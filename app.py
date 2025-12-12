import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="Risque mildiou oignon", layout="wide")
st.title("Risque mildiou oignon — Observé + Prévision")

BASE = "https://api.sencrop.com/v1"
TZ = "Europe/Paris"

# -----------------------------
# Secrets
# -----------------------------
def get_secret(key: str, env_fallback: str | None = None, default: str | None = None) -> str | None:
    if key in st.secrets:
        return str(st.secrets[key]).strip()
    if env_fallback and env_fallback in os.environ:
        return os.environ[env_fallback].strip()
    return default

APPLICATION_ID = get_secret("SENCROP_APPLICATION_ID", "SENCROP_APPLICATION_ID")
APPLICATION_SECRET = get_secret("SENCROP_APPLICATION_SECRET", "SENCROP_APPLICATION_SECRET")
if not APPLICATION_ID or not APPLICATION_SECRET:
    st.error("Secrets manquants: SENCROP_APPLICATION_ID / SENCROP_APPLICATION_SECRET")
    st.stop()

# ✅ Allowlist de stations (IDs séparés par virgule)
# Streamlit Cloud > Settings > Secrets :
# DEVICE_ALLOWLIST = "46431, 12345"
DEVICE_ALLOWLIST = get_secret("DEVICE_ALLOWLIST", "DEVICE_ALLOWLIST", default="")

def parse_allowlist(s: str) -> set[int]:
    out = set()
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            pass
    return out

ALLOW_IDS = parse_allowlist(DEVICE_ALLOWLIST)

# -----------------------------
# HTTP helpers
# -----------------------------
def must(r: requests.Response) -> requests.Response:
    if r.status_code >= 400:
        raise requests.HTTPError(f"HTTP {r.status_code} for {r.url}: {r.text[:800]}")
    return r

@st.cache_data(ttl=45 * 60, show_spinner=False)
def sencrop_token(app_id: str, app_secret: str) -> str:
    r = must(requests.post(
        f"{BASE}/oauth2/token",
        auth=(app_id, app_secret),
        json={"grant_type": "client_credentials", "scope": "user"},
        headers={"Content-Type": "application/json"},
        timeout=30
    ))
    return r.json()["access_token"].strip()

def headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

@st.cache_data(ttl=15 * 60, show_spinner=False)
def sencrop_me(token: str) -> dict:
    return must(requests.get(f"{BASE}/me", headers=headers(token), timeout=30)).json()

@st.cache_data(ttl=15 * 60, show_spinner=False)
def sencrop_devices(token: str, user_id: int) -> dict:
    return must(requests.get(f"{BASE}/users/{user_id}/devices", headers=headers(token), timeout=30)).json()

def resolve_user_id(me_payload: dict) -> int:
    user_id = me_payload.get("item")
    if user_id is None and isinstance(me_payload.get("users"), dict) and len(me_payload["users"]) > 0:
        user_id = int(next(iter(me_payload["users"].keys())))
    if user_id is None:
        raise ValueError("Impossible de déterminer USER_ID depuis /me")
    return int(user_id)

# -----------------------------
# Mildiou helpers
# -----------------------------
def dew_point_c(temp_c, rh_pct):
    a, b = 17.62, 243.12
    rh = np.clip(rh_pct, 1, 100) / 100.0
    gamma = (a * temp_c) / (b + temp_c) + np.log(rh)
    return (b * gamma) / (a - gamma)

@st.cache_data(ttl=10 * 60, show_spinner=False)
def sencrop_hourly(token: str, user_id: int, device_id: int, days: int) -> pd.DataFrame:
    url_hourly = f"{BASE}/users/{user_id}/devices/{device_id}/data/hourly"
    before_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def fetch_one(measure: str, col: str) -> pd.DataFrame:
        r = must(requests.get(
            url_hourly,
            headers=headers(token),
            params={"days": days, "beforeDate": before_iso, "measures": measure},
            timeout=30
        ))
        data = r.json()["measures"]["data"]
        df_raw = pd.DataFrame(data)
        if df_raw.empty:
            return pd.DataFrame(columns=["time", col])

        if "key" in df_raw.columns and measure in df_raw.columns:
            dfm = df_raw[["key", measure]].rename(columns={"key": "time", measure: col})
            dfm["time"] = pd.to_datetime(dfm["time"], utc=True, unit="ms", errors="coerce")
            if dfm["time"].isna().all():
                dfm["time"] = pd.to_datetime(df_raw["key"], utc=True, errors="coerce")
        else:
            raise ValueError(f"Format inattendu. Colonnes reçues: {list(df_raw.columns)}")

        dfm[col] = dfm[col].apply(lambda x: x.get("value") if isinstance(x, dict) else x)
        dfm[col] = pd.to_numeric(dfm[col], errors="coerce")
        return dfm

    df_temp = fetch_one("TEMPERATURE", "temp_c")
    df_rh   = fetch_one("RELATIVE_HUMIDITY", "rh_pct")
    df_rain = fetch_one("RAIN_FALL", "rain_mm")

    df = df_temp.merge(df_rh, on="time", how="outer") \
                .merge(df_rain, on="time", how="outer") \
                .sort_values("time") \
                .reset_index(drop=True)
    return df

@st.cache_data(ttl=30 * 60, show_spinner=False)
def openmeteo_forecast(lat: float, lon: float, days: int, model: str) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": TZ,
        "forecast_days": days,
        "models": model,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
    }
    r = must(requests.get(url, params=params, timeout=30))
    j = r.json()
    h = j["hourly"]
    tloc = pd.to_datetime(h["time"]).tz_localize(TZ)
    return pd.DataFrame({
        "time": tloc.tz_convert("UTC"),
        "temp_c": h["temperature_2m"],
        "rh_pct": h["relative_humidity_2m"],
        "rain_mm": h["precipitation"],
    })

def compute_ep_onion(df_in: pd.DataFrame,
                     rh_wet: float = 88.0,
                     dpd_wet: float = 1.5,
                     t_opt: float = 12.0,
                     min_at_opt: float = 3.0,
                     max_required: float = 10.0,
                     slope: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df_in.dropna(subset=["time"]).copy()
    df2["time_local"] = df2["time"].dt.tz_convert(TZ)
    df2 = df2.sort_values("time_local").reset_index(drop=True)

    df2["td_c"] = dew_point_c(df2["temp_c"], df2["rh_pct"])
    df2["dpd_c"] = df2["temp_c"] - df2["td_c"]

    df2["is_wet"] = (df2["rain_mm"].fillna(0) > 0) | ((df2["rh_pct"] >= rh_wet) & (df2["dpd_c"] <= dpd_wet))

    dt_hours = df2["time_local"].diff().dt.total_seconds().div(3600)
    df2["big_gap"] = dt_hours.fillna(1) > 1.01
    df2["episode_break"] = (df2["is_wet"] != df2["is_wet"].shift(1)) | df2["big_gap"]
    df2["episode_id"] = df2["episode_break"].cumsum()

    wet_points = df2[df2["is_wet"]].copy()
    if wet_points.empty:
        return df2, pd.DataFrame()

    ep = wet_points.groupby("episode_id", as_index=False).agg(
        start=("time_local", "min"),
        end=("time_local", "max"),
        t_mean=("temp_c", "mean"),
        rh_mean=("rh_pct", "mean"),
        rain_sum=("rain_mm", "sum"),
        dpd_mean=("dpd_c", "mean"),
    )
    ep["duration_h"] = ((ep["end"] - ep["start"]).dt.total_seconds() / 3600).round(2) + 1
    ep["required_h"] = np.clip(min_at_opt + slope * np.abs(ep["t_mean"] - t_opt),
                               min_at_opt, max_required)

    ep = ep[ep["duration_h"] >= ep["required_h"]].copy().sort_values("start").reset_index(drop=True)
    if ep.empty:
        return df2, ep

    rh_score   = np.clip((ep["rh_mean"] - 80) / 20, 0, 1)
    t_score    = np.clip(1 - (np.abs(ep["t_mean"] - t_opt) / 10), 0, 1)
    rain_score = np.clip(ep["rain_sum"].fillna(0) / 3.0, 0, 1)
    dur_score  = np.clip(ep["duration_h"] / ep["required_h"], 0, 1)

    ep["mildiou_score"] = (100 * (0.40*rh_score + 0.30*t_score + 0.20*dur_score + 0.10*rain_score)).round(0).astype(int)
    ep["statut"] = np.select(
        [ep["mildiou_score"] >= 70, ep["mildiou_score"] >= 60],
        ["ALERTE", "PRE-ALERTE"],
        default="OK"
    )
    return df2, ep

def hourly_score_onion(temp_c, rh_pct, rain_mm, dpd_c, t_opt=12.0):
    if pd.isna(temp_c) or pd.isna(rh_pct) or pd.isna(dpd_c):
        return np.nan
    rh_s  = np.clip((rh_pct - 80) / 20, 0, 1)
    dpd_s = np.clip(1 - (dpd_c / 3.0), 0, 1)
    t_s   = np.clip(1 - (abs(temp_c - t_opt) / 10), 0, 1)
    r_s   = np.clip((0 if pd.isna(rain_mm) else rain_mm) / 1.0, 0, 1)
    return 100 * (0.35*rh_s + 0.35*dpd_s + 0.25*t_s + 0.05*r_s)

def make_mobile_plot(df2_obs, df2_fc, ep_obs, ep_fc, model_name: str):
    now_local = pd.Timestamp.now(tz=TZ)
    xmin = now_local - pd.Timedelta(days=7)
    xmax = now_local + pd.Timedelta(days=7)

    for d in (df2_obs, df2_fc):
        if d is None or d.empty:
            continue
        d["score_h"] = d.apply(lambda r: hourly_score_onion(r["temp_c"], r["rh_pct"], r["rain_mm"], r["dpd_c"]), axis=1)

    fig = go.Figure()

    if df2_obs is not None and not df2_obs.empty:
        fig.add_trace(go.Scatter(
            x=df2_obs["time_local"], y=df2_obs["score_h"],
            mode="lines", name="Score horaire observé",
            line=dict(color="gray"), opacity=0.35
        ))
    if df2_fc is not None and not df2_fc.empty:
        fig.add_trace(go.Scatter(
            x=df2_fc["time_local"], y=df2_fc["score_h"],
            mode="lines", name="Score horaire prévision",
            line=dict(color="gray", dash="dash"), opacity=0.35
        ))

    def add_points(ep, name_prefix):
        if ep is None or ep.empty:
            return
        red = ep[ep["mildiou_score"] >= 60]
        green = ep[ep["mildiou_score"] < 60]
        if not green.empty:
            fig.add_trace(go.Scatter(
                x=green["start"], y=green["mildiou_score"],
                mode="markers", name=f"{name_prefix} épisodes <60",
                marker=dict(color="green", size=8)
            ))
        if not red.empty:
            fig.add_trace(go.Scatter(
                x=red["start"], y=red["mildiou_score"],
                mode="markers", name=f"{name_prefix} épisodes ≥60",
                marker=dict(color="red", size=10)
            ))

    add_points(ep_obs, "Observé")
    add_points(ep_fc, "Prévision")

    # now vertical
    fig.add_vline(x=now_local, line_dash="dash")

    # ✅ seuils (lignes visibles via shapes)
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=60, y1=60,
                  line=dict(color="orange", width=1, dash="dot"))
    fig.add_annotation(xref="paper", x=0.01, yref="y", y=60, text="Pré-alerte (60)",
                       showarrow=False, font=dict(color="orange", size=12),
                       bgcolor="rgba(255,255,255,0.6)", xanchor="left", yanchor="bottom")

    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=70, y1=70,
                  line=dict(color="red", width=1, dash="dot"))
    fig.add_annotation(xref="paper", x=0.01, yref="y", y=70, text="Alerte (70)",
                       showarrow=False, font=dict(color="red", size=12),
                       bgcolor="rgba(255,255,255,0.6)", xanchor="left", yanchor="bottom")

    fig.update_layout(
        title=f"Risque mildiou oignon — Observé + Prévision ({model_name})",
        xaxis_title="Date",
        yaxis_title="Score (0–100)",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(range=[xmin, xmax]),
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=480
    )
    return fig

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Paramètres")
    obs_days = st.slider("Historique Sencrop (jours)", 1, 15, 7)
    fc_days = st.slider("Prévision (jours)", 1, 10, 7)
    model = st.selectbox("Modèle météo", ["icon_eu", "icon"], index=0)
    st.caption("Allowlist possible via secret DEVICE_ALLOWLIST.")
    if st.button("Rafraîchir maintenant"):
        st.cache_data.clear()
        st.rerun()

# -----------------------------
# Main
# -----------------------------
try:
    tok = sencrop_token(APPLICATION_ID, APPLICATION_SECRET)
    me = sencrop_me(tok)
    user_id = resolve_user_id(me)

    dev_payload = sencrop_devices(tok, user_id)
    items = dev_payload.get("items", [])
    devs = dev_payload.get("devices", {})

    options = []
    id_to_meta = {}

    for did in items:
        did_int = int(did)
        if ALLOW_IDS and did_int not in ALLOW_IDS:
            continue

        d = devs.get(str(did), devs.get(did, {})) if isinstance(devs, dict) else {}
        name = ((d.get("contents") or {}).get("name")) if isinstance(d, dict) else None
        label = f"{name or 'Station'} — {did_int}"
        options.append(label)
        id_to_meta[label] = {"id": did_int, "raw": d}

    if not options:
        st.error("Aucune station trouvée. Vérifie DEVICE_ALLOWLIST dans Secrets.")
        st.stop()

    sel = st.selectbox("Station", options=options, index=0)
    device_id = id_to_meta[sel]["id"]
    dsel = id_to_meta[sel]["raw"]

    loc = (dsel.get("location") or {}) if isinstance(dsel, dict) else {}
    lat = loc.get("latitude")
    lon = loc.get("longitude")

    if lat is None or lon is None:
        st.warning("Coordonnées absentes côté Sencrop. Renseigne-les :")
        lat = st.number_input("Latitude", value=48.856600, format="%.6f")
        lon = st.number_input("Longitude", value=2.352200, format="%.6f")

    with st.spinner("Récupération observé (Sencrop) + calcul risque..."):
        df_obs = sencrop_hourly(tok, user_id, device_id, obs_days)
        df2_obs, ep_obs = compute_ep_onion(df_obs)

    with st.spinner(f"Récupération prévision ({model}) + calcul risque..."):
        df_fc = openmeteo_forecast(float(lat), float(lon), fc_days, model)
        now_utc = pd.Timestamp.now(tz="UTC")
        df_fc_future = df_fc[df_fc["time"] >= now_utc].copy()
        df2_fc, ep_fc = compute_ep_onion(df_fc_future)

    fig = make_mobile_plot(df2_obs, df2_fc, ep_obs, ep_fc, model)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("Erreur pendant l'exécution.")
    st.exception(e)
