import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone
from pathlib import Path

# =========================================================
# CONFIG UI
# =========================================================
st.set_page_config(page_title="Risque mildiou oignon", layout="wide")
st.title("Risque mildiou oignon — Observé + Prévision")

BASE = "https://api.sencrop.com/v1"
TZ = "Europe/Paris"

# =========================================================
# SECRETS
# =========================================================
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

# =========================================================
# HTTP HELPERS
# =========================================================
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

# =========================================================
# MILDEW HELPERS
# =========================================================
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


# =========================================================
# FORECAST/OBS STORAGE + BIAS CORRECTION (simple MOS)
# =========================================================
STORE_DIR = Path("store")
STORE_DIR.mkdir(exist_ok=True)
FC_STORE = STORE_DIR / "forecast_snapshots.csv"
OBS_STORE = STORE_DIR / "obs_hourly.csv"

def append_obs_snapshot(df_obs: pd.DataFrame, device_id: int):
    """Append observed Sencrop hourly data to local store (dedup by time+device)."""
    if df_obs is None or df_obs.empty:
        return
    out = df_obs.copy()
    out["device_id"] = int(device_id)
    cols = ["device_id", "time", "temp_c", "rh_pct", "rain_mm"]
    out = out[cols].copy()
    header = not OBS_STORE.exists()
    out.to_csv(OBS_STORE, mode="a", header=header, index=False)

def append_forecast_snapshot(df_fc: pd.DataFrame, device_id: int, model: str):
    """Append forecast snapshot with issued_at so we can verify later vs station."""
    if df_fc is None or df_fc.empty:
        return
    issued_at = pd.Timestamp.now(tz="UTC")
    out = df_fc.copy()
    out = out.rename(columns={"temp_c": "temp_fc", "rh_pct": "rh_fc", "rain_mm": "rain_fc"})
    out["issued_at_utc"] = issued_at
    out["device_id"] = int(device_id)
    out["model"] = str(model)

    cols = ["issued_at_utc", "device_id", "model", "time", "temp_fc", "rh_fc", "rain_fc"]
    out = out[cols].copy()
    header = not FC_STORE.exists()
    out.to_csv(FC_STORE, mode="a", header=header, index=False)

def _fit_linear(x, y, min_n=60):
    """Return a,b for y ≈ a*x + b, with guards."""
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return 1.0, 0.0
    a, b = np.polyfit(x[m], y[m], 1)
    if not np.isfinite(a) or not np.isfinite(b):
        return 1.0, 0.0
    # avoid crazy slopes
    a = float(np.clip(a, 0.7, 1.3))
    b = float(np.clip(b, -15.0, 15.0))
    return a, b

def _fit_rain_multiplier(r_fc, r_obs, min_n=40):
    """k for r_obs ≈ k*r_fc. Uses only hours with some forecast rain."""
    r_fc = np.asarray(r_fc); r_obs = np.asarray(r_obs)
    m = np.isfinite(r_fc) & np.isfinite(r_obs) & (r_fc > 0.1)
    if m.sum() < min_n:
        return 1.0
    k = float(r_obs[m].sum() / r_fc[m].sum())
    if not np.isfinite(k) or k <= 0:
        return 1.0
    return float(np.clip(k, 0.2, 5.0))

def compute_bias_coeffs(device_id: int, model: str, window_days: int = 30, lead_hours_max: int = 48) -> dict:
    """Learn bias on recent history, limited to short lead times (e.g. <=48h)."""
    if (not FC_STORE.exists()) or (not OBS_STORE.exists()):
        return {"T": {"a": 1.0, "b": 0.0}, "RH": {"a": 1.0, "b": 0.0}, "R": {"k": 1.0}}

    fc = pd.read_csv(FC_STORE, parse_dates=["issued_at_utc", "time"])
    fc = fc[(fc["device_id"] == int(device_id)) & (fc["model"] == str(model))].copy()
    if fc.empty:
        return {"T": {"a": 1.0, "b": 0.0}, "RH": {"a": 1.0, "b": 0.0}, "R": {"k": 1.0}}

    obs = pd.read_csv(OBS_STORE, parse_dates=["time"])
    obs = obs[obs["device_id"] == int(device_id)].copy()
    if obs.empty:
        return {"T": {"a": 1.0, "b": 0.0}, "RH": {"a": 1.0, "b": 0.0}, "R": {"k": 1.0}}

    # Dedup (in case of repeated appends)
    fc = fc.drop_duplicates(subset=["issued_at_utc", "time", "device_id", "model"])
    obs = obs.drop_duplicates(subset=["device_id", "time"])

    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=window_days)

    # Keep only verifiable, recent, past hours
    fc = fc[(fc["time"] >= cutoff) & (fc["time"] <= now)].copy()

    # Filter by lead time (avoid long-range forecast degradation bias)
    lead_h = (fc["time"] - fc["issued_at_utc"]).dt.total_seconds() / 3600.0
    fc = fc[(lead_h >= 0) & (lead_h <= float(lead_hours_max))].copy()
    if fc.empty:
        return {"T": {"a": 1.0, "b": 0.0}, "RH": {"a": 1.0, "b": 0.0}, "R": {"k": 1.0}}

    merged = fc.merge(
        obs.rename(columns={"temp_c": "temp_obs", "rh_pct": "rh_obs", "rain_mm": "rain_obs"})[["time", "temp_obs", "rh_obs", "rain_obs"]],
        on="time",
        how="inner",
    )
    if merged.empty:
        return {"T": {"a": 1.0, "b": 0.0}, "RH": {"a": 1.0, "b": 0.0}, "R": {"k": 1.0}}

    aT, bT = _fit_linear(merged["temp_fc"], merged["temp_obs"])
    aRH, bRH = _fit_linear(merged["rh_fc"], merged["rh_obs"])
    kR = _fit_rain_multiplier(merged["rain_fc"], merged["rain_obs"])

    return {"T": {"a": aT, "b": bT}, "RH": {"a": aRH, "b": bRH}, "R": {"k": kR}}

def apply_bias_to_forecast(df_fc: pd.DataFrame, coeffs: dict) -> pd.DataFrame:
    if df_fc is None or df_fc.empty:
        return df_fc
    out = df_fc.copy()
    out["temp_c"] = coeffs["T"]["a"] * out["temp_c"] + coeffs["T"]["b"]
    out["rh_pct"] = coeffs["RH"]["a"] * out["rh_pct"] + coeffs["RH"]["b"]
    out["rh_pct"] = out["rh_pct"].clip(40, 100)
    out["rain_mm"] = (coeffs["R"]["k"] * out["rain_mm"]).clip(lower=0)
    return out



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

    # Background hourly curves
    if df2_obs is not None and not df2_obs.empty:
        fig.add_trace(go.Scatter(
            x=df2_obs["time_local"], y=df2_obs["score_h"],
            mode="lines", name="Score horaire observé (fond)",
            line=dict(color="gray"), opacity=0.30
        ))
    if df2_fc is not None and not df2_fc.empty:
        fig.add_trace(go.Scatter(
            x=df2_fc["time_local"], y=df2_fc["score_h"],
            mode="lines", name="Score horaire prévision (fond)",
            line=dict(color="gray", dash="dash"), opacity=0.30
        ))

    # Episode points
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

    # Now vertical
    fig.add_vline(x=now_local, line_dash="dash")

    # Thresholds (reliable shapes)
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

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.header("Paramètres")
    obs_days = st.slider("Historique Sencrop (jours)", 1, 15, 7)
    fc_days = st.slider("Prévision (jours)", 1, 10, 7)
    model = st.selectbox("Modèle météo", ["icon_eu", "icon"], index=0)
    st.caption("Optionnel: limite les stations via DEVICE_ALLOWLIST dans Secrets.")
    if st.button("Rafraîchir maintenant"):
        st.cache_data.clear()
        st.rerun()

# =========================================================
# MAIN PIPELINE
# =========================================================
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

    colA, colB = st.columns([2, 1])

    with colB:
        st.subheader("Station")
        st.write(f"**DEVICE_ID**: {device_id}")
        if lat is None or lon is None:
            st.warning("Coordonnées absentes côté Sencrop. Renseigne-les :")
            lat = st.number_input("Latitude", value=48.856600, format="%.6f")
            lon = st.number_input("Longitude", value=2.352200, format="%.6f")
        else:
            st.write(f"**Lat/Lon**: {lat}, {lon}")

    with st.spinner("Récupération observé (Sencrop) + calcul risque..."):
        df_obs = sencrop_hourly(tok, user_id, device_id, obs_days)
        append_obs_snapshot(df_obs, device_id)
        df2_obs, ep_obs = compute_ep_onion(df_obs)

    with st.spinner(f"Récupération prévision ({model}) + calcul risque..."):
        df_fc = openmeteo_forecast(float(lat), float(lon), fc_days, model)
        append_forecast_snapshot(df_fc, device_id=device_id, model=model)

        # Learn bias only on short lead times (<=48h) to avoid long-range degradation effects
        coeffs = compute_bias_coeffs(device_id=device_id, model=model, window_days=30, lead_hours_max=48)
        df_fc_corr = apply_bias_to_forecast(df_fc, coeffs)

        now_utc = pd.Timestamp.now(tz="UTC")
        df_fc_future = df_fc_corr[df_fc_corr["time"] >= now_utc].copy()
        df2_fc, ep_fc = compute_ep_onion(df_fc_future)

        # Sidebar debug (optional but useful)
        try:
            st.sidebar.caption(
                f"Correction biais (30j, ≤48h): "
                f"T a={coeffs['T']['a']:.2f} b={coeffs['T']['b']:+.2f} | "
                f"RH a={coeffs['RH']['a']:.2f} b={coeffs['RH']['b']:+.1f} | "
                f"Pluie k={coeffs['R']['k']:.2f}"
            )
        except Exception:
            pass

    fig = make_mobile_plot(df2_obs, df2_fc, ep_obs, ep_fc, model)

    with colA:
        st.plotly_chart(fig, use_container_width=True)

        # =========================================================
        # PRO SUMMARY
        # =========================================================
        st.subheader("Synthèse")

        def status_badge(level: str):
            if level == "ALERTE":
                st.error("🚨 **ALERTE** (≥ 70) détectée dans la prévision")
            elif level == "PRE-ALERTE":
                st.warning("⚠️ **PRÉ-ALERTE** (≥ 60) détectée dans la prévision")
            else:
                st.success("✅ Pas d’épisode ≥ 60 détecté dans la prévision")

        if ep_fc is None or ep_fc.empty:
            status_badge("OK")
            st.caption("Aucun épisode mouillant significatif détecté dans la prévision.")
        else:
            risk70 = ep_fc[ep_fc["mildiou_score"] >= 70].sort_values("start")
            risk60 = ep_fc[ep_fc["mildiou_score"] >= 60].sort_values("start")

            if not risk70.empty:
                status_badge("ALERTE")
            elif not risk60.empty:
                status_badge("PRE-ALERTE")
            else:
                status_badge("OK")

            top = risk60.head(3)
            if top.empty:
                st.caption("Aucune pré-alerte/alerte (score < 60) sur la fenêtre de prévision.")
            else:
                st.markdown("### Prochaines alertes (prévision)")
                for _, r in top.iterrows():
                    emoji = "🚨" if r["mildiou_score"] >= 70 else "⚠️"
                    st.write(
                        f"{emoji} **{r['statut']}** probable le **{r['start'].strftime('%d/%m %Hh')}** — "
                        f"score **{int(r['mildiou_score'])}** "
                        f"(durée **{r['duration_h']:.1f}h**, T **{r['t_mean']:.1f}°C**, RH **{r['rh_mean']:.0f}%**, pluie **{r['rain_sum']:.1f}mm**)"
                    )

        # =========================================================
        # TABLES
        # =========================================================
        st.subheader("Détails")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Alertes prévues (≥60)", "Tous épisodes prévus", "Épisodes observés", "Dernières données horaires", "Comparaison obs vs prévision"])

        cols = ["start","end","duration_h","required_h","mildiou_score","statut","t_mean","rh_mean","dpd_mean","rain_sum"]

        with tab1:
            if ep_fc is None or ep_fc.empty:
                st.info("Aucun épisode prévisionnel.")
            else:
                risk = ep_fc[ep_fc["mildiou_score"] >= 60].sort_values("start")
                if risk.empty:
                    st.info("Aucune pré-alerte/alerte (score < 60).")
                else:
                    st.dataframe(risk[cols], use_container_width=True)

        with tab2:
            if ep_fc is None or ep_fc.empty:
                st.info("Aucun épisode prévisionnel.")
            else:
                st.dataframe(ep_fc.sort_values("start")[cols], use_container_width=True)

        with tab3:
            if ep_obs is None or ep_obs.empty:
                st.info("Aucun épisode observé.")
            else:
                st.dataframe(ep_obs.sort_values("start").tail(50)[cols], use_container_width=True)

        with tab4:
            st.dataframe(df_obs.tail(72), use_container_width=True)


        with tab5:
            st.markdown("#### Comparaison observation station vs prévision (contrôle qualité)")
            st.caption("Objectif : vérifier visuellement qu'il n'y a pas d'écarts aberrants entre la station et la prévision. "
                       "Les prévisions affichées sont celles sauvegardées lors des exécutions précédentes (snapshots) ; "
                       "la version 'corrigée' applique la correction de biais apprise (si disponible).")

            if (not FC_STORE.exists()) and (not OBS_STORE.exists()):
                st.info("Aucun historique local trouvé (dossier `store/`). Revenez après quelques exécutions de l'app.")
            else:
                # Observations: on utilise df_obs (déjà chargé) et, si disponible, le store pour compléter.
                obs = df_obs.copy()
                obs["time"] = pd.to_datetime(obs["time"], utc=True)
                obs = obs.dropna(subset=["time"]).sort_values("time")

                # Prévisions historisées (snapshots)
                if not FC_STORE.exists():
                    st.info("Pas encore de prévisions historisées (store/forecast_snapshots.csv).")
                else:
                    fc_hist = pd.read_csv(FC_STORE, parse_dates=["issued_at_utc", "time"])
                    fc_hist = fc_hist[(fc_hist["device_id"] == int(device_id)) & (fc_hist["model"] == str(model))].copy()
                    if fc_hist.empty:
                        st.info("Pas de snapshots de prévision pour cette station / ce modèle.")
                    else:
                        # Ne garder que les échéances courtes pour la comparaison (≤ 48h)
                        fc_hist["lead_h"] = (fc_hist["time"] - fc_hist["issued_at_utc"]).dt.total_seconds() / 3600.0
                        fc_hist = fc_hist[(fc_hist["lead_h"] >= 0) & (fc_hist["lead_h"] <= 48)].copy()

                        # Pour chaque heure cible, on garde la dernière prévision émise avant l'heure cible
                        fc_hist = fc_hist.sort_values(["time", "issued_at_utc"])
                        fc_best = fc_hist.groupby("time", as_index=False).tail(1)

                        # Merge avec obs (dernières 72h par défaut)
                        window_h = st.slider("Fenêtre d'affichage (heures)", 24, 168, 72, step=24)
                        tmax = obs["time"].max() if not obs.empty else pd.Timestamp.now(tz="UTC")
                        tmin = tmax - pd.Timedelta(hours=window_h)

                        obs_w = obs[(obs["time"] >= tmin) & (obs["time"] <= tmax)].copy()
                        fc_w = fc_best[(fc_best["time"] >= tmin) & (fc_best["time"] <= tmax)].copy()

                        m = fc_w.merge(
                            obs_w.rename(columns={"temp_c":"temp_obs","rh_pct":"rh_obs","rain_mm":"rain_obs"}),
                            on="time", how="inner"
                        )
                        if m.empty:
                            st.info("Pas assez de recouvrement entre les observations et les snapshots prévisionnels sur la fenêtre choisie.")
                        else:
                            # Prévision brute
                            m = m.rename(columns={"temp_fc":"temp_raw","rh_fc":"rh_raw","rain_fc":"rain_raw"})

                            # Prévision corrigée (avec les coeffs courants)
                            try:
                                m["temp_corr"] = coeffs["T"]["a"]*m["temp_raw"] + coeffs["T"]["b"]
                                m["rh_corr"] = coeffs["RH"]["a"]*m["rh_raw"] + coeffs["RH"]["b"]
                                m["rh_corr"] = m["rh_corr"].clip(0, 100)
                                m["rain_corr"] = (coeffs["R"]["k"]*m["rain_raw"]).clip(lower=0)
                            except Exception:
                                m["temp_corr"] = np.nan
                                m["rh_corr"] = np.nan
                                m["rain_corr"] = np.nan

                            # Erreurs
                            m["err_temp_raw"] = m["temp_raw"] - m["temp_obs"]
                            m["err_temp_corr"] = m["temp_corr"] - m["temp_obs"]
                            m["err_rh_raw"] = m["rh_raw"] - m["rh_obs"]
                            m["err_rh_corr"] = m["rh_corr"] - m["rh_obs"]
                            m["err_rain_raw"] = m["rain_raw"] - m["rain_obs"]
                            m["err_rain_corr"] = m["rain_corr"] - m["rain_obs"]

                            # Graphiques
                            fig_t = go.Figure()
                            fig_t.add_trace(go.Scatter(x=m["time"], y=m["temp_obs"], mode="lines", name="T obs"))
                            fig_t.add_trace(go.Scatter(x=m["time"], y=m["temp_raw"], mode="lines", name="T prévision (snapshot)"))
                            if m["temp_corr"].notna().any():
                                fig_t.add_trace(go.Scatter(x=m["time"], y=m["temp_corr"], mode="lines", name="T prévision corrigée"))
                            fig_t.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), title="Température (°C)")
                            st.plotly_chart(fig_t, use_container_width=True)

                            fig_rh = go.Figure()
                            fig_rh.add_trace(go.Scatter(x=m["time"], y=m["rh_obs"], mode="lines", name="RH obs"))
                            fig_rh.add_trace(go.Scatter(x=m["time"], y=m["rh_raw"], mode="lines", name="RH prévision (snapshot)"))
                            if m["rh_corr"].notna().any():
                                fig_rh.add_trace(go.Scatter(x=m["time"], y=m["rh_corr"], mode="lines", name="RH prévision corrigée"))
                            fig_rh.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), title="Humidité relative (%)")
                            st.plotly_chart(fig_rh, use_container_width=True)

                            fig_p = go.Figure()
                            fig_p.add_trace(go.Bar(x=m["time"], y=m["rain_obs"], name="Pluie obs"))
                            fig_p.add_trace(go.Bar(x=m["time"], y=m["rain_raw"], name="Pluie prévision (snapshot)"))
                            if m["rain_corr"].notna().any():
                                fig_p.add_trace(go.Bar(x=m["time"], y=m["rain_corr"], name="Pluie prévision corrigée"))
                            fig_p.update_layout(barmode="group", height=320, margin=dict(l=10,r=10,t=30,b=10), title="Précipitations (mm/h)")
                            st.plotly_chart(fig_p, use_container_width=True)

                            # Tableau coloré des erreurs (absolues)
                            show_cols = [
                                "time",
                                "temp_obs","temp_raw","temp_corr","err_temp_raw","err_temp_corr",
                                "rh_obs","rh_raw","rh_corr","err_rh_raw","err_rh_corr",
                                "rain_obs","rain_raw","rain_corr","err_rain_raw","err_rain_corr",
                                "lead_h",
                            ]
                            view = m[show_cols].copy()
                            view = view.sort_values("time")

                            def _color_abs_err(s, cap):
                                a = s.abs()
                                n = (a / cap).clip(0, 1)

                                colors = []
                                for v in n.to_numpy():
                                    if v is None or not np.isfinite(v):
                                        colors.append("")
                                        continue

                                    v = float(v)
                                    r = int(220 + (255 - 220) * v)
                                    g = int(255 - (255 - 220) * v)
                                    b = 220
                                    colors.append(f"background-color: rgb({r},{g},{b})")

                                return colors
                            styler = view.style.format({
                                "temp_obs":"{:.1f}","temp_raw":"{:.1f}","temp_corr":"{:.1f}",
                                "err_temp_raw":"{:+.1f}","err_temp_corr":"{:+.1f}",
                                "rh_obs":"{:.0f}","rh_raw":"{:.0f}","rh_corr":"{:.0f}",
                                "err_rh_raw":"{:+.0f}","err_rh_corr":"{:+.0f}",
                                "rain_obs":"{:.2f}","rain_raw":"{:.2f}","rain_corr":"{:.2f}",
                                "err_rain_raw":"{:+.2f}","err_rain_corr":"{:+.2f}",
                                "lead_h":"{:.0f}",
                            })
                            styler = styler.apply(_color_abs_err, subset=["err_temp_raw","err_temp_corr"], cap=3.0)
                            styler = styler.apply(_color_abs_err, subset=["err_rh_raw","err_rh_corr"], cap=15.0)
                            styler = styler.apply(_color_abs_err, subset=["err_rain_raw","err_rain_corr"], cap=2.0)

                            st.markdown("#### Table de contrôle (écarts colorés)")
                            st.dataframe(styler, use_container_width=True)

except Exception as e:
    st.error("Erreur pendant l'exécution.")
    st.exception(e)
