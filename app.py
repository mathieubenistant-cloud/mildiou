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
def get_secret(key, env_fallback=None, default=None):
    if key in st.secrets:
        return str(st.secrets[key]).strip()
    if env_fallback and env_fallback in os.environ:
        return os.environ[env_fallback].strip()
    return default

APPLICATION_ID = get_secret("SENCROP_APPLICATION_ID", "SENCROP_APPLICATION_ID")
APPLICATION_SECRET = get_secret("SENCROP_APPLICATION_SECRET", "SENCROP_APPLICATION_SECRET")

if not APPLICATION_ID or not APPLICATION_SECRET:
    st.error("Secrets Sencrop manquants")
    st.stop()

DEVICE_ALLOWLIST = get_secret("DEVICE_ALLOWLIST", "DEVICE_ALLOWLIST", default="")

def parse_allowlist(s):
    out = set()
    for p in (s or "").split(","):
        try:
            out.add(int(p.strip()))
        except Exception:
            pass
    return out

ALLOW_IDS = parse_allowlist(DEVICE_ALLOWLIST)

# =========================================================
# HTTP HELPERS
# =========================================================
def must(r):
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r

@st.cache_data(ttl=3600)
def sencrop_token(app_id, app_secret):
    r = must(requests.post(
        f"{BASE}/oauth2/token",
        auth=(app_id, app_secret),
        json={"grant_type": "client_credentials", "scope": "user"}
    ))
    return r.json()["access_token"]

def headers(token):
    return {"Authorization": f"Bearer {token}"}

@st.cache_data(ttl=900)
def sencrop_me(token):
    return must(requests.get(f"{BASE}/me", headers=headers(token))).json()

@st.cache_data(ttl=900)
def sencrop_devices(token, user_id):
    return must(requests.get(
        f"{BASE}/users/{user_id}/devices",
        headers=headers(token)
    )).json()

# =========================================================
# METEO
# =========================================================
def dew_point_c(temp, rh):
    a, b = 17.62, 243.12
    rh = np.clip(rh, 1, 100) / 100
    g = (a * temp) / (b + temp) + np.log(rh)
    return (b * g) / (a - g)

@st.cache_data(ttl=600)
def sencrop_hourly(token, user_id, device_id, days):
    url = f"{BASE}/users/{user_id}/devices/{device_id}/data/hourly"
    before = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def fetch(measure, col):
        r = must(requests.get(
            url,
            headers=headers(token),
            params={"days": days, "beforeDate": before, "measures": measure}
        ))
        d = pd.DataFrame(r.json()["measures"]["data"])
        if d.empty:
            return pd.DataFrame(columns=["time", col])
        d[col] = d[measure].apply(lambda x: x.get("value") if isinstance(x, dict) else x)
        d["time"] = pd.to_datetime(d["key"], unit="ms", utc=True)
        return d[["time", col]]

    t = fetch("TEMPERATURE", "temp_c")
    h = fetch("RELATIVE_HUMIDITY", "rh_pct")
    r = fetch("RAIN_FALL", "rain_mm")

    return t.merge(h, on="time", how="outer").merge(r, on="time", how="outer")

@st.cache_data(ttl=1800)
def openmeteo_forecast(lat, lon, days, model):
    r = must(requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "forecast_days": days,
            "timezone": TZ,
            "models": model,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation"
        }
    ))
    h = r.json()["hourly"]
    t = pd.to_datetime(h["time"]).tz_localize(TZ).tz_convert("UTC")
    return pd.DataFrame({
        "time": t,
        "temp_c": h["temperature_2m"],
        "rh_pct": h["relative_humidity_2m"],
        "rain_mm": h["precipitation"]
    })

# =========================================================
# BIAS CORRECTION (SAFE)
# =========================================================
STORE = Path("store")
STORE.mkdir(exist_ok=True)
FC_STORE = STORE / "forecast_snapshots.csv"
OBS_STORE = STORE / "obs_hourly.csv"

def append_obs(df, device_id):
    if df.empty:
        return
    d = df.copy()
    d["device_id"] = device_id
    d.to_csv(OBS_STORE, mode="a", header=not OBS_STORE.exists(), index=False)

def append_fc(df, device_id, model):
    if df.empty:
        return
    d = df.copy()
    d["issued_at"] = pd.Timestamp.now(tz="UTC")
    d["device_id"] = device_id
    d["model"] = model
    d.to_csv(FC_STORE, mode="a", header=not FC_STORE.exists(), index=False)

def compute_bias(device_id, model):
    if not FC_STORE.exists() or not OBS_STORE.exists():
        return {"T": (1, 0), "RH": (1, 0), "R": 1}

    fc = pd.read_csv(FC_STORE, parse_dates=["time", "issued_at"])
    obs = pd.read_csv(OBS_STORE, parse_dates=["time"])

    fc = fc[(fc.device_id == device_id) & (fc.model == model)]
    obs = obs[obs.device_id == device_id]

    m = fc.merge(obs, on="time", suffixes=("_fc", "_obs"))
    if len(m) < 48:
        return {"T": (1, 0), "RH": (1, 0), "R": 1}

    aT, bT = np.polyfit(m.temp_c_fc, m.temp_c_obs, 1)
    aRH, bRH = np.polyfit(m.rh_pct_fc, m.rh_pct_obs, 1)

    aRH = np.clip(aRH, 0.7, 1.3)
    bRH = np.clip(bRH, -15, 15)

    kR = np.clip(m.rain_mm_obs.sum() / max(m.rain_mm_fc.sum(), 0.1), 0.2, 5)

    return {"T": (aT, bT), "RH": (aRH, bRH), "R": kR}

# =========================================================
# MILDEW MODEL
# =========================================================
def compute_ep(df):
    df = df.copy()
    df["td"] = dew_point_c(df.temp_c, df.rh_pct)
    df["dpd"] = df.temp_c - df.td
    df["wet"] = (df.rain_mm.fillna(0) > 0) | ((df.rh_pct >= 88) & (df.dpd <= 1.5))
    df["grp"] = (df.wet != df.wet.shift()).cumsum()
    w = df[df.wet]
    if w.empty:
        return df, pd.DataFrame()

    ep = w.groupby("grp").agg(
        start=("time", "min"),
        end=("time", "max"),
        t=("temp_c", "mean"),
        rh=("rh_pct", "mean"),
        rain=("rain_mm", "sum")
    )
    ep["dur"] = ((ep.end - ep.start).dt.total_seconds() / 3600) + 1
    ep["score"] = (100 * (
        0.4 * np.clip((ep.rh - 80) / 20, 0, 1) +
        0.3 * np.clip(1 - abs(ep.t - 12) / 10, 0, 1) +
        0.2 * np.clip(ep.dur / 3, 0, 1) +
        0.1 * np.clip(ep.rain / 3, 0, 1)
    )).round().astype(int)
    return df, ep

# =========================================================
# APP
# =========================================================
try:
    tok = sencrop_token(APPLICATION_ID, APPLICATION_SECRET)
    me = sencrop_me(tok)
    user_id = list(me["users"].keys())[0]
    devs = sencrop_devices(tok, user_id)

    choices = []
    meta = {}
    for d in devs.get("items", []):
        did = int(d)
        if ALLOW_IDS and did not in ALLOW_IDS:
            continue
        choices.append(str(did))
        meta[str(did)] = devs["devices"][str(d)]

    device = st.selectbox("Station", choices)
    device_id = int(device)
    loc = meta[device].get("location", {})
    lat, lon = loc.get("latitude"), loc.get("longitude")

    obs_days = st.slider("Historique (jours)", 1, 15, 7)
    fc_days = st.slider("Prévision (jours)", 1, 10, 7)
    model = st.selectbox("Modèle météo", ["icon_eu", "icon"])

    df_obs = sencrop_hourly(tok, user_id, device_id, obs_days)
    append_obs(df_obs, device_id)
    df_obs2, ep_obs = compute_ep(df_obs)

    df_fc = openmeteo_forecast(lat, lon, fc_days, model)
    append_fc(df_fc, device_id, model)

    bias = compute_bias(device_id, model)
    aT, bT = bias["T"]
    aRH, bRH = bias["RH"]

    df_fc["temp_c"] = aT * df_fc.temp_c + bT
    df_fc["rh_pct"] = np.clip(aRH * df_fc.rh_pct + bRH, 40, 100)
    df_fc["rain_mm"] = bias["R"] * df_fc.rain_mm

    df_fc2, ep_fc = compute_ep(df_fc[df_fc.time >= pd.Timestamp.now(tz="UTC")])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_obs2.time, y=df_obs2.rh_pct, name="RH obs"))
    fig.add_trace(go.Scatter(x=df_fc2.time, y=df_fc2.rh_pct, name="RH prev corr"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Épisodes prévus")
    st.dataframe(ep_fc)

except Exception as e:
    st.error("Erreur d'exécution")
    st.exception(e)

