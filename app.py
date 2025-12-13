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
    rh = np.clip(rh, 1,
