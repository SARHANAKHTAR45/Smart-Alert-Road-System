"""
app.py — Streamlit UI  ·  Redesigned
Hyper-Local Smart Road Hazard Detection & Real-Time Alert System
Run: streamlit run app.py
"""
import os, time, logging
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from config import JUNCTIONS, EDGES, RISK_CRITICAL, RISK_WARNING, XGB_MODEL_PATH, GNN_MODEL_PATH
from utils import risk_colour, risk_label, risk_emoji, estimate_travel_time_mins, ist_timestamp
from alert_engine import AlertEngine

logging.basicConfig(level=logging.WARNING)

st.set_page_config(layout="wide", page_title="RoadGuard BLR", page_icon="🛣️")

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  color: #c9d1d9;
}

.stApp {
  background: #080c12;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(17,34,64,0.6) 0%, transparent 60%),
    linear-gradient(180deg, #080c12 0%, #050810 100%);
}

/* Subtle grid overlay */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,160,50,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,160,50,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #0b1018 !important;
  border-right: 1px solid rgba(255,160,50,0.12) !important;
}

section[data-testid="stSidebar"] > div {
  padding-top: 1.5rem;
}

/* ── Typography ── */
h1, h2, h3, .rg-heading {
  font-family: 'Rajdhani', sans-serif !important;
  letter-spacing: 0.03em;
}

.stMarkdown code, .mono {
  font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Header Block ── */
.rg-header {
  padding: 2rem 0 1.5rem;
  border-bottom: 1px solid rgba(255,160,50,0.15);
  margin-bottom: 1.5rem;
  position: relative;
}

.rg-header-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: #ffa032;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.rg-header-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: clamp(26px, 4vw, 42px);
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0.04em;
  color: #f0f6fc;
  margin: 0 0 8px;
}

.rg-header-title span { color: #ffa032; }

.rg-header-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: #4a5568;
  letter-spacing: 0.08em;
}

/* ── Live Dot ── */
.live-dot {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 0 0 rgba(34,197,94,0.5);
  animation: livepulse 1.8s ease-in-out infinite;
}
@keyframes livepulse {
  0%  { box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
  60% { box-shadow: 0 0 0 7px rgba(34,197,94,0); }
  100%{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}

.amber-dot {
  background: #ffa032;
  box-shadow: 0 0 0 0 rgba(255,160,50,0.5);
  animation: amberpulse 1.8s ease-in-out infinite;
}
@keyframes amberpulse {
  0%  { box-shadow: 0 0 0 0 rgba(255,160,50,0.5); }
  60% { box-shadow: 0 0 0 7px rgba(255,160,50,0); }
  100%{ box-shadow: 0 0 0 0 rgba(255,160,50,0); }
}

.red-dot {
  background: #f85149;
  box-shadow: 0 0 0 0 rgba(248,81,73,0.5);
  animation: redpulse 1.2s ease-in-out infinite;
}
@keyframes redpulse {
  0%  { box-shadow: 0 0 0 0 rgba(248,81,73,0.5); }
  60% { box-shadow: 0 0 0 8px rgba(248,81,73,0); }
  100%{ box-shadow: 0 0 0 0 rgba(248,81,73,0); }
}

/* ── KPI Cards ── */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 1.5rem;
}

.kpi-card {
  background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px;
  padding: 16px 18px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}

.kpi-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent, #ffa032);
  opacity: 0.7;
}

.kpi-card:hover { border-color: rgba(255,160,50,0.25); }

.kpi-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #5a6882;
  margin-bottom: 6px;
}

.kpi-value {
  font-family: 'Rajdhani', sans-serif;
  font-size: 32px;
  font-weight: 700;
  color: #f0f6fc;
  line-height: 1;
  margin-bottom: 4px;
}

.kpi-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #4a5568;
}

.kpi-accent { color: var(--accent, #ffa032) !important; }

/* ── Route Banner ── */
.route-banner {
  background: linear-gradient(135deg, rgba(0,200,255,0.05), rgba(0,200,255,0.02));
  border: 1px solid rgba(0,200,255,0.2);
  border-radius: 12px;
  padding: 18px 22px;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}

.route-banner::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, #00c8ff, #0080ff);
}

.route-banner-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #00c8ff;
  margin-bottom: 8px;
}

.route-path {
  font-family: 'Rajdhani', sans-serif;
  font-size: 17px;
  font-weight: 600;
  color: #e6edf3;
  line-height: 1.4;
  margin-bottom: 12px;
}

.route-stats {
  display: flex;
  gap: 28px;
  flex-wrap: wrap;
}

.route-stat {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.route-stat-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #4a5568;
}

.route-stat-val {
  font-family: 'Rajdhani', sans-serif;
  font-size: 20px;
  font-weight: 700;
  color: #f0f6fc;
}

.risk-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 20px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.1em;
}

/* ── Push Notification Cards ── */
.notif-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 1.5rem;
}

.notif-card {
  background: rgba(15,22,36,0.9);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 11px;
  padding: 14px 16px;
  display: flex;
  gap: 12px;
  align-items: flex-start;
  position: relative;
  overflow: hidden;
  transition: transform 0.15s, border-color 0.15s;
}

.notif-card:hover {
  transform: translateY(-2px);
  border-color: rgba(255,255,255,0.14);
}

.notif-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: var(--notif-accent, rgba(255,255,255,0.1));
}

.notif-card.critical {
  --notif-accent: rgba(248,81,73,0.6);
  border-color: rgba(248,81,73,0.2);
  background: rgba(248,81,73,0.05);
}

.notif-card.high {
  --notif-accent: rgba(255,160,50,0.6);
  border-color: rgba(255,160,50,0.2);
  background: rgba(255,160,50,0.04);
}

.notif-icon {
  font-size: 26px;
  line-height: 1;
  flex-shrink: 0;
  margin-top: 2px;
}

.notif-content { flex: 1; min-width: 0; }

.notif-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: 14px;
  font-weight: 600;
  color: #e6edf3;
  margin-bottom: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.notif-body {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  color: #6e7f9c;
  line-height: 1.4;
  margin-bottom: 6px;
}

.notif-ts {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #333d4d;
}

.notif-level {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--notif-accent, #4a5568);
  margin-bottom: 2px;
}

/* ── Alert Log ── */
.alert-entry {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 10px 14px;
  border-radius: 8px;
  margin-bottom: 6px;
  border-left: 3px solid var(--alert-color, #4a5568);
  background: rgba(255,255,255,0.02);
  transition: background 0.15s;
}

.alert-entry:hover { background: rgba(255,255,255,0.04); }

.alert-entry.critical {
  --alert-color: #f85149;
  background: rgba(248,81,73,0.06);
}

.alert-entry.warning {
  --alert-color: #ffa032;
  background: rgba(255,160,50,0.05);
}

.alert-entry.info {
  --alert-color: #58a6ff;
  background: rgba(88,166,255,0.04);
}

.alert-badge {
  flex-shrink: 0;
  padding: 2px 7px;
  border-radius: 4px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 1px;
}

.badge-gnn  { background: rgba(79,70,229,0.3);  color: #a5b4fc; border: 1px solid rgba(79,70,229,0.4); }
.badge-yolo { background: rgba(5,150,105,0.3);  color: #6ee7b7; border: 1px solid rgba(5,150,105,0.4); }
.badge-iot  { background: rgba(217,119,6,0.3);  color: #fcd34d; border: 1px solid rgba(217,119,6,0.4); }
.badge-sim  { background: rgba(124,58,237,0.3); color: #c4b5fd; border: 1px solid rgba(124,58,237,0.4); }

.alert-msg {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: #8b949e;
  flex: 1;
  line-height: 1.5;
}

.alert-time {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #3d4654;
  flex-shrink: 0;
  margin-top: 1px;
}

/* ── Sensor Table Customisation ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Section Headers ── */
.section-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255,160,50,0.1);
}

.section-header-text {
  font-family: 'Rajdhani', sans-serif;
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: #e6edf3;
}

.section-tag {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #ffa032;
  background: rgba(255,160,50,0.1);
  border: 1px solid rgba(255,160,50,0.2);
  padding: 2px 8px;
  border-radius: 4px;
}

/* ── Weather Widget ── */
.weather-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}

.weather-item {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 8px;
  padding: 10px 12px;
}

.weather-icon { font-size: 18px; margin-bottom: 4px; }

.weather-val {
  font-family: 'Rajdhani', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: #f0f6fc;
  line-height: 1;
}

.weather-lbl {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  color: #3d4654;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 2px;
}

/* ── Source Status Pills ── */
.sources-grid {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 8px;
}

.source-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 7px;
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  color: #6e7f9c;
}

.source-pill .live-dot, .source-pill .amber-dot { flex-shrink: 0; }

/* ── Performance Panel ── */
.perf-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  margin-top: 12px;
}

.perf-table th {
  text-align: left;
  padding: 10px 14px;
  background: rgba(255,160,50,0.07);
  color: #ffa032;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  border-bottom: 1px solid rgba(255,160,50,0.15);
}

.perf-table td {
  padding: 10px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  color: #8b949e;
  vertical-align: top;
}

.perf-table tr:hover td { background: rgba(255,255,255,0.02); }
.perf-table td:first-child { color: #ffa032; font-weight: 500; }
.perf-table td:nth-child(2) { color: #58a6ff; }

/* ── Tab Styling ── */
button[data-baseweb="tab"] {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 14px !important;
  letter-spacing: 0.04em !important;
}

/* ── Sidebar Labels ── */
.sidebar-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #ffa032;
  margin-bottom: 4px;
  display: block;
}

/* ── Divider ── */
hr { border-color: rgba(255,160,50,0.1) !important; }

/* ── Streamlit Metric Override ── */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px;
  padding: 14px 16px !important;
}

[data-testid="stMetricLabel"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: #4a5568 !important;
}

[data-testid="stMetricValue"] {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  color: #f0f6fc !important;
}

/* ── Button Override ── */
.stButton > button {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  border-radius: 8px !important;
  transition: all 0.15s !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #ffa032, #ff7800) !important;
  border: none !important;
  color: #0d1117 !important;
}

.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #ffb44a, #ff8c1a) !important;
  box-shadow: 0 4px 20px rgba(255,160,50,0.3) !important;
}

.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: #8b949e !important;
}

.stButton > button[kind="secondary"]:hover {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(255,255,255,0.2) !important;
  color: #f0f6fc !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
}

/* ── Info/Warning Boxes ── */
.stInfo {
  background: rgba(88,166,255,0.07) !important;
  border: 1px solid rgba(88,166,255,0.2) !important;
  border-radius: 8px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 12px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #ffa032 !important; }

/* ── Timestamp footer ── */
.rg-footer {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #2a3241;
  text-align: right;
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255,255,255,0.04);
  letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
def _init():
    defs = {
        "xgb_model": None, "gnn_model": None, "gnn_edge_order": None, "gnn_loss_history": [],
        "xgb_metrics": {}, "gnn_metrics": {}, "pipeline_data": {},
        "xgb_risks": {}, "gnn_risks": {}, "graph": None,
        "alert_engine": AlertEngine(), "blocked_edges": set(),
        "path": [], "path_risk": 0.0, "alerts": [], "notifications": [],
        "models_loaded": False, "xgb_ms": 0.0, "gnn_ms": 0.0, "override": None,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_models():
    from xgb_model import load_xgb_model, train_xgb_model
    from gnn_model  import load_gnn_model, train_gnn
    mx = {}
    if not os.path.exists(XGB_MODEL_PATH):
        with st.spinner("⚙  Training XGBoost (first run)…"):
            mx = train_xgb_model()
    xgb = load_xgb_model()
    mg = {}
    if not os.path.exists(GNN_MODEL_PATH):
        with st.spinner("🧠  Training GNN (first run)…"):
            mg = train_gnn()
    gnn, eo, lh = load_gnn_model()
    return xgb, gnn, eo, lh, mx, mg

def ensure_models():
    if not st.session_state.models_loaded:
        xgb, gnn, eo, lh, mx, mg = _load_models()
        st.session_state.xgb_model = xgb
        st.session_state.gnn_model = gnn
        st.session_state.gnn_edge_order = eo
        st.session_state.gnn_loss_history = lh
        st.session_state.xgb_metrics = mx
        st.session_state.gnn_metrics = mg
        st.session_state.models_loaded = True

# ── Prediction cycle ──────────────────────────────────────────────────────────
def run_prediction_cycle(override=None):
    from xgb_model  import predict_edge_risks
    from gnn_model  import build_networkx_graph, update_graph_risks, predict_gnn_risks
    from data_fetcher import fetch_all
    ensure_models()
    data = fetch_all(override=override)
    st.session_state.pipeline_data = data
    xgb_r, xms = predict_edge_risks(st.session_state.xgb_model, data["edge_features"])
    st.session_state.xgb_risks = xgb_r
    st.session_state.xgb_ms = xms
    G = build_networkx_graph(xgb_r)
    gnn_r, gms = predict_gnn_risks(
        st.session_state.gnn_model, st.session_state.gnn_edge_order, G, xgb_r)
    st.session_state.gnn_risks = gnn_r
    st.session_state.gnn_ms = gms
    update_graph_risks(G, gnn_r)
    st.session_state.graph = G
    ae = st.session_state.alert_engine
    blocked, _ = ae.process_risks(gnn_r)
    ae.process_yolo_results(data["yolo_results"])
    ae.process_iot_readings(data["iot_readings"])
    st.session_state.blocked_edges = blocked
    st.session_state.alerts = ae.get_log()
    st.session_state.notifications = ae.get_notifications()

def run_routing(src_id, dst_id):
    from gnn_model import find_ml_path
    G  = st.session_state.graph
    gr = st.session_state.gnn_risks
    if G is None or not gr:
        run_prediction_cycle(override=st.session_state.override)
        G  = st.session_state.graph
        gr = st.session_state.gnn_risks
    path, risk = find_ml_path(G, gr, src_id, dst_id,
                               blocked_edges=st.session_state.blocked_edges)
    st.session_state.path = path
    st.session_state.path_risk = risk

# ── Header ────────────────────────────────────────────────────────────────────
is_live = bool(st.session_state.pipeline_data)
dot_cls = "live-dot" if is_live else "amber-dot"
dot_lbl = "LIVE" if is_live else "STANDBY"

st.markdown(f"""
<div class="rg-header">
  <div class="rg-header-eyebrow">
    <span class="live-dot {dot_cls}"></span>
    ROADGUARD BLR &nbsp;·&nbsp; {dot_lbl} &nbsp;·&nbsp; BANGALORE TRAFFIC COMMAND
  </div>
  <div class="rg-header-title">Hyper-Local <span>Road Hazard</span> Detection</div>
  <div class="rg-header-sub">
    EDGE AI · YOLOV8 CCTV &nbsp;|&nbsp; IOT SENSOR FUSION &nbsp;|&nbsp;
    GRAPHSAGE GNN RISK REFINER &nbsp;|&nbsp; ZERO DIJKSTRA
  </div>
</div>
""", unsafe_allow_html=True)

junc_names = [JUNCTIONS[i]["name"] for i in range(len(JUNCTIONS))]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;font-size:22px;font-weight:700;
         color:#f0f6fc;letter-spacing:0.06em;margin-bottom:1.2rem;padding-bottom:10px;
         border-bottom:1px solid rgba(255,160,50,0.15);">
      🗺️ Route Planner
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-label">Origin Junction</span>', unsafe_allow_html=True)
    src_name = st.selectbox("", junc_names, index=0, key="src_sel", label_visibility="collapsed")

    st.markdown('<span class="sidebar-label">Destination Junction</span>', unsafe_allow_html=True)
    dst_name = st.selectbox("", junc_names, index=7, key="dst_sel", label_visibility="collapsed")

    src_id = junc_names.index(src_name)
    dst_id = junc_names.index(dst_name)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    find_btn    = c1.button("🚀 Find Route",   use_container_width=True, type="primary")
    refresh_btn = c2.button("🔄 Refresh",      use_container_width=True, type="secondary")

    st.markdown("<br>", unsafe_allow_html=True)
    flood_btn = st.button("🌊 Simulate Silk Board Flood",
                          use_container_width=True, type="secondary")
    clear_btn = st.button("🗑️  Clear All Alerts",
                          use_container_width=True, type="secondary")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Weather Widget ──
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;font-size:16px;font-weight:600;
         color:#ffa032;letter-spacing:0.06em;margin-bottom:6px;">
      🌤  Live Weather
    </div>
    """, unsafe_allow_html=True)

    w = st.session_state.pipeline_data.get("weather", {})
    if w:
        st.markdown(f"""
        <div class="weather-grid">
          <div class="weather-item">
            <div class="weather-icon">🌧</div>
            <div class="weather-val">{w.get('rain_mm',0)}</div>
            <div class="weather-lbl">mm/hr Rain</div>
          </div>
          <div class="weather-item">
            <div class="weather-icon">💨</div>
            <div class="weather-val">{w.get('wind_kmph',0)}</div>
            <div class="weather-lbl">km/h Wind</div>
          </div>
          <div class="weather-item">
            <div class="weather-icon">🌡</div>
            <div class="weather-val">{w.get('temperature_c',0)}°</div>
            <div class="weather-lbl">Celsius</div>
          </div>
          <div class="weather-item">
            <div class="weather-icon">👁</div>
            <div class="weather-val">{int(w.get('visibility_m',0))}</div>
            <div class="weather-lbl">m Visibility</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            'color:#3d4654;padding:8px 0;">Press Refresh to load data</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data Sources ──
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;font-size:16px;font-weight:600;
         color:#ffa032;letter-spacing:0.06em;margin-bottom:6px;">
      📡 Data Sources
    </div>
    <div class="sources-grid">
      <div class="source-pill"><span class="live-dot"></span> Open-Meteo Weather API</div>
      <div class="source-pill"><span class="live-dot"></span> Overpass Road Conditions</div>
      <div class="source-pill"><span class="live-dot"></span> YOLOv8 CCTV Detection</div>
      <div class="source-pill"><span class="live-dot"></span> IoT Sensor Network</div>
      <div class="source-pill"><span class="live-dot"></span> Crowdsource Reports</div>
    </div>
    """, unsafe_allow_html=True)

# ── Button handlers ───────────────────────────────────────────────────────────
if refresh_btn or not st.session_state.pipeline_data:
    with st.spinner("Fetching all data sources & running inference…"):
        run_prediction_cycle(override=st.session_state.override)
    st.rerun()

if flood_btn:
    st.session_state.override = {"rain_mm": 50, "has_closure": 1}
    st.session_state.alert_engine.inject_simulation_alert(
        "Silk Board Flood Simulation — rain_mm=50, closure=1", "CRITICAL")
    with st.spinner("Re-running models with flood scenario…"):
        run_prediction_cycle(override=st.session_state.override)
    st.rerun()

if find_btn:
    if src_id == dst_id:
        st.warning("⚠️  Source and destination must be different.")
    else:
        with st.spinner("Running ML greedy routing pipeline…"):
            if not st.session_state.gnn_risks:
                run_prediction_cycle(override=st.session_state.override)
            run_routing(src_id, dst_id)
        st.rerun()

if clear_btn:
    st.session_state.alert_engine.clear_log()
    st.session_state.alert_engine.clear_notifications()
    st.session_state.alerts = []
    st.session_state.notifications = []
    st.rerun()

# ── KPI Row ───────────────────────────────────────────────────────────────────
gnn_r    = st.session_state.gnn_risks
alerts   = st.session_state.alerts
path     = st.session_state.path

n_critical = sum(1 for a in alerts if a.get("level") == "CRITICAL")
n_warning  = sum(1 for a in alerts if a.get("level") == "WARNING")
n_edges    = len(gnn_r)
avg_risk   = round(sum(gnn_r.values()) / max(len(gnn_r), 1), 1) if gnn_r else 0.0
n_blocked  = len(st.session_state.blocked_edges)

accent_critical = "#f85149" if n_critical > 0 else "#22c55e"
accent_blocked  = "#f85149" if n_blocked  > 0 else "#22c55e"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card" style="--accent:{accent_critical}">
    <div class="kpi-label">Critical Alerts</div>
    <div class="kpi-value" style="color:{accent_critical}">{n_critical}</div>
    <div class="kpi-sub">{n_warning} warnings active</div>
  </div>
  <div class="kpi-card" style="--accent:#ffa032">
    <div class="kpi-label">Edges Monitored</div>
    <div class="kpi-value">{n_edges}</div>
    <div class="kpi-sub">directed road segments</div>
  </div>
  <div class="kpi-card" style="--accent:#58a6ff">
    <div class="kpi-label">Avg Network Risk</div>
    <div class="kpi-value">{avg_risk}</div>
    <div class="kpi-sub">GNN-refined · 0–100 scale</div>
  </div>
  <div class="kpi-card" style="--accent:{accent_blocked}">
    <div class="kpi-label">Blocked Routes</div>
    <div class="kpi-value" style="color:{accent_blocked}">{n_blocked}</div>
    <div class="kpi-sub">risk &gt; {RISK_CRITICAL} threshold</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Route Banner ──────────────────────────────────────────────────────────────
if path and len(path) > 1:
    path_names = " → ".join(JUNCTIONS[n]["name"] for n in path)
    travel     = estimate_travel_time_mins(path, st.session_state.gnn_risks)
    avg_r      = st.session_state.path_risk / max(len(path) - 1, 1)
    rl         = risk_label(avg_r)
    badge_col  = {"LOW": "#22c55e", "MODERATE": "#ffd700",
                  "HIGH": "#ffa032", "CRITICAL": "#f85149"}.get(rl, "#f85149")
    st.markdown(f"""
    <div class="route-banner">
      <div class="route-banner-title">◈ ML OPTIMAL PATH IDENTIFIED</div>
      <div class="route-path">{path_names}</div>
      <div class="route-stats">
        <div class="route-stat">
          <span class="route-stat-label">Total Risk Score</span>
          <span class="route-stat-val">{st.session_state.path_risk:.1f}
            <span class="risk-badge" style="background:{badge_col}22;color:{badge_col};
              border:1px solid {badge_col}55;font-size:11px;vertical-align:middle;">{rl}</span>
          </span>
        </div>
        <div class="route-stat">
          <span class="route-stat-label">Est. Travel Time</span>
          <span class="route-stat-val">~{travel} min</span>
        </div>
        <div class="route-stat">
          <span class="route-stat-label">Hops</span>
          <span class="route-stat-val">{len(path) - 1}</span>
        </div>
        <div class="route-stat">
          <span class="route-stat-label">Routing Method</span>
          <span class="route-stat-val" style="font-size:14px;color:#58a6ff;">
            GNN Greedy ML
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
elif path == [] and st.session_state.path_risk == float("inf"):
    st.error("❌  No ML path found — all routes blocked by CRITICAL risk edges.")

# ── Push Notifications Panel ──────────────────────────────────────────────────
notifs = st.session_state.notifications
if notifs:
    st.markdown("""
    <div class="section-header">
      <span class="section-tag">LIVE</span>
      <span class="section-header-text">Push Notifications</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(min(len(notifs), 3))
    for i, n in enumerate(notifs[:3]):
        lvl     = n.get("level", "LOW")
        card_cls = "critical" if lvl == "CRITICAL" else ("high" if lvl in ("HIGH", "WARNING") else "")
        lvl_dot  = '<span class="red-dot live-dot" style="display:inline-block"></span>' \
                   if lvl == "CRITICAL" else \
                   '<span class="amber-dot live-dot" style="display:inline-block"></span>'
        with cols[i]:
            st.markdown(f"""
            <div class="notif-card {card_cls}">
              <div class="notif-icon">{n.get('icon','⚠️')}</div>
              <div class="notif-content">
                <div class="notif-level">{lvl_dot}&nbsp; {lvl}</div>
                <div class="notif-title">{n.get('title','Alert')}</div>
                <div class="notif-body">{n.get('body','')}</div>
                <div class="notif-ts">{n.get('ts','')}</div>
              </div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Live Map",
    "📡  Sensor Data",
    "🤖  Model Insights",
    "🚨  Alert Log",
    "⚡  Performance",
])

# ── TAB 1: MAP ────────────────────────────────────────────────────────────────
with tab1:
    sel_path = set(zip(path, path[1:])) if path else set()
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    for (src, dst, rt, _) in EDGES:
        s, d = JUNCTIONS[src], JUNCTIONS[dst]
        risk  = gnn_r.get((src, dst), 50.0)
        is_p  = (src, dst) in sel_path
        colour = "#00c8ff" if is_p else risk_colour(risk)
        # White halo for readability on satellite background
        folium.PolyLine(
            [[s["lat"], s["lon"]], [d["lat"], d["lon"]]],
            color="white", weight=10 if is_p else 7, opacity=0.3,
        ).add_to(m)
        folium.PolyLine(
            [[s["lat"], s["lon"]], [d["lat"], d["lon"]]],
            color=colour,
            weight=7 if is_p else 4,
            opacity=1.0,
            dash_array=None if is_p else ("6,4" if risk > RISK_CRITICAL else None),
            tooltip=f"<b>{s['name']} → {d['name']}</b><br>Risk: {risk:.1f} — {risk_label(risk)} {risk_emoji(risk)}",
        ).add_to(m)

    for nid, info in JUNCTIONS.items():
        iot_r   = st.session_state.pipeline_data.get("iot_risks", {}).get(nid, 0)
        avg_r   = (sum(gnn_r.get((s, d), 50) for s, d, *_ in EDGES if s == nid)
                   / max(1, sum(1 for s, *_ in EDGES if s == nid)))
        in_p    = nid in path
        yolo    = st.session_state.pipeline_data.get("yolo_results", {}).get(nid, {})
        hazard  = yolo.get("primary_hazard", "none")
        r_col   = "#00c8ff" if in_p else risk_colour(avg_r)
        folium.CircleMarker(
            [info["lat"], info["lon"]],
            radius=13 if in_p else 8,
            color=r_col,
            fill=True,
            fill_color=r_col,
            fill_opacity=0.9,
            weight=2,
            tooltip=(f"<b>{info['name']}</b><br>"
                     f"GNN Risk: <b>{avg_r:.1f}</b><br>"
                     f"IoT Risk: {iot_r:.1f}<br>"
                     f"YOLO: {hazard}"),
        ).add_to(m)

    legend = """
    <div style="position:fixed;bottom:30px;right:12px;z-index:999;
      background:rgba(5,8,14,0.88);color:#e6edf3;
      padding:14px 18px;border-radius:10px;
      border:1px solid rgba(255,160,50,0.25);
      font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:2.1;
      backdrop-filter:blur(6px);box-shadow:0 4px 24px rgba(0,0,0,0.6);">
      <div style="color:#ffa032;font-weight:600;margin-bottom:6px;
           letter-spacing:0.12em;font-size:10px;">RISK LEGEND</div>
      🟢 Low &lt; 30<br>
      🟡 Moderate 30–50<br>
      🟠 High 50–80<br>
      🔴 Critical 80+<br>
      🔵 Selected Path
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    if not gnn_r:
        st.info("👆 Click **Refresh** in the sidebar to load live data and run inference.")
    st_folium(m, width="100%", height=580)

# ── TAB 2: SENSOR DATA ────────────────────────────────────────────────────────
with tab2:
    iot      = st.session_state.pipeline_data.get("iot_readings", {})
    yolo_res = st.session_state.pipeline_data.get("yolo_results", {})

    if iot:
        c_iot, c_yolo = st.columns(2)

        with c_iot:
            st.markdown("""
            <div class="section-header">
              <span class="section-tag">IOT</span>
              <span class="section-header-text">Sensor Readings</span>
            </div>
            """, unsafe_allow_html=True)

            iot_rows = []
            for jid, r in iot.items():
                flood_col = {"CRITICAL": "🔴", "HIGH": "🟠",
                             "MODERATE": "🟡", "LOW": "🟢"}.get(r["flood_risk"], "🟢")
                iot_rows.append({
                    "Junction":    JUNCTIONS[jid]["name"],
                    "Water mm":    r["water_depth_mm"],
                    "Vib g":       r["vibration_g"],
                    "Humidity %":  r["humidity_pct"],
                    "Vehicles":    r["vehicle_count"],
                    "Flood":       f'{flood_col} {r["flood_risk"]}',
                    "IoT Score":   r["iot_risk_score"],
                })
            df_iot = pd.DataFrame(iot_rows)
            st.dataframe(df_iot, use_container_width=True, height=400,
                         column_config={
                             "IoT Score": st.column_config.ProgressColumn(
                                 "IoT Score", min_value=0, max_value=100),
                         })

        with c_yolo:
            st.markdown("""
            <div class="section-header">
              <span class="section-tag">CCTV</span>
              <span class="section-header-text">YOLO Detections</span>
            </div>
            """, unsafe_allow_html=True)

            hazard_emoji = {
                "pothole": "🕳️", "waterlogging": "🌊", "fallen_tree": "🌳",
                "muddy_road": "🟤", "overflowing_drain": "💧",
                "road_collapse": "⚠️", "debris": "🪨", "none": "✅",
            }
            yolo_rows = []
            for jid, r in yolo_res.items():
                hz = r.get("primary_hazard", "none")
                yolo_rows.append({
                    "Junction":      JUNCTIONS[jid]["name"],
                    "Hazard":        f"{hazard_emoji.get(hz,'⚠️')} {hz.replace('_',' ').title()}",
                    "Confidence":    r.get("confidence", 0),
                    "Risk Contrib":  r.get("risk_contribution", 0),
                    "Detections":    len(r.get("detections", [])),
                })
            df_yolo = pd.DataFrame(yolo_rows)
            st.dataframe(df_yolo, use_container_width=True, height=400,
                         column_config={
                             "Confidence": st.column_config.ProgressColumn(
                                 "Confidence", min_value=0, max_value=1,
                                 format="%.0%"),
                             "Risk Contrib": st.column_config.ProgressColumn(
                                 "Risk Contrib", min_value=0, max_value=100),
                         })
    else:
        st.info("Click **Refresh** to load sensor data.")

# ── TAB 3: MODEL INSIGHTS ─────────────────────────────────────────────────────
with tab3:
    ca, cb = st.columns(2)

    with ca:
        st.markdown("""
        <div class="section-header">
          <span class="section-tag">XGB</span>
          <span class="section-header-text">Feature Importance</span>
        </div>
        """, unsafe_allow_html=True)

        fi = st.session_state.xgb_metrics.get("feature_importance", {})
        if fi:
            df_fi = pd.DataFrame(list(fi.items()),
                                 columns=["Feature", "Importance"]).sort_values("Importance")
            fig = px.bar(
                df_fi, x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=[[0, "#1a2a3a"], [0.5, "#ffa032"], [1, "#ff4500"]],
                template="plotly_dark",
            )
            fig.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                font=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            mx = st.session_state.xgb_metrics
            if mx:
                mc1, mc2 = st.columns(2)
                mc1.metric("RMSE", f"{mx.get('rmse', 0):.3f}")
                mc2.metric("R²",   f"{mx.get('r2', 0):.4f}")
        else:
            st.info("Click Refresh to train models and view feature importances.")

    with cb:
        st.markdown("""
        <div class="section-header">
          <span class="section-tag">GNN</span>
          <span class="section-header-text">Training Loss Curve</span>
        </div>
        """, unsafe_allow_html=True)

        lh = st.session_state.gnn_loss_history
        if lh:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(1, len(lh) + 1)),
                y=lh,
                mode="lines",
                line=dict(color="#22c55e", width=2),
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.07)",
                name="MSE Loss",
            ))
            fig2.update_layout(
                height=290,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.04)",
                           title_font_color="#4a5568"),
                yaxis=dict(title="MSE Loss", gridcolor="rgba(255,255,255,0.04)",
                           title_font_color="#4a5568"),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("GNN loss curve appears after training.")

        st.markdown("""
        <div class="section-header" style="margin-top:16px">
          <span class="section-tag">GRAPH</span>
          <span class="section-header-text">Edge Risk Table</span>
        </div>
        """, unsafe_allow_html=True)

        from gnn_model import graph_to_edge_table
        G = st.session_state.graph
        if G:
            df_edge = pd.DataFrame(graph_to_edge_table(G))
            st.dataframe(df_edge, use_container_width=True, height=270,
                         column_config={
                             "Risk": st.column_config.ProgressColumn(
                                 "Risk", min_value=0, max_value=100),
                         })
        else:
            st.info("Run prediction to populate the edge risk table.")

# ── TAB 4: ALERT LOG ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="section-header">
      <span class="section-tag">LIVE</span>
      <span class="section-header-text">Unified Alert Log</span>
    </div>
    """, unsafe_allow_html=True)

    alerts = st.session_state.alerts
    if alerts:
        # Summary bar
        n_c = sum(1 for a in alerts if a.get("level") == "CRITICAL")
        n_w = sum(1 for a in alerts if a.get("level") == "WARNING")
        n_i = len(alerts) - n_c - n_w
        st.markdown(f"""
        <div style="display:flex;gap:10px;margin-bottom:14px;">
          <div style="background:rgba(248,81,73,0.1);border:1px solid rgba(248,81,73,0.3);
               border-radius:6px;padding:6px 14px;font-family:'Rajdhani',sans-serif;
               font-size:15px;font-weight:700;color:#f85149;">
            🔴 {n_c} Critical
          </div>
          <div style="background:rgba(255,160,50,0.1);border:1px solid rgba(255,160,50,0.3);
               border-radius:6px;padding:6px 14px;font-family:'Rajdhani',sans-serif;
               font-size:15px;font-weight:700;color:#ffa032;">
            🟠 {n_w} Warning
          </div>
          <div style="background:rgba(88,166,255,0.07);border:1px solid rgba(88,166,255,0.2);
               border-radius:6px;padding:6px 14px;font-family:'Rajdhani',sans-serif;
               font-size:15px;font-weight:700;color:#58a6ff;">
            🔵 {n_i} Info
          </div>
        </div>
        """, unsafe_allow_html=True)

        src_badge = {
            "GNN":        ("badge-gnn",  "GNN"),
            "YOLO":       ("badge-yolo", "YOLO"),
            "IoT":        ("badge-iot",  "IoT"),
            "SIMULATION": ("badge-sim",  "SIM"),
        }
        div_cls = {
            "CRITICAL": "critical",
            "WARNING":  "warning",
        }

        for a in alerts[:60]:
            lvl  = a.get("level", "INFO")
            src  = a.get("source", "GNN")
            bcls, blbl = src_badge.get(src, ("badge-gnn", src))
            dcls = div_cls.get(lvl, "info")
            ts   = a.get("timestamp", "")
            msg  = a.get("message", "").replace("<", "&lt;").replace(">", "&gt;")

            st.markdown(f"""
            <div class="alert-entry {dcls}">
              <span class="alert-badge {bcls}">{blbl}</span>
              <span class="alert-msg">{msg}</span>
              <span class="alert-time">{ts[-8:] if ts else ''}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;
             font-family:'IBM Plex Mono',monospace;font-size:13px;color:#22c55e;">
          ✅ &nbsp; No active alerts — all routes clear
        </div>
        """, unsafe_allow_html=True)

# ── TAB 5: PERFORMANCE ────────────────────────────────────────────────────────
with tab5:
    xms = st.session_state.xgb_ms
    gms = st.session_state.gnn_ms

    st.markdown("""
    <div class="section-header">
      <span class="section-tag">METRICS</span>
      <span class="section-header-text">Pipeline Performance</span>
    </div>
    """, unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("XGBoost Inference", f"{xms:.2f} ms")
    pc2.metric("GNN Inference",     f"{gms:.2f} ms")
    pc3.metric("Total Pipeline",    f"{xms + gms:.2f} ms")

    st.markdown("<br>", unsafe_allow_html=True)

    fig3 = go.Figure(go.Bar(
        x=["XGBoost", "GNN GraphSAGE", "Total"],
        y=[xms, gms, xms + gms],
        marker=dict(
            color=["#58a6ff", "#22c55e", "#ffa032"],
            opacity=0.85,
            line=dict(width=0),
        ),
        text=[f"{v:.2f} ms" for v in [xms, gms, xms + gms]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=12, color="#8b949e"),
    ))
    fig3.update_layout(
        template="plotly_dark",
        yaxis_title="Latency (ms)",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        bargap=0.35,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <table class="perf-table">
      <thead>
        <tr>
          <th>Stage</th><th>Component</th><th>Input</th><th>Output</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>Data Fusion</td>
          <td>Weather + IoT + YOLO + Crowd</td>
          <td>15-feature edge vectors</td>
        </tr>
        <tr>
          <td>2</td>
          <td>XGBoost</td>
          <td>Edge feature vectors</td>
          <td>Base risk score 0–100</td>
        </tr>
        <tr>
          <td>3</td>
          <td>GraphSAGE GNN</td>
          <td>Graph + XGB scores</td>
          <td>Spatially-refined risk</td>
        </tr>
        <tr>
          <td>4</td>
          <td>Greedy ML Router</td>
          <td>GNN risk map</td>
          <td>Safest path (no Dijkstra)</td>
        </tr>
        <tr>
          <td>5</td>
          <td>Alert Engine</td>
          <td>Risk + Hazard data</td>
          <td>Push notifications</td>
        </tr>
      </tbody>
    </table>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#4a5568;
         margin-top:14px;padding:10px 14px;background:rgba(248,81,73,0.05);
         border:1px solid rgba(248,81,73,0.15);border-radius:6px;">
      ⛔ Zero traditional shortest-path algorithms. All routing decisions are
      driven by XGBoost baseline + GraphSAGE GNN spatial refinement.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
if st.session_state.pipeline_data:
    st.markdown(
        f'<div class="rg-footer">LAST UPDATED · {ist_timestamp()} · ROADGUARD BLR v2.0</div>',
        unsafe_allow_html=True,
    )
