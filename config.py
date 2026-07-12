"""
config.py — Central configuration for Hyper-Local Smart Road Hazard Detection
           & Real-Time Alert System (Bangalore)
"""

import pytz

# ─────────────────────────────────────────────
# Timezone
# ─────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

# ─────────────────────────────────────────────
# Bangalore Center & Bounding Box
# ─────────────────────────────────────────────
BANGALORE_CENTER = (12.9716, 77.5946)
BANGALORE_BBOX   = (12.8340, 77.4600, 13.1390, 77.7840)

# ─────────────────────────────────────────────
# Junction Definitions  (id → metadata)
# ─────────────────────────────────────────────
JUNCTIONS = {
    0:  {"name": "Silk Board Junction",   "lat": 12.9176, "lon": 77.6228, "is_major": 1, "zone": "south"},
    1:  {"name": "KR Puram",              "lat": 13.0050, "lon": 77.6960, "is_major": 1, "zone": "east"},
    2:  {"name": "Hebbal Flyover",         "lat": 13.0450, "lon": 77.5970, "is_major": 1, "zone": "north"},
    3:  {"name": "Marathahalli Bridge",    "lat": 12.9563, "lon": 77.7009, "is_major": 1, "zone": "east"},
    4:  {"name": "Whitefield",             "lat": 12.9698, "lon": 77.7499, "is_major": 0, "zone": "east"},
    5:  {"name": "Electronic City",        "lat": 12.8399, "lon": 77.6770, "is_major": 1, "zone": "south"},
    6:  {"name": "Koramangala",            "lat": 12.9352, "lon": 77.6245, "is_major": 1, "zone": "south"},
    7:  {"name": "Indiranagar 100ft Road", "lat": 12.9784, "lon": 77.6408, "is_major": 0, "zone": "central"},
    8:  {"name": "MG Road",                "lat": 12.9757, "lon": 77.6011, "is_major": 1, "zone": "central"},
    9:  {"name": "Yeshwantpur Junction",   "lat": 13.0234, "lon": 77.5548, "is_major": 1, "zone": "west"},
    10: {"name": "Bannerghatta Road",      "lat": 12.8933, "lon": 77.5976, "is_major": 0, "zone": "south"},
    11: {"name": "Bellary Road Hebbal",    "lat": 13.0489, "lon": 77.5941, "is_major": 0, "zone": "north"},
    12: {"name": "ORR Bellandur",          "lat": 12.9344, "lon": 77.6855, "is_major": 1, "zone": "east"},
    13: {"name": "HSR Layout",             "lat": 12.9116, "lon": 77.6389, "is_major": 0, "zone": "south"},
    14: {"name": "JP Nagar",              "lat": 12.9063, "lon": 77.5858, "is_major": 0, "zone": "south"},
}

JUNCTION_NAMES = {v["name"]: k for k, v in JUNCTIONS.items()}
JUNCTION_IDS   = list(JUNCTIONS.keys())

# ─────────────────────────────────────────────
# Directed Edges (src_id, dst_id, road_type, length_km)
# road_type: 0=arterial, 1=highway, 2=local
# ─────────────────────────────────────────────
EDGES = [
    (0, 6,  0, 3.2),  (0, 13, 0, 2.8),  (0, 5,  1, 8.5),
    (1, 3,  0, 6.4),  (1, 4,  1, 9.1),  (1, 7,  0, 7.2),
    (2, 9,  0, 5.3),  (2, 11, 1, 1.2),  (2, 8,  0, 8.9),
    (3, 4,  0, 4.5),  (3, 12, 0, 3.3),
    (4, 5,  1, 14.2),
    (5, 10, 0, 6.1),  (5, 13, 0, 7.4),
    (6, 7,  0, 4.6),  (6, 13, 2, 2.1),
    (7, 8,  0, 3.8),  (7, 1,  0, 7.2),
    (8, 9,  0, 9.5),  (8, 2,  1, 8.7),
    (9, 11, 1, 5.8),  (9, 14, 0, 6.4),
    (10, 14, 0, 3.9), (10, 13, 0, 2.7),
    (12, 0, 1, 5.2),  (12, 3, 0, 3.3),
    (13, 6, 2, 2.1),  (13, 10, 0, 2.7),
    (14, 10, 0, 3.9), (14, 0, 0, 4.7),
]

# ─────────────────────────────────────────────
# Hazard Types  (for YOLO + IoT simulation)
# ─────────────────────────────────────────────
HAZARD_TYPES = [
    "pothole",
    "waterlogging",
    "fallen_tree",
    "muddy_road",
    "overflowing_drain",
    "road_collapse",
    "debris",
    "none",
]

HAZARD_SEVERITY = {
    "pothole":           35,
    "waterlogging":      55,
    "fallen_tree":       85,
    "muddy_road":        40,
    "overflowing_drain": 65,
    "road_collapse":     95,
    "debris":            50,
    "none":               0,
}

# ─────────────────────────────────────────────
# IoT Sensor IDs  (one per junction, simulated)
# ─────────────────────────────────────────────
IOT_SENSORS = {jid: f"SENSOR_BLR_{jid:03d}" for jid in JUNCTIONS}

# ─────────────────────────────────────────────
# Historical Accident Scores (per junction)
# ─────────────────────────────────────────────
HISTORICAL_ACCIDENTS = {
    0: 9, 1: 7, 2: 6,  3: 7, 4: 5,
    5: 4, 6: 5, 7: 3,  8: 4, 9: 5,
    10: 4, 11: 4, 12: 8, 13: 3, 14: 3,
}

JUNCTION_BASE_PENALTY = {0: 5, 12: 3}   # Chronic congestion points

# ─────────────────────────────────────────────
# Alert Thresholds
# ─────────────────────────────────────────────
RISK_CRITICAL = 80
RISK_WARNING  = 50
RISK_LOW      = 30

# ─────────────────────────────────────────────
# Open-Meteo API
# ─────────────────────────────────────────────
OPENMETEO_URL    = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_PARAMS = {
    "latitude":  12.9716,
    "longitude": 77.5946,
    "current":   "rain,wind_speed_10m,visibility,temperature_2m",
    "timezone":  "Asia/Kolkata",
}

# ─────────────────────────────────────────────
# Overpass API
# ─────────────────────────────────────────────
OVERPASS_URL   = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """
[out:json][timeout:25];
(
  way["construction"](12.8340,77.4600,13.1390,77.7840);
  way["access"="no"](12.8340,77.4600,13.1390,77.7840);
  node["barrier"="gate"](12.8340,77.4600,13.1390,77.7840);
);
out body;
"""

# ─────────────────────────────────────────────
# Model Paths
# ─────────────────────────────────────────────
XGB_MODEL_PATH  = "models/xgb_risk_model.pkl"
GNN_MODEL_PATH  = "models/gnn_risk_model.pt"
YOLO_MODEL_PATH = "models/yolo_hazard_model.pkl"   # Simulated YOLO weights

# ─────────────────────────────────────────────
# Training Params
# ─────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":  200,
    "max_depth":     6,
    "learning_rate": 0.05,
    "random_state":  42,
}
GNN_EPOCHS    = 100
GNN_LR        = 0.001
SYNTH_SAMPLES = 10_000

# ─────────────────────────────────────────────
# Peak / Night / Monsoon
# ─────────────────────────────────────────────
PEAK_HOURS     = list(range(8, 11)) + list(range(17, 21))
NIGHT_HOURS    = list(range(22, 24)) + list(range(0, 6))
MONSOON_MONTHS = [6, 7, 8, 9]

# ─────────────────────────────────────────────
# Simulated CCTV Camera Locations (lat, lon)
# ─────────────────────────────────────────────
CCTV_CAMERAS = {
    jid: {"lat": info["lat"] + 0.001, "lon": info["lon"] + 0.001, "junction_id": jid}
    for jid, info in JUNCTIONS.items()
}
