"""
data_fetcher.py — Multi-source data pipeline
Hyper-Local Smart Road Hazard Detection & Real-Time Alert System

Sources:
  1. Open-Meteo API  → live weather
  2. Overpass API    → road construction / closures
  3. IoT Sensor Network (simulated)
  4. YOLOv8 CCTV Detection (simulated)
  5. Crowdsource reports (simulated)
"""

import requests
import random
import logging
from config import (
    OPENMETEO_URL, OPENMETEO_PARAMS,
    OVERPASS_URL, OVERPASS_QUERY,
    JUNCTIONS, EDGES, HISTORICAL_ACCIDENTS,
    JUNCTION_BASE_PENALTY, HAZARD_SEVERITY,
)
from utils import get_time_features, is_monsoon, clamp

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Weather  (Open-Meteo — no API key needed)
# ─────────────────────────────────────────────

def fetch_weather() -> dict:
    defaults = {"rain_mm": 0.0, "wind_kmph": 10.0, "visibility_m": 5000.0, "temperature_c": 28.0}
    try:
        resp = requests.get(OPENMETEO_URL, params=OPENMETEO_PARAMS, timeout=10)
        resp.raise_for_status()
        current = resp.json().get("current", {})

        rain_mm       = float(current.get("rain", 0) or 0)
        wind_kmph     = float(current.get("wind_speed_10m", 10) or 10)
        visibility_m  = float(current.get("visibility", 5000) or 5000)
        temperature_c = float(current.get("temperature_2m", 28) or 28)

        if is_monsoon():
            rain_mm = rain_mm * 1.5

        return {
            "rain_mm":       round(rain_mm, 2),
            "wind_kmph":     round(wind_kmph, 1),
            "visibility_m":  round(visibility_m, 0),
            "temperature_c": round(temperature_c, 1),
        }
    except Exception as e:
        logger.warning(f"[WeatherFetcher] Failed: {e}. Using defaults.")
        return defaults


# ─────────────────────────────────────────────
# 2. Road Conditions  (Overpass)
# ─────────────────────────────────────────────

def fetch_road_conditions() -> dict:
    defaults = {"has_construction": 0, "has_closure": 0}
    try:
        resp = requests.post(OVERPASS_URL, data=OVERPASS_QUERY, timeout=30)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])

        has_construction = 0
        has_closure      = 0
        for el in elements:
            tags = el.get("tags", {})
            if "construction" in tags or tags.get("highway") == "construction":
                has_construction = 1
            if tags.get("access") == "no" or tags.get("barrier") == "gate":
                has_closure = 1

        return {"has_construction": has_construction, "has_closure": has_closure}
    except Exception as e:
        logger.warning(f"[OverpassFetcher] Failed: {e}. Using defaults.")
        return defaults


# ─────────────────────────────────────────────
# 3. Crowdsource Reports (simulated)
# ─────────────────────────────────────────────

def fetch_crowdsource_reports(weather: dict) -> dict:
    """
    Simulate crowdsourced hazard reports (like Waze/Google Maps user pins).
    In production: poll a Firebase / REST endpoint for user submissions.

    Returns {junction_id: {"count": int, "avg_severity": float}}
    """
    reports = {}
    rain = weather.get("rain_mm", 0)
    for jid in JUNCTIONS:
        # More reports when it rains
        base_count = max(0, int(random.gauss(2 + rain * 0.3, 1.5)))
        if base_count > 0:
            severities = [random.uniform(10, 90) for _ in range(base_count)]
            reports[jid] = {
                "count":        base_count,
                "avg_severity": round(sum(severities) / len(severities), 1),
            }
        else:
            reports[jid] = {"count": 0, "avg_severity": 0.0}
    return reports


# ─────────────────────────────────────────────
# 4. Build fused edge feature vectors
# ─────────────────────────────────────────────

def build_edge_features(
    weather: dict,
    road_conditions: dict,
    yolo_results: dict,
    iot_risks: dict,
    crowdsource: dict,
    override: dict | None = None,
) -> dict:
    """
    Build a multi-source fused feature vector for every directed edge.

    Feature set (12 cols):
        rain_mm, wind_kmph, visibility_m, temperature_c,
        hour_of_day, is_peak_hour, is_night, is_weekend,
        road_type, has_construction, has_closure,
        historical_accidents,
        yolo_risk,      ← from CCTV / Edge AI
        iot_risk,       ← from sensor network
        crowd_severity  ← from crowdsource reports
    """
    time_features = get_time_features()

    w  = {**weather}
    rc = {**road_conditions}
    if override:
        for k, v in override.items():
            if k in w:
                w[k] = v
            elif k in rc:
                rc[k] = v

    edge_features = {}
    for (src, dst, road_type, _length_km) in EDGES:
        rain_adj = w["rain_mm"] + JUNCTION_BASE_PENALTY.get(src, 0) * 0.2

        # YOLO detection contributes risk from source junction camera
        yolo_r    = yolo_results.get(src, {}).get("risk_contribution", 0.0)

        # IoT: average of src and dst sensor risks
        iot_src   = iot_risks.get(src, 0.0)
        iot_dst   = iot_risks.get(dst, 0.0)
        iot_r     = (iot_src + iot_dst) / 2.0

        # Crowdsource: avg severity at source junction
        crowd_r   = crowdsource.get(src, {}).get("avg_severity", 0.0)

        feat = {
            "rain_mm":              clamp(rain_adj, 0, 200),
            "wind_kmph":            w["wind_kmph"],
            "visibility_m":         w["visibility_m"],
            "temperature_c":        w["temperature_c"],
            "hour_of_day":          time_features["hour_of_day"],
            "is_peak_hour":         time_features["is_peak_hour"],
            "is_night":             time_features["is_night"],
            "is_weekend":           time_features["is_weekend"],
            "road_type":            road_type,
            "has_construction":     rc["has_construction"],
            "has_closure":          rc["has_closure"],
            "historical_accidents": HISTORICAL_ACCIDENTS[src],
            "yolo_risk":            round(yolo_r, 1),
            "iot_risk":             round(iot_r, 1),
            "crowd_severity":       round(crowd_r, 1),
        }
        edge_features[(src, dst)] = feat

    return edge_features


# ─────────────────────────────────────────────
# 5. Convenience entry-point
# ─────────────────────────────────────────────

def fetch_all(override: dict | None = None) -> dict:
    """
    Fetch all data sources and return a single pipeline dict.
    """
    from yolo_detector import YOLOHazardDetector
    from iot_sensors   import IoTSensorNetwork

    weather         = fetch_weather()
    road_conditions = fetch_road_conditions()
    crowdsource     = fetch_crowdsource_reports(weather)

    detector   = YOLOHazardDetector(weather=weather)
    yolo_raw   = detector.scan_all_junctions()

    iot_net    = IoTSensorNetwork(weather=weather)
    iot_raw    = iot_net.read_all()
    iot_risks  = {jid: r["iot_risk_score"] for jid, r in iot_raw.items()}

    edge_features = build_edge_features(
        weather, road_conditions,
        yolo_raw, iot_risks, crowdsource,
        override=override,
    )

    return {
        "weather":         weather,
        "road_conditions": road_conditions,
        "yolo_results":    yolo_raw,
        "iot_readings":    iot_raw,
        "iot_risks":       iot_risks,
        "crowdsource":     crowdsource,
        "edge_features":   edge_features,
    }
