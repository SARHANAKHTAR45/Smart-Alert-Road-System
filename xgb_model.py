"""
xgb_model.py — XGBoost Risk Predictor
Hyper-Local Smart Road Hazard Detection & Real-Time Alert System

Stage 1 of the ML pipeline:
  Input:  per-edge feature vectors (weather + IoT + YOLO + crowd)
  Output: baseline risk score 0–100 for every edge
"""

import os, time, logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from config import (
    EDGES, JUNCTIONS, HISTORICAL_ACCIDENTS, JUNCTION_BASE_PENALTY,
    XGB_MODEL_PATH, XGB_PARAMS, SYNTH_SAMPLES,
    PEAK_HOURS, MONSOON_MONTHS, HAZARD_SEVERITY,
)
from utils import clamp

logger = logging.getLogger(__name__)

# Feature columns — must match data_fetcher.build_edge_features keys
FEATURE_COLS = [
    "rain_mm", "wind_kmph", "visibility_m", "temperature_c",
    "hour_of_day", "is_peak_hour", "is_night", "is_weekend",
    "road_type", "has_construction", "has_closure",
    "historical_accidents", "yolo_risk", "iot_risk", "crowd_severity",
]


# ─────────────────────────────────────────────
# Synthetic training data generator
# ─────────────────────────────────────────────

def _generate_training_data(n_samples: int = SYNTH_SAMPLES) -> pd.DataFrame:
    import random

    rows = []
    edge_list = list(EDGES)

    for _ in range(n_samples):
        src, dst, road_type, length_km = random.choice(edge_list)

        rain_mm       = float(np.random.choice(
            [0, 0, 0, np.random.uniform(0.5, 5),
             np.random.uniform(5, 20), np.random.uniform(20, 80)],
            p=[0.30, 0.15, 0.10, 0.20, 0.15, 0.10],
        ))
        wind_kmph     = float(np.random.uniform(0, 60))
        visibility_m  = float(np.random.uniform(500, 10000))
        temperature_c = float(np.random.uniform(18, 40))
        hour_of_day   = random.randint(0, 23)
        is_peak_hour  = int(hour_of_day in PEAK_HOURS)
        is_night      = int(hour_of_day in (list(range(22, 24)) + list(range(0, 6))))
        is_weekend    = random.randint(0, 1)
        has_construction = random.choices([0, 1], weights=[0.85, 0.15])[0]
        has_closure   = random.choices([0, 1], weights=[0.90, 0.10])[0]
        historical_acc = HISTORICAL_ACCIDENTS.get(src, 3)
        yolo_risk     = float(np.random.uniform(0, 95) if rain_mm > 5 else np.random.uniform(0, 40))
        iot_risk      = float(np.random.uniform(0, 80) if rain_mm > 10 else np.random.uniform(0, 30))
        crowd_severity = float(np.random.uniform(0, 70) if rain_mm > 2 else np.random.uniform(0, 20))

        # Ground truth risk formula
        risk = (
            rain_mm * 1.2 +
            (60 - wind_kmph) * 0.1 +
            max(0, (3000 - visibility_m) / 100) +
            is_peak_hour * 8 +
            is_night * 5 +
            has_construction * 15 +
            has_closure * 20 +
            historical_acc * 2.5 +
            JUNCTION_BASE_PENALTY.get(src, 0) * 2 +
            yolo_risk * 0.35 +
            iot_risk * 0.30 +
            crowd_severity * 0.20 +
            (road_type == 2) * 5  # Local roads slightly riskier
        )
        risk = clamp(risk + float(np.random.normal(0, 3)), 0, 100)

        rows.append({
            "rain_mm": rain_mm, "wind_kmph": wind_kmph,
            "visibility_m": visibility_m, "temperature_c": temperature_c,
            "hour_of_day": hour_of_day, "is_peak_hour": is_peak_hour,
            "is_night": is_night, "is_weekend": is_weekend,
            "road_type": road_type, "has_construction": has_construction,
            "has_closure": has_closure, "historical_accidents": historical_acc,
            "yolo_risk": yolo_risk, "iot_risk": iot_risk,
            "crowd_severity": crowd_severity, "risk": risk,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_xgb_model() -> dict:
    """Train XGBoost (GradientBoosting) model and persist to disk."""
    logger.info("Training XGBoost risk model…")
    os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)

    df = _generate_training_data()
    X  = df[FEATURE_COLS]
    y  = df["risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators  = XGB_PARAMS["n_estimators"],
        max_depth     = XGB_PARAMS["max_depth"],
        learning_rate = XGB_PARAMS["learning_rate"],
        random_state  = XGB_PARAMS["random_state"],
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_test, y_pred))

    fi = dict(zip(FEATURE_COLS, model.feature_importances_))
    joblib.dump(model, XGB_MODEL_PATH)

    metrics = {
        "rmse":               round(rmse, 4),
        "r2":                 round(r2, 4),
        "n_train":            len(X_train),
        "feature_importance": {k: round(float(v), 4) for k, v in fi.items()},
    }
    logger.info(f"XGB trained: RMSE={rmse:.3f}, R²={r2:.4f}")
    return metrics


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def load_xgb_model():
    return joblib.load(XGB_MODEL_PATH)


def predict_edge_risks(model, edge_features: dict) -> tuple[dict, float]:
    """
    Run XGBoost inference over all edges.

    Returns
    -------
    xgb_risks : {(src, dst): risk_score}
    infer_ms  : inference latency in milliseconds
    """
    edges   = list(edge_features.keys())
    records = [edge_features[e] for e in edges]

    df = pd.DataFrame(records, columns=FEATURE_COLS)
    # Fill any missing new columns with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    t0     = time.perf_counter()
    preds  = model.predict(df[FEATURE_COLS])
    infer_ms = (time.perf_counter() - t0) * 1000

    xgb_risks = {
        edge: clamp(float(pred), 0, 100)
        for edge, pred in zip(edges, preds)
    }
    return xgb_risks, round(infer_ms, 2)
