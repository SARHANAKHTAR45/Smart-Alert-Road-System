"""
yolo_detector.py — Simulated YOLOv8 Edge-AI Hazard Detector
Hyper-Local Smart Road Hazard Detection & Real-Time Alert System

In a real deployment this module would:
  1. Connect to CCTV/dash-cam feeds via RTSP streams
  2. Run YOLOv8 (ultralytics) inference frame-by-frame
  3. Return bounding boxes + class labels

Here we simulate realistic detection outputs using weighted random sampling
seeded by weather conditions — rain/wind increases hazard probability.
"""

import random
import numpy as np
from config import (
    JUNCTIONS, HAZARD_TYPES, HAZARD_SEVERITY, CCTV_CAMERAS,
    MONSOON_MONTHS,
)
from utils import ist_timestamp, is_monsoon


# ─────────────────────────────────────────────
# Simulated YOLO confidence model
# ─────────────────────────────────────────────

class YOLOHazardDetector:
    """
    Simulates a YOLOv8 multi-class object detector trained on road hazard images.

    Classes detected:
        pothole, waterlogging, fallen_tree, muddy_road,
        overflowing_drain, road_collapse, debris, none

    In production: replace `detect()` with actual ultralytics YOLOv8 call:
        model = YOLO('yolov8n_hazard.pt')
        results = model(frame)
    """

    # Base hazard probability per class under dry conditions
    _BASE_PROBS = {
        "pothole":           0.08,
        "waterlogging":      0.04,
        "fallen_tree":       0.02,
        "muddy_road":        0.05,
        "overflowing_drain": 0.03,
        "road_collapse":     0.01,
        "debris":            0.06,
        "none":              0.71,
    }

    def __init__(self, weather: dict | None = None):
        self.weather = weather or {}
        self._rain_factor = self._compute_rain_factor()

    def _compute_rain_factor(self) -> float:
        """Rain amplifies waterlogging/muddy/drain hazard probability."""
        rain = self.weather.get("rain_mm", 0)
        if rain > 30:
            return 3.5
        elif rain > 10:
            return 2.0
        elif rain > 0:
            return 1.3
        return 1.0

    def _compute_junction_weight(self, junction_id: int) -> float:
        """High-accident junctions get a higher baseline detection weight."""
        from config import HISTORICAL_ACCIDENTS
        acc_score = HISTORICAL_ACCIDENTS.get(junction_id, 3)
        return 1.0 + (acc_score / 20.0)   # 1.0 → 1.45 range

    def detect(self, junction_id: int, num_frames: int = 5) -> dict:
        """
        Simulate multi-frame YOLO detection at a junction's CCTV camera.

        Returns
        -------
        dict:
            junction_id, camera_id, detections (list),
            primary_hazard, confidence, risk_contribution, timestamp
        """
        junc_weight = self._compute_junction_weight(junction_id)

        # Build weighted probability distribution
        probs = {}
        for cls, base_p in self._BASE_PROBS.items():
            if cls in ("waterlogging", "muddy_road", "overflowing_drain"):
                probs[cls] = base_p * self._rain_factor * junc_weight
            elif cls == "none":
                probs[cls] = base_p
            else:
                probs[cls] = base_p * junc_weight

        # Monsoon → extra boost
        if is_monsoon():
            for cls in ("waterlogging", "overflowing_drain", "muddy_road"):
                probs[cls] *= 1.5

        # Normalize to a valid distribution
        total = sum(probs.values())
        norm_probs = {k: v / total for k, v in probs.items()}

        # Simulate detections over multiple frames
        classes = list(norm_probs.keys())
        weights = [norm_probs[c] for c in classes]
        detections = []

        for _ in range(num_frames):
            detected_class = random.choices(classes, weights=weights, k=1)[0]
            if detected_class != "none":
                confidence = round(random.uniform(0.55, 0.97), 2)
                detections.append({
                    "class":      detected_class,
                    "confidence": confidence,
                    "bbox":       self._fake_bbox(),
                })

        # Aggregate: pick highest-severity detection as primary
        primary_hazard = "none"
        best_conf      = 0.0
        if detections:
            detections.sort(key=lambda d: HAZARD_SEVERITY[d["class"]] * d["confidence"], reverse=True)
            primary_hazard = detections[0]["class"]
            best_conf      = detections[0]["confidence"]

        base_risk = HAZARD_SEVERITY[primary_hazard]
        risk_contribution = round(base_risk * best_conf, 1)

        return {
            "junction_id":       junction_id,
            "camera_id":         CCTV_CAMERAS[junction_id],
            "detections":        detections,
            "primary_hazard":    primary_hazard,
            "confidence":        best_conf,
            "risk_contribution": risk_contribution,
            "timestamp":         ist_timestamp(),
        }

    @staticmethod
    def _fake_bbox() -> list:
        """Simulate a bounding box [x1, y1, x2, y2] in pixel space (640×480)."""
        x1 = random.randint(50, 400)
        y1 = random.randint(50, 300)
        x2 = x1 + random.randint(40, 150)
        y2 = y1 + random.randint(30, 120)
        return [x1, y1, min(x2, 639), min(y2, 479)]

    def scan_all_junctions(self) -> dict:
        """Run simulated detection across all CCTV cameras. Returns {jid: result}."""
        results = {}
        for jid in JUNCTIONS:
            results[jid] = self.detect(jid)
        return results
