"""
alert_engine.py — Multi-channel Alert & Notification Engine
Hyper-Local Smart Road Hazard Detection & Real-Time Alert System

Handles:
  - Risk threshold monitoring → CRITICAL / WARNING / INFO alerts
  - Push notification simulation (mobile-style toast cards)
  - Crowdsource alert injection
  - Hazard type tagging per alert
"""

import logging
from collections import deque
from utils import ist_timestamp, risk_label, risk_emoji, format_push_notification
from config import RISK_CRITICAL, RISK_WARNING, JUNCTIONS

logger = logging.getLogger(__name__)

MAX_LOG_SIZE    = 100
MAX_NOTIF_SIZE  = 20


class AlertEngine:
    """
    Unified alert engine watching:
      1. GNN edge risk scores (threshold-based)
      2. YOLO hazard detections (class-based)
      3. IoT sensor anomalies (flood / vibration thresholds)
    """

    def __init__(self):
        self._log:   deque[dict] = deque(maxlen=MAX_LOG_SIZE)
        self._notif: deque[dict] = deque(maxlen=MAX_NOTIF_SIZE)

    # ─────────────────────────────────────────────
    # 1. GNN Risk Monitoring
    # ─────────────────────────────────────────────

    def process_risks(self, gnn_risks: dict) -> tuple[set, list]:
        blocked_edges = set()
        new_alerts    = []

        for (src, dst), risk in gnn_risks.items():
            src_name = JUNCTIONS[src]["name"]
            dst_name = JUNCTIONS[dst]["name"]
            edge_str = f"{src_name} → {dst_name}"

            if risk > RISK_CRITICAL:
                alert = self._make_alert(edge_str, risk, "CRITICAL", "Route blocked — rerouting")
                blocked_edges.add((src, dst))
                self._log.appendleft(alert)
                new_alerts.append(alert)

                # Push notification
                notif = format_push_notification("road_collapse", src_name, risk)
                notif["edge"] = edge_str
                self._notif.appendleft(notif)

            elif risk > RISK_WARNING:
                alert = self._make_alert(edge_str, risk, "WARNING", "Proceed with caution")
                self._log.appendleft(alert)
                new_alerts.append(alert)

        return blocked_edges, new_alerts

    # ─────────────────────────────────────────────
    # 2. YOLO Detection Alerts
    # ─────────────────────────────────────────────

    def process_yolo_results(self, yolo_results: dict) -> list:
        """Fire alerts for YOLO-detected hazards with high confidence."""
        new_alerts = []
        for jid, result in yolo_results.items():
            hazard    = result.get("primary_hazard", "none")
            conf      = result.get("confidence", 0.0)
            risk_val  = result.get("risk_contribution", 0.0)
            junc_name = JUNCTIONS[jid]["name"]

            if hazard != "none" and conf >= 0.65:
                level  = risk_label(risk_val)
                emoji  = risk_emoji(risk_val)
                msg    = (
                    f"[CCTV/{junc_name}] {emoji} {hazard.replace('_',' ').upper()} "
                    f"detected — confidence {conf:.0%}, risk contribution {risk_val:.0f}"
                )
                alert = {
                    "timestamp": ist_timestamp(),
                    "level":     level,
                    "source":    "YOLO",
                    "junction":  junc_name,
                    "hazard":    hazard,
                    "message":   msg,
                }
                self._log.appendleft(alert)
                new_alerts.append(alert)

                if risk_val >= RISK_WARNING:
                    notif = format_push_notification(hazard, junc_name, risk_val)
                    self._notif.appendleft(notif)

        return new_alerts

    # ─────────────────────────────────────────────
    # 3. IoT Sensor Alerts
    # ─────────────────────────────────────────────

    def process_iot_readings(self, iot_readings: dict) -> list:
        """Fire alerts for critical IoT sensor anomalies."""
        new_alerts = []
        for jid, reading in iot_readings.items():
            junc_name = JUNCTIONS[jid]["name"]
            water     = reading.get("water_depth_mm", 0)
            vib       = reading.get("vibration_g", 0)
            flood_r   = reading.get("flood_risk", "LOW")
            iot_score = reading.get("iot_risk_score", 0)

            if flood_r == "CRITICAL":
                msg = (
                    f"[IoT/{junc_name}] 🌊 FLOOD ALERT — water depth {water}mm, "
                    f"IoT risk {iot_score:.0f}/100"
                )
                alert = {
                    "timestamp": ist_timestamp(), "level": "CRITICAL",
                    "source": "IoT", "junction": junc_name,
                    "message": msg,
                }
                self._log.appendleft(alert)
                new_alerts.append(alert)
                notif = format_push_notification("waterlogging", junc_name, iot_score)
                self._notif.appendleft(notif)

            elif vib > 0.35:
                msg = (
                    f"[IoT/{junc_name}] 🕳️ HIGH VIBRATION — {vib:.3f}g "
                    f"(possible pothole / road damage)"
                )
                alert = {
                    "timestamp": ist_timestamp(), "level": "WARNING",
                    "source": "IoT", "junction": junc_name,
                    "message": msg,
                }
                self._log.appendleft(alert)
                new_alerts.append(alert)

        return new_alerts

    # ─────────────────────────────────────────────
    # Simulation & Manual Injection
    # ─────────────────────────────────────────────

    def inject_simulation_alert(self, message: str, level: str = "WARNING") -> None:
        alert = {
            "timestamp": ist_timestamp(),
            "level":     level,
            "source":    "SIMULATION",
            "message":   f"[{level}][SIM][{ist_timestamp()}] {message}",
        }
        self._log.appendleft(alert)

    # ─────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────

    def get_log(self) -> list:
        return list(self._log)

    def get_notifications(self) -> list:
        return list(self._notif)

    def clear_log(self) -> None:
        self._log.clear()

    def clear_notifications(self) -> None:
        self._notif.clear()

    # ─────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────

    @staticmethod
    def _make_alert(edge_str: str, risk: float, level: str, detail: str) -> dict:
        ts  = ist_timestamp()
        emoji = risk_emoji(risk)
        msg = f"[{level}][{ts}] {emoji} {edge_str}  Risk={risk:.1f} — {detail}"
        return {
            "timestamp": ts,
            "level":     level,
            "source":    "GNN",
            "edge":      edge_str,
            "risk":      round(risk, 1),
            "message":   msg,
        }
