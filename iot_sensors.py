"""
iot_sensors.py — Simulated IoT Sensor Network
Hyper-Local Smart Road Hazard Detection & Real-Time Alert System

Simulates physical road sensors that measure:
  - Surface water depth (flood sensor)
  - Road vibration (pothole / vehicle stress)
  - Tilt / strain (structural monitoring)
  - Ambient light (tunnel / road visibility)
  - Temperature + humidity

In a real deployment these would be ESP32/Arduino nodes
publishing MQTT messages to an MQTT broker.
"""

import random
import numpy as np
from config import JUNCTIONS, IOT_SENSORS, HISTORICAL_ACCIDENTS
from utils import ist_timestamp, is_monsoon


class IoTSensorNetwork:
    """
    Simulates a network of IoT road sensors.
    Each junction has one sensor node with multiple measurement channels.
    """

    def __init__(self, weather: dict | None = None):
        self.weather = weather or {}
        self._rain   = self.weather.get("rain_mm", 0)
        self._wind   = self.weather.get("wind_kmph", 10)
        self._temp   = self.weather.get("temperature_c", 28)

    # ─────────────────────────────────────────────
    # Individual sensor readings
    # ─────────────────────────────────────────────

    def _water_depth_mm(self, junction_id: int) -> float:
        """Surface water depth (mm). Rises with rain intensity."""
        base    = self._rain * 1.8
        penalty = HISTORICAL_ACCIDENTS.get(junction_id, 3) * 0.5
        noise   = random.gauss(0, 3)
        return round(max(0, base + penalty + noise), 1)

    def _vibration_g(self, junction_id: int) -> float:
        """Road surface vibration in g-force (pothole indicator)."""
        acc_score = HISTORICAL_ACCIDENTS.get(junction_id, 3)
        base = 0.1 + acc_score * 0.04
        if self._rain > 5:
            base += 0.15   # Wet roads → potholes more pronounced
        return round(max(0.01, random.gauss(base, 0.05)), 3)

    def _tilt_degrees(self) -> float:
        """Road surface tilt (structural health). Should be near 0."""
        return round(random.gauss(0, 0.3), 2)

    def _humidity_pct(self) -> float:
        """Relative humidity %. Wet roads → higher humidity."""
        base = 60 + self._rain * 1.5
        return round(min(100, max(30, random.gauss(base, 5))), 1)

    def _visibility_lux(self) -> float:
        """Ambient light (lux). Low during rain/night."""
        base = 15000
        if self._rain > 0:
            base -= self._rain * 200
        return round(max(200, random.gauss(base, 500)), 0)

    def _vehicle_count(self, junction_id: int) -> int:
        """Simulated vehicle count per 5-minute window at junction."""
        from utils import get_time_features
        tf  = get_time_features()
        base = 40 + HISTORICAL_ACCIDENTS.get(junction_id, 3) * 8
        if tf["is_peak_hour"]:
            base = int(base * 2.2)
        if tf["is_night"]:
            base = int(base * 0.3)
        return max(0, int(random.gauss(base, base * 0.15)))

    # ─────────────────────────────────────────────
    # Aggregate sensor node reading
    # ─────────────────────────────────────────────

    def read_node(self, junction_id: int) -> dict:
        """Return all sensor readings for a junction node."""
        water  = self._water_depth_mm(junction_id)
        vib    = self._vibration_g(junction_id)
        tilt   = self._tilt_degrees()
        hum    = self._humidity_pct()
        lux    = self._visibility_lux()
        vcnt   = self._vehicle_count(junction_id)

        # Derive flood_risk from water depth
        if water > 80:
            flood_risk = "CRITICAL"
        elif water > 40:
            flood_risk = "HIGH"
        elif water > 15:
            flood_risk = "MODERATE"
        else:
            flood_risk = "LOW"

        # IoT-derived risk contribution (0–100)
        iot_risk_score = min(100, (
            (water / 120) * 40 +
            (vib / 0.5) * 30 +
            (max(0, abs(tilt) - 1) / 5) * 15 +
            ((100 - hum) / 100) * 15
        ))

        return {
            "junction_id":    junction_id,
            "sensor_id":      IOT_SENSORS[junction_id],
            "water_depth_mm": water,
            "vibration_g":    vib,
            "tilt_degrees":   tilt,
            "humidity_pct":   hum,
            "visibility_lux": lux,
            "vehicle_count":  vcnt,
            "flood_risk":     flood_risk,
            "iot_risk_score": round(iot_risk_score, 1),
            "is_monsoon":     is_monsoon(),
            "timestamp":      ist_timestamp(),
        }

    def read_all(self) -> dict:
        """Read all sensor nodes. Returns {junction_id: reading_dict}."""
        return {jid: self.read_node(jid) for jid in JUNCTIONS}

    def get_iot_risk_map(self) -> dict:
        """Return {junction_id: iot_risk_score} for all junctions."""
        readings = self.read_all()
        return {jid: r["iot_risk_score"] for jid, r in readings.items()}
