"""
utils.py — Shared helper utilities
Hyper-Local Smart Road Hazard Detection & Real-Time Alert System
"""

from datetime import datetime
from config import IST, RISK_CRITICAL, RISK_WARNING, RISK_LOW, PEAK_HOURS, NIGHT_HOURS, MONSOON_MONTHS


# ─────────────────────────────────────────────
# Time Helpers
# ─────────────────────────────────────────────

def ist_now() -> datetime:
    return datetime.now(IST)

def ist_timestamp() -> str:
    return ist_now().strftime("%Y-%m-%d %H:%M:%S IST")

def get_time_features() -> dict:
    now = ist_now()
    return {
        "hour_of_day":  now.hour,
        "is_peak_hour": int(now.hour in PEAK_HOURS),
        "is_night":     int(now.hour in NIGHT_HOURS),
        "is_weekend":   int(now.weekday() >= 5),
    }

def is_monsoon() -> bool:
    return ist_now().month in MONSOON_MONTHS


# ─────────────────────────────────────────────
# Risk Helpers
# ─────────────────────────────────────────────

def risk_colour(risk: float) -> str:
    if risk >= RISK_CRITICAL:
        return "#ff4d4d"
    elif risk >= RISK_WARNING:
        return "#ff9900"
    elif risk >= RISK_LOW:
        return "#ffd700"
    return "#22c55e"

def risk_label(risk: float) -> str:
    if risk >= RISK_CRITICAL:
        return "CRITICAL"
    elif risk >= RISK_WARNING:
        return "HIGH"
    elif risk >= RISK_LOW:
        return "MODERATE"
    return "LOW"

def risk_emoji(risk: float) -> str:
    if risk >= RISK_CRITICAL:
        return "🔴"
    elif risk >= RISK_WARNING:
        return "🟠"
    elif risk >= RISK_LOW:
        return "🟡"
    return "🟢"

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def estimate_travel_time_mins(path: list, risks: dict, avg_speed_kmph: float = 30.0) -> int:
    """Estimate travel time based on risk-adjusted speed."""
    total_mins = 0
    for i in range(len(path) - 1):
        e = (path[i], path[i + 1])
        r = risks.get(e, 30.0)
        # Higher risk → slower speed
        speed = avg_speed_kmph * max(0.2, 1.0 - (r / 200.0))
        total_mins += (3.5 / speed) * 60  # avg 3.5 km per hop
    return max(1, round(total_mins))


# ─────────────────────────────────────────────
# Notification formatting
# ─────────────────────────────────────────────

def format_push_notification(hazard_type: str, junction_name: str, risk: float) -> dict:
    """Create a push-notification style dict for the alert panel."""
    emoji_map = {
        "pothole":           "🕳️",
        "waterlogging":      "🌊",
        "fallen_tree":       "🌳",
        "muddy_road":        "🟤",
        "overflowing_drain": "💧",
        "road_collapse":     "⚠️",
        "debris":            "🪨",
        "none":              "✅",
    }
    icon = emoji_map.get(hazard_type, "⚠️")
    label = risk_label(risk)
    return {
        "icon":    icon,
        "title":   f"{icon} {hazard_type.replace('_', ' ').title()} Detected",
        "body":    f"At {junction_name} — Risk Level: {label} ({risk:.0f}/100)",
        "level":   label,
        "ts":      ist_timestamp(),
    }
