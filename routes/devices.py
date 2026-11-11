"""
routes/devices.py — Blueprint: Sensor & Kontrol Perangkat Smart Home.

Endpoint:
  GET  /api/sensors/current — Data pembacaan sensor real-time
  GET  /api/devices/status  — Status terkini semua perangkat & aktuator
  POST /api/devices/control — Kendali perangkat (Peltier PWM, Atap, Lampu, Door Lock, Buzzer)
  GET  /api/events          — Riwayat event keamanan & otomasi
"""

from datetime import datetime
from flask import Blueprint, jsonify, request

devices_bp = Blueprint("devices", __name__)

# State perangkat di memori backend
_devices_state = {
    "light_on": False,
    "cooling_on": True,
    "door_locked": True,
    "roof_open": True,
    "buzzer_on": False,
    "porch_camera_active": False,
    "light_auto_mode": True,
    "cooling_auto_mode": True,
    "roof_auto_mode": True,
    "target_temperature": 24.0,
    "peltier_pwm": 75,
    "last_updated": datetime.now().isoformat(),
}

# State sensor real-time
_sensors_state = {
    "temperature": 28.4,
    "humidity": 62.0,
    "light_intensity": 42.0,
    "gas_ppm": 95.0,
    "flame_detected": False,
    "motion_detected": False,
    "rain_detected": False,
    "door_open": False,
    "timestamp": datetime.now().isoformat(),
}

_event_logs = [
    {
        "id": "evt_001",
        "title": "Sistem Aktif",
        "description": "Sistem Smart Home dan modul pengenal wajah siap digunakan.",
        "category": "device",
        "severity": "normal",
        "timestamp": datetime.now().isoformat(),
    }
]


def _recalc_automations():
    """Jalankan aturan otomasi sensor -> aktuator."""
    now_iso = datetime.now().isoformat()
    _devices_state["last_updated"] = now_iso
    _sensors_state["timestamp"] = now_iso

    # 1. PIR & LDR untuk Lampu & Teras (Kamera aktif saat ada gerakan)
    is_motion = _sensors_state["motion_detected"]
    _devices_state["porch_camera_active"] = is_motion
    if _devices_state["light_auto_mode"]:
        _devices_state["light_on"] = (_sensors_state["light_intensity"] < 30.0) or is_motion

    # 2. Flame & MQ-2: HANYA ke Buzzer
    flame = _sensors_state["flame_detected"]
    gas_danger = _sensors_state["gas_ppm"] >= 400.0
    _devices_state["buzzer_on"] = flame or gas_danger

    # 3. Sensor Hujan & Atap Rumah
    if _devices_state["roof_auto_mode"]:
        # Hujan -> atap tertutup, Kering -> atap terbuka
        _devices_state["roof_open"] = not _sensors_state["rain_detected"]

    # 4. Peltier & Modul MOSFET PWM (Penyesuaian Suhu Otomatis)
    if _devices_state["cooling_on"]:
        if _devices_state["cooling_auto_mode"]:
            diff = _sensors_state["temperature"] - _devices_state["target_temperature"]
            if diff > 0:
                pwm = int(min(100, max(20, (diff / 4.0) * 80 + 20)))
            else:
                pwm = 0
            _devices_state["peltier_pwm"] = pwm
        else:
            _devices_state["peltier_pwm"] = 80
    else:
        _devices_state["peltier_pwm"] = 0


@devices_bp.get("/api/sensors/current")
def get_current_sensors():
    """Kembalikan data sensor terkini."""
    _recalc_automations()
    return jsonify({"status": "ok", "data": _sensors_state}), 200


@devices_bp.get("/api/devices/status")
def get_device_status():
    """Kembalikan status perangkat & aktuator."""
    _recalc_automations()
    return jsonify({"status": "ok", "status": _devices_state}), 200


@devices_bp.post("/api/devices/control")
def control_device():
    """
    Kendalikan status perangkat dari aplikasi mobile / ESP32.
    Body JSON: {"device": "roof|cooling|light|door_lock|target_temp|...", "value": val}
    """
    data = request.get_json(silent=True) or {}
    dev_key = data.get("device")
    val = data.get("value")

    if not dev_key:
        return jsonify({"error": "parameter 'device' diperlukan"}), 400

    if dev_key == "light":
        _devices_state["light_on"] = bool(val)
    elif dev_key == "light_auto_mode":
        _devices_state["light_auto_mode"] = bool(val)
    elif dev_key == "cooling":
        _devices_state["cooling_on"] = bool(val)
    elif dev_key == "cooling_auto_mode":
        _devices_state["cooling_auto_mode"] = bool(val)
    elif dev_key == "target_temp":
        _devices_state["target_temperature"] = float(val)
    elif dev_key == "roof":
        _devices_state["roof_open"] = bool(val)
        _event_logs.insert(0, {
            "id": f"evt_{int(datetime.now().timestamp() * 1000)}",
            "title": "Atap Dibuka Manual" if val else "Atap Ditutup Manual",
            "description": "Perintah kontrol atap dari aplikasi mobile.",
            "category": "device",
            "severity": "normal",
            "timestamp": datetime.now().isoformat(),
        })
    elif dev_key == "roof_auto_mode":
        _devices_state["roof_auto_mode"] = bool(val)
    elif dev_key == "door_lock":
        _devices_state["door_locked"] = bool(val)
    elif dev_key == "buzzer":
        _devices_state["buzzer_on"] = bool(val)
    elif dev_key == "peltier_pwm":
        _devices_state["peltier_pwm"] = int(val)
    elif dev_key == "update_sensor":
        # Endpoint untuk ESP32 mengupdate nilai sensor
        if isinstance(val, dict):
            _sensors_state.update(val)

    _recalc_automations()
    return jsonify({"status": _devices_state}), 200


@devices_bp.get("/api/events")
def get_events():
    """Daftar log aktivitas dan peringatan keamanan."""
    return jsonify({"status": "ok", "events": _event_logs[:50]}), 200
