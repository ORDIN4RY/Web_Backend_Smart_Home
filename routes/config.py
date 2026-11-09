"""
routes/config.py — Blueprint: manajemen konfigurasi backend.

Endpoint:
  POST /api/config/threshold      — ubah threshold cosine similarity
  POST /api/config/fas_threshold  — ubah threshold liveness MiniFASNet
  POST /api/config/margin         — ubah margin crop wajah
"""

from flask import Blueprint, jsonify, request, current_app

config_bp = Blueprint("config", __name__)


def _engine():
    return current_app.config["ENGINE"]


def _update_float(field: str, min_val: float = 0.0, max_val: float = 1.0):
    """Helper: validasi dan simpan satu field float ke store.config."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "body harus JSON object"}), 400
    try:
        value = float(body[field])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": f"field '{field}' harus berupa angka"}), 400
    if not min_val <= value <= max_val:
        return jsonify({"error": f"'{field}' harus antara {min_val} dan {max_val}"}), 400
    return value


@config_bp.post("/api/config/threshold")
def set_threshold():
    """
    Ubah threshold cosine similarity untuk face recognition.

    Body JSON: {"threshold": 0.45}
    Response 200: {"threshold": 0.45}
    """
    eng = _engine()
    result = _update_float("threshold")
    if isinstance(result, tuple):
        return result
    eng.store.config["threshold"] = result
    eng.store.save_config()
    return jsonify({"threshold": result}), 200


@config_bp.post("/api/config/fas_threshold")
def set_fas_threshold():
    """
    Ubah threshold liveness MiniFASNet.

    Body JSON: {"fas_threshold": 0.65}
    Response 200: {"fas_threshold": 0.65}
    """
    eng = _engine()
    result = _update_float("fas_threshold")
    if isinstance(result, tuple):
        return result
    eng.store.config["fas_threshold"] = result
    eng.store.save_config()
    # Update juga threshold pada instance fas yang sedang jalan
    if eng.fas:
        eng.fas.threshold = result
    return jsonify({"fas_threshold": result}), 200


@config_bp.post("/api/config/margin")
def set_margin():
    """
    Ubah margin crop bounding box wajah.

    Body JSON: {"margin": 0.1}
    Response 200: {"margin": 0.1}
    """
    eng = _engine()
    result = _update_float("margin")
    if isinstance(result, tuple):
        return result
    eng.store.config["margin"] = result
    eng.store.save_config()
    return jsonify({"margin": result}), 200
