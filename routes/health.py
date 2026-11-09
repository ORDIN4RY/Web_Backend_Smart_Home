"""
routes/health.py — Blueprint: status server & komponen.

Endpoint:
  GET /api/health  — status ringkas semua komponen (model, detektor, kamera)
  GET /api/status  — snapshot penuh hasil recognition terakhir
  GET /api/config  — konfigurasi aktif
"""

from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__)


def _engine():
    return current_app.config["ENGINE"]


@health_bp.get("/api/health")
def api_health():
    """
    Cek apakah server dan semua komponen siap.
    Cocok dipakai sebagai health check dari mobile / load balancer.

    Response 200:
      {
        "status": "ok",
        "versi": "2.0",
        "model": "dimuat" | "belum_dimuat",
        "model_alasan": null | "<error>",
        "fas_model": "dimuat" | "belum_dimuat",
        "fas_alasan": null | "<error>",
        "detektor": "yunet" | "haar" | "tidak_ada",
        "kamera": "ok" | "nonaktif" | "error",
        "wajah_terdaftar": 3,
        "threshold": 0.5,
        "fas_threshold": 0.6
      }
    """
    eng = _engine()
    snap = eng.snapshot()
    return jsonify({
        "status": "ok",
        "versi": "2.0",
        "model": snap["model"],
        "model_alasan": snap["model_alasan"],
        "fas_model": snap["fas_model"],
        "fas_alasan": snap["fas_alasan"],
        "detektor": snap["detektor"],
        "detektor_alasan": snap["detektor_alasan"],
        "kamera": snap["kamera"],
        "wajah_terdaftar": snap["wajah_terdaftar"],
        "threshold": snap["threshold"],
        "fas_threshold": snap["fas_threshold"],
    }), 200


@health_bp.get("/api/status")
def api_status():
    """
    Snapshot penuh hasil recognition terakhir dari worker kamera.
    Mobile app bisa polling endpoint ini untuk mendapatkan hasil realtime.

    Response 200: snapshot lengkap (lihat Engine.snapshot())
    """
    eng = _engine()
    return jsonify(eng.snapshot()), 200


@health_bp.get("/api/config")
def api_config():
    """
    Konfigurasi backend yang sedang aktif.

    Response 200:
      {
        "threshold": 0.5,
        "fas_threshold": 0.6,
        "margin": 0.0,
        "source": "0",
        "frame_skip": 3,
        "mfn_repo": "MobileFaceNet",
        "pretrained": "...",
        "yunet_model": "...",
        "fas_model_dir": "...",
        "data_dir": "api_data"
      }
    """
    eng = _engine()
    cfg = eng.store.config
    return jsonify({
        "threshold": eng.store.threshold(),
        "fas_threshold": eng.store.fas_threshold(),
        "margin": eng.store.margin(),
        "source": eng.source_label,
        "frame_skip": eng.frame_skip,
        "mfn_repo": cfg.get("mfn_repo"),
        "pretrained": cfg.get("pretrained"),
        "yunet_model": cfg.get("yunet_model"),
        "fas_model_dir": cfg.get("fas_model_dir"),
        "data_dir": str(eng.store.dir),
    }), 200
