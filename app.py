"""
app.py — Entrypoint Flask REST API Backend Face Recognition.

Menggantikan api_server.py (http.server) dengan Flask yang modular.
Tanpa UI: semua response adalah JSON, siap dikonsumsi oleh aplikasi mobile.

Jalankan:
  python app.py
  python app.py --source none         # tanpa kamera (mobile-only)
  python app.py --source 0            # webcam lokal
  python app.py --source http://192.168.1.50:81/stream   # ESP32-CAM
  python app.py --host 0.0.0.0 --port 5000   # akses dari jaringan LAN

Uji dengan curl:
  curl http://127.0.0.1:5000/api/health
  curl http://127.0.0.1:5000/api/status
  curl http://127.0.0.1:5000/api/faces
  curl -X POST -H "Content-Type: application/json" \\
       -d '{"nama":"Faris"}' http://127.0.0.1:5000/api/enroll
  curl -X POST -F "nama=Faris" -F "photo=@foto.jpg" \\
       http://127.0.0.1:5000/api/enroll/photo
  curl -X POST -F "photo=@foto.jpg" \\
       http://127.0.0.1:5000/api/recognize
"""

import argparse
import os
import signal
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Factory aplikasi Flask
# ---------------------------------------------------------------------------

def create_app(engine_cfg: dict) -> Flask:
    """
    Buat instance Flask, daftarkan semua Blueprint, dan inisialisasi Engine.

    Args:
        engine_cfg: dict konfigurasi untuk Engine (dari argparse / defaults).
    """
    app = Flask(__name__)

    # CORS: izinkan akses dari semua origin (mobile app / Postman / ESP32)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ----- Inisialisasi Engine (komponen AI) -----
    from core.engine import Engine
    engine = Engine(engine_cfg)
    app.config["ENGINE"] = engine

    # ----- Daftarkan Blueprints -----
    from routes.health import health_bp
    from routes.faces import faces_bp
    from routes.recognize import recognize_bp
    from routes.config import config_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(faces_bp)
    app.register_blueprint(recognize_bp)
    app.register_blueprint(config_bp)

    # ----- Root endpoint: ringkasan API (bukan HTML) -----
    @app.get("/")
    def root():
        return jsonify({
            "nama": "Smart House Face Recognition API",
            "versi": "2.0",
            "endpoints": {
                "GET  /api/health":               "Status semua komponen backend",
                "GET  /api/status":               "Hasil recognition terakhir (dari kamera)",
                "GET  /api/config":               "Konfigurasi aktif",
                "GET  /api/faces":                "Daftar wajah terdaftar",
                "POST /api/enroll":               "Daftarkan wajah dari kamera live",
                "POST /api/enroll/photo":         "Daftarkan wajah dari foto upload (mobile)",
                "POST /api/recognize":            "Kenali wajah dari foto upload (mobile)",
                "DELETE /api/faces/<nama>":       "Hapus wajah dari database",
                "POST /api/config/threshold":     "Ubah cosine threshold",
                "POST /api/config/fas_threshold": "Ubah liveness threshold",
                "POST /api/config/margin":        "Ubah margin crop",
            },
            "dokumentasi": "Semua request/response dalam format JSON",
        }), 200

    # ----- Error handler global -----
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": f"endpoint tidak ditemukan: {e}"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": f"method tidak diizinkan: {e}"}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": f"server error: {e}"}), 500

    return app


# ---------------------------------------------------------------------------
# Argparse & main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Smart House Face Recognition — Flask REST API Backend"
    )
    ap.add_argument("--host", default="127.0.0.1",
                    help="Host server (0.0.0.0 untuk akses LAN, default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument(
        "--source", default="none",
        help="'none' (default, mobile-only) | angka webcam (misal 0) | URL stream MJPEG ESP32-CAM"
    )
    ap.add_argument("--data-dir", default="api_data")
    ap.add_argument("--mfn-repo", default="MobileFaceNet")
    ap.add_argument(
        "--pretrained",
        default=os.path.join("MobileFaceNet", "weights", "mobilefacenet.pt")
    )
    ap.add_argument(
        "--infer-script",
        default=str(Path(__file__).resolve().parent / "infer_mfn.py")
    )
    ap.add_argument("--yunet-model", default="face_detection_yunet_2023mar.onnx")
    ap.add_argument("--fas-model-dir", default="fas/models")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--fas-threshold", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--enroll-frames", type=int, default=5)
    return ap.parse_args()


def build_engine_cfg(args) -> dict:
    """Susun dict konfigurasi Engine dari argparse + default."""
    cfg = {
        "data_dir":      args.data_dir,
        "mfn_repo":      args.mfn_repo,
        "pretrained":    args.pretrained,
        "infer_script":  args.infer_script,
        "yunet_model":   args.yunet_model,
        "fas_model_dir": args.fas_model_dir,
        "source":        args.source,
        "frame_skip":    args.frame_skip,
        "enroll_frames": args.enroll_frames,
    }
    if args.threshold is not None:
        cfg["threshold"] = args.threshold
    if args.fas_threshold is not None:
        cfg["fas_threshold"] = args.fas_threshold
    if args.margin is not None:
        cfg["margin"] = args.margin
    return cfg


def main():
    args = parse_args()
    engine_cfg = build_engine_cfg(args)

    app = create_app(engine_cfg)

    # Graceful shutdown: hentikan engine saat Ctrl+C atau SIGTERM
    def _shutdown(sig, frame):
        print("\n[app] shutdown...")
        engine = app.config.get("ENGINE")
        if engine:
            engine.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[app] Smart House Face Recognition API v2.0", flush=True)
    print(f"[app] running on http://{args.host}:{args.port}/", flush=True)
    print(f"[app] tekan Ctrl+C untuk berhenti", flush=True)

    # Gunakan Waitress (production WSGI) jika tersedia, fallback ke Flask dev
    try:
        from waitress import serve
        print(f"[app] menggunakan Waitress WSGI server", flush=True)
        serve(app, host=args.host, port=args.port, threads=8)
    except ImportError:
        print(
            "[app] waitress tidak terpasang; menggunakan Flask development server.\n"
            "[app] untuk produksi, install: pip install waitress",
            flush=True
        )
        app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
