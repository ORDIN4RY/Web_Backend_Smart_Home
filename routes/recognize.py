"""
routes/recognize.py — Blueprint: face recognition dari foto upload (mobile).

Ini adalah endpoint UTAMA untuk aplikasi mobile.
Mobile mengirim foto -> backend mendeteksi + liveness + recognition -> JSON.

Endpoint:
  POST /api/recognize  — kenali wajah dari foto upload
"""

from flask import Blueprint, jsonify, request, current_app

recognize_bp = Blueprint("recognize", __name__)


def _engine():
    return current_app.config["ENGINE"]


@recognize_bp.post("/api/recognize")
def recognize():
    """
    Kenali wajah dari foto yang diupload aplikasi mobile.

    Request: multipart/form-data
      - photo : file gambar (JPEG/PNG)

    ATAU application/json:
      {"image_base64": "<base64 string>"}

    Response 200 — wajah dikenali:
      {
        "granted": true,
        "wajah": "Faris",
        "skor": 0.87,
        "threshold": 0.5,
        "liveness": true,
        "liveness_skor": 0.95,
        "catatan": "dikenali: Faris"
      }

    Response 200 — wajah tidak dikenali (unknown):
      {
        "granted": false,
        "wajah": null,
        "skor": 0.32,
        "catatan": "wajah REAL, skor di bawah threshold (unknown)"
      }

    Response 403 — spoof terdeteksi:
      {
        "granted": false,
        "liveness": false,
        "catatan": "DITOLAK: spoof / wajah palsu terdeteksi"
      }

    Response 404 — tidak ada wajah di foto:
      {"error": "tidak ada wajah terdeteksi di foto"}
    """
    eng = _engine()
    image_bytes = _extract_image_bytes()
    if isinstance(image_bytes, tuple):          # error response
        return image_bytes

    code, payload = eng.recognize_photo(image_bytes)
    return jsonify(payload), code


# ---------------------------------------------------------------------------
# Helper: ekstrak bytes gambar dari request (multipart atau JSON base64)
# ---------------------------------------------------------------------------

def _extract_image_bytes():
    """
    Kembalikan bytes gambar dari request, atau tuple (response, status_code).
    """
    if request.content_type and "multipart/form-data" in request.content_type:
        if "photo" not in request.files:
            return jsonify({"error": "field 'photo' tidak ada di form-data"}), 400
        data = request.files["photo"].read()
        if not data:
            return jsonify({"error": "file photo kosong"}), 400
        return data

    if request.is_json:
        body = request.get_json(silent=True) or {}
        b64 = body.get("image_base64", "")
        if not b64:
            return jsonify({"error": "field 'image_base64' tidak boleh kosong"}), 400
        try:
            import base64
            return base64.b64decode(b64)
        except Exception:
            return jsonify({"error": "image_base64 tidak valid (bukan base64)"}), 400

    return jsonify({
        "error": "Content-Type harus multipart/form-data atau application/json"
    }), 415
