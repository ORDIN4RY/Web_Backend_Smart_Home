"""
routes/faces.py — Blueprint: manajemen wajah terdaftar (faces.json).

Endpoint:
  GET    /api/faces              — daftar semua wajah terdaftar
  POST   /api/enroll             — daftarkan wajah dari kamera live
  POST   /api/enroll/photo       — daftarkan wajah dari foto upload (mobile)
  DELETE /api/faces/<nama>       — hapus wajah dari database
"""

from flask import Blueprint, jsonify, request, current_app

faces_bp = Blueprint("faces", __name__)


def _engine():
    return current_app.config["ENGINE"]


# ---------------------------------------------------------------------------
# GET /api/faces
# ---------------------------------------------------------------------------

@faces_bp.get("/api/faces")
def list_faces():
    """
    Daftar semua wajah yang sudah terdaftar.

    Response 200:
      {
        "wajah": [
          {"nama": "Faris", "jumlah_frame": 5, "diperbarui": "2026-11-05T10:00:00"},
          ...
        ],
        "total": 3
      }
    """
    eng = _engine()
    faces = eng.store.list_faces()
    return jsonify({"wajah": faces, "total": len(faces)}), 200


# ---------------------------------------------------------------------------
# POST /api/enroll  (kamera live)
# ---------------------------------------------------------------------------

@faces_bp.post("/api/enroll")
def enroll_camera():
    """
    Daftarkan wajah baru dari kamera live backend.
    Cocok dipakai jika backend memiliki webcam yang terpasang.

    Body JSON:
      {"nama": "Faris"}

    Response 200:
      {"nama": "Faris", "jumlah_frame": 5, "pesan": "..."}
    Response 503:
      {"error": "kamera nonaktif; gunakan POST /api/enroll/photo"}
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "body harus JSON object"}), 400

    nama = str(body.get("nama", "")).strip()
    if not nama or len(nama) > 50 or "/" in nama or "\\" in nama:
        return jsonify({
            "error": "nama tidak valid (wajib isi, maks 50 karakter, tanpa / atau \\)"
        }), 400

    eng = _engine()
    code, payload = eng.enroll_from_camera(nama)
    return jsonify(payload), code


# ---------------------------------------------------------------------------
# POST /api/enroll/photo  (upload foto dari mobile)
# ---------------------------------------------------------------------------

@faces_bp.post("/api/enroll/photo")
def enroll_photo():
    """
    Daftarkan wajah dari foto yang diupload oleh aplikasi mobile.

    Request: multipart/form-data
      - nama  : string (nama orang)
      - photo : file gambar (JPEG/PNG)

    ATAU application/json dengan field "nama" + "image_base64" (base64).

    Response 200:
      {"nama": "Faris", "jumlah_frame": 1, "pesan": "..."}
    Response 400:
      {"error": "..."}
    """
    eng = _engine()
    nama = ""
    image_bytes = None

    # --- multipart/form-data (dari mobile Flutter/Android dengan image picker) ---
    if request.content_type and "multipart/form-data" in request.content_type:
        nama = request.form.get("nama", "").strip()
        if "photo" not in request.files:
            return jsonify({"error": "field 'photo' tidak ditemukan di form-data"}), 400
        image_bytes = request.files["photo"].read()

    # --- application/json dengan base64 image ---
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        nama = str(body.get("nama", "")).strip()
        b64 = body.get("image_base64", "")
        if not b64:
            return jsonify({"error": "field 'image_base64' tidak boleh kosong"}), 400
        try:
            import base64
            image_bytes = base64.b64decode(b64)
        except Exception:
            return jsonify({"error": "image_base64 tidak valid"}), 400
    else:
        return jsonify({
            "error": "Content-Type harus multipart/form-data atau application/json"
        }), 415

    if not nama or len(nama) > 50 or "/" in nama or "\\" in nama:
        return jsonify({
            "error": "nama tidak valid (wajib isi, maks 50 karakter, tanpa / atau \\)"
        }), 400
    if not image_bytes:
        return jsonify({"error": "data gambar kosong"}), 400

    code, payload = eng.enroll_from_photo(nama, image_bytes)
    return jsonify(payload), code


# ---------------------------------------------------------------------------
# DELETE /api/faces/<nama>
# ---------------------------------------------------------------------------

@faces_bp.delete("/api/faces/<string:nama>")
def delete_face(nama: str):
    """
    Hapus wajah dari database berdasarkan nama.

    Response 200:
      {"dihapus": "Faris", "pesan": "Wajah berhasil dihapus"}
    Response 404:
      {"error": "wajah tidak ditemukan: Faris"}
    """
    eng = _engine()
    nama = nama.strip()
    if eng.store.remove_face(nama):
        return jsonify({"dihapus": nama, "pesan": f"Wajah '{nama}' berhasil dihapus"}), 200
    return jsonify({"error": f"wajah tidak ditemukan: {nama}"}), 404
