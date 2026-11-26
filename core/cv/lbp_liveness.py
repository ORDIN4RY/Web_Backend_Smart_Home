"""
core/cv/lbp_liveness.py — Tahapan Color-LBP Liveness Detection (Computer Vision Klasik).

Metode (Boulkenafet et al., IEEE TIFS 2016):
  Multi-Color Space Chrominance LBP:
    YCrCb (Cr, Cb) + HSV (H, S) → vektor 1024-D
  Kanal luminansi (Y, V) dibuang agar kebal terhadap variasi pencahayaan.
  Keputusan: 1-NN Relatif Chi-Square (d_real < d_spoof → REAL).

Galeri color_lbp_proto.json dimuat sekali saat modul di-import.
"""

import json
from pathlib import Path
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Muat galeri Multi-Exemplar Color-LBP (Real & Spoof) sekali saat startup
# ---------------------------------------------------------------------------
_PROTO_PATH = Path(__file__).resolve().parent.parent.parent / "api_data" / "color_lbp_proto.json"
COLOR_LBP_REAL = None
COLOR_LBP_SPOOF = None
COLOR_LBP_THRESH = 0.08

if _PROTO_PATH.exists():
    try:
        _data = json.loads(_PROTO_PATH.read_text(encoding="utf-8"))
        COLOR_LBP_THRESH = float(_data.get("threshold", 0.08))
        _real_ex = _data.get("real_exemplars", _data.get("exemplars", []))
        _spoof_ex = _data.get("spoof_exemplars", [])
        if len(_real_ex) > 0:
            COLOR_LBP_REAL = np.asarray(_real_ex, dtype=np.float32)
        if len(_spoof_ex) > 0:
            COLOR_LBP_SPOOF = np.asarray(_spoof_ex, dtype=np.float32)
    except Exception as _err:
        print(f"[cv.lbp_liveness] error loading {_PROTO_PATH}: {_err}")


def check_liveness(face_bgr, real_mat=None, spoof_mat=None, threshold: float = None) -> tuple:
    """
    Jalankan tahapan Color-LBP Liveness secara berurutan (linear):
      1. Resize crop wajah ke 128×128
      2. Konversi BGR → YCrCb & HSV, ambil 4 kanal krominan (Cr, Cb, H, S)
      3. Hitung LBP 8-tetangga + histogram ternormalisasi (256 bins) per kanal → 1024-D
      4. Hitung jarak Chi-Square ke galeri Real dan galeri Spoof
      5. Keputusan 1-NN Relatif: is_real = (d_real < d_spoof)

    Returns:
        (is_real, liveness_score, d_real, d_spoof)
    """
    if face_bgr is None or face_bgr.size == 0:
        return False, 0.0, 1.0, 0.0

    if real_mat is None:
        real_mat = COLOR_LBP_REAL
    if spoof_mat is None:
        spoof_mat = COLOR_LBP_SPOOF
    if threshold is None:
        threshold = COLOR_LBP_THRESH

    # --- Tahap 1: Resize & konversi ruang warna (YCrCb + HSV) ---
    face = cv2.resize(face_bgr, (128, 128))
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    # --- Tahap 2: Ekstraksi LBP pada 4 kanal krominan (Cr, Cb, H, S) ---
    # Kanal luminansi (Y, V) dibuang agar kebal variasi gelap/terang
    shifts = [
        (-1, -1, 0), (-1, 0, 1), (-1, 1, 2),
        (0,  1, 3),  (1,  1, 4), (1,  0, 5),
        (1, -1, 6),  (0, -1, 7),
    ]
    channels = [ycrcb[:, :, 1], ycrcb[:, :, 2], hsv[:, :, 0], hsv[:, :, 1]]
    hists = []

    for ch in channels:
        ch_100 = cv2.resize(ch, (100, 100))
        center = ch_100[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)
        for dy, dx, bit in shifts:
            neighbor = ch_100[1 + dy:99 + dy, 1 + dx:99 + dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)

        h, _ = np.histogram(lbp, bins=256, range=(0, 256))
        h = h.astype(np.float32)
        h /= (h.sum() + 1e-7)
        hists.append(h)

    vec = np.concatenate(hists).astype(np.float32)  # 4 × 256 = 1024-D

    # --- Tahap 3: Jarak Chi-Square ke galeri Real ---
    if real_mat is not None:
        if real_mat.ndim == 2:
            diff_r = (real_mat - vec) ** 2
            denom_r = real_mat + vec + 1e-7
            d_real = float(np.min(0.5 * np.sum(diff_r / denom_r, axis=1)))
        else:
            diff_r = (vec - real_mat) ** 2
            denom_r = vec + real_mat + 1e-7
            d_real = float(0.5 * np.sum(diff_r / denom_r))
    else:
        d_real = 0.50

    # --- Tahap 4: Jarak Chi-Square ke galeri Spoof ---
    if spoof_mat is not None and len(spoof_mat) > 0:
        diff_s = (spoof_mat - vec) ** 2
        denom_s = spoof_mat + vec + 1e-7
        d_spoof = float(np.min(0.5 * np.sum(diff_s / denom_s, axis=1)))
    else:
        d_spoof = 999.0

    # --- Tahap 5: Keputusan 1-NN Relatif (d_real < d_spoof → REAL) ---
    is_real = bool(d_real < d_spoof)
    margin_ratio = (d_spoof - d_real) / (d_spoof + d_real + 1e-7)
    liveness_score = round(float(max(0.0, min(1.0, 0.5 + 0.5 * margin_ratio))), 3)

    return is_real, liveness_score, round(d_real, 4), round(d_spoof, 4)
