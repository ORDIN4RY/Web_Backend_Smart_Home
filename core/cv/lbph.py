"""
core/cv/lbph.py — Tahapan LBPH Face Recognition (Computer Vision Klasik).

Alur end-to-end (sesuai urutan Gatekeeper 1 → Gatekeeper 2):
  1. Decode & resize frame (max 1280px)
  2. Deteksi wajah via Haar Cascade (Viola-Jones), crop ROI BGR
  3. Gatekeeper 1 Liveness: Color-LBP 4 kanal krominans (Cr, Cb, H, S → 1024-D) 1-NN Chi-Square
     -> Jika SPOOF, langsung tolak (403) tanpa memproses tahap LBPH!
  4. Prapemrosesan Tekstur: Grayscale 100×100 + Histogram Equalization
  5. Ekstraksi Fitur LBPH: partisi grid 8×8, LBP 8-tetangga → 16.384-D
  6. Gatekeeper 2 Identitas: Multi-Exemplar 1-NN Chi-Square (χ² ≤ 23.0)
"""

import base64
import cv2
import numpy as np

from core.cv.lbp_liveness import check_liveness, COLOR_LBP_THRESH

LBPH_THRESHOLD_DEFAULT = 23.0
_METODE = "CV Klasik (Haar + Color-LBP + LBPH)"

# Muat Haar Cascade sekali saat modul di-import (tidak di-reload tiap request)
_HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_HAAR_CASCADE = cv2.CascadeClassifier(_HAAR_XML)


def extract_lbp(gray_crop, grid_x: int = 8, grid_y: int = 8, return_image: bool = False):
    """
    Ekstraksi fitur Local Binary Pattern (LBP) spasial 8×8 (16.384-D).
    Dipakai oleh recognize_cv(), enroll_from_photo(), dan build_lbph_dataset.py.
    """
    gray = cv2.resize(gray_crop, (100, 100))
    center = gray[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)

    shifts = [
        (-1, -1, 0), (-1, 0, 1), (-1, 1, 2),
        (0,  1, 3),  (1,  1, 4), (1,  0, 5),
        (1, -1, 6),  (0, -1, 7),
    ]
    for dy, dx, bit in shifts:
        neighbor = gray[1 + dy:99 + dy, 1 + dx:99 + dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hists = []
    for r in np.array_split(lbp, grid_y, axis=0):
        for c in np.array_split(r, grid_x, axis=1):
            h, _ = np.histogram(c, bins=256, range=(0, 256))
            h = h.astype(np.float32)
            h /= (h.sum() + 1e-7)
            hists.append(h)

    vec = np.concatenate(hists)
    return (vec, lbp) if return_image else vec


def recognize_cv(
    image_bytes: bytes,
    lbph_db: dict,
    real_mat=None,
    spoof_mat=None,
    lbp_thresh: float = COLOR_LBP_THRESH,
    lbph_threshold: float = LBPH_THRESHOLD_DEFAULT,
    include_previews: bool = True,
) -> tuple:
    """
    Pipeline pengenalan wajah Computer Vision Klasik secara berurutan (linear).
    Mengembalikan (http_code: int, payload: dict).
    """
    try:
        # --- Tahap 1: Decode & resize gambar (max 1280px) ---
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return 400, {"error": "gambar tidak bisa di-decode"}

        h_f, w_f = frame.shape[:2]
        if max(h_f, w_f) > 1280:
            scale = 1280 / max(h_f, w_f)
            frame = cv2.resize(frame, (int(w_f * scale), int(h_f * scale)), interpolation=cv2.INTER_AREA)

        # --- Tahap 2: Deteksi wajah via Haar Cascade (Viola-Jones) → crop ROI BGR ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _HAAR_CASCADE.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        h_img, w_img = frame.shape[:2]
        if len(faces) == 0:
            if h_img <= 220 and w_img <= 220:
                bx, by, bw, bh = 0, 0, w_img, h_img
                face_crop = frame
            else:
                return 404, {"error": "tidak ada wajah terdeteksi (Haar Cascade)"}
        else:
            bx, by, bw, bh = max(faces, key=lambda f: f[2] * f[3])
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(w_img - bx, bw), min(h_img - by, bh)
            face_crop = frame[by:by + bh, bx:bx + bw]

        bbox_info = {
            "x": int(bx), "y": int(by), "w": int(bw), "h": int(bh),
            "frame_w": int(w_img), "frame_h": int(h_img),
        }

        # --- Tahap 3: Gatekeeper 1 Liveness — Color-LBP 1-NN Chi-Square ---
        is_real, liveness_score, chi2_liveness, chi2_spoof = check_liveness(
            face_crop, real_mat, spoof_mat, threshold=lbp_thresh
        )

        raw_steps = []
        if include_previews:
            # Visualisasi 4 kanal krominans Color-LBP: Cr|Cb (YCrCb) dan H|S (HSV)
            crop_128 = cv2.resize(face_crop, (128, 128))
            ycrcb = cv2.cvtColor(crop_128, cv2.COLOR_BGR2YCrCb)
            hsv = cv2.cvtColor(crop_128, cv2.COLOR_BGR2HSV)
            cr_map = cv2.applyColorMap(ycrcb[:, :, 1], cv2.COLORMAP_JET)
            cb_map = cv2.applyColorMap(ycrcb[:, :, 2], cv2.COLORMAP_OCEAN)
            h_map = cv2.applyColorMap(cv2.convertScaleAbs(hsv[:, :, 0], alpha=255.0 / 180.0), cv2.COLORMAP_HSV)
            s_map = cv2.applyColorMap(hsv[:, :, 1], cv2.COLORMAP_VIRIDIS)
            sep = np.full((128, 4, 3), 30, dtype=np.uint8)
            crcb_pair = np.hstack([cr_map, sep, cb_map])
            hs_pair = np.hstack([h_map, sep, s_map])
            raw_steps.extend([
                (1, "Deteksi Haar", "Crop ROI BGR", face_crop, 280),
                (2, "Krominans Cr | Cb", "Kanal YCrCb (tanpa Y)", crcb_pair, 280),
                (3, "Krominans H | S", "Kanal HSV (tanpa V)", hs_pair, 280),
            ])

        if not is_real:
            # Berhenti di Gatekeeper 1: tidak melanjutkan ke tahap prapemrosesan & LBPH
            http_code = 403
            payload = {
                "granted": False,
                "wajah": None,
                "mode": "cv",
                "metode": _METODE,
                "liveness": False,
                "liveness_skor": liveness_score,
                "distance_chi2_liveness": chi2_liveness,
                "distance_chi2_spoof": chi2_spoof,
                "threshold_chi2_liveness": lbp_thresh,
                "catatan": f"Spoof (d_real={chi2_liveness:.3f} ≥ d_spoof={chi2_spoof:.3f})",
                "bbox": bbox_info,
                "tahapan": [
                    "1. Deteksi Haar Cascade → ROI BGR",
                    f"2. Liveness Color-LBP (Cr,Cb,H,S): d_real={chi2_liveness:.3f} ≥ d_spoof={chi2_spoof:.3f} → SPOOF",
                    "3. Ditolak di Gatekeeper 1 (tidak lanjut ke LBPH)",
                ],
            }
        else:
            # --- Tahap 4: Prapemrosesan Tekstur (hanya jika lolos liveness REAL) ---
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray_100 = cv2.resize(gray, (100, 100))
            equalized = cv2.equalizeHist(gray_100)

            # --- Tahap 5: Ekstraksi Fitur LBPH 8×8 (16.384-D) ---
            lbp_vec, lbp_img = extract_lbp(equalized, return_image=True)

            if include_previews:
                # Render grafik distribusi pola LBP (0..255) rata-rata dari 64 sel grid 8×8
                # Tema terang (Light Neumorphic) agar serasi dan kontras di UI Flutter
                hist_w, hist_h = 340, 170
                hist_canvas = np.full((hist_h, hist_w, 3), (248, 244, 240), dtype=np.uint8)  # #F0F4F8

                mean_256 = lbp_vec.reshape(64, 256).mean(axis=0)
                sampled = mean_256.reshape(64, 4).sum(axis=1)  # 64 bar (tiap bar = 4 kode LBP)
                max_v = float(sampled.max()) if sampled.max() > 0 else 1.0

                pad_l, pad_r, pad_t, pad_b = 14, 14, 24, 22
                plot_w = hist_w - pad_l - pad_r
                plot_h = hist_h - pad_t - pad_b
                y_base = hist_h - pad_b

                # Grid horizontal tipis + sumbu dasar
                for frac in (0.25, 0.5, 0.75, 1.0):
                    yg = int(y_base - frac * plot_h)
                    cv2.line(hist_canvas, (pad_l, yg), (hist_w - pad_r, yg), (228, 220, 212), 1)
                cv2.line(hist_canvas, (pad_l, y_base), (hist_w - pad_r, y_base), (144, 122, 107), 1)

                # Gambar 64 bar biru (#3B82F6 -> BGR 246, 130, 59) dengan celah 1px
                step_x = plot_w / 64.0
                for i, v in enumerate(sampled):
                    bar_h = max(1, int((float(v) / max_v) * plot_h))
                    x1 = int(pad_l + i * step_x)
                    x2 = max(x1 + 1, int(pad_l + (i + 1) * step_x) - 1)
                    cv2.rectangle(hist_canvas, (x1, y_base - bar_h), (x2, y_base - 1), (246, 130, 59), -1)

                cv2.putText(hist_canvas, "Distribusi Pola LBP (0-255)", (pad_l, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (56, 44, 36), 1, cv2.LINE_AA)
                cv2.putText(hist_canvas, "0", (pad_l, hist_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (144, 122, 107), 1, cv2.LINE_AA)
                cv2.putText(hist_canvas, "128", (hist_w // 2 - 12, hist_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (144, 122, 107), 1, cv2.LINE_AA)
                cv2.putText(hist_canvas, "255", (hist_w - pad_r - 22, hist_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (144, 122, 107), 1, cv2.LINE_AA)

                raw_steps.extend([
                    (4, "Prapemrosesan", "Gray 100×100 + EqualizeHist", equalized, 280),
                    (5, "Citra LBP", "8-Tetangga, R=1 (98×98)", lbp_img, 280),
                    (6, "Histogram LBPH", "Grid 8×8 (16.384-D)", hist_canvas, 340),
                ])

            # --- Tahap 6: Gatekeeper 2 Identitas — Multi-Exemplar LBPH Chi-Square ---
            best_name = None
            best_dist = 999.0
            all_distances = {}

            for person, data in lbph_db.items():
                ex_list = data.get("exemplars")
                if ex_list:
                    ex_arr = np.asarray(ex_list, dtype=np.float32)
                    diff = (ex_arr - lbp_vec) ** 2
                    denom = ex_arr + lbp_vec + 1e-7
                    chi2 = float(np.min(0.5 * np.sum(diff / denom, axis=1)))
                else:
                    proto_vec = np.asarray(
                        data.get("prototype", data.get("histogram", [])), dtype=np.float32
                    )
                    chi2 = float(0.5 * np.sum(((lbp_vec - proto_vec) ** 2) / (lbp_vec + proto_vec + 1e-7)))

                all_distances[person] = round(chi2, 2)
                if chi2 < best_dist:
                    best_dist = chi2
                    best_name = person

            granted = bool(best_name and best_dist <= lbph_threshold)
            score_ratio = max(0.0, min(1.0, 1.0 - (best_dist / (lbph_threshold * 1.5))))
            confidence = round(score_ratio, 3)
            wajah_result = best_name if granted else None

            http_code = 200
            payload = {
                "granted": granted,
                "wajah": wajah_result,
                "mode": "cv",
                "metode": _METODE,
                "liveness": True,
                "liveness_skor": liveness_score,
                "distance_chi2_liveness": chi2_liveness,
                "distance_chi2_spoof": chi2_spoof,
                "threshold_chi2_liveness": lbp_thresh,
                "skor": confidence,
                "distance_chi2": round(best_dist, 2),
                "threshold": lbph_threshold,
                "threshold_chi2": lbph_threshold,
                "semua_jarak": all_distances,
                "dimensi_vektor": int(len(lbp_vec)),
                "tahapan": [
                    "1. Deteksi Haar Cascade → ROI BGR",
                    f"2. Liveness Color-LBP (Cr,Cb,H,S): d_real={chi2_liveness:.3f} < d_spoof={chi2_spoof:.3f} → REAL",
                    "3. Prapemrosesan: Gray 100×100 + EqualizeHist",
                    "4. Ekstraksi LBPH: Grid 8×8 (16.384-D)",
                    f"5. Identitas Chi-Square: χ²={best_dist:.2f} ≤ {lbph_threshold} → {'Diterima (' + str(wajah_result) + ')' if granted else 'Ditolak (Unknown)'}",
                ],
                "catatan": (
                    f"{best_name} (χ² = {best_dist:.2f} ≤ {lbph_threshold})"
                    if granted
                    else f"Tidak dikenal (χ² = {best_dist:.2f} > {lbph_threshold})"
                ),
                "bbox": bbox_info,
            }

        # --- Encode semua gambar preview (3 step jika spoof, 6 step jika real) ---
        previews = []
        for step_no, label, desc, img_step, target_sz in raw_steps:
            if img_step is None or img_step.size == 0:
                b64_str = ""
            else:
                sh, sw = img_step.shape[:2]
                if max(sh, sw) > target_sz:
                    sc = target_sz / max(sh, sw)
                    resized = cv2.resize(img_step, (int(sw * sc), int(sh * sc)), interpolation=cv2.INTER_AREA)
                elif max(sh, sw) < 180:
                    sc = 220 / max(sh, sw)
                    resized = cv2.resize(img_step, (int(sw * sc), int(sh * sc)), interpolation=cv2.INTER_NEAREST)
                else:
                    resized = img_step
                _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                b64_str = base64.b64encode(buf).decode("ascii")
            previews.append({"step": step_no, "label": label, "desc": desc, "image": b64_str})

        payload["previews"] = previews
        return http_code, payload
    except Exception as exc:
        return 500, {"error": f"ekstraksi CV gagal: {exc}"}
