"""
core/ai/pipeline.py — Pipeline pengenalan wajah Sistem Cerdas end-to-end.

Sistem Cerdas (Deep Learning):
  Alur: YuNet (deteksi) → MiniFASNet (liveness) + Color-LBP Guard → SFace (embedding) → Cosine similarity

Ekspor:
  recognize_frame_ai(frame, ...)       — jalankan pipeline AI pada satu frame BGR (dipakai live camera & upload)
  recognize_ai(image_bytes, ...)       — endpoint wrapper untuk upload foto (/api/recognize mode="ai")
"""

import cv2
import numpy as np

from core.cv.lbp_liveness import (
    check_liveness,
    COLOR_LBP_REAL,
    COLOR_LBP_SPOOF,
    COLOR_LBP_THRESH,
)
from core.ai.recognizer import Recognizer


def recognize_frame_ai(
    frame,
    detector,
    recognizer,
    fas,
    store,
    real_mat=None,
    spoof_mat=None,
    lbp_thresh: float = COLOR_LBP_THRESH,
) -> dict:
    """
    Jalankan pipeline AI lengkap pada satu frame BGR (dipakai bersama oleh
    worker live camera dan endpoint upload foto).

    Returns:
        dict dengan kunci:
          status        : "ok" | "no_face" | "no_model" | "spoof" | "no_registered_faces" | "crop_too_small"
          accepted      : list deteksi wajah yang lolos filter
          bbox          : dict {x, y, w, h, frame_w, frame_h} | None
          liveness      : bool | None
          liveness_skor : float | None
          wajah         : str | None
          skor          : float | None
          granted       : bool
          catatan       : str
    """
    if real_mat is None:
        real_mat = COLOR_LBP_REAL
    if spoof_mat is None:
        spoof_mat = COLOR_LBP_SPOOF

    details = detector.detect_detailed(frame)
    accepted = [d for d in details if d["accepted"]]

    result = {
        "status": "no_face",
        "accepted": accepted,
        "bbox": None,
        "liveness": None,
        "liveness_skor": None,
        "wajah": None,
        "skor": None,
        "granted": False,
        "catatan": "tidak ada wajah di frame",
    }

    if not accepted:
        return result

    best_d = accepted[0]
    box0 = (best_d["x"], best_d["y"], best_d["w"], best_d["h"])
    result["bbox"] = {
        "x": int(best_d["x"]),
        "y": int(best_d["y"]),
        "w": int(best_d["w"]),
        "h": int(best_d["h"]),
        "frame_w": int(frame.shape[1]),
        "frame_h": int(frame.shape[0]),
    }

    if not recognizer.ready:
        result["status"] = "no_model"
        result["catatan"] = f"wajah terdeteksi, model belum dimuat: {recognizer.reason}"
        return result

    # --- Liveness Layer 1: MiniFASNet (Deep Learning) ---
    is_real, liveness_score = True, 1.0
    catatan_spoof = "DITOLAK: spoof / wajah palsu terdeteksi"
    if fas.ready:
        is_real, liveness_score, _ = fas.check(frame, box0)
        is_real = bool(is_real and liveness_score >= store.fas_threshold())
        if not is_real:
            catatan_spoof = f"DITOLAK: spoof / foto layar terdeteksi (skor real: {liveness_score:.2f})"

    # --- Liveness Layer 2: Color-LBP Guard (CV Klasik — tangkal layar OLED/AMOLED) ---
    if is_real:
        bx, by, bw, bh = box0
        patch_crop = frame[
            max(0, by):min(frame.shape[0], by + bh),
            max(0, bx):min(frame.shape[1], bx + bw),
        ]
        if patch_crop.size > 0:
            is_real_clbp, clbp_score, chi2_live, *_ = check_liveness(
                patch_crop, real_mat, spoof_mat, threshold=lbp_thresh
            )
            if not is_real_clbp:
                is_real = False
                liveness_score = round(float(clbp_score), 3)
                catatan_spoof = (
                    f"DITOLAK: spoof layar HP/cetak terdeteksi via Color-LBP 1-NN "
                    f"(χ² Real: {chi2_live:.3f})"
                )

    result["liveness"] = is_real
    result["liveness_skor"] = round(float(liveness_score), 3)

    if not is_real:
        result["status"] = "spoof"
        result["catatan"] = catatan_spoof
        return result

    # --- Cek wajah terdaftar ---
    faces = store.prototypes()
    if not faces:
        result["status"] = "no_registered_faces"
        result["catatan"] = "wajah REAL, belum ada wajah terdaftar"
        return result

    # --- Embedding SFace/MFN + Cosine Similarity ---
    emb = recognizer.embedding_from_frame(
        frame, box0,
        landmarks=best_d.get("landmarks"),
        raw=best_d.get("raw"),
        margin=store.margin(),
    )
    if emb is None:
        result["status"] = "crop_too_small"
        result["catatan"] = "crop terlalu kecil, dilewati"
        return result

    name, score = Recognizer.best_match(emb, faces)
    granted = bool(score >= store.threshold())
    result.update({
        "status": "ok",
        "wajah": name if granted else None,
        "skor": round(score, 3),
        "granted": granted,
        "catatan": (
            f"dikenali: {name}"
            if granted
            else "wajah REAL, skor di bawah threshold (unknown)"
        ),
    })
    return result


def recognize_ai(
    image_bytes: bytes,
    detector,
    recognizer,
    fas,
    store,
    real_mat=None,
    spoof_mat=None,
    lbp_thresh: float = COLOR_LBP_THRESH,
) -> tuple:
    """
    Pipeline pengenalan wajah upload mobile (/api/recognize mode="ai").
    Mengembalikan (http_code: int, payload: dict).
    """
    if not recognizer.ready:
        return 503, {"error": "model belum dimuat", "alasan": recognizer.reason}

    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return 400, {"error": "gambar tidak bisa di-decode"}

        # Resize ke max 1280px agar YuNet bisa deteksi wajah di foto HP resolusi tinggi
        h_f, w_f = frame.shape[:2]
        if max(h_f, w_f) > 1280:
            scale = 1280 / max(h_f, w_f)
            frame = cv2.resize(frame, (int(w_f * scale), int(h_f * scale)), interpolation=cv2.INTER_AREA)

        res = recognize_frame_ai(
            frame, detector, recognizer, fas, store,
            real_mat=real_mat, spoof_mat=spoof_mat, lbp_thresh=lbp_thresh,
        )

        status = res["status"]
        if status == "no_face":
            return 404, {"error": "tidak ada wajah terdeteksi di foto"}
        if status == "crop_too_small":
            return 400, {"error": "crop wajah terlalu kecil"}
        if status == "spoof":
            return 403, {
                "granted": False,
                "wajah": None,
                "liveness": False,
                "liveness_skor": res["liveness_skor"],
                "catatan": res["catatan"],
                "bbox": res["bbox"],
            }
        if status == "no_registered_faces":
            return 200, {
                "granted": False,
                "wajah": None,
                "liveness": True,
                "liveness_skor": res["liveness_skor"],
                "catatan": res["catatan"],
                "bbox": res["bbox"],
            }

        return 200, {
            "granted": res["granted"],
            "wajah": res["wajah"],
            "skor": res["skor"],
            "threshold": store.threshold(),
            "liveness": True,
            "liveness_skor": res["liveness_skor"],
            "catatan": res["catatan"],
            "bbox": res["bbox"],
        }
    except Exception as exc:
        return 500, {"error": f"recognize_photo (AI) gagal: {exc}"}
