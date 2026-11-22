"""
build_lbph_dataset.py — Ekstrak prototype LBPH dari folder foto dataset.
Jalankan:
  python build_lbph_dataset.py "path/ke/folder_dataset"
"""

import sys
import glob
import json
import cv2
import numpy as np
from core.detector import Detector
from core.engine import _extract_lbp, _resize_for_detection


def build_lbph_dataset(dataset_dir: str, output_file: str = "api_data/faces_lbph.json"):
    detector = Detector("face_detection_yunet_2023mar.onnx")
    db = {}

    for folder in glob.glob(f"{dataset_dir}/*"):
        name = folder.replace("\\", "/").rstrip("/").split("/")[-1]
        if name.lower() in ["spoof", "__pycache__"]:
            continue

        images = glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.jpeg") + glob.glob(f"{folder}/*.png")
        vecs = []

        for p in images:
            frame = cv2.imread(p)
            if frame is None:
                continue
            frame = _resize_for_detection(frame, max_dim=1280)
            details = detector.detect_detailed(frame)
            accepted = [d for d in details if d.get("accepted") or d.get("score", 0) >= 0.5]
            if not accepted:
                continue

            best_d = accepted[0]
            bx, by, bw, bh = int(best_d["x"]), int(best_d["y"]), int(best_d["w"]), int(best_d["h"])
            h_img, w_img = frame.shape[:2]
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(w_img - bx, bw), min(h_img - by, bh)

            face_crop = frame[by:by+bh, bx:bx+bw]
            if face_crop.size == 0:
                continue

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            vecs.append(_extract_lbp(eq))

        if vecs:
            proto = np.mean(vecs, axis=0)
            db[name] = {
                "histogram": proto.tolist(),
                "count": len(vecs),
                "dim": int(len(proto)),
            }
            print(f"[LBPH] {name}: {len(vecs)} foto berhasil diekstrak")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"\nSelesai! {len(db)} profil identitas tersimpan di '{output_file}'.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "../backup testing models/dataset"
    build_lbph_dataset(target)
