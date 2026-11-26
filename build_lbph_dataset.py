"""
build_lbph_dataset.py â€” Ekstrak prototype LBPH dari folder foto dataset.
Menggunakan 100% Classical Computer Vision: Haar Cascade Classifier (Viola-Jones).
Jalankan:
  python build_lbph_dataset.py "path/ke/folder_dataset"
"""

import sys
import os
import glob
import json
import cv2
import numpy as np
from core.cv.lbph import extract_lbp


def build_lbph_dataset(dataset_dir: str = ".", output_file: str = "api_data/faces_lbph.json"):
    # Detektor Klasik Haar Cascade (Viola-Jones)
    xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(xml_path)
    if face_cascade.empty():
        raise RuntimeError(f"Gagal memuat Haar Cascade dari: {xml_path}")

    db = {}
    color_samples = []
    EXCLUDE = ["spoof", "dataset_unknown", "__pycache__", ".ipynb_checkpoints", "models", "mobilefacenet", "fas", "api_data", "core", "routes"]

    for folder in sorted(glob.glob(f"{dataset_dir}/*")):
        if not os.path.isdir(folder):
            continue
        name = folder.replace("\\", "/").rstrip("/").split("/")[-1]
        if name.lower() in EXCLUDE:
            continue

        images = sorted(glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.jpeg") + glob.glob(f"{folder}/*.png"))
        if not images:
            continue

        print(f"Memproses {name} ({len(images)} gambar)...")
        histograms = []

        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            h_f, w_f = frame.shape[:2]
            if max(h_f, w_f) > 640:
                sc = 640 / max(h_f, w_f)
                frame = cv2.resize(frame, (int(w_f * sc), int(h_f * sc)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                h_img, w_img = frame.shape[:2]
                if h_img <= 220 and w_img <= 220:
                    face_crop = frame
                else:
                    continue
            else:
                bx, by, bw, bh = max(faces, key=lambda f: f[2] * f[3])
                h_img, w_img = frame.shape[:2]
                if h_img <= 220 and w_img <= 220:
                    # Sudah merupakan potongan wajah standar (112x112)
                    face_crop = frame
                else:
                    face_crop = frame[max(0, by):by+bh, max(0, bx):bx+bw]
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(face_gray)
            features = extract_lbp(equalized)
            histograms.append(features)

            # Ekstrak Color-LBP Cr+Cb untuk Liveness Prototype (Boulkenafet et al., IEEE TIFS 2016)
            ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
            h_cr = extract_lbp(ycrcb[:, :, 1], grid_x=1, grid_y=1)
            h_cb = extract_lbp(ycrcb[:, :, 2], grid_x=1, grid_y=1)
            color_samples.append(np.concatenate([h_cr, h_cb]))

        if histograms:
            # 1. Simpan K=25 exemplar nyata untuk pencocokan Multi-Exemplar 1-NN
            num_ex = min(25, len(histograms))
            step = len(histograms) / num_ex
            ex_indices = [int(i * step) for i in range(num_ex)]
            exemplars = [histograms[i].tolist() for i in ex_indices]

            # 2. Vektor medoid pusat untuk fallback backward-compatibility
            hist_matrix = np.asarray(histograms, dtype=np.float32)
            mean_vec = np.mean(hist_matrix, axis=0)
            diffs = np.sum((hist_matrix - mean_vec) ** 2, axis=1)
            medoid_idx = int(np.argmin(diffs))
            central_histogram = histograms[medoid_idx].tolist()

            db[name] = {
                "exemplars": exemplars,
                "histogram": central_histogram,
                "count": len(histograms),
                "dim": len(histograms[0])
            }
            print(f"  -> {name}: {len(histograms)} terdeteksi via Haar Cascade, {num_ex} exemplar disimpan.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(db, f)

    if color_samples:
        color_proto = np.mean(color_samples, axis=0).tolist()
        proto_file = os.path.join(os.path.dirname(output_file), "color_lbp_proto.json")
        with open(proto_file, "w", encoding="utf-8") as f:
            json.dump({"proto": color_proto, "dim": len(color_proto), "count": len(color_samples)}, f)
        print(f"Prototipe Color-LBP Liveness disimpan di {proto_file} ({len(color_samples)} sampel).")

    print(f"\nSelesai! Database LBPH Haar Cascade disimpan di {output_file} ({len(db)} profil).")


if __name__ == "__main__":
    dir_target = sys.argv[1] if len(sys.argv) > 1 else "."
    build_lbph_dataset(dir_target)
