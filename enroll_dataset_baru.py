"""
Script enroll dataset baru ke faces_lbph.json
Membaca semua foto dari folder 'dataset baru/<NamaOrang>/*.jpg'
dan mengganti exemplar lama dengan yang baru.

Jalankan: python enroll_dataset_baru.py
"""

import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from datetime import datetime

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
DATASET_DIR  = Path("dataset baru")
LBPH_JSON    = Path("api_data/faces_lbph.json")
BACKUP_DIR   = Path("api_data/backup")
MAX_EXEMPLAR = 50   # simpan semua 50 foto sebagai exemplar

# Mapping nama folder → nama kunci di DB (atur jika berbeda)
NAME_MAP = {
    "Faza":   "Faza",
    "Raffin": "Raffin",
    "Rafy":   "Rafi",   # folder Rafy → kunci Rafi di DB
}
# ──────────────────────────────────────────────────────────────────────────────


def lbp_same_as_engine(roi: np.ndarray) -> np.ndarray:
    """
    LBP IDENTIK dengan _extract_lbp() di engine.py:
    - Resize ke 100x100
    - LBP pada area 98x98 (border 1px dikurangi)
    - Grid 8x8 via array_split
    - Normalisasi per cell
    """
    gray = cv2.resize(roi, (100, 100))
    center = gray[1:-1, 1:-1]   # 98x98
    lbp = np.zeros_like(center, dtype=np.uint8)
    shifts = [
        (-1, -1, 0), (-1, 0, 1), (-1, 1, 2),
        (0,  1,  3), (1,  1, 4), (1,  0, 5),
        (1, -1,  6), (0, -1, 7)
    ]
    for dy, dx, bit in shifts:
        neighbor = gray[1+dy:99+dy, 1+dx:99+dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hists = []
    for row in np.array_split(lbp, 8, axis=0):
        for col in np.array_split(row, 8, axis=1):
            h, _ = np.histogram(col, bins=256, range=(0, 256))
            h = h.astype(np.float32)
            h /= (h.sum() + 1e-7)
            hists.append(h)
    return np.concatenate(hists)   # shape: (16384,)


def extract_lbp_from_file(img_path: Path, haar) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))  # sama dengan engine.py
    roi = cv2.equalizeHist(roi)
    return lbp_same_as_engine(roi)


def main():
    if not DATASET_DIR.exists():
        print(f"[ERROR] Folder '{DATASET_DIR}' tidak ditemukan.")
        return

    # Backup DB lama
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if LBPH_JSON.exists():
        backup_path = BACKUP_DIR / f"faces_lbph_backup_{ts}.json"
        shutil.copy2(LBPH_JSON, backup_path)
        print(f"[BACKUP] DB lama disimpan ke: {backup_path}")
        db = json.loads(LBPH_JSON.read_text(encoding="utf-8"))
    else:
        db = {}
        print("[INFO] faces_lbph.json belum ada, akan dibuat baru.")

    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print(f"\n=== Enroll dari '{DATASET_DIR}' ===\n")
    summary = {}

    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        folder_name = person_dir.name
        db_key = NAME_MAP.get(folder_name, folder_name)  # fallback: nama folder

        imgs = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        if not imgs:
            print(f"[SKIP] {folder_name}: tidak ada file gambar")
            continue

        print(f"[{folder_name}] → kunci DB: '{db_key}' | memproses {len(imgs)} foto...")

        exemplars = []
        failed = []

        for img_path in imgs[:MAX_EXEMPLAR]:
            vec = extract_lbp_from_file(img_path, haar)
            if vec is None:
                failed.append(img_path.name)
                continue
            exemplars.append(vec.tolist())

        if not exemplars:
            print(f"  ⚠ Semua foto gagal deteksi wajah! Cek kualitas gambar.")
            summary[folder_name] = {"status": "GAGAL", "enrolled": 0, "failed": len(failed)}
            continue

        # Hitung prototype (rata-rata exemplar)
        ex_arr = np.array(exemplars, dtype=np.float32)
        prototype = ex_arr.mean(axis=0).tolist()

        # Update DB
        db[db_key] = {
            "exemplars":  exemplars,
            "prototype":  prototype,
            "jumlah":     len(exemplars),
            "source":     f"enroll_dataset_baru/{folder_name}",
            "timestamp":  ts,
        }

        # Hitung self Chi-Square (untuk verifikasi)
        ex_np = np.array(exemplars, dtype=np.float32)
        proto_np = np.array(prototype, dtype=np.float32)
        self_dists = []
        for ex in ex_np:
            d = float(0.5 * np.sum(((ex - proto_np)**2) / (ex + proto_np + 1e-7)))
            self_dists.append(d)
        self_arr = np.array(self_dists)

        print(f"  ✅ Enrolled: {len(exemplars)} foto | Gagal deteksi: {len(failed)}")
        if failed:
            print(f"     Gagal: {', '.join(failed)}")
        print(f"  📊 Self Chi-Square (vs prototype): min={self_arr.min():.4f} mean={self_arr.mean():.4f} max={self_arr.max():.4f}")
        print()

        summary[folder_name] = {
            "status": "OK",
            "db_key": db_key,
            "enrolled": len(exemplars),
            "failed": len(failed),
            "self_chi2_min":  round(float(self_arr.min()), 4),
            "self_chi2_mean": round(float(self_arr.mean()), 4),
            "self_chi2_max":  round(float(self_arr.max()), 4),
        }

    # Simpan DB baru
    LBPH_JSON.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] faces_lbph.json diperbarui ({LBPH_JSON.stat().st_size/1024:.1f} KB)")

    print("\n=== RINGKASAN ===")
    for name, info in summary.items():
        if info["status"] == "OK":
            print(f"  {name:10s} → {info['enrolled']} exemplar | "
                  f"Chi2 mean={info['self_chi2_mean']:.4f} max={info['self_chi2_max']:.4f}")
        else:
            print(f"  {name:10s} → GAGAL (0 exemplar terproses)")

    print("\n✅ Selesai. Restart server: python app.py")


if __name__ == "__main__":
    main()
