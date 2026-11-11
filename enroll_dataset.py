#!/usr/bin/env python3
"""
enroll_dataset.py — Mendaftarkan (enroll) wajah secara batch dari folder dataset.

Struktur folder dataset yang diharapkan:
  dataset/
    Faris/
      foto1.jpg
      foto2.png
    Budi/
      foto_a.jpg
      foto_b.jpg

Jalankan:
  python enroll_dataset.py
  python enroll_dataset.py --dataset path/ke/folder_dataset
  python enroll_dataset.py --dataset dataset --data-dir api_data
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Tambahkan direktori saat ini ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.detector import Detector
from core.recognizer import Recognizer
from core.store import Store


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    ap = argparse.ArgumentParser(description="Batch enroll wajah dari folder dataset ke faces.json")
    ap.add_argument(
        "--dataset",
        default="dataset",
        help="Path ke folder dataset berisi subfolder nama orang (default: 'dataset')",
    )
    ap.add_argument(
        "--data-dir",
        default="api_data",
        help="Path ke folder penyimpanan data backend (default: 'api_data')",
    )
    ap.add_argument(
        "--yunet-model",
        default="face_detection_yunet_2023mar.onnx",
        help="Path ke model YuNet ONNX",
    )
    ap.add_argument(
        "--sface-model",
        default=os.path.join("models", "face_recognition_sface_2021dec.onnx"),
        help="Path ke model SFace ONNX",
    )
    ap.add_argument(
        "--clear",
        action="store_true",
        help="Hapus semua wajah lama di faces.json sebelum mendaftarkan dataset baru",
    )
    return ap.parse_args()


def enroll_dataset(dataset_dir: str, data_dir: str, yunet_model: str, sface_model: str, clear: bool = False):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"[ERROR] Folder dataset tidak ditemukan: '{dataset_path.resolve()}'")
        print("Silakan buat folder dan susun foto dengan struktur:")
        print(f"  {dataset_dir}/")
        print("    NamaOrang1/foto1.jpg")
        print("    NamaOrang2/foto2.jpg")
        return

    print("=" * 60)
    print(" Smart House Face Recognition — Batch Dataset Enrollment (SFace)")
    print("=" * 60)

    # Inisialisasi komponen
    print("[1/3] Memuat model deteksi (YuNet)...")
    detector = Detector(yunet_model)
    if not detector.kind == "yunet":
        print(f"[WARN] Detektor YuNet tidak aktif ({detector.reason}), memakai {detector.kind}")

    print("[2/3] Memuat model pengenalan (SFace)...")
    recognizer = Recognizer(sface_model=sface_model, backend="sface")
    if not recognizer.ready or recognizer.kind != "sface":
        print(f"[ERROR] Model SFace gagal dimuat: {recognizer.reason}")
        return

    print("[3/3] Memuat storage database...")
    store = Store(data_dir)

    if clear:
        print("[!] Menghapus database wajah lama (--clear)...", flush=True)
        store.faces = {}
        store.save_faces()

    # Folder yang diabaikan (bukan nama orang, misal folder anti-spoof)
    skip_names = {"spoof", "fake", "attack", "unknown", "dataset_unknown"}

    # Cari semua subfolder di dataset
    person_dirs = [
        d for d in sorted(dataset_path.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]

    # Filter folder yang valid
    valid_dirs = []
    for d in person_dirs:
        if d.name.lower() in skip_names:
            print(f"ℹ️  Mengabaikan folder '{d.name}' (data anti-spoof/bukan nama orang).", flush=True)
        else:
            valid_dirs.append(d)
    person_dirs = valid_dirs

    if not person_dirs:
        print(f"[WARN] Tidak ada subfolder orang di dalam '{dataset_dir}'.", flush=True)
        print("Pastikan ada subfolder nama orang (misal: 'dataset/Faris/').", flush=True)
        return

    print(f"\nDitemukan {len(person_dirs)} orang di folder '{dataset_dir}'. Mulai ekstraksi fitur...", flush=True)
    print("-" * 60, flush=True)

    total_sukses = 0

    for p_dir in person_dirs:
        name = p_dir.name
        img_files = [f for f in p_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]

        if not img_files:
            print(f"⚠️  [{name}] Tidak ada file gambar (.jpg/.png) di folder ini. Dilewati.", flush=True)
            continue

        print(f"⏳ [{name:15s}] Memproses {len(img_files)} foto...", end="", flush=True)
        embs = []
        gagal_count = 0

        for idx, img_path in enumerate(img_files, start=1):
            if idx % 25 == 0 or idx == len(img_files):
                print(f" [{idx}/{len(img_files)}]", end="", flush=True)

            img = cv2.imread(str(img_path))
            if img is None:
                gagal_count += 1
                continue

            # Deteksi wajah YuNet
            details = detector.detect_detailed(img)
            accepted = [d for d in details if d["accepted"]]
            if not accepted:
                # Coba deteksi tanpa strict_geometry jika foto cropping rapat
                accepted = [d for d in details if d["score"] >= 0.5]

            if not accepted:
                gagal_count += 1
                continue

            best = accepted[0]
            box0 = (best["x"], best["y"], best["w"], best["h"])
            raw = best.get("raw")
            landmarks = best.get("landmarks")

            emb = recognizer.embedding_from_frame(img, box0, landmarks=landmarks, raw=raw)
            if emb is not None:
                embs.append(emb)
            else:
                gagal_count += 1

        print()  # Baris baru setelah progress selesai
        if embs:
            # Rata-ratakan semua embedding agar prototype lebih robust
            prototype = np.mean(embs, axis=0)
            norm = np.linalg.norm(prototype)
            if norm > 0:
                prototype = prototype / norm

            store.set_face(name, prototype, len(embs))
            total_sukses += 1
            print(f"✅ [{name:15s}] Berhasil: {len(embs)} foto terproses" + (f" ({gagal_count} foto tanpa wajah)" if gagal_count > 0 else ""), flush=True)
        else:
            print(f"❌ [{name:15s}] Gagal: Tidak ada wajah yang berhasil dideteksi dari {len(img_files)} foto.", flush=True)

    print("-" * 60)
    print(f"Selesai! {total_sukses} orang berhasil disimpan ke '{store.faces_path}'.")
    print(f"Total wajah aktif di database saat ini: {len(store.list_faces())} orang.")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    enroll_dataset(
        dataset_dir=args.dataset,
        data_dir=args.data_dir,
        yunet_model=args.yunet_model,
        sface_model=args.sface_model,
        clear=args.clear,
    )
