#!/usr/bin/env python3
"""
Coba MFN (MobileFaceNet) TANPA training dulu — buat laptop sekalipun.

Idenya beda dari train_mfn.py:
  train_mfn.py : mengubah bobot MFN lewat fine-tune (butuh dataset besar,
                 waktu lama, enaknya jalan di zakyserver).
  infer_mfn.py : bobot MFN pretrained TIDAK diubah sama sekali. Kita hanya
                 "mendaftarkan" (enroll) tiap orang: hitung embedding dari
                 beberapa foto referensinya, rata-ratakan jadi prototype.
                 Foto baru dikenali dengan cosine similarity ke prototype.

Ini sudah cukup untuk membuktikan pipeline jalan di laptopmu, dan untuk
mengukur apakah MFN pretrained saja sudah cukup akurat sebelum fine-tune.
Satu foto (bukan video realtime) di CPU lama pun biasanya < 1 detik.

Cara pakai (Windows, di Command Prompt / PowerShell):
  git clone https://github.com/foamliu/MobileFaceNet.git
  mkdir MobileFaceNet\\weights
  :: unduh mobilefacenet.pt dari rilis foamliu ke MobileFaceNet\\weights
  python -m venv fr-venv
  fr-venv\\Scripts\\activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu pillow numpy
  python infer_mfn.py --enroll enroll --test test.jpg
  python infer_mfn.py --enroll enroll --test-dir test --threshold 0.5

Struktur enroll (sama seperti dataset training):
  enroll/
    faza/*.jpg
    thoriq/*.jpg

Threshold awal 0.5 cuma titik mulai. Lihat skor cosine yang dicetak:
  - kalau foto orang terdaftar sering di bawah threshold -> turunkan
    sedikit, atau tambah foto enroll yang lebih beragam.
  - kalau foto orang luar sering di atas threshold -> naikkan threshold,
    atau itu tanda memang perlu fine-tune (train_mfn.py).
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
# Sengaja TANPA torchvision: di laptop cukup torch + pillow + numpy,
# dan torchvision adalah penyebab error "No module named 'torchvision'"
# di percobaan pertama. Preprocessing ditulis manual di bawah, hasilnya
# sama dengan ToTensor + Normalize(mean=0.5, std=0.5).


def load_backbone(mfn_repo: str, pretrained: str):
    repo = Path(mfn_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from mobilefacenet import MobileFaceNet
    except ImportError as exc:
        raise SystemExit(
            "Gagal import MobileFaceNet dari repo foamliu.\n"
            f"  --mfn-repo sekarang: {mfn_repo}\n"
            "  Pastikan sudah: git clone "
            "https://github.com/foamliu/MobileFaceNet.git\n"
            f"  Detail: {exc}"
        )
    model = MobileFaceNet()
    if not Path(pretrained).exists():
        raise SystemExit(f"File pretrained tidak ketemu: {pretrained}")
    state = torch.load(pretrained, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[mfn] pretrained dimuat: {pretrained} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    return model


def preprocess(path, img_size: int):
    """Resize 112x112 + normalisasi ke [-1, 1], tanpa torchvision."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H x W x C, 0..1
    arr = (arr - 0.5) / 0.5  # sama dengan Normalize(mean=0.5, std=0.5)
    arr = np.transpose(arr, (2, 0, 1))  # C x H x W
    return torch.from_numpy(arr).unsqueeze(0)


def embedding_of(model, path, img_size: int):
    x = preprocess(path, img_size)
    with torch.no_grad():
        emb = model(x)
        if emb.dim() > 2:
            emb = emb.flatten(1)
        emb = nn.functional.normalize(emb, p=2, dim=1)
    return emb.numpy()[0]


def build_prototypes(model, enroll_dir, img_size: int):
    enroll = Path(enroll_dir)
    persons = sorted(p.name for p in enroll.iterdir() if p.is_dir())
    if not persons:
        raise SystemExit(f"Tidak ada subfolder orang di: {enroll_dir}")
    protos = {}
    for person in persons:
        files = sorted(
            f for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for f in glob.glob(os.path.join(enroll_dir, person, pat))
        )
        if not files:
            print(f"[mfn] lewati {person}: tidak ada foto")
            continue
        embs = []
        for f in files:
            try:
                embs.append(embedding_of(model, f, img_size))
            except Exception as exc:
                print(f"[mfn] gagal baca {f}: {exc}")
        if embs:
            v = np.mean(embs, axis=0)
            protos[person] = v / (np.linalg.norm(v) + 1e-8)
            print(f"[mfn] enroll {person}: {len(embs)} foto")
    if not protos:
        raise SystemExit("Tidak ada prototype yang berhasil dibuat.")
    return protos


def recognise(model, path, img_size: int, protos, threshold):
    emb = embedding_of(model, path, img_size)
    names = list(protos.keys())
    mat = np.stack([protos[n] for n in names])
    sims = mat @ emb
    best = int(np.argmax(sims))
    return names[best], float(sims[best]), dict(zip(names, sims.tolist()))


def main():
    ap = argparse.ArgumentParser(description="Coba MFN pretrained tanpa training.")
    ap.add_argument("--enroll", required=True, help="Folder enroll, satu subfolder per orang")
    ap.add_argument("--test", default="", help="Satu file foto untuk dikenali")
    ap.add_argument("--test-dir", default="", help="Folder foto untuk dikenali (opsional)")
    ap.add_argument("--mfn-repo", default="MobileFaceNet")
    ap.add_argument("--pretrained", default=os.path.join("MobileFaceNet", "weights", "mobilefacenet.pt"))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Cosine di bawah ini = unknown (titik mulai, silakan disetel)")
    ap.add_argument("--img-size", type=int, default=112)
    args = ap.parse_args()

    torch.set_num_threads(4)
    model = load_backbone(args.mfn_repo, args.pretrained)
    protos = build_prototypes(model, args.enroll, args.img_size)

    targets = []
    if args.test:
        targets.append(args.test)
    if args.test_dir:
        targets.extend(sorted(
            f for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for f in glob.glob(os.path.join(args.test_dir, "**", pat), recursive=True)
        ))
    if not targets:
        print("[mfn] enroll selesai. Tambahkan --test foto.jpg untuk mencoba.")
        return
    for path in targets:
        name, score, all_sims = recognise(model, path, args.img_size, protos, args.threshold)
        verdict = name if score >= args.threshold else "UNKNOWN"
        print(f"\n{path} -> {verdict} (cosine={score:.3f}, threshold={args.threshold})")
        for n, s in sorted(all_sims.items(), key=lambda kv: -kv[1]):
            print(f"    {n}: {s:.3f}")


if __name__ == "__main__":
    main()
