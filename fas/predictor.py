# -*- coding: utf-8 -*-
"""
MiniFASNet Predictor untuk verifikasi keaslian wajah (Liveness / Anti-Spoofing).
Mendukung inferensi model tunggal atau ensemble 2 skala (2.7x dan 4.0x).
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .crop import FaceCropper
from .minifasnet import MiniFASNetV1SE, MiniFASNetV2


MODEL_DICT = {
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
}


def _get_kernel(height, width):
    return ((height + 15) // 16, (width + 15) // 16)


def _parse_model_name(filename):
    """
    Parse nama file model seperti:
      '2.7_80x80_MiniFASNetV2.pth' -> h=80, w=80, type='MiniFASNetV2', scale=2.7
      '4_0_0_80x80_MiniFASNetV1SE.pth' -> h=80, w=80, type='MiniFASNetV1SE', scale=4.0
    """
    parts = filename.replace(".pth", "").split("_")
    # Bagian terakhir adalah nama tipe model
    model_type = parts[-1]
    # Bagian sebelum tipe model adalah dimensi (misal 80x80)
    dim_part = parts[-2]
    h, w = (int(x) for x in dim_part.split("x"))

    # Scale ada di bagian depan
    if parts[0] == "4":
        scale = 4.0
    elif parts[0] == "org":
        scale = 1.0
    else:
        try:
            scale = float(parts[0])
        except ValueError:
            scale = 2.7
    return h, w, model_type, scale


class MiniFASNetPredictor:
    def __init__(self, model_dir=None, device=None, threshold=0.60):
        """
        model_dir: path ke folder yang berisi file-file .pth MiniFASNet (default: fas/models)
        device: 'cpu' atau 'cuda' (default: auto)
        threshold: ambang batas skor untuk menyatakan 'Real Face' (default: 0.60)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.cropper = FaceCropper()

        if model_dir is None:
            base_dir = Path(__file__).resolve().parent
            model_dir = base_dir / "models"
        self.model_dir = Path(model_dir)

        self.models = []
        self._load_models()

    def _load_models(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Folder model tidak ditemukan: {self.model_dir}")

        pth_files = sorted(self.model_dir.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"Tidak ada file .pth di folder {self.model_dir}")

        for pth in pth_files:
            try:
                h, w, model_type, scale = _parse_model_name(pth.name)
                kernel = _get_kernel(h, w)
                model_cls = MODEL_DICT.get(model_type, MiniFASNetV2)
                net = model_cls(conv6_kernel=kernel)

                state_dict = torch.load(str(pth), map_location=self.device, weights_only=True)
                clean_state = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state_dict.items()
                }
                net.load_state_dict(clean_state)
                net.to(self.device)
                net.eval()

                self.models.append({
                    "name": pth.name,
                    "net": net,
                    "h": h,
                    "w": w,
                    "scale": scale,
                })
            except Exception as e:
                print(f"[MiniFASNet] Gagal memuat {pth.name}: {e}")

        if not self.models:
            raise RuntimeError("Tidak ada model MiniFASNet yang berhasil dimuat.")

    def _to_tensor(self, img_bgr):
        """
        Konversi BGR OpenCV ke Tensor [1, 3, H, W] float [0.0 - 255.0].
        PENTING: Pre-trained MiniFASNet (Minivision) dilatih dengan format BGR float [0.0 - 255.0]
        tanpa pembagian 255.0. Membagi dengan 255.0 akan merusak rentang aktivasi model.
        """
        tensor = torch.from_numpy(img_bgr.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_bgr, bbox):
        """
        image_bgr: frame / gambar OpenCV (BGR)
        bbox: [x, y, w, h] koordinat wajah

        Return:
          is_real (bool): True jika wajah asli/live, False jika spoof/foto/layar
          real_score (float): skor kepercayaan Real (0.0 s.d 1.0)
          label (int): 1 = Real, 0 atau 2 = Fake
        """
        total_pred = np.zeros(3, dtype=np.float32)

        for item in self.models:
            patch = self.cropper.crop(
                image_bgr,
                bbox,
                scale=item["scale"],
                out_w=item["w"],
                out_h=item["h"],
            )
            x = self._to_tensor(patch)
            with torch.no_grad():
                out = item["net"](x)
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
                total_pred += probs

        avg_pred = total_pred / len(self.models)
        label = int(np.argmax(avg_pred))
        real_score = float(avg_pred[1])  # Class 1 = Real Face

        is_real = (label == 1 and real_score >= self.threshold)
        return is_real, real_score, label
