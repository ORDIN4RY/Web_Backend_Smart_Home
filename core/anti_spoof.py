"""
core/anti_spoof.py — Liveness detection menggunakan MiniFASNet.

Deteksi apakah wajah di depan kamera adalah wajah ASLI (real) atau
wajah palsu (foto cetak, layar HP, topeng, dll).
"""

from pathlib import Path


class AntiSpoofChecker:
    """
    Wrapper MiniFASNet untuk Face Anti-Spoofing.

    Atribut:
      ready     : bool — True jika model berhasil dimuat
      reason    : str  — pesan error jika ready=False
      threshold : float — skor minimum agar dianggap REAL (default 0.60)
    """

    def __init__(self, model_dir: str = None, threshold: float = 0.60):
        self.ready = False
        self.reason = ""
        self.predictor = None
        self.threshold = threshold
        self._load(model_dir)

    def _load(self, model_dir: str):
        try:
            from fas.predictor import MiniFASNetPredictor
            if model_dir is None:
                model_dir = str(Path(__file__).resolve().parent.parent / "fas" / "models")
            self.predictor = MiniFASNetPredictor(
                model_dir=str(model_dir), threshold=self.threshold
            )
            self.ready = True
            print(f"[anti_spoof] MiniFASNet dimuat dari {model_dir}")
        except Exception as exc:
            self.reason = f"MiniFASNet belum siap ({exc})"
            self.ready = False
            print(f"[anti_spoof] {self.reason}")

    def check(self, frame, box: tuple) -> tuple:
        """
        Periksa liveness.

        Args:
            frame : np.ndarray BGR dari kamera
            box   : (x, y, w, h) bounding box wajah

        Returns:
            (is_real: bool, score: float, label: int)
              is_real = True  -> wajah ASLI
              is_real = False -> SPOOF / palsu
              score   -> skor "real" (0.0 – 1.0)
        """
        if not self.ready or self.predictor is None:
            # Jika model belum dimuat, anggap semua real (tidak memblokir alur)
            return True, 1.0, 1
        try:
            return self.predictor.predict(frame, box)
        except Exception as exc:
            print(f"[anti_spoof] error predict: {exc}")
            return True, 1.0, 1

    def check_image_bytes(self, image_bytes: bytes, box: tuple = None) -> tuple:
        """
        Periksa liveness dari bytes gambar (upload dari mobile).
        Jika box=None, gunakan full frame.
        """
        try:
            import numpy as np
            import cv2
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return False, 0.0, 0
            if box is None:
                h, w = frame.shape[:2]
                box = (0, 0, w, h)
            return self.check(frame, box)
        except Exception as exc:
            print(f"[anti_spoof] check_image_bytes error: {exc}")
            return True, 1.0, 1
