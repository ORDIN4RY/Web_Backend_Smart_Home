"""
core/ai/anti_spoof.py — Liveness detection menggunakan MiniFASNet (Deep Learning).

Sistem Cerdas: Neural network-based face anti-spoofing (ensemble MiniFASNetV2 + MiniFASNetV1SE).
"""

from pathlib import Path


class AntiSpoofChecker:
    """
    Wrapper MiniFASNet untuk Face Anti-Spoofing (Sistem Cerdas).

    Atribut:
      ready     : bool  — True jika model berhasil dimuat
      reason    : str   — pesan error jika ready=False
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
                model_dir = str(Path(__file__).resolve().parent.parent.parent / "fas" / "models")
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
        Periksa liveness via MiniFASNet.

        Args:
            frame : np.ndarray BGR dari kamera
            box   : (x, y, w, h) bounding box wajah

        Returns:
            (is_real: bool, score: float, label: int)
        """
        if not self.ready or self.predictor is None:
            return True, 1.0, 1
        try:
            return self.predictor.predict(frame, box)
        except Exception as exc:
            print(f"[anti_spoof] error predict: {exc}")
            return True, 1.0, 1
