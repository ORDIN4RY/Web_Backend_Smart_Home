"""
core/ai/recognizer.py — Face recognition berbasis SFace (OpenCV FaceRecognizerSF)
dengan fallback opsional ke MobileFaceNet (PyTorch).

Sistem Cerdas: Deep Learning-based face embedding.
  SFace → vektor 128-D (L2-normalized), alignment instan via alignCrop, threshold Cosine ≥ 0.363
  MFN   → MobileFaceNet PyTorch fallback
"""

from pathlib import Path
import cv2
import numpy as np

IMG_SIZE = 112
DEFAULT_SFACE_THRESHOLD = 0.363


def clamp_crop(frame, x, y, w, h, margin: float = 0.0):
    """Potong bounding box + margin, dijepit ke dimensi frame."""
    height, width = frame.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * (1 + 2 * margin), h * (1 + 2 * margin)
    x0 = max(0, int(round(cx - nw / 2)))
    y0 = max(0, int(round(cy - nh / 2)))
    x1 = min(width, int(round(cx + nw / 2)))
    y1 = min(height, int(round(cy + nh / 2)))
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None
    return frame[y0:y1, x0:x1]


class Recognizer:
    """
    Wrapper model pengenalan wajah (Sistem Cerdas).
    Mendukung SFace (default) dan MobileFaceNet (opsional fallback).
    """

    def __init__(
        self,
        sface_model: str = "models/face_recognition_sface_2021dec.onnx",
        backend: str = "sface",
        infer_script: str = "infer_mfn.py",
        mfn_repo: str = "MobileFaceNet",
        pretrained: str = "MobileFaceNet/weights/mobilefacenet.pt",
    ):
        self.ready = False
        self.kind = "tidak_ada"
        self.reason = ""
        self.backend = (backend or "sface").lower()
        self._sface = None
        self._mfn_model = None
        self._mfn_mod = None

        if self.backend == "sface":
            self._init_sface(sface_model)
            if not self.ready and Path(pretrained).exists():
                print(f"[recognizer] SFace gagal ({self.reason}); mencoba fallback MobileFaceNet...")
                self._init_mfn(infer_script, mfn_repo, pretrained)
        else:
            self._init_mfn(infer_script, mfn_repo, pretrained)

    def _init_sface(self, sface_model: str):
        p = Path(sface_model)
        if not p.exists():
            self.reason = f"File model SFace tidak ditemukan: {sface_model}"
            return

        try:
            if hasattr(cv2, "FaceRecognizerSF_create"):
                self._sface = cv2.FaceRecognizerSF_create(str(p), "")
            elif hasattr(cv2, "FaceRecognizerSF"):
                self._sface = cv2.FaceRecognizerSF.create(str(p), "")
            else:
                self.reason = "OpenCV tidak memiliki FaceRecognizerSF (perlu opencv-python >= 4.5.4)"
                return

            self.ready = True
            self.kind = "sface"
            print(f"[recognizer] SFace ONNX dimuat dari {sface_model}", flush=True)
        except Exception as exc:
            self.reason = f"Gagal inisialisasi SFace ({exc})"

    def _init_mfn(self, infer_script: str, mfn_repo: str, pretrained: str):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("infer_mfn", infer_script)
            if spec is None or spec.loader is None:
                self.reason = f"infer_mfn.py tidak ketemu: {infer_script}"
                return
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._mfn_mod = mod
            self._mfn_model = mod.load_backbone(mfn_repo, pretrained)
            self.ready = True
            self.kind = "mobilefacenet"
            print(f"[recognizer] MobileFaceNet dimuat dari {pretrained}", flush=True)
        except Exception as exc:
            self.reason = f"Gagal memuat MobileFaceNet ({exc})"

    # ------------------------------------------------------------------
    # Ekstraksi Fitur / Embedding (Deep Learning)
    # ------------------------------------------------------------------

    def embedding_from_crop(self, bgr_crop):
        """Hitung embedding 128-D L2-normalized dari crop gambar wajah (BGR)."""
        if not self.ready or bgr_crop is None:
            return None

        if self.kind == "sface":
            try:
                if bgr_crop.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    bgr_crop = cv2.resize(bgr_crop, (IMG_SIZE, IMG_SIZE))
                feat = self._sface.feature(bgr_crop)
                emb = np.array(feat[0], dtype=np.float32)
                norm = np.linalg.norm(emb)
                return (emb / norm) if norm > 0 else emb
            except Exception as exc:
                print(f"[recognizer] SFace feature error: {exc}")
                return None

        if self.kind == "mobilefacenet" and self._mfn_mod is not None:
            try:
                return self._mfn_mod.infer(self._mfn_model, bgr_crop)
            except Exception as exc:
                print(f"[recognizer] MFN feature error: {exc}")
                return None

        return None

    def _build_sface_face_row(self, box: tuple, landmarks=None, raw=None):
        """Susun array 15-float format YuNet untuk FaceRecognizerSF.alignCrop."""
        x, y, w, h = box
        if raw is not None and len(raw) >= 14:
            vals = [float(raw[i]) for i in range(14)] + [1.0]
            return np.array(vals, dtype=np.float32)

        if landmarks is not None and len(landmarks) >= 5:
            pts = (
                landmarks
                if isinstance(landmarks[0], (list, tuple))
                else [[landmarks[i], landmarks[i + 1]] for i in range(0, len(landmarks), 2)]
            )
            vals = [float(x), float(y), float(w), float(h)]
            for pt in pts[:5]:
                vals.extend([float(pt[0]), float(pt[1])])
            vals.append(1.0)
            return np.array(vals, dtype=np.float32)

        return None

    def embedding_from_frame(self, frame, box: tuple, landmarks=None, raw=None, margin: float = 0.1):
        """
        Hitung embedding dari frame kamera lengkap + bounding box.
        Untuk SFace, prioritaskan alignCrop berbasis 5-landmark agar akurat terhadap pose.
        """
        if self.kind == "sface" and self._sface is not None:
            face_arr = self._build_sface_face_row(box, landmarks=landmarks, raw=raw)
            if face_arr is not None:
                try:
                    aligned = self._sface.alignCrop(frame, face_arr)
                    return self.embedding_from_crop(aligned)
                except Exception:
                    pass

        crop = clamp_crop(frame, *box, margin=margin)
        if crop is not None:
            return self.embedding_from_crop(crop)
        return None

    # ------------------------------------------------------------------
    # Pencocokan / Cosine Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def best_match(embedding, prototypes: dict):
        """
        Cosine similarity (dot product) ke semua prototype.
        Kembalikan (nama, skor) terbaik, atau (None, 0.0) jika kosong.
        """
        if not prototypes or embedding is None:
            return None, 0.0
        names = sorted(prototypes.keys())
        mat = np.stack([prototypes[n]["embedding"] for n in names])
        sims = mat @ embedding
        best = int(np.argmax(sims))
        return names[best], float(sims[best])
