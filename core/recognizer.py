"""
core/recognizer.py — Face recognition berbasis SFace (OpenCV FaceRecognizerSF)
dengan fallback opsional ke MobileFaceNet (PyTorch).

Default backend: SFace (ONNX native via OpenCV DNN).
Keunggulan SFace:
  - Berpasangan resmi dengan YuNet
  - Alignment instan via cv2.FaceRecognizerSF.alignCrop
  - Sangat ringan & cepat di CPU (C++ OpenCV engine, tanpa overhead PyTorch)
  - Vektor fitur 128-D L2-normalized
  - Rekomendasi threshold Cosine: 0.363
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
    Wrapper model pengenalan wajah.
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
            self._mfn_model = mod.load_backbone(mfn_repo, pretrained)
            self.ready = True
            self.kind = "mobilefacenet"
            print(f"[recognizer] MobileFaceNet dimuat dari {pretrained}", flush=True)
        except Exception as exc:
            self.reason = f"Gagal memuat MobileFaceNet ({exc})"

    # ------------------------------------------------------------------
    # Ekstraksi Fitur / Embedding
    # ------------------------------------------------------------------

    def embedding_from_crop(self, bgr_crop):
        """Hitung embedding 128-D dari crop gambar wajah (BGR)."""
        if not self.ready or bgr_crop is None:
            return None

        if self.kind == "sface":
            try:
                # SFace mengharapkan crop 112x112
                if bgr_crop.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    bgr_crop = cv2.resize(bgr_crop, (IMG_SIZE, IMG_SIZE))
                feat = self._sface.feature(bgr_crop)
                emb = np.array(feat[0], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb
            except Exception as exc:
                print(f"[recognizer] SFace feature error: {exc}")
                return None

        # Fallback MFN
        if self.kind == "mobilefacenet" and self._mfn_model is not None:
            try:
                import torch
                import torch.nn as nn
                from PIL import Image
                rgb = bgr_crop[:, :, ::-1]
                img = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                arr = np.transpose(arr, (2, 0, 1))
                x = torch.from_numpy(arr).unsqueeze(0)
                with torch.no_grad():
                    emb = self._mfn_model(x)
                    if emb.dim() > 2:
                        emb = emb.flatten(1)
                    emb = nn.functional.normalize(emb, p=2, dim=1)
                return emb.numpy()[0]
            except Exception:
                return None

        return None

    def embedding_from_frame(self, frame, box, landmarks=None, raw=None, margin: float = 0.0):
        """
        Hitung embedding dari frame lengkap.
        SFace memanfaatkan alignCrop bawaan OpenCV jika raw/landmarks tersedia.
        """
        if not self.ready:
            return None

        if self.kind == "sface":
            # 1. Prioritas: alignCrop menggunakan raw 15-elemen dari YuNet
            if raw is not None:
                try:
                    face_arr = np.asarray(raw, dtype=np.float32)
                    aligned = self._sface.alignCrop(frame, face_arr)
                    return self.embedding_from_crop(aligned)
                except Exception:
                    pass

            # 2. Alternatif: rakit array 15-elemen dari box + landmarks
            if landmarks is not None and len(landmarks) == 5:
                try:
                    face_arr = np.array([
                        box[0], box[1], box[2], box[3],
                        landmarks[0][0], landmarks[0][1],
                        landmarks[1][0], landmarks[1][1],
                        landmarks[2][0], landmarks[2][1],
                        landmarks[3][0], landmarks[3][1],
                        landmarks[4][0], landmarks[4][1],
                        1.0,
                    ], dtype=np.float32)
                    aligned = self._sface.alignCrop(frame, face_arr)
                    return self.embedding_from_crop(aligned)
                except Exception:
                    pass

            # 3. Fallback: crop manual + resize
            crop = clamp_crop(frame, *box, margin=margin)
            if crop is not None:
                return self.embedding_from_crop(crop)
            return None

        # Fallback MFN path
        crop = clamp_crop(frame, *box, margin=margin)
        if crop is not None:
            return self.embedding_from_crop(crop)
        return None

    def embedding_from_image_bytes(self, image_bytes: bytes):
        """Hitung embedding dari raw image bytes (upload mobile)."""
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None, "gambar tidak bisa di-decode"
            crop = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            return self.embedding_from_crop(crop), None
        except Exception as exc:
            return None, str(exc)

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
