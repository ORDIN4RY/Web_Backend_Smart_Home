"""
core/recognizer.py — Face recognition berbasis MobileFaceNet (PyTorch).

Pipeline:
  1. Crop / alignment 5-titik landmark (standar ArcFace 112x112)
  2. Normalisasi pixel: [0,255] -> [-1,1]
  3. Inferensi MobileFaceNet -> embedding 128-D
  4. L2-normalize embedding
  5. Cosine similarity ke semua prototype (dot product karena sudah unit vector)
"""

import importlib.util
from pathlib import Path

IMG_SIZE = 112

# Koordinat 5-titik acuan kanonikal ArcFace / InsightFace (standar 112x112)
REFERENCE_PTS = (
    (38.2946, 51.6963),  # mata kanan (kiri di gambar)
    (73.5318, 51.5014),  # mata kiri (kanan di gambar)
    (56.0252, 71.7366),  # ujung hidung
    (41.5493, 92.3655),  # sudut mulut kanan
    (70.7299, 92.2041),  # sudut mulut kiri
)


# ---------------------------------------------------------------------------
# Helper: Align 5-titik + crop
# ---------------------------------------------------------------------------

def align_face_5pt(frame, landmarks, target_size: int = IMG_SIZE):
    """
    Transformasi afin 5 titik landmark YuNet ke koordinat kanonikal ArcFace.
    Memastikan mata/hidung/bibir selalu di posisi yang sama sebelum masuk MFN.
    Kembalikan crop BGR (target_size x target_size) atau None jika gagal.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if landmarks is None or len(landmarks) != 5:
        return None
    src_pts = numpy_pts(landmarks)
    ref_pts = numpy_pts(REFERENCE_PTS)
    if target_size != IMG_SIZE:
        ref_pts = ref_pts * (float(target_size) / IMG_SIZE)
    tform, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts)
    if tform is None:
        return None
    return cv2.warpAffine(frame, tform, (target_size, target_size), borderValue=0.0)


def numpy_pts(pts):
    import numpy as np
    return np.array(pts, dtype=np.float32)


def clamp_crop(frame, x, y, w, h, margin: float = 0.0):
    """
    Potong bounding box + margin, dijepit ke frame.
    Kembalikan BGR crop atau None jika terlalu kecil.
    Margin harus SAMA antara enroll dan recognition.
    """
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


# ---------------------------------------------------------------------------
# Class Recognizer
# ---------------------------------------------------------------------------

class Recognizer:
    """
    Wrapper inferensi MobileFaceNet.

    Memuat backbone via infer_mfn.py (load_backbone) secara lazy saat
    __init__ — jika torch atau file bobot belum ada, `ready` = False
    dan server tetap bisa menjawab request lain.
    """

    def __init__(
        self,
        infer_script: str = "infer_mfn.py",
        mfn_repo: str = "MobileFaceNet",
        pretrained: str = "MobileFaceNet/weights/mobilefacenet.pt",
    ):
        self.ready = False
        self.reason = ""
        self.model = None
        self._mod = None
        self._load(infer_script, mfn_repo, pretrained)

    def _load(self, infer_script: str, mfn_repo: str, pretrained: str):
        # Cek torch
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.reason = f"torch belum terpasang ({exc})"
            return

        # Load modul infer_mfn.py dinamis (agar tidak butuh restructure infer_mfn)
        spec = importlib.util.spec_from_file_location("infer_mfn", infer_script)
        if spec is None or spec.loader is None:
            self.reason = f"infer_mfn.py tidak ketemu di: {infer_script}"
            return
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:
            self.reason = f"gagal memuat infer_mfn.py ({exc})"
            return

        # Load bobot MobileFaceNet
        try:
            self.model = mod.load_backbone(mfn_repo, pretrained)
        except SystemExit as exc:
            self.reason = str(exc)
            return
        except Exception as exc:
            self.reason = f"gagal memuat bobot MFN ({exc})"
            return

        self._mod = mod
        self.ready = True
        print(f"[recognizer] MobileFaceNet dimuat dari {pretrained}")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embedding_from_crop(self, bgr_crop):
        """
        Hitung embedding 128-D dari crop BGR (sudah di-resize ke IMG_SIZE x IMG_SIZE).
        Kembalikan numpy array float32 yang sudah L2-normalized.
        """
        import numpy as np
        import torch
        import torch.nn as nn
        from PIL import Image

        rgb = bgr_crop[:, :, ::-1]
        img = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5                       # normalisasi ke [-1, 1]
        arr = np.transpose(arr, (2, 0, 1))             # HWC -> CHW
        x = torch.from_numpy(arr).unsqueeze(0)
        with torch.no_grad():
            emb = self.model(x)
            if emb.dim() > 2:
                emb = emb.flatten(1)
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb.numpy()[0]

    def embedding_from_frame(self, frame, box, landmarks=None, margin: float = 0.0):
        """
        Hitung embedding dari frame mentah (BGR) + box + opsional landmarks.
        Prioritas: 5-pt alignment (lebih akurat) > tight crop.
        Kembalikan numpy array atau None.
        """
        if landmarks is not None and len(landmarks) == 5:
            aligned = align_face_5pt(frame, landmarks, IMG_SIZE)
            if aligned is not None:
                return self.embedding_from_crop(aligned)
        crop = clamp_crop(frame, *box, margin=margin)
        if crop is not None:
            return self.embedding_from_crop(crop)
        return None

    def embedding_from_image_bytes(self, image_bytes: bytes):
        """
        Hitung embedding dari bytes gambar (JPEG/PNG dari upload mobile).
        Deteksi wajah dilakukan di sini jika dibutuhkan; atau langsung pakai full frame
        (cocok jika mobile sudah crop wajah sebelum mengirim).
        Kembalikan numpy array atau None.
        """
        try:
            import numpy as np
            import cv2
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None, "gambar tidak bisa di-decode"
            # Gunakan full frame sebagai crop; resize ke 112x112
            return self.embedding_from_crop(cv2.resize(frame, (IMG_SIZE, IMG_SIZE))), None
        except Exception as exc:
            return None, str(exc)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def best_match(embedding, prototypes: dict):
        """
        Cosine similarity (dot product) ke semua prototype.
        Kembalikan (nama, skor) terbaik, atau (None, 0.0) jika kosong.
        """
        import numpy as np

        if not prototypes:
            return None, 0.0
        names = sorted(prototypes.keys())
        mat = np.stack([prototypes[n]["embedding"] for n in names])
        sims = mat @ embedding
        best = int(np.argmax(sims))
        return names[best], float(sims[best])

    # ------------------------------------------------------------------
    # Batch enroll dari direktori (kompatibilitas infer_mfn.py)
    # ------------------------------------------------------------------

    def build_prototypes_from_dir(self, enroll_dir: str):
        """Baca subfolder per orang, hitung prototype; kembalikan dict."""
        if self._mod is None:
            raise RuntimeError("Recognizer belum siap")
        return self._mod.build_prototypes(self.model, enroll_dir, IMG_SIZE)
