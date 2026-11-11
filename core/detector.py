"""
core/detector.py — Deteksi wajah menggunakan YuNet (ONNX via OpenCV) dengan fallback Haar.

Parameter diselaraskan dengan notebook PCV_Realtime_Wajah_Saja.ipynb:
  - YuNet: score_threshold=0.3, nms=0.3, top_k=5000
  - Filter skor akhir default 0.80
  - Filter geometri landmark (strict_geometry=True untuk live, False untuk enroll)
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Helper: validasi geometri landmark YuNet
# ---------------------------------------------------------------------------

def _landmarks_sane(x, y, w, h, landmarks) -> bool:
    """
    Periksa apakah 5 titik landmark masuk akal secara geometri:
      - Semua titik di dalam bounding box (dengan toleransi 5px)
      - Urutan vertikal: mata -> hidung -> mulut
      - Jarak antar-mata minimal 15% lebar box
    """
    (rx, ry), (lx, ly), (nx, ny), (rmx, rmy), (lmx, lmy) = landmarks
    inside = all(
        x - 5 <= px <= x + w + 5 and y - 5 <= py <= y + h + 5
        for (px, py) in landmarks
    )
    eye_y = (ry + ly) / 2
    mouth_y = (rmy + lmy) / 2
    eye_dist = abs(lx - rx)
    return inside and (eye_y < ny < mouth_y) and eye_dist >= 0.15 * w


# ---------------------------------------------------------------------------
# Class Detector
# ---------------------------------------------------------------------------

class Detector:
    """
    Wrapper deteksi wajah.

    Urutan prioritas:
      1. YuNet (membutuhkan file .onnx + opencv-contrib atau opencv>=4.10)
      2. Haar Cascade (bawaan OpenCV, fallback otomatis)

    Atribut publik:
      kind   : "yunet" | "haar" | "tidak_ada"
      reason : pesan error jika gagal inisialisasi
    """

    def __init__(self, yunet_path: str, min_score: float = 0.80, strict_geometry: bool = True):
        self.kind = "tidak_ada"
        self.reason = ""
        self.min_score = min_score
        self.strict_geometry = strict_geometry
        self.last_rejected: list = []
        self._yunet = None
        self._haar = None
        self._eye = None
        self._cv2 = None
        self._size = None
        self._init(yunet_path)

    def _init(self, yunet_path: str):
        try:
            import cv2
        except ImportError as exc:
            self.reason = f"opencv belum terpasang ({exc})"
            return
        self._cv2 = cv2

        # Coba YuNet dulu
        if yunet_path and Path(yunet_path).exists():
            try:
                if hasattr(cv2, "FaceDetectorYN_create"):
                    self._yunet = cv2.FaceDetectorYN_create(
                        str(yunet_path), "", (320, 320), 0.3, 0.3, 5000
                    )
                elif hasattr(cv2, "FaceDetectorYN"):
                    self._yunet = cv2.FaceDetectorYN.create(
                        str(yunet_path), "", (320, 320), 0.3, 0.3, 5000
                    )
                else:
                    raise AttributeError("tidak ada FaceDetectorYN_create / FaceDetectorYN")
                self.kind = "yunet"
                print(f"[detector] YuNet ONNX dimuat dari {yunet_path}", flush=True)
                return
            except Exception as exc:
                self.reason = f"yunet gagal ({exc}); pakai haar."

        # Fallback Haar
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        try:
            haar = cv2.CascadeClassifier(cascade_path)
        except Exception as exc:
            self.reason = f"haar gagal ({exc})"
            return
        if haar.empty():
            self.reason = "cascade haar kosong (mungkin opencv 5.x; pin opencv-python<5)"
            return
        self._haar = haar
        try:
            eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            self._eye = None if eye.empty() else eye
        except Exception:
            self._eye = None
        self.kind = "haar"
        print(f"[detector] menggunakan Haar Cascade (alasan: {self.reason or 'fallback'})", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame) -> list:
        """List (x, y, w, h) yang lolos filter, wajah terbesar dulu."""
        return [
            (d["x"], d["y"], d["w"], d["h"])
            for d in self.detect_detailed(frame)
            if d["accepted"]
        ]

    def detect_detailed(self, frame) -> list:
        """
        Semua kandidat (diterima + ditolak) beserta skor/landmark/alasan.
        Kembalikan list dict: {x, y, w, h, score, landmarks, accepted, reason}.
        Urutan: wajah terbesar dulu (berdasarkan area).
        """
        self.last_rejected = []
        if self.kind == "yunet":
            details = self._detect_yunet(frame)
        elif self.kind == "haar":
            details = self._detect_haar(frame)
        else:
            return []
        details.sort(key=lambda d: d["w"] * d["h"], reverse=True)
        self.last_rejected = [
            {"reason": d["reason"], "score": d["score"]}
            for d in details if not d["accepted"]
        ]
        return details

    # ------------------------------------------------------------------
    # Internal: YuNet
    # ------------------------------------------------------------------

    def _detect_yunet(self, frame) -> list:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._yunet.setInputSize((w, h))
            self._size = (w, h)
        out = self._yunet.detect(frame)
        faces = out[1] if isinstance(out, tuple) else out
        if faces is None:
            return []

        details = []
        for face in faces:
            score = float(face[14]) if len(face) > 14 else float(face[-1])
            fx, fy, fw, fh = (int(v) for v in face[:4])
            landmarks = (
                [(int(face[i]), int(face[i + 1])) for i in range(4, 14, 2)]
                if len(face) >= 14 else None
            )
            accepted, reason = True, ""
            if score < self.min_score:
                accepted, reason = False, "skor_rendah"
            elif self.strict_geometry:
                ratio = fw / max(fh, 1)
                if not 0.65 <= ratio <= 1.40:
                    accepted, reason = False, "rasio_box_aneh"
                elif landmarks is None or not _landmarks_sane(fx, fy, fw, fh, landmarks):
                    accepted, reason = False, "landmark_tidak_masuk_akal"
            details.append({
                "x": fx, "y": fy, "w": fw, "h": fh,
                "score": round(score, 3), "landmarks": landmarks,
                "accepted": accepted, "reason": reason,
                "raw": face,
            })
        return details

    # ------------------------------------------------------------------
    # Internal: Haar
    # ------------------------------------------------------------------

    def _detect_haar(self, frame) -> list:
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        found = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        details = []
        for (fx, fy, fw, fh) in found:
            fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)
            accepted, reason = True, ""
            if self.strict_geometry:
                ratio = fw / max(fh, 1)
                if not 0.65 <= ratio <= 1.40:
                    accepted, reason = False, "rasio_box_aneh"
                elif self._eye is not None:
                    upper = gray[fy: fy + int(fh * 0.6), fx: fx + fw]
                    eyes = self._eye.detectMultiScale(
                        upper, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
                    )
                    if len(eyes) < 1:
                        accepted, reason = False, "tidak_ada_mata"
            details.append({
                "x": fx, "y": fy, "w": fw, "h": fh,
                "score": None, "landmarks": None,
                "accepted": accepted, "reason": reason,
            })
        return details
