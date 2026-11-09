"""
core/grabber.py — Pembaca frame dari webcam atau stream MJPEG ESP32-CAM.

Mendukung:
  - Webcam lokal (source = 0, 1, ...)
  - Stream MJPEG dari ESP32-CAM (source = "http://<ip>:81/stream")
  - Mode nonaktif (source = None) — cocok untuk mode mobile-upload-only
Jika koneksi putus, coba sambung ulang otomatis tiap 3 detik.
"""

import threading
import time


class FrameGrabber:
    """
    Satu VideoCapture yang aman diakses dari banyak thread.

    Atribut publik:
      status : "ok" | "nonaktif" | "error" | "belum_dibuka"
      reason : str — penjelasan jika status != "ok"
    """

    def __init__(self, source):
        """
        Args:
            source: None (nonaktif) | int (indeks webcam) | str (URL MJPEG)
        """
        self.source = source
        self.cap = None
        self.lock = threading.Lock()
        self.status = "belum_dibuka"
        self.reason = ""
        self._cv2 = None
        self._next_retry = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cv2_or_none(self):
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except Exception as exc:
                self._cv2 = False
                self.reason = f"opencv belum terpasang ({exc})"
        return self._cv2 or None

    def _open(self) -> bool:
        cv2 = self._cv2_or_none()
        if cv2 is None:
            self.status = "error"
            return False
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            self.cap = cap
            self.status = "ok"
            self.reason = ""
            return True
        cap.release()
        self.status = "error"
        self.reason = f"tidak bisa membuka sumber {self.source!r}"
        return False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def read(self):
        """
        Baca satu frame BGR.
        Kembalikan np.ndarray atau None (jika nonaktif / error).
        """
        if self.source is None:
            self.status = "nonaktif"
            return None
        with self.lock:
            if self.cap is None:
                if time.time() < self._next_retry:
                    return None
                if not self._open():
                    self._next_retry = time.time() + 3.0
                    return None
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.cap.release()
                self.cap = None
                self.status = "error"
                self.reason = "frame gagal dibaca; coba sambung ulang..."
                self._next_retry = time.time() + 3.0
                return None
            self.status = "ok"
            return frame

    def release(self):
        """Tutup VideoCapture secara eksplisit (saat shutdown)."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        self.status = "nonaktif"
