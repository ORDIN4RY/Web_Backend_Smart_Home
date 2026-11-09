"""
core/engine.py — Engine face recognition: state global + worker loop kamera.

Engine menyatukan semua komponen (Store, Detector, Recognizer, AntiSpoofChecker,
FrameGrabber) dan menyediakan satu antarmuka bersih ke layer Routes Flask.

Ekspor utama:
  Engine        — class yang di-inisialisasi sekali saat startup aplikasi
  engine        — instance global yang dipakai semua Blueprint routes
"""

import threading
import time
from pathlib import Path

from core.store import Store
from core.detector import Detector
from core.recognizer import Recognizer
from core.anti_spoof import AntiSpoofChecker
from core.grabber import FrameGrabber


def _now_iso():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


def _resize_for_detection(frame, max_dim: int = 1280):
    """
    Resize frame agar sisi terpanjang tidak melebihi max_dim.
    YuNet perlu gambar yang tidak terlalu besar (foto HP bisa 3-12 MP).
    Kembalikan frame yang sudah di-resize (atau asli jika sudah kecil).
    """
    import cv2
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """
    Satu objek Engine diinisialisasi di app.py saat startup Flask,
    kemudian disimpan di app.config["ENGINE"] dan diakses dari routes.

    Alur utama:
      - Worker thread membaca frame dari kamera secara terus-menerus.
      - Setiap beberapa frame (frame_skip), deteksi + recognition dijalankan.
      - Hasil disimpan di self._last (thread-safe).
      - Routes Flask membaca self._last melalui snapshot() / get_last().
      - Mobile app dapat juga POST gambar ke /api/recognize (tanpa kamera).
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: dict konfigurasi dari config.json (atau argparse defaults).
                 Kunci yang digunakan:
                   data_dir, yunet_model, mfn_repo, pretrained, infer_script,
                   fas_model_dir, source, frame_skip, enroll_frames
        """
        self.cfg = cfg
        self.frame_skip: int = int(cfg.get("frame_skip", 3))
        self.enroll_frames: int = int(cfg.get("enroll_frames", 5))

        # --- Komponen inti ---
        self.store = Store(cfg.get("data_dir", "api_data"))
        # Terapkan config defaults ke store jika belum ada
        self.store.config.setdefault("threshold", self.store.threshold())
        self.store.config.setdefault("fas_threshold", self.store.fas_threshold())
        self.store.config.setdefault("margin", self.store.margin())
        self.store.save_config()

        source_raw = cfg.get("source")
        if source_raw is None:
            source_raw = self.store.config.get("source", "none")
        self.source, self.source_label = _parse_source(source_raw)

        self.grabber = FrameGrabber(self.source)
        self.detector = Detector(
            cfg.get("yunet_model", "face_detection_yunet_2023mar.onnx")
        )
        self.recognizer = Recognizer(
            infer_script=cfg.get("infer_script", "infer_mfn.py"),
            mfn_repo=cfg.get("mfn_repo", "MobileFaceNet"),
            pretrained=cfg.get("pretrained", "MobileFaceNet/weights/mobilefacenet.pt"),
        )
        self.fas = AntiSpoofChecker(
            model_dir=cfg.get("fas_model_dir", "fas/models"),
            threshold=self.store.fas_threshold(),
        )

        # --- State hasil recognition terakhir ---
        self._lock = threading.Lock()
        self._last = {
            "waktu": None,
            "wajah": None,
            "skor": None,
            "granted": False,
            "jumlah_box": 0,
            "boxes": [],
            "landmarks": [],
            "liveness": None,
            "liveness_skor": None,
            "catatan": "belum ada frame diproses",
        }

        # --- Worker thread (hanya jika ada stream/webcam lokal aktif) ---
        self._stop = threading.Event()
        if self.source is not None:
            self._worker = threading.Thread(
                target=self._worker_loop, daemon=True, name="fr-worker"
            )
            self._worker.start()
        else:
            self._worker = None

    # ------------------------------------------------------------------
    # State (thread-safe)
    # ------------------------------------------------------------------

    def _set_last(self, patch: dict):
        with self._lock:
            self._last.update(patch)

    def get_last(self) -> dict:
        with self._lock:
            return dict(self._last)

    def snapshot(self) -> dict:
        """Snapshot lengkap untuk response /api/status."""
        last = self.get_last()
        faces = self.store.list_faces()
        return {
            # Waktu & kamera
            "waktu": last["waktu"],
            "kamera": self.grabber.status,
            "kamera_alasan": self.grabber.reason or None,
            # Detektor
            "detektor": self.detector.kind,
            "detektor_alasan": self.detector.reason or None,
            # Model recognition
            "model": "dimuat" if self.recognizer.ready else "belum_dimuat",
            "model_alasan": self.recognizer.reason or None,
            # Anti-spoof
            "fas_model": "dimuat" if self.fas.ready else "belum_dimuat",
            "fas_alasan": self.fas.reason or None,
            # Hasil recognition terakhir
            "wajah": last["wajah"],
            "skor": last["skor"],
            "liveness": last["liveness"],
            "liveness_skor": last["liveness_skor"],
            "granted": bool(last["granted"]),
            "jumlah_box": last["jumlah_box"],
            "boxes": last["boxes"],
            "landmarks": last["landmarks"],
            "catatan": last["catatan"],
            # Config aktif
            "threshold": self.store.threshold(),
            "fas_threshold": self.store.fas_threshold(),
            "margin": self.store.margin(),
            # Wajah terdaftar
            "wajah_terdaftar": len(faces),
        }

    # ------------------------------------------------------------------
    # Worker loop (kamera lokal)
    # ------------------------------------------------------------------

    def _worker_loop(self):
        n = 0
        while not self._stop.is_set():
            frame = self.grabber.read()
            if frame is None:
                self._stop.wait(0.5)
                continue
            n += 1
            if n % self.frame_skip != 0:
                continue
            self._process_frame(frame)

    def _process_frame(self, frame):
        """Jalankan deteksi + liveness + recognition pada satu frame."""
        try:
            details = self.detector.detect_detailed(frame)
            accepted = [d for d in details if d["accepted"]]
            boxes = [(d["x"], d["y"], d["w"], d["h"]) for d in accepted]
        except Exception as exc:
            self._set_last({"waktu": _now_iso(), "catatan": f"deteksi gagal: {exc}"})
            return

        patch = {
            "waktu": _now_iso(),
            "jumlah_box": len(boxes),
            "boxes": [list(b) for b in boxes],
            "landmarks": [d.get("landmarks") for d in accepted if d.get("landmarks")],
            "wajah": None,
            "skor": None,
            "liveness": None,
            "liveness_skor": None,
            "granted": False,
        }

        if not boxes:
            patch["catatan"] = "tidak ada wajah di frame"
            self._set_last(patch)
            return

        if not self.recognizer.ready:
            patch["catatan"] = f"wajah terdeteksi, model belum dimuat: {self.recognizer.reason}"
            self._set_last(patch)
            return

        best_d = accepted[0]
        box0 = (best_d["x"], best_d["y"], best_d["w"], best_d["h"])

        # Tahap 1: Liveness
        is_real, liveness_score = True, 1.0
        if self.fas.ready:
            is_real, liveness_score, _ = self.fas.check(frame, box0)
            is_real = bool(is_real and liveness_score >= self.store.fas_threshold())
        patch["liveness"] = is_real
        patch["liveness_skor"] = round(float(liveness_score), 3)

        if not is_real:
            patch["granted"] = False
            patch["catatan"] = f"DITOLAK: spoof terdeteksi (skor real: {liveness_score:.2f})"
            self._set_last(patch)
            return

        # Tahap 2: Recognition
        faces = self.store.prototypes()
        if not faces:
            patch["catatan"] = "wajah REAL, belum ada wajah terdaftar"
            self._set_last(patch)
            return

        try:
            emb = self.recognizer.embedding_from_frame(
                frame, box0,
                landmarks=best_d.get("landmarks"),
                margin=self.store.margin(),
            )
            if emb is None:
                patch["catatan"] = "crop terlalu kecil, dilewati"
            else:
                name, score = Recognizer.best_match(emb, faces)
                granted = score >= self.store.threshold()
                patch.update({
                    "wajah": name if granted else None,
                    "skor": round(score, 3),
                    "granted": granted,
                    "catatan": (
                        f"dikenali: {name} (REAL)"
                        if granted
                        else "wajah REAL, skor di bawah threshold (unknown)"
                    ),
                })
        except Exception as exc:
            patch["catatan"] = f"recognition gagal: {exc}"

        self._set_last(patch)

    # ------------------------------------------------------------------
    # Enroll dari kamera live
    # ------------------------------------------------------------------

    def enroll_from_camera(self, nama: str) -> tuple:
        """
        Ambil beberapa frame dari kamera, hitung prototype, simpan ke store.
        Kembalikan (http_code, payload_dict).
        """
        if not self.recognizer.ready:
            return 503, {"error": "model belum dimuat", "alasan": self.recognizer.reason}
        if self.detector.kind == "tidak_ada":
            return 503, {"error": "detektor tidak tersedia", "alasan": self.detector.reason}
        if self.grabber.status == "nonaktif":
            return 503, {"error": "kamera nonaktif; gunakan POST /api/enroll/photo"}

        import numpy as np
        embs = []
        deadline = time.time() + 10.0
        while len(embs) < self.enroll_frames and time.time() < deadline:
            frame = self.grabber.read()
            if frame is None:
                time.sleep(0.2)
                continue
            try:
                details = self.detector.detect_detailed(frame)
                accepted = [d for d in details if d["accepted"]]
            except Exception:
                accepted = []
            if not accepted:
                continue
            best_d = accepted[0]
            box0 = (best_d["x"], best_d["y"], best_d["w"], best_d["h"])
            # Anti-spoof saat enroll
            if self.fas.ready:
                is_real, _, _ = self.fas.check(frame, box0)
                if not is_real:
                    time.sleep(0.1)
                    continue
            emb = self.recognizer.embedding_from_frame(
                frame, box0,
                landmarks=best_d.get("landmarks"),
                margin=self.store.margin(),
            )
            if emb is not None:
                embs.append(emb)

        if len(embs) < 1:
            return 408, {"error": "timeout: tidak ada wajah REAL terdeteksi dalam 10 detik"}

        prototype = np.mean(embs, axis=0)
        self.store.set_face(nama, prototype, len(embs))
        return 200, {
            "nama": nama,
            "jumlah_frame": len(embs),
            "pesan": f"Wajah '{nama}' berhasil didaftarkan ({len(embs)} frame)",
        }

    # ------------------------------------------------------------------
    # Enroll dari foto upload (mobile)
    # ------------------------------------------------------------------

    def enroll_from_photo(self, nama: str, image_bytes: bytes) -> tuple:
        """
        Daftarkan wajah dari foto yang diupload oleh aplikasi mobile.
        Kembalikan (http_code, payload_dict).
        """
        if not self.recognizer.ready:
            return 503, {"error": "model belum dimuat", "alasan": self.recognizer.reason}
        try:
            import numpy as np
            import cv2
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return 400, {"error": "gambar tidak bisa di-decode"}

            # Resize ke max 1280px agar YuNet bisa deteksi wajah di foto HP resolusi tinggi
            frame = _resize_for_detection(frame, max_dim=1280)

            # Deteksi wajah di foto
            details = self.detector.detect_detailed(frame)
            # Enroll pakai semua detektor (strict_geometry=False lebih cocok untuk foto)
            accepted = [d for d in details if d["accepted"]]
            if not accepted:
                # Coba tanpa filter geometri: gunakan box penuh
                h, w = frame.shape[:2]
                emb = self.recognizer.embedding_from_frame(frame, (0, 0, w, h))
                if emb is None:
                    return 400, {"error": "tidak ada wajah terdeteksi di foto"}
                self.store.set_face(nama, emb, 1)
                return 200, {
                    "nama": nama,
                    "jumlah_frame": 1,
                    "pesan": f"Wajah '{nama}' didaftarkan dari foto (full frame fallback)",
                }

            best_d = accepted[0]
            box0 = (best_d["x"], best_d["y"], best_d["w"], best_d["h"])
            emb = self.recognizer.embedding_from_frame(
                frame, box0,
                landmarks=best_d.get("landmarks"),
                margin=self.store.margin(),
            )
            if emb is None:
                return 400, {"error": "crop wajah terlalu kecil"}

            self.store.set_face(nama, emb, 1)
            return 200, {
                "nama": nama,
                "jumlah_frame": 1,
                "pesan": f"Wajah '{nama}' berhasil didaftarkan dari foto",
            }
        except Exception as exc:
            return 500, {"error": f"enroll_from_photo gagal: {exc}"}

    # ------------------------------------------------------------------
    # Recognize dari foto upload (mobile)
    # ------------------------------------------------------------------

    def recognize_photo(self, image_bytes: bytes) -> tuple:
        """
        Kenali wajah dari foto yang diupload oleh aplikasi mobile.
        Kembalikan (http_code, payload_dict).
        """
        if not self.recognizer.ready:
            return 503, {"error": "model belum dimuat", "alasan": self.recognizer.reason}
        try:
            import numpy as np
            import cv2
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return 400, {"error": "gambar tidak bisa di-decode"}

            # Resize ke max 1280px agar YuNet bisa deteksi wajah di foto HP resolusi tinggi
            frame = _resize_for_detection(frame, max_dim=1280)

            details = self.detector.detect_detailed(frame)
            accepted = [d for d in details if d["accepted"]]

            if not accepted:
                return 404, {"error": "tidak ada wajah terdeteksi di foto"}

            best_d = accepted[0]
            box0 = (best_d["x"], best_d["y"], best_d["w"], best_d["h"])
            bbox_info = {
                "x": int(best_d["x"]),
                "y": int(best_d["y"]),
                "w": int(best_d["w"]),
                "h": int(best_d["h"]),
                "frame_w": int(frame.shape[1]),
                "frame_h": int(frame.shape[0]),
            }

            # Liveness
            is_real, liveness_score = True, 1.0
            if self.fas.ready:
                is_real, liveness_score, _ = self.fas.check(frame, box0)
                is_real = bool(is_real and liveness_score >= self.store.fas_threshold())

            if not is_real:
                return 403, {
                    "granted": False,
                    "wajah": None,
                    "liveness": False,
                    "liveness_skor": round(float(liveness_score), 3),
                    "catatan": "DITOLAK: spoof / wajah palsu terdeteksi",
                    "bbox": bbox_info,
                }

            faces = self.store.prototypes()
            if not faces:
                return 200, {
                    "granted": False,
                    "wajah": None,
                    "liveness": True,
                    "liveness_skor": round(float(liveness_score), 3),
                    "catatan": "wajah REAL, belum ada wajah terdaftar",
                    "bbox": bbox_info,
                }

            emb = self.recognizer.embedding_from_frame(
                frame, box0,
                landmarks=best_d.get("landmarks"),
                margin=self.store.margin(),
            )
            if emb is None:
                return 400, {"error": "crop wajah terlalu kecil"}

            name, score = Recognizer.best_match(emb, faces)
            granted = score >= self.store.threshold()
            return 200, {
                "granted": granted,
                "wajah": name if granted else None,
                "skor": round(score, 3),
                "threshold": self.store.threshold(),
                "liveness": True,
                "liveness_skor": round(float(liveness_score), 3),
                "catatan": (
                    f"dikenali: {name}"
                    if granted
                    else "wajah REAL, skor di bawah threshold (unknown)"
                ),
                "bbox": bbox_info,
            }
        except Exception as exc:
            return 500, {"error": f"recognize_photo gagal: {exc}"}

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Hentikan worker thread dan lepas kamera."""
        self._stop.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.grabber.release()


# ---------------------------------------------------------------------------
# Helper: parse source
# ---------------------------------------------------------------------------

def _parse_source(raw) -> tuple:
    """
    'none' -> (None, 'none')
    '0'    -> (0, '0')      (webcam indeks)
    'http://...' -> (url, url)
    """
    if raw is None:
        return None, "none"
    text = str(raw).strip()
    if text.lower() == "none" or text == "":
        return None, "none"
    if text.isdigit():
        return int(text), text
    return text, text
