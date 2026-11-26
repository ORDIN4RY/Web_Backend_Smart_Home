"""
core/engine.py — Lean orchestrator: state global + worker loop kamera.

Engine menyatukan semua komponen (Store, Detector, Recognizer, AntiSpoofChecker,
FrameGrabber) dan menyediakan satu antarmuka bersih ke layer Routes Flask.

Implementasi detail:
  - Mode "cv"  → didelegasikan ke core.cv.pipeline.recognize_cv
  - Mode "ai"  → didelegasikan ke core.ai.pipeline.recognize_ai
  - Worker loop kamera tetap di sini (state global thread-safe)

Ekspor utama:
  Engine        — class yang di-inisialisasi sekali saat startup aplikasi
  engine        — instance global yang dipakai semua Blueprint routes
"""

import threading
import time
from pathlib import Path

from core.store import Store
from core.grabber import FrameGrabber

# Sistem Cerdas (Deep Learning)
from core.ai import Detector, Recognizer, AntiSpoofChecker, recognize_frame_ai, recognize_ai

# Computer Vision Klasik
from core.cv.lbp_liveness import COLOR_LBP_REAL, COLOR_LBP_SPOOF, COLOR_LBP_THRESH
from core.cv.lbph import extract_lbp, recognize_cv


def _now_iso():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


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

        # Detektor YuNet (Sistem Cerdas — dipakai juga oleh CV pipeline sebagai fallback)
        self.detector = Detector(
            cfg.get("yunet_model", "face_detection_yunet_2023mar.onnx")
        )

        # Recognizer SFace/MFN (Sistem Cerdas)
        self.recognizer = Recognizer(
            sface_model=cfg.get("sface_model", "models/face_recognition_sface_2021dec.onnx"),
            backend=cfg.get("model_backend", self.store.config.get("model_backend", "sface")),
            infer_script=cfg.get("infer_script", "infer_mfn.py"),
            mfn_repo=cfg.get("mfn_repo", "MobileFaceNet"),
            pretrained=cfg.get("pretrained", "MobileFaceNet/weights/mobilefacenet.pt"),
        )

        # Anti-spoof MiniFASNet (Sistem Cerdas)
        self.fas = AntiSpoofChecker(
            model_dir=cfg.get("fas_model_dir", "fas/models"),
            threshold=self.store.fas_threshold(),
        )

        # Galeri Color-LBP (Computer Vision Klasik) — dimuat saat import core.cv.lbp_liveness
        self._clbp_real = COLOR_LBP_REAL
        self._clbp_spoof = COLOR_LBP_SPOOF
        self._clbp_thresh = COLOR_LBP_THRESH

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
            # Model recognition (Sistem Cerdas)
            "model": "dimuat" if self.recognizer.ready else "belum_dimuat",
            "model_alasan": self.recognizer.reason or None,
            # Anti-spoof (Sistem Cerdas)
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
        """Jalankan deteksi + liveness + recognition AI pada satu frame kamera."""
        try:
            res = recognize_frame_ai(
                frame,
                detector=self.detector,
                recognizer=self.recognizer,
                fas=self.fas,
                store=self.store,
                real_mat=self._clbp_real,
                spoof_mat=self._clbp_spoof,
                lbp_thresh=self._clbp_thresh,
            )
        except Exception as exc:
            self._set_last({"waktu": _now_iso(), "catatan": f"deteksi gagal: {exc}"})
            return

        accepted = res["accepted"]
        boxes = [[d["x"], d["y"], d["w"], d["h"]] for d in accepted]
        catatan = res["catatan"]
        if res["status"] == "ok" and res["granted"]:
            catatan = f"{catatan} (REAL)"

        self._set_last({
            "waktu": _now_iso(),
            "jumlah_box": len(boxes),
            "boxes": boxes,
            "landmarks": [d.get("landmarks") for d in accepted if d.get("landmarks")],
            "wajah": res["wajah"],
            "skor": res["skor"],
            "liveness": res["liveness"],
            "liveness_skor": res["liveness_skor"],
            "granted": res["granted"],
            "catatan": catatan,
        })

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
                raw=best_d.get("raw"),
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
            h_f, w_f = frame.shape[:2]
            if max(h_f, w_f) > 1280:
                scale = 1280 / max(h_f, w_f)
                frame = cv2.resize(frame, (int(w_f * scale), int(h_f * scale)), interpolation=cv2.INTER_AREA)

            # Deteksi wajah di foto (Sistem Cerdas: YuNet)
            details = self.detector.detect_detailed(frame)
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
                raw=best_d.get("raw"),
                margin=self.store.margin(),
            )
            if emb is None:
                return 400, {"error": "crop wajah terlalu kecil"}

            # Simpan prototype AI (Sistem Cerdas: SFace embedding)
            self.store.set_face(nama, emb, 1)

            # Simpan prototype LBPH untuk mode Computer Vision Klasik
            try:
                bx, by, bw, bh = int(box0[0]), int(box0[1]), int(box0[2]), int(box0[3])
                h_f, w_f = frame.shape[:2]
                bx, by = max(0, bx), max(0, by)
                bw, bh = min(w_f - bx, bw), min(h_f - by, bh)
                face_crop_enroll = frame[by:by + bh, bx:bx + bw]
                if face_crop_enroll.size > 0:
                    gray_crop = cv2.resize(
                        cv2.cvtColor(face_crop_enroll, cv2.COLOR_BGR2GRAY), (100, 100)
                    )
                    eq_crop = cv2.equalizeHist(gray_crop)
                    lbp_hist = extract_lbp(eq_crop)
                    self.store.set_face_lbph(nama, lbp_hist)
            except Exception as e_lbp:
                print(f"[engine] enroll lbph warning: {e_lbp}")

            return 200, {
                "nama": nama,
                "jumlah_frame": 1,
                "pesan": f"Wajah '{nama}' berhasil didaftarkan dari foto (AI + CV prototype)",
            }
        except Exception as exc:
            return 500, {"error": f"enroll_from_photo gagal: {exc}"}

    # ------------------------------------------------------------------
    # Recognize dari foto upload (mobile)
    # ------------------------------------------------------------------

    def recognize_photo(self, image_bytes: bytes, mode: str = "ai") -> tuple:
        """
        Kenali wajah dari foto yang diupload oleh aplikasi mobile.
        mode: "ai" (Sistem Cerdas) atau "cv" (Computer Vision Klasik)
        Didelegasikan ke core.cv.pipeline atau core.ai.pipeline.
        """
        if mode == "cv":
            lbph_db = self.store.lbph_prototypes()
            return recognize_cv(
                image_bytes,
                lbph_db=lbph_db,
                real_mat=self._clbp_real,
                spoof_mat=self._clbp_spoof,
                lbp_thresh=self._clbp_thresh,
            )

        # mode == "ai" (default)
        return recognize_ai(
            image_bytes,
            detector=self.detector,
            recognizer=self.recognizer,
            fas=self.fas,
            store=self.store,
            real_mat=self._clbp_real,
            spoof_mat=self._clbp_spoof,
            lbp_thresh=self._clbp_thresh,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Hentikan worker thread dan lepas kamera."""
        self._stop.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.grabber.release()
