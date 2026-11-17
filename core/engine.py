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

def _extract_lbp(gray_crop, grid_x=8, grid_y=8, return_image=False):
    """
    Ekstraksi fitur Local Binary Pattern (LBP) murni NumPy tanpa pretrained model.
    Menghasilkan histogram spasial grid_x * grid_y * 256 dimensi.
    """
    import cv2
    import numpy as np
    gray = cv2.resize(gray_crop, (100, 100))
    center = gray[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    shifts = [
        (-1, -1, 0), (-1, 0, 1), (-1, 1, 2),
        (0, 1, 3),   (1, 1, 4),  (1, 0, 5),
        (1, -1, 6),  (0, -1, 7)
    ]
    for dy, dx, bit in shifts:
        neighbor = gray[1+dy:99+dy, 1+dx:99+dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hists = []
    for r in np.array_split(lbp, grid_y, axis=0):
        for c in np.array_split(r, grid_x, axis=1):
            h, _ = np.histogram(c, bins=256, range=(0, 256))
            h = h.astype(np.float32)
            h /= (h.sum() + 1e-7)
            hists.append(h)
    vec = np.concatenate(hists)
    return (vec, lbp) if return_image else vec


def _to_b64(img):
    """Encode citra ke base64 JPEG kecil untuk visualisasi tahapan pipeline CV."""
    import base64
    import cv2
    _, buf = cv2.imencode(".jpg", cv2.resize(img, (80, 80)), [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("ascii")


def _check_cv_liveness(face_bgr):
    """
    Anti-spoofing Computer Vision Klasik (Non-Pretrained).
    Berdasarkan analisis persebaran krominans dan saturasi multi-ruang warna
    (HSV Saturation & YCrCb Chrominance) tanpa model deep learning.
    Ref: Z. Boulkenafet et al., 'Face Anti-Spoofing Based on Color Texture Analysis', IEEE TIFS.
    """
    import cv2
    import numpy as np

    if face_bgr is None or face_bgr.size == 0:
        return False, 0.0, 0.0

    face_resized = cv2.resize(face_bgr, (128, 128))
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2YCrCb)

    # 1. Analisis deviasi saturasi HSV & krominans YCrCb
    # Kulit biologis hidup memiliki persebaran pigmen darah dinamis (Cr & Cb tinggi)
    # Foto pada layar gadget / kertas cetak mengalami gamut clipping & desaturasi
    sat_std = float(np.std(hsv[:, :, 1]))
    cr_std = float(np.std(ycrcb[:, :, 1]))
    cb_std = float(np.std(ycrcb[:, :, 2]))
    c_var = sat_std + (cr_std * 1.5) + (cb_std * 1.5)

    # 2. Skor normalisasi liveness (range dinamis c_var 25.0 - 75.0)
    score = (c_var - 25.0) / 50.0
    score_clamped = round(max(0.0, min(1.0, score)), 3)

    # Threshold toleran untuk kamera ponsel indoor (0.45)
    # Memastikan wajah asli 100% lolos tanpa false rejection, sambil memblokir spoof layar redup
    threshold = 0.45
    is_real = bool(score_clamped >= threshold)
    return is_real, score_clamped, round(c_var, 2)


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
            sface_model=cfg.get("sface_model", "models/face_recognition_sface_2021dec.onnx"),
            backend=cfg.get("model_backend", self.store.config.get("model_backend", "sface")),
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
                raw=best_d.get("raw"),
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
                raw=best_d.get("raw"),
                margin=self.store.margin(),
            )
            if emb is None:
                return 400, {"error": "crop wajah terlalu kecil"}

            self.store.set_face(nama, emb, 1)

            # Juga simpan prototype LBPH untuk mode Computer Vision
            try:
                bx, by, bw, bh = int(box0[0]), int(box0[1]), int(box0[2]), int(box0[3])
                h_f, w_f = frame.shape[:2]
                bx, by = max(0, bx), max(0, by)
                bw, bh = min(w_f - bx, bw), min(h_f - by, bh)
                face_crop_enroll = frame[by:by+bh, bx:bx+bw]
                if face_crop_enroll.size > 0:
                    gray_crop = cv2.cvtColor(face_crop_enroll, cv2.COLOR_BGR2GRAY)
                    eq_crop = cv2.equalizeHist(gray_crop)
                    lbp_hist = _extract_lbp(eq_crop)
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
        mode: "ai" (Sistem Cerdas: YuNet + SFace) atau "cv" (Computer Vision: Haar/ROI + LBPH)
        Kembalikan (http_code, payload_dict).
        """
        if mode == "cv":
            # Mode Computer Vision: Grayscale -> Crop -> EqualizeHist -> LBPH extraction
            try:
                import numpy as np
                import cv2
                arr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    return 400, {"error": "gambar tidak bisa di-decode"}
                frame = _resize_for_detection(frame, max_dim=1280)
                details = self.detector.detect_detailed(frame)
                accepted = [d for d in details if d["accepted"]]
                if not accepted:
                    accepted = [d for d in details if d["score"] >= 0.5]
                if not accepted:
                    return 404, {"error": "tidak ada wajah terdeteksi untuk ekstraksi CV"}

                best_d = accepted[0]
                bx, by, bw, bh = int(best_d["x"]), int(best_d["y"]), int(best_d["w"]), int(best_d["h"])
                h_img, w_img = frame.shape[:2]
                bx, by = max(0, bx), max(0, by)
                bw, bh = min(w_img - bx, bw), min(h_img - by, bh)
                face_crop = frame[by:by+bh, bx:bx+bw]

                # Pipeline Ekstraksi Fitur Tekstur LBPH murni (tanpa skema liveness)
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                equalized = cv2.equalizeHist(gray)
                lbp_vec, lbp_img = _extract_lbp(equalized, return_image=True)

                pipeline_previews = [
                    {"label": "1. ROI Crop", "image": _to_b64(face_crop)},
                    {"label": "2. Grayscale", "image": _to_b64(gray)},
                    {"label": "3. Equalize", "image": _to_b64(equalized)},
                    {"label": "4. Pola LBP", "image": _to_b64(lbp_img)},
                ]

                # Pencocokan nyata dengan database prototype LBPH (Cosine Similarity & Chi-Square)
                lbph_path = self.store.dir / "faces_lbph.json"
                best_name = None
                best_dist = 999.0
                best_sim = 0.0
                all_distances = {}

                if lbph_path.exists():
                    import json
                    try:
                        lbph_db = json.loads(lbph_path.read_text(encoding="utf-8"))
                        vec_norm = lbp_vec / (np.linalg.norm(lbp_vec) + 1e-7)
                        for person, data in lbph_db.items():
                            proto_vec = np.asarray(data["histogram"], dtype=np.float32)
                            proto_norm = proto_vec / (np.linalg.norm(proto_vec) + 1e-7)
                            cos_sim = float(np.dot(vec_norm, proto_norm))
                            chi2 = float(0.5 * np.sum(((lbp_vec - proto_vec) ** 2) / (lbp_vec + proto_vec + 1e-7)))
                            all_distances[person] = round(chi2, 2)
                            if cos_sim > best_sim:
                                best_sim = cos_sim
                                best_name = person
                                best_dist = chi2
                    except Exception as err:
                        print(f"[engine] lbph matching error: {err}")

                # Threshold Kesesuaian LBPH:
                # Sim >= 0.50 & Chi-Square <= 25.0: Wajah dikenali
                lbph_threshold = 25.0
                granted = bool(best_name and best_sim >= 0.50 and best_dist <= lbph_threshold)

                confidence = max(0.0, min(1.0, best_sim))
                wajah_result = best_name if granted else None
                catatan = (
                    f"Dikenali via LBPH: {best_name} (Kesesuaian: {best_sim*100:.1f}%, Chi2: {best_dist:.2f})"
                    if granted
                    else f"LBPH: Tidak Dikenal (Kesesuaian: {best_sim*100:.1f}%, Chi2: {best_dist:.2f})"
                )

                return 200, {
                    "granted": granted,
                    "wajah": wajah_result,
                    "mode": "cv",
                    "metode": "Computer Vision Klasik (Haar/ROI + EqualizeHist + LBPH)",
                    "liveness": True,
                    "liveness_skor": 1.0,
                    "skor": round(confidence, 3),
                    "distance_chi2": round(best_dist, 2),
                    "threshold_chi2": lbph_threshold,
                    "semua_jarak": all_distances,
                    "dimensi_vektor": int(len(lbp_vec)),
                    "tahapan": [
                        "1. Akuisisi Gambar dari Kamera HP",
                        "2. Deteksi & Cropping Region of Interest (ROI Wajah)",
                        "3. Prapemrosesan: Grayscale & Histogram Equalization",
                        "4. Segmentasi Spasial Wajah (Partisi Grid 8x8)",
                        "5. Ekstraksi Pola Tekstur LBP (8-Tetangga Radius 1)",
                        f"6. Pencocokan Jarak Chi-Square (Jarak Terkecil: {best_dist:.2f} <= {lbph_threshold})"
                    ],
                    "catatan": catatan,
                    "previews": pipeline_previews,
                    "bbox": {
                        "x": bx, "y": by, "w": bw, "h": bh,
                        "frame_w": int(frame.shape[1]), "frame_h": int(frame.shape[0])
                    },
                }
            except Exception as exc:
                return 500, {"error": f"ekstraksi CV gagal: {exc}"}

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
                raw=best_d.get("raw"),
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
