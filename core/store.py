"""
core/store.py — Penyimpanan prototype embedding (faces.json) dan konfigurasi (config.json).

Thread-safe menggunakan threading.Lock.
"""

import json
import threading
from pathlib import Path

DEFAULT_THRESHOLD = 0.5
DEFAULT_FAS_THRESHOLD = 0.60
DEFAULT_MARGIN = 0.0


def _now_iso():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


class Store:
    """
    Mengelola dua file di data_dir:
      - faces.json  : prototype embedding per orang
      - config.json : threshold, margin, path model, dsb.
    Semua operasi baca/tulis dilindungi lock agar aman dari banyak thread.
    """

    def __init__(self, data_dir: str = "api_data"):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.faces_path = self.dir / "faces.json"
        self.lock = threading.Lock()
        # faces: nama -> {embedding: np.ndarray, jumlah_frame: int, diperbarui: str}
        self.faces: dict = {}
        self.config: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Load & Save
    # ------------------------------------------------------------------

    def _load(self):
        """Muat config dan faces dari disk saat startup."""
        # Config
        if self.config_path.exists():
            try:
                self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[store] config rusak, pakai default ({exc})")
                self.config = {}

        # Faces / Prototypes
        with self.lock:
            self.faces = {}
            if self.faces_path.exists():
                try:
                    import numpy as np
                    raw = json.loads(self.faces_path.read_text(encoding="utf-8"))
                    for nama, rec in raw.items():
                        self.faces[nama] = {
                            "embedding": np.asarray(rec["embedding"], dtype=np.float32),
                            "jumlah_frame": int(rec.get("jumlah_frame", 0)),
                            "diperbarui": rec.get("diperbarui", ""),
                        }
                    print(f"[store] {len(self.faces)} wajah dimuat dari {self.faces_path.name}")
                except Exception as exc:
                    print(f"[store] {self.faces_path.name} gagal dimuat ({exc}); mulai kosong.")
                    self.faces = {}

    def save_config(self):
        """Tulis config ke disk."""
        with self.lock:
            self.config_path.write_text(
                json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def save_faces(self):
        """Tulis semua prototype ke disk."""
        with self.lock:
            raw = {
                nama: {
                    "embedding": rec["embedding"].tolist(),
                    "jumlah_frame": rec["jumlah_frame"],
                    "diperbarui": rec["diperbarui"],
                }
                for nama, rec in self.faces.items()
            }
            self.faces_path.write_text(
                json.dumps(raw, ensure_ascii=False), encoding="utf-8"
            )

    # ------------------------------------------------------------------
    # Faces CRUD
    # ------------------------------------------------------------------

    def set_face(self, nama: str, embedding, jumlah_frame: int):
        """
        Simpan (atau timpa) prototype wajah.
        Embedding WAJIB dinormalisasi L2 agar dot-product == cosine-similarity.
        """
        import numpy as np

        vec = np.asarray(embedding, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        with self.lock:
            self.faces[nama] = {
                "embedding": vec,
                "jumlah_frame": jumlah_frame,
                "diperbarui": _now_iso(),
            }
        self.save_faces()

    def remove_face(self, nama: str) -> bool:
        """Hapus wajah; kembalikan True jika sebelumnya ada."""
        with self.lock:
            existed = nama in self.faces
            self.faces.pop(nama, None)
        if existed:
            self.save_faces()
        return existed

    def list_faces(self) -> list:
        """List semua wajah (tanpa embedding raw)."""
        with self.lock:
            return [
                {
                    "nama": nama,
                    "jumlah_frame": rec["jumlah_frame"],
                    "diperbarui": rec["diperbarui"],
                }
                for nama, rec in sorted(self.faces.items())
            ]

    def prototypes(self) -> dict:
        """Snapshot nama -> record (termasuk embedding) untuk matching."""
        with self.lock:
            return dict(self.faces)

    def has_face(self, nama: str) -> bool:
        with self.lock:
            return nama in self.faces

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def threshold(self) -> float:
        return float(self.config.get("threshold", DEFAULT_THRESHOLD))

    def fas_threshold(self) -> float:
        return float(self.config.get("fas_threshold", DEFAULT_FAS_THRESHOLD))

    def margin(self) -> float:
        return float(self.config.get("margin", DEFAULT_MARGIN))
