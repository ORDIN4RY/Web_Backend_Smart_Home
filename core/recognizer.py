# Shim kompatibilitas backward — gunakan core.ai.recognizer untuk kode baru
from core.ai.recognizer import Recognizer, clamp_crop, IMG_SIZE, DEFAULT_SFACE_THRESHOLD  # noqa: F401

__all__ = ["Recognizer", "clamp_crop", "IMG_SIZE", "DEFAULT_SFACE_THRESHOLD"]
