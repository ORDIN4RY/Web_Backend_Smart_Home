# Shim kompatibilitas backward — gunakan core.ai.detector untuk kode baru
from core.ai.detector import Detector, _landmarks_sane  # noqa: F401

__all__ = ["Detector", "_landmarks_sane"]
