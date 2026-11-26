# Shim kompatibilitas backward — gunakan core.ai.anti_spoof untuk kode baru
from core.ai.anti_spoof import AntiSpoofChecker  # noqa: F401

__all__ = ["AntiSpoofChecker"]
