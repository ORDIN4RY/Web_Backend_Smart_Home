"""
core/ai/__init__.py — Sistem Cerdas (Deep Learning) package.

Ekspor:
  Detector           — YuNet ONNX face detector (+ Haar fallback)
  Recognizer         — SFace ONNX / MFN face embedding + Cosine matching
  AntiSpoofChecker   — MiniFASNet liveness detection
  recognize_frame_ai — pipeline AI per frame (live camera / upload)
  recognize_ai       — pipeline AI endpoint upload (/api/recognize mode="ai")
"""

from .detector import Detector
from .recognizer import Recognizer
from .anti_spoof import AntiSpoofChecker
from .pipeline import recognize_frame_ai, recognize_ai

__all__ = [
    "Detector",
    "Recognizer",
    "AntiSpoofChecker",
    "recognize_frame_ai",
    "recognize_ai",
]
