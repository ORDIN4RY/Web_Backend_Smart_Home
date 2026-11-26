"""
core/cv/__init__.py — Computer Vision Klasik package (2 file: lbp_liveness & lbph).

Ekspor:
  check_liveness         — Color-LBP 1-NN Chi-Square liveness detection
  COLOR_LBP_REAL         — galeri exemplar real (N × 1024)
  COLOR_LBP_SPOOF        — galeri exemplar spoof (M × 1024)
  COLOR_LBP_THRESH       — threshold default Color-LBP
  extract_lbp            — LBP spasial 8×8 → 16.384-D
  recognize_cv           — pipeline pengenalan wajah CV klasik lengkap
  LBPH_THRESHOLD_DEFAULT — 23.0
"""

from .lbp_liveness import check_liveness, COLOR_LBP_REAL, COLOR_LBP_SPOOF, COLOR_LBP_THRESH
from .lbph import extract_lbp, recognize_cv, LBPH_THRESHOLD_DEFAULT

__all__ = [
    "check_liveness",
    "COLOR_LBP_REAL",
    "COLOR_LBP_SPOOF",
    "COLOR_LBP_THRESH",
    "extract_lbp",
    "recognize_cv",
    "LBPH_THRESHOLD_DEFAULT",
]
