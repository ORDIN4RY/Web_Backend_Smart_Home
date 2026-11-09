# -*- coding: utf-8 -*-
"""
Modul cropping wajah dengan scale factor untuk MiniFASNet.
MiniFASNet membutuhkan konteks area sekitar wajah (skala 2.7x dan/atau 4.0x)
untuk menganalisis pola spoofing seperti batas kertas, pantulan layar, atau moiré.
"""

import cv2
import numpy as np


class FaceCropper:
    @staticmethod
    def _get_expanded_box(src_w, src_h, bbox, scale):
        """
        bbox: [x, y, w, h]
        scale: faktor perbesaran box wajah (misal 2.7 atau 4.0)
        """
        x, y, box_w, box_h = bbox[:4]

        # Cegah skala melebihi batas gambar
        if box_h > 0 and box_w > 0:
            scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        else:
            scale = 1.0

        new_width = box_w * scale
        new_height = box_h * scale
        center_x = x + box_w / 2.0
        center_y = y + box_h / 2.0

        left = center_x - new_width / 2.0
        top = center_y - new_height / 2.0
        right = center_x + new_width / 2.0
        bottom = center_y + new_height / 2.0

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > src_w - 1:
            left -= (right - src_w + 1)
            right = src_w - 1
        if bottom > src_h - 1:
            top -= (bottom - src_h + 1)
            bottom = src_h - 1

        left = max(0, int(left))
        top = max(0, int(top))
        right = min(src_w - 1, int(right))
        bottom = min(src_h - 1, int(bottom))

        return left, top, right, bottom

    def crop(self, img, bbox, scale=2.7, out_w=80, out_h=80):
        """
        Memotong patch gambar di sekitar bbox wajah dengan scale tertentu dan me-resize ke (out_w, out_h).
        """
        src_h, src_w = img.shape[:2]
        left, top, right, bottom = self._get_expanded_box(src_w, src_h, bbox, scale)
        patch = img[top : bottom + 1, left : right + 1]
        if patch.size == 0:
            return cv2.resize(img, (out_w, out_h))
        return cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
