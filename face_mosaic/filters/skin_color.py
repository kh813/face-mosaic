"""
Skin color filter using YCbCr color space.
Rejects non-skin objects (bronze/stone statues, monuments, metal) while reliably
preserving East Asian, Caucasian, Black, Hispanic, South Asian human skin tones.
"""

from typing import Tuple
import numpy as np
import cv2

class SkinColorFilter:
    def __init__(self, min_skin_ratio: float = 0.15):
        """
        min_skin_ratio: Minimum fraction of pixels inside face bounding box that must fall in human skin tone gamut.
        """
        self.min_skin_ratio = min_skin_ratio

    def is_skin_tone(self, crop_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate if a cropped BGR image contains sufficient human skin tones.
        Uses standard YCbCr chrominance bounds for broad human skin tones:
        Cb in [77, 127], Cr in [133, 173]
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return True, 1.0  # Fallback to safe: keep if empty

        # Convert to YCbCr
        ycbcr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        # OpenCV cv2.COLOR_BGR2YCrCb gives channels (Y, Cr, Cb)
        y = ycbcr[:, :, 0]
        cr = ycbcr[:, :, 1]
        cb = ycbcr[:, :, 2]

        # Broad human skin tone range in Cr/Cb
        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        
        total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
        if total_pixels == 0:
            return True, 1.0

        skin_pixels = np.count_nonzero(skin_mask)
        skin_ratio = skin_pixels / total_pixels

        # If skin_ratio is above threshold, it's valid human skin
        # Default policy: prioritize preventing omission of real humans
        is_valid = bool(skin_ratio >= self.min_skin_ratio)
        return is_valid, float(skin_ratio)

