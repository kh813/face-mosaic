"""
Illustration and Anime/Drawing face filter module.
Distinguishes real human faces from drawings, anime/cartoon characters,
and illustrated billboard/poster faces.
"""

from typing import Tuple, Optional
import numpy as np
import cv2

class IllustrationFilter:
    """
    Evaluates whether a detected face crop is an illustration/drawing or a real human face.
    Real human faces have:
      - Continuous natural gradient lighting and microscopic skin texture.
      - Rich color gradations across the skin (higher entropy / standard deviation).
    Illustrations / cartoon characters have:
      - Cel-shaded flat fills (uniform skin patches with near-zero local variance).
      - High edge sharpness (ink line contours) surrounding completely flat regions.
      - Highly quantized color palettes (low color entropy).
    """

    def __init__(
        self,
        flatness_threshold: float = 0.65,
        min_crop_size: int = 16
    ):
        """
        flatness_threshold: Fraction of skin pixels having near-zero local variance
                            above which the crop is classified as an illustration.
        min_crop_size: Minimum crop width/height to evaluate; smaller crops are treated as human.
        """
        self.flatness_threshold = flatness_threshold
        self.min_crop_size = min_crop_size

    def is_human_face(self, crop_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate if a cropped BGR image is a real human face (True) or an illustration (False).
        Returns:
            (is_human: bool, human_score: float)
            human_score: 1.0 (definitely real human) down to 0.0 (definitely illustration).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return True, 1.0

        h, w = crop_bgr.shape[:2]
        if min(h, w) < self.min_crop_size:
            # Resolution too low to inspect micro-texture safely -> keep safe (assume human)
            return True, 1.0

        # 1. Extract skin region in YCrCb color space
        ycbcr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        y = ycbcr[:, :, 0].astype(np.float32)
        cr = ycbcr[:, :, 1]
        cb = ycbcr[:, :, 2]

        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        skin_pixel_count = np.count_nonzero(skin_mask)

        # If very little skin tone, could be a black-and-white manga or non-skin illustration
        if skin_pixel_count < 0.10 * (h * w):
            # Evaluate overall luminance flatness across central region
            eval_mask = np.ones((h, w), dtype=bool)
        else:
            eval_mask = skin_mask

        # 2. Local standard deviation (texture variance) in 5x5 window
        kernel_size = 5
        mean = cv2.blur(y, (kernel_size, kernel_size))
        sq_mean = cv2.blur(y ** 2, (kernel_size, kernel_size))
        local_var = np.maximum(0.0, sq_mean - (mean ** 2))
        local_std = np.sqrt(local_var)

        # Pixels with local standard deviation < 3.0 are considered completely flat (solid fill)
        flat_pixels = (local_std < 3.0) & eval_mask
        eval_pixel_count = np.count_nonzero(eval_mask)
        if eval_pixel_count == 0:
            return True, 1.0

        flat_ratio = np.count_nonzero(flat_pixels) / eval_pixel_count

        # 3. Overall skin standard deviation
        overall_std = float(np.std(y[eval_mask]))

        # 4. Color palette diversity (quantization check)
        # Quantize BGR to 16 levels per channel (4096 bins)
        quantized = (crop_bgr >> 4).astype(np.uint16)
        color_keys = (quantized[:, :, 0] << 8) | (quantized[:, :, 1] << 4) | quantized[:, :, 2]
        unique_colors_in_eval = len(np.unique(color_keys[eval_mask]))
        color_diversity_ratio = unique_colors_in_eval / max(1, eval_pixel_count)

        # Decision heuristic:
        # Illustrations have high flat_ratio (> 0.65) AND low overall skin standard deviation (< 12.0)
        # Real humans have natural lighting gradients and micro-texture (flat_ratio typically 0.10 - 0.45, overall_std > 14.0)
        is_illustration = (flat_ratio >= self.flatness_threshold) and (overall_std < 14.0 or color_diversity_ratio < 0.05)

        if is_illustration:
            human_score = max(0.0, 1.0 - flat_ratio)
            return False, float(human_score)
        else:
            human_score = min(1.0, 1.0 - (flat_ratio * 0.5))
            return True, float(human_score)
