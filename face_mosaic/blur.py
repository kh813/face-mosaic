"""
Face blur and mosaic rendering module for face-mosaic.
Implements Gaussian blur and Mosaic pixelation with margin expansion.
"""

from typing import List, Tuple
import numpy as np
import cv2

class FaceBlurrer:
    def __init__(
        self,
        blur_type: str = "gaussian",
        strength: int = 51,
        mosaic_block_size: int = 28,
        margin_x: float = 0.50,
        margin_y: float = 0.50
    ):
        self.blur_type = blur_type.lower()
        self.strength = strength if strength % 2 == 1 else strength + 1  # ensure odd
        self.mosaic_block_size = max(2, mosaic_block_size)
        self.margin_x = max(0.0, margin_x)
        self.margin_y = max(0.0, margin_y)

    def expand_bbox(self, bbox: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """
        Expand bounding box [x1, y1, x2, y2] by margin ratios and clip to image boundaries.
        """
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1

        dx = w * self.margin_x
        dy = h * self.margin_y

        ex1 = int(max(0, np.floor(x1 - dx)))
        ey1 = int(max(0, np.floor(y1 - dy)))
        ex2 = int(min(img_w, np.ceil(x2 + dx)))
        ey2 = int(min(img_h, np.ceil(y2 + dy)))

        return ex1, ey1, ex2, ey2

    def apply_mosaic_to_roi(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h <= 0 or w <= 0:
            return roi
        
        # Downscale then upscale with nearest neighbor
        small_w = max(1, w // self.mosaic_block_size)
        small_h = max(1, h // self.mosaic_block_size)
        
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return mosaic

    def apply_gaussian_to_roi(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h <= 0 or w <= 0:
            return roi
        
        ksize = min(self.strength, min(h, w))
        if ksize % 2 == 0:
            ksize -= 1
        if ksize < 3:
            ksize = 3
            
        return cv2.GaussianBlur(roi, (ksize, ksize), 0)

    def apply_blur(self, frame: np.ndarray, bboxes: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply blur/mosaic to frame at specified bboxes.
        Returns:
            (processed_frame, mask)
            where mask is a uint8 binary image (255 where blurred, 0 elsewhere).
        """
        out_frame = frame.copy()
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for bbox in bboxes:
            x1, y1, x2, y2 = self.expand_bbox(bbox, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = out_frame[y1:y2, x1:x2]
            if self.blur_type == "mosaic":
                blurred_roi = self.apply_mosaic_to_roi(roi)
            else:
                blurred_roi = self.apply_gaussian_to_roi(roi)

            out_frame[y1:y2, x1:x2] = blurred_roi
            mask[y1:y2, x1:x2] = 255

        return out_frame, mask
