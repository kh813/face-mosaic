"""
Static photo / poster / billboard filter module.
Optional filter (default OFF) that identifies faces on static posters or advertisements
when camera moves relative to static background.
"""

from typing import List, Tuple
import numpy as np
import cv2

class StaticPhotoFilter:
    def __init__(
        self,
        min_static_frames: int = 30,
        motion_similarity_threshold: float = 0.95,
        texture_diff_threshold: float = 0.05
    ):
        self.min_static_frames = min_static_frames
        self.motion_similarity_threshold = motion_similarity_threshold
        self.texture_diff_threshold = texture_diff_threshold

    def is_static_advertisement(self, track_crops: List[np.ndarray]) -> bool:
        """
        Check if the sequence of face crops across the track has near-zero internal variation.
        """
        if len(track_crops) < self.min_static_frames:
            return False  # Not enough history to judge safely

        # Compare consecutive crops
        diffs = []
        for i in range(1, len(track_crops)):
            prev = track_crops[i - 1]
            curr = track_crops[i]
            if prev.shape != curr.shape:
                curr_resized = cv2.resize(curr, (prev.shape[1], prev.shape[0]))
            else:
                curr_resized = curr

            diff = np.mean(np.abs(prev.astype(np.float32) - curr_resized.astype(np.float32))) / 255.0
            diffs.append(diff)

        avg_diff = float(np.mean(diffs))
        
        # If internal appearance variation is almost zero over a long duration
        if avg_diff < self.texture_diff_threshold:
            return True  # Static ad
            
        return False
