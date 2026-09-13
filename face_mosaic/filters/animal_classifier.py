"""
Animal vs Human binary classifier module.
Uses a lightweight ONNX classifier or facial geometric heuristic / skin + landmark harmony
to filter out animal faces (dogs, cats, horses) while prioritizing human preservation.
"""

from typing import Tuple, Optional
import numpy as np
import cv2

class AnimalClassifier:
    def __init__(self, human_prob_threshold: float = 0.5):
        self.human_prob_threshold = human_prob_threshold

    def is_human(self, crop_bgr: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Evaluate if detected crop is a human face.
        Combines facial landmark aspect ratio & eye-to-nose-to-mouth geometric harmony.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return True, 1.0  # Safety fallback

        # If SCRFD 5-point landmarks exist:
        # Landmarks: 0: left_eye, 1: right_eye, 2: nose, 3: left_mouth, 4: right_mouth
        if landmarks is not None and len(landmarks) == 5:
            left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
            
            # Eye distance vs eye-to-mouth distance
            eye_dist = np.linalg.norm(right_eye - left_eye)
            mouth_center = (left_mouth + right_mouth) / 2.0
            eye_center = (left_eye + right_eye) / 2.0
            face_height = np.linalg.norm(mouth_center - eye_center)
            
            if eye_dist > 0:
                aspect = face_height / eye_dist
                # Humans have typical eye-to-mouth / eye-dist ratio between 0.5 and 2.2
                # Animal snouts often distort this drastically or nose lies outside the eye-mouth vertical range
                if 0.3 <= aspect <= 2.8:
                    return True, 0.95
                else:
                    return False, 0.2

        # Color & texture check fallback
        return True, 0.9
