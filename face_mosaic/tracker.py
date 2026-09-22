"""
Tracklet association, temporal smoothing, track-level filtering,
and bidirectional frame padding (forward/backward blur extension).
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from .detector import FaceDetection
from .filters.skin_color import SkinColorFilter
from .filters.animal_classifier import AnimalClassifier
from .filters.static_photo import StaticPhotoFilter
from .config import FiltersConfig

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

class Tracklet:
    def __init__(self, track_id: int, start_frame: int, detection: FaceDetection, crop: np.ndarray):
        self.track_id = track_id
        self.start_frame = start_frame
        self.last_frame = start_frame
        
        # Frame index -> Bounding Box [x1, y1, x2, y2]
        self.frames: Dict[int, np.ndarray] = {start_frame: detection.bbox}
        # Frame index -> Score
        self.scores: Dict[int, float] = {start_frame: detection.score}
        # Frame index -> Landmarks
        self.landmarks: Dict[int, Optional[np.ndarray]] = {start_frame: detection.landmarks}
        # Sequence of face crops for track-level evaluation
        self.crops: List[np.ndarray] = [crop] if crop is not None else []
        
        self.is_valid: Optional[bool] = None  # Judged at track level

    def add_detection(self, frame_idx: int, detection: FaceDetection, crop: np.ndarray):
        self.last_frame = frame_idx
        self.frames[frame_idx] = detection.bbox
        self.scores[frame_idx] = detection.score
        self.landmarks[frame_idx] = detection.landmarks
        if crop is not None:
            self.crops.append(crop)

    @property
    def latest_bbox(self) -> np.ndarray:
        return self.frames[self.last_frame]

class FaceTracker:
    """
    Online tracker with batch completion & track-level filtering.
    Associates frame detections across time, applies track-level filters,
    interpolates missing frames, and expands blur backward & forward.
    """
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 8,
        padding_backward: int = 8,
        padding_forward: int = 8,
        filters_config: Optional[FiltersConfig] = None
    ):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.padding_backward = padding_backward
        self.padding_forward = padding_forward
        
        self.next_track_id = 0
        self.active_tracks: List[Tracklet] = []
        self.finished_tracks: List[Tracklet] = []

        # Filters
        self.filters_config = filters_config or FiltersConfig()
        self.skin_filter = SkinColorFilter(min_skin_ratio=self.filters_config.skin_color_filter.min_skin_ratio)
        self.animal_filter = AnimalClassifier(human_prob_threshold=self.filters_config.animal_filter.human_prob_threshold)
        self.static_filter = StaticPhotoFilter(
            min_static_frames=self.filters_config.static_photo_filter.min_static_frames,
            motion_similarity_threshold=self.filters_config.static_photo_filter.motion_similarity_threshold,
            texture_diff_threshold=self.filters_config.static_photo_filter.texture_diff_threshold
        )

    def extract_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(np.floor(bbox[0])))
        y1 = max(0, int(np.floor(bbox[1])))
        x2 = min(w, int(np.ceil(bbox[2])))
        y2 = min(h, int(np.ceil(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def update(self, frame_idx: int, frame: np.ndarray, detections: List[FaceDetection]):
        """
        Process detections for frame_idx.
        """
        matched_tracks = set()
        matched_dets = set()

        if self.active_tracks and detections:
            # Build IoU matrix
            ious = np.zeros((len(self.active_tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.active_tracks):
                for d_idx, det in enumerate(detections):
                    ious[t_idx, d_idx] = compute_iou(track.latest_bbox, det.bbox)

            # Greedy matching
            while True:
                max_iou = np.max(ious)
                if max_iou < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(ious), ious.shape)
                if t_idx in matched_tracks or d_idx in matched_dets:
                    ious[t_idx, d_idx] = 0.0
                    continue

                crop = self.extract_crop(frame, detections[d_idx].bbox)
                self.active_tracks[t_idx].add_detection(frame_idx, detections[d_idx], crop)
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                ious[t_idx, :] = 0.0
                ious[:, d_idx] = 0.0

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                crop = self.extract_crop(frame, det.bbox)
                track = Tracklet(self.next_track_id, frame_idx, det, crop)
                self.next_track_id += 1
                self.active_tracks.append(track)

        # Retire stale tracks
        still_active = []
        for t_idx, track in enumerate(self.active_tracks):
            if frame_idx - track.last_frame > self.max_missing_frames:
                self.finished_tracks.append(track)
            else:
                still_active.append(track)
        self.active_tracks = still_active

    def finish(self):
        """Called when video ends to finalize remaining active tracks."""
        self.finished_tracks.extend(self.active_tracks)
        self.active_tracks.clear()

    def evaluate_track_validity(self, track: Tracklet, img_w: int = 1920, img_h: int = 1080) -> bool:
        """
        Evaluate if track is a genuine human face to be blurred.
        Track-level judgment ensures no flickering.
        """
        # Exclusion masks check
        if self.filters_config.exclusion_masks:
            # Check center of track
            all_bboxes = list(track.frames.values())
            avg_center_x = np.mean([(b[0] + b[2]) / 2.0 for b in all_bboxes]) / img_w
            avg_center_y = np.mean([(b[1] + b[3]) / 2.0 for b in all_bboxes]) / img_h
            for mask in self.filters_config.exclusion_masks:
                if len(mask) == 4:
                    mx1, my1, mx2, my2 = mask
                    if mx1 <= avg_center_x <= mx2 and my1 <= avg_center_y <= my2:
                        return False

        # Skin color check on sample crops
        if self.filters_config.skin_color_filter.enabled and track.crops:
            # Check if crops have non-zero content
            valid_crops = [c for c in track.crops if c is not None and np.any(c > 0)]
            if valid_crops:
                skin_valid_votes = []
                for crop in valid_crops[::max(1, len(valid_crops) // 5)]:
                    is_skin, _ = self.skin_filter.is_skin_tone(crop)
                    skin_valid_votes.append(is_skin)
                # Safe side: if at least some crops show human skin, consider valid
                if skin_valid_votes and not any(skin_valid_votes):
                    return False  # Pure statue/bronze without human skin tone


        # Animal filter
        if self.filters_config.animal_filter.enabled and track.crops:
            human_votes = []
            sample_frames = list(track.frames.keys())[::max(1, len(track.frames) // 5)]
            for f in sample_frames:
                lm = track.landmarks.get(f)
                crop = track.crops[0]  # Representative crop
                is_h, _ = self.animal_filter.is_human(crop, lm)
                human_votes.append(is_h)
            if human_votes and not any(human_votes):
                return False

        # Static photo filter
        if self.filters_config.static_photo_filter.enabled and track.crops:
            if self.static_filter.is_static_advertisement(track.crops):
                return False

        return True

    def get_frame_bboxes_map(self, total_frames: int, img_w: int, img_h: int) -> Dict[int, List[np.ndarray]]:
        """
        Build frame_idx -> list of [x1, y1, x2, y2] bounding boxes across all frames,
        applying:
        1. Track-level validity filtering
        2. Linear interpolation for missing frames inside track
        3. Backward (N) and Forward (M) padding extension
        """
        self.finish()
        
        frame_map: Dict[int, List[np.ndarray]] = {i: [] for i in range(total_frames)}

        for track in self.finished_tracks:
            # Check track-level validity
            if not self.evaluate_track_validity(track, img_w, img_h):
                continue

            detected_frames = sorted(track.frames.keys())
            if not detected_frames:
                continue

            min_f = detected_frames[0]
            max_f = detected_frames[-1]

            # Interpolate inside the track duration
            for f in range(min_f, max_f + 1):
                if 0 <= f < total_frames:
                    if f in track.frames:
                        bbox = track.frames[f]
                    else:
                        # Linear interpolation between closest prior and subsequent detections
                        prev_f = max(k for k in detected_frames if k < f)
                        next_f = min(k for k in detected_frames if k > f)
                        alpha = (f - prev_f) / (next_f - prev_f)
                        bbox = (1 - alpha) * track.frames[prev_f] + alpha * track.frames[next_f]
                    frame_map[f].append(bbox)

            # Apply backward padding (before start)
            start_bbox = track.frames[min_f]
            for pad_i in range(1, self.padding_backward + 1):
                target_f = min_f - pad_i
                if 0 <= target_f < total_frames:
                    frame_map[target_f].append(start_bbox)

            # Apply forward padding (after end)
            end_bbox = track.frames[max_f]
            for pad_i in range(1, self.padding_forward + 1):
                target_f = max_f + pad_i
                if 0 <= target_f < total_frames:
                    frame_map[target_f].append(end_bbox)

        return frame_map
