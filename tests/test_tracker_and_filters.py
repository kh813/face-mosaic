import pytest
import numpy as np
from face_mosaic.detector import FaceDetection
from face_mosaic.tracker import FaceTracker, compute_iou
from face_mosaic.filters import SkinColorFilter, AnimalClassifier, StaticPhotoFilter
from face_mosaic.config import FiltersConfig

def test_iou_calculation():
    b1 = np.array([0, 0, 10, 10])
    b2 = np.array([5, 5, 15, 15])
    # Inter: [5, 5, 10, 10] -> 25. Union: 100 + 100 - 25 = 175 -> 25/175 = 1/7
    iou = compute_iou(b1, b2)
    assert abs(iou - (1.0 / 7.0)) < 1e-4

def test_skin_color_filter():
    filt = SkinColorFilter(min_skin_ratio=0.15)
    # Natural skin tone in BGR (e.g. B=160, G=180, R=220)
    skin_img = np.zeros((50, 50, 3), dtype=np.uint8)
    skin_img[:, :] = (160, 180, 220)
    is_skin, ratio = filt.is_skin_tone(skin_img)
    assert is_skin is True
    assert ratio > 0.8

    # Blue non-skin tone (B=255, G=0, R=0)
    blue_img = np.zeros((50, 50, 3), dtype=np.uint8)
    blue_img[:, :] = (255, 0, 0)
    is_skin, ratio = filt.is_skin_tone(blue_img)
    assert is_skin is False
    assert ratio < 0.05

def test_tracker_forward_backward_padding():
    tracker = FaceTracker(
        iou_threshold=0.3,
        padding_backward=3,
        padding_forward=3
    )

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Face appears at frame 10, 11, 12
    det1 = FaceDetection(bbox=np.array([50, 50, 100, 100]), score=0.9)
    det2 = FaceDetection(bbox=np.array([52, 52, 102, 102]), score=0.9)
    det3 = FaceDetection(bbox=np.array([54, 54, 104, 104]), score=0.9)

    tracker.update(10, frame, [det1])
    tracker.update(11, frame, [det2])
    tracker.update(12, frame, [det3])

    frame_map = tracker.get_frame_bboxes_map(total_frames=20, img_w=200, img_h=200)

    # Frame 7, 8, 9 should have backward padding bboxes
    assert len(frame_map[7]) == 1
    assert len(frame_map[8]) == 1
    assert len(frame_map[9]) == 1

    # Detected frames 10, 11, 12
    assert len(frame_map[10]) == 1
    assert len(frame_map[11]) == 1
    assert len(frame_map[12]) == 1

    # Frame 13, 14, 15 should have forward padding bboxes
    assert len(frame_map[13]) == 1
    assert len(frame_map[14]) == 1
    assert len(frame_map[15]) == 1

    # Frame 6 and Frame 16 should be empty
    assert len(frame_map[6]) == 0
    assert len(frame_map[16]) == 0
