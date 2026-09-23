import pytest
import numpy as np
import cv2
from face_mosaic.filters.illustration import IllustrationFilter
from face_mosaic.blur import FaceBlurrer
from face_mosaic.tracker import FaceTracker, Tracklet
from face_mosaic.detector import FaceDetection
from face_mosaic.config import FiltersConfig, IllustrationFilterConfig, BlurConfig

def test_round_mosaic_and_gaussian_blur_shape():
    """Verify that FaceBlurrer with shape='ellipse' blurs only inside the ellipse."""
    blurrer_round = FaceBlurrer(blur_type="mosaic", mosaic_block_size=8, margin_x=0.0, margin_y=0.0, shape="ellipse")
    blurrer_rect = FaceBlurrer(blur_type="mosaic", mosaic_block_size=8, margin_x=0.0, margin_y=0.0, shape="rect")

    # Create a test frame 100x100 with a gradient pattern
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(100):
        for x in range(100):
            frame[y, x] = [x * 2, y * 2, (x + y)]

    bbox = np.array([20, 20, 80, 80], dtype=np.float32)  # 60x60 square

    # Apply round mosaic
    out_round, mask_round = blurrer_round.apply_blur(frame, [bbox])
    # Apply rect mosaic
    out_rect, mask_rect = blurrer_rect.apply_blur(frame, [bbox])

    # Check center (50, 50): both should be blurred (mask == 255)
    assert mask_round[50, 50] == 255
    assert mask_rect[50, 50] == 255
    assert not np.array_equal(out_round[50, 50], frame[50, 50])

    # Check near corner (21, 21):
    # For rect, corner (21, 21) is inside the bbox -> mask == 255 and blurred
    assert mask_rect[21, 21] == 255
    # For round (ellipse with radius 30 centered at 50, distance to 21 is ~41 > 30):
    # Corner (21, 21) is OUTSIDE the ellipse -> mask == 0 and original pixel preserved!
    assert mask_round[21, 21] == 0
    assert np.array_equal(out_round[21, 21], frame[21, 21])

    # Verify Gaussian blur also works with round shape
    blurrer_gauss_round = FaceBlurrer(blur_type="gaussian", strength=15, margin_x=0.0, margin_y=0.0, shape="ellipse")
    out_gauss, mask_gauss = blurrer_gauss_round.apply_blur(frame, [bbox])
    assert mask_gauss[50, 50] == 255
    assert mask_gauss[21, 21] == 0
    assert np.array_equal(out_gauss[21, 21], frame[21, 21])

def test_illustration_filter_real_human_vs_flat_drawing():
    """Verify that flat-shaded anime/illustrations are rejected, while real skin is accepted."""
    filter_instance = IllustrationFilter(flatness_threshold=0.65)

    # 1. Simulate a flat-shaded anime face (uniform skin tone with black line edges)
    # Skin color in BGR: [180, 200, 240] (light anime skin)
    flat_crop = np.full((64, 64, 3), [180, 200, 240], dtype=np.uint8)
    # Add black ink line contour/edges (typical anime eyes/outline)
    flat_crop[10:14, 20:30] = [10, 10, 10]
    flat_crop[10:14, 35:45] = [10, 10, 10]
    flat_crop[45:48, 25:40] = [10, 10, 10]

    is_human, score = filter_instance.is_human_face(flat_crop)
    # Flat anime face must be recognized as illustration (is_human == False)
    assert is_human is False
    assert score < 0.5

    # 2. Simulate a real human face crop:
    # Rich skin gradient lighting + natural micro-texture variance
    np.random.seed(42)
    real_crop = np.zeros((64, 64, 3), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            # Lighting gradient across cheek/forehead + natural micro-variation
            grad = (x * 0.5 + y * 0.8)
            noise = np.random.normal(0, 5)
            b = np.clip(130 + grad * 0.3 + noise, 0, 255)
            g = np.clip(160 + grad * 0.5 + noise, 0, 255)
            r = np.clip(210 + grad * 0.6 + noise, 0, 255)
            real_crop[y, x] = [b, g, r]

    is_human_real, score_real = filter_instance.is_human_face(real_crop)
    # Real human face must be preserved (is_human == True)
    assert is_human_real is True
    assert score_real > 0.5

def test_illustration_filter_safety_on_small_crops():
    """Very small crops should default to safe human preservation."""
    filter_instance = IllustrationFilter(min_crop_size=16)
    small_crop = np.full((12, 12, 3), [180, 200, 240], dtype=np.uint8)
    is_human, score = filter_instance.is_human_face(small_crop)
    assert is_human is True
    assert score == 1.0

def test_tracker_excludes_illustration_track():
    """Verify that FaceTracker rejects tracks containing flat-shaded illustrations."""
    tracker = FaceTracker(
        filters_config=FiltersConfig(
            illustration_filter=IllustrationFilterConfig(enabled=True)
        )
    )

    # Flat anime drawing crop
    flat_crop = np.full((64, 64, 3), [180, 200, 240], dtype=np.uint8)
    flat_crop[10:14, 20:30] = [10, 10, 10]
    flat_crop[45:48, 25:40] = [10, 10, 10]

    det = FaceDetection(bbox=[100, 100, 200, 200], score=0.85)
    track = Tracklet(track_id=1, start_frame=0, detection=det, crop=flat_crop)
    # Add multiple frames of flat illustration
    for f in range(1, 10):
        track.add_detection(f, det, flat_crop.copy())

    # Evaluation should reject this illustration
    assert tracker.evaluate_track_validity(track) is False
