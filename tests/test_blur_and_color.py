import pytest
import numpy as np
from face_mosaic.blur import FaceBlurrer
from face_mosaic.color import compute_color_difference, ColorMetadata

def test_blur_expand_bbox():
    blurrer = FaceBlurrer(margin_x=0.2, margin_y=0.2)
    # bbox: [100, 100, 200, 200] -> w=100, h=100 -> dx=20, dy=20 -> [80, 80, 220, 220]
    ex1, ey1, ex2, ey2 = blurrer.expand_bbox(np.array([100, 100, 200, 200]), 1000, 1000)
    assert ex1 == 80
    assert ey1 == 80
    assert ex2 == 220
    assert ey2 == 220

def test_gaussian_and_mosaic_blur():
    blurrer_g = FaceBlurrer(blur_type="gaussian", strength=31)
    blurrer_m = FaceBlurrer(blur_type="mosaic", mosaic_block_size=10)
    
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    bboxes = [np.array([50, 50, 150, 150])]

    g_out, g_mask = blurrer_g.apply_blur(img, bboxes)
    assert g_out.shape == img.shape
    assert np.any(g_mask == 255)
    assert np.any(g_mask == 0)

    m_out, m_mask = blurrer_m.apply_blur(img, bboxes)
    assert m_out.shape == img.shape
    assert np.any(m_mask == 255)

def test_color_difference_metric():
    img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    img2 = img1.copy()
    # Blur a patch
    img2[20:40, 20:40] = 0
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 255

    # Color difference excluding mask should be 0.0
    res = compute_color_difference(img1, img2, ignore_mask=mask)
    assert res["mae"] == 0.0
    assert res["delta_e_approx"] == 0.0

def test_color_metadata_ffmpeg_args():
    meta = ColorMetadata(
        color_space="bt2020nc",
        color_transfer="smpte2084",
        color_primaries="bt2020",
        color_range="tv"
    )
    args = meta.to_ffmpeg_args()
    assert "-colorspace" in args and "bt2020nc" in args
    assert "-color_trc" in args and "smpte2084" in args
    assert "-color_primaries" in args and "bt2020" in args
    assert "-color_range" in args and "tv" in args
