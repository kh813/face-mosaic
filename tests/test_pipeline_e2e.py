import pytest
from pathlib import Path
import tempfile
import numpy as np
import cv2
import subprocess

from face_mosaic.config import AppConfig
from face_mosaic.pipeline import ProcessingPipeline
from face_mosaic.detector import SCRFDDetector
from face_mosaic.cli import collect_video_files

@pytest.fixture(scope="session")
def synthetic_video_path():
    """Create a temporary 1-second 30fps test video with a moving face-like circle."""
    temp_dir = tempfile.mkdtemp()
    video_path = Path(temp_dir) / "test_walk.mp4"
    
    fps = 30
    width = 320
    height = 240
    duration_secs = 1
    total_frames = fps * duration_secs

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(total_frames):
        frame = np.full((height, width, 3), 100, dtype=np.uint8) # Background
        # Draw moving face circle
        cx = int(50 + (width - 100) * (i / total_frames))
        cy = int(height / 2)
        # Skin color circle
        cv2.circle(frame, (cx, cy), 30, (160, 180, 220), -1)
        writer.write(frame)

    writer.release()
    yield str(video_path)

def test_scrfd_detector_initialization():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    model_file = models_dir / "scrfd_2.5g_bnkps.onnx"
    if not model_file.exists():
        pytest.skip("Model scrfd_2.5g_bnkps.onnx not found")

    detector = SCRFDDetector(str(model_file))
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(test_img)
    assert isinstance(results, list)

def test_pipeline_execution(synthetic_video_path):
    models_dir = Path(__file__).resolve().parent.parent / "models"
    model_file = models_dir / "scrfd_2.5g_bnkps.onnx"
    if not model_file.exists():
        pytest.skip("Model scrfd_2.5g_bnkps.onnx not found")

    config = AppConfig()
    config.model.name = "scrfd_2.5g_bnkps.onnx"
    config.output.codec = "h264" # Fast test encoder
    config.output.preset = "ultrafast"

    pipeline = ProcessingPipeline(config, model_dir=str(models_dir))
    out_file = Path(synthetic_video_path).with_name("test_walk_blurred.mp4")

    res = pipeline.process_video(synthetic_video_path, output_path=str(out_file))

    assert res["status"] == "success"
    assert res["total_frames"] == 30
    assert out_file.exists()
    assert out_file.stat().st_size > 1000

def test_cli_collect_video_files(synthetic_video_path):
    p = Path(synthetic_video_path)
    files = collect_video_files([str(p.parent)])
    assert any(f.name == "test_walk.mp4" for f in files)
