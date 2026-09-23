import pytest
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from face_mosaic.config import AppConfig
from face_mosaic.pipeline import ProcessingPipeline
from face_mosaic.detector import FaceDetection
from face_mosaic.tracker import FaceTracker, Tracklet
from face_mosaic.checkpoint import VideoCheckpoint

def test_crop_memory_isolation_and_capping():
    """
    Verify that:
    1. extract_crop returns an independent copy (crop.base is not frame)
    2. Tracklet.crops is capped at 35 to prevent unbounded memory growth
    """
    tracker = FaceTracker()
    large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bbox = np.array([100, 100, 300, 300], dtype=np.float32)

    crop = tracker.extract_crop(large_frame, bbox)
    # Must NOT hold reference to large_frame
    assert crop.base is None or crop.base is not large_frame
    assert crop.shape[0] <= 128 and crop.shape[1] <= 128

    # Test Tracklet crop capping
    det = FaceDetection(bbox=bbox, score=0.95)
    tracklet = Tracklet(track_id=1, start_frame=0, detection=det, crop=crop)
    for f in range(1, 100):
        tracklet.add_detection(f, det, crop)

    assert len(tracklet.crops) <= 35


def test_checkpoint_lifecycle(tmp_path):
    out_video = tmp_path / "test_out.mp4"
    in_video = tmp_path / "test_in.mp4"
    in_video.write_bytes(b"dummy video data for size and mtime")

    chk = VideoCheckpoint.for_video(out_video)
    assert not chk.exists()

    chk.init_state(in_video, total_frames=100, width=1920, height=1080, fps=30.0, model_name="scrfd_10g", conf_threshold=0.5)
    assert chk.exists()
    assert chk.matches_input(in_video, model_name="scrfd_10g", conf_threshold=0.5)
    assert not chk.matches_input(in_video, model_name="different_model", conf_threshold=0.5)

    # Append detections
    dets = [FaceDetection(bbox=np.array([10, 20, 30, 40]), score=0.9)]
    for i in range(150):
        chk.append_detection(i, dets)
    chk.flush_detections(last_frame=149)

    cached = chk.load_cached_detections()
    assert len(cached) == 150
    assert 0 in cached and 149 in cached
    assert cached[0][0].score == 0.9

    # Save Pass 1 completed
    bboxes_map = {i: [np.array([10, 20, 30, 40], dtype=np.float32)] for i in range(100)}
    chk.save_pass1_completed(bboxes_map, total_frames=100)
    assert chk.is_pass1_completed()

    loaded_map = chk.load_frame_bboxes_map()
    assert loaded_map is not None
    assert len(loaded_map) == 100
    assert isinstance(loaded_map[0][0], np.ndarray)

    # Cleanup
    chk.cleanup()
    assert not chk.exists()


def test_pipeline_resume_skips_pass1(tmp_path):
    """
    Test that when Pass 1 checkpoint exists, ProcessingPipeline reuses it
    and does not run face detection again.
    """
    # Create small video
    video_path = tmp_path / "resume_test.mp4"
    fps = 30
    width = 320
    height = 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for _ in range(10):
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    config = AppConfig()
    config.output.codec = "h264"
    config.output.preset = "ultrafast"
    pipeline = ProcessingPipeline(config)

    mock_detector = MagicMock()
    mock_detector.active_provider = "CPU"
    mock_detector.detect.return_value = [
        FaceDetection(bbox=np.array([50, 50, 100, 100]), score=0.9)
    ]
    pipeline.detector = mock_detector

    out_file = tmp_path / "output_resume.mp4"

    # Pre-create completed Pass 1 checkpoint
    chk = VideoCheckpoint.for_video(out_file)
    chk.init_state(video_path, total_frames=10, width=width, height=height, fps=fps, model_name=config.model.name, conf_threshold=config.model.conf_threshold)
    bboxes_map = {i: [np.array([50, 50, 100, 100], dtype=np.float32)] for i in range(10)}
    chk.save_pass1_completed(bboxes_map, total_frames=10)

    # Now run pipeline with resume=True
    progress_calls = []
    def on_prog(phase, cur, tot, el):
        progress_calls.append(phase)

    res = pipeline.process_video(
        str(video_path),
        output_path=str(out_file),
        progress_callback=on_prog,
        resume=True
    )

    assert res["status"] == "success"
    # Detector should NOT have been called because Pass 1 was cached!
    assert mock_detector.detect.call_count == 0
    assert any("Cached" in p for p in progress_calls)
    assert out_file.exists()
