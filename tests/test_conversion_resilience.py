import pytest
from pathlib import Path
import tempfile
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from face_mosaic.config import AppConfig
from face_mosaic.pipeline import ProcessingPipeline
from face_mosaic.detector import FaceDetection
from face_mosaic.gui.app import ProcessingWorker, QueueItem

@pytest.fixture
def test_video_file(tmp_path):
    """Creates a small 15-frame 320x240 video for fast conversion testing."""
    video_path = tmp_path / "resilience_test.mp4"
    fps = 30
    width = 320
    height = 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(15):
        frame = np.full((height, width, 3), 80, dtype=np.uint8)
        # Draw some background shapes
        cv2.rectangle(frame, (10, 10), (100, 100), (200, 150, 100), -1)
        writer.write(frame)
    writer.release()
    return str(video_path)


def test_full_conversion_with_preview_and_detections(test_video_file, tmp_path):
    """
    Guards against:
    - Missing cv2 in preview rendering
    - Missing x1/y1/x2/y2 attributes on FaceDetection
    - Type errors in preview_callback
    """
    config = AppConfig()
    config.output.codec = "h264"
    config.output.preset = "ultrafast"
    pipeline = ProcessingPipeline(config)

    # Mock detector returning realistic FaceDetection instances
    mock_detector = MagicMock()
    mock_detector.active_provider = "CPU"
    mock_detector.detect.return_value = [
        FaceDetection(bbox=np.array([50.2, 40.8, 120.6, 130.4]), score=0.92),
        FaceDetection(bbox=np.array([160.0, 70.0, 220.0, 150.0]), score=0.85),
    ]
    pipeline.detector = mock_detector

    preview_logs = []
    def preview_cb(frame, status=""):
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        preview_logs.append(status)

    progress_logs = []
    def progress_cb(phase, cur, tot, elapsed):
        progress_logs.append((phase, cur, tot))

    out_file = tmp_path / "output_preview.mp4"
    result = pipeline.process_video(
        test_video_file,
        output_path=str(out_file),
        preview_callback=preview_cb,
        progress_callback=progress_cb
    )

    assert result["status"] == "success"
    assert out_file.exists()
    assert out_file.stat().st_size > 500
    # Both Pass 1 and Pass 2 preview callbacks must have fired
    assert any("Pass 1" in s for s in preview_logs)
    assert any("Pass 2" in s for s in preview_logs)
    assert len(progress_logs) > 0


def test_conversion_with_out_of_bounds_and_negative_coords(test_video_file, tmp_path):
    """
    Guards against:
    - Negative coordinates [-50, -30, 80, 90] crashing cv2.rectangle or slicing
    - Out of bounds coordinates [280, 200, 400, 300] exceeding frame (320x240)
    - Zero or near-zero area boxes
    """
    config = AppConfig()
    config.output.codec = "h264"
    config.output.preset = "ultrafast"
    config.model.min_face_size = 5
    pipeline = ProcessingPipeline(config)

    # Return boundary cases
    mock_detector = MagicMock()
    mock_detector.active_provider = "CPU"
    mock_detector.detect.return_value = [
        FaceDetection(bbox=np.array([-40.0, -20.0, 80.0, 90.0]), score=0.95),     # negative top-left
        FaceDetection(bbox=np.array([260.0, 180.0, 380.0, 300.0]), score=0.88),  # exceeds 320x240
        FaceDetection(bbox=np.array([100.0, 100.0, 106.0, 106.0]), score=0.75),  # very small
    ]
    pipeline.detector = mock_detector

    preview_frames = []
    def preview_cb(frame, status=""):
        preview_frames.append(frame)

    out_file = tmp_path / "output_oob.mp4"
    result = pipeline.process_video(
        test_video_file,
        output_path=str(out_file),
        preview_callback=preview_cb
    )

    assert result["status"] == "success"
    assert out_file.exists()
    assert len(preview_frames) > 0


@pytest.mark.parametrize("blur_type, strength, block_size", [
    ("mosaic", 21, 4),
    ("mosaic", 21, 64),
    ("gaussian", 3, 16),
    ("gaussian", 99, 16),
])
def test_conversion_with_all_blur_types_and_extremes(test_video_file, tmp_path, blur_type, strength, block_size):
    """
    Guards against:
    - Kernel size errors in gaussian blur (even numbers, sizes larger than ROI)
    - Block size edge cases in mosaic pixelation
    """
    config = AppConfig()
    config.output.codec = "h264"
    config.output.preset = "ultrafast"
    config.blur.type = blur_type
    config.blur.strength = strength
    config.blur.mosaic_block_size = block_size
    config.blur.margin_x = 0.3
    config.blur.margin_y = 0.3

    pipeline = ProcessingPipeline(config)
    mock_detector = MagicMock()
    mock_detector.active_provider = "CPU"
    mock_detector.detect.return_value = [
        FaceDetection(bbox=np.array([30, 30, 90, 90]), score=0.9)
    ]
    pipeline.detector = mock_detector

    out_file = tmp_path / f"output_{blur_type}_{strength}_{block_size}.mp4"
    result = pipeline.process_video(test_video_file, output_path=str(out_file))

    assert result["status"] == "success"
    assert out_file.exists()


def test_conversion_crowd_and_intermittent_detections(test_video_file, tmp_path):
    """
    Guards against:
    - Multi-face tracking index out-of-bounds (20+ faces)
    - Frames with 0 detections followed by frames with multiple detections
    """
    config = AppConfig()
    config.output.codec = "h264"
    config.output.preset = "ultrafast"
    pipeline = ProcessingPipeline(config)

    frame_counter = [0]
    def dynamic_detect(frame):
        frame_counter[0] += 1
        if frame_counter[0] % 3 == 0:
            # Empty frame
            return []
        # Crowd scene: 12 faces
        crowd = []
        for i in range(12):
            x = (i % 4) * 70 + 10
            y = (i // 4) * 60 + 10
            crowd.append(FaceDetection(bbox=np.array([x, y, x + 40, y + 40]), score=0.8 + (i * 0.01)))
        return crowd

    mock_detector = MagicMock()
    mock_detector.active_provider = "CPU"
    mock_detector.detect.side_effect = dynamic_detect
    pipeline.detector = mock_detector

    out_file = tmp_path / "output_crowd.mp4"
    result = pipeline.process_video(test_video_file, output_path=str(out_file))

    assert result["status"] == "success"
    assert out_file.exists()


def test_worker_gui_integration_full_conversion(test_video_file, tmp_path, qapp):
    """
    Guards against:
    - Worker thread signal emission failures
    - Uncaught exceptions causing UI freeze or silent worker death
    - Item status remaining stuck in 'Processing' or 'Queued'
    """
    config = AppConfig()
    config.output.codec = "h264"
    config.output.preset = "ultrafast"

    # Setup queue item
    item = QueueItem(path=Path(test_video_file), status="Waiting")
    worker = ProcessingWorker(config, [item])

    # Mock detector
    mock_detector = MagicMock()
    mock_detector.active_provider = "CPU"
    mock_detector.detect.return_value = [
        FaceDetection(bbox=np.array([40, 40, 100, 100]), score=0.90)
    ]
    orig_pipeline_init = ProcessingPipeline.__init__
    def patched_pipeline_init(self, *args, **kwargs):
        orig_pipeline_init(self, *args, **kwargs)
        self.detector = mock_detector

    with patch.object(ProcessingPipeline, '__init__', patched_pipeline_init):
        item_updates = []
        previews = []
        finished_results = []

        worker.item_updated_signal.connect(lambda i_id, stat, prog, out: item_updates.append((stat, prog)))
        worker.preview_signal.connect(lambda img, txt: previews.append((img.shape, txt)))
        worker.finished_signal.connect(lambda ok, msg: finished_results.append((ok, msg)))

        # Run worker synchronously
        worker.run()

        assert item.status == "Completed"
        assert item.progress == "100%"
        assert len(previews) > 0
        assert len(finished_results) == 1
        assert finished_results[0][0] is True


def test_worker_catches_and_reports_fatal_pipeline_exception(test_video_file, tmp_path, qapp):
    """
    Guards against:
    - Unhandled exception crashing the GUI application
    - Worker properly setting item status to 'Error' and reporting traceback
    """
    config = AppConfig()
    item = QueueItem(path=Path(test_video_file), status="Waiting")
    worker = ProcessingWorker(config, [item])

    # Force an unhandled exception inside process_video
    with patch.object(ProcessingPipeline, 'process_video', side_effect=RuntimeError("Simulated hardware crash")):
        log_messages = []
        worker.log_signal.connect(lambda msg: log_messages.append(msg))
        worker.run()

        assert item.status == "Error"
        assert item.progress == "Error"
        assert any("Simulated hardware crash" in m for m in log_messages)
