"""
Processing pipeline orchestrator for face-mosaic.
Performs 2-pass processing:
Pass 1: Detect and track faces across all frames to establish tracklets and temporal bounds.
Pass 2: Render blur/mosaic with forward/backward padding and pipe directly to ffmpeg encoder.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import time
import numpy as np

from .config import AppConfig
from .io_utils import probe_video, VideoReader, VideoWriter
from .detector import SCRFDDetector
from .tracker import FaceTracker
from .blur import FaceBlurrer
from .color import compute_color_difference

class ProcessingPipeline:
    def __init__(self, config: AppConfig, model_dir: Optional[str] = None):
        self.config = config
        
        # Resolve model path
        if model_dir:
            base_model_dir = Path(model_dir)
        else:
            base_model_dir = Path(__file__).resolve().parent.parent / "models"
        self.model_path = base_model_dir / config.model.name

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Please run 'python scripts/download_models.py' first."
            )

        print(f"Loading SCRFD Face Detector: {self.model_path.name}")
        self.detector = SCRFDDetector(
            model_path=str(self.model_path),
            conf_threshold=self.config.model.conf_threshold,
            nms_threshold=self.config.model.nms_threshold
        )

        self.blurrer = FaceBlurrer(
            blur_type=self.config.blur.type,
            strength=self.config.blur.strength,
            mosaic_block_size=self.config.blur.mosaic_block_size,
            margin_x=self.config.blur.margin_x,
            margin_y=self.config.blur.margin_y
        )

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a single video file.
        Returns a dict of metrics and results.
        """
        start_time = time.time()
        in_path = Path(input_path).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        # Determine output path
        if output_path is None:
            out_dir = Path(self.config.output.output_dir) if self.config.output.output_dir else in_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_filename = f"{in_path.stem}{self.config.output.filename_suffix}.mp4"
            out_path = out_dir / out_filename
        else:
            out_path = Path(output_path).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

        info = probe_video(str(in_path))
        tracker = FaceTracker(
            iou_threshold=self.config.tracking.iou_threshold,
            max_missing_frames=self.config.tracking.max_missing_frames,
            padding_backward=self.config.tracking.padding_backward,
            padding_forward=self.config.tracking.padding_forward,
            filters_config=self.config.filters
        )

        # ----------------- PASS 1: Face Detection & Tracking -----------------
        print(f"\n[Pass 1/2] Detecting faces in {in_path.name} ({info.width}x{info.height}, {info.fps:.2f} fps)...")
        reader1 = VideoReader(str(in_path))
        frame_idx = 0
        
        pass1_start = time.time()
        for frame in reader1:
            # Filter detections by min_face_size
            detections = self.detector.detect(frame)
            valid_dets = [
                d for d in detections 
                if min(d.width, d.height) >= self.config.model.min_face_size
            ]
            tracker.update(frame_idx, frame, valid_dets)
            frame_idx += 1
            if progress_callback:
                progress_callback("Detection", frame_idx, info.total_frames or frame_idx, time.time() - pass1_start)

        reader1.close()
        total_frames = frame_idx
        print(f"Pass 1 complete. Total frames: {total_frames}. Establishing tracklets...")

        # Build blur bbox map for all frames (with padding and interpolation)
        frame_bboxes_map = tracker.get_frame_bboxes_map(total_frames, reader1.out_width, reader1.out_height)

        # ----------------- PASS 2: Blur Rendering & Encoding -----------------
        print(f"[Pass 2/2] Rendering blur/mosaic and encoding to {out_path.name}...")
        writer = VideoWriter(
            output_path=str(out_path),
            input_info=info,
            width=reader1.out_width,
            height=reader1.out_height,
            fps=info.fps,
            codec=self.config.output.codec,
            crf=self.config.output.crf,
            preset=self.config.output.preset,
            preserve_color_tags=self.config.output.preserve_color_tags
        )

        reader2 = VideoReader(str(in_path))
        pass2_start = time.time()
        
        mae_samples = []
        delta_e_samples = []
        
        for f_idx, frame in enumerate(reader2):
            bboxes = frame_bboxes_map.get(f_idx, [])
            blurred_frame, mask = self.blurrer.apply_blur(frame, bboxes)
            writer.write_frame(blurred_frame)
            
            # Sample color difference on 5% of frames
            if f_idx % max(1, total_frames // 20) == 0:
                diff_metrics = compute_color_difference(frame, blurred_frame, mask)
                mae_samples.append(diff_metrics["mae"])
                delta_e_samples.append(diff_metrics["delta_e_approx"])

            if progress_callback:
                progress_callback("Rendering", f_idx + 1, total_frames, time.time() - pass2_start)

        reader2.close()
        writer.close()

        total_elapsed = time.time() - start_time
        fps_proc = total_frames / total_elapsed if total_elapsed > 0 else 0.0

        avg_mae = float(np.mean(mae_samples)) if mae_samples else 0.0
        avg_delta_e = float(np.mean(delta_e_samples)) if delta_e_samples else 0.0

        result = {
            "input": str(in_path),
            "output": str(out_path),
            "total_frames": total_frames,
            "elapsed_seconds": total_elapsed,
            "processing_fps": fps_proc,
            "color_mae": avg_mae,
            "color_delta_e": avg_delta_e,
            "status": "success"
        }
        
        print(f"Finished {in_path.name} -> {out_path.name} in {total_elapsed:.2f}s ({fps_proc:.2f} fps). Color MAE: {avg_mae:.4f}")
        return result
