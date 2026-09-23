"""
Command Line Interface for face-mosaic.
Handles single file, multiple files, and recursive folder batch processing.
Supports skip-on-error behavior and outputs summary reports.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List
from tqdm import tqdm

from .config import AppConfig
from .pipeline import ProcessingPipeline

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

def collect_video_files(inputs: List[str], recursive: bool = True) -> List[Path]:
    """
    Collect video files from single files, list of files, and directories.
    """
    collected = []
    for item in inputs:
        p = Path(item).resolve()
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                collected.append(p)
            else:
                print(f"[Warning] Skipping non-video file: {p.name}")
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for sub_p in p.glob(pattern):
                if sub_p.is_file() and sub_p.suffix.lower() in VIDEO_EXTENSIONS:
                    # Skip already blurred output files to avoid infinite loops
                    if "_blurred" not in sub_p.stem:
                        collected.append(sub_p.resolve())
        else:
            print(f"[Warning] Path does not exist: {item}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in collected:
        if str(f) not in seen:
            seen.add(str(f))
            unique_files.append(f)
            
    return unique_files

def main():
    parser = argparse.ArgumentParser(
        description="face-mosaic: Automated offline face blurring/mosaic tool for walk videos"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="Input video file(s) or folder(s) to process"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical user interface (GUI)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run in folder watch mode (continuously monitor input folder for new videos)"
    )


    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="SCRFD model name (e.g. scrfd_2.5g_bnkps.onnx, scrfd_10g_bnkps.onnx, scrfd_34g_gnkps.onnx)"
    )
    parser.add_argument(
        "--blur-type", "-b",
        choices=["gaussian", "mosaic"],
        default=None,
        help="Blur algorithm (gaussian or mosaic)"
    )
    parser.add_argument(
        "--strength", "-s",
        type=int,
        default=None,
        help="Gaussian blur strength (odd int) or mosaic block size"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (0.0 - 1.0)"
    )
    parser.add_argument(
        "--pad-back",
        type=int,
        default=None,
        help="Frames to pad backward before face appearance (default: 3)"
    )
    parser.add_argument(
        "--pad-fwd",
        type=int,
        default=None,
        help="Frames to pad forward after face disappearance (default: 3)"
    )
    parser.add_argument(
        "--codec",
        choices=["hevc", "h264"],
        default=None,
        help="Video encoder codec (hevc or h264)"
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="Constant Rate Factor (default: 18)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--shape",
        choices=["ellipse", "rect"],
        default=None,
        help="Blur/mosaic shape (ellipse for round, rect for rectangular)"
    )
    parser.add_argument(
        "--no-illustration-filter",
        action="store_true",
        help="Disable automatic illustration / anime face exclusion"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search in directories"
    )

    args = parser.parse_args()

    if args.gui or len(args.inputs) == 0:
        from .gui import run_app
        run_app()
        return

    # Load configuration

    config = AppConfig.load(args.config)

    # CLI Overrides
    if args.model:
        config.model.name = args.model
    elif args.config is None:
        from .hardware import detect_hardware_profile
        hw = detect_hardware_profile()
        config.model.name = hw.recommended_model
        print(f"[Hardware Auto-Select] {hw.reason}")
    if args.blur_type:
        config.blur.type = args.blur_type
    if args.strength:
        if config.blur.type == "mosaic":
            config.blur.mosaic_block_size = args.strength
        else:
            config.blur.strength = args.strength
    if args.conf is not None:
        config.model.conf_threshold = args.conf
    if args.pad_back is not None:
        config.tracking.padding_backward = args.pad_back
    if args.pad_fwd is not None:
        config.tracking.padding_forward = args.pad_fwd
    if args.codec:
        config.output.codec = args.codec
    if args.crf is not None:
        config.output.crf = args.crf
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.shape:
        config.blur.shape = args.shape
    if args.no_illustration_filter:
        config.filters.illustration_filter.enabled = False

    # Folder Watch Mode
    if args.watch:
        from .watcher import FolderWatcher
        target_dir = args.inputs[0] if args.inputs else "."
        print(f"Starting Folder Watcher on: {target_dir}")
        watcher = FolderWatcher(target_dir, config=config)
        watcher.start()
        return

    video_files = collect_video_files(args.inputs, recursive=not args.no_recursive)

    if not video_files:
        print("Error: No video files found matching the specified inputs.")
        sys.exit(1)

    print(f"Found {len(video_files)} video file(s) for batch processing.")

    try:
        pipeline = ProcessingPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    success_list = []
    failed_list = []

    for idx, video_path in enumerate(video_files, 1):
        print(f"\n=======================================================")
        print(f"[{idx}/{len(video_files)}] Processing: {video_path.name}")
        print(f"=======================================================")
        
        # Progress callback with tqdm
        pbar = None
        current_phase = [None]
        
        def on_progress(phase: str, current: int, total: int, elapsed: float):
            nonlocal pbar
            if current_phase[0] != phase:
                if pbar:
                    pbar.close()
                current_phase[0] = phase
                pbar = tqdm(total=total, desc=f"  {phase}", unit="frame")
            if pbar:
                pbar.n = current
                pbar.refresh()

        try:
            res = pipeline.process_video(str(video_path), progress_callback=on_progress)
            if pbar:
                pbar.close()
            success_list.append(res)
        except Exception as e:
            if pbar:
                pbar.close()
            print(f"\n[Error] Failed to process {video_path.name}: {e}")
            # Spec requirement: continue on error
            failed_list.append({
                "input": str(video_path),
                "error": str(e),
                "status": "failed"
            })

    # Summary Report
    print("\n=======================================================")
    print("                BATCH PROCESSING SUMMARY               ")
    print("=======================================================")
    print(f"Total videos queued: {len(video_files)}")
    print(f"Successfully processed: {len(success_list)}")
    print(f"Failed / Skipped: {len(failed_list)}")

    if success_list:
        print("\nSuccessful files:")
        for s in success_list:
            print(f"  ✓ {Path(s['input']).name} -> {Path(s['output']).name} ({s['elapsed_seconds']:.1f}s, {s['processing_fps']:.1f} fps)")

    if failed_list:
        print("\nFailed files:")
        for f in failed_list:
            print(f"  ✗ {Path(f['input']).name}: {f['error']}")

    print("=======================================================\n")
    if failed_list and not success_list:
        sys.exit(1)

if __name__ == "__main__":
    main()
