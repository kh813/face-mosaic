"""
Video I/O and ffmpeg process wrapper module for face-mosaic.
Handles reading video frames via ffmpeg pipe (with rotation & color tags),
extracting audio, querying metadata with ffprobe, and writing processed frames.
"""

import json
import subprocess
import os
import sys
import shutil
import threading
import queue
import time
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
import numpy as np

from .color import ColorMetadata, parse_color_metadata

def get_subprocess_kwargs() -> Dict[str, Any]:
    """
    Platform-specific subprocess kwargs to prevent console (cmd.exe)
    windows from popping up when spawning ffmpeg/ffprobe on Windows.
    """
    kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = si
    return kwargs

_qsv_supported: Optional[bool] = None

def supports_qsv() -> bool:
    """Check if Intel Quick Sync Video (QSV) is supported by ffmpeg."""
    global _qsv_supported
    if _qsv_supported is not None:
        return _qsv_supported
    try:
        ffmpeg_bin = get_ffmpeg_cmd()
        res = subprocess.run(
            [ffmpeg_bin, "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **get_subprocess_kwargs()
        )
        _qsv_supported = "hevc_qsv" in res.stdout
    except Exception:
        _qsv_supported = False
    return _qsv_supported

def get_ffprobe_cmd() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path
    candidates = [
        Path(__file__).resolve().parent.parent / "bin" / "ffprobe.exe",
        Path.home() / "bin" / "ffprobe.exe",
        Path("C:/ffmpeg/bin/ffprobe.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffprobe.exe"),
        Path(__file__).resolve().parent.parent / "ffprobe.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "ffprobe"

def get_ffmpeg_cmd() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    candidates = [
        Path(__file__).resolve().parent.parent / "bin" / "ffmpeg.exe",
        Path.home() / "bin" / "ffmpeg.exe",
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path(__file__).resolve().parent.parent / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "ffmpeg"

class VideoInfo:
    def __init__(self, probe_data: Dict[str, Any], filepath: str):
        self.filepath = filepath
        self.probe_data = probe_data
        
        # Find video stream
        video_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"]
        if not video_streams:
            raise ValueError(f"No video stream found in {filepath}")
        self.video_stream = video_streams[0]
        
        # Audio stream existence
        audio_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"]
        self.has_audio = len(audio_streams) > 0
        self.audio_stream = audio_streams[0] if self.has_audio else None

        self.width = int(self.video_stream.get("width", 0))
        self.height = int(self.video_stream.get("height", 0))
        
        # Frame rate calculation
        r_fps = self.video_stream.get("r_frame_rate", "30/1")
        avg_fps = self.video_stream.get("avg_frame_rate", "30/1")
        try:
            num, den = map(int, avg_fps.split("/"))
            self.fps = num / den if den != 0 else 30.0
        except Exception:
            self.fps = 30.0
            
        # Check VFR/CFR warning
        try:
            r_num, r_den = map(int, r_fps.split("/"))
            r_val = r_num / r_den if r_den != 0 else self.fps
            if abs(r_val - self.fps) > 1.0:
                print(f"[Warning] Variable frame rate (VFR) detected in {filepath} (avg_fps: {self.fps:.2f}, r_fps: {r_val:.2f})")
        except Exception:
            pass

        # Total frames
        nb_frames = self.video_stream.get("nb_frames")
        if nb_frames and nb_frames.isdigit():
            self.total_frames = int(nb_frames)
        else:
            # Estimate from duration
            duration = float(probe_data.get("format", {}).get("duration", 0.0))
            self.total_frames = int(duration * self.fps) if duration > 0 else 0

        # Rotation
        self.rotation = 0
        side_data_list = self.video_stream.get("side_data_list", [])
        for sd in side_data_list:
            if "rotation" in sd:
                self.rotation = int(sd["rotation"])
        if "tags" in self.video_stream and "rotate" in self.video_stream["tags"]:
            self.rotation = int(self.video_stream["tags"]["rotate"])

        # Color Metadata
        self.color_metadata = parse_color_metadata(self.video_stream)

def probe_video(filepath: str) -> VideoInfo:
    """Run ffprobe to get detailed stream and color info."""
    ffprobe_bin = get_ffprobe_cmd()
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-print_format", "json",
        str(filepath)
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **get_subprocess_kwargs()
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"ffprobe executable not found ('{ffprobe_bin}'). "
            f"Please ensure FFmpeg is installed and added to PATH or placed in the bin/ directory."
        )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {filepath}: {res.stderr}")
    data = json.loads(res.stdout)
    return VideoInfo(data, str(filepath))

class VideoReader:
    """
    Reads frames as raw BGR24 arrays using ffmpeg subprocess pipe.
    Automatically handles autorotation filter.
    """
    def __init__(self, filepath: str):
        self.filepath = str(filepath)
        self.info = probe_video(filepath)
        
        # When autorotate filter is applied by ffmpeg, 90/270 degree rotation swaps width and height
        self.needs_swap = abs(self.info.rotation) in (90, 270)
        self.out_width = self.info.height if self.needs_swap else self.info.width
        self.out_height = self.info.width if self.needs_swap else self.info.height
        
        # Build ffmpeg command
        ffmpeg_bin = get_ffmpeg_cmd()
        cmd = [
            ffmpeg_bin,
            "-v", "error",
            "-i", self.filepath,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1"
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
                **get_subprocess_kwargs()
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"ffmpeg executable not found ('{ffmpeg_bin}'). "
                f"Please ensure FFmpeg is installed and added to PATH or placed in the bin/ directory."
            )
        self.frame_size = self.out_width * self.out_height * 3

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while True:
            raw_bytes = self.process.stdout.read(self.frame_size)
            if not raw_bytes or len(raw_bytes) < self.frame_size:
                ret = self.process.poll()
                if ret is not None and ret != 0:
                    err = self.process.stderr.read().decode('utf-8', errors='replace')
                    raise RuntimeError(f"FFmpeg read error (code {ret}): {err}")
                break
            frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.out_height, self.out_width, 3))
            yield frame

    def close(self):
        if self.process:
            self.process.stdout.close()
            self.process.wait()

class PrefetchedVideoReader:
    """
    Wraps VideoReader with a background worker thread and queue.
    Prefetches decoded frames so GPU inference does not block on FFmpeg I/O.
    """
    def __init__(self, filepath: str, queue_size: int = 16):
        self.reader = VideoReader(filepath)
        self.queue_size = queue_size
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.error: Optional[Exception] = None
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    @property
    def info(self) -> VideoInfo:
        return self.reader.info

    @property
    def out_width(self) -> int:
        return self.reader.out_width

    @property
    def out_height(self) -> int:
        return self.reader.out_height

    def _worker(self):
        try:
            for frame in self.reader:
                if self.stop_event.is_set():
                    break
                self.queue.put(frame)
        except Exception as e:
            self.error = e
        finally:
            self.queue.put(None)  # End sentinel

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while True:
            frame = self.queue.get()
            if frame is None:
                if self.error:
                    raise self.error
                break
            yield frame

    def close(self):
        self.stop_event.set()
        self.reader.close()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

class VideoWriter:
    """
    Pipes processed raw BGR24 frames into ffmpeg encoder.
    Applies H.264 / HEVC encoding with CRF quality and preserves color metadata & audio.
    Supports Intel Quick Sync Video (QSV) hardware encoding with automatic fallback.
    """
    def __init__(
        self,
        output_path: str,
        input_info: VideoInfo,
        width: int,
        height: int,
        fps: float,
        codec: str = "hevc",
        crf: int = 18,
        preset: str = "medium",
        preserve_color_tags: bool = True,
        use_hardware_accel: bool = True
    ):
        self.output_path = str(output_path)
        self.input_info = input_info
        self.width = width
        self.height = height
        self.fps = fps
        
        QSV_PRESET_MAP = {
            "ultrafast": "veryfast",
            "superfast": "veryfast",
            "veryfast": "veryfast",
            "faster": "faster",
            "fast": "fast",
            "medium": "medium",
            "slow": "slow",
            "slower": "slower",
            "veryslow": "veryslow",
        }

        ffmpeg_bin = get_ffmpeg_cmd()

        def build_cmd(use_qsv: bool):
            if use_qsv:
                enc = "hevc_qsv" if codec.lower() in ("hevc", "h265", "x265") else "h264_qsv"
                qsv_p = QSV_PRESET_MAP.get(preset.lower(), "medium")
                enc_args = ["-c:v", enc, "-global_quality", str(crf), "-preset", qsv_p, "-pix_fmt", "nv12"]
            else:
                enc = "libx265" if codec.lower() in ("hevc", "h265", "x265") else "libx264"
                enc_args = ["-c:v", enc, "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]

            c = [
                ffmpeg_bin,
                "-y",
                "-v", "error",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", f"{fps:.4f}",
                "-i", "pipe:0"
            ]
            if input_info.has_audio:
                c.extend(["-i", input_info.filepath, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"])
            else:
                c.extend(["-map", "0:v:0"])
            c.extend(enc_args)
            if preserve_color_tags and input_info.color_metadata:
                c.extend(input_info.color_metadata.to_ffmpeg_args())
            c.append(self.output_path)
            return c

        self.process = None
        if use_hardware_accel and supports_qsv():
            try:
                cmd_qsv = build_cmd(use_qsv=True)
                proc = subprocess.Popen(
                    cmd_qsv,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **get_subprocess_kwargs()
                )
                # Quick healthcheck
                time.sleep(0.05)
                if proc.poll() is None:
                    self.process = proc
                else:
                    proc.wait()
            except Exception:
                pass

        if self.process is None:
            cmd_cpu = build_cmd(use_qsv=False)
            try:
                self.process = subprocess.Popen(
                    cmd_cpu,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **get_subprocess_kwargs()
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"ffmpeg executable not found ('{ffmpeg_bin}'). "
                    f"Please ensure FFmpeg is installed and added to PATH or placed in the bin/ directory."
                )

    def write_frame(self, frame: np.ndarray):
        assert frame.shape[0] == self.height and frame.shape[1] == self.width, "Frame dimensions mismatch"
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process:
            self.process.stdin.close()
            stderr_output = self.process.stderr.read()
            self.process.wait()
            if self.process.returncode != 0:
                raise RuntimeError(f"ffmpeg encoding failed with code {self.process.returncode}: {stderr_output.decode('utf-8', errors='ignore')}")
