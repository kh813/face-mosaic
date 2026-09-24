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
import cv2

from .color import ColorMetadata, parse_color_metadata

def get_subprocess_kwargs() -> Dict[str, Any]:
    """
    Platform-specific subprocess kwargs to prevent console (cmd.exe)
    windows from popping up when spawning ffmpeg/ffprobe on Windows.
    Safely resolves Windows-specific attributes when running on non-Windows platforms.
    """
    kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        # 0x08000000 is CREATE_NO_WINDOW on Windows
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        startupinfo_cls = getattr(subprocess, "STARTUPINFO", None)
        if startupinfo_cls is not None:
            si = startupinfo_cls()
            flags = getattr(subprocess, "STARTF_USESHOWWINDOW", 1)
            si.dwFlags |= flags
            si.wShowWindow = 0  # SW_HIDE
            kwargs["startupinfo"] = si
    return kwargs

_supported_encoders: Optional[set] = None

def get_supported_encoders() -> set:
    """Query available ffmpeg encoders once and cache."""
    global _supported_encoders
    if _supported_encoders is not None:
        return _supported_encoders
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
        _supported_encoders = set()
        for line in res.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith("V"):
                _supported_encoders.add(parts[1])
    except Exception:
        _supported_encoders = set()
    return _supported_encoders

def supports_qsv() -> bool:
    """Check if Intel Quick Sync Video (QSV) is supported by ffmpeg."""
    return "hevc_qsv" in get_supported_encoders()

def supports_videotoolbox() -> bool:
    """Check if Apple VideoToolbox (Apple Silicon / Mac) is supported by ffmpeg."""
    return "hevc_videotoolbox" in get_supported_encoders()

def supports_amf() -> bool:
    """Check if AMD Advanced Media Framework (AMF / Ryzen APU & Radeon) is supported by ffmpeg."""
    return "hevc_amf" in get_supported_encoders()

def supports_nvenc() -> bool:
    """Check if NVIDIA NVENC is supported by ffmpeg."""
    return "hevc_nvenc" in get_supported_encoders()

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
    Reads frames as raw BGR24 arrays.
    When scale is None, utilizes cv2.VideoCapture for native in-process hardware accelerated
    decoding (Media Foundation / D3D11 / FFmpeg), bypassing Windows pipe context-switch limits.
    When scale is specified (e.g. Pass 1), uses multithreaded FFmpeg subprocess with -vf scale.
    """
    def __init__(self, filepath: str, scale: Optional[Tuple[int, int]] = None):
        self.filepath = str(filepath)
        self.info = probe_video(filepath)
        self.scale = scale
        
        # When autorotate filter is applied by ffmpeg, 90/270 degree rotation swaps width and height
        self.needs_swap = abs(self.info.rotation) in (90, 270)
        self.orig_width = self.info.height if self.needs_swap else self.info.width
        self.orig_height = self.info.width if self.needs_swap else self.info.height

        if scale is not None:
            self.out_width, self.out_height = scale
        else:
            self.out_width = self.orig_width
            self.out_height = self.orig_height
        
        self.cap = None
        self.process = None

        # Full resolution decoding (Pass 2): native hardware acceleration via cv2.VideoCapture
        if scale is None:
            try:
                cap = cv2.VideoCapture(self.filepath)
                if cap.isOpened():
                    self.cap = cap
            except Exception:
                self.cap = None

        if self.cap is None:
            # Multithreaded FFmpeg decode pipe (supports arbitrary scaling filters)
            ffmpeg_bin = get_ffmpeg_cmd()
            cmd = [
                ffmpeg_bin,
                "-v", "error",
                "-threads", "0",
                "-i", self.filepath,
            ]
            if scale is not None:
                cmd.extend(["-vf", f"scale={self.out_width}:{self.out_height}"])
            cmd.extend([
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "pipe:1"
            ])
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
        if self.cap is not None:
            while self.cap is not None and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                except Exception:
                    break
                if not ret or frame is None:
                    break
                yield frame
        else:
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
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.process:
            self.process.stdout.close()
            self.process.wait()
            self.process = None

class PrefetchedVideoReader:
    """
    Wraps VideoReader with a background worker thread and queue.
    Prefetches decoded frames so GPU inference does not block on FFmpeg I/O.
    """
    def __init__(self, filepath: str, queue_size: int = 16, scale: Optional[Tuple[int, int]] = None):
        self.reader = VideoReader(filepath, scale=scale)
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
    def orig_width(self) -> int:
        return self.reader.orig_width

    @property
    def orig_height(self) -> int:
        return self.reader.orig_height

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
        # Unblock worker thread if it is waiting on queue.put
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)
        self.reader.close()

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
        crf: int = 23,
        preset: str = "medium",
        preserve_color_tags: bool = True,
        use_hardware_accel: bool = True,
        bit_depth: str = "auto"
    ):
        self.output_path = str(output_path)
        self.input_info = input_info
        self.width = width
        self.height = height
        self.fps = fps
        self.bit_depth = bit_depth
        
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

        def build_cmd(hw_mode: str):
            """
            hw_mode options:
             - 'videotoolbox': Apple Silicon / macOS hardware encoder
             - 'qsv': Intel Quick Sync Video (Intel Arc / Core Ultra)
             - 'amf': AMD Advanced Media Framework (Ryzen APU / Radeon)
             - 'nvenc': NVIDIA NVENC
             - 'cpu': libx265 / libx264 software encoder (multi-threaded)
            """
            is_hevc = codec.lower() in ("hevc", "h265", "x265")
            input_is_10bit = bool(input_info.color_metadata and input_info.color_metadata.pix_fmt in ("yuv420p10le", "p010le", "p010"))
            
            if bit_depth.lower() == "8bit":
                is_10bit = False
            elif bit_depth.lower() == "10bit":
                is_10bit = True
            else:  # "auto"
                # Keep 10-bit for HEVC if input was 10-bit; for H.264 default to 8-bit for maximum compatibility
                is_10bit = input_is_10bit and is_hevc

            if hw_mode == "videotoolbox":
                enc = "hevc_videotoolbox" if is_hevc else "h264_videotoolbox"
                # VideoToolbox quality: scale 1-100 (65 is sweet spot corresponding to crf 18-20)
                vt_q = str(max(40, min(90, 85 - (crf - 15) * 2)))
                if is_10bit and is_hevc:
                    pix = "p010le"
                    enc_args = ["-c:v", enc, "-profile:v", "main10", "-q:v", vt_q, "-pix_fmt", pix]
                else:
                    pix = "nv12"
                    enc_args = ["-c:v", enc, "-q:v", vt_q, "-pix_fmt", pix]
            elif hw_mode == "qsv":
                enc = "hevc_qsv" if is_hevc else "h264_qsv"
                qsv_p = QSV_PRESET_MAP.get(preset.lower(), "medium")
                pix = "p010le" if (is_10bit and is_hevc) else "nv12"
                enc_args = ["-c:v", enc, "-global_quality", str(crf), "-preset", qsv_p, "-pix_fmt", pix]
            elif hw_mode == "amf":
                enc = "hevc_amf" if is_hevc else "h264_amf"
                pix = "p010le" if (is_10bit and is_hevc) else "nv12"
                enc_args = ["-c:v", enc, "-quality", "quality", "-rc", "cqp", "-qp_p", str(crf), "-qp_i", str(crf), "-pix_fmt", pix]
            elif hw_mode == "nvenc":
                enc = "hevc_nvenc" if is_hevc else "h264_nvenc"
                pix = "p010le" if (is_10bit and is_hevc) else "nv12"
                enc_args = ["-c:v", enc, "-preset", "p5", "-cq", str(crf), "-pix_fmt", pix]
            else:
                # CPU software encode: heavily utilize multi-core threads (Ryzen, etc.)
                enc = "libx265" if is_hevc else "libx264"
                pix = "yuv420p10le" if (is_10bit and is_hevc) else "yuv420p"
                enc_args = ["-c:v", enc, "-crf", str(crf), "-preset", preset, "-pix_fmt", pix, "-threads", "0"]

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
            c.extend(["-r", f"{fps:.4f}"])
            if preserve_color_tags and input_info.color_metadata:
                c.extend(input_info.color_metadata.to_ffmpeg_args())
                if is_hevc and is_10bit:
                    c.extend(input_info.color_metadata.to_bsf_args(codec))
            # Enable FastStart for MP4 / MOV containers for instant seeking & web playback
            if Path(self.output_path).suffix.lower() in (".mp4", ".mov", ".m4v"):
                c.extend(["-movflags", "+faststart"])
            c.append(self.output_path)
            return c

        self.process = None
        self.active_encoder = "cpu"

        # Determine target hardware encoder based on platform and detected capability
        if use_hardware_accel and width >= 640 and height >= 480:
            target_hw = None
            if sys.platform == "darwin" and supports_videotoolbox():
                target_hw = "videotoolbox"
            elif supports_qsv():
                target_hw = "qsv"
            elif supports_amf():
                target_hw = "amf"
            elif supports_nvenc():
                target_hw = "nvenc"

            if target_hw:
                try:
                    cmd_hw = build_cmd(hw_mode=target_hw)
                    proc = subprocess.Popen(
                        cmd_hw,
                        stdin=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        **get_subprocess_kwargs()
                    )
                    # Quick healthcheck
                    time.sleep(0.05)
                    if proc.poll() is None:
                        self.process = proc
                        self.active_encoder = target_hw
                    else:
                        proc.wait()
                except Exception:
                    pass

        if self.process is None:
            cmd_cpu = build_cmd(hw_mode="cpu")
            try:
                self.process = subprocess.Popen(
                    cmd_cpu,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **get_subprocess_kwargs()
                )
                self.active_encoder = "cpu"
            except FileNotFoundError:
                raise RuntimeError(
                    f"ffmpeg executable not found ('{ffmpeg_bin}'). "
                    f"Please ensure FFmpeg is installed and added to PATH or placed in the bin/ directory."
                )

        # Asynchronous background pipe writer thread (decouples rendering from stdin pipe)
        self.queue: queue.Queue = queue.Queue(maxsize=16)
        self.writer_error: Optional[Exception] = None
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def _writer_worker(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                if isinstance(item, np.ndarray):
                    self.process.stdin.write(item.tobytes())
                else:
                    self.process.stdin.write(item)
        except Exception as e:
            self.writer_error = e

    def write_frame(self, frame: np.ndarray):
        if self.writer_error:
            raise self.writer_error
        assert frame.shape[0] == self.height and frame.shape[1] == self.width, "Frame dimensions mismatch"
        self.queue.put(frame)

    def close(self):
        if self.process:
            self.queue.put(None)
            self.writer_thread.join()
            if self.writer_error:
                raise self.writer_error
            self.process.stdin.close()
            stderr_output = self.process.stderr.read()
            self.process.wait()
            if self.process.returncode != 0:
                raise RuntimeError(f"ffmpeg encoding failed with code {self.process.returncode}: {stderr_output.decode('utf-8', errors='ignore')}")
