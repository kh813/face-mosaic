"""
Video I/O and ffmpeg process wrapper module for face-mosaic.
Handles reading video frames via ffmpeg pipe (with rotation & color tags),
extracting audio, querying metadata with ffprobe, and writing processed frames.
"""

import json
import subprocess
import os
import sys
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
import numpy as np

from .color import ColorMetadata, parse_color_metadata

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
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-print_format", "json",
        str(filepath)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", self.filepath,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vsync", "0",
            "pipe:1"
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        self.frame_size = self.out_width * self.out_height * 3

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while True:
            raw_bytes = self.process.stdout.read(self.frame_size)
            if not raw_bytes or len(raw_bytes) < self.frame_size:
                break
            frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.out_height, self.out_width, 3))
            yield frame

    def close(self):
        if self.process:
            self.process.stdout.close()
            self.process.wait()

class VideoWriter:
    """
    Pipes processed raw BGR24 frames into ffmpeg encoder.
    Applies H.264 / HEVC encoding with CRF quality and preserves color metadata & audio.
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
        preserve_color_tags: bool = True
    ):
        self.output_path = str(output_path)
        self.input_info = input_info
        self.width = width
        self.height = height
        self.fps = fps
        
        # Determine encoder
        encoder = "libx265" if codec.lower() in ("hevc", "h265", "x265") else "libx264"
        
        cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps:.4f}",
            "-i", "pipe:0"  # raw frames from stdin
        ]
        
        # Include original audio if available
        if input_info.has_audio:
            cmd.extend([
                "-i", input_info.filepath,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "copy"  # copy original audio stream without re-encoding
            ])
        else:
            cmd.extend([
                "-map", "0:v:0"
            ])
            
        # Video encoding settings
        cmd.extend([
            "-c:v", encoder,
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p"  # Ensure universal compatibility
        ])
        
        # Color tags preservation
        if preserve_color_tags and input_info.color_metadata:
            color_args = input_info.color_metadata.to_ffmpeg_args()
            cmd.extend(color_args)
            
        cmd.append(self.output_path)
        
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

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
