"""
Color space management, metadata extraction, and color difference verification.
Implements BT.2020/BT.709 color metadata preservation and Delta-E / MAE evaluation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import cv2

@dataclass
class ColorMetadata:
    color_space: Optional[str] = None
    color_transfer: Optional[str] = None
    color_primaries: Optional[str] = None
    color_range: Optional[str] = None
    pix_fmt: Optional[str] = None

    def to_ffmpeg_args(self) -> list:
        """
        Generate ffmpeg output arguments to strictly preserve static color tags.
        """
        args = []
        if self.color_space:
            args.extend(["-colorspace", self.color_space])
        if self.color_transfer:
            args.extend(["-color_trc", self.color_transfer])
        if self.color_primaries:
            args.extend(["-color_primaries", self.color_primaries])
        if self.color_range:
            args.extend(["-color_range", self.color_range])
        return args

    def to_bsf_args(self, codec: str = "hevc") -> list:
        """
        Generate HEVC bitstream filter arguments to embed VUI colour metadata
        directly into the HEVC NAL units (essential for QSV / hardware encoders).
        """
        if codec.lower() not in ("hevc", "h265", "x265"):
            return []

        PRIMARIES_MAP = {
            "bt709": 1,
            "bt2020": 9,
        }
        TRANSFER_MAP = {
            "bt709": 1,
            "smpte170m": 6,
            "smpte2084": 16,
            "arib-std-b67": 18,
        }
        MATRIX_MAP = {
            "bt709": 1,
            "smpte170m": 6,
            "bt2020nc": 9,
            "bt2020_ncl": 9,
            "bt2020c": 10,
            "bt2020_cl": 10,
        }

        items = []
        if self.color_primaries and self.color_primaries.lower() in PRIMARIES_MAP:
            items.append(f"colour_primaries={PRIMARIES_MAP[self.color_primaries.lower()]}")
        if self.color_transfer and self.color_transfer.lower() in TRANSFER_MAP:
            items.append(f"transfer_characteristics={TRANSFER_MAP[self.color_transfer.lower()]}")
        if self.color_space and self.color_space.lower() in MATRIX_MAP:
            items.append(f"matrix_coefficients={MATRIX_MAP[self.color_space.lower()]}")

        if items:
            return ["-bsf:v", f"hevc_metadata={':'.join(items)}"]
        return []

def parse_color_metadata(stream_info: Dict[str, Any]) -> ColorMetadata:
    """
    Extract color metadata from ffprobe stream dictionary.
    """
    return ColorMetadata(
        color_space=stream_info.get("color_space"),
        color_transfer=stream_info.get("color_transfer"),
        color_primaries=stream_info.get("color_primaries"),
        color_range=stream_info.get("color_range"),
        pix_fmt=stream_info.get("pix_fmt")
    )

def compute_color_difference(
    frame_orig: np.ndarray,
    frame_proc: np.ndarray,
    ignore_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute mean absolute error (MAE) and Delta-E (in Lab space) between two frames,
    ignoring blurred regions if ignore_mask is provided.
    
    ignore_mask: 2D uint8 array where 255 indicates blurred region to ignore, 0 indicates evaluate.
    """
    assert frame_orig.shape == frame_proc.shape, "Frames must have the same shape"
    
    if ignore_mask is not None:
        valid_mask = (ignore_mask == 0)
        if not np.any(valid_mask):
            return {"mae": 0.0, "delta_e_approx": 0.0, "max_error": 0.0}
        orig_pixels = frame_orig[valid_mask]
        proc_pixels = frame_proc[valid_mask]
    else:
        orig_pixels = frame_orig.reshape(-1, 3)
        proc_pixels = frame_proc.reshape(-1, 3)
        
    diff = np.abs(orig_pixels.astype(np.float32) - proc_pixels.astype(np.float32))
    mae = float(np.mean(diff))
    max_error = float(np.max(diff))
    
    # Approximate Delta-E in Lab space
    orig_bgr = orig_pixels.reshape(-1, 1, 3).astype(np.uint8)
    proc_bgr = proc_pixels.reshape(-1, 1, 3).astype(np.uint8)
    
    orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    proc_lab = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    delta_e = float(np.mean(np.sqrt(np.sum((orig_lab - proc_lab) ** 2, axis=2))))
    
    return {
        "mae": mae,
        "delta_e_approx": delta_e,
        "max_error": max_error
    }
