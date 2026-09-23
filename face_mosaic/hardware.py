"""
Hardware diagnostic and optimal model auto-selection module for face-mosaic.
Detects CPU architecture, total RAM, and available GPU/NPU acceleration providers
(DirectML, CoreML, CUDA) to choose the best SCRFD model.
"""

from dataclasses import dataclass
from typing import Optional, List
import os
import sys
import platform
import subprocess

@dataclass
class HardwareProfile:
    cpu_name: str
    logical_cores: int
    ram_gb: float
    gpu_providers: List[str]
    is_high_spec: bool
    recommended_model: str
    reason: str

def get_cpu_name() -> str:
    """Get human-readable CPU model string."""
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(val).strip()
        elif sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            return out
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"

def get_total_ram_gb() -> float:
    """Get total physical RAM in gigabytes."""
    try:
        if sys.platform == "win32":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        elif sys.platform == "darwin" or sys.platform.startswith("linux"):
            bytes_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            return bytes_ram / (1024 ** 3)
    except Exception:
        pass
    return 8.0  # Fallback assumption

def get_available_gpu_providers() -> List[str]:
    """Get active accelerated execution providers (OpenVINO, DirectML, CoreML, CUDA)."""
    providers = []
    try:
        import openvino as ov
        core = ov.Core()
        devs = core.available_devices
        if "GPU" in devs and "NPU" in devs:
            providers.append("OpenVINO (GPU+NPU)")
        elif "GPU" in devs:
            providers.append("OpenVINO (GPU)")
        elif "NPU" in devs:
            providers.append("OpenVINO (NPU)")
    except Exception:
        pass

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        accelerators = ["CoreMLExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider"]
        for p in accelerators:
            if p in available and p not in providers:
                providers.append(p)
    except Exception:
        pass
    return providers

def detect_hardware_profile() -> HardwareProfile:
    """
    Analyze system specs and determine the optimal SCRFD model.
    Criteria:
    - High-spec: GPU acceleration available (DirectML / CoreML / CUDA) AND (RAM >= 15GB OR Cores >= 8)
      -> scrfd_34g_gnkps.onnx (Maximum Accuracy)
    - Mid-spec: GPU acceleration available OR (RAM >= 12GB AND Cores >= 6)
      -> scrfd_10g_bnkps.onnx (Balanced Default)
    - Low-spec: CPU-only and (RAM < 8GB OR Cores <= 4)
      -> scrfd_2.5g_bnkps.onnx (Fastest)
    """
    cpu = get_cpu_name()
    cores = os.cpu_count() or 4
    ram = get_total_ram_gb()
    gpus = get_available_gpu_providers()

    has_gpu = len(gpus) > 0

    if has_gpu and (ram >= 15.0 or cores >= 8):
        rec_model = "scrfd_34g_gnkps.onnx"
        provider_name = gpus[0].replace("ExecutionProvider", "")
        reason = f"High-performance system detected ({cpu}, {ram:.1f}GB RAM, {provider_name} GPU). Auto-selected 34G (High Accuracy)."
        is_high = True
    elif has_gpu or (ram >= 12.0 and cores >= 6):
        rec_model = "scrfd_10g_bnkps.onnx"
        reason = f"Mid-range system detected ({cpu}, {ram:.1f}GB RAM). Auto-selected 10G (Balanced Default)."
        is_high = False
    else:
        rec_model = "scrfd_2.5g_bnkps.onnx"
        reason = f"Resource-constrained environment ({cpu}, {ram:.1f}GB RAM, CPU-only). Auto-selected 2.5G (Fast)."
        is_high = False

    return HardwareProfile(
        cpu_name=cpu,
        logical_cores=cores,
        ram_gb=ram,
        gpu_providers=gpus,
        is_high_spec=is_high,
        recommended_model=rec_model,
        reason=reason
    )
