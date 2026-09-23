import pytest
import sys
from unittest.mock import patch
from face_mosaic.hardware import detect_hardware_profile, get_cpu_name, get_total_ram_gb, get_hardware_vendor
from face_mosaic.io_utils import (
    supports_qsv,
    supports_videotoolbox,
    supports_amf,
    supports_nvenc,
    get_supported_encoders
)

def test_detect_hardware_profile():
    profile = detect_hardware_profile()
    assert profile.cpu_name != ""
    assert profile.logical_cores >= 1
    assert profile.ram_gb > 0
    assert profile.recommended_model in ("scrfd_34g_gnkps.onnx", "scrfd_10g_bnkps.onnx", "scrfd_2.5g_bnkps.onnx")
    assert profile.reason != ""
    assert profile.hardware_vendor in ("apple_silicon", "amd_ryzen", "intel", "generic")

def test_high_spec_machine_recommendation():
    # If 16GB+ RAM and GPU available, should recommend 34G model
    profile = detect_hardware_profile()
    if len(profile.gpu_providers) > 0 and profile.ram_gb >= 15.0:
        assert profile.recommended_model == "scrfd_34g_gnkps.onnx"
        assert profile.is_high_spec is True

def test_apple_silicon_hardware_profile_mock():
    with patch("face_mosaic.hardware.get_hardware_vendor", return_value="apple_silicon"), \
         patch("face_mosaic.hardware.get_total_ram_gb", return_value=16.0), \
         patch("face_mosaic.hardware.get_cpu_name", return_value="Apple M3 Max"):
        profile = detect_hardware_profile()
        assert profile.hardware_vendor == "apple_silicon"
        assert profile.recommended_model == "scrfd_34g_gnkps.onnx"
        assert "Apple Silicon" in profile.reason
        assert "VideoToolbox" in profile.reason

def test_amd_ryzen_hardware_profile_mock():
    with patch("face_mosaic.hardware.get_hardware_vendor", return_value="amd_ryzen"), \
         patch("face_mosaic.hardware.get_total_ram_gb", return_value=32.0), \
         patch("face_mosaic.hardware.get_available_gpu_providers", return_value=["DmlExecutionProvider"]), \
         patch("face_mosaic.hardware.get_cpu_name", return_value="AMD Ryzen 9 7950X"):
        profile = detect_hardware_profile()
        assert profile.hardware_vendor == "amd_ryzen"
        assert profile.recommended_model == "scrfd_34g_gnkps.onnx"
        assert "AMD Ryzen" in profile.reason

def test_supported_encoders_detection():
    encoders = get_supported_encoders()
    assert isinstance(encoders, set)
    # Functions should return boolean
    assert isinstance(supports_qsv(), bool)
    assert isinstance(supports_videotoolbox(), bool)
    assert isinstance(supports_amf(), bool)
    assert isinstance(supports_nvenc(), bool)
