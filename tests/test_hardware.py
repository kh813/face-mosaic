import pytest
from face_mosaic.hardware import detect_hardware_profile, get_cpu_name, get_total_ram_gb

def test_detect_hardware_profile():
    profile = detect_hardware_profile()
    assert profile.cpu_name != ""
    assert profile.logical_cores >= 1
    assert profile.ram_gb > 0
    assert profile.recommended_model in ("scrfd_34g_gnkps.onnx", "scrfd_10g_bnkps.onnx", "scrfd_2.5g_bnkps.onnx")
    assert profile.reason != ""

def test_high_spec_machine_recommendation():
    # If 16GB+ RAM and GPU available, should recommend 34G model
    profile = detect_hardware_profile()
    if len(profile.gpu_providers) > 0 and profile.ram_gb >= 15.0:
        assert profile.recommended_model == "scrfd_34g_gnkps.onnx"
        assert profile.is_high_spec is True
