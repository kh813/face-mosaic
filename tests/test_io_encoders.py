import pytest
from unittest.mock import MagicMock, patch
from face_mosaic.io_utils import VideoWriter, VideoInfo
from face_mosaic.color import ColorMetadata

def _create_mock_process():
    proc = MagicMock()
    proc.poll.return_value = None
    proc.returncode = 0
    proc.stderr.read.return_value = b""
    return proc

def test_videowriter_encoder_cmd_branches():
    mock_input_info = MagicMock(spec=VideoInfo)
    mock_input_info.filepath = "input.mp4"
    mock_input_info.has_audio = True
    mock_input_info.color_metadata = ColorMetadata(
        color_space="bt2020nc",
        color_primaries="bt2020",
        color_transfer="arib-std-b67",
        color_range="tv",
        pix_fmt="yuv420p10le"
    )

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=True), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=False), \
         patch("sys.platform", "darwin"):
        
        mock_popen.return_value = _create_mock_process()

        writer = VideoWriter(
            output_path="output_mac.mp4",
            input_info=mock_input_info,
            width=3840,
            height=2160,
            fps=60.0,
            codec="hevc",
            crf=18,
            use_hardware_accel=True
        )

        assert writer.active_encoder == "videotoolbox"
        called_cmd = mock_popen.call_args[0][0]
        assert "-c:v" in called_cmd
        enc_idx = called_cmd.index("-c:v")
        assert called_cmd[enc_idx + 1] == "hevc_videotoolbox"
        assert "-profile:v" in called_cmd
        assert "main10" in called_cmd
        writer.close()

def test_videowriter_qsv_cmd_branches():
    mock_input_info = MagicMock(spec=VideoInfo)
    mock_input_info.filepath = "input.mp4"
    mock_input_info.has_audio = False
    mock_input_info.color_metadata = None

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=False), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=True), \
         patch("sys.platform", "win32"):
        
        mock_popen.return_value = _create_mock_process()

        writer = VideoWriter(
            output_path="output_intel.mp4",
            input_info=mock_input_info,
            width=1920,
            height=1080,
            fps=30.0,
            codec="hevc",
            crf=18,
            use_hardware_accel=True
        )

        assert writer.active_encoder == "qsv"
        called_cmd = mock_popen.call_args[0][0]
        assert "hevc_qsv" in called_cmd
        writer.close()

def test_videowriter_cpu_threads_fallback():
    mock_input_info = MagicMock(spec=VideoInfo)
    mock_input_info.filepath = "input.mp4"
    mock_input_info.has_audio = False
    mock_input_info.color_metadata = None

    with patch("subprocess.Popen") as mock_popen, \
         patch("face_mosaic.io_utils.get_ffmpeg_cmd", return_value="ffmpeg"), \
         patch("face_mosaic.io_utils.supports_videotoolbox", return_value=False), \
         patch("face_mosaic.io_utils.supports_qsv", return_value=False), \
         patch("face_mosaic.io_utils.supports_amf", return_value=False), \
         patch("face_mosaic.io_utils.supports_nvenc", return_value=False):
        
        mock_popen.return_value = _create_mock_process()

        writer = VideoWriter(
            output_path="output_cpu.mp4",
            input_info=mock_input_info,
            width=1920,
            height=1080,
            fps=30.0,
            codec="hevc",
            crf=18,
            use_hardware_accel=True
        )

        assert writer.active_encoder == "cpu"
        called_cmd = mock_popen.call_args[0][0]
        assert "libx265" in called_cmd
        assert "-threads" in called_cmd
        assert "0" in called_cmd
        writer.close()
