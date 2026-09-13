"""
face_mosaic GUI application using PySide6 (Qt for Python).
Features:
- Video file / folder selection
- Real-time parameter tuning (Blur type, Strength, Confidence, Padding)
- Multi-threaded processing (UI remains responsive)
- Live progress bars and status logs
"""

import sys
import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
    QCheckBox, QMessageBox
)

from face_mosaic.config import AppConfig
from face_mosaic.pipeline import ProcessingPipeline
from face_mosaic.cli import collect_video_files

class ProcessingWorker(QThread):
    progress_signal = Signal(str, int, int) # (phase, current, total)
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, config: AppConfig, video_files: List[Path]):
        super().__init__()
        self.config = config
        self.video_files = video_files
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            pipeline = ProcessingPipeline(self.config)
            self.log_signal.emit(f"Initialized pipeline with model: {self.config.model.name} on {pipeline.detector.active_provider}")

            for idx, video_path in enumerate(self.video_files, 1):
                if self._is_cancelled:
                    self.log_signal.emit("Processing cancelled by user.")
                    self.finished_signal.emit(False, "Cancelled")
                    return

                self.log_signal.emit(f"\n[{idx}/{len(self.video_files)}] Starting: {video_path.name}")

                def on_progress(phase: str, current: int, total: int, elapsed: float):
                    self.progress_signal.emit(f"[{video_path.name}] {phase}", current, total)

                try:
                    res = pipeline.process_video(str(video_path), progress_callback=on_progress)
                    self.log_signal.emit(
                        f"✓ Finished: {video_path.name} in {res['elapsed_seconds']:.1f}s "
                        f"({res['processing_fps']:.1f} fps, Color MAE: {res['color_mae']:.4f})"
                    )
                except Exception as e:
                    self.log_signal.emit(f"✗ Failed {video_path.name}: {e}")

            self.finished_signal.emit(True, "All tasks completed successfully.")
        except Exception as e:
            self.log_signal.emit(f"Fatal Error: {e}")
            self.finished_signal.emit(False, str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("face-mosaic: Automated Face Blur Tool")
        self.resize(850, 680)

        self.config = AppConfig()
        self.selected_files: List[Path] = []
        self.worker: Optional[ProcessingWorker] = None

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)

        # 1. File Input Group
        file_group = QGroupBox("1. Select Videos")
        file_layout = QVBoxLayout(file_group)
        btn_layout = QHBoxLayout()

        self.btn_select_files = QPushButton("Add Video File(s)...")
        self.btn_select_files.clicked.connect(self._select_files)
        self.btn_select_folder = QPushButton("Add Folder...")
        self.btn_select_folder.clicked.connect(self._select_folder)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_files)

        btn_layout.addWidget(self.btn_select_files)
        btn_layout.addWidget(self.btn_select_folder)
        btn_layout.addWidget(self.btn_clear)
        file_layout.addLayout(btn_layout)

        self.lbl_file_count = QLabel("No videos selected.")
        file_layout.addWidget(self.lbl_file_count)
        main_layout.addWidget(file_group)

        # 2. Parameters Group
        param_group = QGroupBox("2. Settings & Parameters")
        grid = QGridLayout(param_group)

        # Model
        grid.addWidget(QLabel("Model:"), 0, 0)
        self.combo_model = QComboBox()
        self.combo_model.addItems(["scrfd_10g_bnkps.onnx (Default)", "scrfd_2.5g_bnkps.onnx (Fast)", "scrfd_34g_gnkps.onnx (High Acc)"])
        grid.addWidget(self.combo_model, 0, 1)

        # Blur Type
        grid.addWidget(QLabel("Blur Type:"), 0, 2)
        self.combo_blur_type = QComboBox()
        self.combo_blur_type.addItems(["Gaussian", "Mosaic"])
        self.combo_blur_type.currentTextChanged.connect(self._on_blur_type_changed)
        grid.addWidget(self.combo_blur_type, 0, 3)

        # Blur Strength / Block Size
        self.lbl_strength = QLabel("Blur Strength:")
        grid.addWidget(self.lbl_strength, 1, 0)
        self.spin_strength = QSpinBox()
        self.spin_strength.setRange(3, 199)
        self.spin_strength.setSingleStep(2)
        self.spin_strength.setValue(31)
        grid.addWidget(self.spin_strength, 1, 1)

        # Confidence Threshold
        grid.addWidget(QLabel("Detection Confidence:"), 1, 2)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.50)
        grid.addWidget(self.spin_conf, 1, 3)

        # Padding Backward (N)
        grid.addWidget(QLabel("Pad Backward (frames):"), 2, 0)
        self.spin_pad_back = QSpinBox()
        self.spin_pad_back.setRange(0, 30)
        self.spin_pad_back.setValue(3)
        grid.addWidget(self.spin_pad_back, 2, 1)

        # Padding Forward (M)
        grid.addWidget(QLabel("Pad Forward (frames):"), 2, 2)
        self.spin_pad_fwd = QSpinBox()
        self.spin_pad_fwd.setRange(0, 30)
        self.spin_pad_fwd.setValue(3)
        grid.addWidget(self.spin_pad_fwd, 2, 3)

        # Codec & CRF
        grid.addWidget(QLabel("Output Codec:"), 3, 0)
        self.combo_codec = QComboBox()
        self.combo_codec.addItems(["HEVC (H.265)", "H.264"])
        grid.addWidget(self.combo_codec, 3, 1)

        grid.addWidget(QLabel("CRF Quality:"), 3, 2)
        self.spin_crf = QSpinBox()
        self.spin_crf.setRange(0, 51)
        self.spin_crf.setValue(18)
        grid.addWidget(self.spin_crf, 3, 3)

        main_layout.addWidget(param_group)

        # 3. Execution & Progress
        exec_group = QGroupBox("3. Execution & Log")
        exec_layout = QVBoxLayout(exec_group)

        self.btn_start = QPushButton("Start Processing")
        self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; height: 36px; background-color: #007AFF; color: white;")
        self.btn_start.clicked.connect(self._toggle_processing)
        exec_layout.addWidget(self.btn_start)

        self.lbl_progress_status = QLabel("Idle")
        exec_layout.addWidget(self.lbl_progress_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        exec_layout.addWidget(self.progress_bar)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        exec_layout.addWidget(self.txt_log)

        main_layout.addWidget(exec_group)

    def _on_blur_type_changed(self, text: str):
        if text.lower() == "mosaic":
            self.lbl_strength.setText("Mosaic Block Size:")
            self.spin_strength.setValue(16)
        else:
            self.lbl_strength.setText("Blur Strength (odd):")
            self.spin_strength.setValue(31)

    def _select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", "Video Files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm)"
        )
        if paths:
            self.selected_files.extend(collect_video_files(paths))
            self._update_file_count()

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Videos")
        if folder:
            self.selected_files.extend(collect_video_files([folder]))
            self._update_file_count()

    def _clear_files(self):
        self.selected_files.clear()
        self._update_file_count()

    def _update_file_count(self):
        # Remove duplicates
        seen = set()
        unique = []
        for f in self.selected_files:
            if str(f) not in seen:
                seen.add(str(f))
                unique.append(f)
        self.selected_files = unique
        self.lbl_file_count.setText(f"{len(self.selected_files)} video file(s) selected.")

    def _toggle_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.btn_start.setText("Start Processing")
            self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; height: 36px; background-color: #007AFF; color: white;")
            return

        if not self.selected_files:
            QMessageBox.warning(self, "No Videos", "Please select at least one video file or folder first.")
            return

        # Build config
        cfg = AppConfig()
        model_selection = self.combo_model.currentText().split()[0]
        cfg.model.name = model_selection
        cfg.model.conf_threshold = self.spin_conf.value()

        cfg.blur.type = self.combo_blur_type.currentText().lower()
        if cfg.blur.type == "mosaic":
            cfg.blur.mosaic_block_size = self.spin_strength.value()
        else:
            val = self.spin_strength.value()
            cfg.blur.strength = val if val % 2 == 1 else val + 1

        cfg.tracking.padding_backward = self.spin_pad_back.value()
        cfg.tracking.padding_forward = self.spin_pad_fwd.value()

        codec_str = "hevc" if "HEVC" in self.combo_codec.currentText() else "h264"
        cfg.output.codec = codec_str
        cfg.output.crf = self.spin_crf.value()

        self.btn_start.setText("Stop / Cancel Processing")
        self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; height: 36px; background-color: #FF3B30; color: white;")

        self.worker = ProcessingWorker(cfg, self.selected_files)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    @Slot(str, int, int)
    def _on_progress(self, phase_str: str, current: int, total: int):
        self.lbl_progress_status.setText(f"{phase_str}: {current} / {total}")
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))

    @Slot(str)
    def _on_log(self, text: str):
        self.txt_log.append(text)

    @Slot(bool, str)
    def _on_finished(self, success: bool, msg: str):
        self.btn_start.setText("Start Processing")
        self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; height: 36px; background-color: #007AFF; color: white;")
        self.lbl_progress_status.setText(msg)
        if success:
            QMessageBox.information(self, "Completed", "All videos have been processed successfully!")

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
