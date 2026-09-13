import pytest
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication
from face_mosaic.gui.app import MainWindow

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_gui_window_initialization(qapp):
    window = MainWindow()
    assert window.windowTitle() == "face-mosaic: Automated Face Blur Tool"
    assert window.combo_model.count() >= 3
    assert window.spin_pad_back.value() == 3
    assert window.spin_pad_fwd.value() == 3
    assert window.spin_crf.value() == 18

def test_gui_blur_type_switch(qapp):
    window = MainWindow()
    window.combo_blur_type.setCurrentText("Mosaic")
    assert window.lbl_strength.text() == "Mosaic Block Size:"
    window.combo_blur_type.setCurrentText("Gaussian")
    assert window.lbl_strength.text() == "Blur Strength (odd):"
