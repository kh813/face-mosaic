import pytest
import tempfile
import time
from pathlib import Path
from face_mosaic.watcher import FolderWatcher
from face_mosaic.config import AppConfig

def test_folder_watcher_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        watcher = FolderWatcher(tmp_dir, poll_interval_seconds=0.1)
        assert watcher.watch_dir == Path(tmp_dir).resolve()
        assert len(watcher.processed_files) == 0

def test_folder_watcher_scan_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "dummy.mp4"
        p.write_bytes(b"dummy_data_1234567890")
        
        watcher = FolderWatcher(tmp_dir, poll_interval_seconds=0.1)
        # Check that existing file is registered in initial set
        assert str(p.resolve()) in watcher.processed_files
