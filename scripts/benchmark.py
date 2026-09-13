import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from face_mosaic.detector import SCRFDDetector


def benchmark():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models = [
        "scrfd_2.5g_bnkps.onnx",
        "scrfd_10g_bnkps.onnx",
        "scrfd_34g_gnkps.onnx",
    ]

    # Standard 4K frame resized to detection size (640x640)
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    num_iterations = 50

    print("==========================================================")
    print("      SCRFD ONNX Model Benchmark on Apple Silicon        ")
    print("==========================================================")
    print(f"Test frame resolution: 1920x1080 -> 640x640 detection input")
    print(f"Iterations: {num_iterations} frames per model\n")

    for model_name in models:
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"[-] {model_name}: not found (skipped)")
            continue

        try:
            detector = SCRFDDetector(str(model_path))
            # Warm-up
            for _ in range(5):
                _ = detector.detect(test_img)

            t0 = time.time()
            for _ in range(num_iterations):
                _ = detector.detect(test_img)
            t1 = time.time()

            elapsed = t1 - t0
            fps = num_iterations / elapsed
            latency_ms = (elapsed / num_iterations) * 1000

            print(f"[+] {model_name:24s} | {fps:6.1f} FPS | {latency_ms:5.1f} ms / frame")
        except Exception as e:
            print(f"[!] {model_name}: Error {e}")

    print("==========================================================\n")

if __name__ == "__main__":
    benchmark()
