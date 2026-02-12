from ultralytics import YOLO
import time
import cv2
import numpy as np

def benchmark_model(model_path, iterations=50):
    print(f"Loading model: {model_path}...")
    try:
        model = YOLO(model_path, task="detect")
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None

    # Dummy image (640x640)
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    print("Warming up...")
    for _ in range(10):
        model(img, verbose=False)

    print(f"Benchmarking ({iterations} iterations)...")
    start_time = time.time()
    for _ in range(iterations):
        model(img, verbose=False)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    fps = 1.0 / avg_time
    print(f"Model: {model_path} | Avg Latency: {avg_time*1000:.2f} ms | FPS: {fps:.2f}")
    return avg_time

if __name__ == "__main__":
    print("--- PyTorch (.pt) Benchmark ---")
    pt_time = benchmark_model("best.pt")

    print("\n--- ONNX (.onnx) Benchmark ---")
    onnx_time = benchmark_model("best.onnx")

    if pt_time and onnx_time:
        speedup = pt_time / onnx_time
        print(f"\nSummary: ONNX is {speedup:.2f}x faster on this machine.")
