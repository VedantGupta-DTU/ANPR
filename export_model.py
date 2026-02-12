from ultralytics import YOLO
import os

def export_model(model_path, format='onnx'):
    """
    Export a YOLOv8 model to a different format (e.g., ONNX, TFLite).
    Adaptive to dynamic size (dynamic=True).
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)
    
    print(f"Exporting to {format} (dynamic=True)...")
    # Export the model
    # dynamic=True allows for variable image sizes (useful for different camera resolutions)
    # simplify=True simplifies the graph for better compatibility
    path = model.export(format=format, dynamic=True, simplify=True, opset=17)
    
    print(f"Export complete! Saved to: {path}")
    print("\nTo use this model on Raspberry Pi:")
    print(f"1. Copy '{path}' to your Pi.")
    print("2. Install onnxruntime: pip install onnxruntime")
    print(f"3. Update your detector to load '{os.path.basename(path)}'")

if __name__ == "__main__":
    # Export the currently used 'best.pt'
    MODEL_PATH = "best.pt"
    export_model(MODEL_PATH)
