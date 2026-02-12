"""
License Plate Recognition Pipeline
Combines YOLO detection with OCR for end-to-end plate recognition
Supports: DeepSeek OCR-2 (GPU) → PaddleOCR (CPU) → EasyOCR (CPU fallback)
"""
import os
import cv2
import json
import torch
from datetime import datetime
from typing import List, Optional
from plate_detector import PlateDetector
from ocr_reader import OCRReader, PaddleOCRReader, EasyOCRReader, EnsembleOCRReader
import config


class LicensePlateRecognizer:
    """End-to-end license plate recognition pipeline"""
    
    def __init__(self, ocr_engine: str = "auto", device: str = None):
        """
        Initialize the recognition pipeline
        
        Args:
            ocr_engine: OCR engine to use:
                'auto'     - DeepSeek if GPU, else PaddleOCR
                'paddle'   - PaddleOCR (best CPU accuracy)
                'easyocr'  - EasyOCR (lighter weight)
                'ensemble' - PaddleOCR + EasyOCR (picks best)
                'deepseek' - DeepSeek OCR-2 (requires GPU)
            device: Device for inference ('cuda' or 'cpu')
        """
        print("=" * 50)
        print("Initializing License Plate Recognition Pipeline")
        print("=" * 50)
        
        self.detector = PlateDetector()
        
        # Resolve 'auto' engine choice
        if ocr_engine == "auto":
            if torch.cuda.is_available():
                ocr_engine = "deepseek"
                print("\n[INFO] CUDA GPU detected → using DeepSeek OCR-2")
            else:
                ocr_engine = "paddle"
                print("\n[INFO] No CUDA GPU → using PaddleOCR (high accuracy, CPU)")
        
        # Initialize chosen OCR engine
        engines = {
            "paddle": lambda: PaddleOCRReader(),
            "easyocr": lambda: EasyOCRReader(),
            "ensemble": lambda: EnsembleOCRReader(),
            "deepseek": lambda: OCRReader(device=device),
        }
        
        self.ocr_engine_name = ocr_engine
        self.ocr = engines[ocr_engine]()
        print(f"Using OCR engine: {ocr_engine}")
        
        # Lazy load OCR model
        self._ocr_loaded = False
    
    def _ensure_ocr_loaded(self):
        """Lazy load OCR model when first needed"""
        if not self._ocr_loaded:
            self.ocr.load_model()
            self._ocr_loaded = True
    
    def process_image(self, image_path: str, save_crops: bool = True, 
                      save_visualization: bool = True) -> List[dict]:
        """
        Process a single image through the full pipeline
        
        Args:
            image_path: Path to the input image
            save_crops: Save cropped plate images
            save_visualization: Save image with detection boxes
            
        Returns:
            List of results with plate info and OCR text
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*50}")
        
        results = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Step 1: Detect and crop plates
        print("\n[1/2] Detecting license plates...")
        crops = self.detector.crop_plates(image_path)
        print(f"      Found {len(crops)} plate(s)")
        
        if len(crops) == 0:
            print("      No plates detected!")
            return results
        
        # Step 2: OCR on each crop
        print("\n[2/2] Extracting text with OCR...")
        self._ensure_ocr_loaded()
        
        crop_dir = os.path.join(config.OUTPUT_DIR, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        
        for i, (crop_img, detection) in enumerate(crops):
            # Save crop temporarily (or permanently if requested)
            crop_path = os.path.join(crop_dir, f"{base_name}_plate_{i}.jpg")
            cv2.imwrite(crop_path, crop_img)
            
            # Run OCR
            try:
                text = self.ocr.read_plate(crop_path)
                print(f"      Plate {i+1}: {text} (conf: {detection['confidence']:.2f})")
            except Exception as e:
                text = f"OCR Error: {str(e)}"
                print(f"      Plate {i+1}: Error - {e}")
            
            results.append({
                'plate_index': i,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'text': text,
                'crop_path': crop_path if save_crops else None
            })
            
            if not save_crops:
                os.remove(crop_path)
        
        # Save visualization with annotations
        if save_visualization:
            self._save_annotated_image(image_path, results)
        
        return results
    
    def _save_annotated_image(self, image_path: str, results: List[dict]):
        """Save image with bounding boxes and OCR text"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            text = r['text']
            conf = r['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw text background
            label = f"{text} ({conf:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(image, (x1, y2), (x1 + label_w + 10, y2 + label_h + 15), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(image, label, (x1 + 5, y2 + label_h + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Save
        vis_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, os.path.basename(image_path))
        cv2.imwrite(vis_path, image)
        print(f"\n      Visualization saved: {vis_path}")
    
    def process_directory(self, input_dir: str) -> dict:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing images
            
        Returns:
            Dictionary mapping image paths to results
        """
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Not a directory: {input_dir}")
        
        # Find all images
        images = [f for f in os.listdir(input_dir) 
                  if os.path.splitext(f)[1].lower() in config.IMAGE_EXTENSIONS]
        
        print(f"\nFound {len(images)} images to process")
        
        all_results = {}
        for img_name in images:
            img_path = os.path.join(input_dir, img_name)
            try:
                results = self.process_image(img_path)
                all_results[img_path] = results
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                all_results[img_path] = {'error': str(e)}
        
        # Save summary
        self._save_summary(all_results)
        
        return all_results
    
    def _save_summary(self, all_results: dict):
        """Save a JSON summary of all results"""
        summary_path = os.path.join(config.OUTPUT_DIR, "results_summary.json")
        
        # Prepare JSON-serializable results
        json_results = {}
        for path, results in all_results.items():
            if isinstance(results, dict) and 'error' in results:
                json_results[path] = results
            else:
                json_results[path] = [
                    {k: v for k, v in r.items()} 
                    for r in results
                ]
        
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': json_results
            }, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Results summary saved: {summary_path}")
        print(f"{'='*50}")


def main():
    """CLI for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="License Plate Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image (auto-selects best engine)
  python pipeline.py -i test_images/car.jpg
  
  # Process all images with PaddleOCR
  python pipeline.py -i test_images/ --engine paddle
  
  # Use ensemble mode (PaddleOCR + EasyOCR, picks best)
  python pipeline.py -i test_images/ --engine ensemble
        """
    )
    
    parser.add_argument("--input", "-i", required=True, 
                       help="Input image or directory")
    parser.add_argument("--output", "-o", default=config.OUTPUT_DIR,
                       help="Output directory")
    parser.add_argument("--engine", choices=["auto", "paddle", "easyocr", "ensemble", "deepseek"],
                       default="auto", help="OCR engine (default: auto)")
    parser.add_argument("--use-easyocr", action="store_true",
                       help="Shortcut for --engine easyocr")
    parser.add_argument("--no-crops", action="store_true",
                       help="Don't save cropped plate images")
    parser.add_argument("--no-vis", action="store_true",
                       help="Don't save visualizations")
    
    args = parser.parse_args()
    
    # Handle legacy flag
    engine = "easyocr" if args.use_easyocr else args.engine
    
    # Override output directory
    config.OUTPUT_DIR = args.output
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize pipeline
    pipeline = LicensePlateRecognizer(ocr_engine=engine)
    
    # Process
    if os.path.isfile(args.input):
        results = pipeline.process_image(
            args.input,
            save_crops=not args.no_crops,
            save_visualization=not args.no_vis
        )
        
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        for r in results:
            print(f"  Plate: {r['text']}")
            print(f"  Confidence: {r['confidence']:.2f}")
            print(f"  Bounding Box: {r['bbox']}")
            print()
    else:
        pipeline.process_directory(args.input)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
