"""
License Plate Detection Module using YOLO
Uses a pre-trained YOLO model (best.pt) to detect and crop license plates
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from typing import List, Tuple, Optional
import config


class PlateDetector:
    """YOLO-based license plate detector"""
    
    def __init__(self, model_path: str = None, confidence: float = None):
        """
        Initialize the plate detector
        
        Args:
            model_path: Path to YOLO model (.pt file)
            confidence: Minimum confidence threshold for detections
        """
        self.model_path = model_path or config.MODEL_PATH
        self.confidence = confidence or config.DETECTION_CONFIDENCE
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading YOLO model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        print("Model loaded successfully!")
    
    def detect(self, image_path: str) -> List[dict]:
        """
        Detect license plates in an image using ensemble (original + CLAHE).
        Falls back to multi-scale tiling for small/distant plates (e.g. CCTV).
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box
            - confidence: detection confidence score
            - class_name: detected class name
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # 1. Run detection on original image
        results1 = self.model(image, conf=self.confidence, verbose=False)
        detections1 = self._parse_results(results1)
        
        # 2. Run detection on CLAHE enhanced image (improves low-light detection)
        clahe_img = self._enhance_image(image)
        results2 = self.model(clahe_img, conf=self.confidence, verbose=False)
        detections2 = self._parse_results(results2)
        
        # 3. Merge results with NMS
        all_detections = detections1 + detections2
        final_detections = self._nms(all_detections)
        
        # 4. If nothing found, try multi-scale tiling (for small/distant plates)
        if not final_detections:
            print("      [INFO] No plates at full resolution — trying multi-scale tiling...")
            tile_detections = self._detect_with_tiles(image)
            if tile_detections:
                final_detections = tile_detections
        
        return final_detections
    
    def _detect_with_tiles(self, image: np.ndarray, 
                           grid: Tuple[int, int] = (2, 2),
                           overlap: float = 0.25,
                           scale: float = 2.0) -> List[dict]:
        """
        Split image into overlapping tiles, upscale each, and run detection.
        Maps bounding boxes back to the original image coordinate space.
        This catches small/distant plates that YOLO misses at full resolution.
        
        Args:
            image:   Full-resolution BGR image
            grid:    (rows, cols) tile grid
            overlap: Fraction of tile dimension used as overlap between tiles
            scale:   Upscale factor applied to each tile before detection
            
        Returns:
            Deduplicated list of detections in original image coordinates
        """
        h, w = image.shape[:2]
        rows, cols = grid
        
        # Compute tile dimensions with overlap
        tile_h = int(h / rows * (1 + overlap))
        tile_w = int(w / cols * (1 + overlap))
        step_y = int(h / rows)
        step_x = int(w / cols)
        
        all_tile_dets = []
        
        for r in range(rows):
            for c in range(cols):
                # Tile origin (clamped)
                y0 = max(0, r * step_y - int(tile_h * overlap / 2))
                x0 = max(0, c * step_x - int(tile_w * overlap / 2))
                y1 = min(h, y0 + tile_h)
                x1 = min(w, x0 + tile_w)
                
                tile = image[y0:y1, x0:x1]
                
                # Upscale the tile
                tile_up = cv2.resize(tile, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
                
                # Run YOLO on the upscaled tile
                results = self.model(tile_up, conf=self.confidence, verbose=False)
                dets = self._parse_results(results)
                
                # Map detections back to original image coordinates
                for det in dets:
                    bx1, by1, bx2, by2 = det['bbox']
                    det['bbox'] = [
                        int(bx1 / scale) + x0,
                        int(by1 / scale) + y0,
                        int(bx2 / scale) + x0,
                        int(by2 / scale) + y0,
                    ]
                    all_tile_dets.append(det)
        
        # Deduplicate overlapping detections from adjacent tiles
        return self._nms(all_tile_dets)
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement for low-light detection"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _parse_results(self, results) -> List[dict]:
        """Convert YOLO results object to list of dicts"""
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_name': class_name
                })
        return detections

    def _nms(self, detections: List[dict], iou_thresh: float = 0.5) -> List[dict]:
        """Apply Non-Maximum Suppression to merge overlapping boxes"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            curr = detections.pop(0)
            keep.append(curr)
            
            # Remove overlapping boxes
            remaining = []
            for det in detections:
                iou = self._calculate_iou(curr['bbox'], det['bbox'])
                if iou < iou_thresh:
                    remaining.append(det)
            detections = remaining
            
        return keep

    def _calculate_iou(self, box1, box2) -> float:
        """Calculate Intersection over Union (IoU) of two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter_area == 0:
            return 0.0
            
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area
    
    def crop_plates(self, image_path: str, padding: int = 5) -> List[Tuple[np.ndarray, dict]]:
        """
        Detect and crop license plates from an image
        
        Args:
            image_path: Path to the input image
            padding: Extra pixels to add around the detected plate
            
        Returns:
            List of tuples (cropped_image, detection_info)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Get detections
        detections = self.detect(image_path)
        
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Adaptive padding: use larger padding for small (bike-sized) plates
            plate_area = (x2 - x1) * (y2 - y1)
            if plate_area < config.SMALL_PLATE_AREA_THRESHOLD:
                pad = max(padding, config.SMALL_PLATE_PADDING)
            else:
                pad = padding
            
            # Add padding (clamp to image bounds)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            # Crop the plate region
            cropped = image[y1:y2, x1:x2]
            crops.append((cropped, det))
        
        return crops
    
    def save_crops(self, image_path: str, output_dir: str = None, padding: int = 5) -> List[str]:
        """
        Detect, crop, and save license plates to files
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save cropped images
            padding: Extra pixels to add around the detected plate
            
        Returns:
            List of paths to saved cropped images
        """
        output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        crops = self.crop_plates(image_path, padding)
        
        saved_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for i, (crop, det) in enumerate(crops):
            filename = f"{base_name}_plate_{i}_{det['confidence']:.2f}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, crop)
            saved_paths.append(save_path)
            print(f"Saved plate crop: {save_path}")
        
        return saved_paths
    
    def visualize(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            
        Returns:
            Image with bounding boxes drawn
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        detections = self.detect(image_path)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Plate: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Saved visualization: {output_path}")
        
        return image


def main():
    """CLI for plate detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="License Plate Detection")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", default=config.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--visualize", "-v", action="store_true", help="Save visualizations")
    parser.add_argument("--confidence", "-c", type=float, default=config.DETECTION_CONFIDENCE,
                       help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    detector = PlateDetector(confidence=args.confidence)
    
    # Get list of images
    if os.path.isfile(args.input):
        images = [args.input]
    else:
        images = [os.path.join(args.input, f) for f in os.listdir(args.input)
                  if os.path.splitext(f)[1].lower() in config.IMAGE_EXTENSIONS]
    
    print(f"\nProcessing {len(images)} images...")
    
    for image_path in images:
        print(f"\n--- Processing: {image_path} ---")
        
        # Detect and save crops
        crops = detector.save_crops(image_path, args.output)
        print(f"Found {len(crops)} plates")
        
        # Optionally save visualization
        if args.visualize:
            vis_dir = os.path.join(args.output, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, os.path.basename(image_path))
            detector.visualize(image_path, vis_path)
    
    print(f"\nDone! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
