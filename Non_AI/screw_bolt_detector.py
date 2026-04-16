"""
Enhanced OpenCV-based Screw and Bolt Detection
Author: Sanjay
Task: Count items in images using Non-AI techniques (OpenCV)
Accuracy Target: > 95%
"""

import cv2
import numpy as np
from pathlib import Path
import json


class ScrewBoltDetector:
    """Advanced detector for screws and bolts using traditional CV techniques."""
    
    def __init__(self, min_area=100, max_area=10000, blur_kernel=(5, 5), threshold=60):
        """
        Initialize detector with configurable parameters.
        
        Args:
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            blur_kernel: Gaussian blur kernel size
            threshold: Binary threshold value
        """
        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel = blur_kernel
        self.threshold = threshold
    
    def is_hexagonal_or_circular(self, contour):
        """
        Detect hexagonal (bolt head) or circular shapes.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            bool: True if shape matches hexagonal or circular pattern
        """
        # Approximate contour with polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Hexagonal detection (5-7 sides allows for noise)
        if 5 <= len(approx) <= 7:
            return True
        
        # Circular detection using circularity metric
        if len(approx) > 7:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
            return circularity > 0.6
        
        return False
    
    def has_center_hole(self, roi_image):
        """
        Detect if ROI has a center hole (indicates bolt).
        
        Args:
            roi_image: Region of interest image (BGR)
            
        Returns:
            bool: True if center hole detected
        """
        # Convert to grayscale and preprocess
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        _, roi_thresh = cv2.threshold(roi_blur, self.threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find internal contours
        inner_contours, _ = cv2.findContours(
            roi_thresh, 
            cv2.RETR_CCOMP, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Check for hexagonal or circular holes
        for inner_cnt in inner_contours:
            area = cv2.contourArea(inner_cnt)
            if 20 < area < 1000:
                if self.is_hexagonal_or_circular(inner_cnt):
                    return True
        
        return False
    
    def detect_objects(self, image):
        """
        Detect and classify screws and bolts in image.
        
        Args:
            image: Input BGR image
            
        Returns:
            tuple: (annotated_image, screw_count, bolt_count, detections)
        """
        output = image.copy()
        
        # Preprocessing pipeline
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        _, thresh = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        screw_count = 0
        bolt_count = 0
        detections = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            roi = image[y:y+h, x:x+w]
            
            # Classify based on center hole detection
            if self.has_center_hole(roi):
                label = 'Bolt'
                bolt_count += 1
                color = (0, 0, 255)  # Red for bolts
            else:
                label = 'Screw'
                screw_count += 1
                color = (0, 255, 0)  # Green for screws
            
            # Store detection info
            detections.append({
                'type': label,
                'bbox': (x, y, w, h),
                'area': int(area)
            })
            
            # Draw bounding box and label
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                output, 
                label, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        return output, screw_count, bolt_count, detections
    
    def process_dataset(self, image_dir, output_dir=None):
        """
        Process entire dataset directory.
        
        Args:
            image_dir: Path to directory containing images
            output_dir: Optional path to save annotated images
            
        Returns:
            dict: Results summary
        """
        image_path = Path(image_dir)
        results = {}
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for img_file in image_path.glob('*.jpg'):
            image = cv2.imread(str(img_file))
            
            if image is None:
                print(f"Warning: Could not read {img_file}")
                continue
            
            # Detect objects
            annotated, screws, bolts, detections = self.detect_objects(image)
            
            results[img_file.name] = {
                'screws': screws,
                'bolts': bolts,
                'total': screws + bolts,
                'detections': detections
            }
            
            print(f"{img_file.name} -> Screws: {screws}, Bolts: {bolts}, Total: {screws + bolts}")
            
            # Save annotated image if output directory specified
            if output_dir:
                output_file = output_path / f"annotated_{img_file.name}"
                cv2.imwrite(str(output_file), annotated)
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect screws and bolts in images')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', help='Output directory for annotated images')
    parser.add_argument('--save-json', help='Save results to JSON file')
    parser.add_argument('--min-area', type=int, default=100, help='Minimum contour area')
    parser.add_argument('--max-area', type=int, default=10000, help='Maximum contour area')
    parser.add_argument('--threshold', type=int, default=60, help='Binary threshold value')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ScrewBoltDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        threshold=args.threshold
    )
    
    input_path = Path(args.input)
    
    # Process directory or single image
    if input_path.is_dir():
        results = detector.process_dataset(input_path, args.output)
        
        # Save results to JSON if specified
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.save_json}")
    
    elif input_path.is_file():
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: Could not read image {input_path}")
            return
        
        annotated, screws, bolts, detections = detector.detect_objects(image)
        print(f"Screws: {screws}, Bolts: {bolts}, Total: {screws + bolts}")
        
        # Save annotated image if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"annotated_{input_path.name}"
            cv2.imwrite(str(output_file), annotated)
            print(f"Annotated image saved to {output_file}")
    
    else:
        print(f"Error: {input_path} is neither a file nor directory")


if __name__ == '__main__':
    main()
