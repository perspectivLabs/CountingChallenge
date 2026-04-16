"""
AI-based Screw and Bolt Detection using Segment Anything Model (SAM)
Author: Sanjay
Task: Count items in images using AI techniques
Accuracy Target: > 95%
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import urllib.request


class SAMObjectCounter:
    """Object counter using Meta's Segment Anything Model."""
    
    def __init__(self, model_type='vit_b', device=None):
        """
        Initialize SAM-based object counter.
        
        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            device: Device to run model on (auto-detects if None)
        """
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam = None
        self.mask_generator = None
        self.checkpoint_path = None
        
        print(f"Using device: {self.device}")
    
    def download_checkpoint(self, force=False):
        """
        Download SAM checkpoint if not exists.
        
        Args:
            force: Force re-download even if file exists
        """
        checkpoint_urls = {
            'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
        }
        
        checkpoint_file = f"sam_{self.model_type}_checkpoint.pth"
        self.checkpoint_path = checkpoint_file
        
        if Path(checkpoint_file).exists() and not force:
            print(f"Checkpoint already exists: {checkpoint_file}")
            return
        
        url = checkpoint_urls[self.model_type]
        print(f"Downloading {self.model_type} checkpoint...")
        print(f"URL: {url}")
        
        try:
            urllib.request.urlretrieve(url, checkpoint_file)
            print(f"Downloaded to {checkpoint_file}")
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            raise
    
    def load_model(self):
        """Load SAM model and initialize mask generator."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "segment-anything not installed. "
                "Run: pip install segment-anything"
            )
        
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            self.download_checkpoint()
        
        print(f"Loading SAM model ({self.model_type})...")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(self.device)
        
        # Initialize automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        print("Model loaded successfully")
    
    def filter_masks_by_area(
        self, 
        masks: List[Dict], 
        min_area_multiplier: float = 0.3,
        max_area_multiplier: float = 2.0
    ) -> List[Dict]:
        """
        Filter masks based on median area.
        
        Args:
            masks: List of mask dictionaries
            min_area_multiplier: Minimum area as multiple of median
            max_area_multiplier: Maximum area as multiple of median
            
        Returns:
            Filtered list of masks
        """
        if not masks:
            return []
        
        areas = np.array([m['area'] for m in masks])
        median_area = np.median(areas)
        
        min_area = min_area_multiplier * median_area
        max_area = max_area_multiplier * median_area
        
        filtered = [m for m in masks if min_area < m['area'] < max_area]
        
        print(f"Area filtering: {len(masks)} -> {len(filtered)} masks")
        print(f"Median area: {median_area:.0f}, Range: [{min_area:.0f}, {max_area:.0f}]")
        
        return filtered
    
    @staticmethod
    def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Intersection over Union between two masks.
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            IoU score
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-8)
    
    @staticmethod
    def compute_containment(mask_small: np.ndarray, mask_large: np.ndarray) -> float:
        """
        Compute what percentage of small mask is contained in large mask.
        
        Args:
            mask_small: Smaller mask
            mask_large: Larger mask
            
        Returns:
            Containment ratio
        """
        intersection = np.logical_and(mask_small, mask_large).sum()
        return intersection / (mask_small.sum() + 1e-8)
    
    def remove_duplicates(
        self,
        masks: List[Dict],
        iou_threshold: float = 0.6,
        containment_threshold: float = 0.8
    ) -> List[Dict]:
        """
        Remove overlapping/duplicate masks.
        
        Args:
            masks: List of mask dictionaries
            iou_threshold: IoU threshold for duplicate detection
            containment_threshold: Containment threshold
            
        Returns:
            List of unique masks
        """
        if not masks:
            return []
        
        # Sort by area (largest first)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        final_masks = []
        
        for mask in sorted_masks:
            keep = True
            
            for existing_mask in final_masks:
                # Check IoU
                iou = self.compute_iou(
                    mask['segmentation'], 
                    existing_mask['segmentation']
                )
                
                if iou > iou_threshold:
                    keep = False
                    break
                
                # Check containment
                containment = self.compute_containment(
                    mask['segmentation'],
                    existing_mask['segmentation']
                )
                
                if containment > containment_threshold:
                    keep = False
                    break
            
            if keep:
                final_masks.append(mask)
        
        print(f"Duplicate removal: {len(sorted_masks)} -> {len(final_masks)} masks")
        
        return final_masks
    
    def detect_objects(
        self,
        image: np.ndarray,
        min_area_mult: float = 0.3,
        max_area_mult: float = 2.0,
        iou_threshold: float = 0.6,
        containment_threshold: float = 0.8
    ) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        Detect and count objects in image.
        
        Args:
            image: Input BGR image
            min_area_mult: Minimum area multiplier
            max_area_mult: Maximum area multiplier
            iou_threshold: IoU threshold for duplicate removal
            containment_threshold: Containment threshold
            
        Returns:
            Tuple of (annotated_image, object_count, mask_details)
        """
        if self.mask_generator is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        print("Generating masks...")
        raw_masks = self.mask_generator.generate(image_rgb)
        print(f"Raw masks detected: {len(raw_masks)}")
        
        # Filter by area
        filtered_masks = self.filter_masks_by_area(
            raw_masks,
            min_area_mult,
            max_area_mult
        )
        
        # Remove duplicates
        final_masks = self.remove_duplicates(
            filtered_masks,
            iou_threshold,
            containment_threshold
        )
        
        # Create overlay visualization
        overlay = image_rgb.copy()
        
        for i, mask in enumerate(final_masks):
            seg = mask['segmentation']
            color = np.random.randint(0, 255, 3)
            overlay[seg] = overlay[seg] * 0.5 + color * 0.5
        
        # Convert back to BGR for OpenCV
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        return overlay_bgr, len(final_masks), final_masks
    
    def process_dataset(
        self,
        image_dir: str,
        output_dir: str = None,
        save_json: str = None,
        **kwargs
    ) -> Dict:
        """
        Process entire dataset directory.
        
        Args:
            image_dir: Path to directory containing images
            output_dir: Optional path to save annotated images
            save_json: Optional path to save results JSON
            **kwargs: Additional arguments for detect_objects
            
        Returns:
            Dictionary of results
        """
        image_path = Path(image_dir)
        results = {}
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for img_file in sorted(image_path.glob('*.jpg')):
            print(f"\n{'='*60}")
            print(f"Processing: {img_file.name}")
            print('='*60)
            
            image = cv2.imread(str(img_file))
            
            if image is None:
                print(f"Warning: Could not read {img_file}")
                continue
            
            # Detect objects
            annotated, count, masks = self.detect_objects(image, **kwargs)
            
            results[img_file.name] = {
                'count': count,
                'masks': [
                    {
                        'area': int(m['area']),
                        'bbox': [int(x) for x in m['bbox']],
                        'predicted_iou': float(m['predicted_iou'])
                    }
                    for m in masks
                ]
            }
            
            print(f"\n✓ Final count: {count} objects")
            
            # Save annotated image
            if output_dir:
                output_file = output_path / f"annotated_{img_file.name}"
                cv2.imwrite(str(output_file), annotated)
                print(f"Saved: {output_file}")
        
        # Save results to JSON
        if save_json:
            with open(save_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {save_json}")
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Count objects using Segment Anything Model'
    )
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image or directory')
    parser.add_argument('--output', '-o', 
                       help='Output directory for annotated images')
    parser.add_argument('--save-json', 
                       help='Save results to JSON file')
    parser.add_argument('--model-type', default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    parser.add_argument('--min-area-mult', type=float, default=0.3,
                       help='Minimum area multiplier')
    parser.add_argument('--max-area-mult', type=float, default=2.0,
                       help='Maximum area multiplier')
    parser.add_argument('--iou-threshold', type=float, default=0.6,
                       help='IoU threshold for duplicate removal')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download checkpoint and exit')
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = SAMObjectCounter(model_type=args.model_type)
    
    if args.download_only:
        counter.download_checkpoint()
        return
    
    # Load model
    counter.load_model()
    
    input_path = Path(args.input)
    
    # Process directory or single image
    if input_path.is_dir():
        results = counter.process_dataset(
            input_path,
            args.output,
            args.save_json,
            min_area_mult=args.min_area_mult,
            max_area_mult=args.max_area_mult,
            iou_threshold=args.iou_threshold
        )
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for filename, data in results.items():
            print(f"{filename}: {data['count']} objects")
        print(f"\nTotal images processed: {len(results)}")
    
    elif input_path.is_file():
        image = cv2.imread(str(input_path))
        
        if image is None:
            print(f"Error: Could not read image {input_path}")
            return
        
        annotated, count, masks = counter.detect_objects(
            image,
            min_area_mult=args.min_area_mult,
            max_area_mult=args.max_area_mult,
            iou_threshold=args.iou_threshold
        )
        
        print(f"\n✓ Detected {count} objects")
        
        # Save annotated image
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"annotated_{input_path.name}"
            cv2.imwrite(str(output_file), annotated)
            print(f"Saved: {output_file}")
    
    else:
        print(f"Error: {input_path} is neither a file nor directory")


if __name__ == '__main__':
    main()
