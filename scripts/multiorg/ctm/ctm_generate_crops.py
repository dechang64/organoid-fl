"""
Generate crops from MultiOrg SAM2 results for CTM training.

This script:
1. Reads SAM2 results JSON (16198 detections with bbox + TP/FP labels)
2. For each detection, crops the bbox region from the original image
3. Adds padding (20% of bbox size) for context
4. Saves as PNG with cache_key naming
5. Outputs a metadata JSON for the dataset

Usage (on 冬生's Windows machine with original MultiOrg images):
    python ctm_generate_crops.py \
        --sam2-results results/multiorg_sam2_zeroshot/multiorg_sam2_results.json \
        --images-root "D:\datasets\mutliorg\MultiOrg_v2" \
        --output-dir data/ctm_crops \
        --pad-ratio 0.2

Output:
    data/ctm_crops/
        Macros_Plate_15_image_0_0.png
        Macros_Plate_15_image_0_1.png
        ...
        ctm_metadata.json
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Handle imports for both Windows and Linux
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not found. Install with: pip install tifffile")

try:
    from PIL import Image
except ImportError:
    print("Error: PIL/Pillow required. Install with: pip install Pillow")
    sys.exit(1)


def load_tiff(path: str) -> np.ndarray:
    """Load 16-bit TIFF safely (uses tifffile to avoid PIL segfault on Windows)."""
    if HAS_TIFFFILE:
        return tifffile.imread(path)
    else:
        # Fallback to PIL (may segfault on Windows for 16-bit TIFFs)
        return np.array(Image.open(path))


def normalize_16bit_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize 16-bit image to 8-bit for saving as PNG."""
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        arr = ((arr.astype(np.float64) - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        arr = np.zeros_like(arr, dtype=np.uint8)
    return arr


def crop_bbox(
    image: np.ndarray,
    bbox: list,
    pad_ratio: float = 0.2,
    min_size: int = 32,
) -> np.ndarray:
    """
    Crop bbox region with padding.
    
    Args:
        image: [H, W] or [H, W, C] array
        bbox: [x1, y1, x2, y2] in original image coordinates
        pad_ratio: fraction of bbox size to add as padding
        min_size: minimum crop size
    
    Returns:
        Cropped region as np.ndarray
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # Add padding
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(image.shape[1], int(x2 + pad_w))
    y2 = min(image.shape[0], int(y2 + pad_h))
    
    # Ensure minimum size
    if x2 - x1 < min_size:
        cx = (x1 + x2) / 2
        x1 = max(0, int(cx - min_size / 2))
        x2 = min(image.shape[1], int(cx + min_size / 2))
    if y2 - y1 < min_size:
        cy = (y1 + y2) / 2
        y1 = max(0, int(cy - min_size / 2))
        y2 = min(image.shape[0], int(cy + min_size / 2))
    
    return image[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description='Generate crops for CTM training')
    parser.add_argument('--sam2-results', type=str, required=True,
                        help='Path to multiorg_sam2_results.json')
    parser.add_argument('--images-root', type=str, required=True,
                        help='Root directory of MultiOrg images (e.g. D:\\datasets\\mutliorg\\MultiOrg_v2)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for crops')
    parser.add_argument('--pad-ratio', type=float, default=0.2,
                        help='Padding ratio around bbox')
    parser.add_argument('--max-crops', type=int, default=None,
                        help='Max number of crops (for testing)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing crops')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load SAM2 results
    with open(args.sam2_results, encoding='utf-8') as f:
        sam2_data = json.load(f)
    
    per_image = sam2_data['per_image']
    print(f"Loaded {len(per_image)} images with {sum(img['n_det'] for img in per_image)} detections")
    
    # Image cache (avoid re-reading same TIFF)
    image_cache = {}
    max_cache = 20  # Cache up to 20 images
    
    # Generate crops
    metadata = []
    n_generated = 0
    n_skipped = 0
    n_errors = 0
    
    for img_info in per_image:
        image_name = img_info['image']  # e.g. "Macros/Plate_15/image_0"
        detections = img_info['detections']
        
        # Construct image path
        # image_name format: "Macros/Plate_15/image_0"
        # images_root: "D:\datasets\mutliorg\MultiOrg_v2"
        # Full path: images_root\train\Macros\Plate_15\image_0\image_0.tiff
        parts = image_name.split('/')
        cls_name = parts[0]  # Macros or Normal
        plate_name = parts[1]  # Plate_15
        img_folder = parts[2]  # image_0
        
        # Try train/ then test/
        for split in ['train', 'test']:
            tiff_path = os.path.join(args.images_root, split, cls_name, plate_name, img_folder, f'{img_folder}.tiff')
            if os.path.exists(tiff_path):
                break
        else:
            # Try without split prefix
            tiff_path = os.path.join(args.images_root, cls_name, plate_name, img_folder, f'{img_folder}.tiff')
        
        if not os.path.exists(tiff_path):
            print(f"  [SKIP] Image not found: {tiff_path}")
            n_skipped += len(detections)
            continue
        
        # Load image (with cache)
        cache_key_img = image_name
        if cache_key_img in image_cache:
            image = image_cache[cache_key_img]
        else:
            try:
                image = load_tiff(tiff_path)
                # Normalize 16-bit to 8-bit
                if image.dtype != np.uint8:
                    image = normalize_16bit_to_uint8(image)
                # Convert to RGB (3 channels)
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3 and image.shape[2] == 1:
                    image = np.concatenate([image] * 3, axis=-1)
                
                # Cache management
                if len(image_cache) >= max_cache:
                    image_cache.clear()
                image_cache[cache_key_img] = image
            except Exception as e:
                print(f"  [ERROR] Failed to load {tiff_path}: {e}")
                n_errors += len(detections)
                continue
        
        # Generate crops for each detection
        for det_idx, det in enumerate(detections):
            cache_key = f"{image_name.replace('/', '_')}_{det_idx}"
            crop_path = os.path.join(args.output_dir, f'{cache_key}.png')
            
            # Skip if exists and not overwriting
            if os.path.exists(crop_path) and not args.overwrite:
                n_skipped += 1
                continue
            
            # Crop
            bbox = det['bbox']  # [x1, y1, x2, y2]
            try:
                crop = crop_bbox(image, bbox, pad_ratio=args.pad_ratio)
                
                # Skip empty crops
                if crop.size == 0 or min(crop.shape[:2]) < 8:
                    n_skipped += 1
                    continue
                
                # Save as PNG
                Image.fromarray(crop).save(crop_path)
                n_generated += 1
                
                # Add to metadata
                metadata.append({
                    'cache_key': cache_key,
                    'image': image_name,
                    'det_idx': det_idx,
                    'bbox': bbox,
                    'rfdetr_conf': det.get('confidence', 0.0),
                    'matched': det.get('matched', False),  # TP/FP label
                    'match_iou': det.get('match_iou', 0.0),
                    'area': det.get('area', 0),
                    'circularity': det.get('circularity', 0),
                    'solidity': det.get('solidity', 0),
                    'aspect_ratio': det.get('aspect_ratio', 0),
                })
            except Exception as e:
                print(f"  [ERROR] Crop failed for {cache_key}: {e}")
                n_errors += 1
        
        if n_generated % 500 == 0 and n_generated > 0:
            print(f"  Generated {n_generated} crops...")
        
        if args.max_crops and n_generated >= args.max_crops:
            print(f"Reached max_crops={args.max_crops}, stopping")
            break
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, 'ctm_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Summary
    n_tp = sum(1 for m in metadata if m['matched'])
    n_fp = len(metadata) - n_tp
    print(f"\n{'='*60}")
    print(f"Crop Generation Complete")
    print(f"{'='*60}")
    print(f"  Generated: {n_generated}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Errors:    {n_errors}")
    print(f"  Metadata:  {len(metadata)} entries ({n_tp} TP + {n_fp} FP)")
    print(f"  Output:    {args.output_dir}")
    print(f"  Metadata:  {metadata_path}")


if __name__ == '__main__':
    main()
