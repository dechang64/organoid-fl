"""Generate bbox crops from SAM2 results JSON for VLM verification.

Usage:
    python scripts\multiorg\generate_crops.py --json results\multiorg_sam2_zeroshot\multiorg_sam2_results.json --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\phase2_vlm_100 --max-tp 50 --max-fp 50
"""
import argparse, json, os, random, sys
import numpy as np
from pathlib import Path

try:
    import tifffile
except ImportError:
    tifffile = None

import cv2


def load_sam2_results(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_image_path(src_dir, image_field):
    image_name = os.path.basename(image_field)
    return os.path.join(src_dir, image_field, f"{image_name}.tiff")


def load_image(path):
    if path.endswith('.tiff') or path.endswith('.tif'):
        if tifffile is None:
            raise ImportError("pip install tifffile")
        arr = tifffile.imread(path)
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            arr = ((arr.astype(np.float64) - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr
    else:
        from PIL import Image
        return np.array(Image.open(path).convert('RGB'))


def crop_detection(img_arr, bbox, padding=0.1):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px, py = int(w * padding), int(h * padding)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(img_arr.shape[1], x2 + px)
    y2 = min(img_arr.shape[0], y2 + py)
    return img_arr[y1:y2, x1:x2]


def draw_bbox(crop, bbox, padding=0.1):
    h, w = crop.shape[:2]
    x1 = max(0, int(bbox[0]) - int((bbox[2] - bbox[0]) * padding))
    y1 = max(0, int(bbox[1]) - int((bbox[3] - bbox[1]) * padding))
    x2 = min(w, int(bbox[2]) - x1 + int((bbox[2] - bbox[0]) * padding))
    y2 = min(h, int(bbox[3]) - y1 + int((bbox[3] - bbox[1]) * padding))
    cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return crop


def save_crop(crop, path):
    ext = os.path.splitext(path)[1]
    result, encoded = cv2.imencode(ext, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    encoded.tofile(path)


def main():
    parser = argparse.ArgumentParser(description='Generate crops for VLM verification')
    parser.add_argument('--json', required=True)
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', default='results/phase2_vlm_crops')
    parser.add_argument('--max-tp', type=int, default=50)
    parser.add_argument('--max-fp', type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"[ERROR] JSON not found: {args.json}")
        sys.exit(1)
    if not os.path.isdir(args.src):
        print(f"[ERROR] Source dir not found: {args.src}")
        sys.exit(1)

    output_dir = Path(args.dst)
    crops_dir = output_dir / 'crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.json}")
    sam2_data = load_sam2_results(args.json)

    tp_list, fp_list = [], []
    for img_data in sam2_data.get('per_image', []):
        if not isinstance(img_data, dict) or 'detections' not in img_data:
            continue
        image_field = img_data.get('image', '')
        for i, det in enumerate(img_data['detections']):
            entry = {
                'image': image_field,
                'det_idx': i,
                'bbox': det.get('bbox', []),
                'confidence': float(det.get('confidence', 0)),
                'matched': det.get('matched', False),
            }
            if entry['matched']:
                tp_list.append(entry)
            else:
                fp_list.append(entry)

    random.seed(42)
    tp_sample = random.sample(tp_list, min(args.max_tp, len(tp_list)))
    fp_sample = random.sample(fp_list, min(args.max_fp, len(fp_list)))
    all_dets = tp_sample + fp_sample
    print(f"Found: {len(tp_list)} TP, {len(fp_list)} FP")
    print(f"Sampled: {len(tp_sample)} TP, {len(fp_sample)} FP")

    image_cache = {}
    for idx, det in enumerate(all_dets):
        cache_key = f"{det['image']}_{det['det_idx']}".replace('/', '_').replace('\\', '_')
        print(f"[{idx+1}/{len(all_dets)}] {cache_key} ({'TP' if det['matched'] else 'FP'})")

        img_path = resolve_image_path(args.src, det['image'])
        if img_path not in image_cache:
            if not os.path.exists(img_path):
                print(f"  [SKIP] Image not found: {img_path}")
                continue
            print(f"  Loaded: {img_path}")
            image_cache[img_path] = load_image(img_path)

        img_arr = image_cache[img_path]
        crop = crop_detection(img_arr, det['bbox'])
        crop = draw_bbox(crop, det['bbox'])
        crop_path = crops_dir / f"{cache_key}.png"
        save_crop(crop, str(crop_path))
        det['cache_key'] = cache_key
        det['crop_path'] = str(crop_path)

    metadata_path = output_dir / 'crop_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_dets, f, indent=2, ensure_ascii=False)
    print(f"\nCrops: {crops_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == '__main__':
    main()
