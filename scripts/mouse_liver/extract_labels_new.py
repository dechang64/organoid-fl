r"""
从鼠肝红线标注图提取 YOLO 格式标签

Usage:
    python scripts\mouse_liver\extract_labels_new.py --annot D:\datasets\mouse_liver_new\annot --dst D:\datasets\mouse_liver_new\labels
"""
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def extract_yolo_labels(annot_path, img_w, img_h):
    """从红线标注图提取 YOLO 格式 bbox"""
    img = np.array(Image.open(annot_path).convert('RGB'))
    h, w = img.shape[:2]

    r, g, b = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)
    red_mask = (r > 150) & (r - g > 50) & (r - b > 50)
    red_uint8 = (red_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(red_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 500]

    labels = []
    for c in valid:
        x, y, cw, ch = cv2.boundingRect(c)
        xc = (x + cw / 2) / w
        yc = (y + ch / 2) / h
        bw = cw / w
        bh = ch / h
        labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot', required=True, help='Annotated images directory')
    parser.add_argument('--orig', required=True, help='Original images directory (for naming)')
    parser.add_argument('--dst', required=True, help='Output YOLO labels directory')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    annot_files = sorted(f for f in os.listdir(args.annot) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    orig_files = sorted(f for f in os.listdir(args.orig) if f.lower().endswith(('.jpg', '.png', '.jpeg')))

    if len(annot_files) != len(orig_files):
        print(f"[WARN] annot={len(annot_files)} vs orig={len(orig_files)} mismatch!")

    n = min(len(annot_files), len(orig_files))
    total = 0
    for i in range(n):
        annot_path = os.path.join(args.annot, annot_files[i])
        labels = extract_yolo_labels(annot_path, 0, 0)
        # Use ORIGINAL image name for the label file
        out_name = Path(orig_files[i]).stem + '.txt'
        with open(os.path.join(args.dst, out_name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels))
        total += len(labels)
        print(f"  [{i}] {orig_files[i]} ↔ {annot_files[i]}: {len(labels)} organoids")

    print(f"\nDone: {n} pairs, {total} organoids total")
    print(f"Labels saved to: {args.dst}")


if __name__ == '__main__':
    main()
