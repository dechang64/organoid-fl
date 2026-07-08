"""
诊断脚本: 检查 B1/B2/B3 的训练图像是否正确配对
在冬生本地运行:
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\mouse_liver\\v2\\check_pairing.py --data-root D:\\datasets\\mouse_liver_split

输出:
    1. 每个 batch 每张训练图的尺寸
    2. B1/B2 同名图片的 MD5 对比 (如果相同 = 图片被错误共享)
    3. 标注框在图片上是否可见 (简单的像素统计)
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np


def md5_file(path):
    """计算文件 MD5"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def check_batch(batch_name, data_root):
    """检查单个 batch 的图片-标签配对"""
    batch_dir = Path(data_root) / batch_name
    train_img_dir = batch_dir / 'full' / 'train' / 'images'
    train_lbl_dir = batch_dir / 'full' / 'train' / 'labels'

    if not train_img_dir.exists():
        print(f"  [ERROR] {train_img_dir} not found")
        return {}, {}

    print(f"\n{'='*60}")
    print(f"  {batch_name} — train images")
    print(f"{'='*60}")

    img_info = {}
    for img_path in sorted(train_img_dir.iterdir()):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
            continue

        img = Image.open(img_path)
        w, h = img.size
        md5 = md5_file(img_path)

        # 检查对应的 label
        lbl_path = train_lbl_dir / (img_path.stem + '.txt')
        n_boxes = 0
        if lbl_path.exists():
            with open(lbl_path) as f:
                n_boxes = len([l for l in f if l.strip()])

        # 检查标注框区域的像素特征
        lbl_stats = ""
        if lbl_path.exists() and n_boxes > 0:
            img_np = np.array(img.convert('L'))
            ih, iw = img_np.shape
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = int((cx - bw/2) * iw)
                        y1 = int((cy - bh/2) * ih)
                        x2 = int((cx + bw/2) * iw)
                        y2 = int((cy + bh/2) * ih)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(iw, x2), min(ih, y2)
                        region = img_np[y1:y2, x1:x2]
                        if region.size > 0:
                            mean_val = region.mean()
                            std_val = region.std()
                            lbl_stats += f"    box: ({x1},{y1})-({x2},{y2}) mean={mean_val:.1f} std={std_val:.1f}\n"

        print(f"  {img_path.name}: {w}x{h} md5={md5[:12]} boxes={n_boxes}")
        if lbl_stats:
            print(lbl_stats, end='')

        img_info[img_path.name] = {
            'size': (w, h),
            'md5': md5,
            'n_boxes': n_boxes,
        }

    return img_info


def check_cross_batch(b1_info, b2_info, b3_info):
    """检查不同 batch 是否有相同图片 (MD5 相同)"""
    print(f"\n{'='*60}")
    print(f"  Cross-batch MD5 comparison")
    print(f"{'='*60}")

    all_md5s = {}
    for batch_name, info in [('b1', b1_info), ('b2', b2_info), ('b3', b3_info)]:
        for img_name, data in info.items():
            md5 = data['md5']
            if md5 not in all_md5s:
                all_md5s[md5] = []
            all_md5s[md5].append(f"{batch_name}/{img_name}")

    found_dup = False
    for md5, locations in all_md5s.items():
        if len(locations) > 1:
            found_dup = True
            print(f"  [DUPLICATE] md5={md5[:12]} found in: {', '.join(locations)}")

    if not found_dup:
        print("  No duplicates found — all images are unique across batches")

    # Check same-name images
    print(f"\n  Same-name images across batches:")
    b1_names = set(b1_info.keys())
    b2_names = set(b2_info.keys())
    b3_names = set(b3_info.keys())

    common_12 = b1_names & b2_names
    common_13 = b1_names & b3_names
    common_23 = b2_names & b3_names

    if common_12:
        print(f"  B1∩B2 same names: {sorted(common_12)}")
        for name in sorted(common_12):
            md5_1 = b1_info[name]['md5']
            md5_2 = b2_info[name]['md5']
            match = "SAME!" if md5_1 == md5_2 else "different"
            print(f"    {name}: b1_md5={md5_1[:12]} b2_md5={md5_2[:12]} → {match}")

    if common_13:
        print(f"  B1∩B3 same names: {sorted(common_13)}")
        for name in sorted(common_13):
            md5_1 = b1_info[name]['md5']
            md5_3 = b3_info[name]['md5']
            match = "SAME!" if md5_1 == md5_3 else "different"
            print(f"    {name}: b1_md5={md5_1[:12]} b3_md5={md5_3[:12]} → {match}")

    if common_23:
        print(f"  B2∩B3 same names: {sorted(common_23)}")
        for name in sorted(common_23):
            md5_2 = b2_info[name]['md5']
            md5_3 = b3_info[name]['md5']
            match = "SAME!" if md5_2 == md5_3 else "different"
            print(f"    {name}: b2_md5={md5_2[:12]} b3_md5={md5_3[:12]} → {match}")

    if not common_12 and not common_13 and not common_23:
        print("  No same-name images across batches")


def check_original_data(data_root):
    """检查原始数据目录结构"""
    # mouse_liver_split 的上层是 mouse_liver_correct
    parent = Path(data_root).parent
    correct_dir = parent / 'mouse_liver_correct'

    print(f"\n{'='*60}")
    print(f"  Original data structure")
    print(f"{'='*60}")

    if not correct_dir.exists():
        print(f"  {correct_dir} not found, trying alternatives...")
        # Try other paths
        for name in ['mouse_liver_correct', 'mouse_liver']:
            correct_dir = parent / name
            if correct_dir.exists():
                break

    if not correct_dir.exists():
        print(f"  Original data directory not found")
        return

    print(f"  Original data: {correct_dir}")

    for batch_dir in sorted(correct_dir.iterdir()):
        if batch_dir.is_dir():
            print(f"\n  {batch_dir.name}/")
            for sub in sorted(batch_dir.iterdir()):
                if sub.is_dir():
                    n_files = len(list(sub.iterdir()))
                    print(f"    {sub.name}/ ({n_files} files)")
                else:
                    print(f"    {sub.name}")

            # Check annotations.json
            annot = batch_dir / 'annotations.json'
            if annot.exists():
                with open(annot, encoding='utf-8') as f:
                    ann = json.load(f)
                print(f"    annotations.json: {len(ann)} entries")
                if ann:
                    print(f"    First entry: {ann[0]}")


def main():
    parser = argparse.ArgumentParser(description='Check image-label pairing for mouse liver v2')
    parser.add_argument('--data-root', default=r'D:\datasets\mouse_liver_split',
                        help='Path to mouse_liver_split')
    args = parser.parse_args()

    data_root = args.data_root
    print(f"Data root: {data_root}")

    # Check original data structure
    check_original_data(data_root)

    # Check each batch
    b1_info = check_batch('b1', data_root)
    b2_info = check_batch('b2', data_root)
    b3_info = check_batch('b3', data_root)

    # Cross-batch comparison
    check_cross_batch(b1_info, b2_info, b3_info)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
