"""
诊断 B1/B2 图像是否被错误共享
在冬生本地运行:
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\mouse_liver\\v2\\check_image_pairing.py
"""
import hashlib
import os
from pathlib import Path
from PIL import Image
import json

SPLIT_ROOT = Path(r"D:\datasets\mouse_liver_split")
ORIG_ROOT = Path(r"D:\datasets\mouse_liver_correct")


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def check_batch_images():
    """检查各 batch 训练图的 MD5，看 B1/B2 同名图是否相同"""
    print("=" * 70)
    print("1. 检查 split 数据中 B1/B2/B3 同名图片是否为同一文件 (MD5)")
    print("=" * 70)

    batches = ['b1', 'b2', 'b3']
    all_imgs = {}

    for batch in batches:
        img_dir = SPLIT_ROOT / batch / 'full' / 'train' / 'images'
        if not img_dir.exists():
            print(f"  [ERROR] {img_dir} not found")
            continue

        all_imgs[batch] = {}
        for img_file in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.JPG')):
            md5 = md5_file(img_file)
            w, h = Image.open(img_file).size
            all_imgs[batch][img_file.name] = {'md5': md5, 'size': (w, h), 'path': str(img_file)}
            print(f"  {batch}/{img_file.name}: {w}x{h}, MD5={md5[:12]}")

    # Cross-batch comparison
    print("\n  --- Cross-batch MD5 comparison ---")
    if 'b1' in all_imgs and 'b2' in all_imgs:
        b1_names = set(all_imgs['b1'].keys())
        b2_names = set(all_imgs['b2'].keys())
        common = b1_names & b2_names
        if common:
            print(f"  B1 and B2 have {len(common)} same-named images:")
            for name in sorted(common):
                b1_md5 = all_imgs['b1'][name]['md5']
                b2_md5 = all_imgs['b2'][name]['md5']
                b1_size = all_imgs['b1'][name]['size']
                b2_size = all_imgs['b2'][name]['size']
                same = "SAME FILE!" if b1_md5 == b2_md5 else "different"
                print(f"    {name}: B1={b1_size} {b1_md5[:12]} | B2={b2_size} {b2_md5[:12]} → {same}")
        else:
            print("  B1 and B2 have no same-named images")

    if 'b1' in all_imgs and 'b3' in all_imgs:
        b1_names = set(all_imgs['b1'].keys())
        b3_names = set(all_imgs['b3'].keys())
        common = b1_names & b3_names
        if common:
            print(f"  B1 and B3 have {len(common)} same-named images:")
            for name in sorted(common):
                b1_md5 = all_imgs['b1'][name]['md5']
                b3_md5 = all_imgs['b3'][name]['md5']
                same = "SAME FILE!" if b1_md5 == b3_md5 else "different"
                print(f"    {name}: B1={b1_md5[:12]} | B3={b3_md5[:12]} → {same}")


def check_original_structure():
    """检查原始数据目录结构"""
    print("\n" + "=" * 70)
    print("2. 检查原始数据目录结构")
    print("=" * 70)

    for batch in ['batch1', 'batch2', 'batch3', 'b1', 'b2', 'b3']:
        batch_dir = ORIG_ROOT / batch
        if not batch_dir.exists():
            continue

        print(f"\n  {batch_dir}")
        for item in sorted(batch_dir.iterdir()):
            if item.is_dir():
                n_files = len(list(item.iterdir()))
                print(f"    {item.name}/ ({n_files} items)")
                # 如果是 images/ 目录，列出前几个文件名
                if 'image' in item.name.lower():
                    for f in sorted(item.iterdir())[:5]:
                        print(f"      {f.name}")
                    if n_files > 5:
                        print(f"      ... ({n_files} total)")
            else:
                print(f"    {item.name} ({item.stat().st_size} bytes)")

    # 也检查 data_root 下的 yolo_format
    yolo_dir = ORIG_ROOT / 'yolo_format'
    if yolo_dir.exists():
        print(f"\n  {yolo_dir} (SHARED)")
        for sub in sorted(yolo_dir.iterdir()):
            if sub.is_dir():
                n = len(list(sub.iterdir()))
                print(f"    {sub.name}/ ({n} items)")
                if 'image' in sub.name.lower():
                    for f in sorted(sub.iterdir())[:10]:
                        print(f"      {f.name}")
                    if n > 10:
                        print(f"      ... ({n} total)")


def check_label_image_match():
    """检查每张训练图的标注是否合理（框不出界）"""
    print("\n" + "=" * 70)
    print("3. 检查标注框是否在图片范围内")
    print("=" * 70)

    for batch in ['b1', 'b2', 'b3']:
        img_dir = SPLIT_ROOT / batch / 'full' / 'train' / 'images'
        lbl_dir = SPLIT_ROOT / batch / 'full' / 'train' / 'labels'
        if not img_dir.exists():
            continue

        print(f"\n  {batch}:")
        for img_file in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.JPG')):
            lbl_file = lbl_dir / (img_file.stem + '.txt')
            if not lbl_file.exists():
                print(f"    {img_file.name}: NO LABEL!")
                continue

            img = Image.open(img_file)
            w, h = img.size

            with open(lbl_file, encoding='utf-8') as f:
                lines = f.readlines()

            issues = []
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    issues.append(f"    line {i}: malformed: {line.strip()}")
                    continue
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = cx - bw/2
                y1 = cy - bh/2
                x2 = cx + bw/2
                y2 = cy + bh/2
                if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                    issues.append(f"    line {i}: box out of bounds! ({x1:.3f},{y1:.3f})-({x2:.3f},{y2:.3f})")

            status = f"{len(lines)} boxes" if not issues else f"{len(lines)} boxes, ISSUES:"
            print(f"    {img_file.name} ({w}x{h}): {status}")
            for issue in issues:
                print(f"      {issue}")


def check_test_images():
    """检查 B1/B2 test 图像是否相同"""
    print("\n" + "=" * 70)
    print("4. 检查 B1/B2 test 图像是否为同一文件")
    print("=" * 70)

    for batch in ['b1', 'b2', 'b3']:
        test_dir = SPLIT_ROOT / batch / 'test' / 'images'
        if not test_dir.exists():
            continue
        print(f"\n  {batch} test:")
        for img_file in sorted(test_dir.glob('*.jpg')) + sorted(test_dir.glob('*.JPG')):
            md5 = md5_file(img_file)
            w, h = Image.open(img_file).size
            print(f"    {img_file.name}: {w}x{h}, MD5={md5[:12]}")

    # Direct comparison
    b1_test = SPLIT_ROOT / 'b1' / 'test' / 'images'
    b2_test = SPLIT_ROOT / 'b2' / 'test' / 'images'
    if b1_test.exists() and b2_test.exists():
        b1_files = {f.name: md5_file(f) for f in b1_test.glob('*.jpg')}
        b2_files = {f.name: md5_file(f) for f in b2_test.glob('*.jpg')}
        common = set(b1_files.keys()) & set(b2_files.keys())
        if common:
            print(f"\n  B1/B2 test 同名文件对比:")
            for name in sorted(common):
                same = "SAME FILE!" if b1_files[name] == b2_files[name] else "different"
                print(f"    {name}: {same}")


def main():
    print(f"Split root: {SPLIT_ROOT}")
    print(f"Original data root: {ORIG_ROOT}")

    check_original_structure()
    check_batch_images()
    check_test_images()
    check_label_image_match()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
