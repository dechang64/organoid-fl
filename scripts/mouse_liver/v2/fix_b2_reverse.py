"""
修复 batch2 图片反序问题
image_00 ↔ 90_2 (应该 ↔ 81_2)
image_01 ↔ 89_2 (应该 ↔ 82_2)
...
image_09 ↔ 81_2 (应该 ↔ 90_2)

修复方案: 反转 images/ 目录中的文件名
  image_00.jpg ← → image_09.jpg
  image_01.jpg ← → image_08.jpg
  ...

Usage:
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\mouse_liver\\v2\\fix_b2_reverse.py
"""
import os
import shutil
from pathlib import Path

BATCH2_DIR = Path(r"D:\datasets\mouse_liver_correct\batch2\images")


def main():
    print(f"Fixing B2 image reverse in: {BATCH2_DIR}")

    if not BATCH2_DIR.exists():
        print(f"[ERROR] {BATCH2_DIR} not found")
        return

    # List all images sorted
    images = sorted([f for f in os.listdir(BATCH2_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    n = len(images)
    print(f"Found {n} images")
    print(f"Current order (reversed):")
    for i, f in enumerate(images):
        print(f"  {f} → should be image_{n-1-i:02d}.jpg")

    # Rename to temp names first (avoid collision)
    print(f"\nStep 1: Rename to temp...")
    for i, f in enumerate(images):
        src = BATCH2_DIR / f
        tmp = BATCH2_DIR / f"__tmp_{i:02d}__"
        shutil.move(str(src), str(tmp))

    # Now reverse: __tmp_00__ → image_09, __tmp_01__ → image_08, ...
    print(f"Step 2: Rename to reversed...")
    for i in range(n):
        tmp = BATCH2_DIR / f"__tmp_{i:02d}__"
        new_name = f"image_{n-1-i:02d}.jpg"
        dst = BATCH2_DIR / new_name
        shutil.move(str(tmp), str(dst))
        print(f"  __tmp_{i:02d}__ → {new_name}")

    # Also reverse the labels to match
    labels_dir = Path(r"D:\datasets\mouse_liver_correct\batch2\labels")
    if labels_dir.exists():
        print(f"\nStep 3: Reverse labels in: {labels_dir}")
        labels = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Rename to temp
        for i, f in enumerate(labels):
            src = labels_dir / f
            tmp = labels_dir / f"__tmp_{i:02d}__"
            shutil.move(str(src), str(tmp))

        # Reverse
        for i in range(len(labels)):
            tmp = labels_dir / f"__tmp_{i:02d}__"
            new_name = f"image_{n-1-i:02d}.txt"
            dst = labels_dir / new_name
            shutil.move(str(tmp), str(dst))
            print(f"  __tmp_{i:02d}__ → {new_name}")

    print(f"\n{'='*60}")
    print("Done! B2 images and labels have been reversed.")
    print("Now verify with:")
    print("  python scripts\\mouse_liver\\v2\\check_image_naming.py")
    print("Then regenerate split:")
    print("  python scripts\\mouse_liver\\v2\\prepare_data.py --data-root D:\\datasets\\mouse_liver_correct --output D:\\datasets\\mouse_liver_split")
    print("Then retrain B2:")
    print("  python scripts\\mouse_liver\\v2\\train_full.py --batch b2 --data-root D:\\datasets\\mouse_liver_split")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
