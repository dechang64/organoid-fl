"""
检查 images/image_XX.jpg 是否对应 annotations.json 里的正确原图
通过对比 annotated/ 目录中的文件来验证

逻辑:
  annotations.json 说: image_00.jpg 的 source_annotated 是 "微信图片_xxx.jpg"
  annotated/微信图片_xxx.jpg 应该是 image_00.jpg 加了红线标注
  如果两者像素相似 (忽略红线), 说明配对正确
  如果不相似, 说明图片重命名搞错了

Usage:
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\mouse_liver\\v2\\check_image_naming.py
"""
import json
import os
import numpy as np
from PIL import Image
from pathlib import Path

DATA_ROOT = Path(r"D:\datasets\mouse_liver_correct")


def image_similarity(img1_path, img2_path):
    """对比两张图的相似度 (忽略红色像素)
    返回: 相似度 0-1, 或 -1 如果尺寸不同
    """
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))

    if img1.shape != img2.shape:
        return -1.0, img1.shape, img2.shape

    # 忽略红色像素 (标注线)
    r, g, b = img1[:,:,0].astype(int), img1[:,:,1].astype(int), img1[:,:,2].astype(int)
    red_mask = (r > 150) & (r - g > 50) & (r - b > 50)

    # 在非红色区域计算相似度
    non_red = ~red_mask
    if non_red.sum() == 0:
        return 0.0, img1.shape, img2.shape

    diff = np.abs(img1[non_red].astype(int) - img2[non_red].astype(int))
    mean_diff = diff.mean()
    similarity = 1.0 - mean_diff / 255.0

    return similarity, img1.shape, img2.shape


def check_batch(batch_name):
    """检查单个 batch 的图片命名"""
    batch_dir = DATA_ROOT / batch_name
    if not batch_dir.exists():
        print(f"\n[SKIP] {batch_dir} not found")
        return

    annot_path = batch_dir / 'annotations.json'
    if not annot_path.exists():
        print(f"\n[SKIP] {annot_path} not found")
        return

    with open(annot_path, encoding='utf-8') as f:
        annotations = json.load(f)

    images_dir = batch_dir / 'images'
    annotated_dir = batch_dir / 'annotated'

    if not annotated_dir.exists():
        print(f"\n[SKIP] {annotated_dir} not found")
        return

    print(f"\n{'='*70}")
    print(f"  {batch_name}: {len(annotations)} images")
    print(f"  images dir: {images_dir}")
    print(f"  annotated dir: {annotated_dir}")
    print(f"{'='*70}")

    # 获取 annotated 目录下所有文件
    annot_files = sorted([f for f in os.listdir(annotated_dir)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    n_correct = 0
    n_wrong = 0
    n_unknown = 0

    for ann in annotations:
        img_name = ann['image']  # image_00.jpg
        source_annotated = ann['source_annotated']  # 微信图片_xxx.jpg
        source_original = ann.get('source_original', '?')

        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"  [MISSING] {img_name}")
            continue

        # 在 annotated 目录找 source_annotated
        # 文件名可能是 GBK 编码的, 尝试多种匹配
        annot_path_expected = annotated_dir / source_annotated

        if not annot_path_expected.exists():
            # 尝试模糊匹配
            found = None
            for af in annot_files:
                if source_annotated in af or af in source_annotated:
                    found = annotated_dir / af
                    break
            if found is None:
                # 尝试按文件大小匹配
                img_size = os.path.getsize(img_path)
                for af in annot_files:
                    af_path = annotated_dir / af
                    af_size = os.path.getsize(af_path)
                    if abs(af_size - img_size) < 1000:  # 文件大小接近
                        found = af_path
                        break
            annot_path_expected = found

        if annot_path_expected is None or not annot_path_expected.exists():
            print(f"  [UNKNOWN] {img_name}: source_annotated='{source_annotated}' not found in annotated/")
            n_unknown += 1
            continue

        # 对比 images/image_XX.jpg 和 annotated/source_annotated
        sim, shape1, shape2 = image_similarity(str(img_path), str(annot_path_expected))

        if sim < 0:
            print(f"  [SIZE DIFF] {img_name} vs {annot_path_expected.name}: {shape1} vs {shape2}")
            n_wrong += 1
        elif sim > 0.95:
            print(f"  [OK]      {img_name} ↔ {annot_path_expected.name}: sim={sim:.4f}")
            n_correct += 1
        else:
            # 可能配对了错误的标注图, 尝试所有标注图找最佳匹配
            best_sim = -1
            best_match = None
            for af in annot_files:
                af_path = annotated_dir / af
                s, _, _ = image_similarity(str(img_path), str(af_path))
                if s > best_sim:
                    best_sim = s
                    best_match = af

            if best_match == annot_path_expected.name:
                print(f"  [LOW SIM] {img_name} ↔ {annot_path_expected.name}: sim={sim:.4f} (best match but low)")
                n_correct += 1
            else:
                print(f"  [WRONG]   {img_name} ↔ {annot_path_expected.name}: sim={sim:.4f}")
                print(f"            best match: {best_match} sim={best_sim:.4f}")
                n_wrong += 1

    print(f"\n  Summary: {n_correct} correct, {n_wrong} wrong, {n_unknown} unknown")


def main():
    print(f"Data root: {DATA_ROOT}")

    for batch in ['batch1', 'batch2', 'batch3']:
        check_batch(batch)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
