r"""
从鼠肝红线标注图提取 YOLO 格式标签 (v2 — 修复文件名配对 bug)

Bug: v1 按 sorted 文件名索引配对标注图和原图
  标注图: 微信图片_20260522151759_46_2.jpg
  原图:   20260109-152624-142.jpg
  排序顺序不一致 → 标签张冠李戴 → B2 训练 loss=15.4, mAP=0

Fix: v2 通过 annotations.json 获取正确对应关系
  annotations.json 的每条记录有:
    'image': 'image_00.jpg'           ← 重命名后的图名
    'source_annotated': '微信图片_xxx.jpg'  ← 标注图原始文件名
    'source_original': '20260109-xxx.jpg'   ← 原图原始文件名
    'bboxes': [{'x':..., 'y':..., 'w':..., 'h':...}]
    'image_size': [W, H]

  v2 直接用 annotations.json 的 bboxes 生成标签
  不再依赖红线检测, 彻底避免文件名配对问题

Usage:
    python scripts\mouse_liver\extract_labels_v2.py --data-root D:\datasets\mouse_liver_correct

    会处理 batch1/batch2/batch3, 在每个 batch 目录下生成 labels/ 目录
"""
import argparse
import json
import os
from pathlib import Path


def bbox_to_yolo(bbox, img_w, img_h):
    """annotations.json bbox → YOLO format (normalized cx cy w h)"""
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def process_batch(batch_dir, annot_dir=None):
    """处理单个 batch, 从 annotations.json 生成 YOLO labels"""
    annot_json_path = batch_dir / 'annotations.json'
    if not annot_json_path.exists():
        print(f"  [ERROR] {annot_json_path} not found")
        return 0

    with open(annot_json_path, encoding='utf-8') as f:
        annotations = json.load(f)

    # labels 输出目录
    labels_dir = batch_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 清空旧标签
    for old in labels_dir.glob('*.txt'):
        old.unlink()

    n_images = 0
    n_boxes = 0
    for entry in annotations:
        img_name = entry['image']  # 'image_00.jpg'
        bboxes = entry.get('bboxes', [])
        img_size = entry.get('image_size', [0, 0])
        img_w, img_h = img_size[0], img_size[1]

        # 生成 YOLO label
        lines = [bbox_to_yolo(bb, img_w, img_h) for bb in bboxes]
        out_name = Path(img_name).stem + '.txt'
        with open(labels_dir / out_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        n_images += 1
        n_boxes += len(lines)
        print(f"  {out_name}: {len(lines)} boxes (image_size={img_w}x{img_h})")

    return n_images, n_boxes


def main():
    parser = argparse.ArgumentParser(
        description='从 annotations.json 生成 YOLO labels (修复文件名配对 bug)'
    )
    parser.add_argument('--data-root', default=r'D:\datasets\mouse_liver_correct',
                        help='Root directory containing batch1/batch2/batch3')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    print(f"Data root: {data_root}")

    for batch_name in ['batch1', 'batch2', 'batch3']:
        batch_dir = data_root / batch_name
        if not batch_dir.exists():
            print(f"\n[SKIP] {batch_dir} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {batch_name}")
        print(f"{'='*60}")

        n_img, n_box = process_batch(batch_dir)
        print(f"\n  Total: {n_img} images, {n_box} organoid boxes")
        print(f"  Labels saved to: {batch_dir / 'labels'}")

    print(f"\n{'='*60}")
    print("Done! Now re-run prepare_data.py to regenerate the split:")
    print(f"  python scripts\\mouse_liver\\v2\\prepare_data.py --data-root {data_root} --output D:\\datasets\\mouse_liver_split")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
