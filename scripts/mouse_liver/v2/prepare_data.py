r"""
鼠肝 organoid 实验数据分配脚本

数据分配方案 (冬生提出, 2026-07-06):
  B1 (10张, 2592×1944): 6 train + 2 test + 2 val
  B2 (10张, 4000×3000): 6 train + 2 test + 2 val
  B3 (20张, 4000×3000): 12 train + 4 test + 4 val

关键改进:
  - 独立 test set (不参与训练/选模型)
  - val 只用于选模型 (early stopping best.pt)
  - test 只用于最终评估
  - seed=42 保证可复现

RF-DETR 数据格式 (Roboflow YOLO):
  dataset_dir/
    data.yaml
    train/
      images/
      labels/
    valid/
      images/
      labels/
    test/           (可选)
      images/
      labels/

  RF-DETR 不读 data.yaml 里的 train/val 路径!
  它用固定路径: root/train/images, root/valid/images
  (源码: rfdetr/datasets/__init__.py build_roboflow_from_yolo)

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\v2\prepare_data.py --data-root D:\datasets\mouse_liver_correct --output D:\datasets\mouse_liver_split

输出结构:
    {output}/
    ├── b1/
    │   ├── full/           # 全量训练集 (6张)
    │   │   ├── data.yaml
    │   │   ├── train/
    │   │   │   ├── images/
    │   │   │   └── labels/
    │   │   └── valid/      # 符号链接到 ../val/
    │   │       ├── images/
    │   │       └── labels/
    │   ├── fewshot/        # 3-shot (同 full 结构)
    │   ├── val/            # 验证集 (2张)
    │   │   ├── images/
    │   │   └── labels/
    │   └── test/           # 测试集 (2张)
    │       ├── images/
    │       └── labels/
    ├── b2/ (同结构)
    └── b3/ (同结构, full=12, fewshot=3, val=4, test=4)

Note:
    - fewshot 的 3 张从 full 的 train 中选取 (前3张, 固定)
    - val 和 test 互斥 (不重叠)
    - full/fewshot 的 valid/ 复制 val 的数据 (RF-DETR 要 valid/ 不是 val/)
"""
import argparse
import os
import json
import shutil
import random
from pathlib import Path


# 数据分配方案
SPLIT_PLAN = {
    'b1': {'total': 10, 'train': 6, 'test': 2, 'val': 2, 'fewshot': 3},
    'b2': {'total': 10, 'train': 6, 'test': 2, 'val': 2, 'fewshot': 3},
    'b3': {'total': 20, 'train': 12, 'test': 4, 'val': 4, 'fewshot': 3},
}

DEFAULT_DATA_ROOT = r"D:\datasets\mouse_liver_correct"
DEFAULT_OUTPUT = r"D:\datasets\mouse_liver_split"


def split_batch(batch_name, plan, data_root, output_root, seed=42):
    """为单个 batch 创建 train/val/test/fewshot split"""
    batch_dir = Path(data_root) / batch_name.replace('b', 'batch')
    if not batch_dir.exists():
        batch_dir = Path(data_root) / batch_name
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    # 读取 annotations.json 获取图片列表
    annot_path = batch_dir / 'annotations.json'
    if not annot_path.exists():
        raise FileNotFoundError(f"annotations.json not found: {annot_path}")

    with open(annot_path, encoding='utf-8') as f:
        annotations = json.load(f)

    # 获取所有图片名
    image_names = [item['image'] for item in annotations]
    print(f"\n{batch_name}: {len(image_names)} images from {batch_dir}")

    # 固定随机种子, 打乱顺序
    random.seed(seed)
    shuffled = image_names.copy()
    random.shuffle(shuffled)

    # 分配: train + test + val
    n_train = plan['train']
    n_test = plan['test']
    n_val = plan['val']
    n_fewshot = plan['fewshot']

    train_files = shuffled[:n_train]
    test_files = shuffled[n_train:n_train + n_test]
    val_files = shuffled[n_train + n_test:n_train + n_test + n_val]

    # fewshot: 从 train 中取前 n_fewshot 张
    fewshot_files = train_files[:n_fewshot]

    print(f"  Train ({n_train}): {train_files}")
    print(f"  Test  ({n_test}): {test_files}")
    print(f"  Val   ({n_val}): {val_files}")
    print(f"  Few-shot ({n_fewshot}): {fewshot_files}")

    # 创建输出目录
    batch_output = Path(output_root) / batch_name

    # 清理旧数据
    if batch_output.exists():
        shutil.rmtree(batch_output)

    # 查找原始图片和标签
    src_img_dir = batch_dir / 'images'
    src_lbl_dir = batch_dir / 'labels'

    if not src_img_dir.exists():
        src_img_dir = batch_dir / 'yolo_format' / 'images'
        src_lbl_dir = batch_dir / 'yolo_format' / 'labels'

    if not src_img_dir.exists():
        src_img_dir = Path(data_root) / 'yolo_format' / 'images'

    if not src_lbl_dir.exists():
        src_lbl_dir = batch_dir / 'labels'

    print(f"  Source images: {src_img_dir} (exists={src_img_dir.exists()})")
    print(f"  Source labels: {src_lbl_dir} (exists={src_lbl_dir.exists()})")

    def copy_files(files, dst_img_dir, dst_lbl_dir):
        """复制图片和标签到目标目录"""
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_name in files:
            src_img = src_img_dir / img_name
            if not src_img.exists():
                src_img = src_img_dir / (os.path.splitext(img_name)[0] + '.jpg')
            if src_img.exists():
                shutil.copy2(src_img, dst_img_dir / src_img.name)
            else:
                print(f"  [WARN] Image not found: {src_img}")
            lbl_name = os.path.splitext(src_img.name)[0] + '.txt'
            src_lbl = src_lbl_dir / lbl_name
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl_dir / lbl_name)
            else:
                print(f"  [WARN] Label not found: {src_lbl}")

    # === 创建 val/ 目录 (独立, 不在任何训练目录下) ===
    val_dir = batch_output / 'val'
    copy_files(val_files, val_dir / 'images', val_dir / 'labels')

    # === 创建 test/ 目录 ===
    test_dir = batch_output / 'test'
    copy_files(test_files, test_dir / 'images', test_dir / 'labels')

    # === 创建 full/ 目录 (RF-DETR Roboflow 格式) ===
    # full/
    #   data.yaml
    #   train/images/ + train/labels/  (训练数据)
    #   valid/images/ + valid/labels/  (验证数据, 复制 val 的)
    for split_name, train_files_list in [('full', train_files), ('fewshot', fewshot_files)]:
        split_dir = batch_output / split_name
        train_sub = split_dir / 'train'
        valid_sub = split_dir / 'valid'

        # train/ 子目录
        copy_files(train_files_list, train_sub / 'images', train_sub / 'labels')

        # valid/ 子目录 (复制 val 的数据)
        copy_files(val_files, valid_sub / 'images', valid_sub / 'labels')

        # data.yaml (RF-DETR 不读这个, 但留着做参考)
        yaml_content = f"""# RF-DETR Roboflow YOLO format
# train data: train/images/
# valid data: valid/images/ (copied from val split)
path: {split_dir.resolve()}
train: train/images
val: valid/images
nc: 1
names: ['organoid']
"""
        yaml_path = split_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"  {split_name}/ created (train={len(train_files_list)}, valid={len(val_files)})")

    # === 创建 test/ 的 data.yaml (用于 evaluate.py) ===
    test_yaml = test_dir / 'data.yaml'
    yaml_content = f"""path: {test_dir.resolve()}
train: images
val: images
nc: 1
names: ['organoid']
"""
    with open(test_yaml, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # 保存 split 信息
    split_info = {
        'batch': batch_name,
        'seed': seed,
        'total': len(image_names),
        'splits': {
            'full': {'count': len(train_files), 'files': train_files},
            'fewshot': {'count': len(fewshot_files), 'files': fewshot_files},
            'val': {'count': len(val_files), 'files': val_files},
            'test': {'count': len(test_files), 'files': test_files},
        }
    }
    info_path = batch_output / 'split_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"  Split info saved: {info_path}")
    return split_info


def main():
    parser = argparse.ArgumentParser(description='Prepare mouse liver organoid experiment data splits')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                        help=f'Root directory containing batch1/batch2/batch3 (default: {DEFAULT_DATA_ROOT})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output directory (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"\nSplit plan:")
    for batch, plan in SPLIT_PLAN.items():
        print(f"  {batch}: {plan['train']} train + {plan['test']} test + {plan['val']} val + {plan['fewshot']} few-shot")

    os.makedirs(args.output, exist_ok=True)

    all_info = {}
    for batch_name, plan in SPLIT_PLAN.items():
        info = split_batch(batch_name, plan, args.data_root, args.output, args.seed)
        all_info[batch_name] = info

    global_info_path = Path(args.output) / 'split_info.json'
    with open(global_info_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"Global split info: {global_info_path}")
    print(f"Done! All splits created under: {args.output}")


if __name__ == '__main__':
    main()
