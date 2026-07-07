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

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\v2\prepare_data.py --data-root D:\datasets\mouse_liver_correct --output D:\datasets\mouse_liver_split

    # 或用默认路径 (DATA_BASE 在 fl_sequential.py 里定义)
    python scripts\mouse_liver\v2\prepare_data.py

输出结构:
    {output}/
    ├── b1/
    │   ├── full/           # 全量训练集 (6张)
    │   │   ├── images/
    │   │   ├── labels/
    │   │   └── data.yaml
    │   ├── fewshot/        # 3-shot 训练集 (3张, 从full的6张中选)
    │   │   ├── images/
    │   │   ├── labels/
    │   │   └── data.yaml
    │   ├── val/            # 验证集 (2张, 选模型)
    │   │   ├── images/
    │   │   └── labels/
    │   └── test/           # 测试集 (2张, 最终评估)
    │       ├── images/
    │       └── labels/
    ├── b2/ (同结构)
    └── b3/ (同结构, full=12, fewshot=3, val=4, test=4)

Note:
    - fewshot 的 3 张从 full 的 train 中选取 (前3张, 固定)
    - val 和 test 互斥 (不重叠)
    - 所有 split 都用 images/labels 目录名 (Ultralytics img2label_paths 铁律)
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
        # 也试试 batch1 vs b1
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

    # 创建目录结构
    splits = {
        'full': train_files,       # 全量训练
        'fewshot': fewshot_files,  # 3-shot
        'val': val_files,          # 验证集
        'test': test_files,        # 测试集
    }

    for split_name, files in splits.items():
        split_dir = batch_output / split_name
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)

        # 查找原始图片和标签
        # 优先顺序: batch_dir/images > batch_dir/yolo_format/images > data_root/yolo_format/images
        src_img_dir = batch_dir / 'images'
        src_lbl_dir = batch_dir / 'labels'

        if not src_img_dir.exists():
            src_img_dir = batch_dir / 'yolo_format' / 'images'
            src_lbl_dir = batch_dir / 'yolo_format' / 'labels'

        if not src_img_dir.exists():
            # 云 VM 上 batch 图片可能在全局 yolo_format/images
            src_img_dir = Path(data_root) / 'yolo_format' / 'images'

        if not src_lbl_dir.exists():
            src_lbl_dir = batch_dir / 'labels'

        print(f"  Source images: {src_img_dir} (exists={src_img_dir.exists()})")
        print(f"  Source labels: {src_lbl_dir} (exists={src_lbl_dir.exists()})")

        for img_name in files:
            # 复制图片
            src_img = src_img_dir / img_name
            if not src_img.exists():
                # 尝试 .jpg
                src_img = src_img_dir / (os.path.splitext(img_name)[0] + '.jpg')
            if src_img.exists():
                shutil.copy2(src_img, split_dir / 'images' / src_img.name)
            else:
                print(f"  [WARN] Image not found: {src_img}")

            # 复制标签
            lbl_name = os.path.splitext(src_img.name)[0] + '.txt'
            src_lbl = src_lbl_dir / lbl_name
            if src_lbl.exists():
                shutil.copy2(src_lbl, split_dir / 'labels' / lbl_name)
            else:
                print(f"  [WARN] Label not found: {src_lbl}")

        # 创建 data.yaml (full 和 fewshot 需要, val/test 不需要)
        if split_name in ('full', 'fewshot'):
            # val 指向 val split 的 images 目录 (绝对路径)
            val_dir = batch_output / 'val' / 'images'
            yaml_content = f"""path: {split_dir.resolve()}
train: images
val: {val_dir.resolve()}
nc: 1
names: ['organoid']
"""
            yaml_path = split_dir / 'data.yaml'
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            print(f"  {split_name}/data.yaml created (val → {val_dir})")

    # 创建 test 的 data.yaml (用于评估, val=test)
    test_yaml = batch_output / 'test' / 'data.yaml'
    test_img_dir = batch_output / 'test' / 'images'
    yaml_content = f"""path: {(batch_output / 'test').resolve()}
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

    # 保存全局 split 信息
    global_info_path = Path(args.output) / 'split_info.json'
    with open(global_info_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"Global split info: {global_info_path}")
    print(f"Done! All splits created under: {args.output}")


if __name__ == '__main__':
    main()
