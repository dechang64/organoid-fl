r"""
鼠肝 organoid 集中式训练 (天花板) — RF-DETR small

把 B1+B2+B3 的所有 train 数据合到一个目录训练
- B1: 6 train, B2: 6 train, B3: 12 train → 共 24 train
- val 用 B1+B2+B3 的 val 合并 (共 8 val)
- test 各 batch 独立评估

参数: resolution=640 (混合分辨率折中), batch=2, grad_accum=1
  640 是 RF-DETR Small 默认 resolution, 在 12GB 上 batch=2 安全
  不用 768 (B2/B3 的小目标) 也不用 544 (B1 的大目标)
  640 是折中 — 集中式训练不看单 batch 最优, 看整体

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\v2\train_central.py --data-root D:\datasets\mouse_liver_split --output runs\mouse_liver_v2

输出:
    runs\mouse_liver_v2\central\  — checkpoint + 训练日志
"""
import argparse
import os
import sys
import json
import shutil
from pathlib import Path


EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10
SEED = 42
RESOLUTION = 640
BATCH_SIZE = 2
GRAD_ACCUM = 1


def prepare_central_dataset(data_root, output_dir):
    """把 B1+B2+B3 的 train 和 val 合并到一个目录"""
    central_dir = Path(output_dir) / 'central'
    train_img_dir = central_dir / 'train' / 'images'
    train_lbl_dir = central_dir / 'train' / 'labels'
    valid_img_dir = central_dir / 'valid' / 'images'
    valid_lbl_dir = central_dir / 'valid' / 'labels'

    for d in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        # 清空旧文件
        for old in d.iterdir():
            try: old.unlink()
            except: pass

    # 合并各 batch 的 train 数据
    for batch in ['b1', 'b2', 'b3']:
        batch_train_img = Path(data_root) / batch / 'full' / 'train' / 'images'
        batch_train_lbl = Path(data_root) / batch / 'full' / 'train' / 'labels'

        if not batch_train_img.exists():
            print(f"  [WARN] {batch} train images not found: {batch_train_img}")
            continue

        for img_file in batch_train_img.glob('*.[jJ][pP][gG]'):
            # 加 batch 前缀避免重名
            dst_name = f"{batch}_{img_file.name}"
            shutil.copy2(img_file, train_img_dir / dst_name)
            lbl_file = batch_train_lbl / (img_file.stem + '.txt')
            if lbl_file.exists():
                shutil.copy2(lbl_file, train_lbl_dir / f"{batch}_{lbl_file.name}")

        # 合并 valid 数据 (从各 batch 的 valid 目录)
        batch_valid_img = Path(data_root) / batch / 'full' / 'valid' / 'images'
        batch_valid_lbl = Path(data_root) / batch / 'full' / 'valid' / 'labels'

        if batch_valid_img.exists():
            for img_file in batch_valid_img.glob('*.[jJ][pP][gG]'):
                dst_name = f"{batch}_{img_file.name}"
                shutil.copy2(img_file, valid_img_dir / dst_name)
                lbl_file = batch_valid_lbl / (img_file.stem + '.txt')
                if lbl_file.exists():
                    shutil.copy2(lbl_file, valid_lbl_dir / f"{batch}_{lbl_file.name}")

    n_train = len(list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.JPG')))
    n_valid = len(list(valid_img_dir.glob('*.jpg')) + list(valid_img_dir.glob('*.JPG')))
    print(f"  Central dataset: {n_train} train, {n_valid} valid")

    # 写 data.yaml
    data_yaml = central_dir / 'data.yaml'
    with open(data_yaml, 'w', encoding='utf-8') as f:
        f.write(f"path: {central_dir.resolve()}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: valid/images\n")
        f.write(f"nc: 1\nnames: ['organoid']\n")

    return central_dir, data_yaml


def train_central(data_root, output_base, epochs=EPOCHS):
    """集中式训练 (天花板)"""
    from rfdetr import RFDETRSmall

    central_dir, data_yaml = prepare_central_dataset(data_root, output_base)

    output_dir = Path(output_base) / 'central'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training CENTRAL (ceiling)")
    print(f"{'='*60}")
    print(f"  Data: {central_dir}")
    print(f"  Pretrained: COCO (RF-DETR default)")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Grad accum: {GRAD_ACCUM}")
    print(f"  Epochs: {epochs}")
    print(f"  Seed: {SEED}")
    print(f"  Output: {output_dir}")

    assert RESOLUTION % 32 == 0

    model = RFDETRSmall(num_classes=1)

    train_kwargs = {
        'dataset_dir': str(central_dir.resolve()),
        'epochs': epochs,
        'grad_accum_steps': GRAD_ACCUM,
        'resolution': RESOLUTION,
        'batch_size': BATCH_SIZE,
        'output_dir': str(output_dir.resolve()),
        'early_stopping': True,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'seed': SEED,
        'num_workers': 0,
    }

    print(f"  kwargs: {train_kwargs}")
    model.train(**train_kwargs)

    ckpt_regular = output_dir / 'checkpoint_best_regular.pth'
    print(f"\nTraining complete!")
    print(f"  Regular: {ckpt_regular} (exists={ckpt_regular.exists()})")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='RF-DETR centralized training (ceiling)')
    parser.add_argument('--data-root', default=r'D:\datasets\mouse_liver_split')
    parser.add_argument('--output', default='runs/mouse_liver_v2')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    train_central(args.data_root, args.output, args.epochs)

    print(f"\n{'='*60}")
    print(f"Central training done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
