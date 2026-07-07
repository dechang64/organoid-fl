r"""
鼠肝 organoid 全量训练 — RF-DETR small

每个 batch 用全量 train (B1=6, B2=6, B3=12) 训练, 在 val 上选 best.pt, 在 test 上评估

参数 (基于文献调研 2026-07-06):
  resolution: B1=544, B2/B3=768 (12GB GPU 限制)
  grad_accum_steps: 4
  epochs: 20, early_stopping_patience: 10
  seed: 42

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    # 跑单个 batch
    python scripts\mouse_liver\v2\train_full.py --batch b1 --data-root D:\datasets\mouse_liver_split --pretrained output\checkpoint_best_regular.pth

    # 跑全部 batch
    python scripts\mouse_liver\v2\train_full.py --batch all --data-root D:\datasets\mouse_liver_split --pretrained output\checkpoint_best_regular.pth

输出:
    runs\mouse_liver_v2\{batch}\full\  — checkpoint + 训练日志
"""
import argparse
import os
import sys
import json
from pathlib import Path


# 参数决策 (基于文献调研)
BATCH_RESOLUTION = {
    'b1': 544,   # 2592×1944, organoid 161px@640, 大目标, 544 够用
    'b2': 768,   # 4000×3000, organoid 40px@640 → 48px@768, 小目标需高分辨率
    'b3': 768,   # 4000×3000, organoid 47px@640 → 56px@768
}

EPOCHS = 20
GRAD_ACCUM = 4
EARLY_STOPPING_PATIENCE = 10
SEED = 42


def train_batch(batch_name, data_root, pretrained_path, output_base, epochs=EPOCHS):
    """训练单个 batch 的全量模型"""
    from rfdetr import RFDETRSmall

    resolution = BATCH_RESOLUTION[batch_name]
    batch_dir = Path(data_root) / batch_name / 'full'
    data_yaml = batch_dir / 'data.yaml'

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    output_dir = Path(output_base) / batch_name / 'full'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {batch_name} (full)")
    print(f"{'='*60}")
    print(f"  Data: {batch_dir}")
    print(f"  Pretrained: {pretrained_path}")
    print(f"  Resolution: {resolution}")
    print(f"  Epochs: {epochs}")
    print(f"  Grad accum: {GRAD_ACCUM}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Seed: {SEED}")
    print(f"  Output: {output_dir}")

    # RF-DETR resolution 必须能被 32 整除
    assert resolution % 32 == 0, f"resolution must be divisible by 32, got {resolution}"

    model = RFDETRSmall(pretrain_weights=pretrained_path, num_classes=1)

    train_kwargs = {
        'dataset_dir': str(batch_dir.resolve()),
        'epochs': epochs,
        'grad_accum_steps': GRAD_ACCUM,
        'resolution': resolution,
        'output_dir': str(output_dir.resolve()),
        'early_stopping': True,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'seed': SEED,
        'num_workers': 2,
    }

    print(f"  kwargs: {train_kwargs}")
    model.train(**train_kwargs)

    # 检查 checkpoint
    ckpt_regular = output_dir / 'checkpoint_best_regular.pth'
    ckpt_ema = output_dir / 'checkpoint_best_ema.pth'
    print(f"\nTraining complete!")
    print(f"  Regular: {ckpt_regular} (exists={ckpt_regular.exists()})")
    print(f"  EMA: {ckpt_ema} (exists={ckpt_ema.exists()})")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='RF-DETR full training on mouse liver organoids')
    parser.add_argument('--batch', required=True, choices=['b1', 'b2', 'b3', 'all'],
                        help='Which batch to train')
    parser.add_argument('--data-root', default=r'D:\datasets\mouse_liver_split',
                        help='Root directory containing b1/b2/b3 splits')
    parser.add_argument('--pretrained', required=True,
                        help='Pretrained RF-DETR checkpoint (MultiOrg)')
    parser.add_argument('--output', default='runs/mouse_liver_v2',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    batches = ['b1', 'b2', 'b3'] if args.batch == 'all' else [args.batch]

    for batch in batches:
        train_batch(batch, args.data_root, args.pretrained, args.output, args.epochs)

    print(f"\n{'='*60}")
    print(f"All batches done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
