r"""
鼠肝 organoid few-shot 训练 — RF-DETR small 3-shot

每个 batch 用 3 张图训练 (NeurIPS FSOD 标准), 在 val 上选 best.pt, 在 test 上评估

参数 (基于文献调研 2026-07-06):
  resolution: B1=544, B2/B3=768
  grad_accum_steps: 4
  epochs: 20, early_stopping_patience: 10
  seed: 42
  3-shot: 从 full 的 6 张 train 中取前 3 张

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    # 跑单个 batch
    python scripts\mouse_liver\v2\train_fewshot.py --batch b2 --data-root D:\datasets\mouse_liver_split --pretrained output\checkpoint_best_regular.pth

    # 跑全部 batch
    python scripts\mouse_liver\v2\train_fewshot.py --batch all --data-root D:\datasets\mouse_liver_split --pretrained output\checkpoint_best_regular.pth

输出:
    runs\mouse_liver_v2\{batch}\fewshot\  — checkpoint + 训练日志
"""
import argparse
import os
import sys
import json
from pathlib import Path


BATCH_RESOLUTION = {
    'b1': 544,
    'b2': 768,
    'b3': 768,
}

EPOCHS = 20
GRAD_ACCUM = 4
EARLY_STOPPING_PATIENCE = 10
SEED = 42


def train_batch_fewshot(batch_name, data_root, pretrained_path, output_base, epochs=EPOCHS):
    """3-shot 训练"""
    from rfdetr import RFDETRSmall

    resolution = BATCH_RESOLUTION[batch_name]
    batch_dir = Path(data_root) / batch_name / 'fewshot'
    data_yaml = batch_dir / 'data.yaml'

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    output_dir = Path(output_base) / batch_name / 'fewshot'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {batch_name} (3-shot)")
    print(f"{'='*60}")
    print(f"  Data: {batch_dir}")
    print(f"  Pretrained: {pretrained_path}")
    print(f"  Resolution: {resolution}")
    print(f"  Epochs: {epochs}")
    print(f"  Grad accum: {GRAD_ACCUM}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Seed: {SEED}")
    print(f"  Output: {output_dir}")

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

    ckpt_regular = output_dir / 'checkpoint_best_regular.pth'
    ckpt_ema = output_dir / 'checkpoint_best_ema.pth'
    print(f"\nTraining complete!")
    print(f"  Regular: {ckpt_regular} (exists={ckpt_regular.exists()})")
    print(f"  EMA: {ckpt_ema} (exists={ckpt_ema.exists()})")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='RF-DETR 3-shot training on mouse liver organoids')
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
        train_batch_fewshot(batch, args.data_root, args.pretrained, args.output, args.epochs)

    print(f"\n{'='*60}")
    print(f"All batches done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
