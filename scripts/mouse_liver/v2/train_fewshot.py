r"""
鼠肝 organoid 跨域迁移 — B1 checkpoint → B2/B3 3-shot 微调

实验设计 (2026-07-06 确认):
  Step 2 跨域迁移:
    B1→B2 zeroshot: B1 模型直接推理 B2 (不微调, 用 evaluate.py 跑)
    B1→B2 few-shot: B1 checkpoint + B2 3张微调
    B1→B3 zeroshot: B1 模型直接推理 B3
    B1→B3 few-shot: B1 checkpoint + B3 3张微调

本脚本只做 few-shot 微调 (zeroshot 用 evaluate.py)
pretrained = B1 的全量训练 checkpoint (runs\mouse_liver_v2\b1\full\checkpoint_best_regular.pth)

参数:
  resolution: B2/B3 用 768 (B1 checkpoint 在 544 训练, 但微调用 768)
  batch_size: 1, grad_accum: 1
  epochs: 20, early_stopping_patience: 10
  seed: 42

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    # B1→B2 few-shot
    python scripts\mouse_liver\v2\train_fewshot.py --target b2 --data-root D:\datasets\mouse_liver_split --b1-ckpt runs\mouse_liver_v2\b1\full\checkpoint_best_regular.pth

    # B1→B3 few-shot
    python scripts\mouse_liver\v2\train_fewshot.py --target b3 --data-root D:\datasets\mouse_liver_split --b1-ckpt runs\mouse_liver_v2\b1\full\checkpoint_best_regular.pth

    # 全部 (B2+B3)
    python scripts\mouse_liver\v2\train_fewshot.py --target all --data-root D:\datasets\mouse_liver_split --b1-ckpt runs\mouse_liver_v2\b1\full\checkpoint_best_regular.pth

输出:
    runs\mouse_liver_v2\{b2,b3}\fewshot\  — checkpoint + 训练日志
"""
import argparse
import os
import sys
import json
from pathlib import Path


TARGET_CONFIG = {
    'b2': {'resolution': 768, 'batch_size': 1, 'grad_accum': 1},
    'b3': {'resolution': 768, 'batch_size': 1, 'grad_accum': 1},
}

EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10
SEED = 42


def train_fewshot(target_batch, data_root, b1_checkpoint, output_base, epochs=EPOCHS):
    """B1 checkpoint → target batch 3-shot 微调"""
    from rfdetr import RFDETRSmall

    config = TARGET_CONFIG[target_batch]
    resolution = config['resolution']
    batch_size = config['batch_size']
    grad_accum = config['grad_accum']

    # fewshot 数据目录 (prepare_data.py 创建)
    batch_dir = Path(data_root) / target_batch / 'fewshot'
    data_yaml = batch_dir / 'data.yaml'

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    if not Path(b1_checkpoint).exists():
        raise FileNotFoundError(f"B1 checkpoint not found: {b1_checkpoint}")

    output_dir = Path(output_base) / target_batch / 'fewshot'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training B1→{target_batch} (3-shot transfer)")
    print(f"{'='*60}")
    print(f"  Data: {batch_dir}")
    print(f"  Pretrained: B1 full checkpoint ({b1_checkpoint})")
    print(f"  Resolution: {resolution}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accum: {grad_accum}")
    print(f"  Epochs: {epochs}")
    print(f"  Seed: {SEED}")
    print(f"  Output: {output_dir}")

    assert resolution % 32 == 0

    # 用 B1 的全量训练 checkpoint 做 pretrained
    model = RFDETRSmall(pretrain_weights=b1_checkpoint, num_classes=1)

    train_kwargs = {
        'dataset_dir': str(batch_dir.resolve()),
        'epochs': epochs,
        'grad_accum_steps': grad_accum,
        'resolution': resolution,
        'batch_size': batch_size,
        'output_dir': str(output_dir.resolve()),
        'early_stopping': True,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'seed': SEED,
        'num_workers': 0,
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
    parser = argparse.ArgumentParser(description='B1→B2/B3 3-shot transfer training')
    parser.add_argument('--target', required=True, choices=['b2', 'b3', 'all'],
                        help='Target batch (B1 is source)')
    parser.add_argument('--data-root', default=r'D:\datasets\mouse_liver_split')
    parser.add_argument('--b1-ckpt', required=True,
                        help='B1 full training checkpoint (pretrained)')
    parser.add_argument('--output', default='runs/mouse_liver_v2')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    targets = ['b2', 'b3'] if args.target == 'all' else [args.target]

    for target in targets:
        train_fewshot(target, args.data_root, args.b1_ckpt, args.output, args.epochs)

    print(f"\n{'='*60}")
    print(f"All few-shot transfers done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
