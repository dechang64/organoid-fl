r"""
鼠肝 organoid 全量训练 — RF-DETR small

每个 batch 用全量 train (B1=6, B2=6, B3=12) 训练, 在 val 上选 best.pt, 在 test 上评估

参数 (基于文献调研 2026-07-06 + 12GB 实测):
  resolution: B1=544, B2/B3=768 (organoid 小需高分辨率)
  batch_size: B1=4 (544 安全), B2/B3=1 (768 在 12GB 上 batch>1 会 OOM)
  grad_accum_steps: 1 (数据量少, 每 step 都更新)
  epochs: 20, early_stopping_patience: 10
  seed: 42

⚠️ 不要用 batch_size='auto'!
  RF-DETR 的 auto_batch 探测会先试大 batch, 768 resolution 直接 OOM 崩溃
  (2026-07-07 冬生本地实测: auto_batch probe 在 generalized_box_iou 处 OOM)

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    # 跑单个 batch
    python scripts\mouse_liver\v2\train_full.py --batch b1 --data-root D:\datasets\mouse_liver_split 

    # 跑全部 batch
    python scripts\mouse_liver\v2\train_full.py --batch all --data-root D:\datasets\mouse_liver_split 

输出:
    runs\mouse_liver_v2\{batch}\full\  — checkpoint + 训练日志
"""
import argparse
import os
import sys
import json
from pathlib import Path


# 参数决策 (基于文献调研 + 12GB 实测)
# B1: 2592×1944, organoid ~161px@640, 大目标, 544 够用, batch=4 安全
# B2: 4000×3000, organoid ~40px@640 → 48px@768, 小目标需高分辨率
# B3: 4000×3000, organoid ~47px@640 → 56px@768
# 768 是 512 默认的 2.25x 显存, batch=4 必 OOM, 用 batch=1 + grad_accum=1 (每 step 更新)
BATCH_CONFIG = {
    'b1': {'resolution': 544, 'batch_size': 4, 'grad_accum': 1},   # 等效 batch=4
    'b2': {'resolution': 768, 'batch_size': 1, 'grad_accum': 1},   # 等效 batch=1
    'b3': {'resolution': 768, 'batch_size': 1, 'grad_accum': 1},   # 等效 batch=1
}

EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10
SEED = 42


def train_batch(batch_name, data_root, output_base, epochs=EPOCHS):
    """训练单个 batch 的全量模型 (COCO 预训练, 从头训练)"""
    from rfdetr import RFDETRSmall

    config = BATCH_CONFIG[batch_name]
    resolution = config['resolution']
    batch_size = config['batch_size']
    grad_accum = config['grad_accum']

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
    print(f"  Pretrained: COCO (RF-DETR default)")
    print(f"  Resolution: {resolution}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accum: {grad_accum} (effective batch={batch_size * grad_accum})")
    print(f"  Epochs: {epochs}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Seed: {SEED}")
    print(f"  Output: {output_dir}")

    # RF-DETR resolution 必须能被 32 整除
    assert resolution % 32 == 0, f"resolution must be divisible by 32, got {resolution}"

    model = RFDETRSmall(num_classes=1)  # COCO 预训练, 不用 MultiOrg checkpoint

    train_kwargs = {
        'dataset_dir': str(batch_dir.resolve()),
        'epochs': epochs,
        'grad_accum_steps': grad_accum,
        'resolution': resolution,
        'batch_size': batch_size,  # 固定 batch_size, 不用 'auto'
        'output_dir': str(output_dir.resolve()),
        'early_stopping': True,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'seed': SEED,
        'num_workers': 0,  # Windows 必须用 0, 和 FL 一致
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
    parser.add_argument('--output', default='runs/mouse_liver_v2',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    batches = ['b1', 'b2', 'b3'] if args.batch == 'all' else [args.batch]

    for batch in batches:
        train_batch(batch, args.data_root, args.output, args.epochs)

    print(f"\n{'='*60}")
    print(f"All batches done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
