#!/usr/bin/env python3
"""
RF-DETR Fine-tune on MultiOrg — NMS-free Transformer 检测器

关键优势:
- NMS-free: 对密集 organoid 场景无 NMS 误删问题
- 60+ mAP on COCO: 首个 real-time SOTA transformer
- 12GB VRAM: RF-DETR Base at 560px 可跑 (auto batch)

Usage (冬生本地):
    # 安装
    pip install rfdetr

    # 训练 RF-DETR Base
    python train_rfdetr.py --data D:\\datasets\\MultiOrg_v3_512\\data.yaml --model base --epochs 200

    # 训练 RF-DETR Small（更快，12GB 更稳）
    python train_rfdetr.py --data D:\\datasets\\MultiOrg_v3_512\\data.yaml --model small --epochs 200

    # 评估
    python train_rfdetr.py --eval --checkpoint path/to/checkpoint.pt --data D:\\datasets\\MultiOrg_v3_512\\data.yaml
"""

import argparse
import os
import sys
import yaml
import time
import json
from pathlib import Path


def check_install():
    """检查 rfdetr 是否已安装"""
    try:
        import rfdetr
        print(f"rfdetr version: {rfdetr.__version__ if hasattr(rfdetr, '__version__') else 'unknown'}")
        return True
    except ImportError:
        print("[ERROR] rfdetr not installed. Run: pip install rfdetr")
        print("  Requires: Python>=3.10, PyTorch>=2.0, CUDA-compatible GPU")
        return False


def load_data_yaml(yaml_path):
    """加载 YOLO 格式 data.yaml"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"Dataset config:")
    print(f"  path: {data.get('path')}")
    print(f"  train: {data.get('train')}")
    print(f"  val: {data.get('val')}")
    print(f"  nc: {data.get('nc')}")
    print(f"  names: {data.get('names')}")
    return data


def train_rfdetr(data_yaml, model_variant='base', epochs=200, imgsz=512,
                 batch_size=None, output_dir='./runs/rfdetr'):
    """Fine-tune RF-DETR on MultiOrg.

    Args:
        model_variant: 'nano', 'small', 'base', 'large', 'medium'
        epochs: 200 for small datasets
        imgsz: 512 for MultiOrg patches
        batch_size: None = auto (recommended for 12GB VRAM)
    """
    if not check_install():
        return

    from rfdetr import RFDETRBase, RFDETRSmall, RFDETRLarge, RFDETRMedium
    from rfdetr import RFDETRNano

    data = load_data_yaml(data_yaml)
    assert data['nc'] == 1, f"MultiOrg v3 expects 1 class, got {data['nc']}"

    # 选择模型
    model_map = {
        'nano': RFDETRNano,
        'small': RFDETRSmall,
        'base': RFDETRBase,
        'medium': RFDETRMedium,
        'large': RFDETRLarge,
    }
    if model_variant not in model_map:
        print(f"[ERROR] Unknown model: {model_variant}. Choose from {list(model_map.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"RF-DETR {model_variant.upper()} Fine-tune on MultiOrg")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {'auto' if batch_size is None else batch_size}")
    print(f"  Output: {output_dir}")

    # 实例化模型
    ModelClass = model_map[model_variant]
    model = ModelClass()

    # 训练
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    dataset_dir = os.path.dirname(os.path.abspath(data_yaml))
    train_kwargs = {
        'dataset_dir': dataset_dir,
        'epochs': epochs,
        'grad_accum_steps': 4,  # 3060 12GB 需要梯度累积
        'img_size': imgsz,
    }
    if batch_size is not None:
        train_kwargs['batch_size'] = batch_size

    print(f"\nStarting training...")
    print(f"  kwargs: {train_kwargs}")

    model.train(**train_kwargs)

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nTraining completed in {hours:.2f} hours ({elapsed:.0f}s)")

    # 评估
    print(f"\nEvaluating on validation set...")
    val_dir = os.path.join(os.path.dirname(data_yaml), 'test')
    metrics = model.evaluate(val_dir)
    print(f"\nValidation metrics:")
    print(json.dumps(metrics, indent=2, default=str))

    # 保存结果
    results = {
        'model': f'rf-detr-{model_variant}',
        'epochs': epochs,
        'imgsz': imgsz,
        'training_time_hours': hours,
        'metrics': metrics,
    }
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


def evaluate_rfdetr(checkpoint_path, data_yaml, imgsz=512):
    """评估训练好的 RF-DETR 模型"""
    if not check_install():
        return

    from rfdetr import RFDETRBase

    data = load_data_yaml(data_yaml)
    model = RFDETRBase(pretrain_weights=checkpoint_path)

    val_dir = os.path.join(os.path.dirname(data_yaml), 'test')
    metrics = model.evaluate(val_dir)
    print(f"\nEvaluation results:")
    print(json.dumps(metrics, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description='RF-DETR Fine-tune on MultiOrg')
    parser.add_argument('--data', required=True, help='Path to data.yaml')
    parser.add_argument('--model', default='base',
                        choices=['nano', 'small', 'base', 'medium', 'large'],
                        help='RF-DETR model variant')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='None=auto (recommended for 12GB)')
    parser.add_argument('--output', default='./runs/rfdetr')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluation only')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint path for evaluation')
    args = parser.parse_args()

    if args.eval:
        if not args.checkpoint:
            print("[ERROR] --checkpoint required for --eval mode")
            sys.exit(1)
        evaluate_rfdetr(args.checkpoint, args.data, args.imgsz)
    else:
        train_rfdetr(args.data, args.model, args.epochs, args.imgsz,
                     args.batch_size, args.output)


if __name__ == '__main__':
    main()
