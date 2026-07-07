r"""
鼠肝 organoid 联邦学习 — 顺序链式 FL (v2)

实验矩阵:
  --gate: none(FedAvg) / soft(EWA) / hard(加权聚合, v2优化)
  --order: b1_b2_b3 / b3_b2_b1

v2 改进:
  - hard gate: 性能超标的节点参与加权聚合 (不是直接替换)
    new_global = w*local + (1-w)*global, w = signal_local/(signal_local+global_signal)
    不达标的节点权重=0 (跳过)
  - resolution: B1=544, B2/B3=768 (基于文献调研)
  - seed=42

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    # 跑单个实验
    python scripts\mouse_liver\v2\run_fl.py --gate hard --order b1_b2_b3 --tag F1

    # 跑全部实验矩阵
    for tag in F1 F2 F3 F4; do
        python scripts\mouse_liver\v2\run_fl.py --gate ... --tag $tag
    done

实验矩阵:
  F1: gate=none,    order=b1_b2_b3  (FedAvg baseline)
  F2: gate=soft,    order=b1_b2_b3  (EWA)
  F3: gate=hard,    order=b1_b2_b3  (加权聚合, v2)
  F4: gate=hard,    order=b3_b2_b1  (反序)

输出:
    runs\mouse_liver_v2\fl\{tag}\  — 每轮 global model + 评估
"""
import argparse
import os
import sys
import json
import time
import copy
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# 导入复用函数
sys.path.insert(0, str(Path(__file__).parent.parent))
from fl_sequential import (
    fedavg_aggregate, compute_ewa_weights, get_node_order, copy_sd_to_model, save_model_ckpt,
    safe_path, load_model, release_model,
    BATCH_SIZE, DEVICE, EWA_WARMUP_ROUNDS, OPTIMIZER, LR0,
)
# Windows 多进程会崩溃, 必须用 WORKERS=0 (fl_sequential.py 验证过)
WORKERS = 0

# v2 参数
NUM_ROUNDS = 10
LOCAL_EPOCHS = 10
IMGSZ = 640  # YOLO 统一用 640 (FL 不用 RF-DETR, 用 YOLOv12n)
CLOSE_MOSAIC = 5  # 最后5轮关闭 mosaic
# WORKERS, OPTIMIZER, LR0 从 fl_sequential import (保持一致)
SEED = 42

DATA_BASE = r"D:\datasets\mouse_liver_split"
BATCH_RESOLUTION = {'b1': 544, 'b2': 768, 'b3': 768}
OUTPUT_BASE = os.environ.get('MOUSE_LIVER_RUNS', r"runs") + r"\mouse_liver_v2\fl"


def hard_gate_v2(global_sd, local_sd, signal_local, global_signal_cache, n_local, global_data_count):
    """v2 硬门控: 性能超标的节点参与加权聚合 (不是替换)"""
    if global_sd is None:
        # 首个节点: 直接用本地模型
        return local_sd, signal_local, n_local, [1.0, 0.0]

    if signal_local > global_signal_cache:
        # 性能达标: 加权聚合
        # 权重 = signal_local / (signal_local + global_signal)
        # 性能越好, 本地权重越高
        total_signal = signal_local + global_signal_cache
        w_local = signal_local / total_signal if total_signal > 0 else 0.5
        w_global = 1 - w_local
        new_global = fedavg_aggregate([local_sd, global_sd], [w_local, w_global])
        new_signal = w_local * signal_local + w_global * global_signal_cache
        new_data_count = n_local + global_data_count
        return new_global, new_signal, new_data_count, [round(w_local, 4), round(w_global, 4)]
    else:
        # 性能不达标: 保持全局模型不变
        return global_sd, global_signal_cache, global_data_count, [0.0, 1.0]


def run_fl_experiment(gate, order, tag, data_root=DATA_BASE, output_base=OUTPUT_BASE, signal='mAP'):
    """运行单个 FL 实验"""
    import torch
    from ultralytics import YOLO

    output_dir = Path(output_base) / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(output_dir / 'experiment_log.txt', 'w', encoding='utf-8')

    def log_msg(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log_msg(f"{'='*60}")
    log_msg(f"FL Experiment: {tag}")
    log_msg(f"  Gate: {gate}")
    log_msg(f"  Order: {order}")
    log_msg(f"  Rounds: {NUM_ROUNDS}")
    log_msg(f"  Local epochs: {LOCAL_EPOCHS}")
    log_msg(f"  Close mosaic: {CLOSE_MOSAIC}")
    log_msg(f"  Seed: {SEED}")
    log_msg(f"  Output: {output_dir}")

    # 节点顺序
    nodes = order.split('_')
    log_msg(f"  Nodes: {nodes}")

    # 准备 val set (统一验证集, 用 b1 的 val)
    val_yaml = Path(data_root) / 'b1' / 'val' / 'data.yaml'
    if not val_yaml.exists():
        val_dir = Path(data_root) / 'b1' / 'val'
        val_yaml = val_dir / 'data.yaml'
        val_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(val_yaml, 'w', encoding='utf-8') as f:
            f.write(f"path: {val_dir.resolve()}\ntrain: images\nval: images\nnc: 1\nnames: ['organoid']\n")

    # === FL 从头训练 (COCO 预训练, 不用 init_model warmup) ===
    # 第一轮第一个节点: YOLO('yolo12n.pt') (COCO 80类) → train → 自动变 1 类
    # Ultralytics train() 会根据 data.yaml 的 nc 自动调整检测头
    # 后续节点: load_model(global_ckpt) → 1 类 → copy_sd_to_model(global_sd) → train
    # 
    # 保存全局模型: 用当前 model 对象 (train 后 val 前, 未 fuse) 保存
    # 不需要 init_ckpt 做 template

    # 初始化全局状态
    global_sd = None
    global_signal_cache = None
    global_data_count = 0
    global_model_path = None  # 第一轮第一个节点训练后才有
    template_ckpt = None  # 第一轮第一个节点训练后保存 (1类 template)

    # 本地最优模型 (local gate 用, 保留接口)
    node_best_sd = {}
    node_best_map = {}

    local_metrics_history = []
    all_round_results = []

    for round_idx in range(NUM_ROUNDS):
        log_msg(f"\n{'='*60}")
        log_msg(f"Round {round_idx + 1}/{NUM_ROUNDS}")
        log_msg(f"{'='*60}")

        node_order = get_node_order(order, round_idx, local_metrics_history)
        round_local_metrics = []
        round_weights = []

        for node_name in node_order:
            log_msg(f"\n  [{node_name}] Training...")

            # 准备节点数据
            node_data_dir = Path(data_root) / node_name / 'full'
            if not node_data_dir.exists():
                log_msg(f"    [WARN] {node_data_dir} not found, skipping")
                continue

            # 创建 fl_split (复用 fl_sequential 的 write_node_yaml)
            fl_split_dir = output_dir / f'round_{round_idx+1}' / node_name
            fl_split_dir.mkdir(parents=True, exist_ok=True)

            # 写 data.yaml
            node_yaml = fl_split_dir / 'data.yaml'
            train_img_dir = node_data_dir / 'train' / 'images'
            train_lbl_dir = node_data_dir / 'train' / 'labels'

            # 创建 fl_split 子目录 (images/labels 必须叫这名, Ultralytics img2label_paths 铁律)
            img_link_dir = fl_split_dir / 'images'
            lbl_link_dir = fl_split_dir / 'labels'
            img_link_dir.mkdir(parents=True, exist_ok=True)
            lbl_link_dir.mkdir(parents=True, exist_ok=True)

            # 清空旧文件
            for old in img_link_dir.iterdir():
                try: old.unlink()
                except: pass
            for old in lbl_link_dir.iterdir():
                try: old.unlink()
                except: pass

            # 复制图片和标签到 fl_split
            for img_file in train_img_dir.glob('*.[jJ][pP][gG]'):
                shutil.copy2(img_file, img_link_dir / img_file.name)
            for lbl_file in train_lbl_dir.glob('*.txt'):
                shutil.copy2(lbl_file, lbl_link_dir / lbl_file.name)

            # 写 data.yaml (train 用相对路径 "images", 指向 fl_split_dir/images/)
            # 铁律: train 必须是相对路径, 不能是绝对路径
            # 否则 Ultralytics 在原路径找标签, 不在 fl_split/labels/ 找
            node_yaml = fl_split_dir / 'data.yaml'
            with open(node_yaml, 'w', encoding='utf-8') as f:
                f.write(f"path: {safe_path(str(fl_split_dir.resolve()))}\n")
                f.write(f"train: images\n")
                f.write(f"val: {safe_path(str((val_yaml.parent / 'images').resolve()))}\n")
                f.write(f"nc: 1\nnames: ['organoid']\n")

            # 加载模型: 从头训练 (COCO 预训练)
            # 第一轮第一个节点: YOLO('yolo12n.pt') (COCO 80类), train() 自动改检测头为 1 类
            # 后续节点: load_model(global_model_path) (1类) + copy_sd_to_model(global_sd)
            if global_model_path is not None and global_sd is not None:
                # 后续节点: 用上一轮的 global checkpoint 做 template
                model = load_model(str(global_model_path))
                loaded = copy_sd_to_model(model, global_sd)
                log_msg(f"    Loaded global_sd: {loaded}/{len(global_sd)} keys")
            else:
                # 第一轮第一个节点: 从 COCO 预训练开始
                model = load_model('yolo12n.pt')
                log_msg(f"    First node: starting from COCO pretrained yolo12n.pt")

            # 训练 (参数和 fl_sequential.py 保持一致, 确保可比性)
            results = model.train(
                data=str(node_yaml),
                epochs=LOCAL_EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH_SIZE,
                device=DEVICE,
                workers=WORKERS,
                cache=False,
                lr0=LR0,
                optimizer=OPTIMIZER,
                seed=SEED,
                cos_lr=True,
                close_mosaic=CLOSE_MOSAIC,
                project=str(fl_split_dir),
                name='train',
                exist_ok=True,
                verbose=False,
            )

            # 铁律: state_dict 必须在 val 前取 (model.val() 会 fuse, 不可逆)
            local_sd = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
            n_local = len(list(train_img_dir.glob('*.[jJ][pP][gG]')))

            # 第一轮第一个节点: 保存 model 为 template (1类, 未 fuse)
            # 后续轮用这个 template 保存 global_sd
            if template_ckpt is None:
                template_ckpt = output_dir / 'template.pt'
                save_model_ckpt(model, str(template_ckpt))
                log_msg(f"    Template saved (first node): {template_ckpt}")

            # 评估本地模型 (val 会 fuse model, 但 local_sd 已提取)
            metrics = model.val(
                data=str(val_yaml),
                imgsz=IMGSZ,
                device=DEVICE,
                project=str(fl_split_dir),
                name=f'r{round_idx+1}_{node_name}_val',
                exist_ok=True,
            )
            mAP50 = float(metrics.box.map50)
            mAP5095 = float(metrics.box.map)
            signal_local = mAP5095 if 'mAP' in signal else mAP50

            log_msg(f"    mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}, signal={signal_local:.4f}")

            # 门控聚合
            if gate == 'none':
                # FedAvg
                if global_sd is None:
                    global_sd = local_sd
                    global_data_count = n_local
                    w = [1.0, 0.0]
                else:
                    w_local = n_local / (n_local + global_data_count)
                    global_sd = fedavg_aggregate([local_sd, global_sd], [w_local, 1-w_local])
                    global_data_count += n_local
                    w = [round(w_local, 4), round(1-w_local, 4)]
            elif gate == 'soft':
                # EWA (前 warmup 轮用 FedAvg)
                if global_sd is None or round_idx < EWA_WARMUP_ROUNDS:
                    if global_sd is None:
                        global_sd = local_sd
                        global_data_count = n_local
                        w = [1.0, 0.0]
                    else:
                        w_local = n_local / (n_local + global_data_count)
                        global_sd = fedavg_aggregate([local_sd, global_sd], [w_local, 1-w_local])
                        global_data_count += n_local
                        w = [round(w_local, 4), round(1-w_local, 4)]
                else:
                    maps = [max(signal_local, 1e-8), max(global_signal_cache or 0, 1e-8)]
                    total = sum(maps)
                    w_local, w_global = maps[0]/total, maps[1]/total
                    global_sd = fedavg_aggregate([local_sd, global_sd], [w_local, w_global])
                    global_signal_cache = w_local * signal_local + w_global * (global_signal_cache or 0)
                    w = [round(w_local, 4), round(w_global, 4)]
            elif gate == 'hard':
                # v2 硬门控: 加权聚合 (不是替换)
                global_sd, global_signal_cache, global_data_count, w = hard_gate_v2(
                    global_sd, local_sd, signal_local, global_signal_cache, n_local, global_data_count
                )
                if w[0] > 0:
                    log_msg(f"    [hard] 聚合: w_local={w[0]:.4f}, w_global={w[1]:.4f}")
                else:
                    log_msg(f"    [hard] 跳过: signal={signal_local:.4f} <= global={global_signal_cache:.4f}")

            round_weights.append(w)
            round_local_metrics.append({
                'node': node_name,
                'mAP50': mAP50,
                'mAP50-95': mAP5095,
                'signal': signal_local,
                'weights': w,
            })

            # 释放模型
            del model
            torch.cuda.empty_cache()

        local_metrics_history.append(round_local_metrics)

        # 保存全局模型 (用 template_ckpt 做 template, 1类)
        # template_ckpt 是第一轮第一个节点训练后保存的 1 类模型
        # 后续轮用它做 template, copy_sd_to_model 写入 global_sd
        if global_sd is not None and template_ckpt is not None:
            global_model_path = output_dir / f'round_{round_idx+1}' / 'global.pt'
            global_model_path.parent.mkdir(parents=True, exist_ok=True)
            g_model = load_model(str(template_ckpt))
            loaded = copy_sd_to_model(g_model, global_sd)
            log_msg(f"  Save global: loaded {loaded}/{len(global_sd)} keys")
            save_model_ckpt(g_model, str(global_model_path))
            release_model(g_model)

        # 评估全局模型 (用 load_model 加载完整 checkpoint)
        if global_sd is not None:
            model = load_model(str(global_model_path))
            metrics = model.val(
                data=str(val_yaml),
                imgsz=IMGSZ,
                device=DEVICE,
                project=str(output_dir / f'round_{round_idx+1}'),
                name='global_val',
                exist_ok=True,
            )
            global_mAP50 = float(metrics.box.map50)
            global_mAP5095 = float(metrics.box.map)
            log_msg(f"\n  Global model: mAP50={global_mAP50:.4f}, mAP50-95={global_mAP5095:.4f}")
            all_round_results.append({
                'round': round_idx + 1,
                'global_mAP50': global_mAP50,
                'global_mAP50-95': global_mAP5095,
                'local_metrics': round_local_metrics,
            })
            del model
            torch.cuda.empty_cache()

    # 保存最终结果
    results_path = output_dir / 'fl_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'tag': tag,
            'gate': gate,
            'order': order,
            'num_rounds': NUM_ROUNDS,
            'local_epochs': LOCAL_EPOCHS,
            'rounds': all_round_results,
        }, f, indent=2, ensure_ascii=False)

    log_msg(f"\n{'='*60}")
    log_msg(f"Experiment {tag} complete!")
    log_msg(f"  Final global mAP50={all_round_results[-1]['global_mAP50']:.4f}")
    log_msg(f"  Final global mAP50-95={all_round_results[-1]['global_mAP50-95']:.4f}")
    log_msg(f"  Results: {results_path}")

    log_file.close()
    return all_round_results


def main():
    parser = argparse.ArgumentParser(description='Mouse liver FL experiment (v2)')
    parser.add_argument('--gate', required=True, choices=['none', 'soft', 'hard'])
    parser.add_argument('--order', default='b1_b2_b3', choices=['b1_b2_b3', 'b3_b2_b1', 'random', 'quality'])
    parser.add_argument('--tag', required=True, help='Experiment tag (e.g. F1)')
    parser.add_argument('--data-root', default=DATA_BASE)
    parser.add_argument('--output', default=OUTPUT_BASE)
    parser.add_argument('--signal', default='mAP', choices=['mAP', 'mAP50'], help='Signal for gate weighting')
    args = parser.parse_args()

    run_fl_experiment(args.gate, args.order, args.tag, args.data_root, args.output, args.signal)


if __name__ == '__main__':
    main()
