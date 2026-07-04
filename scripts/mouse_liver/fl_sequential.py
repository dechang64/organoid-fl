r"""
鼠肝方案2-扩展: 顺序链式 FL — 多维度实验矩阵

实验维度:
  --gate: none(无条件聚合) / hard(性能更好才更新) / soft(EWA加权) / local(本地门控存储)
  --order: b1_b2_b3 / b3_b2_b1 / random / quality
  --signal: mAP50 / mAP
  --margin: 硬门控的性能提升阈值 (默认0, 严格>)

实验矩阵 (8组):
  E4: sequential, gate=none,    order=b1_b2_b3
  E5: sequential, gate=hard,    order=b1_b2_b3
  E6: sequential, gate=soft,    order=b1_b2_b3
  E7: sequential, gate=hard,    order=b3_b2_b1
  E8: sequential, gate=local,   order=b1_b2_b3

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    # 跑单个实验
    python scripts\mouse_liver\fl_sequential.py --gate hard --order b1_b2_b3 --tag E5

    # 跑全部实验矩阵 (建议)
    python scripts\mouse_liver\run_sequential_matrix.bat

输出:
    runs\mouse_liver_fl_seq\{tag}\  — 每轮 global model + 评估 + 可视化
    runs\mouse_liver_fl_seq\comparison.png  — 全实验对比图
"""
import os, sys, json, time, copy, shutil, argparse
import numpy as np

DATA_BASE = r"D:\datasets\mouse_liver_data"
BATCH_DIRS = {
    'b1': os.path.join(DATA_BASE, 'batch1'),
    'b2': os.path.join(DATA_BASE, 'batch2'),
    'b3': os.path.join(DATA_BASE, 'batch3'),
}
OUTPUT_BASE = r"runs\mouse_liver_fl_seq"
NUM_ROUNDS = 10
WORKERS = 0
LOCAL_EPOCHS = 10
IMGSZ = 640
BATCH_SIZE = 4
DEVICE = 'cuda'
EWA_WARMUP_ROUNDS = 2

VAL_INDICES = {
    'b1': [7, 8, 9],
    'b2': [7, 8, 9],
    'b3': [17, 18, 19],
}

def log(msg):
    print(msg, flush=True)

def safe_path(p):
    return p.replace('\\', '/')


# ============ 数据准备 (复用 fl_train.py 逻辑) ============

def write_node_yaml(data_dir, node_name):
    node_yaml = os.path.join(data_dir, 'data.yaml')
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')
    train_img_dir = os.path.join(data_dir, 'train_images')
    train_lbl_dir = os.path.join(data_dir, 'train_labels')
    for d in [train_img_dir, train_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        for old in os.listdir(d):
            try:
                os.remove(os.path.join(d, old))
            except:
                pass
    val_indices = set(VAL_INDICES.get(node_name, []))
    for img_file in sorted(os.listdir(img_dir)):
        name = os.path.splitext(img_file)[0]
        try:
            idx = int(name.split('_')[1])
        except (IndexError, ValueError):
            continue
        if idx in val_indices:
            continue
        shutil.copy2(os.path.join(img_dir, img_file), os.path.join(train_img_dir, img_file))
        lbl_file = img_file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(lbl_dir, lbl_file)):
            shutil.copy2(os.path.join(lbl_dir, lbl_file), os.path.join(train_lbl_dir, lbl_file))
    with open(node_yaml, 'w') as f:
        f.write(f'path: {safe_path(data_dir)}\ntrain: train_images\nval: train_images\nnc: 1\nnames: [\'organoid\']\n')
    for cache_name in ['labels.cache', 'train_images.cache', 'train_labels.cache']:
        cache = os.path.join(data_dir, cache_name)
        if os.path.exists(cache):
            try:
                os.remove(cache)
            except PermissionError:
                pass
    return node_yaml


def prepare_val_set(output_dir):
    val_dir = os.path.join(output_dir, 'val_set')
    if os.path.exists(val_dir):
        try:
            shutil.rmtree(val_dir)
        except PermissionError:
            for root, dirs, files in os.walk(val_dir, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except:
                        pass
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    for node, ddir in BATCH_DIRS.items():
        img_src = os.path.join(ddir, 'images')
        lbl_src = os.path.join(ddir, 'labels')
        for idx in VAL_INDICES[node]:
            fname = f'image_{idx:02d}'
            img_path = os.path.join(img_src, f'{fname}.jpg')
            lbl_path = os.path.join(lbl_src, f'{fname}.txt')
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                shutil.copy2(img_path, os.path.join(val_dir, 'images', f'{node}_{fname}.jpg'))
                shutil.copy2(lbl_path, os.path.join(val_dir, 'labels', f'{node}_{fname}.txt'))
    val_cache = os.path.join(val_dir, 'labels.cache')
    if os.path.exists(val_cache):
        try:
            os.remove(val_cache)
        except PermissionError:
            pass
    val_yaml = os.path.join(output_dir, 'val.yaml')
    with open(val_yaml, 'w') as f:
        f.write(f'path: {safe_path(os.path.abspath(val_dir))}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
    return val_yaml


# ============ 模型加载 ============

def load_model(ckpt_path):
    from ultralytics import YOLO
    return YOLO(ckpt_path)

def release_model(model):
    import torch
    if hasattr(model, 'model'):
        del model.model
    if hasattr(model, 'predictor'):
        del model.predictor
    if hasattr(model, 'trainer'):
        del model.trainer
    model.__dict__.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============ 聚合 ============

def fedavg_aggregate(state_dicts, weights=None):
    import torch
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    avg = {}
    for key in state_dicts[0]:
        if state_dicts[0][key].dtype in (torch.int32, torch.int64):
            avg[key] = state_dicts[0][key].clone()
            continue
        avg[key] = sum(w * sd[key].cpu().float() for sd, w in zip(state_dicts, weights))
    return avg

def compute_ewa_weights(client_metrics, signal="mAP"):
    key = "mAP" if signal == "mAP" else "mAP50"
    maps = [max(m[key], 1e-8) for m in client_metrics]
    total = sum(maps)
    if total == 0:
        return [1.0 / len(maps)] * len(maps)
    return [m / total for m in maps]

def copy_sd_to_model(model, sd):
    """把 state_dict copy 到模型, 返回匹配数"""
    model_sd = model.model.state_dict()
    loaded = 0
    for key in sd:
        if key in model_sd and sd[key].shape == model_sd[key].shape:
            model_sd[key].copy_(sd[key])
            loaded += 1
    return loaded

def save_model_ckpt(model, path):
    import torch
    from copy import deepcopy
    from datetime import datetime
    from ultralytics import __version__
    ckpt = {
        'model': deepcopy(model.model).float(),
        'date': datetime.now().isoformat(),
        'version': __version__,
        'license': 'AGPL-3.0 License',
        'docs': 'https://docs.ultralytics.com',
        'train_args': dict(model.overrides) if hasattr(model, 'overrides') and hasattr(model.overrides, '__dict__') else {},
    }
    torch.save(ckpt, path)


# ============ 顺序链式 FL ============

def get_node_order(order_str, round_idx, local_metrics_history=None):
    """根据 order 策略返回节点顺序"""
    if order_str == 'b1_b2_b3':
        return ['b1', 'b2', 'b3']
    elif order_str == 'b3_b2_b1':
        return ['b3', 'b2', 'b1']
    elif order_str == 'random':
        import random
        nodes = ['b1', 'b2', 'b3']
        random.seed(round_idx)
        random.shuffle(nodes)
        return nodes
    elif order_str == 'quality' and local_metrics_history:
        # 按上一轮 mAP 降序排列
        last = local_metrics_history[-1] if local_metrics_history else []
        if len(last) == 3:
            sorted_nodes = sorted(range(3), key=lambda i: last[i].get('mAP50', 0), reverse=True)
            return [['b1', 'b2', 'b3'][i] for i in sorted_nodes]
        return ['b1', 'b2', 'b3']
    return ['b1', 'b2', 'b3']


def run_sequential_fl(args, init_ckpt, val_yaml, output_dir):
    """顺序链式 FL 主循环"""
    from ultralytics import YOLO

    strat_dir = os.path.join(output_dir, args.tag)
    os.makedirs(strat_dir, exist_ok=True)

    global_ckpt = init_ckpt
    global_sd = None  # 当前全局 state_dict (unfused)
    global_signal_cache = None  # 缓存全局模型性能 (避免重复评估)
    global_data_count = 0  # 全局已聚合的累积数据量 (用于 running average)
    fl_history = []
    local_metrics_history = []

    # 本地门控: 每个节点存储的"最佳"模型参数
    node_best_sd = {}  # {node_name: sd}
    node_best_map = {}  # {node_name: best_mAP}

    for round_idx in range(NUM_ROUNDS):
        node_order = get_node_order(args.order, round_idx, local_metrics_history)
        log(f"\n  --- {args.tag} Round {round_idx+1}/{NUM_ROUNDS} (order: {'→'.join(node_order)}) ---")

        round_local_metrics = []
        round_weights = []

        # 顺序处理每个节点
        for node_pos, node_name in enumerate(node_order):
            data_dir = BATCH_DIRS[node_name]
            log(f"    [{node_pos+1}/{len(node_order)}] 训练 {node_name}...")
            node_yaml = write_node_yaml(data_dir, node_name)

            # 加载当前全局模型
            model = load_model(global_ckpt)

            t0 = time.time()
            model.train(data=node_yaml, epochs=LOCAL_EPOCHS, imgsz=IMGSZ, batch=BATCH_SIZE,
                        device=DEVICE, workers=WORKERS, cache=False,
                        project=strat_dir, name=f'r{round_idx}_{node_name}',
                        exist_ok=True, cos_lr=True, close_mosaic=5, verbose=False)
            dt = time.time() - t0
            log(f"      {dt/60:.1f} min")

            # 在 val 前取 state_dict (unfused)
            local_sd = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}

            # 评估本地模型 (val 会 fuse, 但 sd 已取)
            val_res = model.val(data=val_yaml, device=DEVICE, project=strat_dir,
                                name=f'r{round_idx}_{node_name}_val', exist_ok=True)
            mAP50 = float(val_res.box.map50)
            mAP5095 = float(val_res.box.map)
            log(f"      {node_name}: mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}")

            release_model(model)
            model = None

            round_local_metrics.append({
                'node': node_name,
                'mAP50': round(mAP50, 4),
                'mAP': round(mAP5095, 4),
            })

            n_local = len(os.listdir(os.path.join(data_dir, 'train_images')))
            signal_local = mAP5095 if args.signal == 'mAP' else mAP50

            # === 更新全局参数 ===
            if args.gate == 'none':
                # 无门控: running average (累积数据量加权)
                if global_sd is None:
                    global_sd = local_sd
                    global_data_count = n_local
                else:
                    w_local = n_local / (n_local + global_data_count)
                    weights = [w_local, 1 - w_local]
                    global_sd = fedavg_aggregate([local_sd, global_sd], weights)
                    global_data_count += n_local
                round_weights.append(None)

            elif args.gate == 'hard':
                # 硬门控: 本地性能 > 全局性能 + margin 才更新
                if global_sd is None:
                    global_sd = local_sd
                    global_signal_cache = signal_local
                    global_data_count = n_local
                    log(f"      [gate] 初始化全局模型 ({args.signal}={signal_local:.4f})")
                else:
                    if signal_local > global_signal_cache + args.margin:
                        old_signal = global_signal_cache
                        global_sd = local_sd
                        global_signal_cache = signal_local
                        global_data_count = n_local
                        log(f"      [gate] 更新全局 ({signal_local:.4f} > {old_signal:.4f} + {args.margin})")
                    else:
                        log(f"      [gate] 保留旧全局 ({signal_local:.4f} <= {global_signal_cache:.4f} + {args.margin})")
                round_weights.append(None)

            elif args.gate == 'soft':
                # 软门控: EWA 加权 (前 warmup 轮用 FedAvg)
                if global_sd is None or round_idx < EWA_WARMUP_ROUNDS:
                    if global_sd is None:
                        global_sd = local_sd
                        global_data_count = n_local
                    else:
                        w_local = n_local / (n_local + global_data_count)
                        global_sd = fedavg_aggregate([local_sd, global_sd], [w_local, 1 - w_local])
                        global_data_count += n_local
                    round_weights.append(None)
                else:
                    # EWA: 按性能信号加权
                    maps = [max(signal_local, 1e-8), max(global_signal_cache or 0, 1e-8)]
                    total = sum(maps)
                    w_local, w_global = maps[0] / total, maps[1] / total
                    global_sd = fedavg_aggregate([local_sd, global_sd], [w_local, w_global])
                    round_weights.append([round(w_local, 4), round(w_global, 4)])
                    log(f"      [soft] weights: local={w_local:.3f}, global={w_global:.3f}")

            elif args.gate == 'local':
                # 本地门控: 各节点自己决定是否存储
                if node_name not in node_best_map or signal_local > node_best_map[node_name] + args.margin:
                    node_best_sd[node_name] = local_sd
                    node_best_map[node_name] = signal_local
                    log(f"      [local] {node_name} 存储新模型 ({signal_local:.4f})")
                else:
                    log(f"      [local] {node_name} 保留旧模型 ({signal_local:.4f} <= {node_best_map[node_name]:.4f})")

                # 全局模型 = 所有节点最佳模型的平均
                if len(node_best_sd) == 3:
                    sds = [node_best_sd[n] for n in ['b1', 'b2', 'b3'] if n in node_best_sd]
                    global_sd = fedavg_aggregate(sds)
                round_weights.append(None)

            # 保存更新后的全局模型
            if global_sd is not None:
                global_ckpt = os.path.join(strat_dir, f'global_r{round_idx}.pt')
                g_model = load_model(init_ckpt)
                loaded = copy_sd_to_model(g_model, global_sd)
                log(f"      Loaded {loaded}/{len(global_sd)} keys")
                save_model_ckpt(g_model, global_ckpt)
                release_model(g_model)

        local_metrics_history.append(round_local_metrics)

        # 评估全局模型
        g_model = load_model(global_ckpt)
        val_res = g_model.val(data=val_yaml, device=DEVICE, project=strat_dir,
                              name=f'r{round_idx}_global_val', exist_ok=True)
        g_mAP50 = float(val_res.box.map50)
        g_mAP5095 = float(val_res.box.map)
        log(f"    ★ Global: mAP50={g_mAP50:.4f}, mAP50-95={g_mAP5095:.4f}, P={val_res.box.mp:.4f}, R={val_res.box.mr:.4f}")

        # 更新全局性能缓存 (round-end 评估后)
        global_signal_cache = g_mAP5095 if args.signal == 'mAP' else g_mAP50

        fl_history.append({
            'round': round_idx + 1,
            'global_mAP50': round(g_mAP50, 4),
            'global_mAP': round(g_mAP5095, 4),
            'global_P': round(float(val_res.box.mp), 4),
            'global_R': round(float(val_res.box.mr), 4),
            'local': round_local_metrics,
            'weights': round_weights if round_weights else None,
            'node_order': node_order,
        })

        release_model(g_model)

    return fl_history


# ============ 可视化 ============

def plot_experiment(history, tag, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rounds = list(range(1, NUM_ROUNDS + 1))

    # mAP50 收敛曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = [r['global_mAP50'] for r in history]
    ax.plot(rounds, vals, 'o-', linewidth=2, markersize=6, label=tag)
    ax.set_xlabel('FL Round', fontsize=13)
    ax.set_ylabel('Global mAP50', fontsize=13)
    ax.set_title(f'{tag} — Convergence', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{tag}_convergence.png'), dpi=150)
    plt.close(fig)

    # 本地模型趋势
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, node in enumerate(['b1', 'b2', 'b3']):
        vals = []
        for r in history:
            for lm in r['local']:
                if lm['node'] == node:
                    vals.append(lm['mAP50'])
                    break
        if vals:
            ax.plot(rounds[:len(vals)], vals, 'o-', label=node, linewidth=2, markersize=5)
    ax.set_xlabel('FL Round', fontsize=13)
    ax.set_ylabel('Local mAP50', fontsize=13)
    ax.set_title(f'{tag} — Local mAP50 per Node', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{tag}_local_mAP50.png'), dpi=150)
    plt.close(fig)

    log(f"  可视化 → {output_dir}/{tag}_convergence.png")
    log(f"  可视化 → {output_dir}/{tag}_local_mAP50.png")


def visualize_detections(ckpt_path, img_dir, lbl_dir, dst_dir, device='cuda', imgsz=640):
    import cv2
    from ultralytics import YOLO
    model = YOLO(ckpt_path)
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    for img_file in images:
        img_path = os.path.join(img_dir, img_file)
        orig = cv2.imread(img_path)
        h, w = orig.shape[:2]
        results = model.predict(img_path, device=device, imgsz=imgsz, verbose=False, conf=0.25)
        vis = orig.copy()
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f'{conf:.2f}', (x1, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        lbl_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        lbl_path = os.path.join(lbl_dir, lbl_file)
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = map(float, parts)
                        gx1 = int((xc - bw/2) * w)
                        gy1 = int((yc - bh/2) * h)
                        gx2 = int((xc + bw/2) * w)
                        gy2 = int((yc + bh/2) * h)
                        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        cv2.putText(vis, 'Green=Detection  Red=GT', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(dst_dir, f'det_{img_file}'), vis)
    release_model(model)
    return len(images)


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description='顺序链式 FL 实验')
    parser.add_argument('--gate', choices=['none', 'hard', 'soft', 'local'], default='hard')
    parser.add_argument('--order', choices=['b1_b2_b3', 'b3_b2_b1', 'random', 'quality'], default='b1_b2_b3')
    parser.add_argument('--signal', choices=['mAP50', 'mAP'], default='mAP', help='门控性能信号')
    parser.add_argument('--margin', type=float, default=0.0, help='硬门控性能提升阈值')
    parser.add_argument('--tag', default='E5', help='实验标签')
    args = parser.parse_args()

    log(f"\n{'='*60}")
    log(f"实验: {args.tag}")
    log(f"  gate={args.gate}, order={args.order}, signal={args.signal}, margin={args.margin}")
    log(f"{'='*60}")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 准备 val_set
    val_yaml = prepare_val_set(OUTPUT_BASE)

    # 训练 init_model (复用 fl_train.py 的 init_model.pt)
    init_ckpt = os.path.join('runs', 'mouse_liver_fl', 'init_model.pt')
    if not os.path.exists(init_ckpt):
        log("\n=== Step 1: 训练 init_model (1 epoch, B1 数据) ===")
        from ultralytics import YOLO
        init_ckpt = os.path.join(OUTPUT_BASE, 'init_model.pt')
        node_yaml = write_node_yaml(BATCH_DIRS['b1'], 'b1')
        model = YOLO('yolo12n.pt')
        model.train(data=node_yaml, epochs=1, imgsz=IMGSZ, batch=BATCH_SIZE,
                    device=DEVICE, workers=WORKERS, cache=False,
                    project=OUTPUT_BASE, name='init', exist_ok=True,
                    cos_lr=True, verbose=False)
        import torch
        from copy import deepcopy
        from datetime import datetime
        from ultralytics import __version__
        ckpt = {
            'model': deepcopy(model.model).float(),
            'date': datetime.now().isoformat(),
            'version': __version__,
            'license': 'AGPL-3.0 License',
            'docs': 'https://docs.ultralytics.com',
            'train_args': dict(model.overrides) if hasattr(model, 'overrides') and hasattr(model.overrides, '__dict__') else {},
        }
        torch.save(ckpt, init_ckpt)
        log(f"init_model saved: {init_ckpt}")
        release_model(model)

    # === 顺序链式 FL ===
    history = run_sequential_fl(args, init_ckpt, val_yaml, OUTPUT_BASE)

    # 保存结果
    result_path = os.path.join(OUTPUT_BASE, f'{args.tag}_results.json')
    with open(result_path, 'w') as f:
        json.dump({
            'tag': args.tag,
            'config': vars(args),
            'history': history,
        }, f, indent=2)

    # 汇总
    log(f"\n{'='*60}")
    log(f"{args.tag} 汇总")
    log(f"{'='*60}")
    log(f"\n{'Round':<8} {'Global mAP50':<15} {'Global mAP50-95':<18}")
    log("-" * 41)
    for r in history:
        log(f"  R{r['round']:<6} {r['global_mAP50']:<15} {r['global_mAP']:<18}")
    log(f"\n最终 (Round {NUM_ROUNDS}): mAP50={history[-1]['global_mAP50']:.4f}, mAP50-95={history[-1]['global_mAP']:.4f}")

    # 可视化
    log(f"\n生成可视化...")
    plot_experiment(history, args.tag, OUTPUT_BASE)

    # 检测叠加图
    final_ckpt = os.path.join(OUTPUT_BASE, args.tag, f'global_r{NUM_ROUNDS-1}.pt')
    if os.path.exists(final_ckpt):
        vis_dir = os.path.join(OUTPUT_BASE, 'visualization', args.tag)
        os.makedirs(vis_dir, exist_ok=True)
        val_img_dir = os.path.join(OUTPUT_BASE, 'val_set', 'images')
        val_lbl_dir = os.path.join(OUTPUT_BASE, 'val_set', 'labels')
        n = visualize_detections(final_ckpt, val_img_dir, val_lbl_dir, vis_dir, DEVICE, IMGSZ)
        log(f"  {n} 张检测叠加图 → {vis_dir}")

    log(f"\n{args.tag} 完成! 结果: {result_path}")

if __name__ == '__main__':
    main()
