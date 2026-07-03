r"""
鼠肝方案2: 联邦学习 — 3节点(B1/B2/B3)分别训练 → FedAvg聚合

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\fl_train.py

数据路径 (本地):
    D:\datasets\mouse_liver_data\batch1\  (10张 2592x1944, 23 organoids)
    D:\datasets\mouse_liver_data\batch2\  (10张 2592x1944, 23 organoids)
    D:\datasets\mouse_liver_data\batch3\  (20张 4000x3000, 40 organoids)

输出:
    runs\mouse_liver_fl\  — 每轮 global model + 评估结果

关键设计 (避免已知坑):
    - YOLO('yolo12n.pt') 是 80 类 COCO，load_state_dict 会 shape mismatch
    - Round 0 先训练 1 epoch 得到 1 类 base model (init_model.pt)
    - 后续轮次用 model.ckpt = torch.load(init_model.pt) 加载, 跳过 YOLO() 构造器
    - data.yaml 每次覆盖为绝对路径, 清除 labels.cache
"""
import os, sys, json, time, copy, shutil
import torch
import yaml

# ============ 配置 ============
DATA_BASE = r"D:\datasets\mouse_liver_data"
BATCH_DIRS = {
    'b1': os.path.join(DATA_BASE, 'batch1'),
    'b2': os.path.join(DATA_BASE, 'batch2'),
    'b3': os.path.join(DATA_BASE, 'batch3'),
}
OUTPUT_DIR = r"runs\mouse_liver_fl"
NUM_ROUNDS = 5
LOCAL_EPOCHS = 10
IMGSZ = 640
BATCH_SIZE = 4
DEVICE = 'cuda'  # 冬生本地 GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)

# ============ FedAvg ============
def fed_avg(state_dicts, weights):
    """加权平均 — 只平均 float tensor, 跳过 int (num_batches_tracked)"""
    avg_sd = {}
    for key in state_dicts[0]:
        if state_dicts[0][key].dtype.is_floating_point:
            avg_sd[key] = sum(sd[key].float() * w for sd, w in zip(state_dicts, weights))
        else:
            avg_sd[key] = state_dicts[0][key]  # 直接取第一个
    return avg_sd

# ============ 数据准备 ============
def prepare_val_set():
    """每批各取 2 张做统一 val set"""
    val_dir = os.path.join(OUTPUT_DIR, 'val_set')
    # 清理旧数据
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

    for node, ddir in BATCH_DIRS.items():
        img_src = os.path.join(ddir, 'images')
        lbl_src = os.path.join(ddir, 'labels')
        for idx in range(2):
            fname = f'image_{idx:02d}'
            shutil.copy2(os.path.join(img_src, f'{fname}.jpg'),
                        os.path.join(val_dir, 'images', f'{node}_{fname}.jpg'))
            shutil.copy2(os.path.join(lbl_src, f'{fname}.txt'),
                        os.path.join(val_dir, 'labels', f'{node}_{fname}.txt'))

    val_yaml = os.path.join(OUTPUT_DIR, 'val.yaml')
    with open(val_yaml, 'w') as f:
        f.write(f"path: {os.path.abspath(val_dir)}\ntrain: images\nval: images\nnc: 1\nnames: ['organoid']\n")
    return val_yaml

def write_node_yaml(data_dir):
    """每次覆盖 data.yaml 为绝对路径, 清除 cache"""
    node_yaml = os.path.join(data_dir, 'data.yaml')
    with open(node_yaml, 'w') as f:
        f.write(f"path: {data_dir}\ntrain: images\nval: images\nnc: 1\nnames: ['organoid']\n")
    cache = os.path.join(data_dir, 'labels.cache')
    if os.path.exists(cache):
        os.remove(cache)
    return node_yaml

# ============ 模型加载 (绕过 80 类坑) ============
def load_model_from_ckpt(ckpt_path):
    """从 checkpoint 加载模型 — 用 ultralytics 内部 API 绕过 YOLO() 构造器"""
    from ultralytics import YOLO
    # YOLO(ckpt_path) 会读取 checkpoint 里的 nc, 不会重建 80 类
    # 前提: ckpt 是 ultralytics save 的完整模型 (不是纯 state_dict)
    return YOLO(ckpt_path)

# ============ 主程序 ============
def main():
    from ultralytics import YOLO

    val_yaml = prepare_val_set()
    log(f"Val set: {val_yaml}")

    # === Step 1: 训练 init_model (1 epoch, 从 yolo12n.pt 开始) ===
    # 这一步把 80 类 COCO 模型适配为 1 类 organoid 模型
    init_ckpt = os.path.join(OUTPUT_DIR, 'init_model.pt')
    if not os.path.exists(init_ckpt):
        log("\n=== Step 1: 训练 init_model (1 epoch, B1 数据) ===")
        node_yaml = write_node_yaml(BATCH_DIRS['b1'])
        model = YOLO('yolo12n.pt')
        model.train(data=node_yaml, epochs=1, imgsz=IMGSZ, batch=BATCH_SIZE,
                    device=DEVICE, workers=8, cache=False,
                    project=OUTPUT_DIR, name='init', exist_ok=True,
                    cos_lr=True, verbose=False)
        # 保存 init_model — 用 model.save() 保存完整模型 (不是 state_dict)
        model.save(init_ckpt)
        log(f"init_model saved: {init_ckpt}")
    else:
        log(f"init_model 已存在: {init_ckpt}")

    # === Step 2: FL 训练 ===
    fl_history = []
    global_ckpt = init_ckpt  # 第一轮从 init_model 开始

    for round_idx in range(NUM_ROUNDS):
        log(f"\n{'='*60}")
        log(f"FL Round {round_idx+1}/{NUM_ROUNDS}")
        log(f"{'='*60}")

        local_weights = []
        local_sizes = []
        local_metrics = []

        for node_name, data_dir in BATCH_DIRS.items():
            log(f"\n  训练节点 {node_name} ({data_dir})...")
            node_yaml = write_node_yaml(data_dir)

            # 从 global_ckpt 加载模型 (不是 YOLO('yolo12n.pt'))
            model = load_model_from_ckpt(global_ckpt)
            t0 = time.time()
            model.train(data=node_yaml, epochs=LOCAL_EPOCHS, imgsz=IMGSZ, batch=BATCH_SIZE,
                        device=DEVICE, workers=8, cache=False,
                        project=OUTPUT_DIR, name=f'r{round_idx}_{node_name}',
                        exist_ok=True, cos_lr=True, close_mosaic=5, verbose=False)
            dt = time.time() - t0
            log(f"    训练完成: {dt/60:.1f} min")

            # 评估本地模型
            val_res = model.val(data=val_yaml, project=OUTPUT_DIR,
                                name=f'r{round_idx}_{node_name}_val', exist_ok=True)
            log(f"    {node_name} val: mAP50={val_res.box.map50:.4f}")

            # 提取权重 — 用 model.ckpt (完整 checkpoint) 而非 state_dict
            # 因为 load_model_from_ckpt 用 YOLO(ckpt_path) 加载
            local_ckpt_path = os.path.join(OUTPUT_DIR, f'r{round_idx}_{node_name}_local.pt')
            model.save(local_ckpt_path)

            # 提取 state_dict 用于 FedAvg
            sd = model.model.state_dict()
            local_weights.append(sd)
            local_sizes.append(len(os.listdir(os.path.join(data_dir, 'images'))))
            local_metrics.append({
                'node': node_name,
                'mAP50': round(float(val_res.box.map50), 4),
                'mAP50-95': round(float(val_res.box.map), 4),
            })

        # === FedAvg ===
        log(f"\n  FedAvg 聚合 (weights: {local_sizes})...")
        total_size = sum(local_sizes)
        weights = [s / total_size for s in local_sizes]
        avg_sd = fed_avg(local_weights, weights)

        # 保存 global model — 加载 init_model 完整 ckpt, 覆盖 state_dict
        # ckpt['model'] 是 DetectionModel 对象, 用 load_state_dict 覆盖权重
        global_ckpt = os.path.join(OUTPUT_DIR, f'global_r{round_idx}.pt')
        init_full_ckpt = torch.load(init_ckpt, map_location='cpu', weights_only=False)
        init_full_ckpt['model'].load_state_dict(avg_sd, strict=True)
        torch.save(init_full_ckpt, global_ckpt)
        log(f"  Global model saved: {global_ckpt}")

        # 评估全局模型
        global_model = load_model_from_ckpt(global_ckpt)
        val_res = global_model.val(data=val_yaml, project=OUTPUT_DIR,
                                    name=f'r{round_idx}_global_val', exist_ok=True)
        log(f"\n  ★ Global val: mAP50={val_res.box.map50:.4f}, P={val_res.box.mp:.4f}, R={val_res.box.mr:.4f}")

        fl_history.append({
            'round': round_idx + 1,
            'global_mAP50': round(float(val_res.box.map50), 4),
            'global_mAP50-95': round(float(val_res.box.map), 4),
            'global_P': round(float(val_res.box.mp), 4),
            'global_R': round(float(val_res.box.mr), 4),
            'local': local_metrics,
        })

    # 保存历史
    with open(os.path.join(OUTPUT_DIR, 'fl_history.json'), 'w') as f:
        json.dump(fl_history, f, indent=2)

    log(f"\n{'='*60}")
    log("FL 训练完成!")
    log(f"{'='*60}")
    for h in fl_history:
        log(f"  Round {h['round']}: global mAP50={h['global_mAP50']:.4f}")
        for l in h['local']:
            log(f"    {l['node']}: mAP50={l['mAP50']:.4f}")

if __name__ == '__main__':
    main()
