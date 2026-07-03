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
"""
import os, sys, json, time, copy, shutil
import torch

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
    """加权平均"""
    avg_sd = {}
    for key in state_dicts[0]:
        avg_sd[key] = sum(sd[key] * w for sd, w in zip(state_dicts, weights))
    return avg_sd

# ============ 主程序 ============
def main():
    from ultralytics import YOLO
    
    # 统一 val set: 每批各取 2 张 (B1: image_00,01  B2: image_00,01  B3: image_00,01)
    # 创建临时 val 数据集
    val_dir = os.path.join(OUTPUT_DIR, 'val_set')
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    
    for node, ddir in BATCH_DIRS.items():
        img_src = os.path.join(ddir, 'images')
        lbl_src = os.path.join(ddir, 'labels')
        for idx in range(2):  # 取前2张做 val
            fname = f'image_{idx:02d}'
            shutil.copy2(os.path.join(img_src, f'{fname}.jpg'),
                        os.path.join(val_dir, 'images', f'{node}_{fname}.jpg'))
            shutil.copy2(os.path.join(lbl_src, f'{fname}.txt'),
                        os.path.join(val_dir, 'labels', f'{node}_{fname}.txt'))
    
    # 清除 val labels.cache
    val_cache = os.path.join(val_dir, 'labels.cache')
    if os.path.exists(val_cache):
        os.remove(val_cache)
    
    val_yaml = os.path.join(OUTPUT_DIR, 'val.yaml')
    with open(val_yaml, 'w') as f:
        f.write(f'path: {os.path.abspath(val_dir)}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
    
    # 初始全局模型 — 自动下载 yolo12n.pt
    global_model_path = 'yolo12n.pt'
    if not os.path.exists(global_model_path):
        log("下载 yolo12n.pt ...")
        YOLO('yolo12n.pt')
    fl_history = []
    
    for round_idx in range(NUM_ROUNDS):
        log(f"\n{'='*60}")
        log(f"FL Round {round_idx+1}/{NUM_ROUNDS}")
        log(f"{'='*60}")
        
        local_weights = []
        local_sizes = []
        local_metrics = []
        
        for node_name, data_dir in BATCH_DIRS.items():
            log(f"\n  训练节点 {node_name} ({data_dir})...")
            
            # data.yaml for this node — always overwrite with absolute path
            node_yaml = os.path.join(data_dir, 'data.yaml')
            with open(node_yaml, 'w') as f:
                f.write(f'path: {data_dir}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
            # Remove stale cache
            cache = os.path.join(data_dir, 'labels.cache')
            if os.path.exists(cache):
                os.remove(cache)
            
            model = YOLO('yolo12n.pt')  # 基础架构 (避免 YOLO() 80类重建)
            # 如果有上一轮的 global model, 加载聚合权重
            if round_idx > 0:
                prev_global = os.path.join(OUTPUT_DIR, f'global_r{round_idx-1}.pt')
                if os.path.exists(prev_global):
                    avg_sd_prev = torch.load(prev_global, map_location='cpu')
                    model.model.load_state_dict(avg_sd_prev)
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
            
            # 提取权重
            sd = model.model.state_dict()
            local_weights.append(sd)
            local_sizes.append(len(os.listdir(os.path.join(data_dir, 'images'))))
            local_metrics.append({
                'node': node_name,
                'mAP50': round(float(val_res.box.map50), 4),
                'mAP50-95': round(float(val_res.box.map), 4),
            })
        
        # FedAvg
        log(f"\n  FedAvg 聚合 (weights: {local_sizes})...")
        total_size = sum(local_sizes)
        weights = [s / total_size for s in local_sizes]
        avg_sd = fed_avg(local_weights, weights)
        
        # 保存全局模型 — 用 torch.save 直接保存 state_dict 避免 YOLO() 80类重建问题
        global_model_path = os.path.join(OUTPUT_DIR, f'global_r{round_idx}.pt')
        torch.save(avg_sd, global_model_path)
        
        # 评估全局模型 — 加载 avg_sd 到新模型
        global_model = YOLO('yolo12n.pt')  # 基础架构
        global_model.model.load_state_dict(avg_sd)  # 加载聚合权重
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
