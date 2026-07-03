r"""
鼠肝方案2: 联邦学习 — 3节点(B1/B2/B3) × 3策略(FedAvg/EWA/FedProx)对比

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\fl_train.py

数据路径 (本地):
    D:\datasets\mouse_liver_data\batch1\  (10张 2592x1944, 23 organoids)
    D:\datasets\mouse_liver_data\batch2\  (10张 2592x1944, 23 organoids)
    D:\datasets\mouse_liver_data\batch3\  (20张 4000x3000, 40 organoids)

输出:
    runs\mouse_liver_fl\  — 每轮 global model + 评估结果 + 对比图

三种聚合策略:
    1. FedAvg — 按数据量加权平均 (baseline)
    2. EWA — 用 mAP50-95 作为质量信号加权 (低熵/高质量客户端权重更高)
    3. FedProx — proximal term 近似, (1-μ)·w_local + μ·w_global

关键设计 (避免已知坑):
    - YOLO('yolo12n.pt') 是 80 类 COCO, load_state_dict 会 shape mismatch
    - Round 0 先训练 init_model (1 epoch B1) 得到 1 类模型
    - 后续轮次用 YOLO(saved_ckpt) 加载, 跳过 80 类重建
    - data.yaml path 用正斜杠 (Windows \b 会被 YAML 当转义)
    - data.yaml 每次覆盖, 清除 labels.cache
    - EWA warmup: 前 2 轮用 FedAvg (首轮模型未收敛, EWA 信号不可靠)
    - workers=0: Windows spawn 会导致 torch DLL 重复加载 (WinError 1455)
    - torch 延迟导入: 避免顶层 import 在 worker 中重复加载 DLL
    - 每轮 del model + torch.cuda.empty_cache(): 防止 GPU 显存泄漏
"""
import os, sys, json, time, copy, shutil
import numpy as np

# 注意: torch 延迟导入 — Windows spawn 模式下每个 worker 都会重新 import 脚本,
# 顶层 import torch 会导致 DLL 重复加载, 页面文件耗尽 (WinError 1455)

# ============ 配置 ============
DATA_BASE = r"D:\datasets\mouse_liver_data"
BATCH_DIRS = {
    'b1': os.path.join(DATA_BASE, 'batch1'),
    'b2': os.path.join(DATA_BASE, 'batch2'),
    'b3': os.path.join(DATA_BASE, 'batch3'),
}
OUTPUT_DIR = r"runs\mouse_liver_fl"
NUM_ROUNDS = 5
WORKERS = 0  # Windows: 0=主进程加载, 避免 spawn 子进程重复加载 torch DLL (WinError 1455)
LOCAL_EPOCHS = 10
IMGSZ = 640
BATCH_SIZE = 4
DEVICE = 'cuda'
EWA_WARMUP_ROUNDS = 2   # 前 2 轮用 FedAvg, 第 3 轮起用 EWA
FedProx_MU = 0.01        # FedProx 插值系数

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)

def safe_path(p):
    """Windows 路径转 YAML 安全格式 (正斜杠)"""
    return p.replace('\\', '/')

# ============ 聚合策略 ============

def fedavg_aggregate(state_dicts, weights=None):
    """FedAvg: 按数据量加权平均"""
    import torch
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    avg = {}
    for key in state_dicts[0]:
        if state_dicts[0][key].dtype in (torch.int32, torch.int64):
            avg[key] = state_dicts[0][key].clone()
            continue
        avg[key] = sum(w * sd[key].float() for sd, w in zip(state_dicts, weights))
    return avg


def compute_ewa_weights(client_metrics, signal="mAP"):
    """EWA: 用 mAP50-95 作为质量信号加权 (低熵/高质量客户端权重更高)
    
    在完整 ewa-fed 框架中, 权重是 exp(-H/α), H 是预测熵。
    对于 YOLO 检测, 用 mAP50-95 作为质量信号 (比 mAP50 更有区分度)。
    """
    key = "mAP" if signal == "mAP" else "mAP50"
    maps = [max(m[key], 1e-8) for m in client_metrics]  # 防 0
    total = sum(maps)
    if total == 0:
        return [1.0 / len(maps)] * len(maps)
    return [m / total for m in maps]


def fedprox_interpolate(local_sd, global_sd, mu=0.01):
    """FedProx 近似: (1-μ)·w_local + μ·w_global
    
    FedProx 在 loss 中加 μ/2·||w - w_global||², 拉近本地和全局。
    Ultralytics 不支持自定义 loss, 用权重插值近似。
    """
    import torch
    result = {}
    for key in local_sd:
        if local_sd[key].dtype in (torch.int32, torch.int64):
            result[key] = local_sd[key].clone()
            continue
        result[key] = (1 - mu) * local_sd[key].float() + mu * global_sd[key].float()
    return result


# ============ 数据准备 ============

def write_node_yaml(data_dir):
    """为每个节点写 data.yaml (绝对路径, 正斜杠) + 清 labels.cache"""
    node_yaml = os.path.join(data_dir, 'data.yaml')
    with open(node_yaml, 'w') as f:
        f.write(f'path: {safe_path(data_dir)}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
    cache = os.path.join(data_dir, 'labels.cache')
    if os.path.exists(cache):
        try:
            os.remove(cache)
        except PermissionError:
            pass  # Windows 文件锁, 下次训练会自动覆盖
    return node_yaml


def prepare_val_set():
    """统一 val set: 每批各取前 2 张 (共 6 张)"""
    val_dir = os.path.join(OUTPUT_DIR, 'val_set')
    if os.path.exists(val_dir):
        try:
            shutil.rmtree(val_dir)
        except PermissionError:
            # Windows 文件锁, 尝试删除内容
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
        for idx in range(2):
            fname = f'image_{idx:02d}'
            shutil.copy2(os.path.join(img_src, f'{fname}.jpg'),
                        os.path.join(val_dir, 'images', f'{node}_{fname}.jpg'))
            shutil.copy2(os.path.join(lbl_src, f'{fname}.txt'),
                        os.path.join(val_dir, 'labels', f'{node}_{fname}.txt'))
    
    # 清 val labels.cache
    val_cache = os.path.join(val_dir, 'labels.cache')
    if os.path.exists(val_cache):
        try:
            os.remove(val_cache)
        except PermissionError:
            pass
    
    val_yaml = os.path.join(OUTPUT_DIR, 'val.yaml')
    with open(val_yaml, 'w') as f:
        f.write(f'path: {safe_path(os.path.abspath(val_dir))}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
    return val_yaml


# ============ 模型加载 ============

def load_model(ckpt_path):
    """从 checkpoint 加载模型 — YOLO(ckpt) 会读取 checkpoint 里的 nc, 不会重建 80 类"""
    from ultralytics import YOLO
    return YOLO(ckpt_path)


def release_model(model):
    """释放模型 GPU 显存 — model 是 YOLO 对象"""
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


# ============ 单策略 FL 训练 ============

def run_fl_strategy(strategy_name, init_ckpt, val_yaml):
    """运行一轮完整的 FL 训练 (指定聚合策略)"""
    from ultralytics import YOLO
    
    strat_dir = os.path.join(OUTPUT_DIR, strategy_name)
    os.makedirs(strat_dir, exist_ok=True)
    
    global_ckpt = init_ckpt
    fl_history = []
    
    for round_idx in range(NUM_ROUNDS):
        log(f"\n  --- {strategy_name} Round {round_idx+1}/{NUM_ROUNDS} ---")
        
        local_weights = []
        local_sizes = []
        local_metrics = []
        global_sd_for_fedprox = None
        
        for node_name, data_dir in BATCH_DIRS.items():
            log(f"    训练 {node_name}...")
            node_yaml = write_node_yaml(data_dir)
            
            # 从 global_ckpt 加载模型
            model = load_model(global_ckpt)
            
            # FedProx: 保存训练前的 global state_dict
            if strategy_name == "fedprox":
                global_sd_for_fedprox = {k: v.clone() for k, v in model.model.state_dict().items()}
            
            t0 = time.time()
            model.train(data=node_yaml, epochs=LOCAL_EPOCHS, imgsz=IMGSZ, batch=BATCH_SIZE,
                        device=DEVICE, workers=WORKERS, cache=False,
                        project=strat_dir, name=f'r{round_idx}_{node_name}',
                        exist_ok=True, cos_lr=True, close_mosaic=5, verbose=False)
            dt = time.time() - t0
            log(f"      {dt/60:.1f} min")
            
            # 评估本地模型
            val_res = model.val(data=val_yaml, device=DEVICE, project=strat_dir,
                                name=f'r{round_idx}_{node_name}_val', exist_ok=True)
            mAP50 = float(val_res.box.map50)
            mAP5095 = float(val_res.box.map)
            log(f"      {node_name}: mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}")
            
            # 提取 state_dict — 不直接用 model.model.state_dict()
            # 因为训练后模型可能被 fuse (有 conv.bias, 无 bn.weight)
            # 而 init_ckpt 的 model 是 unfused (有 bn.weight, 无 conv.bias)
            # 两者 key 不匹配会导致 load_state_dict 失败
            # 解法: model.save() 保存 ckpt → torch.load 取 state_dict
            # save 的 ckpt 格式和 init_ckpt 一致, 保证 key 匹配
            import torch
            local_ckpt_path = os.path.join(strat_dir, f'r{round_idx}_{node_name}_local.pt')
            model.save(local_ckpt_path)
            local_ckpt = torch.load(local_ckpt_path, map_location='cpu', weights_only=False)
            sd = local_ckpt['model'].state_dict()
            
            # FedProx: interpolate local toward global
            if strategy_name == "fedprox":
                sd = fedprox_interpolate(sd, global_sd_for_fedprox, mu=FedProx_MU)
            
            local_weights.append(sd)
            local_sizes.append(len(os.listdir(os.path.join(data_dir, 'images'))))
            local_metrics.append({
                'node': node_name,
                'mAP50': round(mAP50, 4),
                'mAP': round(mAP5095, 4),
            })
            
            # 释放模型显存
            release_model(model)
            model = None  # 覆盖调用方引用, 触发 GC
        
        # 聚合
        if strategy_name == "ewa":
            if round_idx < EWA_WARMUP_ROUNDS:
                log(f"    EWA warmup (round {round_idx+1}): using FedAvg")
                weights = [s / sum(local_sizes) for s in local_sizes]
                avg_sd = fedavg_aggregate(local_weights, weights)
            else:
                weights = compute_ewa_weights(local_metrics, signal="mAP")
                log(f"    EWA weights: {[round(w,3) for w in weights]}")
                avg_sd = fedavg_aggregate(local_weights, weights)
        else:
            weights = [s / sum(local_sizes) for s in local_sizes]
            avg_sd = fedavg_aggregate(local_weights, weights)
        
        # 保存 global model
        # avg_sd 来自 save+load 的 ckpt (格式和 init_ckpt 一致)
        # 用 YOLO(init_ckpt) 创建模型, load_state_dict (key 匹配), 再 save
        import torch
        global_ckpt = os.path.join(strat_dir, f'global_r{round_idx}.pt')
        global_model = load_model(init_ckpt)
        global_model.model.load_state_dict(avg_sd, strict=True)
        global_model.save(global_ckpt)
        release_model(global_model)
        global_model = None
        log(f"    Global model saved: {global_ckpt}")
        
        # 评估全局模型
        global_model = load_model(global_ckpt)
        val_res = global_model.val(data=val_yaml, device=DEVICE, project=strat_dir,
                                    name=f'r{round_idx}_global_val', exist_ok=True)
        g_mAP50 = float(val_res.box.map50)
        g_mAP5095 = float(val_res.box.map)
        log(f"    ★ Global: mAP50={g_mAP50:.4f}, mAP50-95={g_mAP5095:.4f}, P={val_res.box.mp:.4f}, R={val_res.box.mr:.4f}")
        
        fl_history.append({
            'round': round_idx + 1,
            'global_mAP50': round(g_mAP50, 4),
            'global_mAP': round(g_mAP5095, 4),
            'global_P': round(float(val_res.box.mp), 4),
            'global_R': round(float(val_res.box.mr), 4),
            'local': local_metrics,
            'weights': [round(w, 4) for w in weights],
        })
        
        release_model(global_model)
        global_model = None
    
    return fl_history


# ============ 主程序 ============

def main():
    from ultralytics import YOLO
    
    # 检查数据目录存在
    for name, ddir in BATCH_DIRS.items():
        if not os.path.exists(ddir):
            log(f"❌ 数据目录不存在: {ddir}")
            sys.exit(1)
        n_imgs = len([f for f in os.listdir(os.path.join(ddir, 'images')) if f.endswith('.jpg')])
        n_lbls = len([f for f in os.listdir(os.path.join(ddir, 'labels')) if f.endswith('.txt')])
        log(f"  {name}: {n_imgs} images, {n_lbls} labels")
    
    val_yaml = prepare_val_set()
    log(f"Val set: {val_yaml}")
    
    # === Step 1: 训练 init_model (1 epoch, 从 yolo12n.pt 开始) ===
    init_ckpt = os.path.join(OUTPUT_DIR, 'init_model.pt')
    if not os.path.exists(init_ckpt):
        log("\n=== Step 1: 训练 init_model (1 epoch, B1 数据) ===")
        node_yaml = write_node_yaml(BATCH_DIRS['b1'])
        model = YOLO('yolo12n.pt')
        model.train(data=node_yaml, epochs=1, imgsz=IMGSZ, batch=BATCH_SIZE,
                    device=DEVICE, workers=WORKERS, cache=False,
                    project=OUTPUT_DIR, name='init', exist_ok=True,
                    cos_lr=True, verbose=False)
        model.save(init_ckpt)
        log(f"init_model saved: {init_ckpt}")
        release_model(model)
        model = None
    else:
        log(f"init_model 已存在: {init_ckpt}")
    
    # === Step 2: 三策略对比 ===
    all_results = {}
    for strategy in ['fedavg', 'ewa', 'fedprox']:
        log(f"\n{'='*60}")
        log(f"策略: {strategy.upper()}")
        log(f"{'='*60}")
        history = run_fl_strategy(strategy, init_ckpt, val_yaml)
        all_results[strategy] = history
    
    # 保存对比结果
    with open(os.path.join(OUTPUT_DIR, 'fl_comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # === 汇总 ===
    log(f"\n{'='*60}")
    log("三策略对比汇总")
    log(f"{'='*60}")
    log(f"\n{'Round':<8} {'FedAvg':<12} {'EWA':<12} {'FedProx':<12}")
    log("-"*44)
    for r in range(NUM_ROUNDS):
        vals = []
        for s in ['fedavg', 'ewa', 'fedprox']:
            v = all_results[s][r]['global_mAP50']
            vals.append(f"{v:.4f}")
        log(f"  R{r+1:<6} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")
    
    log(f"\n最终 (Round {NUM_ROUNDS}):")
    for s in ['fedavg', 'ewa', 'fedprox']:
        final = all_results[s][-1]
        log(f"  {s}: mAP50={final['global_mAP50']:.4f}, mAP50-95={final['global_mAP']:.4f}")

if __name__ == '__main__':
    main()
