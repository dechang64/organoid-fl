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
NUM_ROUNDS = 10
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
        # 统一到 CPU 避免设备不一致
        avg[key] = sum(w * sd[key].cpu().float() for sd, w in zip(state_dicts, weights))
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
    """FedProx 近似: (1-μ)·w_local + μ·w_global — 只插值匹配的 key"""
    import torch
    result = {}
    for key in local_sd:
        if key in global_sd and local_sd[key].shape == global_sd[key].shape:
            if local_sd[key].dtype in (torch.int32, torch.int64):
                result[key] = local_sd[key].clone()
                continue
            result[key] = (1 - mu) * local_sd[key].float() + mu * global_sd[key].float()
        else:
            # key 不匹配 (fused/unfused 差异), 保留 local
            result[key] = local_sd[key].clone()
    return result


# ============ 数据准备 ============

# 每批用于 val 的图片索引 (其余用于训练)
VAL_INDICES = {
    'b1': [7, 8, 9],       # B1 有 image_00~09, val 取后 3 张
    'b2': [7, 8, 9],       # B2 有 image_00~09
    'b3': [17, 18, 19],    # B3 有 image_00~19, val 取后 3 张
}

def write_node_yaml(data_dir, node_name):
    """为每个节点写 data.yaml — train 用排除 val 的图, val 用同批 val 图"""
    import glob
    node_yaml = os.path.join(data_dir, 'data.yaml')
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')
    
    # 创建 train_split 子目录 (排除 val 图)
    train_img_dir = os.path.join(data_dir, 'train_images')
    train_lbl_dir = os.path.join(data_dir, 'train_labels')
    for d in [train_img_dir, train_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        # 清空旧文件
        for old in os.listdir(d):
            try:
                os.remove(os.path.join(d, old))
            except:
                pass
    
    val_indices = set(VAL_INDICES.get(node_name, []))
    for img_file in sorted(os.listdir(img_dir)):
        # image_XX.jpg → idx
        name = os.path.splitext(img_file)[0]  # image_XX
        try:
            idx = int(name.split('_')[1])
        except (IndexError, ValueError):
            continue
        if idx in val_indices:
            continue  # 跳过 val 图
        # 复制到 train_split
        shutil.copy2(os.path.join(img_dir, img_file), os.path.join(train_img_dir, img_file))
        lbl_file = img_file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(lbl_dir, lbl_file)):
            shutil.copy2(os.path.join(lbl_dir, lbl_file), os.path.join(train_lbl_dir, lbl_file))
    
    with open(node_yaml, 'w') as f:
        f.write(f'path: {safe_path(data_dir)}\ntrain: train_images\nval: train_images\nnc: 1\nnames: [\'organoid\']\n')
    # 清 cache
    for cache_name in ['labels.cache', 'train_images.cache', 'train_labels.cache']:
        cache = os.path.join(data_dir, cache_name)
        if os.path.exists(cache):
            try:
                os.remove(cache)
            except PermissionError:
                pass
    return node_yaml


def prepare_val_set():
    """统一 val set: 每批取 VAL_INDICES 指定的图 (共 9 张)"""
    val_dir = os.path.join(OUTPUT_DIR, 'val_set')
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
    """从 checkpoint 加载模型"""
    from ultralytics import YOLO
    return YOLO(ckpt_path)

def load_sam2_predictor(device='cuda'):
    """加载 SAM2 predictor — 和 sam2_segment.py 同样的方式"""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
        # 尝试常见 checkpoint 路径
        ckpt_candidates = [
            'sam2_checkpoints/sam2_hiera_small.pt',
            'weights/sam2_hiera_small.pt',
            'sam2_hiera_small.pt',
        ]
        checkpoint = None
        for c in ckpt_candidates:
            if os.path.exists(c):
                checkpoint = c
                break
        if not checkpoint:
            log("    SAM2 checkpoint 未找到 (sam2_hiera_small.pt), 使用 fallback")
            return None
        for cfg in ['sam2_hiera_s', 'sam2_hiera_small']:
            try:
                model = build_sam2(cfg, checkpoint, device=device)
                predictor = SAM2ImagePredictor(model)
                log(f"    SAM2 loaded: {cfg} / {checkpoint}")
                return predictor
            except Exception:
                continue
        log("    SAM2 加载失败, 使用 fallback")
        return None
    except ImportError:
        log("    sam2 包未安装, 使用 fallback")
        return None

def release_sam2(predictor):
    """释放 SAM2 predictor"""
    if predictor is not None:
        import torch
        try:
            del predictor.model
            del predictor
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        
        node_names = list(BATCH_DIRS.keys())
        last_unfused_model = None  # 保留最后一个训练后 (未 val) 的模型作 template
        for node_idx, (node_name, data_dir) in enumerate(BATCH_DIRS.items()):
            log(f"    训练 {node_name}...")
            node_yaml = write_node_yaml(data_dir, node_name)
            
            # 从 global_ckpt 加载模型
            model = load_model(global_ckpt)
            
            # FedProx: 保存训练前的 global state_dict (统一到 CPU)
            if strategy_name == "fedprox":
                global_sd_for_fedprox = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
            
            t0 = time.time()
            model.train(data=node_yaml, epochs=LOCAL_EPOCHS, imgsz=IMGSZ, batch=BATCH_SIZE,
                        device=DEVICE, workers=WORKERS, cache=False,
                        project=strat_dir, name=f'r{round_idx}_{node_name}',
                        exist_ok=True, cos_lr=True, close_mosaic=5, verbose=False)
            dt = time.time() - t0
            log(f"      {dt/60:.1f} min")
            
            # 提取 state_dict — 在 val 之前取! 
            # model.val() 会 fuse model.model (不可逆), 导致 key 从 691→239
            sd = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
            
            # 评估本地模型 (val 会 fuse, 但 sd 已取)
            val_res = model.val(data=val_yaml, device=DEVICE, project=strat_dir,
                                name=f'r{round_idx}_{node_name}_val', exist_ok=True)
            mAP50 = float(val_res.box.map50)
            mAP5095 = float(val_res.box.map)
            log(f"      {node_name}: mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}")
            
            # FedProx: interpolate local toward global
            if strategy_name == "fedprox":
                sd = fedprox_interpolate(sd, global_sd_for_fedprox, mu=FedProx_MU)
            
            local_weights.append(sd)
            local_sizes.append(len(os.listdir(os.path.join(data_dir, 'train_images'))))
            local_metrics.append({
                'node': node_name,
                'mAP50': round(mAP50, 4),
                'mAP': round(mAP5095, 4),
            })
            
            # 保留最后一个本地模型 (训练后, val 前) 作为 global template
            # 但 val 已经 fuse 了 model.model... 
            # 所以不能直接用 last_model, 需要重新加载 global_ckpt 作 template
            release_model(model)
            model = None
        
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
        # 用 init_ckpt (unfused) 作为 template, 和 avg_sd (unfused) 同格式
        import torch
        from copy import deepcopy
        from datetime import datetime
        from ultralytics import __version__
        global_ckpt = os.path.join(strat_dir, f'global_r{round_idx}.pt')
        
        # 用 init_ckpt 作 template (unfused, 691 keys, 和 avg_sd 同格式)
        global_model = load_model(init_ckpt)
        model_sd = global_model.model.state_dict()
        loaded = 0
        for key in avg_sd:
            if key in model_sd and avg_sd[key].shape == model_sd[key].shape:
                model_sd[key].copy_(avg_sd[key])
                loaded += 1
        log(f"    Loaded {loaded}/{len(avg_sd)} keys")
        
        # 用 torch.save 保存 (不用 model.save 避免 fuse)
        gckpt = {
            'model': deepcopy(global_model.model).float(),
            'date': datetime.now().isoformat(),
            'version': __version__,
            'license': 'AGPL-3.0 License',
            'docs': 'https://docs.ultralytics.com',
            'train_args': dict(global_model.overrides) if hasattr(global_model, 'overrides') and hasattr(global_model.overrides, '__dict__') else {},
        }
        torch.save(gckpt, global_ckpt)
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
    # 强制重新生成 init_model — 旧版脚本可能用 model.save() 保存了 fused 模型
    if os.path.exists(init_ckpt):
        os.remove(init_ckpt)
    if not os.path.exists(init_ckpt):
        log("\n=== Step 1: 训练 init_model (1 epoch, B1 数据) ===")
        node_yaml = write_node_yaml(BATCH_DIRS['b1'], 'b1')
        model = YOLO('yolo12n.pt')
        model.train(data=node_yaml, epochs=1, imgsz=IMGSZ, batch=BATCH_SIZE,
                    device=DEVICE, workers=WORKERS, cache=False,
                    project=OUTPUT_DIR, name='init', exist_ok=True,
                    cos_lr=True, verbose=False)
        # 不用 model.save() — 它保存 fused 模型 (Conv+BN 合并, 不可逆)
        # 用 torch.save 保存 model.model 对象 (unfused), 保持和训练后 state_dict 一致
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
    
    # === Step 3: 可视化 ===
    log(f"\n{'='*60}")
    log("生成可视化...")
    log(f"{'='*60}")
    plot_fl_curves(all_results, OUTPUT_DIR)
    
    # 用每策略最终全局模型在 val_set 上推理, 画检测叠加图
    log("\n生成检测可视化 (最终全局模型 → val_set)...")
    for strategy in ['fedavg', 'ewa', 'fedprox']:
        final_ckpt = os.path.join(OUTPUT_DIR, strategy, f'global_r{NUM_ROUNDS-1}.pt')
        if os.path.exists(final_ckpt):
            vis_dir = os.path.join(OUTPUT_DIR, 'visualization', strategy)
            os.makedirs(vis_dir, exist_ok=True)
            val_img_dir = os.path.join(OUTPUT_DIR, 'val_set', 'images')
            val_lbl_dir = os.path.join(OUTPUT_DIR, 'val_set', 'labels')
            n = visualize_detections(final_ckpt, val_img_dir, val_lbl_dir, vis_dir, DEVICE, IMGSZ)
            log(f"  {strategy}: {n} 张检测叠加图 → {vis_dir}")
        else:
            log(f"  {strategy}: checkpoint 不存在, 跳过")
    
    # 轮廓可视化: YOLO检测 → SAM2分割 → 像素级轮廓 (和人工红线标注同格式)
    log("\n生成轮廓可视化 (YOLO+SAM2 → 像素级mask轮廓)...")
    sam2_predictor = load_sam2_predictor(DEVICE)
    val_img_dir = os.path.join(OUTPUT_DIR, 'val_set', 'images')
    val_lbl_dir = os.path.join(OUTPUT_DIR, 'val_set', 'labels')
    for strategy in ['fedavg', 'ewa', 'fedprox']:
        final_ckpt = os.path.join(OUTPUT_DIR, strategy, f'global_r{NUM_ROUNDS-1}.pt')
        if not os.path.exists(final_ckpt):
            log(f"  {strategy}: checkpoint 不存在, 跳过")
            continue
        contour_dir = os.path.join(OUTPUT_DIR, 'visualization', strategy, 'contours')
        os.makedirs(contour_dir, exist_ok=True)
        try:
            n = visualize_contours_yolo_sam2(
                final_ckpt, val_img_dir, val_lbl_dir, contour_dir, 
                DEVICE, IMGSZ, sam2_predictor)
            log(f"  {strategy}: {n} 张轮廓图 → {contour_dir}")
        except Exception as e:
            log(f"  {strategy}: 轮廓可视化失败: {e}")
    release_sam2(sam2_predictor)
    sam2_predictor = None


def visualize_contours_yolo_sam2(yolo_ckpt, img_dir, lbl_dir, dst_dir, device='cuda', imgsz=640, sam2_predictor=None):
    """YOLO 检测 → SAM2 分割 → 像素级轮廓叠加图
    
    输出两种图:
    - mask_*.jpg: SAM2 mask (绿色半透明) + GT bbox (红色)
    - contour_*.jpg: SAM2 轮廓线 (绿色) + GT 轮廓线 (红色), 和人工标注同格式
    
    sam2_predictor: 预加载的 SAM2ImagePredictor, None 则用 fallback
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO
    
    sam2_ok = sam2_predictor is not None
    
    model = YOLO(yolo_ckpt)
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    
    for img_file in images:
        img_path = os.path.join(img_dir, img_file)
        orig = cv2.imread(img_path)
        h, w = orig.shape[:2]
        
        # YOLO 检测
        results = model.predict(img_path, device=device, imgsz=imgsz, verbose=False, conf=0.25)
        boxes = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append([x1, y1, x2, y2])
        
        # SAM2 分割
        masks = []
        if sam2_ok and boxes:
            import torch as _torch
            img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            sam2_predictor.set_image(img_rgb)
            for box in boxes:
                with _torch.no_grad():
                    mask_preds, _, _ = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.array(box, dtype=np.float32),
                        multimask_output=False,
                    )
                masks.append(mask_preds[0])
        elif boxes:
            # Fallback: 用 bbox 椭圆 mask
            for x1, y1, x2, y2 in boxes:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask, ((x1+x2)//2, (y1+y2)//2), ((x2-x1)//2, (y2-y1)//2), 0, 0, 360, 255, -1)
                masks.append(mask.astype(bool))
        
        # 图1: mask 叠加 (绿色半透明)
        mask_vis = orig.copy()
        overlay = mask_vis.copy()
        for mask in masks:
            colored = np.zeros_like(orig)
            colored[mask] = (0, 255, 0)  # 绿色
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
        # 画 GT bbox (红色)
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
                        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        cv2.putText(overlay, 'Green=SAM2 mask  Red=GT bbox', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(dst_dir, f'mask_{img_file}'), overlay)
        
        # 图2: 轮廓线 (和人工红线标注同格式)
        contour_vis = orig.copy()
        # 检测轮廓 (绿色)
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 3)
        # GT 轮廓 (红色) — 从 bbox 画矩形轮廓
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
                        cv2.rectangle(contour_vis, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        cv2.putText(contour_vis, 'Green=SAM2 contour  Red=GT', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(dst_dir, f'contour_{img_file}'), contour_vis)
    
    release_model(model)
    return len(images)


def plot_fl_curves(all_results, output_dir):
    """绘制三策略 FL 收敛曲线 + EWA 权重演化"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    rounds = list(range(1, NUM_ROUNDS + 1))
    
    # 图1: mAP50 收敛曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'fedavg': '#2196F3', 'ewa': '#FF5722', 'fedprox': '#4CAF50'}
    labels = {'fedavg': 'FedAvg', 'ewa': 'EWA', 'fedprox': 'FedProx'}
    for s in ['fedavg', 'ewa', 'fedprox']:
        vals = [r['global_mAP50'] for r in all_results[s]]
        ax.plot(rounds, vals, 'o-', color=colors[s], label=labels[s], linewidth=2, markersize=6)
    ax.set_xlabel('FL Round', fontsize=13)
    ax.set_ylabel('Global mAP50', fontsize=13)
    ax.set_title('FL Convergence — Mouse Liver Organoid Detection', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fl_convergence_mAP50.png'), dpi=150)
    plt.close(fig)
    
    # 图2: mAP50-95 收敛曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in ['fedavg', 'ewa', 'fedprox']:
        vals = [r['global_mAP'] for r in all_results[s]]
        ax.plot(rounds, vals, 'o-', color=colors[s], label=labels[s], linewidth=2, markersize=6)
    ax.set_xlabel('FL Round', fontsize=13)
    ax.set_ylabel('Global mAP50-95', fontsize=13)
    ax.set_title('FL Convergence (mAP50-95) — Mouse Liver', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fl_convergence_mAP5095.png'), dpi=150)
    plt.close(fig)
    
    # 图3: EWA 权重演化 (跳过 warmup 轮, 从 EWA 启动轮开始)
    ewa_weights = []
    for r_idx, r in enumerate(all_results['ewa']):
        if r_idx >= EWA_WARMUP_ROUNDS and r['weights']:
            ewa_weights.append(r['weights'])
    if ewa_weights:
        fig, ax = plt.subplots(figsize=(10, 6))
        ewa_rounds = list(range(EWA_WARMUP_ROUNDS + 1, EWA_WARMUP_ROUNDS + 1 + len(ewa_weights)))
        for i, node in enumerate(['B1', 'B2', 'B3']):
            w = [rw[i] for rw in ewa_weights]
            ax.plot(ewa_rounds, w, 'o-', label=node, linewidth=2, markersize=6)
        ax.set_xlabel('FL Round', fontsize=13)
        ax.set_ylabel('EWA Weight', fontsize=13)
        ax.set_title('EWA Weight Evolution (after warmup)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ewa_rounds)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'ewa_weights_evolution.png'), dpi=150)
        plt.close(fig)
    
    # 图4: 本地模型 mAP50 趋势 (每策略每节点)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, s in enumerate(['fedavg', 'ewa', 'fedprox']):
        ax = axes[idx]
        for i, node in enumerate(['B1', 'B2', 'B3']):
            vals = [r['local'][i]['mAP50'] for r in all_results[s]]
            ax.plot(rounds, vals, 'o-', label=node, linewidth=2, markersize=5)
        ax.set_title(f'{labels[s]} — Local mAP50', fontsize=13)
        ax.set_xlabel('FL Round', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Local mAP50', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rounds)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'local_mAP50_per_node.png'), dpi=150)
    plt.close(fig)
    
    log(f"  收敛曲线 → {output_dir}/fl_convergence_mAP50.png")
    log(f"  收敛曲线 → {output_dir}/fl_convergence_mAP5095.png")
    log(f"  EWA权重演化 → {output_dir}/ewa_weights_evolution.png")
    log(f"  本地模型趋势 → {output_dir}/local_mAP50_per_node.png")


def visualize_detections(ckpt_path, img_dir, lbl_dir, dst_dir, device='cuda', imgsz=640):
    """用模型推理, 画检测框(绿) + GT框(红) 叠加图"""
    import cv2
    from ultralytics import YOLO
    
    model = YOLO(ckpt_path)
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    
    for img_file in images:
        img_path = os.path.join(img_dir, img_file)
        orig = cv2.imread(img_path)
        h, w = orig.shape[:2]
        
        # 推理
        results = model.predict(img_path, device=device, imgsz=imgsz, verbose=False, conf=0.25)
        
        # 画检测框 (绿色)
        vis = orig.copy()
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f'{conf:.2f}', (x1, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 画 GT 框 (红色)
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
        
        # 标注
        cv2.putText(vis, 'Green=Detection  Red=GT', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out_path = os.path.join(dst_dir, f'det_{img_file}')
        cv2.imwrite(out_path, vis)
    
    release_model(model)
    return len(images)

if __name__ == '__main__':
    main()
