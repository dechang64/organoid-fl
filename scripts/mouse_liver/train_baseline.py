r"""
鼠肝 Baseline: 各节点独立训练 + 集中式上界 (无 FL)

与 FL 实验严格可比:
  - 同 val_set (9张混合: B1×3 + B2×3 + B3×3)
  - 同 train/val split (VAL_INDICES 排除 val 图)
  - 同模型 (yolo12n.pt COCO 预训练)
  - 同 imgsz=640, batch=4
  - 100ep, patience=None, close_mosaic=50 (FL: 10ep×10round, close_mosaic=5 → 50% 无 mosaic)

三组 baseline:
  1. B1 独立 (7 train)    — 节点独立下界
  2. B2 独立 (7 train)    — 节点独立下界
  3. B3 独立 (17 train)   — 节点独立下界
  4. 集中式 (31 train)    — B1+B2+B3 合并训练, FL 上界

实验设计参考: research/mouse_liver/fl_experiment_design.md
评估基准: 统一 val_set (维度 D)

输出:
  runs\mouse_liver_baseline\{node}\  — 各节点训练结果
  runs\mouse_liver_baseline\centralized\  — 集中式训练结果
  runs\mouse_liver_baseline\baseline_results.json — 汇总

Usage:
  cd C:\Users\decha\organoid-fl
  .\.venv\Scripts\activate
  python scripts\mouse_liver\train_baseline.py
"""
import os, sys, json, time, shutil

DATA_BASE = r"D:\datasets\mouse_liver_data"
BATCH_DIRS = {
    'b1': os.path.join(DATA_BASE, 'batch1'),
    'b2': os.path.join(DATA_BASE, 'batch2'),
    'b3': os.path.join(DATA_BASE, 'batch3'),
}
OUTPUT_BASE = r"runs\mouse_liver_baseline"
FL_SEQ_DIR = r"runs\mouse_liver_fl_seq"  # 复用 FL 的 val_set

IMGSZ = 640
BATCH_SIZE = 4
WORKERS = 0
DEVICE = 'cuda'
EPOCHS = 100
# close_mosaic: FL 每轮 10ep, close_mosaic=5 → 50% 无 mosaic
# baseline 100ep → close_mosaic=50 匹配 FL 的 50% 比例
CLOSE_MOSAIC = 50
# 不用 patience: FL 每轮固定跑 10ep, baseline 也不早停
PATIENCE = None

VAL_INDICES = {
    'b1': [7, 8, 9],
    'b2': [7, 8, 9],
    'b3': [17, 18, 19],
}


def log(msg):
    print(msg, flush=True)


def safe_path(p):
    return p.replace('\\', '/')


def release_model(model):
    """释放模型 GPU 内存"""
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


def write_node_yaml(data_dir, node_name):
    """与 fl_sequential.py 完全一致的 data.yaml

    重要: Ultralytics img2label_paths 用 os.sep+'images'+os.sep → os.sep+'labels'+os.sep
    做路径替换。目录必须叫 images/labels。
    """
    node_yaml = os.path.join(data_dir, 'data.yaml')
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')

    # 创建 baseline_split 子目录 (不覆盖 FL 的 fl_split)
    split_dir = os.path.join(data_dir, 'baseline_split')
    train_img_dir = os.path.join(split_dir, 'images')
    train_lbl_dir = os.path.join(split_dir, 'labels')
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
            continue  # 排除 val 图
        shutil.copy2(os.path.join(img_dir, img_file), os.path.join(train_img_dir, img_file))
        lbl_file = img_file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(lbl_dir, lbl_file)):
            shutil.copy2(os.path.join(lbl_dir, lbl_file), os.path.join(train_lbl_dir, lbl_file))

    with open(node_yaml, 'w') as f:
        f.write(f'path: {safe_path(split_dir)}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
    # 清 cache (Ultralytics 缓存在 labels/ 目录下)
    cache_path = os.path.join(split_dir, 'labels', 'labels.cache')
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except PermissionError:
            pass
    return node_yaml


def prepare_val_set():
    """复用 FL 的 val_set (如果已存在) 或重新创建

    val_set: 9 张混合图 (B1×3 + B2×3 + B3×3)
    与 fl_sequential.py prepare_val_set 完全一致
    """
    val_dir = os.path.join(FL_SEQ_DIR, 'val_set')
    val_yaml = os.path.join(FL_SEQ_DIR, 'val.yaml')

    # 如果 val_set 已存在且完整，直接复用
    if os.path.exists(val_yaml) and os.path.exists(os.path.join(val_dir, 'images')):
        # 清旧 cache 确保新鲜
        val_cache = os.path.join(val_dir, 'labels', 'labels.cache')
        if os.path.exists(val_cache):
            try:
                os.remove(val_cache)
            except PermissionError:
                pass
        return val_yaml

    # 否则重新创建
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
    with open(val_yaml, 'w') as f:
        f.write(f'path: {safe_path(os.path.abspath(val_dir))}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
    return val_yaml


def main():
    from ultralytics import YOLO

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    val_yaml = prepare_val_set()
    log(f"val_yaml: {val_yaml}")

    results = {}

    for node_name in ['b1', 'b2', 'b3']:
        data_dir = BATCH_DIRS[node_name]
        n_val = len(VAL_INDICES[node_name])
        n_total = len([f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith(('.jpg', '.png'))])
        n_train = n_total - n_val  # 排除 val 图后的训练图数
        log(f"\n{'='*60}")
        log(f"训练 {node_name} (独立 baseline, {EPOCHS}ep close_mosaic={CLOSE_MOSAIC}, {n_train} train images)")
        log(f"{'='*60}")

        node_yaml = write_node_yaml(data_dir, node_name)
        node_dir = os.path.join(OUTPUT_BASE, node_name)
        os.makedirs(node_dir, exist_ok=True)

        model = YOLO('yolo12n.pt')
        t0 = time.time()
        train_kwargs = dict(
            data=node_yaml,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            cache=False,
            project=OUTPUT_BASE,
            name=node_name,
            exist_ok=True,
            cos_lr=True,
            close_mosaic=CLOSE_MOSAIC,
            verbose=False,
        )
        if PATIENCE is not None:
            train_kwargs['patience'] = PATIENCE
        model.train(**train_kwargs)
        dt = time.time() - t0
        log(f"  训练时间: {dt/60:.1f} min")

        # 用 FL 同一 val_set 评估 (统一基准)
        val_res = model.val(data=val_yaml, device=DEVICE, project=node_dir,
                            name='val_fl_set', exist_ok=True)
        mAP50 = float(val_res.box.map50)
        mAP5095 = float(val_res.box.map)
        precision = float(val_res.box.mp)
        recall = float(val_res.box.mr)

        log(f"  ★ {node_name} → FL val_set: mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}, P={precision:.4f}, R={recall:.4f}")

        results[node_name] = {
            'n_train_images': n_train,
            'fl_val_mAP50': round(mAP50, 4),
            'fl_val_mAP5095': round(mAP5095, 4),
            'fl_val_P': round(precision, 4),
            'fl_val_R': round(recall, 4),
            'train_time_min': round(dt / 60, 1),
        }

        release_model(model)

    # === 集中式上界: B1+B2+B3 合并训练 ===
    log(f"\n{'='*60}")
    log(f"训练 centralized (B1+B2+B3 合并, {EPOCHS}ep close_mosaic={CLOSE_MOSAIC})")
    log(f"{'='*60}")

    # 合并三批训练图到 centralized_split
    cent_dir = os.path.join(OUTPUT_BASE, 'centralized_split')
    cent_img_dir = os.path.join(cent_dir, 'images')
    cent_lbl_dir = os.path.join(cent_dir, 'labels')
    for d in [cent_img_dir, cent_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        for old in os.listdir(d):
            try:
                os.remove(os.path.join(d, old))
            except:
                pass

    total_train = 0
    for node_name in ['b1', 'b2', 'b3']:
        data_dir = BATCH_DIRS[node_name]
        img_dir = os.path.join(data_dir, 'images')
        lbl_dir = os.path.join(data_dir, 'labels')
        val_indices = set(VAL_INDICES[node_name])
        for img_file in sorted(os.listdir(img_dir)):
            name = os.path.splitext(img_file)[0]
            try:
                idx = int(name.split('_')[1])
            except (IndexError, ValueError):
                continue
            if idx in val_indices:
                continue
            # 加前缀避免不同 batch 同名文件覆盖
            dst_img = f'{node_name}_{img_file}'
            shutil.copy2(os.path.join(img_dir, img_file), os.path.join(cent_img_dir, dst_img))
            lbl_file = img_file.replace('.jpg', '.txt')
            if os.path.exists(os.path.join(lbl_dir, lbl_file)):
                shutil.copy2(os.path.join(lbl_dir, lbl_file), os.path.join(cent_lbl_dir, f'{node_name}_{lbl_file}'))
            total_train += 1

    cent_yaml = os.path.join(cent_dir, 'data.yaml')
    with open(cent_yaml, 'w') as f:
        f.write(f'path: {safe_path(cent_dir)}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
    cache_path = os.path.join(cent_lbl_dir, 'labels.cache')
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except PermissionError:
            pass

    log(f"  合并训练图: {total_train} 张")

    cent_model = YOLO('yolo12n.pt')
    t0 = time.time()
    cent_model.train(
        data=cent_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        cache=False,
        project=OUTPUT_BASE,
        name='centralized',
        exist_ok=True,
        cos_lr=True,
        close_mosaic=CLOSE_MOSAIC,
        verbose=False,
    )
    dt = time.time() - t0
    log(f"  训练时间: {dt/60:.1f} min")

    cent_val_res = cent_model.val(data=val_yaml, device=DEVICE,
                                   project=os.path.join(OUTPUT_BASE, 'centralized'),
                                   name='val_fl_set', exist_ok=True)
    cent_mAP50 = float(cent_val_res.box.map50)
    cent_mAP5095 = float(cent_val_res.box.map)
    cent_p = float(cent_val_res.box.mp)
    cent_r = float(cent_val_res.box.mr)
    log(f"  ★ centralized → FL val_set: mAP50={cent_mAP50:.4f}, mAP50-95={cent_mAP5095:.4f}, P={cent_p:.4f}, R={cent_r:.4f}")

    results['centralized'] = {
        'n_train_images': total_train,
        'fl_val_mAP50': round(cent_mAP50, 4),
        'fl_val_mAP5095': round(cent_mAP5095, 4),
        'fl_val_P': round(cent_p, 4),
        'fl_val_R': round(cent_r, 4),
        'train_time_min': round(dt / 60, 1),
    }
    release_model(cent_model)

    # 保存汇总
    result_path = os.path.join(OUTPUT_BASE, 'baseline_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\n{'='*60}")
    log(f"Baseline 汇总 (统一 val_set)")
    log(f"{'='*60}")
    log(f"\n{'节点':<14} {'训练图':<8} {'mAP50':<12} {'mAP50-95':<12} {'P':<10} {'R':<10}")
    log("-" * 66)
    for node in ['b1', 'b2', 'b3', 'centralized']:
        r = results[node]
        log(f"  {node:<12} {r['n_train_images']:<8} {r['fl_val_mAP50']:<12} {r['fl_val_mAP5095']:<12} {r['fl_val_P']:<10} {r['fl_val_R']:<10}")

    log(f"\n结果: {result_path}")


if __name__ == '__main__':
    main()
