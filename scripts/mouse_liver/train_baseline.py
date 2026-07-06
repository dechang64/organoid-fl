r"""
鼠肝 Baseline: 各节点独立训练 + 集中式上界 (无 FL)

复用 fl_sequential.py 的: write_node_yaml, prepare_val_set, release_model, log, safe_path, VAL_INDICES, BATCH_DIRS

与 FL 实验严格可比:
  - 同 val_set (9张混合: B1×3 + B2×3 + B3×3)
  - 同 train/val split (VAL_INDICES 排除 val 图)
  - 同模型 (yolo12n.pt COCO 预训练)
  - 同 imgsz=640, batch=4
  - 100ep, close_mosaic=50 (FL: 10ep×10round, close_mosaic=5 → 50% 无 mosaic)

四组 baseline:
  1. B1 独立 (7 train)    — 节点独立下界
  2. B2 独立 (7 train)    — 节点独立下界
  3. B3 独立 (17 train)   — 节点独立下界
  4. 集中式 (31 train)    — B1+B2+B3 合并训练, FL 上界

Usage:
  cd C:\Users\decha\organoid-fl
  .\.venv\Scripts\activate
  python scripts\mouse_liver\train_baseline.py
"""
import os, sys, json, time, shutil

# 复用 fl_sequential.py 的所有基础设施
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_sequential import (
    BATCH_DIRS, VAL_INDICES, IMGSZ, BATCH_SIZE, WORKERS, DEVICE,
    log, safe_path, release_model, write_node_yaml, prepare_val_set,
)

OUTPUT_BASE = r"runs\mouse_liver_baseline"
FL_SEQ_DIR = r"runs\mouse_liver_fl_seq"

EPOCHS = 100
# close_mosaic: FL 每轮 10ep, close_mosaic=5 → 50% 无 mosaic
# baseline 100ep → close_mosaic=50 匹配 FL 的 50% 比例
CLOSE_MOSAIC = 50


def write_centralized_yaml():
    """合并 B1+B2+B3 训练图 (排除各批 val 图) 到 centralized_split"""
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
            if not img_file.endswith(('.jpg', '.png')):
                continue
            name = os.path.splitext(img_file)[0]
            try:
                idx = int(name.split('_')[1])
            except (IndexError, ValueError):
                continue
            if idx in val_indices:
                continue
            # 加节点前缀避免不同 batch 同名文件覆盖
            dst_img = f'{node_name}_{img_file}'
            shutil.copy2(os.path.join(img_dir, img_file), os.path.join(cent_img_dir, dst_img))
            lbl_file = img_file.replace('.jpg', '.txt')
            if os.path.exists(os.path.join(lbl_dir, lbl_file)):
                shutil.copy2(os.path.join(lbl_dir, lbl_file), os.path.join(cent_lbl_dir, f'{node_name}_{lbl_file}'))
            total_train += 1

    cent_yaml = os.path.join(cent_dir, 'data.yaml')
    with open(cent_yaml, 'w') as f:
        f.write(f'path: {safe_path(cent_dir)}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')
    # 清 cache
    cache_path = os.path.join(cent_lbl_dir, 'labels.cache')
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except PermissionError:
            pass
    return cent_yaml, total_train


def train_one(name, node_yaml, n_train, imgsz=None, batch=None):
    """训练单个模型并在 FL val_set 上评估

    imgsz/batch 可选, 默认用全局 IMGSZ/BATCH_SIZE
    评估始终用 IMGSZ (640) 保持可比性
    """
    from ultralytics import YOLO

    if imgsz is None:
        imgsz = IMGSZ
    if batch is None:
        batch = BATCH_SIZE

    log(f"\n{'='*60}")
    log(f"训练 {name} ({EPOCHS}ep close_mosaic={CLOSE_MOSAIC}, {n_train} train images, imgsz={imgsz}, batch={batch})")
    log(f"{'='*60}")

    model = YOLO('yolo12n.pt')
    t0 = time.time()
    model.train(
        data=node_yaml,
        epochs=EPOCHS,
        imgsz=imgsz,
        batch=batch,
        device=DEVICE,
        workers=WORKERS,
        cache=False,
        project=OUTPUT_BASE,
        name=name,
        exist_ok=True,
        cos_lr=True,
        close_mosaic=CLOSE_MOSAIC,
        verbose=False,
    )
    dt = time.time() - t0
    log(f"  训练时间: {dt/60:.1f} min")

    # 用 FL 同一 val_set 评估
    val_res = model.val(data=val_yaml, device=DEVICE,
                        project=os.path.join(OUTPUT_BASE, name),
                        name='val_fl_set', exist_ok=True)
    mAP50 = float(val_res.box.map50)
    mAP5095 = float(val_res.box.map)
    precision = float(val_res.box.mp)
    recall = float(val_res.box.mr)
    log(f"  ★ {name} → FL val_set: mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}, P={precision:.4f}, R={recall:.4f}")

    release_model(model)

    return {
        'n_train_images': n_train,
        'fl_val_mAP50': round(mAP50, 4),
        'fl_val_mAP5095': round(mAP5095, 4),
        'fl_val_P': round(precision, 4),
        'fl_val_R': round(recall, 4),
        'train_time_min': round(dt / 60, 1),
    }


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 复用 FL 的 val_set
    global val_yaml
    val_yaml = prepare_val_set(FL_SEQ_DIR)
    log(f"val_yaml: {val_yaml}")

    results = {}

    # === 各节点独立训练 ===
    for node_name in ['b1', 'b2', 'b3']:
        data_dir = BATCH_DIRS[node_name]
        n_val = len(VAL_INDICES[node_name])
        n_total = len([f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith(('.jpg', '.png'))])
        n_train = n_total - n_val

        # 复用 fl_sequential.write_node_yaml (会写 fl_split/)
        node_yaml = write_node_yaml(data_dir, node_name)
        results[node_name] = train_one(node_name, node_yaml, n_train)

    # === 集中式上界 ===
    cent_yaml, total_train = write_centralized_yaml()
    results['centralized'] = train_one('centralized', cent_yaml, total_train)

    # === E9: B3 独立 imgsz=1280 ===
    log(f"\n{'#'*60}")
    log(f"# E9: B3 独立 imgsz=1280 (验证 B3 高分辨率下能否学到)")
    log(f"{'#'*60}")
    b3_yaml = write_node_yaml(BATCH_DIRS['b3'], 'b3')
    n_b3 = len([f for f in os.listdir(os.path.join(BATCH_DIRS['b3'], 'images')) if f.endswith(('.jpg', '.png'))]) - len(VAL_INDICES['b3'])
    results['b3_1280'] = train_one('b3_1280', b3_yaml, n_b3, imgsz=1280, batch=2)

    # === E11: 集中式 imgsz=1280 ===
    log(f"\n{'#'*60}")
    log(f"# E11: 集中式 imgsz=1280 (控制变量: 集中式高分辨率上界)")
    log(f"{'#'*60}")
    results['centralized_1280'] = train_one('centralized_1280', cent_yaml, total_train, imgsz=1280, batch=2)

    # 保存汇总
    result_path = os.path.join(OUTPUT_BASE, 'baseline_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\n{'='*60}")
    log(f"Baseline 汇总 (统一 val_set, 评估 imgsz=640)")
    log(f"{'='*60}")
    log(f"\n{'节点':<20} {'训练图':<8} {'imgsz':<8} {'mAP50':<12} {'mAP50-95':<12} {'P':<10} {'R':<10}")
    log("-" * 78)
    for node in ['b1', 'b2', 'b3', 'b3_1280', 'centralized', 'centralized_1280']:
        if node not in results:
            continue
        r = results[node]
        img = '1280' if '1280' in node else '640'
        log(f"  {node:<18} {r['n_train_images']:<8} {img:<8} {r['fl_val_mAP50']:<12} {r['fl_val_mAP5095']:<12} {r['fl_val_P']:<10} {r['fl_val_R']:<10}")

    log(f"\n结果: {result_path}")


if __name__ == '__main__':
    main()
