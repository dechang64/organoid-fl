r"""
鼠肝 Baseline: 各节点独立训练 (无 FL)

与 FL 实验严格可比:
  - 同 val_set (9张混合: B1×3 + B2×3 + B3×3)
  - 同 train/val split (VAL_INDICES 排除 val 图)
  - 同模型 (yolo12n.pt COCO 预训练)
  - 同 imgsz=640, batch=4
  - 100ep patience=50 (与 M5 一致, FL 用 10ep/round × 10 round = 100ep 等价)

输出:
  runs\mouse_liver_baseline\{node}\  — 训练结果
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
PATIENCE = 50

VAL_INDICES = {
    'b1': [7, 8, 9],
    'b2': [7, 8, 9],
    'b3': [17, 18, 19],
}


def log(msg):
    print(msg, flush=True)


def safe_path(p):
    return p.replace('\\', '/')


def write_node_yaml(data_dir, node_name):
    """与 fl_sequential.py 完全一致的 data.yaml"""
    node_yaml = os.path.join(data_dir, 'data.yaml')
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')

    split_dir = os.path.join(data_dir, 'fl_split')
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
            continue
        shutil.copy2(os.path.join(img_dir, img_file), os.path.join(train_img_dir, img_file))
        lbl_file = img_file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(lbl_dir, lbl_file)):
            shutil.copy2(os.path.join(lbl_dir, lbl_file), os.path.join(train_lbl_dir, lbl_file))

    with open(node_yaml, 'w') as f:
        f.write(f'path: {safe_path(split_dir)}\ntrain: images\nval: images\nnc: 1\nnames: [\'organoid\']\n')

    cache_path = os.path.join(split_dir, 'labels', 'labels.cache')
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except PermissionError:
            pass
    return node_yaml


def prepare_val_set():
    """复用 FL 的 val_set (如果已存在) 或重新创建"""
    val_yaml = os.path.join(FL_SEQ_DIR, 'val.yaml')
    if os.path.exists(val_yaml):
        return val_yaml
    # 不存在则创建
    os.makedirs(os.path.join(FL_SEQ_DIR, 'val_set', 'images'), exist_ok=True)
    os.makedirs(os.path.join(FL_SEQ_DIR, 'val_set', 'labels'), exist_ok=True)
    for node, ddir in BATCH_DIRS.items():
        img_src = os.path.join(ddir, 'images')
        lbl_src = os.path.join(ddir, 'labels')
        for idx in VAL_INDICES[node]:
            fname = f'image_{idx:02d}'
            img_path = os.path.join(img_src, f'{fname}.jpg')
            lbl_path = os.path.join(lbl_src, f'{fname}.txt')
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                shutil.copy2(img_path, os.path.join(FL_SEQ_DIR, 'val_set', 'images', f'{node}_{fname}.jpg'))
                shutil.copy2(lbl_path, os.path.join(FL_SEQ_DIR, 'val_set', 'labels', f'{node}_{fname}.txt'))
    val_dir = os.path.join(FL_SEQ_DIR, 'val_set')
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
        log(f"\n{'='*60}")
        log(f"训练 {node_name} (独立 baseline, {EPOCHS}ep patience={PATIENCE})")
        log(f"{'='*60}")

        node_yaml = write_node_yaml(data_dir, node_name)
        node_dir = os.path.join(OUTPUT_BASE, node_name)
        os.makedirs(node_dir, exist_ok=True)

        model = YOLO('yolo12n.pt')
        t0 = time.time()
        model.train(
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
            close_mosaic=5,
            patience=PATIENCE,
            verbose=True,
        )
        dt = time.time() - t0
        log(f"  训练时间: {dt/60:.1f} min")

        # 用 FL 同一 val_set 评估
        val_res = model.val(data=val_yaml, device=DEVICE, project=node_dir,
                            name='val_fl_set', exist_ok=True)
        mAP50 = float(val_res.box.map50)
        mAP5095 = float(val_res.box.map)
        precision = float(val_res.box.mp)
        recall = float(val_res.box.mr)

        log(f"  ★ {node_name} → FL val_set: mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}, P={precision:.4f}, R={recall:.4f}")

        # 也用各自 val 图评估 (节点内 val)
        node_val_res = model.val(data=node_yaml, device=DEVICE, project=node_dir,
                                  name='val_own', exist_ok=True)
        own_mAP50 = float(node_val_res.box.map50)
        own_mAP5095 = float(node_val_res.box.map)
        own_p = float(node_val_res.box.mp)
        own_r = float(node_val_res.box.mr)

        log(f"  ★ {node_name} → own val: mAP50={own_mAP50:.4f}, mAP50-95={own_mAP5095:.4f}, P={own_p:.4f}, R={own_r:.4f}")

        results[node_name] = {
            'fl_val_mAP50': round(mAP50, 4),
            'fl_val_mAP5095': round(mAP5095, 4),
            'fl_val_P': round(precision, 4),
            'fl_val_R': round(recall, 4),
            'own_val_mAP50': round(own_mAP50, 4),
            'own_val_mAP5095': round(own_mAP5095, 4),
            'own_val_P': round(own_p, 4),
            'own_val_R': round(own_r, 4),
            'train_time_min': round(dt / 60, 1),
        }

        del model

    # 保存汇总
    result_path = os.path.join(OUTPUT_BASE, 'baseline_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\n{'='*60}")
    log(f"Baseline 汇总")
    log(f"{'='*60}")
    log(f"\n{'节点':<8} {'FL val mAP50':<15} {'FL val mAP50-95':<18} {'Own val mAP50':<15} {'Own val mAP50-95':<18}")
    log("-" * 74)
    for node in ['b1', 'b2', 'b3']:
        r = results[node]
        log(f"  {node:<6} {r['fl_val_mAP50']:<15} {r['fl_val_mAP5095']:<18} {r['own_val_mAP50']:<15} {r['own_val_mAP5095']:<18}")

    log(f"\n结果: {result_path}")


if __name__ == '__main__':
    main()
