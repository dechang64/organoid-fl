"""
fl_detect_sim.py — Organoid YOLO Detection FL Simulation (v2)
==============================================================

Phase 2: YOLO26n on Intestinal Organoid detection dataset (840 images, 23K bboxes)
3 clients, 4 strategies (FedAvg / FedProx / EWA-v2 / EWA-FedProx), 3 Non-IID scenarios.

Dataset: intestinal_organoid/OrganoidDataset/ (YOLO format, ready for Ultralytics)
  - 756 train / 84 val images
  - 4 classes: organoid0, organoid1, organoid3, spheroid
  - 23,065 bounding box annotations
  - Long-tail ratio: 5.3:1

Baseline (yolo26n, imgsz=1280, 100 epochs):
  - mAP50=0.852, mAP50-95=0.596, training=0.97h

Usage (on local GPU machine, e.g. RTX 3060):
    # Full run: all 3 scenarios × 4 strategies
    python fl_detect_sim.py --data data/intestinal_organoid/OrganoidDataset/data.yaml --weights yolo26n.pt --rounds 20 --epochs 2 --device 0 --imgsz 1280 --batch 8

    # Quick test
    python fl_detect_sim.py --data ... --weights yolo26n.pt --quick --device 0

    # Single scenario
    python fl_detect_sim.py --data ... --weights yolo26n.pt --scenarios extreme --strategies FedAvg EWA-v2 --rounds 20 --epochs 2

    # With EWA warmup (default: 2 rounds)
    python fl_detect_sim.py --data ... --weights yolo26n.pt --warmup 2
"""

import sys, os, json, time, shutil, argparse, copy, gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

CLASS_NAMES = ["organoid0", "organoid1", "organoid3", "spheroid"]
NUM_CLASSES = 4

# ─── Scenario Definitions ─────────────────────────────────────────────
SCENARIOS = {
    "iid": {
        "dominant_ratio": 0.33,  # ~IID: each client gets ~33% of each class
        "description": "IID Balanced — each client sees similar class distribution",
    },
    "moderate": {
        "dominant_ratio": 0.70,
        "description": "Moderate Non-IID — 70% dominant class per client",
    },
    "extreme": {
        "dominant_ratio": 0.95,
        "description": "Extreme Non-IID — 95% dominant class per client",
    },
}

STRATEGIES_ALL = ["FedAvg", "FedProx", "EWA-v2", "EWA-FedProx"]


# ─── Non-IID Data Split ───────────────────────────────────────────────

def split_non_iid(data_yaml_path, output_dir, n_clients=3, dominant_ratio=0.8, seed=42):
    """Split organoid detection dataset into n_clients with class-skewed distributions.

    Client 0: organoid0 (cystic) dominant
    Client 1: organoid1 + organoid3 (maturing) dominant
    Client 2: spheroid dominant

    Uses image-level assignment: an image goes to client(s) based on its dominant class.
    """
    import yaml

    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    # base_path: use cfg['path'] if present, otherwise infer from data_yaml location
    if 'path' in cfg and cfg['path']:
        base_path = Path(cfg['path'])
    else:
        base_path = Path(data_yaml_path).resolve().parent
    names = cfg['names']
    nc = cfg['nc']

    dominant_map = {
        0: [0],       # organoid0 specialist
        1: [1, 2],    # maturing organoids
        2: [3],       # spheroid specialist
    }

    # Read training labels
    label_dir = base_path / 'train' / 'labels'
    image_dir = base_path / 'train' / 'images'

    # Handle path prefix issues (e.g., "dataset/train" → "train")
    if not label_dir.exists():
        # Try stripping common prefixes
        for prefix in ['dataset/', 'dataset\\', '']:
            alt_label = base_path / f'{prefix}train' / 'labels'
            alt_image = base_path / f'{prefix}train' / 'images'
            if alt_label.exists():
                label_dir = alt_label
                image_dir = alt_image
                break

    if not label_dir.exists():
        # Last resort: search for labels directory
        candidates = list(base_path.rglob('labels'))
        if candidates:
            label_dir = candidates[0]
            image_dir = label_dir.parent.parent / 'images'
            if not image_dir.exists():
                image_dir = label_dir.parent / 'images'

    class_to_files = defaultdict(list)
    all_label_files = sorted(label_dir.glob('*.txt'))

    for lf in all_label_files:
        stem = lf.stem
        with open(lf) as f:
            classes_in_file = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes_in_file.add(int(parts[0]))
        for cls in classes_in_file:
            class_to_files[cls].append(stem)

    print(f"Dataset: {len(all_label_files)} training images, {nc} classes")
    for c in range(nc):
        print(f"  Class {c} ({names[c]}): {len(class_to_files[c])} images")

    # Split per client
    rng = np.random.RandomState(seed)
    client_stems = {i: set() for i in range(n_clients)}

    for cid in range(n_clients):
        dom_classes = dominant_map[cid]
        non_dom_classes = [c for c in range(nc) if c not in dom_classes]

        for cls in dom_classes:
            files = list(class_to_files[cls])
            rng.shuffle(files)
            n_take = int(dominant_ratio * len(files))
            client_stems[cid].update(files[:n_take])

        n_non_dom = len(non_dom_classes)
        for cls in non_dom_classes:
            files = list(class_to_files[cls])
            rng.shuffle(files)
            n_take = max(1, int((1 - dominant_ratio) * len(files) / n_non_dom))
            client_stems[cid].update(files[:n_take])

    # Create client datasets (YOLO format)
    client_yamls = []
    for cid in range(n_clients):
        client_dir = Path(output_dir) / f"client_{cid}"
        img_train = client_dir / "images" / "train"
        img_val = client_dir / "images" / "val"
        lbl_train = client_dir / "labels" / "train"
        lbl_val = client_dir / "labels" / "val"

        for d in [img_train, img_val, lbl_train, lbl_val]:
            d.mkdir(parents=True, exist_ok=True)

        # Copy training files (use copy2 for Windows compatibility)
        for stem in client_stems[cid]:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                src_img = image_dir / f"{stem}{ext}"
                if src_img.exists():
                    dst_img = img_train / f"{stem}{ext}"
                    if not dst_img.exists():
                        shutil.copy2(str(src_img), str(dst_img))
                    break

            src_lbl = label_dir / f"{stem}.txt"
            dst_lbl = lbl_train / f"{stem}.txt"
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.copy2(str(src_lbl), str(dst_lbl))

        # Copy ALL validation files (shared val set)
        val_img_dir = base_path / 'val' / 'images'
        val_lbl_dir = base_path / 'val' / 'labels'

        # Handle path prefix for val too
        if not val_img_dir.exists():
            for prefix in ['dataset/', 'dataset\\', '']:
                alt = base_path / f'{prefix}val' / 'images'
                if alt.exists():
                    val_img_dir = alt
                    val_lbl_dir = base_path / f'{prefix}val' / 'labels'
                    break

        for vf in sorted(val_img_dir.glob('*')):
            dst = img_val / vf.name
            if not dst.exists():
                shutil.copy2(str(vf), str(dst))
        for vf in sorted(val_lbl_dir.glob('*')):
            dst = lbl_val / vf.name
            if not dst.exists():
                shutil.copy2(str(vf), str(dst))

        # Write data.yaml
        client_yaml = {
            'path': str(client_dir).replace('\\', '/'),
            'train': 'images/train',
            'val': 'images/val',
            'nc': nc,
            'names': names,
        }
        yaml_path = client_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(client_yaml, f, default_flow_style=False)

        client_yamls.append(str(yaml_path))

        n_train = len(list(img_train.glob('*')))
        n_val = len(list(img_val.glob('*')))
        dom_names = [names[c] for c in dominant_map[cid]]
        print(f"  Client {cid}: {n_train} train / {n_val} val, dominant={dom_names}")

    return client_yamls, dominant_map


def split_iid(data_yaml_path, output_dir, n_clients=3, seed=42):
    """IID split: random shuffle, evenly distribute."""
    import yaml

    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    if 'path' in cfg and cfg['path']:
        base_path = Path(cfg['path'])
    else:
        base_path = Path(data_yaml_path).resolve().parent
    names = cfg['names']
    nc = cfg['nc']

    label_dir = base_path / 'train' / 'labels'
    image_dir = base_path / 'train' / 'images'

    if not label_dir.exists():
        for prefix in ['dataset/', 'dataset\\', '']:
            alt_label = base_path / f'{prefix}train' / 'labels'
            if alt_label.exists():
                label_dir = alt_label
                image_dir = base_path / f'{prefix}train' / 'images'
                break

    all_stems = sorted([lf.stem for lf in label_dir.glob('*.txt')])
    rng = np.random.RandomState(seed)
    rng.shuffle(all_stems)

    chunk_size = len(all_stems) // n_clients
    client_stems = {}
    for cid in range(n_clients):
        start = cid * chunk_size
        end = start + chunk_size if cid < n_clients - 1 else len(all_stems)
        client_stems[cid] = set(all_stems[start:end])

    dominant_map = {0: [0], 1: [1, 2], 2: [3]}  # placeholder for IID

    client_yamls = []
    for cid in range(n_clients):
        client_dir = Path(output_dir) / f"client_{cid}"
        img_train = client_dir / "images" / "train"
        img_val = client_dir / "images" / "val"
        lbl_train = client_dir / "labels" / "train"
        lbl_val = client_dir / "labels" / "val"

        for d in [img_train, img_val, lbl_train, lbl_val]:
            d.mkdir(parents=True, exist_ok=True)

        for stem in client_stems[cid]:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                src_img = image_dir / f"{stem}{ext}"
                if src_img.exists():
                    dst_img = img_train / f"{stem}{ext}"
                    if not dst_img.exists():
                        shutil.copy2(str(src_img), str(dst_img))
                    break
            src_lbl = label_dir / f"{stem}.txt"
            dst_lbl = lbl_train / f"{stem}.txt"
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.copy2(str(src_lbl), str(dst_lbl))

        val_img_dir = base_path / 'val' / 'images'
        val_lbl_dir = base_path / 'val' / 'labels'
        if not val_img_dir.exists():
            for prefix in ['dataset/', 'dataset\\', '']:
                alt = base_path / f'{prefix}val' / 'images'
                if alt.exists():
                    val_img_dir = alt
                    val_lbl_dir = base_path / f'{prefix}val' / 'labels'
                    break

        for vf in sorted(val_img_dir.glob('*')):
            dst = img_val / vf.name
            if not dst.exists():
                shutil.copy2(str(vf), str(dst))
        for vf in sorted(val_lbl_dir.glob('*')):
            dst = lbl_val / vf.name
            if not dst.exists():
                shutil.copy2(str(vf), str(dst))

        client_yaml = {
            'path': str(client_dir).replace('\\', '/'),
            'train': 'images/train',
            'val': 'images/val',
            'nc': nc,
            'names': names,
        }
        yaml_path = client_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(client_yaml, f, default_flow_style=False)
        client_yamls.append(str(yaml_path))

        n_train = len(list(img_train.glob('*')))
        print(f"  Client {cid}: {n_train} train (IID)")

    return client_yamls, dominant_map


# ─── Aggregation Strategies ───────────────────────────────────────────

def fedavg_aggregate(state_dicts, weights=None):
    """Weighted average of model state dicts."""
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
    """EWA: weight by quality signal (mAP50-95 recommended for detection)."""
    key = "mAP" if signal == "mAP" else "mAP50"
    maps = [m[key] for m in client_metrics]
    total = sum(maps)
    if total == 0:
        return [1.0 / len(maps)] * len(maps)
    return [m / total for m in maps]


def fedprox_interpolate(local_sd, global_sd, mu=0.01):
    """Approximate FedProx by interpolating local weights toward global."""
    result = {}
    for key in local_sd:
        if local_sd[key].dtype in (torch.int32, torch.int64):
            result[key] = local_sd[key].clone()
            continue
        # Ensure device match
        g = global_sd[key].float().to(local_sd[key].device)
        result[key] = (1 - mu) * local_sd[key].float() + mu * g
    return result


# ─── YOLO Training & Evaluation ───────────────────────────────────────

def fix_data_yaml(data_yaml_path):
    """Ensure data.yaml has 'path' field and valid train/val directories."""
    import yaml as _yaml

    with open(data_yaml_path) as f:
        data_cfg = _yaml.safe_load(f)

    need_fix = False
    yaml_dir = Path(data_yaml_path).resolve().parent

    # Set path if missing
    if 'path' not in data_cfg or not data_cfg['path']:
        data_cfg['path'] = str(yaml_dir)
        need_fix = True

    # Verify train/val directories exist under path; strip prefixes if needed
    base = Path(data_cfg['path'])
    for split_key in ['train', 'val', 'test']:
        if split_key not in data_cfg:
            continue
        split_dir = base / data_cfg[split_key]
        if not split_dir.exists():
            # Try stripping 'dataset/' prefix
            stripped = data_cfg[split_key].replace('dataset/', '').replace('dataset\\', '')
            if (base / stripped).exists():
                data_cfg[split_key] = stripped
                need_fix = True

    if need_fix:
        with open(data_yaml_path, 'w') as f:
            _yaml.dump(data_cfg, f, default_flow_style=False)
        print(f"  Fixed data.yaml: path={data_cfg['path']}, train={data_cfg.get('train')}, val={data_cfg.get('val')}")

    return data_yaml_path


def train_client_yolo(model_path, data_yaml, epochs, device, client_id, project_dir,
                      imgsz=640, batch=8):
    """Train a single FL client using YOLO."""
    from ultralytics import YOLO

    # Fix data.yaml before training
    data_yaml = fix_data_yaml(data_yaml)

    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        amp=True,
        project=str(project_dir),
        name=client_id,
        exist_ok=True,
        verbose=False,
        plots=False,
    )

    mAP50 = float(results.box.map50) if results.box.map50 is not None else 0.0
    mAP = float(results.box.map) if results.box.map is not None else 0.0

    per_class = {}
    if results.box.maps is not None:
        for i, ap in enumerate(results.box.maps):
            per_class[f"class_{i}"] = float(ap)

    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else Path(project_dir) / client_id
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    weights_path = best_path if best_path.exists() else last_path

    if not weights_path.exists():
        candidates = list(save_dir.rglob("best.pt")) + list(save_dir.rglob("last.pt"))
        if candidates:
            weights_path = candidates[0]

    return {
        "client_id": client_id,
        "mAP50": mAP50,
        "mAP": mAP,
        "per_class_ap": per_class,
        "weights_path": str(weights_path),
    }


def evaluate_global(model_path, data_yaml, device, imgsz=640, batch=8):
    """Evaluate global model on full validation set."""
    from ultralytics import YOLO

    # Fix data.yaml before eval
    data_yaml = fix_data_yaml(data_yaml)

    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        amp=True,
        verbose=False,
    )

    mAP50 = float(results.box.map50) if results.box.map50 is not None else 0.0
    mAP = float(results.box.map) if results.box.map is not None else 0.0

    per_class = {}
    if results.box.maps is not None:
        for i, ap in enumerate(results.box.maps):
            per_class[f"class_{i}"] = float(ap)

    return {"mAP50": mAP50, "mAP": mAP, "per_class_ap": per_class}


# ─── FL Training Loop ─────────────────────────────────────────────────

def run_fl_simulation(args):
    """Main FL simulation loop."""
    from ultralytics import YOLO

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = output_dir / "runs"

    # Fix main data.yaml
    args.data = fix_data_yaml(args.data)

    print("=" * 70)
    print("Organoid YOLO Detection FL Simulation (v2)")
    print(f"Dataset: {args.data}")
    print(f"Model: {args.weights}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Strategies: {args.strategies}")
    print(f"Rounds: {args.rounds}, Epochs/round: {args.epochs}")
    print(f"Image size: {args.imgsz}, Batch: {args.batch}")
    print(f"EWA warmup: {args.warmup} rounds")
    print(f"Device: {args.device}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1: Adapt pretrained model to num_classes (80→4)
    # Ultralytics auto-reshapes detection head during training, but state_dict
    # loading requires matching architecture. Train 1 epoch to get 4-class model.
    print("\n[1/3] Adapting pretrained model to 4-class (1 epoch warmup)...")
    global_init = output_dir / "global_init_4class.pt"
    if global_init.exists():
        print(f"  Found existing adapted model: {global_init}")
    else:
        adapt_model = YOLO(args.weights)
        adapt_result = adapt_model.train(
            data=args.data, epochs=1, imgsz=args.imgsz, batch=args.batch,
            device=args.device, project=str(project_dir), name="model_adapt",
            exist_ok=True, verbose=False)
        adapted_weights = Path(adapt_result.save_dir) / "weights" / "last.pt"
        shutil.copy2(str(adapted_weights), str(global_init))
        print(f"  Adapted model saved: {global_init}")
        del adapt_model
        gc.collect()
        torch.cuda.empty_cache()

    # Step 2: Run scenarios
    all_scenario_results = {}

    for scenario_name in args.scenarios:
        scenario_cfg = SCENARIOS.get(scenario_name, SCENARIOS["moderate"])
        dominant_ratio = scenario_cfg["dominant_ratio"]

        print(f"\n{'='*70}")
        print(f"[2/3] Scenario: {scenario_name} (dominant_ratio={dominant_ratio})")
        print(f"  {scenario_cfg['description']}")
        print(f"{'='*70}")

        # Split dataset for this scenario
        scenario_data_dir = output_dir / f"client_data_{scenario_name}"
        if dominant_ratio <= 0.34:
            client_yamls, dominant_map = split_iid(
                args.data, scenario_data_dir, n_clients=3, seed=42)
        else:
            client_yamls, dominant_map = split_non_iid(
                args.data, scenario_data_dir,
                n_clients=3, dominant_ratio=dominant_ratio, seed=42)

        # Run strategies
        strategy_results = {}

        for strategy in args.strategies:
            print(f"\n{'='*70}")
            print(f"[3/3] Scenario={scenario_name}, Strategy: {strategy}")
            print(f"{'='*70}")

            global_path = output_dir / f"global_{scenario_name}_{strategy}_r0.pt"
            shutil.copy2(global_init, global_path)

            rounds_data = []

            for r in range(args.rounds):
                t0 = time.time()
                print(f"\n  Round {r+1}/{args.rounds}...", flush=True)

                global_model = YOLO(str(global_path))
                global_sd = global_model.model.state_dict()

                client_results = []
                client_state_dicts = []

                for cid in range(3):
                    print(f"    Training client {cid}...", end=" ", flush=True)
                    cid_str = f"{scenario_name}_{strategy}_r{r}_c{cid}"

                    result = train_client_yolo(
                        model_path=str(global_path),
                        data_yaml=client_yamls[cid],
                        epochs=args.epochs,
                        device=args.device,
                        client_id=cid_str,
                        project_dir=project_dir,
                        imgsz=args.imgsz,
                        batch=args.batch,
                    )
                    result["client_id"] = cid
                    result["dominant_classes"] = dominant_map[cid]
                    client_results.append(result)
                    print(f"mAP50={result['mAP50']:.4f} mAP50-95={result['mAP']:.4f}", flush=True)

                    # Print per-class AP
                    pca = result.get("per_class_ap", {})
                    pca_str = " | ".join(f"{CLASS_NAMES[i]}:{pca.get(f'class_{i}',0):.3f}"
                                         for i in range(NUM_CLASSES))
                    print(f"      Per-class AP: {pca_str}", flush=True)

                    if Path(result["weights_path"]).exists():
                        client_model = YOLO(result["weights_path"])
                        client_sd = client_model.model.state_dict()

                        if strategy in ("FedProx", "EWA-FedProx"):
                            client_sd = fedprox_interpolate(client_sd, global_sd, mu=args.mu)

                        client_state_dicts.append(client_sd)
                        del client_model

                    gc.collect()
                    torch.cuda.empty_cache()

                # Aggregate
                if not client_state_dicts:
                    agg_path = output_dir / f"global_{scenario_name}_{strategy}_r{r+1}.pt"
                    shutil.copy2(str(global_path), str(agg_path))
                else:
                    if strategy in ("EWA-v2", "EWA-FedProx"):
                        if r < args.warmup:
                            # Warmup: use FedAvg for first N rounds
                            agg_sd = fedavg_aggregate(client_state_dicts)
                            print(f"    EWA warmup round {r+1}/{args.warmup}, using FedAvg", flush=True)
                        else:
                            w = compute_ewa_weights(client_results, signal="mAP")
                            print(f"    EWA(mAP50-95) weights: {[f'{wi:.4f}' for wi in w]}", flush=True)
                            agg_sd = fedavg_aggregate(client_state_dicts, w)
                    elif strategy == "FedProx":
                        agg_sd = fedavg_aggregate(client_state_dicts)
                    else:  # FedAvg
                        agg_sd = fedavg_aggregate(client_state_dicts)

                    agg_path = output_dir / f"global_{scenario_name}_{strategy}_r{r+1}.pt"
                    # Use torch.load/save directly to avoid YOLO() reconstructing 80-class arch
                    ckpt = torch.load(str(global_path), map_location='cpu', weights_only=False)
                    model_obj = ckpt['model']
                    base_sd = model_obj.state_dict()
                    base_sd.update(agg_sd)
                    model_obj.load_state_dict(base_sd)
                    ckpt['model'] = model_obj
                    torch.save(ckpt, str(agg_path))
                    del ckpt, model_obj, base_sd, agg_sd

                # Evaluate
                print(f"    Evaluating global model...", end=" ", flush=True)
                eval_result = evaluate_global(str(agg_path), args.data, args.device,
                                              imgsz=args.imgsz, batch=args.batch)
                print(f"mAP50={eval_result['mAP50']:.4f}, mAP50-95={eval_result['mAP']:.4f}", flush=True)

                # Print per-class AP for global model
                pca = eval_result.get("per_class_ap", {})
                pca_str = " | ".join(f"{CLASS_NAMES[i]}:{pca.get(f'class_{i}',0):.3f}"
                                     for i in range(NUM_CLASSES))
                print(f"    Global per-class AP: {pca_str}", flush=True)

                elapsed = time.time() - t0
                round_data = {
                    "round": r + 1,
                    "global_mAP50": eval_result["mAP50"],
                    "global_mAP": eval_result["mAP"],
                    "global_per_class": eval_result["per_class_ap"],
                    "client_results": [{
                        "client_id": cr["client_id"],
                        "mAP50": cr["mAP50"],
                        "mAP": cr["mAP"],
                        "per_class_ap": cr["per_class_ap"],
                        "dominant_classes": cr["dominant_classes"],
                    } for cr in client_results],
                    "elapsed_sec": round(elapsed, 1),
                }
                rounds_data.append(round_data)

                global_path = agg_path
                del global_model, client_state_dicts
                gc.collect()
                torch.cuda.empty_cache()

                print(f"    Round {r+1} done in {elapsed:.0f}s", flush=True)

            strategy_results[strategy] = rounds_data

            with open(output_dir / f"{scenario_name}_{strategy}_rounds.json", "w") as f:
                json.dump(rounds_data, f, indent=2, ensure_ascii=False, default=str)

        all_scenario_results[scenario_name] = strategy_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": "Intestinal Organoid Detection (840 imgs, 23K bboxes, 4 classes)",
            "model": args.weights,
            "imgsz": args.imgsz,
            "rounds": args.rounds,
            "epochs": args.epochs,
            "clients": 3,
            "warmup": args.warmup,
            "mu": args.mu,
            "device": args.device,
            "scenarios": args.scenarios,
            "strategies": args.strategies,
        },
        "scenarios": {},
    }

    for scenario_name, strategy_results in all_scenario_results.items():
        summary["scenarios"][scenario_name] = {}
        print(f"\n  Scenario: {scenario_name}")
        for strategy, rounds in strategy_results.items():
            final = rounds[-1]
            best = max(rounds, key=lambda x: x["global_mAP"])
            summary["scenarios"][scenario_name][strategy] = {
                "final_mAP50": round(final["global_mAP50"], 4),
                "final_mAP": round(final["global_mAP"], 4),
                "best_mAP": round(best["global_mAP"], 4),
                "best_mAP50": round(best["global_mAP50"], 4),
                "best_round": best["round"],
                "final_per_class": final["global_per_class"],
                "total_time_sec": sum(r["elapsed_sec"] for r in rounds),
            }
            print(f"    {strategy:15s}: final_mAP50-95={final['global_mAP']:.4f}, "
                  f"best={best['global_mAP']:.4f} (round {best['round']})")

    with open(output_dir / "fl_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_dir}/")
    print(f"Summary: {output_dir / 'fl_summary.json'}")
    return summary


# ─── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organoid YOLO Detection FL Simulation (v2)")
    parser.add_argument("--data", required=True,
                        help="Path to intestinal organoid data.yaml")
    parser.add_argument("--weights", default="yolo26n.pt",
                        help="Path to pretrained weights (default: yolo26n.pt)")
    parser.add_argument("--output", default="./fl_detect_results",
                        help="Output directory")
    parser.add_argument("--rounds", type=int, default=20, help="FL rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (640 or 1280)")
    parser.add_argument("--device", default="0", help="CUDA device (0, cpu)")
    parser.add_argument("--scenarios", nargs="+", default=["iid", "moderate", "extreme"],
                        choices=["iid", "moderate", "extreme"],
                        help="Non-IID scenarios to run")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES_ALL,
                        choices=STRATEGIES_ALL,
                        help="FL strategies to run")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx μ")
    parser.add_argument("--warmup", type=int, default=2,
                        help="EWA warmup rounds (use FedAvg for first N rounds)")
    parser.add_argument("--quick", action="store_true", help="Quick test: 2 rounds, 1 epoch")

    args = parser.parse_args()
    if args.quick:
        args.rounds = 2
        args.epochs = 1
        args.scenarios = ["extreme"]
        args.strategies = ["FedAvg", "EWA-v2"]

    run_fl_simulation(args)
