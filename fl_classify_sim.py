"""
fl_classify_sim.py — Organoid Classification FL Simulation
===========================================================

Phase 1: ResNet18 classifier on cropped organoid patches (23K, 4 classes)
3 clients, 3 strategies (FedAvg / FedProx / EWA), Non-IID splits.

Dataset: organoid_patches/ (cropped from Intestinal Organoid YOLO detection set)
  - organoid0 (cystic): 11,911 patches
  - organoid1 (early):  5,510 patches
  - organoid3 (late):   3,366 patches
  - spheroid:           2,265 patches
  Long-tail ratio: 5.3:1

Usage (on local GPU machine, e.g. RTX 3060):
    # Full run
    python fl_classify_sim.py --data ./organoid_patches --rounds 10 --epochs 5 --device 0

    # Quick test
    python fl_classify_sim.py --data ./organoid_patches --rounds 2 --epochs 2 --device 0 --quick

    # Specific strategies
    python fl_classify_sim.py --data ./organoid_patches --strategies FedAvg EWA-v2 --device 0
"""

import sys, os, json, time, shutil, argparse, copy, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
from datetime import datetime
from collections import defaultdict, OrderedDict
from PIL import Image

CLASS_NAMES = ["organoid0", "organoid1", "organoid3", "spheroid"]
NUM_CLASSES = 4


# ─── Dataset ──────────────────────────────────────────────────────────

class OrganoidPatchDataset(Dataset):
    """Folder-based classification dataset for organoid patches."""

    def __init__(self, root_dir, split="train", transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.classes = sorted([d.name for d in Path(root_dir).joinpath(split).iterdir()
                               if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = Path(root_dir) / split / cls_name
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class SubsetDataset(Dataset):
    """Subset of a dataset by indices."""

    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# ─── Model ────────────────────────────────────────────────────────────

def create_model(num_classes=NUM_CLASSES):
    """ResNet18 pretrained on ImageNet, replace final FC layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─── Non-IID Data Split ───────────────────────────────────────────────

def split_non_iid_class_skew(data_root, output_dir, n_clients=3, dominant_ratio=0.8,
                              seed=42):
    """Split organoid patches into n_clients with class-skewed distributions.

    Organoid has 4 classes with natural long-tail:
      organoid0: 11,911 | organoid1: 5,510 | organoid3: 3,366 | spheroid: 2,265

    Client dominant class assignment:
      Client 0: organoid0 (cystic) dominant
      Client 1: organoid1 + organoid3 (maturing) dominant
      Client 2: spheroid dominant

    Each client gets dominant_ratio of its dominant class samples.
    """
    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dominant class map
    dominant_map = {
        0: [0],       # organoid0 specialist
        1: [1, 2],    # maturing organoids specialist
        2: [3],       # spheroid specialist
    }

    # Load training dataset to get per-class indices
    train_transform = transforms.Compose([transforms.Resize((224, 224))])
    full_train = OrganoidPatchDataset(data_root, split="train", transform=train_transform)

    # Group by class
    class_indices = defaultdict(list)
    for idx in range(len(full_train)):
        _, label = full_train[idx]
        class_indices[label].append(idx)

    print(f"Training set: {len(full_train)} samples, {NUM_CLASSES} classes")
    for c in range(NUM_CLASSES):
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(class_indices[c])} samples")

    # Split per client
    client_indices = {i: set() for i in range(n_clients)}

    for cid in range(n_clients):
        dom_classes = dominant_map[cid]
        non_dom_classes = [c for c in range(NUM_CLASSES) if c not in dom_classes]

        # Dominant classes: take dominant_ratio
        for cls in dom_classes:
            indices = list(class_indices[cls])
            rng.shuffle(indices)
            n_take = int(dominant_ratio * len(indices))
            client_indices[cid].update(indices[:n_take])

        # Non-dominant: distribute (1-dominant_ratio) evenly
        n_non_dom = len(non_dom_classes)
        for cls in non_dom_classes:
            indices = list(class_indices[cls])
            rng.shuffle(indices)
            n_take = max(1, int((1 - dominant_ratio) * len(indices) / n_non_dom))
            client_indices[cid].update(indices[:n_take])

    # Print distribution
    for cid in range(n_clients):
        # Count per class
        cls_counts = defaultdict(int)
        for idx in client_indices[cid]:
            _, label = full_train[idx]
            cls_counts[label] += 1
        dom_names = [CLASS_NAMES[c] for c in dominant_map[cid]]
        dist_str = " | ".join(f"{CLASS_NAMES[c]}:{cls_counts[c]}" for c in range(NUM_CLASSES))
        print(f"  Client {cid} ({', '.join(dom_names)}): {len(client_indices[cid])} samples — {dist_str}")

    # Save split info
    split_info = {
        "n_clients": n_clients,
        "dominant_ratio": dominant_ratio,
        "dominant_map": {str(k): v for k, v in dominant_map.items()},
        "class_names": CLASS_NAMES,
        "client_counts": {
            str(cid): {
                "total": len(client_indices[cid]),
                "per_class": {CLASS_NAMES[c]: sum(1 for idx in client_indices[cid]
                            if full_train[idx][1] == c) for c in range(NUM_CLASSES)}
            } for cid in range(n_clients)
        },
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Build client datasets
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_aug = OrganoidPatchDataset(data_root, split="train", transform=train_transform)
    full_val = OrganoidPatchDataset(data_root, split="val", transform=val_transform)

    client_datasets = []
    for cid in range(n_clients):
        train_subset = SubsetDataset(full_train_aug, sorted(client_indices[cid]))
        # All clients share the same validation set
        client_datasets.append({"train": train_subset, "val": full_val})

    return client_datasets, dominant_map


def split_iid(data_root, output_dir, n_clients=3, seed=42):
    """IID split: random shuffle, evenly distribute."""
    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([transforms.Resize((224, 224))])
    full_train = OrganoidPatchDataset(data_root, split="train", transform=train_transform)

    all_indices = list(range(len(full_train)))
    rng.shuffle(all_indices)

    chunk_size = len(all_indices) // n_clients
    client_indices = {}
    for cid in range(n_clients):
        start = cid * chunk_size
        end = start + chunk_size if cid < n_clients - 1 else len(all_indices)
        client_indices[cid] = set(all_indices[start:end])

    dominant_map = {0: [0], 1: [1, 2], 2: [3]}  # placeholder

    print(f"IID split: {len(full_train)} samples → {n_clients} clients")
    for cid in range(n_clients):
        cls_counts = defaultdict(int)
        for idx in client_indices[cid]:
            _, label = full_train[idx]
            cls_counts[label] += 1
        dist_str = " | ".join(f"{CLASS_NAMES[c]}:{cls_counts[c]}" for c in range(NUM_CLASSES))
        print(f"  Client {cid}: {len(client_indices[cid])} samples — {dist_str}")

    # Build datasets
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_aug = OrganoidPatchDataset(data_root, split="train", transform=train_transform)
    full_val = OrganoidPatchDataset(data_root, split="val", transform=val_transform)

    client_datasets = []
    for cid in range(n_clients):
        train_subset = SubsetDataset(full_train_aug, sorted(client_indices[cid]))
        client_datasets.append({"train": train_subset, "val": full_val})

    return client_datasets, dominant_map


# ─── Training & Evaluation ───────────────────────────────────────────

def train_client(model, train_loader, epochs, device, lr=1e-3):
    """Train a single client model, return state dict and metrics."""
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += imgs.size(0)
        acc = correct / total if total > 0 else 0

    return model


def evaluate_model(model, val_loader, device):
    """Evaluate model, return accuracy + per-class accuracy."""
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            for p, l in zip(pred, labels):
                class_total[l.item()] += 1
                if p.item() == l.item():
                    class_correct[l.item()] += 1

    acc = correct / total if total > 0 else 0
    per_class = {CLASS_NAMES[c]: class_correct[c] / class_total[c]
                 if class_total[c] > 0 else 0 for c in range(NUM_CLASSES)}

    return {"accuracy": acc, "per_class_acc": per_class}


# ─── Aggregation Strategies ───────────────────────────────────────────

def fedavg_aggregate(state_dicts, weights=None):
    """Weighted average of model state dicts."""
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    avg = OrderedDict()
    for key in state_dicts[0]:
        if state_dicts[0][key].dtype in (torch.int32, torch.int64):
            avg[key] = state_dicts[0][key].clone()
            continue
        avg[key] = sum(w * sd[key].float() for sd, w in zip(state_dicts, weights))
    return avg


def compute_ewa_weights(client_metrics, signal="accuracy"):
    """Entropy-Weighted Aggregation: weight by quality signal (higher → more weight).

    For classification, signal = validation accuracy.
    For PCB detection, signal = mAP50-95. Same formula, different signal.
    """
    scores = [m[signal] for m in client_metrics]
    total = sum(scores)
    if total == 0:
        return [1.0 / len(scores)] * len(scores)
    return [s / total for s in scores]


def fedprox_interpolate(local_sd, global_sd, mu=0.01):
    """Approximate FedProx by interpolating local weights toward global."""
    result = OrderedDict()
    for key in local_sd:
        if local_sd[key].dtype in (torch.int32, torch.int64):
            result[key] = local_sd[key].clone()
            continue
        result[key] = (1 - mu) * local_sd[key].float() + mu * global_sd[key].float()
    return result


# ─── FL Training Loop ─────────────────────────────────────────────────

def run_fl_simulation(args):
    """Main FL simulation loop."""

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = f"cuda:{args.device}" if args.device.isdigit() else args.device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Training will be very slow on CPU.")
        print("Recommended: Run on local machine with RTX 3060 or better.")

    print("=" * 70)
    print("Organoid Classification FL Simulation")
    print(f"Dataset: {args.data}")
    print(f"Strategies: {args.strategies}")
    print(f"Rounds: {args.rounds}, Epochs/round: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Non-IID ratio: {args.dominant_ratio}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1: Split data
    print("\n[1/3] Splitting dataset...")
    if args.dominant_ratio <= 0.34:
        client_datasets, dominant_map = split_iid(
            args.data, output_dir / "split_info", n_clients=3, seed=42)
    else:
        client_datasets, dominant_map = split_non_iid_class_skew(
            args.data, output_dir / "split_info",
            n_clients=3, dominant_ratio=args.dominant_ratio, seed=42)

    # Step 2: Initialize global model
    print("\n[2/3] Initializing global model (ResNet18)...")
    global_model = create_model(NUM_CLASSES)
    global_sd = global_model.state_dict()
    del global_model

    # Step 3: Run FL for each strategy
    all_results = {}

    for strategy in args.strategies:
        print(f"\n{'='*70}")
        print(f"[3/3] Strategy: {strategy}")
        print(f"{'='*70}")

        # Reset global state dict
        current_sd = copy.deepcopy(global_sd)
        rounds_data = []

        for r in range(args.rounds):
            t0 = time.time()
            print(f"\n  Round {r+1}/{args.rounds}...", flush=True)

            client_state_dicts = []
            client_metrics = []

            for cid in range(3):
                print(f"    Training client {cid}...", end=" ", flush=True)
                model = create_model(NUM_CLASSES)
                model.load_state_dict(copy.deepcopy(current_sd))

                loader = DataLoader(client_datasets[cid]["train"],
                                    batch_size=args.batch_size, shuffle=True,
                                    num_workers=0, pin_memory=True)

                model = train_client(model, loader, args.epochs, device, lr=args.lr)

                # Evaluate on shared validation set
                val_loader = DataLoader(client_datasets[cid]["val"],
                                        batch_size=args.batch_size, shuffle=False,
                                        num_workers=0, pin_memory=True)
                metrics = evaluate_model(model, val_loader, device)
                print(f"acc={metrics['accuracy']:.4f}", flush=True)

                # Get state dict (apply FedProx if needed)
                local_sd = model.state_dict()
                if strategy == "FedProx":
                    local_sd = fedprox_interpolate(local_sd, current_sd, mu=args.mu)

                client_state_dicts.append(local_sd)
                client_metrics.append(metrics)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Aggregate
            if strategy in ("EWA", "EWA-v2"):
                w = compute_ewa_weights(client_metrics, signal="accuracy")
                print(f"    EWA weights: {[f'{wi:.4f}' for wi in w]}", flush=True)
                agg_sd = fedavg_aggregate(client_state_dicts, w)
            elif strategy == "FedProx":
                agg_sd = fedavg_aggregate(client_state_dicts)
            else:  # FedAvg
                agg_sd = fedavg_aggregate(client_state_dicts)

            current_sd = agg_sd

            # Evaluate global model on validation
            global_model = create_model(NUM_CLASSES)
            global_model.load_state_dict(current_sd)
            val_loader = DataLoader(client_datasets[0]["val"],
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)
            global_metrics = evaluate_model(global_model, val_loader, device)
            del global_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - t0
            round_data = {
                "round": r + 1,
                "global_accuracy": round(global_metrics["accuracy"], 4),
                "global_per_class": global_metrics["per_class_acc"],
                "client_results": [{
                    "client_id": cid,
                    "accuracy": round(client_metrics[cid]["accuracy"], 4),
                    "per_class_acc": client_metrics[cid]["per_class_acc"],
                    "dominant_classes": dominant_map.get(cid, []),
                } for cid in range(3)],
                "elapsed_sec": round(elapsed, 1),
            }
            rounds_data.append(round_data)

            print(f"    Global acc={global_metrics['accuracy']:.4f} "
                  f"({elapsed:.0f}s)", flush=True)

        all_results[strategy] = rounds_data

        with open(output_dir / f"{strategy}_rounds.json", "w") as f:
            json.dump(rounds_data, f, indent=2, ensure_ascii=False, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": "Intestinal Organoid Patches (23K, 4 classes)",
            "model": "ResNet18 (ImageNet pretrained)",
            "rounds": args.rounds,
            "epochs": args.epochs,
            "clients": 3,
            "dominant_ratio": args.dominant_ratio,
            "mu": args.mu,
            "device": str(device),
            "strategies": args.strategies,
        },
        "strategies": {},
    }

    for strategy, rounds in all_results.items():
        final = rounds[-1]
        best = max(rounds, key=lambda x: x["global_accuracy"])
        summary["strategies"][strategy] = {
            "final_acc": final["global_accuracy"],
            "best_acc": best["global_accuracy"],
            "final_per_class": final["global_per_class"],
            "total_time_sec": sum(r["elapsed_sec"] for r in rounds),
        }
        print(f"  {strategy}: final_acc={final['global_accuracy']:.4f}, "
              f"best_acc={best['global_accuracy']:.4f}")

    with open(output_dir / "fl_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_dir}/")
    return summary


# ─── Experiment Matrix Runner ─────────────────────────────────────────

NON_IID_LEVELS = {
    "iid":      {"dominant_ratio": 0.33, "desc": "IID (random 1/3 split)"},
    "mild":     {"dominant_ratio": 0.60, "desc": "Mild Non-IID (60% dominant)"},
    "moderate": {"dominant_ratio": 0.80, "desc": "Moderate Non-IID (80% dominant)"},
    "extreme":  {"dominant_ratio": 0.95, "desc": "Extreme Non-IID (95% dominant)"},
}

STRATEGIES = ["FedAvg", "FedProx", "EWA-v2"]

MU_VALUES = [0.001, 0.01, 0.05, 0.1]


def run_experiment_matrix(args):
    """Run the full experiment matrix: 4 Non-IID × 3 strategies + μ sensitivity."""

    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    ckpt_path = base_output / "checkpoint.json"
    completed = {}
    if args.resume and ckpt_path.exists():
        with open(ckpt_path) as f:
            completed = json.load(f)
        print(f"Resuming: {len(completed)} experiments completed")

    # Centralized baseline
    if "baseline" not in completed:
        print("\nEvaluating centralized baseline...")
        device_str = f"cuda:{args.device}" if args.device.isdigit() else args.device
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        model = create_model(NUM_CLASSES).to(device)

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        full_train = OrganoidPatchDataset(args.data, split="train", transform=val_transform)
        val_ds = OrganoidPatchDataset(args.data, split="val", transform=val_transform)
        train_loader = DataLoader(full_train, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

        model = train_client(model, train_loader, epochs=20, device=device, lr=1e-3)
        baseline_metrics = evaluate_model(model, val_loader, device)
        completed["baseline"] = baseline_metrics
        print(f"  Baseline: acc={baseline_metrics['accuracy']:.4f}")
        _save_checkpoint(ckpt_path, completed)
        del model
        gc.collect()

    # Phase 1: Core Matrix
    print(f"\n{'='*70}")
    print("PHASE 1: Core Experiment Matrix")
    print(f"  {len(NON_IID_LEVELS)} Non-IID levels × {len(STRATEGIES)} Strategies")
    print(f"{'='*70}")

    for niid_name, niid_cfg in NON_IID_LEVELS.items():
        exp_key = f"{niid_name}"
        if exp_key in completed:
            print(f"\n  {exp_key} — SKIPPED (completed)")
            continue

        print(f"\n{'='*70}")
        print(f"  {exp_key}: {niid_cfg['desc']}")
        print(f"{'='*70}")

        exp_output = base_output / "experiments" / exp_key

        # Create modified args for this experiment
        exp_args = argparse.Namespace(**vars(args))
        exp_args.output = str(exp_output)
        exp_args.dominant_ratio = niid_cfg["dominant_ratio"]
        exp_args.strategies = STRATEGIES

        result = run_fl_simulation(exp_args)
        completed[exp_key] = {
            "non_iid": niid_name,
            "dominant_ratio": niid_cfg["dominant_ratio"],
            "strategies": result["strategies"],
        }
        _save_checkpoint(ckpt_path, completed)

    # Phase 2: FedProx μ Sensitivity
    print(f"\n{'='*70}")
    print("PHASE 2: FedProx μ Sensitivity (Extreme Non-IID)")
    print(f"{'='*70}")

    for mu_val in MU_VALUES:
        mu_key = f"mu_{mu_val}"
        if mu_key in completed:
            print(f"  {mu_key} — SKIPPED (completed)")
            continue

        print(f"\n  FedProx μ={mu_val}...")
        mu_output = base_output / "experiments" / mu_key
        mu_args = argparse.Namespace(**vars(args))
        mu_args.output = str(mu_output)
        mu_args.dominant_ratio = 0.95
        mu_args.strategies = ["FedProx"]
        mu_args.mu = mu_val

        result = run_fl_simulation(mu_args)
        completed[mu_key] = {
            "mu": mu_val,
            "strategies": result["strategies"],
        }
        _save_checkpoint(ckpt_path, completed)

    # Generate summary
    _generate_summary(base_output, completed)
    print(f"\nAll experiments complete! Results in: {base_output}/")


def _save_checkpoint(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _generate_summary(base_output, completed):
    """Print and save summary table."""
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Non-IID':<12} {'Strategy':<10} {'Final Acc':>10} {'Best Acc':>10} {'Time':>8}")
    print("-" * 55)

    for key, val in completed.items():
        if key in ("baseline",) or key.startswith("mu_"):
            continue
        if not isinstance(val, dict) or "strategies" not in val:
            continue
        for strat, metrics in val["strategies"].items():
            print(f"{key:<12} {strat:<10} {metrics['final_acc']:>10.4f} "
                  f"{metrics['best_acc']:>10.4f} {metrics.get('total_time_sec',0):>7.0f}s")

    # Save as JSON
    with open(base_output / "experiment_summary.json", "w") as f:
        json.dump(completed, f, indent=2, ensure_ascii=False, default=str)


# ─── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organoid Classification FL Simulation")
    parser.add_argument("--data", default="./organoid_patches",
                        help="Path to organoid_patches directory")
    parser.add_argument("--output", default="./fl_classify_results",
                        help="Output directory")
    parser.add_argument("--rounds", type=int, default=10, help="FL rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cpu)")
    parser.add_argument("--strategies", nargs="+", default=["FedAvg", "FedProx", "EWA-v2"],
                        help="FL strategies")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx μ")
    parser.add_argument("--dominant-ratio", type=float, default=0.80,
                        help="Non-IID dominant class ratio")
    parser.add_argument("--matrix", action="store_true",
                        help="Run full experiment matrix (4 Non-IID × 3 strategies)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick test: 2 rounds, 2 epochs")

    args = parser.parse_args()
    if args.quick:
        args.rounds = 2
        args.epochs = 2

    if args.matrix:
        run_experiment_matrix(args)
    else:
        run_fl_simulation(args)
