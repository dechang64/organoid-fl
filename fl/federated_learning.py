"""
Federated learning simulation for organoid image classification.

Pure PyTorch implementation of FedAvg — no Ray, no Flower server threading issues.
Each client trains locally, server aggregates via FedAvg.

Usage:
    python3 federated_learning.py --rounds 10 --clients 3
"""

import sys
import os
import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


# ============================================================
# Model
# ============================================================

class OrganoidClassifier(nn.Module):
    """MLP classifier for organoid stage classification."""
    
    def __init__(self, input_dim=512, num_classes=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        return self.net(x)


def get_params(model):
    return [p.data.cpu().numpy() for p in model.parameters()]


def set_params(model, params):
    state = model.state_dict()
    for (name, _), p in zip(state.items(), params):
        state[name] = torch.tensor(p)
    model.load_state_dict(state)


def fedavg_aggregate(client_params_list, client_weights):
    """FedAvg: weighted average of client parameters."""
    total = sum(client_weights)
    aggregated = []
    for layer_idx in range(len(client_params_list[0])):
        weighted = sum(
            params[layer_idx] * (w / total)
            for params, w in zip(client_params_list, client_weights)
        )
        aggregated.append(weighted)
    return aggregated


# ============================================================
# Client training
# ============================================================

def train_client(model, train_loader, val_loader, local_epochs, lr):
    """Train a client locally and return params + metrics."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model.train()
    for epoch in range(local_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    
    train_acc = correct / total
    train_loss = epoch_loss / len(train_loader)
    
    # Evaluate
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            out = model(X)
            loss = criterion(out, y)
            val_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    
    val_acc = correct / total
    val_loss = val_loss / len(val_loader)
    
    return get_params(model), {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


# ============================================================
# VectorDB Integration
# ============================================================

def store_features_in_vectordb(features, labels, paths, classes, port=50061):
    """Store features in Rust VectorDB via gRPC."""
    try:
        import grpc
        import vectordb_pb2 as pb
        import vectordb_pb2_grpc as pb_grpc
        
        channel = grpc.insecure_channel(f'localhost:{port}')
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = pb_grpc.VectorDBStub(channel)
        
        batch_size = 100
        total = 0
        for i in range(0, len(features), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(features))):
                batch.append(pb.Vector(
                    id=f"organoid_{j:04d}",
                    values=features[j].tolist(),
                    metadata={
                        "class": str(classes[labels[j]]),
                        "path": str(paths[j]),
                    }
                ))
            resp = stub.Insert(pb.InsertRequest(vectors=batch))
            total += resp.inserted
        
        stats = stub.Stats(pb.StatsRequest())
        print(f"  VectorDB: {stats.total_vectors} vectors (dim={stats.dimension})")
        
        # Test search
        query = features[0].tolist()
        results = stub.Search(pb.SearchRequest(query=query, k=3))
        print(f"  Search test (query=organoid_0000, k=3):")
        for r in results.results:
            print(f"    {r.id}: dist={r.distance:.4f}, class={r.metadata.get('class', '?')}")
        
        channel.close()
        return True
    except Exception as e:
        print(f"  VectorDB not available (skipping): {e}")
        return False


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Organoid Federated Learning")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vectordb-port", type=int, default=50061)
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Organoid Federated Learning Simulation")
    print("=" * 60)
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.clients}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    # Load features
    feat_path = Path(__file__).parent / "features.npz"
    if not feat_path.exists():
        print("Features not found. Run feature_extractor.py first.")
        return
    
    data = np.load(feat_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    paths = list(data["paths"])
    classes = list(data["classes"])
    
    print(f"Dataset: {len(features)} samples, {len(classes)} classes: {classes}")
    print(f"Feature dim: {features.shape[1]}")
    print()
    
    # Store in VectorDB
    print("Storing features in VectorDB...")
    store_features_in_vectordb(features, labels, paths, classes, args.vectordb_port)
    print()
    
    # Partition data (non-IID: each client gets different class distribution)
    n = len(features)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    partitions = []
    part_size = n // args.clients
    for i in range(args.clients):
        start = i * part_size
        end = start + part_size if i < args.clients - 1 else n
        partitions.append(indices[start:end])
    
    # Create client data
    client_data = []
    for cid, part in enumerate(partitions):
        X = torch.tensor(features[part], dtype=torch.float32)
        y = torch.tensor(labels[part], dtype=torch.long)
        n_train = int(0.8 * len(X))
        train_loader = DataLoader(TensorDataset(X[:n_train], y[:n_train]),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X[n_train:], y[n_train:]),
                                batch_size=args.batch_size)
        client_data.append((train_loader, val_loader, len(X[:n_train])))
        
        dist = {}
        for lbl in labels[part]:
            dist[str(classes[lbl])] = dist.get(str(classes[lbl]), 0) + 1
        print(f"  Client {cid}: train={len(X[:n_train])}, val={len(X[n_train:])}, dist={dist}")
    
    print()
    print("Starting federated learning...")
    print("-" * 60)
    
    # Initialize global model
    global_model = OrganoidClassifier(input_dim=features.shape[1], num_classes=len(classes))
    global_params = get_params(global_model)
    
    history = []
    
    for rnd in range(1, args.rounds + 1):
        print(f"\n--- Round {rnd}/{args.rounds} ---")
        
        # Each client trains locally
        client_params_list = []
        client_weights = []
        client_metrics = []
        
        for cid, (train_loader, val_loader, n_train) in enumerate(client_data):
            local_model = OrganoidClassifier(input_dim=features.shape[1], num_classes=len(classes))
            set_params(local_model, global_params)
            
            params, metrics = train_client(local_model, train_loader, val_loader,
                                           args.local_epochs, args.lr)
            client_params_list.append(params)
            client_weights.append(n_train)
            client_metrics.append(metrics)
            
            print(f"  Client {cid}: "
                  f"train_loss={metrics['train_loss']:.4f}, train_acc={metrics['train_acc']:.4f} | "
                  f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc']:.4f}")
        
        # FedAvg aggregation
        global_params = fedavg_aggregate(client_params_list, client_weights)
        
        # Round summary
        avg_val_acc = np.mean([m["val_acc"] for m in client_metrics])
        avg_val_loss = np.mean([m["val_loss"] for m in client_metrics])
        print(f"  >>> Round avg: val_loss={avg_val_loss:.4f}, val_acc={avg_val_acc:.4f}")
        
        history.append({
            "round": rnd,
            "avg_val_acc": avg_val_acc,
            "avg_val_loss": avg_val_loss,
            "client_metrics": client_metrics,
        })
    
    # Final evaluation
    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    
    print("\nFinal model evaluation per client:")
    for cid, (train_loader, val_loader, _) in enumerate(client_data):
        final_model = OrganoidClassifier(input_dim=features.shape[1], num_classes=len(classes))
        set_params(final_model, global_params)
        final_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                out = final_model(X)
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        print(f"  Client {cid}: accuracy = {correct/total:.4f} ({correct}/{total})")
    
    # Save model
    final_model = OrganoidClassifier(input_dim=features.shape[1], num_classes=len(classes))
    set_params(final_model, global_params)
    model_path = Path(__file__).parent / "model_final.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save history
    hist_path = Path(__file__).parent / "training_history.npz"
    np.savez(hist_path,
             rounds=[h["round"] for h in history],
             avg_val_acc=[h["avg_val_acc"] for h in history],
             avg_val_loss=[h["avg_val_loss"] for h in history])
    print(f"History saved: {hist_path}")
    
    # Best round
    best = max(history, key=lambda h: h["avg_val_acc"])
    print(f"\nBest round: {best['round']} (val_acc={best['avg_val_acc']:.4f})")


if __name__ == "__main__":
    main()
