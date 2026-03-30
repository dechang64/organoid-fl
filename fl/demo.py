"""
End-to-end demo: Organoid Federated Learning Platform

Demonstrates the complete pipeline:
1. Start Rust VectorDB
2. Generate synthetic organoid images
3. Extract ResNet18 features
4. Store features in VectorDB
5. Run federated learning (FedAvg)
6. Query VectorDB for similar organoid images
7. Classify new images with the trained model

Usage:
    python3 demo.py
"""

import subprocess
import sys
import os
import time
import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


# ============================================================
# Model
# ============================================================

class OrganoidClassifier(nn.Module):
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

def fedavg(client_params_list, client_weights):
    """FedAvg aggregation."""
    total = sum(client_weights)
    aggregated = []
    for layer_params in zip(*client_params_list):
        weighted = sum(w * p for w, p in zip(client_weights, layer_params))
        aggregated.append(weighted / total)
    return aggregated

def train_local(model, loader, epochs=3, lr=0.01):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_correct, total = 0, 0, 0
    for _ in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            _, pred = torch.max(out, 1)
            total_correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / total, total_correct / total

def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            _, pred = torch.max(out, 1)
            total_correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / total, total_correct / total


# ============================================================
# Pipeline Steps
# ============================================================

def start_vectordb(port=50061, dimension=512):
    import grpc
    import vectordb_pb2 as pb
    import vectordb_pb2_grpc as pb_grpc
    try:
        ch = grpc.insecure_channel(f'localhost:{port}')
        grpc.channel_ready_future(ch).result(timeout=2)
        ch.close()
        print("[1/7] VectorDB already running")
        return None
    except:
        pass
    print("[1/7] Starting VectorDB...")
    binary = Path(__file__).parent.parent / "target" / "release" / "organoid-vectordb"
    proc = subprocess.Popen(
        [str(binary), "--dimension", str(dimension), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3)
    try:
        ch = grpc.insecure_channel(f'localhost:{port}')
        grpc.channel_ready_future(ch).result(timeout=5)
        ch.close()
        print(f"       Running on port {port} (dim={dimension})")
        return proc
    except Exception as e:
        print(f"       Failed: {e}")
        return None


def generate_data(data_dir, n_per_class=200):
    from generate_data import generate_dataset
    data_dir = Path(data_dir)
    if data_dir.exists() and len(list(data_dir.rglob("*.png"))) >= n_per_class * 3:
        print(f"[2/7] Using existing dataset ({len(list(data_dir.rglob('*.png')))} images)")
        return
    print(f"[2/7] Generating {n_per_class * 3} synthetic organoid images...")
    generate_dataset(str(data_dir), n_per_class=n_per_class, img_size=128)


def extract_features(data_dir, features_path):
    features_path = Path(features_path)
    if features_path.exists():
        print(f"[3/7] Using cached features")
        return np.load(features_path, allow_pickle=True)
    from feature_extractor import ResNet18Extractor, extract_dataset
    print("[3/7] Extracting ResNet18 features...")
    model = ResNet18Extractor()
    extract_dataset(str(data_dir), model, str(features_path))
    return np.load(features_path, allow_pickle=True)


def store_in_vectordb(features_path, port=50061):
    import grpc
    import vectordb_pb2 as pb
    import vectordb_pb2_grpc as pb_grpc
    data = np.load(features_path, allow_pickle=True)
    features, labels, paths = data["features"], data["labels"], data["paths"]
    classes = list(data["classes"])
    print(f"[4/7] Storing {len(features)} vectors in VectorDB...")
    ch = grpc.insecure_channel(f'localhost:{port}')
    grpc.channel_ready_future(ch).result(timeout=5)
    stub = pb_grpc.VectorDBStub(ch)
    batch_size = 100
    total = 0
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        vectors = [pb.Vector(
            id=f"organoid_{i+j:04d}",
            values=feat.tolist(),
            metadata={"class": str(classes[label]), "path": str(path)},
        ) for j, (feat, label, path) in enumerate(zip(batch, labels[i:i+batch_size], paths[i:i+batch_size]))]
        resp = stub.Insert(pb.InsertRequest(vectors=vectors))
        total += resp.inserted
    stats = stub.Stats(pb.StatsRequest())
    print(f"       Stored {total} vectors (DB total: {stats.total_vectors})")
    ch.close()
    return classes


def run_federated_learning(features_path, rounds=10, n_clients=3):
    data = np.load(features_path, allow_pickle=True)
    features, labels = data["features"], data["labels"]
    classes = list(data["classes"])
    n_classes = len(classes)
    input_dim = features.shape[1]
    print(f"[5/7] Federated learning: {rounds} rounds, {n_clients} clients")
    print("-" * 50)
    # Non-IID split
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    splits = np.array_split(indices, n_clients)
    client_data = []
    for split in splits:
        X = torch.tensor(features[split], dtype=torch.float32)
        y = torch.tensor(labels[split], dtype=torch.long)
        n = int(0.8 * len(X))
        client_data.append((
            DataLoader(TensorDataset(X[:n], y[:n]), batch_size=32, shuffle=True),
            DataLoader(TensorDataset(X[n:], y[n:]), batch_size=32),
            len(split),
        ))
    global_params = get_params(OrganoidClassifier(input_dim, n_classes))
    for rnd in range(1, rounds + 1):
        client_params, client_weights = [], []
        for cid, (tr, va, ns) in enumerate(client_data):
            m = OrganoidClassifier(input_dim, n_classes)
            set_params(m, global_params)
            tr_loss, tr_acc = train_local(m, tr, epochs=3, lr=0.01)
            va_loss, va_acc = evaluate(m, va)
            print(f"  R{rnd} Client{cid}: train={tr_acc:.3f} val={va_acc:.3f}")
            client_params.append(get_params(m))
            client_weights.append(ns)
        global_params = fedavg(client_params, client_weights)
    # Final eval
    print("-" * 50)
    final_model = OrganoidClassifier(input_dim, n_classes)
    set_params(final_model, global_params)
    for cid, (_, va, _) in enumerate(client_data):
        _, acc = evaluate(final_model, va)
        print(f"  Final Client{cid}: accuracy={acc:.4f}")
    return final_model, classes


def demo_search(port=50061, k=5):
    import grpc
    import vectordb_pb2 as pb
    import vectordb_pb2_grpc as pb_grpc
    print(f"[6/7] Similarity search (k={k})...")
    ch = grpc.insecure_channel(f'localhost:{port}')
    grpc.channel_ready_future(ch).result(timeout=5)
    stub = pb_grpc.VectorDBStub(ch)
    np.random.seed(123)
    query = np.random.randn(512).astype(np.float32)
    t0 = time.time()
    results = stub.Search(pb.SearchRequest(query=query.tolist(), k=k))
    ms = (time.time() - t0) * 1000
    print(f"       Search: {ms:.1f}ms")
    for r in results.results:
        print(f"       {r.id}: dist={r.distance:.4f} class={r.metadata.get('class','?')}")
    ch.close()


def demo_classify(model, classes, data_dir):
    from feature_extractor import ResNet18Extractor, TRANSFORM
    print("[7/7] Classifying a random organoid image...")
    extractor = ResNet18Extractor()
    test_images = list(Path(data_dir).rglob("*.png"))
    img_path = test_images[np.random.randint(len(test_images))]
    true_class = img_path.parent.name
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        feat = extractor(tensor).squeeze(0)
    model.eval()
    with torch.no_grad():
        out = model(feat.unsqueeze(0))
        probs = torch.softmax(out, dim=1)
        idx = torch.argmax(probs, dim=1).item()
    pred = classes[idx]
    conf = probs[0][idx].item()
    mark = "✓" if pred == true_class else "✗"
    print(f"       File: {img_path.name}")
    print(f"       True: {true_class} | Predicted: {pred} ({conf:.2%}) {mark}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Organoid Federated Learning Platform")
    print("  End-to-End Demo")
    print("=" * 60)
    print()

    base = Path(__file__).parent
    data_dir = base / "data"
    feat_path = base / "features.npz"
    port = 50061

    proc = start_vectordb(port=port, dimension=512)
    try:
        generate_data(data_dir, n_per_class=200)
        extract_features(data_dir, feat_path)
        classes = store_in_vectordb(feat_path, port=port)
        model, fl_classes = run_federated_learning(feat_path, rounds=10, n_clients=3)
        demo_search(port=port, k=5)
        demo_classify(model, fl_classes, data_dir)
        print()
        print("=" * 60)
        print("  Demo Complete!")
        print("=" * 60)
    finally:
        if proc:
            proc.terminate()
            print("\nVectorDB stopped.")

if __name__ == "__main__":
    main()
