<div align="center">

# Organoid-FL

### Federated Learning Platform for Organoid Image Analysis

**Rust Vector DB + HNSW + PyTorch FedAvg + gRPC + Blockchain Audit + Web Dashboard**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Paper](docs/paper.md) · [Architecture](docs/architecture.md) · [API Reference](docs/api.md)

</div>

---

## 🎯 What is Organoid-FL?

Organoid-FL is an **end-to-end federated learning platform** designed for medical organoid image analysis. It enables multiple hospitals/research labs to collaboratively train AI models **without sharing patient data**.

### Key Results

| Metric | Value |
|--------|-------|
| Classification Accuracy | **99.17%** |
| Model | ResNet-18 (pretrained) |
| Aggregation | FedAvg |
| Vector Search | HNSW (kNN) |
| Audit | SHA-256 Blockchain |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Web Dashboard                         │
│                    (Axum + Static HTML)                       │
├──────────────────────────────────────────────────────────────┤
│                     gRPC Service Layer                        │
│              (Tonic + Protocol Buffers)                       │
├────────────┬──────────────┬──────────────┬───────────────────┤
│  Vector DB │  Federated   │  Blockchain  │   Image Store     │
│  (HNSW)    │  Training    │  Audit Chain │   (File System)   │
│            │  (FedAvg)    │  (SHA-256)   │                   │
├────────────┴──────────────┴──────────────┴───────────────────┤
│                    SQLite Metadata Store                      │
└──────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Rust Vector Database (HNSW)
- **Custom HNSW implementation** with const-generic dimensions
- Euclidean distance metric for organoid feature vectors
- Sub-millisecond kNN search
- Thread-safe with RwLock

#### 2. Federated Learning (PyTorch)
- **ResNet-18** backbone with transfer learning
- **FedAvg** aggregation across multiple clients
- gRPC communication between server and clients
- Support for heterogeneous data distributions

#### 3. Blockchain Audit Chain
- SHA-256 hash chain for immutable operation logging
- Every model update, data access, and prediction is recorded
- Chain verification API for compliance auditing

#### 4. Web Dashboard
- Real-time training progress visualization
- Model performance metrics
- Vector search interface
- Audit log browser

---

## 🚀 Quick Start

### Prerequisites
- Rust 1.70+
- Python 3.10+
- PyTorch 2.0+

### Build & Run

```bash
# Clone
git clone https://github.com/dechang64/organoid-fl.git
cd organoid-fl

# Build Rust server
cargo build --release

# Run server
./target/release/organoid-fl
# gRPC server ready on 0.0.0.0:50051
# Web dashboard ready on http://0.0.0.0:3000

# Train (another terminal)
cd python
pip install -r requirements.txt
python train.py --config config.yaml
```

---

## 📡 gRPC API

```protobuf
service OrganoidService {
  rpc UploadImage(UploadRequest) returns (UploadResponse);
  rpc SearchSimilar(SearchRequest) returns (SearchResponse);
  rpc StartTraining(TrainingRequest) returns (TrainingResponse);
  rpc GetModelStatus(StatusRequest) returns (ModelStatus);
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc GetAuditLog(LogRequest) returns (LogResponse);
}
```

### Example: Search Similar Organoid Images

```python
import grpc
import organoid_pb2 as pb
import organoid_pb2_grpc as rpc

channel = grpc.insecure_channel('localhost:50051')
stub = rpc.OrganoidServiceStub(channel)

# Search for similar organoid images
response = stub.SearchSimilar(pb.SearchRequest(
    image_id="ORG001",
    k=5
))

for result in response.results:
    print(f"ID: {result.image_id}, Distance: {result.distance:.4f}")
```

---

## 🧰 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Server | Rust (tonic, prost, axum) | High-performance gRPC + HTTP |
| Vector Search | HNSW (hnsw crate) | Approximate nearest neighbor |
| ML Training | Python (PyTorch, ResNet18) | Federated model training |
| Serialization | Protocol Buffers | gRPC message format |
| Database | SQLite (rusqlite) | Metadata storage |
| Audit | SHA-256 (sha2 crate) | Blockchain audit chain |
| Async Runtime | Tokio | Asynchronous I/O |

---

## 📁 Project Structure

```
organoid-fl/
├── src/
│   ├── main.rs              # Server entry point
│   ├── lib.rs               # Module declarations
│   ├── vector_db.rs         # HNSW vector index
│   ├── hnsw_index.rs        # HNSW implementation
│   ├── grpc_service.rs      # gRPC service handlers
│   ├── blockchain.rs        # Audit chain
│   ├── db.rs                # SQLite metadata
│   └── web_dashboard.rs     # Web UI
├── proto/
│   └── organoid.proto       # Protocol Buffer definitions
├── python/
│   ├── train.py             # Federated training script
│   ├── client.py            # gRPC client
│   ├── model.py             # ResNet-18 model
│   └── requirements.txt
├── data/                    # Data directory (gitignored)
├── Cargo.toml
└── README.md
```

---

## 🔬 Research Context

This project is part of research on **privacy-preserving medical AI** at XJTLU. Organoids — lab-grown mini-organs — are increasingly used in drug discovery and personalized medicine. Training accurate AI models requires large, diverse datasets, but medical data is subject to strict privacy regulations (HIPAA, GDPR).

Organoid-FL solves this by enabling **collaborative model training without data sharing**:

1. Each hospital trains locally on its own organoid images
2. Only model updates (gradients) are shared via gRPC
3. Server aggregates updates using FedAvg
4. All operations are recorded on an immutable audit chain

---

## 📄 License

MIT

---

<div align="center">

**Organoid-FL** — Privacy-preserving AI for medical organoid analysis

</div>
