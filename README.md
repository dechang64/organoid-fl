# Organoid Federated Learning Platform (类器官图像联邦学习平台)

A federated learning platform for organoid image analysis, featuring a Rust vector database and PyTorch-based federated training.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                 Python Application Layer              │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Synthetic   │  │ Feature      │  │ Federated    │  │
│  │ Data Gen    │  │ Extraction   │  │ Learning     │  │
│  │ (600 imgs)  │→ │ (ResNet18)   │→ │ (FedAvg)     │  │
│  └────────────┘  └──────┬───────┘  └──────────────┘  │
│                         │ 512-dim features            │
│                    gRPC (protobuf)                     │
├─────────────────────────┼────────────────────────────┤
│                         ▼                             │
│              Rust Vector Database                     │
│  ┌──────────────────────────────────────────────┐    │
│  │  gRPC Server (tonic 0.13)                    │    │
│  │  ┌──────────┐  ┌─────────────────────────┐   │    │
│  │  │ HashMap  │  │ kNN Search              │   │    │
│  │  │ Storage  │  │ (cosine distance)       │   │    │
│  │  └──────────┘  └─────────────────────────┘   │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

## Phase 1: Vector Database MVP ✅

- [x] Rust gRPC server (tonic + prost)
- [x] Vector storage with metadata
- [x] kNN search (cosine distance)
- [x] Insert / Search / Delete / Stats API
- [x] Python gRPC client
- [x] Unit tests (3/3)

## Phase 2: Federated Learning ✅

- [x] Synthetic organoid image generation (600 images, 3 classes)
- [x] ResNet18 feature extraction (512-dim)
- [x] FedAvg federated learning (3 clients, 10 rounds)
- [x] VectorDB integration (feature storage + similarity search)
- [x] End-to-end demo pipeline
- [x] **Final accuracy: 99.17%** (3-client avg)

### Classes

| Class | Description | Samples |
|-------|-------------|---------|
| `healthy` | Normal organoid morphology | 200 |
| `early_stage` | Early abnormal changes | 200 |
| `late_stage` | Advanced pathology | 200 |

## Quick Start

```bash
# Build Rust VectorDB
cd organoid-fl
cargo build --release

# Run full demo (generates data, extracts features, trains, searches)
cd fl
python3 demo.py

# Or run components individually:
python3 generate_data.py          # Generate synthetic images
python3 feature_extractor.py      # Extract ResNet18 features
python3 federated_learning.py     # Run federated learning
```

## API (gRPC)

| Method | Description |
|--------|-------------|
| `Insert(InsertRequest)` | Batch insert vectors with metadata |
| `Search(SearchRequest)` | k-nearest neighbor search |
| `Delete(DeleteRequest)` | Delete vectors by ID |
| `Stats(StatsRequest)` | Database statistics |

## Phase 3: Platform Integration (Planned)

- [ ] Web dashboard
- [ ] Blockchain audit logging
- [ ] HNSW index (scale to 100k+ vectors)
- [ ] Real organoid image dataset integration

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector DB | Rust, tonic 0.13, prost 0.13 |
| Feature Extraction | Python, PyTorch, ResNet18 |
| Federated Learning | PyTorch, FedAvg (pure implementation) |
| Communication | gRPC, protobuf |
| Data | Synthetic organoid images (PIL) |

## Project Structure

```
organoid-fl/
├── Cargo.toml              # Rust dependencies
├── build.rs                # Protobuf compilation
├── proto/
│   └── vectordb.proto      # gRPC service definition
├── src/
│   ├── main.rs             # Server entry point
│   ├── lib.rs              # Module declarations
│   ├── db.rs               # Vector storage & search
│   └── grpc.rs             # gRPC service implementation
├── python/                 # Generated Python gRPC stubs
├── fl/
│   ├── demo.py             # End-to-end demo
│   ├── generate_data.py    # Synthetic organoid image generator
│   ├── feature_extractor.py # ResNet18 feature extraction
│   ├── federated_learning.py # FedAvg training
│   ├── data/               # Generated images (600)
│   ├── features.npz        # Cached features
│   ├── model_final.pt      # Trained model
│   └── training_history.npz # Training metrics
└── test_client.py          # Python gRPC integration test
```
