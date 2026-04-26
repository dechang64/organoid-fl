<div align="center">

# Organoid-FL

### Federated Learning Platform for Organoid Image Analysis

**Rust HNSW VectorDB + PyTorch FedAvg + gRPC + Blockchain Audit + Streamlit Dashboard**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live Demo](#) · [Architecture](#-architecture) · [Research Modules](#-research-modules)

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

## 🚀 Quick Start (3 ways)

### Option A: Local (no Docker)
```bash
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

### Option B: Docker
```bash
cp .env.example .env
docker compose up -d
# Open http://localhost:8501
```

### Option C: Streamlit Community Cloud (Free, shareable URL)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect repo → deploy
4. Get `https://your-app.streamlit.app`

---

## 📊 Research Modules

| Module | What It Does |
|--------|-------------|
| 🏠 Dashboard | Platform overview, key metrics, quick actions |
| 📁 Data Explorer | Generate/view synthetic organoid data, distribution analysis |
| 🔄 FL Training | Interactive FedAvg training with real-time convergence plots |
| 🔍 Vector Search | HNSW approximate nearest neighbor search visualization |
| ⛓️ Audit Chain | Blockchain-style immutable operation log browser |
| 📈 Model Analysis | Confusion matrix, per-class metrics, client gap analysis |
| 🔬 Research | Methodology, paper references, theoretical foundation |

---

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Hospital A  │    │  Hospital B  │    │  Hospital C  │
│  (Local Data)│    │  (Local Data)│    │  (Local Data)│
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       │  gradients       │  gradients       │  gradients
       ▼                  ▼                  ▼
┌──────────────────────────────────────────────────┐
│              FL Aggregation Server               │
│              (FedAvg / FedProx)                  │
└──────────────────┬───────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
┌────────────┐ ┌────────┐ ┌──────────┐
│ Rust HNSW  │ │ Audit  │ │ Streamlit│
│ VectorDB   │ │ Chain  │ │ Dashboard│
│ (gRPC)     │ │(SHA256)│ │          │
└────────────┘ └────────┘ └──────────┘
```

---

## 📁 Project Structure

```
organoid-fl/
├── app.py                    # Streamlit entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker deployment
├── docker-compose.yml        # Service orchestration
├── .env.example              # Configuration template
├── src/                      # Rust VectorDB (optional backend)
│   ├── main.rs               # Server entry point
│   ├── db.rs                 # Vector database core
│   ├── hnsw_index.rs         # HNSW approximate search
│   ├── blockchain.rs         # SHA-256 audit chain
│   ├── grpc.rs               # gRPC service
│   ├── web.rs                # HTTP dashboard
│   └── lib.rs                # Module exports
├── modules/                  # Streamlit page modules
│   ├── dashboard.py          # Overview dashboard
│   ├── data_explorer.py      # Data management
│   ├── fl_training.py        # FL training UI
│   ├── vector_search.py      # Vector search UI
│   ├── audit_chain.py        # Audit chain browser
│   ├── model_analysis.py     # Model performance
│   └── research.py           # Methodology & references
├── analysis/                 # Core analysis logic
│   ├── fl_engine.py          # FedAvg engine
│   ├── vector_engine.py      # In-memory vector search
│   └── audit_engine.py       # Audit chain engine
├── visualization/            # Plotly charts
│   └── charts.py             # All chart functions
├── data/                     # Data layer
│   └── synthetic.py          # Synthetic data generator
├── utils/                    # Utilities
│   ├── constants.py          # Config & references
│   └── helpers.py            # Helper functions
├── tests/                    # Python tests
│   ├── test_fl_engine.py
│   ├── test_vector_engine.py
│   └── test_audit_engine.py
├── fl/                       # Legacy CLI demo (preserved)
├── proto/                    # Protocol Buffers
└── README.md
```

---

## 🔬 Research Context

This project is part of research on **privacy-preserving medical AI**. Organoids — lab-grown mini-organs — are increasingly used in drug discovery and personalized medicine. Training accurate AI models requires large, diverse datasets, but medical data is subject to strict privacy regulations (HIPAA, GDPR).

Organoid-FL solves this by enabling **collaborative model training without data sharing**:

1. Each hospital trains locally on its own organoid images
2. Only model updates (gradients) are shared via gRPC
3. Server aggregates updates using FedAvg
4. All operations are recorded on an immutable audit chain

---

## 🧪 Running Tests

```bash
# Python tests
pytest tests/ -v

# Rust tests (if Rust toolchain available)
cd src && cargo test
```

---

## 📄 License

MIT

---

<div align="center">

**Organoid-FL** — Privacy-preserving AI for medical organoid analysis

</div>
