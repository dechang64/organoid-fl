<div align="center">

# Organoid-FL

### Federated Learning Platform for Organoid Image Analysis

**Rust HNSW VectorDB + YOLOv11 + DINOv2 + SAM2 + PyTorch FedAvg + Streamlit Dashboard**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-9b59b6?logo=ultralytics)](https://docs.ultralytics.com/)
[![DINOv2](https://img.shields.io/badge/DINOv2-Meta-blueviolet)](https://github.com/facebookresearch/dinov2)
[![SAM2](https://img.shields.io/badge/SAM2-Meta-green)](https://github.com/facebookresearch/sam2)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🎯 What is Organoid-FL?

Organoid-FL is an **end-to-end federated learning platform** for medical organoid image analysis. It enables multiple hospitals/research labs to collaboratively train AI models **without sharing patient data**.

### Multi-Task Vision Pipeline

| Model | Task | Output |
|-------|------|--------|
| **YOLOv11** | Detection | Bounding boxes + class + confidence |
| **DINOv2** | Feature Extraction | 768-dim self-supervised embeddings |
| **SAM2** | Segmentation | Pixel-level masks + morphology metrics |
| **FedAvg** | Aggregation | Privacy-preserving model updates |

### Key Results

| Metric | Intestinal Organoid | MultiOrg | Mouse Liver |
|--------|---------------------|----------|-------------|
| Classification Accuracy | **99.17%** | — | — |
| Detection mAP50 | — | **0.885** (v12s, 1280px) | 0.96 (RF-DETR) |
| Detection mAP50-95 | — | **0.624** (v12s, 1280px) | — |
| Feature Dim | **768** (DINOv2 base) | — | — |
| Detection Model | YOLOv11 (3.2M–20.1M params) | YOLOv12s (9.2M params) | RF-DETR-base |
| Segmentation | SAM2 (pixel-level) | — | — |
| Vector Search | HNSW (kNN) | — | — |
| Audit | SHA-256 Blockchain | — | — |

### 🔬 SAM2 Self-Distillation Cross-Domain (2026-07-16)

**Method**: Use SAM2 zero-shot mask as independent signal to train classifier head,
avoiding circular logic when RF-DETR conf fails cross-domain.

**Three-tier evidence chain**:

| Experiment | Samples | Zero-shot AUC | Distilled AUC | Improvement |
|---|---|---|---|---|
| Simulation (Intestinal 500+200) | 700 | 0.6616 | **0.9437** | **+28.22%** |
| Real cross-domain 50+30 | 10578 | 0.7480 | **0.8857** | **+13.77%** |
| Real cross-domain 200+100 | 40503 | 0.7679 | **0.8589** | **+9.10%** |
| MPM pilot (34 patches) | 3131 | 0.9779 | 1.0000 | +2.21% |

**Adaptive distillation strategy** (paper innovation):

| Zero-shot AUC | Strategy | Formula | Improvement |
|---|---|---|---|
| **< 0.70** | Classifier alone | `distilled = classifier_prob` | **+31.76%** |
| 0.70 - 0.85 | Distilled | `conf × classifier_prob` | **+9.10%** |
| > 0.85 | Zero-shot | `distilled = conf` | maintain |

**Key insight**: Traditional `conf × classifier_prob` is suboptimal when RF-DETR conf
fails cross-domain. Classifier alone wins in 2/3 experiments.

### 🔢 Organoid Counting Application (2026-09-15)

**Enterprise requirement**: F1 ≥ 0.90, Precision ≥ 0.90, Recall ≥ 0.90, CV ≤ 1%, speed ≤ 1 min/image.

**Dataset**: Intestinal Organoid (Zenodo 6768583) — 840 images, 23,065 annotations, 4 classes.

**Counting pipeline**: RF-DETR detection → SAM2 mask → morphology features → TP/FP classifier → counting.

| Method | F1 | Precision | Recall | MAE | R² | Over-count | Meets F1≥0.90? |
|--------|-----|-----------|--------|-----|-----|-----------|----------------|
| Baseline (no filter) | 0.000 | 0.286 | 0.934 | 43.87 | -4.58 | +149.3% | ✗ |
| Confidence filter | 0.945 | 0.929 | 0.961 | 1.54 | 0.991 | +3.4% | ✓ |
| Morphology (GB) | 0.944 | 0.932 | 0.957 | 1.58 | 0.991 | +2.8% | ✓ |
| **+ SAM2 self-distillation** | **1.000** | **1.000** | **0.999** | **0.04** | **0.9999** | **-0.0%** | **✓** |

**Key finding**: SAM2 mask quality provides a near-perfect TP/FP signal (AUC=1.000),
enabling F1=1.000 and counting R²=0.9999 — fully meeting enterprise requirements.

**Enterprise compliance** (9 requirements):

| # | Requirement | Status | Details |
|---|-------------|--------|---------|
| 1 | Multi-format (jpg/tiff/bmp) | ✅ | PIL/cv2 support |
| 2 | Multi-class (4 types) | ✅ | 4-class YOLO detection |
| 3 | Normal/apoptosis distinction | ⚠️ | Future work |
| 4 | Overlap separation | ✅ | SAM2 instance mask |
| 5 | Morphology metrics | ✅ | Area/perimeter/circularity/solidity/AR |
| 6 | API/LIMS integration | ✅ | FastAPI + Streamlit |
| 7 | F1 ≥ 0.90 | ✅ | F1=1.000 |
| 8 | CV ≤ 1% | ✅ | Deterministic inference |
| 9 | Speed ≤ 1 min | ✅ | ~31s/image |

📄 Full report: `docs/organoid_counting_research_report.md`
📊 Results: `results/counting_intestinal/`

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

### Option C: Streamlit Community Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect repo → deploy
4. Get `https://your-app.streamlit.app`

---

## 📊 Platform Modules (11 pages)

| Page | Description |
|------|-------------|
| 🏠 Dashboard | Platform overview, key metrics, quick actions |
| 📁 Data Explorer | Generate/upload organoid images, data distribution |
| 🔄 FL Training | Interactive FedAvg training with real-time charts |
| 🎯 Detection | YOLOv11 organoid detection, counting, classification |
| ✂️ Segmentation | SAM2 pixel-level masks, morphology analysis |
| 🌌 Feature Space | DINOv2 t-SNE/UMAP feature visualization |
| 🧩 Multi-Task FL | Detection + classification + segmentation jointly |
| 🔍 Vector Search | HNSW approximate nearest neighbor search |
| ⛓️ Audit Chain | Blockchain-style immutable operation log |
| 📈 Model Analysis | Confusion matrix, per-client gap analysis |
| 🔬 Research | Methodology, paper references, architecture |

---

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Hospital A  │    │  Hospital B  │    │  Hospital C  │
│  (Local Data)│    │  (Local Data)│    │  (Local Data)│
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       │  YOLO + DINOv2 + SAM2 weights      │
       │  (no images shared)                 │
       └──────────────┬──────────────────────┘
                      │
              ┌───────┴───────┐
              │  FedAvg Server │
              │  Aggregation   │
              └───────┬───────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ YOLOv11 │ │ DINOv2  │ │  SAM2   │
    │ Detect  │ │ Classify│ │ Segment │
    └─────────┘ └─────────┘ └─────────┘
```

---

## 📁 Project Structure

```
organoid-fl/
├── app.py                    # Streamlit entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker deployment
├── docker-compose.yml        # Container orchestration
├── .env.example              # Configuration template
├── src/                      # Rust VectorDB (optional backend)
│   ├── main.rs               # gRPC + HTTP server
│   ├── db.rs                 # VectorDB core
│   ├── hnsw_index.rs         # HNSW ANN search
│   ├── blockchain.rs         # SHA-256 audit chain
│   ├── grpc.rs               # gRPC service
│   └── web.rs                # REST API + dashboard
├── modules/                  # Streamlit page modules
│   ├── dashboard.py          # 🏠 Overview
│   ├── data_explorer.py      # 📁 Data management
│   ├── fl_training.py        # 🔄 FedAvg training
│   ├── detection.py          # 🎯 YOLOv11 detection
│   ├── segmentation.py       # ✂️ SAM2 segmentation
│   ├── feature_space.py      # 🌌 DINOv2 visualization
│   ├── multi_task.py         # 🧩 Multi-task FL
│   ├── vector_search.py      # 🔍 HNSW search
│   ├── audit_chain.py        # ⛓️ Audit browser
│   ├── model_analysis.py     # 📈 Performance analysis
│   └── research.py           # 🔬 Methodology
├── analysis/                 # Core analysis engines
│   ├── fl_engine.py          # FedAvg engine
│   ├── multi_task_fl.py      # Multi-task FL engine
│   ├── detector.py           # YOLOv11 detector
│   ├── feature_extractor_v2.py  # DINOv2 + ResNet18
│   ├── segmentor.py          # SAM2 segmentor
│   ├── vector_engine.py      # In-memory vector search
│   └── audit_engine.py       # SHA-256 audit chain
├── visualization/            # Plotly charts
│   └── charts.py             # All chart functions
├── data/                     # Data layer
│   └── synthetic.py          # Synthetic data generator
├── utils/                    # Utilities
│   ├── constants.py          # Config & references
│   └── helpers.py            # Helper functions
├── tests/                    # Python tests (24 passed)
│   ├── test_fl_engine.py
│   ├── test_vector_engine.py
│   └── test_audit_engine.py
├── fl/                       # Legacy CLI demo (preserved)
├── proto/                    # Protocol Buffers
└── README.md
```

---

## 🔬 Research Context

This project is part of research on **privacy-preserving medical AI** at XJTLU. Organoids — lab-grown mini-organs — are increasingly used in drug discovery and personalized medicine.

### Multi-Task Vision Pipeline

1. **YOLOv11** detects individual organoids (bounding boxes + classification)
2. **DINOv2** extracts 768-dim self-supervised features (no labels needed)
3. **SAM2** segments organoids at pixel level (morphology metrics)
4. **FedAvg** aggregates model updates across hospitals (no data sharing)
5. **HNSW** enables fast similarity search over feature embeddings
6. **Blockchain** provides tamper-evident audit trail for compliance

### Key References

- McMahan et al. (2017) — FedAvg: Communication-Efficient Learning
- Li et al. (2020) — FedProx: Federated Optimization in Heterogeneous Networks
- Oquab et al. (2023) — DINOv2: Learning Robust Visual Features
- Jocher et al. (2025) — YOLOv11: Ultralytics
- Ravi et al. (2024) — SAM 2: Segment Anything in Images and Videos

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

**Organoid-FL** — Privacy-preserving multi-task AI for medical organoid analysis

</div>
