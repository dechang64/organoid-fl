# ── modules/research.py ──
"""
Research Page
=============
Methodology, paper references, and theoretical foundation.
"""

import streamlit as st
from utils.constants import REFERENCES


def render():
    st.markdown(
        '<div class="main-header"><h1>🔬 Research</h1>'
        '<p>Theoretical foundation, methodology, and paper references</p></div>',
        unsafe_allow_html=True,
    )

    # ── Overview ──
    st.markdown("""
    ### Organoid-FL: Privacy-Preserving Medical AI

    Organoid-FL is an **end-to-end federated learning platform** for medical organoid
    image analysis. Organoids — lab-grown mini-organs — are increasingly used in drug
    discovery and personalized medicine. Training accurate AI models requires large,
    diverse datasets, but medical data is subject to strict privacy regulations
    (HIPAA, GDPR).

    This platform solves the tension by enabling **collaborative model training
    without data sharing**.
    """)

    st.markdown("---")

    # ── Architecture ──
    st.markdown("### 🏗️ System Architecture")

    st.markdown("""
    ```
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Hospital A  │    │  Hospital B  │    │  Hospital C  │
    │  (Local Data)│    │  (Local Data)│    │  (Local Data)│
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           │                  │                  │
           │  Model Updates   │  Model Updates   │  Model Updates
           │  (gradients)     │  (gradients)     │  (gradients)
           ▼                  ▼                  ▼
    ┌──────────────────────────────────────────────────────┐
    │              FL Aggregation Server (FedAvg)           │
    │         ┌─────────────────────────────────┐          │
    │         │  Rust HNSW VectorDB + gRPC      │          │
    │         │  SHA-256 Blockchain Audit Chain  │          │
    │         └─────────────────────────────────┘          │
    └──────────────────────────────────────────────────────┘
    ```
    """)

    # ── Key Components ──
    st.markdown("### 🔧 Key Components")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🧠 Feature Extraction
        - **ResNet-18** pretrained on ImageNet
        - 512-dimensional feature vectors
        - Frozen backbone, no fine-tuning needed
        """)

    with col2:
        st.markdown("""
        #### 🔗 Federated Learning
        - **FedAvg** aggregation (McMahan 2017)
        - Configurable Non-IID splits
        - Per-client local training
        """)

    with col3:
        st.markdown("""
        #### 🔍 Vector Search
        - **HNSW** approximate kNN (Malkov 2018)
        - Rust implementation for performance
        - gRPC interface for Python clients
        """)

    st.markdown("---")

    # ── Papers ──
    st.markdown("### 📚 Key References")

    papers = [
        ("McMahan et al., 2017",
         "Communication-Efficient Learning of Deep Networks from Decentralized Data",
         "AISTATS 2017 — Introduced FedAvg, the foundational FL aggregation algorithm."),
        ("Li et al., 2020",
         "Federated Optimization in Heterogeneous Networks",
         "MLSys 2020 — FedProx: handles statistical heterogeneity across clients."),
        ("Malkov & Yashunin, 2018",
         "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW",
         "IEEE TPAMI — HNSW graph-based ANN search with state-of-the-art recall/speed."),
        ("Kairouz et al., 2021",
         "Advances and Open Problems in Federated Learning",
         "Foundations and Trends — Comprehensive survey of FL methods and challenges."),
        ("Bonawitz et al., 2019",
         "Towards Federated Learning at Scale: A System Design",
         "MLSys 2019 — Production FL system design at Google scale."),
    ]

    for title, subtitle, desc in papers:
        with st.expander(f"📄 {title}"):
            st.markdown(f"**{subtitle}**\n\n{desc}")

    # ── Future Work ──
    st.markdown("---")
    st.markdown("### 🚀 Future Directions")

    st.markdown("""
    - **FedProx / SCAFFOLD**: Handling heterogeneous data distributions
    - **Differential Privacy**: Formal privacy guarantees (ε-delta)
    - **Secure Aggregation**: Cryptographic protection of model updates
    - **Asynchronous FL**: Remove synchronization barrier between clients
    - **Multi-modal Fusion**: Combine imaging + genomic + clinical data
    """)
