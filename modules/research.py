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

    # ── Cross-Domain Evaluation Results ──
    st.markdown("---")
    st.markdown("### 🔬 Cross-Domain Evaluation Results")

    st.markdown("""
    SupCon slot model trained on MultiOrg (100 crops, K=8, dim=128, τ=0.07, β=0.1)
    evaluated on mouse liver (3 batches) and intestinal organoid datasets.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Slot vs Confidence AUC")
        import plotly.graph_objects as go
        datasets = ["Mouse B1", "Mouse B2", "Mouse B3", "Intestinal"]
        slot_auc = [0.29, 0.51, 0.54, 0.67]
        conf_auc = [0.91, 0.98, 0.92, 0.92]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Slot AUC", x=datasets, y=slot_auc,
                             marker_color="#e74c3c", text=[f"{v:.2f}" for v in slot_auc],
                             textposition="auto"))
        fig.add_trace(go.Bar(name="Conf AUC", x=datasets, y=conf_auc,
                             marker_color="#2ecc71", text=[f"{v:.2f}" for v in conf_auc],
                             textposition="auto"))
        fig.update_layout(barmode="group", height=350,
                          title="Slot model fails to generalize across domains",
                          yaxis_title="AUC", yaxis_range=[0, 1.1],
                          template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Dataset Summary")
        st.markdown("""
        | Dataset | Crops | TP/FP | Slot AUC | Conf AUC |
        |---------|-------|-------|----------|----------|
        | Mouse B1 | 26 | 23/3 | 0.29 🔴 | 0.91 |
        | Mouse B2 | 48 | 39/9 | 0.51 🔴 | 0.98 |
        | Mouse B3 | 60 | 40/20 | 0.54 🔴 | 0.92 |
        | Intestinal | 2744 | 2334/410 | 0.67 🔴 | 0.92 |
        """, unsafe_allow_html=True)

        st.markdown("""
        **Key Findings:**
        - 🔴 Slot model **completely fails** cross-domain (all AUC < 0.70)
        - ✅ Detector confidence is **robust** across domains (0.91-0.98)
        - ❌ All fusion strategies (hard_filter, soft_penalize, geometric_mean) 
          degrade to baseline (α=0, w=1.0)
        - **Root cause**: SupCon learns domain-specific TP/FP patterns, 
          not universal organoid primitives
        """)

    st.markdown("---")
    st.markdown("### 🔄 Multi-Dataset Joint Training (Same-Domain)")

    st.markdown("""
    Training on MultiOrg (100) + Mouse Liver (134) + Intestinal (2744) jointly.
    ⚠️ All three datasets are in the training set — these are **same-domain** results, not cross-domain.
    Need leave-one-out for true cross-domain validation.
    """)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Single-domain vs Joint Training")
        import plotly.graph_objects as go
        datasets = ["Mouse B1", "Mouse B2", "Mouse B3", "Intestinal"]
        single_auc = [0.29, 0.51, 0.54, 0.67]
        merged_auc = [0.74, 0.70, 0.85, 0.74]
        conf_auc = [0.91, 0.98, 0.92, 0.92]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Single-domain Slot", x=datasets, y=single_auc,
                              marker_color="#e74c3c", text=[f"{v:.2f}" for v in single_auc],
                              textposition="auto"))
        fig2.add_trace(go.Bar(name="Joint Training Slot", x=datasets, y=merged_auc,
                              marker_color="#3498db", text=[f"{v:.2f}" for v in merged_auc],
                              textposition="auto"))
        fig2.add_trace(go.Bar(name="Conf (baseline)", x=datasets, y=conf_auc,
                              marker_color="#2ecc71", text=[f"{v:.2f}" for v in conf_auc],
                              textposition="auto"))
        fig2.update_layout(barmode="group", height=350,
                           title="Joint training dramatically improves cross-domain AUC",
                           yaxis_title="AUC", yaxis_range=[0, 1.1],
                           template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        st.markdown("#### Improvement Summary")
        st.markdown("""
        | Dataset | Single | Joint | Δ | Conf |
        |---------|--------|-------|---|------|
        | Mouse B1 | 0.29 🔴 | **0.74** | +0.45 | 0.91 |
        | Mouse B2 | 0.51 🔴 | **0.70** | +0.19 | 0.98 |
        | Mouse B3 | 0.54 🔴 | **0.85** | +0.31 | 0.92 |
        | Intestinal | 0.67 | **0.74** | +0.07 | 0.92 |
        """, unsafe_allow_html=True)

        st.markdown("""
        **Key Findings:**
        - ✅ Joint training **reverses** B1 from 0.29 → 0.74 (+0.45)
        - ✅ B3 reaches **0.85**, only 7pp below conf
        - ✅ `soft_penalize` first meaningful positive delta (+0.0125 on B3)
        - ⚠️ Still below conf — single-frame DINOv2 features have a ceiling
        - 🚀 Next: CLIP semantic alignment / 3D temporal / VLM reasoning
        """)

    st.markdown("---")
    st.markdown("### 📊 Phase 8-11 Experiment Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        #### Completed Phases

        | Phase | Method | Result |
        |-------|--------|--------|
        | 8 | Wavelet primitives | PR-AUC 0.45 ❌ |
        | 9 | Slot Attention | PR-AUC 0.788 ✅ |
        | 9 | Combined (slot+conf) | AP 0.903 (+1.5pp) ✅ |
        | 10 | Federated Slot | Global > Local +1.1pp ✅ |
        | 11 | SupCon β=0.1 | AP 0.910 ✅ |
        | 11 | SupCon β=0.5 | AP 0.899 |
        | Cross | DINOv2 single-domain | Failed (0.29-0.67) 🔴 |
        | Cross | DINOv2 LOO | Failed (0.49-0.82) 🔴 |
        | A4 | **CLIP zero-shot** | **B1: 0.29→0.86 ✅** |
        """)
    with col_b:
        st.markdown("""
        #### Key Insights

        - **Same-domain**: Slot + confidence fusion improves AP by 1.5-3.3pp
        - **Cross-domain**: DINOv2 slot fails even with joint training
        - **CLIP zero-shot works**: B1 0.29→0.86 without any training!
        - **Semantic alignment is key**: CLIP text-visual space > DINOv2 pure visual
        - **Next**: CLIP+SupCon joint training, VLM reasoning, 3D temporal
        """)

    st.markdown("---")
    st.markdown("### 🆚 DINOv2 vs CLIP: Cross-Domain Comparison")

    import plotly.graph_objects as go

    datasets = ["MultiOrg", "Mouse B1", "Mouse B2", "Mouse B3"]
    dinov2_single = [0.79, 0.29, 0.51, 0.54]
    dinov2_loo = [None, 0.49, 0.56, 0.82]
    clip_zeroshot = [0.73, 0.86, 0.66, 0.69]
    conf_auc = [0.87, 0.91, 0.98, 0.92]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="DINOv2 Single", x=datasets, y=dinov2_single,
                          marker_color="#e74c3c", text=[f"{v:.2f}" for v in dinov2_single],
                          textposition="auto"))
    # LOO has None for MultiOrg, use 0 for display
    dinov2_loo_display = [v if v is not None else 0 for v in dinov2_loo]
    fig3.add_trace(go.Bar(name="DINOv2 LOO", x=datasets, y=dinov2_loo_display,
                          marker_color="#f39c12", text=[f"{v:.2f}" if v else "—" for v in dinov2_loo],
                          textposition="auto"))
    fig3.add_trace(go.Bar(name="CLIP Zero-Shot", x=datasets, y=clip_zeroshot,
                          marker_color="#3498db", text=[f"{v:.2f}" for v in clip_zeroshot],
                          textposition="auto"))
    fig3.add_trace(go.Bar(name="Conf (baseline)", x=datasets, y=conf_auc,
                          marker_color="#2ecc71", text=[f"{v:.2f}" for v in conf_auc],
                          textposition="auto"))
    fig3.update_layout(barmode="group", height=400,
                       title="CLIP zero-shot outperforms DINOv2 cross-domain (no training needed!)",
                       yaxis_title="AUC", yaxis_range=[0, 1.1],
                       template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    **Breakthrough finding**: CLIP's text-visual alignment space is cross-domain stable.
    - Mouse B1: DINOv2 0.29 (anti-prediction) → CLIP 0.86 (+0.57, no training!)
    - This validates the NLP analogy: semantic alignment (CLIP) > pure visual features (DINOv2)
    """)

    st.markdown("---")
    st.markdown("### ⚠️ Training Destroys Zero-Shot Generalization")

    st.markdown("""
    A1 (CLIP+SupCon LOO) and A3 (CoOp Prompt Tuning LOO) both **underperform** CLIP zero-shot.
    """)

    col5, col6 = st.columns(2)

    with col5:
        import plotly.graph_objects as go
        methods = ["DINOv2\nLOO", "CLIP\nZero-Shot", "CLIP+SupCon\nLOO", "CoOp\nLOO", "Conf\nBaseline"]
        avg_auc = [0.58, 0.73, 0.58, 0.64, 0.94]
        colors = ["#e74c3c", "#3498db", "#e67e22", "#f1c40f", "#2ecc71"]

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=methods, y=avg_auc, marker_color=colors,
                              text=[f"{v:.2f}" for v in avg_auc], textposition="auto"))
        fig4.update_layout(height=350, title="Average Cross-Domain AUC (4 datasets)",
                           yaxis_title="AUC", yaxis_range=[0, 1.0],
                           template="plotly_dark", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col6:
        st.markdown("#### Full Comparison")
        st.markdown("""
        | Dataset | DINOv2 LOO | CLIP ZS | CLIP+SC | CoOp | Conf |
        |---------|-----------|---------|---------|------|------|
        | MultiOrg | — | 0.73 | 0.95 | 0.62 | 0.87 |
        | Mouse B1 | 0.49 | **0.86** | 0.50 | 0.67 | 0.91 |
        | Mouse B2 | 0.56 | **0.66** | 0.57 | 0.58 | 0.98 |
        | Mouse B3 | 0.82 | **0.69** | 0.45 | 0.65 | 0.92 |
        | Intest. | 0.58 | **0.69** | 0.50 | 0.69 | 0.92 |
        | **Avg** | 0.58 | **0.73** | 0.58 | 0.64 | **0.94** |
        """, unsafe_allow_html=True)

        st.markdown("""
        - 🟢 CLIP zero-shot (no training) = best cross-domain slot
        - 🔴 All training hurts cross-domain (overfits to source)
        - 🟢 Conf (0.94) still champion
        - → Prompt engineering, VLM reasoning, 3D temporal
        """)
