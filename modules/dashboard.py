# ── modules/dashboard.py ──
"""
Dashboard Page
==============
Platform overview: key metrics, system status, quick actions.
"""

import streamlit as st
import numpy as np
import time
from utils.constants import CLASS_INFO, REFERENCES, COLORS


def render():
    st.markdown(
        '<div class="main-header"><h1>🏠 Organoid-FL Dashboard</h1>'
        '<p>Privacy-preserving federated learning for medical organoid image analysis</p></div>',
        unsafe_allow_html=True,
    )

    # ── Key Metrics ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Model Accuracy",
            value=st.session_state.get("best_accuracy", "—"),
            delta="FedAvg" if st.session_state.get("best_accuracy") else None,
        )

    with col2:
        st.metric(
            label="FL Rounds",
            value=st.session_state.get("total_rounds", "—"),
            delta=f"{st.session_state.get('n_clients', 3)} clients",
        )

    with col3:
        st.metric(
            label="Vector DB",
            value=f"{st.session_state.get('vector_count', 0):,} vectors",
            delta=f"dim={st.session_state.get('vector_dim', 512)}",
        )

    with col4:
        audit_valid = st.session_state.get("audit_valid", True)
        st.metric(
            label="Audit Chain",
            value=f"{st.session_state.get('audit_blocks', 0)} blocks",
            delta="✅ Valid" if audit_valid else "⚠️ Invalid",
        )

    st.markdown("---")

    # ── Platform Architecture ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🏗️ Architecture")

        st.markdown("""
        ```
        ┌─────────────┐     gRPC      ┌──────────────┐
        │  Hospital A  │ ────────────▶ │              │
        │  Hospital B  │ ────────────▶ │  Rust Vector │
        │  Hospital C  │ ────────────▶ │  DB (HNSW)   │
        └─────────────┘               │              │
                                      │  + Audit     │
        ┌─────────────┐               │    Chain     │
        │  PyTorch    │ ◀──────────── │              │
        │  FedAvg     │   gradients   └──────────────┘
        └─────────────┘
        ```
        """)

        st.markdown("""
        **Tech Stack:**
        - **Backend:** Rust (HNSW index, gRPC, blockchain audit)
        - **ML:** PyTorch (ResNet-18 features, FedAvg aggregation)
        - **Protocol:** gRPC + Protocol Buffers
        - **Audit:** SHA-256 blockchain-style immutable log
        """)

    with col_right:
        st.subheader("📊 Key Results")

        results_data = {
            "Metric": ["Classification Accuracy", "Model", "Aggregation", "Vector Search", "Audit"],
            "Value": ["99.17%", "ResNet-18 (pretrained)", "FedAvg", "HNSW (kNN)", "SHA-256 Blockchain"],
        }
        st.dataframe(results_data, use_container_width=True, hide_index=True)

        st.subheader("🔬 Research Context")
        st.markdown("""
        Organoid-FL enables **collaborative AI model training** across
        hospitals/research labs **without sharing patient data**.

        - Each hospital trains locally on its own organoid images
        - Only model updates (gradients) are shared via gRPC
        - Server aggregates updates using FedAvg
        - All operations recorded on immutable audit chain
        """)

    st.markdown("---")

    # ── Quick Actions ──
    st.subheader("⚡ Quick Actions")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("🔄 Run FL Training", use_container_width=True, type="primary"):
            st.switch_page("pages/02_📊_FL_Training.py")

    with col_b:
        if st.button("🔍 Vector Search", use_container_width=True):
            st.switch_page("pages/03_🔍_Vector_Search.py")

    with col_c:
        if st.button("⛓️ View Audit Chain", use_container_width=True):
            st.switch_page("pages/04_⛓️_Audit_Chain.py")

    # ── References ──
    with st.expander("📚 References"):
        for key, ref in REFERENCES.items():
            st.markdown(f"**{key}**: {ref}")
