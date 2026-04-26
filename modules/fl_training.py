# ── modules/fl_training.py ──
"""
FL Training Page
================
Configure and run federated learning with real-time visualization.
"""

import streamlit as st
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import (
    DEFAULT_ROUNDS, DEFAULT_CLIENTS, DEFAULT_LR,
    DEFAULT_BATCH_SIZE, DEFAULT_LOCAL_EPOCHS,
    ORGANOID_CLASSES, CLASS_INFO, REFERENCES, COLORS,
)
from utils.helpers import generate_synthetic_features, split_federated_data
from analysis.fl_engine import FLEngine
from visualization.charts import fl_convergence, accuracy_heatmap, confusion_matrix_plot


def render():
    st.markdown(
        '<div class="main-header"><h1>🔄 Federated Learning Training</h1>'
        '<p>Train organoid classification models across multiple hospitals without sharing data</p></div>',
        unsafe_allow_html=True,
    )

    # ── Configuration ──
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_rounds = st.slider("FL Rounds", 1, 50, DEFAULT_ROUNDS)
            n_clients = st.slider("Clients (Hospitals)", 2, 10, DEFAULT_CLIENTS)
        with col2:
            lr = st.slider("Learning Rate", 0.0001, 0.01, DEFAULT_LR, format="%.4f")
            local_epochs = st.slider("Local Epochs", 1, 10, DEFAULT_LOCAL_EPOCHS)
        with col3:
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            non_iid = st.slider("Non-IID Degree", 0.0, 1.0, 0.0, step=0.1)
            hidden_dim = st.selectbox("Hidden Dim", [64, 128, 256], index=1)

    # ── Data Source ──
    st.markdown("### 📊 Data")
    use_demo = st.checkbox("Use built-in demo data (recommended for quick start)", value=True)

    if use_demo:
        n_samples = st.slider("Demo samples", 300, 1500, 600, step=100)
        dim = 512

    # ── Run Training ──
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Generate or load data
        if use_demo:
            with st.spinner("Generating synthetic features..."):
                features, labels, class_names = generate_synthetic_features(
                    n_samples=n_samples, dim=dim, n_classes=len(ORGANOID_CLASSES),
                )
        else:
            st.error("Custom data loading not yet implemented. Use demo data.")
            return

        # Split across clients
        client_data = split_federated_data(features, labels, n_clients=n_clients, non_iid=non_iid)

        # Initialize engine
        engine = FLEngine(
            input_dim=dim,
            num_classes=len(class_names),
            hidden_dim=hidden_dim,
            lr=lr,
            local_epochs=local_epochs,
            batch_size=batch_size,
        )

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()

        history = []

        def on_progress(rnd, metrics):
            progress = (rnd + 1) / n_rounds
            progress_bar.progress(progress)
            status_text.text(
                f"Round {rnd + 1}/{n_rounds} | "
                f"Val Acc: {metrics['val_acc'] * 100:.2f}% | "
                f"Val Loss: {metrics['val_loss']:.4f}"
            )
            history.append(metrics)

        # Run
        with st.spinner("Training..."):
            t0 = time.time()
            engine.train(
                client_data,
                n_rounds=n_rounds,
                progress_callback=on_progress,
            )
            total_time = time.time() - t0

        progress_bar.progress(1.0)
        status_text.text(f"✅ Training complete in {total_time:.1f}s")

        # Update session state
        best = max(history, key=lambda h: h["val_acc"])
        st.session_state["best_accuracy"] = f"{best['val_acc'] * 100:.2f}%"
        st.session_state["total_rounds"] = n_rounds
        st.session_state["n_clients"] = n_clients
        st.session_state["fl_history"] = history
        st.session_state["fl_engine"] = engine
        st.session_state["fl_class_names"] = class_names
        st.session_state["fl_total_time"] = total_time

        st.success(
            f"**Best accuracy: {best['val_acc'] * 100:.2f}%** (Round {best['round'] + 1}) | "
            f"Total time: {total_time:.1f}s"
        )

    # ── Results Visualization ──
    if "fl_history" in st.session_state and st.session_state["fl_history"]:
        history = st.session_state["fl_history"]
        class_names = st.session_state.get("fl_class_names", ORGANOID_CLASSES)

        st.markdown("---")
        st.markdown("### 📈 Training Results")

        tab_conv, tab_heat, tab_cm = st.tabs(["Convergence", "Accuracy Heatmap", "Confusion Matrix"])

        with tab_conv:
            fig = fl_convergence(history)
            st.plotly_chart(fig, use_container_width=True)

        with tab_heat:
            fig = accuracy_heatmap(history, class_names)
            st.plotly_chart(fig, use_container_width=True)

        with tab_cm:
            fig = confusion_matrix_plot(history, class_names)
            st.plotly_chart(fig, use_container_width=True)

        # ── Methodology ──
        with st.expander("📖 Methodology"):
            st.markdown("""
            **FedAvg** (McMahan et al., 2017)

            1. **Local Training**: Each client $k$ trains on local data $\\mathcal{D}_k$ for $E$ epochs:
               $$w_k^{t+1} = w_k^t - \\eta \\nabla F_k(w_k^t)$$

            2. **Aggregation**: Server averages client updates:
               $$w^{t+1} = \\sum_{k=1}^{K} \\frac{|\\mathcal{D}_k|}{|\\mathcal{D}|} w_k^{t+1}$$

            3. **Repeat** for $T$ rounds.

            **Non-IID Setting**: When `non_iid > 0`, data is distributed unevenly across clients,
            simulating real-world hospital scenarios where patient populations differ.
            """)
            for key, ref in REFERENCES.items():
                st.markdown(f"- **{key}**: {ref}")
