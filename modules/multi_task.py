# ── modules/multi_task.py ──
"""
Multi-Task FL Page
==================
Train detection + classification + segmentation simultaneously.
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
from utils.helpers import generate_synthetic_features
from analysis.multi_task_fl import MultiTaskFLEngine
from visualization.charts import fl_convergence


def render():
    st.markdown(
        '<div class="main-header"><h1>🧩 Multi-Task Federated Learning</h1>'
        '<p>Train detection + classification + segmentation across hospitals simultaneously</p></div>',
        unsafe_allow_html=True,
    )

    # ── Architecture Diagram ──
    st.markdown("""
    ### System Architecture

    ```
    Hospital A ──┐                    ┌── YOLOv11 (detection)
                 │    FedAvg          ├── DINOv2 + Classifier
    Hospital B ──┼──────────────────→ │
                 │    Aggregation     └── SAM2 (segmentation)
    Hospital C ──┘
    ```

    Each hospital trains locally. Only **model weights** are shared — **no patient images**.
    """)

    st.markdown("---")

    # ── Configuration ──
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_rounds = st.slider("FL Rounds", 1, 50, DEFAULT_ROUNDS)
            n_clients = st.slider("Clients", 2, 10, DEFAULT_CLIENTS)
        with col2:
            feature_dim = st.selectbox("Feature Dim", [384, 512, 768, 1024], index=2)
            lr = st.slider("Learning Rate", 0.0001, 0.01, DEFAULT_LR, format="%.4f")
        with col3:
            local_epochs = st.slider("Local Epochs", 1, 10, DEFAULT_LOCAL_EPOCHS)
            enable_detector = st.checkbox("YOLO Detection", value=True)
            enable_segmentor = st.checkbox("SAM2 Segmentation", value=False)

    # ── Run Training ──
    if st.button("🚀 Start Multi-Task FL Training", type="primary"):
        with st.status("Training...", expanded=True) as status:
            # Generate synthetic features
            st.write("📦 Generating synthetic organoid features...")
            features, labels, class_names = generate_synthetic_features(
                n_samples=600, dim=feature_dim
            )

            # Initialize engine
            st.write("🔧 Initializing multi-task FL engine...")
            engine = MultiTaskFLEngine(
                input_dim=feature_dim,
                num_classes=len(class_names),
                lr=lr,
                local_epochs=local_epochs,
                batch_size=DEFAULT_BATCH_SIZE,
            )

            # Progress callback
            progress_container = st.empty()

            def on_progress(rnd, metrics):
                progress_container.markdown(
                    f"**Round {rnd + 1}/{n_rounds}** — "
                    f"Val Acc: {metrics['val_acc']:.1%} | "
                    f"Val Loss: {metrics['val_loss']:.4f} | "
                    f"Time: {metrics['elapsed']:.1f}s"
                )

            # Train
            st.write("🔄 Running federated training...")
            t0 = time.time()
            history = engine.run(
                features, labels,
                n_clients=n_clients,
                rounds=n_rounds,
                progress_callback=on_progress,
            )
            total_time = time.time() - t0

            # Store results
            st.session_state["mt_history"] = history
            st.session_state["mt_class_names"] = class_names
            st.session_state["mt_total_time"] = total_time
            st.session_state["mt_n_clients"] = n_clients

            status.update(label="Training Complete!", state="complete")

    # ── Results ──
    if "mt_history" in st.session_state:
        history = st.session_state["mt_history"]
        class_names = st.session_state["mt_class_names"]
        total_time = st.session_state["mt_total_time"]

        st.markdown("---")
        st.markdown("### 📈 Training Results")

        best = max(history, key=lambda h: h["val_acc"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Accuracy", f"{best['val_acc'] * 100:.2f}%", f"Round {best['round'] + 1}")
        col2.metric("Final Accuracy", f"{history[-1]['val_acc'] * 100:.2f}%")
        col3.metric("Final Loss", f"{history[-1]['val_loss']:.4f}")
        col4.metric("Total Time", f"{total_time:.1f}s")

        # Convergence chart
        fig = fl_convergence(history)
        st.plotly_chart(fig, use_container_width=True)

        # Per-client breakdown
        if history and "client_metrics" in history[-1]:
            import plotly.graph_objects as go
            last = history[-1]
            clients = [f"Client {m['client']}" for m in last["client_metrics"]]
            accs = [m["train_acc"] * 100 for m in last["client_metrics"]]

            fig = go.Figure(go.Bar(
                x=clients, y=accs,
                marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                              "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4",
                              "#ec4899", "#14b8a6"][:len(clients)],
                text=[f"{a:.1f}%" for a in accs],
                textposition="auto",
            ))
            fig.update_layout(
                title="Per-Client Training Accuracy",
                template="plotly_white",
                yaxis_title="Accuracy (%)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Methodology ──
    with st.expander("📖 Multi-Task FL Methodology"):
        st.markdown("""
        **Multi-Task Federated Learning**

        Three models trained in parallel across hospitals:

        | Model | Task | Shared Weights | Frozen |
        |-------|------|---------------|--------|
        | YOLOv11 | Detection (bbox + class) | Detection head + FPN | Backbone |
        | DINOv2 + Linear | Classification | Linear head only | Full ViT |
        | SAM2 | Segmentation | Prompt encoder | Image encoder + mask decoder |

        **FedAvg Aggregation:**
        $$w^{t+1} = \\sum_{k=1}^{K} \\frac{|\\mathcal{D}_k|}{|\\mathcal{D}|} w_k^{t+1}$$

        Applied independently to each model's trainable parameters.

        **Communication Cost:**
        - YOLOv11n detection head: ~2M params (~8MB per round)
        - DINOv2 classifier head: ~2.3K params (~9KB per round)
        - SAM2 prompt encoder: ~4M params (~16MB per round)
        - **Total per round: ~24MB** (vs sharing images: ~100s of MB)
        """)
