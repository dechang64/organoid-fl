# ── modules/model_analysis.py ──
"""
Model Analysis Page
===================
Post-training model performance analysis: confusion matrix, per-class metrics, ROC.
"""

import streamlit as st
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import ORGANOID_CLASSES, CLASS_INFO, COLORS


def render():
    st.markdown(
        '<div class="main-header"><h1>📈 Model Analysis</h1>'
        '<p>Post-training performance analysis and model comparison</p></div>',
        unsafe_allow_html=True,
    )

    if "fl_history" not in st.session_state or not st.session_state["fl_history"]:
        st.info("Run FL training first to see model analysis.")
        if st.button("Go to Training", type="primary"):
            st.session_state["current_page"] = "fl_training"
        return

    history = st.session_state["fl_history"]
    class_names = st.session_state.get("fl_class_names", ORGANOID_CLASSES)
    total_time = st.session_state.get("fl_total_time", 0)

    # ── Summary Metrics ──
    best = max(history, key=lambda h: h["val_acc"])
    last = history[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Accuracy", f"{best['val_acc'] * 100:.2f}%", f"Round {best['round'] + 1}")
    col2.metric("Final Accuracy", f"{last['val_acc'] * 100:.2f}%")
    col3.metric("Final Loss", f"{last['val_loss']:.4f}")
    col4.metric("Total Time", f"{total_time:.1f}s")

    st.markdown("---")

    # ── Per-Round Breakdown ──
    st.markdown("### 📊 Round-by-Round Breakdown")

    import pandas as pd
    rows = []
    for h in history:
        row = {
            "Round": h["round"] + 1,
            "Val Accuracy (%)": f"{h['val_acc'] * 100:.2f}",
            "Val Loss": f"{h['val_loss']:.4f}",
            "Avg Train Acc (%)": f"{h['avg_train_acc'] * 100:.2f}",
            "Avg Train Loss": f"{h['avg_train_loss']:.4f}",
            "Time (s)": f"{h['elapsed']:.2f}",
        }
        # Per-client metrics
        for cm in h.get("client_metrics", []):
            row[f"Client {cm['client']} Acc (%)"] = f"{cm['train_acc'] * 100:.2f}"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Convergence Analysis ──
    st.markdown("### 📉 Convergence Analysis")

    tab_speed, tab_gap = st.tabs(["Convergence Speed", "Client Gap"])

    with tab_speed:
        # Find round where accuracy first exceeds 95%
        threshold = 0.95
        rounds_to_threshold = None
        for h in history:
            if h["val_acc"] >= threshold:
                rounds_to_threshold = h["round"] + 1
                break

        if rounds_to_threshold:
            st.success(f"Reached {threshold * 100:.0f}% accuracy in **{rounds_to_threshold} rounds**")
        else:
            st.warning(f"Did not reach {threshold * 100:.0f}% accuracy in {len(history)} rounds")

        # Accuracy improvement rate
        if len(history) >= 2:
            first_acc = history[0]["val_acc"]
            last_acc = history[-1]["val_acc"]
            improvement = last_acc - first_acc
            st.metric("Total Improvement", f"{improvement * 100:.2f}%")
            st.metric("Avg Improvement/Round", f"{improvement / len(history) * 100:.4f}%")

    with tab_gap:
        # Client accuracy gap analysis
        if history and "client_metrics" in history[-1]:
            import plotly.graph_objects as go
            last_metrics = history[-1]["client_metrics"]
            clients = [f"Client {m['client']}" for m in last_metrics]
            accs = [m["train_acc"] * 100 for m in last_metrics]

            fig = go.Figure(go.Bar(
                x=clients, y=accs,
                marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"]][:len(clients)],
                text=[f"{a:.1f}%" for a in accs],
                textposition="auto",
            ))
            fig.update_layout(
                title="Per-Client Training Accuracy (Final Round)",
                template="plotly_white",
                yaxis_title="Accuracy (%)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            gap = max(accs) - min(accs)
            if gap > 5:
                st.warning(f"⚠️ Client accuracy gap: {gap:.1f}%. Consider more rounds or FedProx.")
            else:
                st.success(f"✅ Client accuracy gap: {gap:.1f}%. Well-converged.")
