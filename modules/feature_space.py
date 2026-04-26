# ── modules/feature_space.py ──
"""
Feature Space Page
==================
Visualize DINOv2 / ResNet feature embeddings in 2D/3D.
"""

import streamlit as st
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import ORGANOID_CLASSES, CLASS_INFO, COLORS
from utils.helpers import generate_synthetic_features


def render():
    st.markdown(
        '<div class="main-header"><h1>🌌 Feature Space Explorer</h1>'
        '<p>Visualize organoid embeddings with t-SNE and UMAP dimensionality reduction</p></div>',
        unsafe_allow_html=True,
    )

    # ── Configuration ──
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("Samples", 100, 2000, 600, step=100)
    with col2:
        method = st.selectbox("Reduction Method", ["t-SNE", "UMAP", "PCA"])
    with col3:
        feature_dim = st.selectbox("Feature Dim", [128, 256, 512, 768], index=3)

    # ── Generate or Load Features ──
    if st.button("Generate & Visualize", type="primary"):
        with st.spinner("Generating features and reducing dimensions..."):
            features, labels, class_names = generate_synthetic_features(
                n_samples=n_samples, dim=feature_dim
            )

            # Dimensionality reduction
            reduced = _reduce_dimensions(features, method)

            # Store in session
            st.session_state["fs_features"] = features
            st.session_state["fs_labels"] = labels
            st.session_state["fs_reduced"] = reduced
            st.session_state["fs_class_names"] = class_names
            st.session_state["fs_method"] = method

    # ── Visualization ──
    if "fs_reduced" in st.session_state:
        reduced = st.session_state["fs_reduced"]
        labels = st.session_state["fs_labels"]
        class_names = st.session_state["fs_class_names"]
        method = st.session_state["fs_method"]

        tab_2d, tab_3d = st.tabs(["2D View", "3D View"])

        with tab_2d:
            fig = _plot_2d(reduced[:, :2], labels, class_names, method)
            st.plotly_chart(fig, use_container_width=True)

        with tab_3d:
            if reduced.shape[1] >= 3:
                fig = _plot_3d(reduced[:, :3], labels, class_names, method)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("3D view requires t-SNE or UMAP with n_components=3.")

        # ── Cluster Analysis ──
        st.markdown("---")
        st.markdown("### 📊 Cluster Quality")

        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        features = st.session_state["fs_features"]

        sil_score = silhouette_score(features, labels)
        ch_score = calinski_harabasz_score(features, labels)

        col_a, col_b = st.columns(2)
        col_a.metric("Silhouette Score", f"{sil_score:.3f}",
                     delta="Good" if sil_score > 0.5 else "Fair" if sil_score > 0.25 else "Poor")
        col_b.metric("Calinski-Harabasz", f"{ch_score:.1f}")

        # Per-class stats
        import pandas as pd
        rows = []
        for cls_id, cls_name in enumerate(class_names):
            cls_mask = labels == cls_id
            cls_feats = features[cls_mask]
            centroid = cls_feats.mean(axis=0)
            spread = np.mean(np.linalg.norm(cls_feats - centroid, axis=1))
            rows.append({
                "Class": cls_name,
                "Count": int(cls_mask.sum()),
                "Avg Spread": f"{spread:.2f}",
                "Intra-cluster SD": f"{np.std(np.linalg.norm(cls_feats - centroid, axis=1)):.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        st.info("Click 'Generate & Visualize' to explore the feature space.")

    # ── Methodology ──
    with st.expander("📖 Methodology"):
        st.markdown("""
        **Feature Extraction:**
        - **DINOv2** (Meta, 2023): Self-supervised ViT, 768-dim CLS token
        - **ResNet-18** (He et al., 2016): Supervised CNN, 512-dim

        **Dimensionality Reduction:**
        - **t-SNE** (van der Maaten & Hinton, 2008): Non-linear, preserves local structure
        - **UMAP** (McInnes et al., 2018): Non-linear, faster than t-SNE, preserves more global structure
        - **PCA** (Pearson, 1901): Linear, fast baseline

        **Cluster Quality Metrics:**
        - **Silhouette Score** (-1 to 1): How well-separated are clusters
        - **Calinski-Harabasz**: Ratio of between-cluster to within-cluster variance
        """)


def _reduce_dimensions(features: np.ndarray, method: str, n_components: int = 3) -> np.ndarray:
    """Reduce feature dimensions for visualization."""
    if method == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    elif method == "UMAP":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, random_state=42)
        except ImportError:
            st.warning("UMAP not installed. Falling back to t-SNE.")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:  # t-SNE
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)

    return reducer.fit_transform(features)


def _plot_2d(coords: np.ndarray, labels: np.ndarray, class_names: list[str], method: str):
    """2D scatter plot of reduced features."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for cls_id, cls_name in enumerate(class_names):
        mask = labels == cls_id
        color = CLASS_INFO.get(cls_name, {}).get("color", "#3b82f6")
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=cls_name,
            marker=dict(size=5, color=color, opacity=0.7),
        ))

    fig.update_layout(
        title=f"{method} Feature Space (2D)",
        template="plotly_white",
        height=600,
        xaxis_title=f"{method}-1",
        yaxis_title=f"{method}-2",
    )
    return fig


def _plot_3d(coords: np.ndarray, labels: np.ndarray, class_names: list[str], method: str):
    """3D scatter plot of reduced features."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for cls_id, cls_name in enumerate(class_names):
        mask = labels == cls_id
        color = CLASS_INFO.get(cls_name, {}).get("color", "#3b82f6")
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=coords[mask, 2],
            mode="markers",
            name=cls_name,
            marker=dict(size=4, color=color, opacity=0.7),
        ))

    fig.update_layout(
        title=f"{method} Feature Space (3D)",
        template="plotly_white",
        height=700,
        scene=dict(
            xaxis_title=f"{method}-1",
            yaxis_title=f"{method}-2",
            zaxis_title=f"{method}-3",
        ),
    )
    return fig
