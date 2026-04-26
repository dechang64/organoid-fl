# ── modules/vector_search.py ──
"""
Vector Search Page
==================
Search the vector database for similar organoid images.
"""

import streamlit as st
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import ORGANOID_CLASSES, CLASS_INFO, COLORS
from utils.helpers import generate_synthetic_features
from analysis.vector_engine import VectorEngine
from visualization.charts import knn_distance_chart


def render():
    st.markdown(
        '<div class="main-header"><h1>🔍 Vector Search</h1>'
        '<p>Find similar organoid images using HNSW approximate nearest neighbor search</p></div>',
        unsafe_allow_html=True,
    )

    # ── Initialize Vector DB ──
    if "vector_engine" not in st.session_state:
        st.session_state["vector_engine"] = VectorEngine(dimension=512)
        st.session_state["vector_count"] = 0
        st.session_state["vector_dim"] = 512

    engine = st.session_state["vector_engine"]

    # ── Populate DB ──
    if engine.__len__() == 0:
        st.info("Vector DB is empty. Generate features to populate it.")
        n_vectors = st.slider("Number of vectors to generate", 100, 1000, 300, step=50)
        if st.button("Generate & Index Features", type="primary"):
            with st.spinner("Generating features and building index..."):
                features, labels, class_names = generate_synthetic_features(n_samples=n_vectors, dim=512)
                ids = [f"organoid_{i:04d}" for i in range(len(features))]
                metadata = [{"class": class_names[l], "label": int(l)} for l in labels]
                engine.bulk_insert(ids, features, metadata)
                st.session_state["vector_count"] = len(engine)
                st.session_state["vector_features"] = features
                st.session_state["vector_labels"] = labels
                st.session_state["vector_class_names"] = class_names
                st.success(f"Indexed {len(engine)} vectors (dim=512)")
                st.rerun()

    if engine.__len__() == 0:
        return

    # ── Search ──
    st.markdown("---")
    st.markdown("### 🔎 Search")

    col1, col2 = st.columns([1, 3])

    with col1:
        k = st.slider("k (neighbors)", 1, 20, 5)
        search_mode = st.radio("Query mode", ["Random sample", "By class"], horizontal=True)

        if search_mode == "By class":
            query_class = st.selectbox("Query class", ORGANOID_CLASSES)
        else:
            query_class = None

        if st.button("🔍 Search", type="primary", use_container_width=True):
            features = st.session_state["vector_features"]
            labels = st.session_state["vector_labels"]
            class_names = st.session_state["vector_class_names"]

            # Pick query
            if query_class and query_class in class_names:
                cls_idx = class_names.index(query_class)
                candidates = np.where(labels == cls_idx)[0]
                query_idx = np.random.choice(candidates)
            else:
                query_idx = np.random.randint(len(features))

            query_vec = features[query_idx]
            query_label = class_names[labels[query_idx]]

            # Search
            results = engine.search(query_vec, k=k)

            st.session_state["search_results"] = results
            st.session_state["search_query_idx"] = query_idx
            st.session_state["search_query_label"] = query_label

    with col2:
        if "search_results" in st.session_state:
            results = st.session_state["search_results"]
            query_label = st.session_state["search_query_label"]
            query_idx = st.session_state["search_query_idx"]

            st.markdown(f"**Query:** `organoid_{query_idx:04d}` ({CLASS_INFO.get(query_label, {}).get('label', query_label)})")

            # Results table
            import pandas as pd
            rows = []
            for rank, (vid, score) in enumerate(results, 1):
                meta = engine.metadata.get(vid, {})
                cls = meta.get("class", "unknown")
                rows.append({
                    "Rank": rank,
                    "ID": vid,
                    "Class": cls,
                    "Similarity": f"{score:.4f}",
                    "Match": "✅" if cls == query_label else "❌",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Distance chart
            fig = knn_distance_chart(results, query_label)
            st.plotly_chart(fig, use_container_width=True)

            matches = sum(1 for vid, _ in results if engine.metadata.get(vid, {}).get("class") == query_label)
            st.metric("Class Match Rate", f"{matches}/{len(results)} ({matches / len(results) * 100:.0f}%)")

    # ── DB Stats ──
    with st.expander("📊 Vector DB Statistics"):
        stats = engine.get_stats()
        col_a, col_b = st.columns(2)
        col_a.metric("Total Vectors", stats["total_vectors"])
        col_b.metric("Dimension", stats["dimension"])

        # Class distribution
        if "vector_labels" in st.session_state:
            import pandas as pd
            labels = st.session_state["vector_labels"]
            class_names = st.session_state["vector_class_names"]
            dist = pd.DataFrame({
                "Class": class_names,
                "Count": [int((labels == i).sum()) for i in range(len(class_names))],
            })
            st.dataframe(dist, use_container_width=True, hide_index=True)

    # ── Methodology ──
    with st.expander("📖 HNSW Search Methodology"):
        st.markdown("""
        **HNSW** (Hierarchical Navigable Small World) — Malkov & Yashunin, 2018

        A multi-layer graph structure for approximate nearest neighbor search:
        - **Layer 0**: Full graph with all vectors
        - **Upper layers**: Sparse graphs with fewer, longer-range connections
        - **Search**: Greedy descent from top layer to bottom

        **Parameters:**
        - `M=16`: Max connections per layer
        - `M0=32`: Max connections at layer 0
        - `ef=50`: Search beam width

        **Trade-off**: ~95% recall at 10-100x speedup vs brute-force.
        """)
