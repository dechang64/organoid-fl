# ── modules/data_explorer.py ──
"""
Data Explorer Page
==================
Upload organoid images or use built-in demo data.
Visualize data distribution and sample images.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image

from utils.constants import ORGANOID_CLASSES, CLASS_INFO, COLORS
from data.synthetic import generate_organoid_image


def render():
    st.markdown(
        '<div class="main-header"><h1>📁 Data Explorer</h1>'
        '<p>Manage organoid image data — upload or generate synthetic samples</p></div>',
        unsafe_allow_html=True,
    )

    tab_gen, tab_upload, tab_dist = st.tabs(["🧪 Generate Data", "📤 Upload Images", "📊 Distribution"])

    with tab_gen:
        st.subheader("Synthetic Organoid Data Generator")
        st.markdown("Generate realistic organoid-like microscopy images for testing.")

        col1, col2 = st.columns(2)
        with col1:
            n_per_class = st.slider("Images per class", 50, 500, 200, step=50)
            img_size = st.selectbox("Image size", [64, 128, 224, 256], index=1)
        with col2:
            seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999)
            stage = st.selectbox("Preview stage", ORGANOID_CLASSES)

        # Preview
        st.markdown("**Preview:**")
        preview_cols = st.columns(5)
        for i, col in enumerate(preview_cols):
            img = generate_organoid_image(size=img_size, stage=stage, seed=seed + i)
            col.image(img, caption=f"Seed {seed + i}", use_container_width=True)

        if st.button("Generate Full Dataset", type="primary"):
            from data.synthetic import generate_dataset
            with st.spinner("Generating images..."):
                output_dir = Path("data/synthetic_organoids")
                generate_dataset(str(output_dir), n_per_class=n_per_class, img_size=img_size)
                st.session_state["data_dir"] = str(output_dir)
                st.session_state["data_generated"] = True
                st.success(f"Generated {n_per_class * 3} images in {output_dir}")

    with tab_upload:
        st.subheader("Upload Organoid Images")
        st.markdown("Upload images organized by class folder, or individual images.")

        uploaded = st.file_uploader(
            "Choose organoid images",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
        )

        if uploaded:
            st.info(f"Uploaded {len(uploaded)} images")
            preview_cols = st.columns(min(5, len(uploaded)))
            for i, f in enumerate(uploaded[:5]):
                img = Image.open(f)
                preview_cols[i].image(img, caption=f.name, use_container_width=True)

    with tab_dist:
        st.subheader("Data Distribution")

        # Generate demo distribution
        import plotly.graph_objects as go
        import plotly.express as px

        n_samples = st.slider("Total samples", 100, 1500, 600, step=100)
        n_clients = st.slider("Number of clients", 2, 8, 3)
        non_iid = st.slider("Non-IID degree", 0.0, 1.0, 0.0, step=0.1)

        # Simulate distribution
        np.random.seed(42)
        base = n_samples // n_clients
        data = []
        for cid in range(n_clients):
            for cls in ORGANOID_CLASSES:
                if non_iid > 0:
                    # Non-IID: primary class gets more
                    primary = cid % len(ORGANOID_CLASSES)
                    if cls == ORGANOID_CLASSES[primary]:
                        count = int(base * (1 + non_iid * 2))
                    else:
                        count = int(base * (1 - non_iid * 0.5))
                else:
                    count = base // len(ORGANOID_CLASSES)
                data.append({"Client": f"Client {cid + 1}", "Class": cls, "Count": max(count, 5)})

        df_dist = __import__("pandas", fromlist=["DataFrame"]).DataFrame(data)

        fig = px.bar(
            df_dist, x="Client", y="Count", color="Class",
            color_discrete_map={cls: CLASS_INFO[cls]["color"] for cls in ORGANOID_CLASSES},
            barmode="stack",
            title=f"Data Distribution (Non-IID={non_iid:.1f})",
        )
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

        if non_iid > 0:
            st.warning(f"⚠️ Non-IID={non_iid:.1f}: Data is unevenly distributed across clients. "
                       "This is realistic but makes FL convergence harder.")
