# ── modules/segmentation.py ──
"""
Segmentation Page
=================
SAM2 pixel-level organoid segmentation with morphology analysis.
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import CLASS_INFO, COLORS


def render():
    st.markdown(
        '<div class="main-header"><h1>✂️ Organoid Segmentation</h1>'
        '<p>SAM2 pixel-level segmentation with morphological analysis</p></div>',
        unsafe_allow_html=True,
    )

    # ── Pipeline Selection ──
    st.markdown("""
    **Segmentation Pipeline:**
    1. **YOLOv11** detects organoids → bounding boxes
    2. **SAM2** uses boxes as prompts → pixel-level masks
    3. **Morphology** extracted from masks → shape metrics
    """)

    # ── Image Upload ──
    uploaded = st.file_uploader(
        "Upload organoid image for segmentation",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="seg_upload",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_array = np.array(image)

        col_orig, col_seg = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(image, use_container_width=True)

        with col_seg:
            st.markdown("**Segmentation**")
            with st.spinner("Running SAM2 segmentation..."):
                try:
                    # Try full pipeline: YOLO → SAM2
                    from analysis.detector import OrganoidDetector
                    from analysis.segmentor import OrganoidSegmentor

                    # Step 1: YOLO detection
                    detector = OrganoidDetector(model_size="n")
                    detections = detector.detect(img_array, conf_threshold=0.25)

                    if detections:
                        # Step 2: SAM2 segmentation
                        segmentor = OrganoidSegmentor()
                        seg_results = segmentor.segment_from_detections(img_array, detections)

                        if seg_results:
                            # Draw segmentation overlay
                            seg_overlay = _draw_segmentation(img_array, seg_results)
                            st.image(seg_overlay, use_container_width=True)

                            # Morphology table
                            st.markdown("---")
                            st.markdown("### 📐 Morphological Analysis")

                            import pandas as pd
                            rows = [r.to_dict() for r in seg_results]
                            df = pd.DataFrame(rows)
                            df_display = df[[
                                "area", "perimeter", "circularity",
                                "solidity", "aspect_ratio", "eccentricity"
                            ]]
                            df_display.columns = [
                                "Area (px)", "Perimeter", "Circularity",
                                "Solidity", "Aspect Ratio", "Eccentricity"
                            ]
                            st.dataframe(df_display, use_container_width=True, hide_index=True)

                            # Distribution charts
                            _plot_morphology_charts(seg_results)

                            # Quality assessment
                            st.markdown("---")
                            st.markdown("### 📊 Quality Assessment")
                            avg_circ = np.mean([r.circularity for r in seg_results])
                            avg_solid = np.mean([r.solidity for r in seg_results])

                            if avg_circ > 0.7:
                                st.success(f"✅ Avg circularity: {avg_circ:.2f} — well-formed organoids")
                            elif avg_circ > 0.4:
                                st.warning(f"⚠️ Avg circularity: {avg_circ:.2f} — moderate irregularity")
                            else:
                                st.error(f"❌ Avg circularity: {avg_circ:.2f} — highly irregular morphology")

                            if avg_solid > 0.8:
                                st.success(f"✅ Avg solidity: {avg_solid:.2f} — convex shapes")
                            else:
                                st.warning(f"⚠️ Avg solidity: {avg_solid:.2f} — concave boundaries detected")
                        else:
                            st.warning("SAM2 segmentation returned no masks.")
                    else:
                        st.warning("No organoids detected by YOLO. Cannot run segmentation.")

                except ImportError as e:
                    st.error(f"Missing dependency: {e}")
                    st.info("""
                    Install required packages:
                    ```
                    pip install ultralytics
                    pip install git+https://github.com/facebookresearch/sam2.git
                    pip install opencv-python
                    ```
                    """)
                except Exception as e:
                    st.error(f"Segmentation failed: {e}")

    else:
        with st.expander("🧪 Demo Mode"):
            st.markdown("""
            Upload a real organoid microscopy image to see the full
            YOLO → SAM2 segmentation pipeline.

            **Morphology Metrics Explained:**
            - **Circularity** (0–1): 1 = perfect circle. Healthy organoids tend to be round.
            - **Solidity** (0–1): 1 = fully convex. Low values indicate concave/irregular boundaries.
            - **Aspect Ratio**: Width/height of fitted ellipse. 1 = circular.
            - **Eccentricity** (0–1): 0 = circle, 1 = elongated/line-like.
            """)

    # ── Methodology ──
    with st.expander("📖 SAM2 Methodology"):
        st.markdown("""
        **SAM2** (Segment Anything Model 2, Meta AI, 2024)

        The most advanced universal image segmentation model:
        - **Promptable**: Accepts boxes, points, or text as prompts
        - **Zero-shot**: Works on unseen organoid types without retraining
        - **Real-time**: Processes images in <1s on GPU

        **Our Pipeline:**
        1. YOLOv11 detects organoid bounding boxes
        2. Boxes are passed as prompts to SAM2
        3. SAM2 predicts pixel-level masks
        4. Morphological metrics extracted from masks

        **Federated Learning:**
        - SAM2 backbone is frozen (shared, no training needed)
        - Only prompt encoder weights are fine-tuned per hospital
        - Prompt encoder weights aggregated via FedAvg
        """)


def _draw_segmentation(image: np.ndarray, results: list) -> np.ndarray:
    """Draw segmentation masks as colored overlay."""
    try:
        import cv2
        img = image.copy()
        overlay = img.copy()

        colors = [
            (34, 197, 94, 100),    # green
            (245, 158, 11, 100),   # amber
            (239, 68, 68, 100),    # red
            (59, 130, 246, 100),   # blue
            (168, 85, 247, 100),   # purple
            (6, 182, 212, 100),    # cyan
        ]

        for i, r in enumerate(results):
            color = colors[i % len(colors)][:3]
            mask = r.mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            cv2.fillPoly(overlay, contours, (*color, 40) if len(color) == 3 else color)

        # Blend
        result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        return result
    except ImportError:
        return image


def _plot_morphology_charts(results: list):
    """Plot morphology distribution charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(results)
    indices = list(range(1, n + 1))
    circularities = [r.circularity for r in results]
    solidities = [r.solidity for r in results]
    areas = [r.area for r in results]
    eccentricities = [r.eccentricity for r in results]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Circularity", "Solidity", "Area Distribution", "Eccentricity"),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(x=indices, y=circularities, marker_color=COLORS["primary"],
                         name="Circularity"), row=1, col=1)
    fig.add_trace(go.Bar(x=indices, y=solidities, marker_color=COLORS["secondary"],
                         name="Solidity"), row=1, col=2)
    fig.add_trace(go.Bar(x=indices, y=areas, marker_color=COLORS["accent"],
                         name="Area"), row=2, col=1)
    fig.add_trace(go.Bar(x=indices, y=eccentricities, marker_color="#ef4444",
                         name="Eccentricity"), row=2, col=2)

    fig.update_layout(
        template="plotly_white",
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
