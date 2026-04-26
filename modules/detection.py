# ── modules/detection.py ──
"""
Detection Page
==============
YOLOv11 organoid detection: upload image → annotated results.
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import CLASS_INFO, COLORS
from analysis.detector import OrganoidDetector


def render():
    st.markdown(
        '<div class="main-header"><h1>🎯 Organoid Detection</h1>'
        '<p>YOLOv11-based organoid detection, classification, and counting</p></div>',
        unsafe_allow_html=True,
    )

    # ── Model Selection ──
    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox(
            "Model Size",
            options=["n (nano, 3.2M)", "s (small, 9.4M)", "m (medium, 20.1M)"],
            index=0,
        )
        model_key = model_size[0]
    with col2:
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, step=0.05)

    # ── Image Upload ──
    st.markdown("---")
    st.subheader("📤 Upload Organoid Image")

    uploaded = st.file_uploader(
        "Choose an organoid microscopy image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_array = np.array(image)

        col_preview, col_result = st.columns(2)

        with col_preview:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        with col_result:
            st.markdown("**Detection Results**")
            with st.spinner("Running YOLOv11 detection..."):
                try:
                    detector = OrganoidDetector(model_size=model_key)
                    detections = detector.detect(img_array, conf_threshold=conf_threshold)
                    summary = detector.summary(detections)

                    if detections:
                        # Summary metrics
                        st.metric("Organoids Detected", summary["total"])
                        st.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}")
                        st.metric("Avg Area", f"{summary['avg_area']:.0f} px²")

                        # Per-class breakdown
                        st.markdown("**Class Breakdown:**")
                        for cls_name, count in summary["classes"].items():
                            info = CLASS_INFO.get(cls_name, {})
                            st.markdown(
                                f"{info.get('emoji', '⚪')} **{cls_name}**: {count}"
                            )

                        # Detection details table
                        import pandas as pd
                        rows = [d.to_dict() for d in detections]
                        df = pd.DataFrame(rows)
                        df_display = df[["class_name", "confidence", "area", "width", "height"]]
                        df_display.columns = ["Class", "Confidence", "Area (px²)", "Width", "Height"]
                        df_display["Confidence"] = df_display["Confidence"].apply(lambda x: f"{x:.1%}")
                        st.dataframe(df_display, use_container_width=True, hide_index=True)

                        # Annotated image
                        st.markdown("---")
                        st.markdown("**Annotated Image**")
                        annotated = _draw_detections(img_array, detections)
                        st.image(annotated, use_container_width=True)

                    else:
                        st.warning("No organoids detected. Try lowering the confidence threshold.")

                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    st.info("Make sure `ultralytics` is installed: `pip install ultralytics`")

    else:
        # Demo mode
        with st.expander("🧪 Demo Mode (Synthetic Data)"):
            st.markdown("""
            Upload a real organoid microscopy image to see YOLOv11 detection in action.

            **Expected input:**
            - Organoid microscopy images (brightfield/phase contrast)
            - Format: PNG, JPG, or TIFF
            - Recommended size: 640×640 or larger

            **What YOLOv11 detects:**
            - Individual organoid boundaries (bounding boxes)
            - Classification: healthy / early_stage / late_stage
            - Confidence score per detection
            """)

    # ── Methodology ──
    with st.expander("📖 YOLOv11 Methodology"):
        st.markdown("""
        **YOLOv11** (Ultralytics, 2025)

        You Only Look Once — single-pass detection:
        1. **Backbone**: CSPDarknet extracts features at multiple scales
        2. **Neck**: FPN + PANet fuses multi-scale features
        3. **Head**: Predicts bounding boxes, classes, and confidence

        **Federated Learning Integration:**
        - Each hospital trains YOLO locally on its organoid images
        - Detection head + FPN weights are shared via FedAvg
        - No raw images leave the hospital

        **Performance:**
        | Model | Params | mAP50 | Speed |
        |-------|--------|-------|-------|
        | YOLO11n | 3.2M | ~70% | 1.5ms |
        | YOLO11s | 9.4M | ~75% | 2.5ms |
        | YOLO11m | 20.1M | ~80% | 5.0ms |
        """)


def _draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on image."""
    try:
        import cv2
        img = image.copy()
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            color_map = {
                "healthy": (34, 197, 94),
                "early_stage": (245, 158, 11),
                "late_stage": (239, 68, 68),
            }
            color = color_map.get(d.class_name, (59, 130, 246))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.0%}"
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
    except ImportError:
        return image
