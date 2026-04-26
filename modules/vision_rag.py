# ── modules/vision_rag.py ──
"""
Vision RAG Page
===============
Cross-hospital similar case retrieval and diagnostic report generation.
"""

import streamlit as st
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import CLASS_INFO, COLORS
from analysis.vision_rag import VisionRAG, encode_morphology, MORPHOLOGY_FEATURES


def render():
    st.markdown(
        '<div class="main-header"><h1>🔍 Vision RAG</h1>'
        '<p>Cross-hospital similar case retrieval — images never leave your hospital</p></div>',
        unsafe_allow_html=True,
    )

    # ── Initialize ──
    if "vision_rag" not in st.session_state:
        rag = VisionRAG()
        rag.populate_demo(n_cases=150, n_hospitals=4, seed=42)
        st.session_state["vision_rag"] = rag

    rag: VisionRAG = st.session_state["vision_rag"]

    # ── Stats Bar ──
    stats = rag.get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", stats["total_cases"])
    col2.metric("Hospitals", stats["hospitals"])
    col3.metric("Vector DB Size", stats["vector_db_size"])
    col4.metric("Feature Dim", rag.feature_dim)

    st.markdown("---")

    # ── Two Modes ──
    tab_query, tab_register, tab_explore = st.tabs([
        "🔎 Query Similar Cases",
        "📝 Register New Case",
        "📊 Explore Database",
    ])

    # ── Query Tab ──
    with tab_query:
        st.markdown("### Enter Morphology Metrics")
        st.caption("These would normally come from YOLO + SAM2 analysis of a new image.")

        col_a, col_b = st.columns(2)

        with col_a:
            area = st.number_input("Area (px²)", 100, 50000, 5000, step=100)
            perimeter = st.number_input("Perimeter (px)", 50, 2000, 300, step=10)
            circularity = st.slider("Circularity", 0.0, 1.0, 0.65, step=0.01)
            solidity = st.slider("Solidity", 0.3, 1.0, 0.85, step=0.01)
            aspect_ratio = st.slider("Aspect Ratio", 0.3, 5.0, 1.2, step=0.1)
            eccentricity = st.slider("Eccentricity", 0.0, 1.0, 0.3, step=0.01)

        with col_b:
            n_organoids = st.number_input("Organoid Count", 1, 30, 5)
            avg_area = st.number_input("Avg Area (px²)", 100, 30000, 4000, step=100)
            std_area = st.number_input("Area Std Dev", 0, 15000, 1000, step=100)

            st.markdown("**Class Distribution (from YOLO)**")
            p_healthy = st.slider("P(healthy)", 0.0, 1.0, 0.5, step=0.05)
            p_early = st.slider("P(early_stage)", 0.0, 1.0, 0.3, step=0.05)
            p_late = max(0, 1.0 - p_healthy - p_early)
            st.caption(f"P(late_stage) = {p_late:.2f} (auto-computed)")

        k = st.slider("Number of similar cases", 3, 30, 10, step=1)
        min_sim = st.slider("Min similarity", 0.0, 1.0, 0.3, step=0.05)

        if st.button("🔍 Search Similar Cases", type="primary"):
            morphology = {
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "solidity": solidity,
                "aspect_ratio": aspect_ratio,
                "eccentricity": eccentricity,
                "n_organoids": n_organoids,
                "avg_area": avg_area,
                "std_area": std_area,
                "class_distribution": [p_healthy, p_early, p_late],
            }

            with st.spinner("Searching across hospitals..."):
                similar = rag.query(morphology, k=k, min_similarity=min_sim)
                report = rag.generate_report(morphology, similar)

            # ── Results ──
            if report["status"] == "no_similar_cases":
                st.warning(report["message"])
            else:
                # Diagnosis summary
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Top Diagnosis", report["top_diagnosis"].replace("_", " "))
                col_r2.metric("Confidence", f"{report['diagnosis_confidence']:.1%}")
                col_r3.metric("Cases Found", report["similar_cases_found"])

                # Diagnosis breakdown
                st.markdown("### Diagnosis Breakdown")
                breakdown = report["diagnosis_breakdown"]
                cols = st.columns(len(breakdown))
                for i, (diag, count) in enumerate(breakdown.items()):
                    with cols[i]:
                        st.metric(diag.replace("_", " "), count)

                # Recommendations
                st.markdown("### 📋 Recommendations")
                for rec in report["recommendations"]:
                    st.markdown(f"- {rec}")

                # Similar cases table
                st.markdown("### 📑 Similar Cases")
                if similar:
                    table_data = []
                    for case_id, sim, case in similar:
                        table_data.append({
                            "Case ID": case_id,
                            "Hospital": case.hospital_id,
                            "Diagnosis": case.diagnosis.replace("_", " "),
                            "Similarity": f"{sim:.3f}",
                            "Confidence": f"{case.confidence:.2f}",
                            "Circularity": f"{case.morphology.get('circularity', 0):.2f}",
                            "Eccentricity": f"{case.morphology.get('eccentricity', 0):.2f}",
                        })
                    st.dataframe(table_data, use_container_width=True, hide_index=True)

                # Similarity distribution chart
                _plot_similarity_distribution(similar)

    # ── Register Tab ──
    with tab_register:
        st.markdown("### Register a New Case")
        st.caption("In production, this happens automatically after YOLO + SAM2 analysis.")

        with st.form("register_form"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                hospital = st.selectbox("Hospital", ["hospital_1", "hospital_2", "hospital_3", "hospital_4"])
                diagnosis = st.selectbox("Diagnosis", ["healthy", "early_stage", "late_stage"])
                confidence = st.slider("Model Confidence", 0.0, 1.0, 0.85)
            with col_f2:
                reg_circ = st.slider("Circularity", 0.0, 1.0, 0.7)
                reg_ecc = st.slider("Eccentricity", 0.0, 1.0, 0.2)
                reg_area = st.number_input("Area", 100, 50000, 5000)

            report_text = st.text_area("Diagnostic Report", height=100,
                                       value="Organoid morphology consistent with classification.")

            submitted = st.form_submit_button("📝 Register Case", type="primary")
            if submitted:
                morphology = {
                    "area": reg_area,
                    "perimeter": 300,
                    "circularity": reg_circ,
                    "solidity": 0.85,
                    "aspect_ratio": 1.1,
                    "eccentricity": reg_ecc,
                    "n_organoids": 5,
                    "avg_area": reg_area,
                    "std_area": 1000,
                    "class_distribution": [0.7, 0.2, 0.1] if diagnosis == "healthy" else
                                          [0.1, 0.3, 0.6] if diagnosis == "late_stage" else
                                          [0.2, 0.5, 0.3],
                }
                case_id = rag.register_case(
                    morphology=morphology,
                    diagnosis=diagnosis,
                    confidence=confidence,
                    report=report_text,
                    hospital_id=hospital,
                )
                st.success(f"Case {case_id} registered successfully!")

    # ── Explore Tab ──
    with tab_explore:
        st.markdown("### Database Overview")

        # Hospital distribution
        hosp_dist = stats["hospital_distribution"]
        diag_dist = stats["diagnosis_distribution"]

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**Cases per Hospital**")
            for hosp, count in hosp_dist.items():
                st.write(f"- {hosp}: {count} cases")
        with col_e2:
            st.markdown("**Diagnosis Distribution**")
            for diag, count in diag_dist.items():
                st.write(f"- {diag.replace('_', ' ')}: {count} cases")

        _plot_database_charts(rag)

    # ── Methodology ──
    with st.expander("📖 Vision RAG Methodology"):
        st.markdown("""
        **Vision-to-Text RAG Pipeline**

        ```
        New Image
            │
            ▼
        YOLOv11 Detection ──→ Bounding Boxes
            │
            ▼
        SAM2 Segmentation ──→ Pixel Masks
            │
            ▼
        Morphology Extraction ──→ {circularity, area, eccentricity, ...}
            │
            ▼
        Feature Encoding ──→ Normalized Vector (12-dim)
            │
            ▼
        HNSW Vector Search ──→ Top-K Similar Cases
            │
            ▼
        Federated Report Collection ──→ Anonymized Reports
            │
            ▼
        Local LLM Generation ──→ Structured Diagnosis
        ```

        **Privacy Guarantees:**
        - ❌ Patient images are NEVER shared
        - ❌ Raw pixel data never leaves the hospital
        - ✅ Only morphology feature vectors are shared (12 floats)
        - ✅ Diagnostic reports are anonymized before sharing
        - ✅ LLM generation runs entirely on the querying hospital

        **Communication Cost:**
        - Query: 12 floats (48 bytes)
        - Response: K anonymized reports (~1-5 KB)
        - **vs. sharing images: ~100s of KB per image**
        """)


def _plot_similarity_distribution(similar_cases):
    """Plot similarity score distribution."""
    import plotly.graph_objects as go

    if not similar_cases:
        return

    case_ids = [cid for cid, _, _ in similar_cases]
    similarities = [sim for _, sim, _ in similar_cases]
    diagnoses = [c.diagnosis.replace("_", " ") for _, _, c in similar_cases]
    hospitals = [c.hospital_id for _, _, c in similar_cases]

    color_map = {
        "healthy": "#22c55e",
        "early stage": "#f59e0b",
        "late stage": "#ef4444",
    }
    colors = [color_map.get(d, "#3b82f6") for d in diagnoses]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=case_ids,
        y=similarities,
        marker_color=colors,
        text=[f"{s:.3f}" for s in similarities],
        textposition="auto",
    ))

    fig.update_layout(
        title="Similarity Scores (color = diagnosis)",
        template="plotly_white",
        yaxis_title="Cosine Similarity",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_database_charts(rag: VisionRAG):
    """Plot database distribution charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    stats = rag.get_stats()
    hosp = stats["hospital_distribution"]
    diag = stats["diagnosis_distribution"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cases per Hospital", "Diagnosis Distribution"),
        horizontal_spacing=0.15,
    )

    fig.add_trace(go.Bar(
        x=list(hosp.keys()),
        y=list(hosp.values()),
        marker_color=["#3b82f6", "#8b5cf6", "#06b6d4", "#ec4899"][:len(hosp)],
        name="Hospital",
    ), row=1, col=1)

    diag_colors = {
        "healthy": "#22c55e",
        "early_stage": "#f59e0b",
        "late_stage": "#ef4444",
    }
    fig.add_trace(go.Bar(
        x=[d.replace("_", " ") for d in diag.keys()],
        y=list(diag.values()),
        marker_color=[diag_colors.get(d, "#3b82f6") for d in diag.keys()],
        name="Diagnosis",
    ), row=1, col=2)

    fig.update_layout(template="plotly_white", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
