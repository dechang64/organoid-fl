# ── modules/fl_training.py ──
"""
FL Training Page — with Phase 1 (Classification) + Phase 2 (YOLO Detection)

Phase 1: ResNet18 on 23K organoid patches (4 classes)
Phase 2: YOLOv12n on 840 intestinal organoid images (23K bboxes)
"""

import streamlit as st
import numpy as np
import json
import csv
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import (
    DEFAULT_ROUNDS, DEFAULT_CLIENTS, DEFAULT_LR,
    DEFAULT_BATCH_SIZE, DEFAULT_LOCAL_EPOCHS,
    ORGANOID_CLASSES, CLASS_INFO, REFERENCES,
)

# ── Data directories ──
RESULTS_DIR = Path(__file__).parent.parent / "results" / "fl_training"
PHASE2_DIR = Path(__file__).parent.parent / "results" / "phase2_yolo"
PATCHES_DIR = Path(__file__).parent.parent / "data" / "organoid_patches"
DETECT_DIR = Path(__file__).parent.parent / "data" / "intestinal_organoid" / "OrganoidDataset"


def load_phase1_results():
    """Load Phase 1 classification FL results."""
    results = {}
    for f in RESULTS_DIR.glob("classify_*_rounds.json"):
        name = f.stem.replace("classify_", "").replace("_rounds", "")
        try:
            results[name] = json.load(open(f))
        except Exception:
            pass
    return results


def load_phase2_results():
    """Load Phase 2 YOLO detection FL results."""
    p2 = {}
    # Convergence
    conv_file = PHASE2_DIR / "convergence_results.csv"
    if conv_file.exists():
        with open(conv_file) as f:
            p2["convergence"] = list(csv.DictReader(f))
    # μ sensitivity
    mu_file = PHASE2_DIR / "mu_sensitivity.csv"
    if mu_file.exists():
        with open(mu_file) as f:
            p2["mu_sensitivity"] = list(csv.DictReader(f))
    # Round-level data
    for name in ["ewa_rounds", "fedavg_rounds"]:
        rf = PHASE2_DIR / f"{name}.json"
        if rf.exists():
            with open(rf) as f:
                p2[name] = json.load(f)
    # Summary
    sf = PHASE2_DIR / "fl_summary.json"
    if sf.exists():
        with open(sf) as f:
            p2["summary"] = json.load(f)
    # Experiment matrix
    mf = PHASE2_DIR / "experiment_matrix.csv"
    if mf.exists():
        with open(mf) as f:
            p2["experiment_matrix"] = list(csv.DictReader(f))
    return p2


def check_data_availability():
    """Check which datasets are available."""
    has_patches = PATCHES_DIR.exists() and (PATCHES_DIR / "data.yaml").exists()
    has_detect = DETECT_DIR.exists() and (DETECT_DIR / "data.yaml").exists()
    has_phase1_results = bool(load_phase1_results())
    has_phase2_results = bool(PHASE2_DIR.exists() and list(PHASE2_DIR.glob("*.csv")))
    return has_patches, has_detect, has_phase1_results, has_phase2_results


def render():
    st.markdown(
        '<div class="main-header"><h1>🔄 Federated Learning</h1>'
        '<p>Organoid image analysis with privacy-preserving distributed training</p></div>',
        unsafe_allow_html=True,
    )

    # ── Data availability check ──
    has_patches, has_detect, has_p1_res, has_p2_res = check_data_availability()

    # ── Architecture diagram ──
    st.markdown("""
    ### 🌐 Federated Learning Architecture

    ```
    Client A (organoid0 specialist: cystic)     ──┐
    Client B (organoid1+3 specialist: maturing) ──┼──→ Aggregation Server ──→ Global Model
    Client C (spheroid specialist)              ──┘
    ```

    **Dataset**: Intestinal Organoid (840 images, 23K annotations, 4 classes) | **Long-tail**: 5.3:1
    """)

    # ── Tab layout ──
    tab_names = [
        "📊 Phase 1: Classification",
        "🚀 Phase 2: YOLO Detection",
        "🎯 EWA Effective Interval",
        "📖 Methodology",
    ]
    tabs = st.tabs(tab_names)

    # ══════════════════════════════════════════════════════════════
    # Tab 1: Phase 1 — Classification (ResNet18, 23K patches)
    # ══════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("📊 Phase 1: ResNet18 Classification on Organoid Patches")
        st.markdown("""
        **23,052 cropped organoid patches** from real mouse intestinal organoid microscopy images.

        | Class | Train | Val | Total | Description |
        |-------|-------|-----|-------|-------------|
        | organoid0 | 9,487 | 2,424 | 11,911 | Cystic non-budding |
        | organoid1 | 4,520 | 990 | 5,510 | Early organoid |
        | organoid3 | 2,743 | 623 | 3,366 | Late organoid |
        | spheroid | 1,743 | 522 | 2,265 | Spheroid |
        | **Total** | **18,493** | **4,559** | **23,052** | |

        **FL Setup**: 3 clients with Non-IID data distribution, comparing FedAvg / FedProx / EWA-v2.
        """)

        if not has_patches:
            st.warning("📋 Organoid patches not found. Run `python data/crop_patches.py` first.")
            st.code("python data/crop_patches.py  # crops 23K patches from YOLO detection set")

        if has_p1_res:
            p1 = load_phase1_results()
            strategies = list(p1.keys())

            # Summary table
            st.markdown("### Results Summary")
            table_data = []
            for name in strategies:
                d = p1[name]
                if isinstance(d, list) and d:
                    best = max(r.get("global_acc", 0) for r in d)
                    final = d[-1].get("global_acc", 0)
                    table_data.append({
                        "Strategy": name,
                        "Best Accuracy": f"{best:.1%}",
                        "Final Accuracy": f"{final:.1%}",
                    })
            if table_data:
                st.dataframe(table_data, use_container_width=True, hide_index=True)

            # Convergence curves
            st.markdown("### Convergence Curves")
            chart_data = {}
            for name in strategies:
                d = p1[name]
                if isinstance(d, list):
                    chart_data[name] = [r.get("global_acc", 0) for r in d]
            if chart_data:
                st.line_chart(chart_data, use_container_width=True)
                st.caption("Global accuracy across FL rounds")
        else:
            st.info("🧪 No Phase 1 results yet. Run the simulation on your local GPU:")
            st.code("""
# Quick test (2 rounds)
python fl_classify_sim.py --data ./organoid_patches --quick --device 0

# Full experiment matrix (4 Non-IID × 3 strategies)
python fl_classify_sim.py --data ./organoid_patches --matrix --device 0
            """, language="bash")

        # Dataset stats visualization
        if has_patches:
            st.markdown("### 📈 Class Distribution")
            class_counts = {}
            for cls in ORGANOID_CLASSES:
                cls_dir = PATCHES_DIR / "train" / cls
                if cls_dir.exists():
                    class_counts[cls] = len(list(cls_dir.glob("*.jpg")))
            if class_counts:
                st.bar_chart(class_counts, use_container_width=True)
                st.caption("Training set class distribution (long-tail 5.3:1 → natural Non-IID)")

    # ══════════════════════════════════════════════════════════════
    # Tab 2: Phase 2 — YOLO Detection
    # ══════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("🚀 Phase 2: YOLOv12n Object Detection + EWA Active Aggregation")
        st.markdown("""
        **Upgrade from ResNet18 → YOLOv12n**: Phase 2 extends FL from classification to detection,
        and evaluates EWA as an active aggregation strategy.

        | | Phase 1 (ResNet18) | Phase 2 (YOLOv12n) |
        |---|---|---|
        | **Task** | Image Classification | Object Detection |
        | **Metric** | Accuracy | mAP50-95 |
        | **Data** | 23K patches (cropped) | 840 full images (1280×960) |
        | **Annotations** | 1 label/image | 23,065 bboxes |
        | **Clients** | 3 (class-dominant) | 3 (class-specialist) |
        | **Strategies** | FedAvg/FedProx/EWA-v2 | FedAvg/FedProx/EWA-v2/EWA-FedProx |
        """)

        if not has_detect:
            st.warning("📋 Detection dataset not found. Download from Zenodo:")
            st.code("""
# Download Intestinal Organoid dataset
cd data
curl -L -o OrganoidDataset.zip "https://zenodo.org/api/records/6768583/files/OrganoidDataset.zip/content"
unzip OrganoidDataset.zip -d intestinal_organoid
            """, language="bash")

        p2 = load_phase2_results()

        if p2.get("convergence"):
            conv = p2["convergence"]
            ewa_rounds = p2.get("ewa_rounds", [])
            fedavg_rounds = p2.get("fedavg_rounds", [])
            mu_data = p2.get("mu_sensitivity", [])

            # ── Strategy Comparison ──
            st.markdown("### 📊 Strategy Comparison (Final mAP50-95)")
            strat_names = ["FedAvg", "FedProx", "EWA-v2", "EWA-FedProx"]
            scenario_labels = ["IID Balanced", "Moderate Non-IID", "Extreme Non-IID"]
            scenario_keys = ["iid_balanced", "moderate_balanced", "extreme_extreme"]

            strat_data = {}
            for strat in strat_names:
                vals = []
                for sk in scenario_keys:
                    match = [r for r in conv if r["scenario"] == sk and r["strategy"] == strat]
                    vals.append(float(match[0]["final_mAP"]) if match else 0)
                strat_data[strat] = vals

            st.bar_chart(strat_data, use_container_width=True)
            st.caption("Final mAP50-95 for each strategy across Non-IID scenarios")

            # ── Training Curves ──
            st.markdown("### 📈 Training Curves (Extreme Non-IID)")
            if ewa_rounds or fedavg_rounds:
                curve_data = {}
                if fedavg_rounds:
                    curve_data["FedAvg (global mAP)"] = [r["global_mAP"] for r in fedavg_rounds]
                if ewa_rounds:
                    curve_data["EWA-v2 (global mAP)"] = [r["global_mAP"] for r in ewa_rounds]
                    for ci, cr in enumerate(ewa_rounds[0]["client_results"]):
                        dom = cr.get("dominant_classes", [])
                        curve_data[f"Client {ci} (dom: {dom})"] = [r["client_results"][ci]["mAP"] for r in ewa_rounds]

                st.line_chart(curve_data, use_container_width=True)
                st.caption("Global and per-client mAP50-95 over FL rounds")

            # ── Specialist Advantage ──
            if ewa_rounds:
                st.markdown("### 🎯 Specialist Advantage")
                sa_data = {}
                for ci, cr in enumerate(ewa_rounds[0]["client_results"]):
                    dom = cr.get("dominant_classes", [])
                    gaps = []
                    for r in ewa_rounds:
                        c = r["client_results"][ci]
                        pca = c["per_class_ap"]
                        dom_avg = np.mean([pca[f"class_{dc}"] for dc in dom]) if dom else 0
                        nondom_avg = np.mean([pca[f"class_{j}"] for j in range(4) if j not in dom])
                        gaps.append(dom_avg - nondom_avg)
                    sa_data[f"Client {ci} (dom: {dom})"] = gaps

                st.line_chart(sa_data, use_container_width=True)
                st.caption("Positive gap = specialist advantage preserved. EWA protects minority expertise.")

            # ── μ Sensitivity ──
            if mu_data:
                st.markdown("### 🔧 μ Sensitivity (FedProx)")
                mu_chart = {}
                for sk, sl in [("moderate_balanced", "Moderate"), ("extreme_extreme", "Extreme")]:
                    rows = sorted([r for r in mu_data if r["scenario"] == sk], key=lambda x: float(x["mu"]))
                    mu_chart[f"{sl} (Final)"] = [float(r["final_mAP"]) for r in rows]
                    mu_chart[f"{sl} (Best)"] = [float(r["best_mAP"]) for r in rows]

                st.line_chart(mu_chart, use_container_width=True)

            # ── Full Results Table ──
            st.markdown("### 📋 Full Results")
            scenario_display = {"iid_balanced": "IID", "moderate_balanced": "Moderate", "extreme_extreme": "Extreme"}
            table_rows = []
            for r in conv:
                table_rows.append({
                    "Scenario": scenario_display.get(r["scenario"], r["scenario"]),
                    "Strategy": r["strategy"],
                    "Final mAP": f"{float(r['final_mAP']):.4f}",
                    "Best mAP": f"{float(r['best_mAP']):.4f}",
                    "Gap": f"{float(r['best_mAP']) - float(r['final_mAP']):.4f}",
                    "Conv Round": r["convergence_round"],
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

            # ── Export ──
            col_a, col_b = st.columns(2)
            with col_a:
                csv_lines = ["scenario,strategy,final_mAP,best_mAP,convergence_round,peak_round"]
                for r in conv:
                    csv_lines.append(f"{r['scenario']},{r['strategy']},{r['final_mAP']},{r['best_mAP']},{r['convergence_round']},{r['peak_round']}")
                st.download_button("📥 Convergence CSV", "\n".join(csv_lines), "phase2_convergence.csv", "text/csv")
            with col_b:
                json_str = json.dumps(p2, indent=2, ensure_ascii=False)
                st.download_button("📥 All Results JSON", json_str, "phase2_results.json", "application/json")
        else:
            st.info("🧪 No Phase 2 results yet. Run the simulation on your local GPU:")
            st.code("""
# Quick test (2 rounds)
python fl_detect_sim.py --data ./intestinal_organoid/OrganoidDataset/data.yaml --quick --device 0

# Full run (10 rounds, 3 strategies)
python fl_detect_sim.py --data ./intestinal_organoid/OrganoidDataset/data.yaml --device 0
            """, language="bash")

    # ══════════════════════════════════════════════════════════════
    # Tab 3: EWA Effective Interval
    # ══════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("🎯 EWA Effective Interval")
        st.markdown("""
        **Core finding**: EWA's benefit is **conditional on data heterogeneity**.

        | Non-IID Level | EWA vs FedAvg | Verdict |
        |---------------|---------------|---------|
        | IID Balanced | −5.7pp | ⚠️ Harmful |
        | Moderate Non-IID | −1.3pp | ⚠️ Neutral |
        | Extreme Non-IID | +1.1pp | ✅ Beneficial |

        **EWA weights clients by entropy** — when data is IID, entropy is uniform and EWA ≈ FedAvg.
        When data is extremely Non-IID, entropy correctly identifies specialist clients and upweights them.
        The "effective interval" is where EWA's signal exceeds noise.

        ### Hypothesis: Organoid Long-tail → Natural Non-IID

        The Intestinal Organoid dataset's 5.3:1 class ratio creates natural Non-IID:
        - Client A (organoid0 specialist) has **high confidence** on its dominant class
        - Client C (spheroid specialist) has **rare but valuable** minority knowledge
        - EWA should automatically weight Client C higher for spheroid detection

        This is a stronger test than the PCB experiments because the long-tail is **intrinsic to the data**,
        not artificially created by sampling.
        """)

        p2 = load_phase2_results()
        if p2.get("convergence"):
            conv = p2["convergence"]

            # EWA Interval Cards
            int_cols = st.columns(3)
            ewa_interval = [
                ("IID Balanced", "iid_balanced", "⚠️ Harmful"),
                ("Moderate Non-IID", "moderate_balanced", "⚠️ Neutral"),
                ("Extreme Non-IID", "extreme_extreme", "✅ Beneficial"),
            ]
            for i, (label, scenario, verdict) in enumerate(ewa_interval):
                fedavg_match = [r for r in conv if r["scenario"] == scenario and r["strategy"] == "FedAvg"]
                ewa_match = [r for r in conv if r["scenario"] == scenario and r["strategy"] == "EWA-v2"]
                if fedavg_match and ewa_match:
                    delta = float(ewa_match[0]["final_mAP"]) - float(fedavg_match[0]["final_mAP"])
                    with int_cols[i]:
                        st.metric(
                            label=label,
                            value=f"{delta:+.4f}",
                            delta=f"{delta*100:+.1f}pp vs FedAvg",
                        )
                        st.caption(verdict)

        # Visual: Expected organoid EWA behavior
        st.markdown("### 📊 Expected EWA Behavior on Organoid Data")
        st.markdown("""
        ```
        Non-IID Level:  IID ─── Mild ─── Moderate ─── Extreme
        EWA Benefit:     -5.7pp   -2.1pp    -1.3pp     +1.1pp
                                       ↑
                               Organoid 5.3:1 long-tail
                               should land here or right
        ```

        **Prediction**: The organoid dataset's natural long-tail (5.3:1) places it in the
        "Moderate to Extreme" Non-IID range where EWA should show **positive benefit**.
        This would validate EWA's generalizability beyond industrial defect detection.
        """)

    # ══════════════════════════════════════════════════════════════
    # Tab 4: Methodology
    # ══════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("""
        ### FedAvg (McMahan et al., 2017)

        1. **Local Training**: Each client $k$ trains on local data for $E$ epochs:
           $$w_k^{t+1} = w_k^t - \\eta \\nabla F_k(w_k^t)$$

        2. **Aggregation**: Server averages client updates:
           $$w^{t+1} = \\sum_{k=1}^{K} \\frac{|\\mathcal{D}_k|}{|\\mathcal{D}|} w_k^{t+1}$$

        ### FedProx (Li et al., 2020)

        Adds proximal term to prevent client drift:
        $$\\min_w F_k(w) + \\frac{\\mu}{2} \\|w - w^t\\|^2$$

        Approximated via weight interpolation: $w_k^{t+1} = (1-\\mu) w_k^{local} + \\mu w^t$

        ### EWA-v2: Entropy-Weighted Aggregation

        Weights clients by inverse prediction entropy (confidence):
        $$\\alpha_k = \\frac{\\exp(-H_k / \\tau)}{\\sum_j \\exp(-H_j / \\tau)}$$

        Where $H_k = -\\sum_c p_{k,c} \\log p_{k,c}$ is the prediction entropy on a held-out set.

        **Key insight**: Specialist clients have low entropy (high confidence) → higher weight.
        This is most effective when clients have complementary expertise (extreme Non-IID).

        ### Non-IID Client Design (Organoid)

        | Client | Dominant Classes | Ratio | Specialty |
        |--------|-----------------|-------|-----------|
        | A | organoid0 | 80% | Cystic specialist |
        | B | organoid1 + organoid3 | 80% | Maturing specialist |
        | C | spheroid | 80% | Spheroid specialist |

        This creates moderate-to-extreme Non-IID that should fall within EWA's effective interval.
        """)

        st.markdown("---")
        st.markdown("### References")
        for key, ref in REFERENCES.items():
            st.markdown(f"- **{key}**: {ref}")
