# ── modules/surf_2026.py ──
"""
SURF 2026 Summer Research Projects
===================================
SIPPO-funded 18-student summer research program organized in three tracks:
- Track A: Organoid Image Analysis (5 students)
- Track B: Vector Quality Assessment (5 students)
- Track C: MPM Biomarker Database (6 students)
- Track D: Other (2 students)

Each project includes: specification, mentor, deliverable, integration with
organoid-fl platform, and contribution to the three-grant pipeline:
  SURF 2026 (proxy) → Ruijin Medical-Engineering (20 real MPM) → NSFC (100-center)
"""

import streamlit as st
import json
import os
from pathlib import Path


# ── Project catalog ─────────────────────────────────────────────────────────
PROJECTS = [
    # ── Track A: Organoid Image Analysis (5 students) ──
    {
        "id": "P1",
        "track": "A",
        "title": "Mouse liver organoid detection: RF-DETR vs YOLOv12",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 3,
        "duration_weeks": 8,
        "deliverables": [
            "Trained RF-DETR + YOLOv12 models on mouse_liver_split",
            "Cross-validation on b1/b2/b3 batches",
            "Written report comparing accuracy/speed/params",
        ],
        "platform_integration": "results/mouse_liver_v2/ → Streamlit 'Detection' tab",
        "nsfc_link": "Provides baseline for organoid detection module (Paper Section 3.1)",
        "status": "open",
    },
    {
        "id": "P6",
        "track": "A",
        "title": "MultiOrg cross-lab organoid detection benchmark",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 4,
        "duration_weeks": 8,
        "deliverables": [
            "YOLOv12s + freebies training at imgsz=1088",
            "SAHI sliding-window inference + Soft-NMS",
            "SOTA comparison table (vs Deliod, Orga-Dete, MultiOrg SSD)",
        ],
        "platform_integration": "results/multiorg_v5/ → 'Detection' tab + SOTA tracker",
        "nsfc_link": "Cross-lab generalization evidence (Paper Section 5.2)",
        "status": "open",
    },
    {
        "id": "P-A3",
        "track": "A",
        "title": "CLIP zero-shot cross-domain organoid classifier",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 4,
        "duration_weeks": 8,
        "deliverables": [
            "CLIP ViT-L/14 zero-shot on 4 organoid datasets",
            "Prompt engineering comparison (5 prompt templates)",
            "LOO cross-domain AUC report",
        ],
        "platform_integration": "results/clip_zeroshot/ → 'Feature Space' tab",
        "nsfc_link": "Cross-domain generalization without training (Paper Section 5.3)",
        "status": "open",
        "custom": True,
    },
    {
        "id": "P-A4",
        "track": "A",
        "title": "SAM2 self-distillation fine-tuning for MPM",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 5,
        "duration_weeks": 8,
        "deliverables": [
            "SAM2 zero-shot mask generation on 34 MPM patches",
            "Self-distillation: SAM2 pseudo-GT → classifier head fine-tune",
            "Performance comparison vs from-scratch RF-DETR",
        ],
        "platform_integration": "results/sam2_mpm/ → 'Segmentation' tab + 'Research' tab",
        "nsfc_link": "Direct Ruijin deliverable, demonstrates 10-20 sample viability",
        "status": "open",
        "custom": True,
    },
    {
        "id": "P-A5",
        "track": "A",
        "title": "Visual primitives framework: Phase 1-5 ablation",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 5,
        "duration_weeks": 8,
        "deliverables": [
            "Phase 1 perception + Phase 2 VLM + Phase 3 HNM + Phase 4 SupCon + Phase 5 FL",
            "Ablation table showing contribution of each phase",
            "Position paper draft (8 pages)",
        ],
        "platform_integration": "Full integration with 'FL Training' + 'Research' tabs",
        "nsfc_link": "Core methodology paper (Paper Section 4)",
        "status": "open",
        "custom": True,
    },
    # ── Track B: Vector Quality Assessment (5 students) ──
    {
        "id": "P7",
        "track": "B",
        "title": "HNSW vs FAISS benchmark on organoid embeddings",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 3,
        "duration_weeks": 8,
        "deliverables": [
            "Recall@k / latency / memory comparison",
            "Rust HNSW vs Python FAISS benchmark script",
            "Stress test with 100K embeddings",
        ],
        "platform_integration": "→ 'Vector Search' tab benchmark data",
        "nsfc_link": "FedCtx vector DB validation (Paper Section 6.2)",
        "status": "open",
    },
    {
        "id": "P10",
        "track": "B",
        "title": "Vector quality scoring: HNSW neighborhood consistency",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 4,
        "duration_weeks": 8,
        "deliverables": [
            "Quality score definition (k-NN entropy + modularity)",
            "Per-dataset quality report (4 organoid datasets)",
            "Visualization of low-quality vs high-quality embeddings",
        ],
        "platform_integration": "→ 'Vector Search' tab quality sub-tab",
        "nsfc_link": "Vector quality assurance for FedCtx (Paper Section 6.3)",
        "status": "open",
    },
    {
        "id": "P-B3",
        "track": "B",
        "title": "Cross-modal embedding alignment (image + gene + drug)",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 5,
        "duration_weeks": 8,
        "deliverables": [
            "Multi-modal contrastive learning (CLIP-style) on organoid + gene + drug",
            "Cross-modal retrieval demo (text → image, drug → organoid)",
            "Alignment quality metrics",
        ],
        "platform_integration": "→ 'Vision RAG' tab + 'Vector Search' tab",
        "nsfc_link": "Direct Ruijin deliverable, 3-modal alignment quality",
        "status": "open",
        "custom": True,
    },
    {
        "id": "P-B4",
        "track": "B",
        "title": "FedCtx federated semantic search stress test",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 4,
        "duration_weeks": 8,
        "deliverables": [
            "Multi-client FedCtx deployment script",
            "Latency / accuracy under 5-client federation",
            "Network partition tolerance test",
        ],
        "platform_integration": "→ 'FL Training' tab + 'Vector Search' tab",
        "nsfc_link": "Federated infrastructure validation (Paper Section 6)",
        "status": "open",
        "custom": True,
    },
    {
        "id": "P-B5",
        "track": "B",
        "title": "Embedding visualization dashboard (UMAP + quality heatmap)",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 3,
        "duration_weeks": 8,
        "deliverables": [
            "Interactive UMAP plot with hover metadata",
            "Quality heatmap per dataset",
            "Streamlit integration module",
        ],
        "platform_integration": "→ New 'Vector Viz' sub-tab under 'Feature Space' tab",
        "nsfc_link": "Vector quality visualization (Paper Section 6.4)",
        "status": "open",
        "custom": True,
    },
    # ── Track C: MPM Biomarker Database (6 students) ──
    {
        "id": "P13",
        "track": "C",
        "title": "MPM literature mining: biomarker database construction",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 3,
        "duration_weeks": 8,
        "deliverables": [
            "PubMed + Semantic Scholar API scraping",
            "Biomarker database (CSV/SQL): 200+ markers, 50+ papers",
            "Streamlit browse interface",
        ],
        "platform_integration": "→ New 'MPM Markers' tab",
        "nsfc_link": "MPM biomarker database foundation (Paper Section 7)",
        "status": "open",
    },
    {
        "id": "P14",
        "track": "C",
        "title": "MPM clinical data integration: TCGA-MESO + TCIA",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 4,
        "duration_weeks": 8,
        "deliverables": [
            "TCGA-MESO API integration (87 cases)",
            "TCIA WSI download script (3446 images, CC BY 4.0)",
            "De-identification + ICD-10 mapping",
        ],
        "platform_integration": "→ 'Data Explorer' tab + 'MPM Markers' tab",
        "nsfc_link": "Public dataset baseline (Paper Section 7.1)",
        "status": "open",
    },
    {
        "id": "P17-A",
        "track": "C",
        "title": "MPM H&E WSI patch extraction + quality filter",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 4,
        "duration_weeks": 8,
        "deliverables": [
            "WSI patch extraction at 256x / 512x",
            "Tissue vs background classifier",
            "3446 WSI → ~500K patches pipeline",
        ],
        "platform_integration": "→ 'Data Explorer' tab MPM section",
        "nsfc_link": "WSI baseline dataset (Paper Section 7.2)",
        "status": "open",
    },
    {
        "id": "P17-B",
        "track": "C",
        "title": "MesoGraph paper reproduction + comparison",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 4,
        "duration_weeks": 8,
        "deliverables": [
            "MesoGraph open-source code reproduction",
            "Comparison with our RF-DETR pipeline",
            "Performance + interpretability report",
        ],
        "platform_integration": "→ 'Model Analysis' tab SOTA comparison",
        "nsfc_link": "SOTA comparison (Paper Section 7.3)",
        "status": "open",
    },
    {
        "id": "P-C5",
        "track": "C",
        "title": "MPM class activation map (CAM) explainability",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 3,
        "duration_weeks": 8,
        "deliverables": [
            "Grad-CAM on RF-DETR predictions",
            "Comparison: TP vs FP activation patterns",
            "Clinical interpretability report",
        ],
        "platform_integration": "→ 'Explainability' tab MPM section",
        "nsfc_link": "Clinical AI interpretability (Paper Section 7.4)",
        "status": "open",
        "custom": True,
    },
    {
        "id": "P-C6",
        "track": "C",
        "title": "MPM LLM-as-Judge: VLM quality assessment",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 4,
        "duration_weeks": 8,
        "deliverables": [
            "GLM-4V / GPT-4o prompt template library (5 templates)",
            "Quality scoring on 34 MPM patches",
            "Expert comparison (3 clinical doctors)",
        ],
        "platform_integration": "→ New 'VLM Judge' sub-tab under 'Explainability'",
        "nsfc_link": "Direct Ruijin deliverable, LLM-as-Judge validation (Paper Section 8)",
        "status": "open",
        "custom": True,
    },
    # ── Track D: Other (2 students) ──
    {
        "id": "P-D1",
        "track": "D",
        "title": "Federated learning privacy attack simulation",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 5,
        "duration_weeks": 8,
        "deliverables": [
            "Gradient inversion attack (DLG / iDLG)",
            "Membership inference attack",
            "Defense mechanism: differential privacy + gradient compression",
        ],
        "platform_integration": "→ New 'Privacy Audit' tab",
        "nsfc_link": "Privacy attack evidence (Paper Section 8.2)",
        "status": "open",
        "custom": True,
    },
    {
        "id": "P-D2",
        "track": "D",
        "title": "Survey: Agentic AI in biomedical research",
        "student": "TBD",
        "mentor": "Dechang",
        "difficulty": 3,
        "duration_weeks": 8,
        "deliverables": [
            "49-reference survey paper draft",
            "Bibliography + citation audit",
            "Submission-ready manuscript",
        ],
        "platform_integration": "→ 'Research' tab bibliography",
        "nsfc_link": "Methodology paper (companion)",
        "status": "in_progress",
        "custom": True,
    },
]

TRACK_INFO = {
    "A": {
        "name": "Track A: Organoid Image Analysis",
        "color": "#3b82f6",
        "icon": "🧬",
        "lead": "Dechang + Sarah Best (advisory)",
        "slots": 5,
    },
    "B": {
        "name": "Track B: Vector Quality Assessment",
        "color": "#10b981",
        "icon": "🔍",
        "lead": "Dechang",
        "slots": 5,
    },
    "C": {
        "name": "Track C: MPM Biomarker Database",
        "color": "#f59e0b",
        "icon": "🩺",
        "lead": "Dechang + Xie Mengyan (advisory)",
        "slots": 6,
    },
    "D": {
        "name": "Track D: Other / Open Topics",
        "color": "#8b5cf6",
        "icon": "📚",
        "lead": "Dechang",
        "slots": 2,
    },
}


def render():
    st.markdown(
        '<div class="main-header"><h1>🎓 SURF 2026 Summer Research</h1>'
        '<p>18 students × 8 weeks · 3 tracks · Integrated with organoid-fl platform</p></div>',
        unsafe_allow_html=True,
    )

    # ── Top-level overview ─────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    track_a_count = sum(1 for p in PROJECTS if p["track"] == "A")
    track_b_count = sum(1 for p in PROJECTS if p["track"] == "B")
    track_c_count = sum(1 for p in PROJECTS if p["track"] == "C")
    track_d_count = sum(1 for p in PROJECTS if p["track"] == "D")
    col_a.metric("Track A · Organoid", f"{track_a_count} / 5", "Image Analysis")
    col_b.metric("Track B · Vector", f"{track_b_count} / 5", "Quality Assessment")
    col_c.metric("Track C · MPM", f"{track_c_count} / 6", "Biomarker Database")
    col_d.metric("Track D · Open", f"{track_d_count} / 2", "Surveys + Attacks")

    st.markdown("---")

    # ── Three-grant pipeline banner ─────────────────────────────────────
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1e3a8a 0%, #065f46 50%, #92400e 100%);
                    padding: 1.2rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
          <h3 style="margin: 0 0 0.4rem 0;">🔗 Three-Grant Pipeline</h3>
          <div style="font-size: 0.95rem; opacity: 0.95;">
            <b>SURF 2026</b> (proxy validation, this program) →
            <b>Ruijin Medical-Engineering 2026-27</b> (20 real MPM PDO) →
            <b>NSFC General Program 2027-30</b> (100-center multi-site)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Filter ──────────────────────────────────────────────────────────
    filter_col, status_col, _ = st.columns([2, 2, 3])
    track_filter = filter_col.selectbox(
        "Filter by track",
        ["All", "A: Organoid", "B: Vector", "C: MPM", "D: Open"],
        index=0,
    )
    status_filter = status_col.selectbox(
        "Filter by status", ["All", "open", "in_progress", "done"], index=0
    )

    filtered = PROJECTS
    if track_filter != "All":
        track_key = track_filter[0]
        filtered = [p for p in filtered if p["track"] == track_key]
    if status_filter != "All":
        filtered = [p for p in filtered if p["status"] == status_filter]

    # ── Project cards ───────────────────────────────────────────────────
    st.markdown(f"### 📋 {len(filtered)} Project(s)")
    for proj in filtered:
        track_info = TRACK_INFO[proj["track"]]
        with st.container(border=True):
            header_col, meta_col, diff_col = st.columns([5, 3, 2])
            header_col.markdown(
                f"#### {track_info['icon']} {proj['id']}: {proj['title']}"
            )
            meta_col.markdown(
                f"Mentor: **{proj['mentor']}** · Duration: {proj['duration_weeks']} weeks"
            )
            diff_col.metric("Difficulty", "★" * proj["difficulty"])

            # Custom tag
            if proj.get("custom"):
                st.caption("Custom topic (designed by Dechang, not in original SIPPO list)")

            # Deliverables
            st.markdown("**Deliverables**")
            for d in proj["deliverables"]:
                st.markdown(f"- {d}")

            # Integration
            st.info(f"🔗 Platform integration: {proj['platform_integration']}")

            # NSFC link
            st.success(f"🎯 NSFC contribution: {proj['nsfc_link']}")

            # Status
            status_emoji = {"open": "🟡", "in_progress": "🔵", "done": "🟢"}.get(
                proj["status"], "⚪"
            )
            st.markdown(f"**Status**: {status_emoji} {proj['status']}")

    # ── Gap experiments ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Gap Experiments (NSFC must-do, not covered by SURF)")

    gap_data = [
        {
            "id": "A2",
            "topic": "Diffusion synthetic data validation",
            "metric": "FID-medical < 40, downstream mAP ≥ 80%",
            "owner": "Dechang (local GPU needed)",
            "deadline": "2027 Q1",
        },
        {
            "id": "C2",
            "topic": "Privacy attack empirical test (DLG/iDLG)",
            "metric": "Reconstruction PSNR baseline + defense gain",
            "owner": "P-D1 student + Dechang",
            "deadline": "2027 Q2",
        },
        {
            "id": "C3",
            "topic": "Communication efficiency stress test",
            "metric": "5-client federation, bandwidth vs accuracy",
            "owner": "P-B4 student",
            "deadline": "2027 Q2",
        },
        {
            "id": "D2",
            "topic": "Clinical validation design (Ruijin 20 cases)",
            "metric": "Trial protocol + IRB application",
            "owner": "Xie Mengyan + Dechang",
            "deadline": "2026 Q4",
        },
    ]
    st.dataframe(gap_data, use_container_width=True, hide_index=True)

    # ── Timeline ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 8-Week Timeline (all projects)")

    timeline = [
        ("Week 1-2", "Onboarding + literature review + setup"),
        ("Week 3-4", "Mid-term check: data preparation + initial experiments"),
        ("Week 5-6", "Core experiments + Streamlit integration"),
        ("Week 7-8", "Final report + presentation + paper draft (top students)"),
    ]
    for week, desc in timeline:
        st.markdown(f"- **{week}**: {desc}")

    # ── Deliverables to grants ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 How SURF 2026 outputs feed into Ruijin + NSFC grants")

    st.markdown(
        """
        | SURF 2026 Output | Ruijin Use | NSFC Use |
        |---|---|---|
        | P1 Mouse liver detection | Algorithm baseline | Multi-organ validation |
        | P6 MultiOrg SOTA | Cross-lab evidence | 100-center scaling |
        | P-A3 CLIP zero-shot | Cross-domain method | Privacy-preserving inference |
        | P-A4 SAM2 self-distillation | Direct Ruijin deliverable | MPM 100-case extension |
        | P-A5 Visual primitives | Methodology paper | Agentic AI framework |
        | P-B3 Cross-modal alignment | 3-modal quality score | Drug-organoid matching |
        | P-C5 MPM Grad-CAM | Clinical interpretability | Regulatory submission |
        | P-C6 VLM-as-Judge | Direct Ruijin LLM-as-Judge | Multi-center QA |
        | P-D1 Privacy attacks | DP defense evidence | Privacy claim validation |
        | P-D2 Agentic AI survey | Methodology companion | Companion paper |
        """
    )

    # ── SAM2 Self-Distillation 3-tier evidence chain ─────────────────────
    st.markdown("---")
    st.markdown("### 🔬 SAM2 自蒸馏三层证据链 (2026-07-16)")

    st.markdown(
        """
        **方法**：用 SAM2 zero-shot mask 作独立信号训练分类头，避免 RF-DETR conf 循环论证

        | 实验 | 样本数 | Zero-shot AUC | Distilled AUC | 提升 |
        |---|---|---|---|---|
        | Simulation v2 (Intestinal) | 700 | 0.6616 | **0.9437** | **+28.22%** |
        | 真实跨域 50+30 | 10578 | 0.7480 | **0.8857** | **+13.77%** |
        | 真实跨域 200+100 | 40503 | 0.7679 | **0.8589** | **+9.10%** |
        | ResNet50 200+84 | 11974 | 0.9156 | 0.9281 | +1.25% |
        | MPM 试点 (34 patch) | 3131 | 0.9779 | 1.0000 | +2.21% |
        """
    )

    st.markdown("#### 🎯 自适应蒸馏策略（论文创新点）")
    st.markdown(
        """
        **关键发现**：传统 `distilled = conf × classifier_prob` 不是最优！
        当 RF-DETR conf 跨域失效时，conf 是噪声，乘 classifier 反而拉低性能。

        | Zero-shot AUC | 策略 | 公式 | 提升 |
        |---|---|---|---|
        | **< 0.70** | Classifier alone | `distilled = classifier_prob` | **+31.76%** |
        | 0.70 - 0.85 | Distilled | `conf × classifier_prob` | **+9.10%** |
        | > 0.85 | Zero-shot | `distilled = conf` | maintain |

        **四实验对比**：Classifier alone 在 2/3 实验中胜出 distilled (conf×cls)
        """
    )

    st.markdown("#### 📊 ResNet50 真特征 vs 图像统计")
    st.markdown(
        """
        | 特征 | 维度 | Classifier alone AUC | 结论 |
        |---|---|---|---|
        | 图像统计 | 30 | 0.8202 | distilled 赢 (+9.10%) |
        | **ResNet50** | **2048** | **0.9938** | **classifier alone 赢** |

        ResNet50 真特征单独分类 AUC=0.9938（接近完美），但 distilled 0.9281 反而更低
        ——conf 乘法压低了高置信度样本。
        """
    )

    st.markdown("#### 📁 实验代码与结果")
    st.markdown(
        """
        - `scripts/mpm/sam2_self_distillation.py` — MPM 34 patch 原始脚本
        - `scripts/mpm/self_distillation_intestinal_sim.py` — Simulation v2
        - `scripts/mpm/self_distillation_intestinal_real.py` — 真实跨域 RF-DETR
        - `scripts/mpm/self_distillation_intestinal_resnet.py` — ResNet50 真特征
        - `scripts/mpm/comprehensive_analysis.py` — 四实验综合对比
        - `scripts/mpm/adaptive_strategy_analysis.py` — 自适应策略分析
        - `results/sd_comprehensive_analysis/` — 综合分析报告
        """
    )
