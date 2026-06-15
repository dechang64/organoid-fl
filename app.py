"""
Organoid-FL — 类器官智能分析平台
Streamlit Cloud App

Features:
    - Upload microscopy image
    - AI organoid detection (healthy / early_stage / late_stage)
    - Mock SAM2 segmentation with morphology metrics
    - Federated learning simulation across hospitals
    - Convergence visualization
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
import os

from analysis.detector import OrganoidDetector, Detection
from analysis.segmentor import OrganoidSegmentor, SegmentationResult
from analysis.fl_engine import FLEngine as OrganoidFL
from utils.constants import (
    ORGANOID_CLASSES, CLASS_INFO, DEFAULT_ROUNDS, DEFAULT_CLIENTS,
    DEFAULT_LR, DEFAULT_BATCH_SIZE, DEFAULT_LOCAL_EPOCHS, COLORS,
    DEFAULT_INPUT_DIM, DEFAULT_HIDDEN_DIM,
)
from utils.helpers import generate_synthetic_features, split_data_non_iid


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Organoid-FL · 类器官智能分析",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        color: #059669;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #666;
    }
    .class-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin: 0.25rem;
        font-weight: 600;
    }
    .badge-healthy { background: #22c55e; color: white; }
    .badge-early { background: #f59e0b; color: white; }
    .badge-late { background: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
if "detector" not in st.session_state:
    st.session_state.detector = OrganoidDetector(mode="mock")
if "segmentor" not in st.session_state:
    st.session_state.segmentor = OrganoidSegmentor(mode="mock")
if "last_detections" not in st.session_state:
    st.session_state.last_detections = []
if "last_image" not in st.session_state:
    st.session_state.last_image = None


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=64)
    st.title("类器官分析")
    st.caption("Organoid-FL v0.1")

    st.divider()

    st.subheader("⚙️ 设置")
    detect_mode = st.selectbox(
        "检测模式",
        ["mock (快速演示)", "yolo (需要GPU)"],
        index=0,
    )
    st.session_state.detector = OrganoidDetector(
        mode="mock" if "mock" in detect_mode else "yolo"
    )

    seg_mode = st.selectbox(
        "分割模式",
        ["mock (快速演示)", "sam2 (需要GPU)"],
        index=0,
    )
    st.session_state.segmentor = OrganoidSegmentor(
        mode="mock" if "mock" in seg_mode else "sam2"
    )

    st.divider()

    st.subheader("🏥 实验室信息")
    lab_id = st.text_input("实验室编号", value="lab_bwh_01", placeholder="e.g., lab_bwh_01")
    institution = st.selectbox(
        "机构",
        ["Brigham and Women's Hospital", "Mayo Clinic", "Johns Hopkins", "Custom"],
        index=0,
    )
    passage = st.number_input("传代数", value=5, min_value=1, max_value=50)
    cell_line = st.text_input("细胞系", value="iPSC-derived", placeholder="e.g., iPSC-derived")


# ============================================================
# Main Page
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🧬 Organoid-FL</h1>
    <p>基于联邦学习的多中心类器官智能分析平台</p>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_segment, tab_fl, tab_about = st.tabs([
    "🔍 检测", "📐 分割", "🤝 联邦学习", "ℹ️ 关于"
])


# ============================================================
# Tab 1: Detection
# ============================================================
with tab_detect:
    st.header("类器官智能检测")
    st.caption("支持3类状态：Healthy（健康）、Early Stage（早期分化）、Late Stage（晚期分化）")

    uploaded = st.file_uploader(
        "上传显微镜图像",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="detect_upload",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("原始图像")
            st.image(img, use_container_width=True)
            st.caption(f"尺寸: {img.width} × {img.height} px")

        with col2:
            with st.spinner("🔍 正在检测类器官..."):
                t0 = time.time()
                detections = st.session_state.detector.detect(img_array, lab_id)
                dt = (time.time() - t0) * 1000

            # Draw detections
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)

            CLASS_COLORS = {
                "healthy": "#22c55e",
                "early_stage": "#f59e0b",
                "late_stage": "#ef4444",
            }

            for d in detections:
                color = CLASS_COLORS.get(d.class_name, "#888")
                draw.rectangle([d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]], outline=color, width=2)
                label = f"{d.class_name} {d.confidence:.0%}"
                draw.rectangle([d.bbox[0], d.bbox[1]-16, d.bbox[0]+len(label)*9+6, d.bbox[1]], fill=color)
                draw.text((d.bbox[0]+3, d.bbox[1]-14), label, fill="white")

            st.subheader("检测结果")
            st.image(annotated, use_container_width=True)
            st.caption(f"检测耗时: {dt:.0f}ms")

        # Store for segmentation tab
        st.session_state.last_detections = detections
        st.session_state.last_image = img_array

        # Summary
        summary = st.session_state.detector.summary(detections)
        st.divider()
        st.subheader("📊 检测统计")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("检测总数", summary["total"])
        with m2:
            st.metric("平均置信度", f"{summary['avg_confidence']:.1%}")
        with m3:
            st.metric("平均面积", f"{summary['avg_area']:.0f} px²")
        with m4:
            st.metric("面积范围", f"{summary['min_area']:.0f} - {summary['max_area']:.0f}")

        # Class breakdown
        if summary["classes"]:
            st.subheader("分类分布")
            for cls_name, count in summary["classes"].items():
                info = CLASS_INFO.get(cls_name, {})
                emoji = info.get("emoji", "⚪")
                label = info.get("label", cls_name)
                color = info.get("color", "#888")
                st.markdown(f"{emoji} **{label}**: {count} 个")

        # Detection table
        if detections:
            st.subheader("详细结果")
            table_data = []
            for i, d in enumerate(detections):
                table_data.append({
                    "#": i + 1,
                    "类型": d.class_name,
                    "置信度": f"{d.confidence:.1%}",
                    "中心": f"({d.cx:.0f}, {d.cy:.0f})",
                    "面积": f"{d.area:.0f} px²",
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("请上传显微镜图像开始检测")


# ============================================================
# Tab 2: Segmentation
# ============================================================
with tab_segment:
    st.header("类器官像素级分割")
    st.caption("基于SAM2的精确分割，提取形态学指标")

    if not st.session_state.last_detections:
        st.warning("请先在「检测」页面上传图像并完成检测")
    else:
        img_array = st.session_state.last_image
        detections = st.session_state.last_detections

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("原始图像 + 检测框")
            annotated = Image.fromarray(img_array)
            draw = ImageDraw.Draw(annotated)
            CLASS_COLORS = {"healthy": "#22c55e", "early_stage": "#f59e0b", "late_stage": "#ef4444"}
            for d in detections:
                color = CLASS_COLORS.get(d.class_name, "#888")
                draw.rectangle([d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]], outline=color, width=2)
            st.image(annotated, use_container_width=True)

        with col2:
            with st.spinner("📐 正在分割类器官..."):
                t0 = time.time()
                seg_results = st.session_state.segmentor.segment(img_array, detections)
                dt = (time.time() - t0) * 1000

            # Create segmentation overlay
            overlay = img_array.copy()
            for i, seg in enumerate(seg_results):
                mask = seg.mask if hasattr(seg, 'mask') else np.zeros(img_array.shape[:2], dtype=np.uint8)
                color_map = {
                    "healthy": np.array([34, 197, 94]),
                    "early_stage": np.array([245, 158, 11]),
                    "late_stage": np.array([239, 68, 68]),
                }
                color = color_map.get(seg.defect_type if hasattr(seg, 'defect_type') else "healthy", np.array([100, 100, 255]))
                overlay[mask > 0] = (overlay[mask > 0] * 0.5 + color * 0.5).astype(np.uint8)

            st.subheader("分割结果")
            st.image(overlay, use_container_width=True)
            st.caption(f"分割耗时: {dt:.0f}ms | 分割数: {len(seg_results)}")

        # Morphology table
        if seg_results:
            st.subheader("📊 形态学指标")
            morph_data = []
            for i, seg in enumerate(seg_results):
                morph_data.append({
                    "#": i + 1,
                    "类型": seg.defect_type if hasattr(seg, 'defect_type') else "unknown",
                    "面积": f"{seg.area} px²",
                    "周长": f"{seg.perimeter:.1f} px",
                    "圆度": f"{seg.circularity:.3f}",
                    "实心度": f"{seg.solidity:.3f}",
                    "长宽比": f"{seg.aspect_ratio:.2f}",
                    "离心率": f"{seg.eccentricity:.3f}",
                })
            st.dataframe(morph_data, use_container_width=True, hide_index=True)

            # Morphology interpretation
            st.subheader("🔬 形态学解读")
            for i, seg in enumerate(seg_results):
                cls = seg.defect_type if hasattr(seg, 'defect_type') else "unknown"
                info = CLASS_INFO.get(cls, {})
                emoji = info.get("emoji", "⚪")
                label = info.get("label", cls)

                interpretations = []
                if seg.circularity > 0.8:
                    interpretations.append("高圆度 → 接近球形，结构规整")
                elif seg.circularity < 0.5:
                    interpretations.append("低圆度 → 不规则形态，可能异常")
                if seg.solidity > 0.9:
                    interpretations.append("高实心度 → 边缘光滑")
                elif seg.solidity < 0.7:
                    interpretations.append("低实心度 → 边缘凹陷，可能出芽")
                if seg.aspect_ratio > 1.5:
                    interpretations.append("高长宽比 → 拉伸形态")

                st.markdown(f"**{emoji} #{i+1} {label}**: {'; '.join(interpretations) if interpretations else '形态正常'}")


# ============================================================
# Tab 3: Federated Learning
# ============================================================
with tab_fl:
    st.header("联邦学习模拟")
    st.caption("多中心类器官分类联邦训练（FedAvg）")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ 训练参数")
        n_rounds = st.number_input("联邦轮数", value=DEFAULT_ROUNDS, min_value=1, max_value=50)
        n_clients = st.number_input("医院数量", value=DEFAULT_CLIENTS, min_value=2, max_value=10)
        lr = st.number_input("学习率", value=DEFAULT_LR, min_value=0.0001, max_value=0.1, format="%.4f")
        local_epochs = st.number_input("本地轮数", value=DEFAULT_LOCAL_EPOCHS, min_value=1, max_value=10)
        batch_size = st.number_input("批大小", value=DEFAULT_BATCH_SIZE, min_value=8, max_value=128)
        n_samples = st.number_input("每院样本数", value=600, min_value=100, max_value=5000, step=100)

        run_fl = st.button("🚀 开始训练", type="primary", use_container_width=True)

    with col2:
        if run_fl:
            with st.spinner("🤝 联邦训练中..."):
                # Generate synthetic data
                features, labels, class_names = generate_synthetic_features(
                    n_samples=n_samples * n_clients,
                    dim=DEFAULT_INPUT_DIM,
                    n_classes=3,
                )

                # Run FL
                engine = OrganoidFL(
                    input_dim=DEFAULT_INPUT_DIM,
                    num_classes=3,
                    hidden_dim=DEFAULT_HIDDEN_DIM,
                    lr=lr,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                )

                progress = st.empty()
                def on_progress(rnd, metrics):
                    progress.text(f"轮次 {rnd}/{n_rounds} — 验证准确率: {metrics['val_acc']:.1%}")

                history = engine.run(
                    features=features,
                    labels=labels,
                    n_clients=n_clients,
                    rounds=n_rounds,
                    progress_callback=on_progress,
                )

            st.success("✅ 训练完成！")

            # Metrics
            final = history[-1]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("最终训练准确率", f"{final['avg_train_acc']:.1%}")
            with m2:
                st.metric("最终验证准确率", f"{final['val_acc']:.1%}")
            with m3:
                st.metric("最终验证损失", f"{final['val_loss']:.3f}")
            with m4:
                st.metric("联邦轮数", n_rounds)

            # Convergence chart
            st.subheader("📈 收敛曲线")
            rounds = [h["round"] for h in history]
            train_acc = [h["avg_train_acc"] for h in history]
            val_acc = [h["val_acc"] for h in history]
            train_loss = [h["avg_train_loss"] for h in history]
            val_loss = [h["val_loss"] for h in history]

            chart_data = {
                "轮次": rounds,
                "训练准确率": train_acc,
                "验证准确率": val_acc,
                "训练损失": train_loss,
                "验证损失": val_loss,
            }
            st.line_chart(chart_data, x="轮次", y=["训练准确率", "验证准确率"])

            # Per-client breakdown
            st.subheader("🏥 各医院训练详情")
            for cm in final["client_metrics"]:
                st.markdown(f"""
                **医院 {cm['client'] + 1}** — 损失: {cm['train_loss']:.3f} | 准确率: {cm['train_acc']:.1%}
                """)
        else:
            st.info("配置参数后点击「开始训练」")


# ============================================================
# Tab 4: About
# ============================================================
with tab_about:
    st.header("关于 Organoid-FL")

    st.markdown("""
    ### 🧬 项目简介

    **Organoid-FL** 是一个基于联邦学习的多中心类器官智能分析平台，实现跨医院协同训练，医学图像数据不出院。

    ### 🔬 核心技术

    | 技术 | 用途 |
    |------|------|
    | YOLOv11 | 类器官检测与分类 |
    | SAM2 | 像素级精确分割 |
    | DINOv2 | 768维特征提取 |
    | Grad-CAM | 模型可解释性 |
    | FedAvg | 多中心联邦聚合 |
    | HNSW | 相似类器官检索 |

    ### 🏥 分类能力

    | 类型 | 标签 | 描述 |
    |------|------|------|
    | 🟢 Healthy | 健康 | 正常类器官形态 |
    | 🟡 Early Stage | 早期分化 | 初步分化迹象 |
    | 🔴 Late Stage | 晚期分化 | 明显分化/异常 |

    ### 📊 形态学指标

    - 面积（像素数）
    - 周长（边界长度）
    - 圆度（4π·面积/周长²）
    - 实心度（面积/凸包面积）
    - 长宽比（长短轴比）
    - 离心率（0=圆，1=线）

    ### 📄 开源协议

    Apache-2.0 | [GitHub](https://github.com/dechang64)
    """)
