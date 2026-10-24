"""HITL 标注流水线 MVP —— Streamlit 嵌入页。

把 mvp/hitl-demo/ 的单文件 HTML 通过 components.html（iframe srcdoc）
原样嵌入 Streamlit。纯静态内容，无任何 Python 侧依赖（除 streamlit 本身）。

用法一（并入 app.py 的 radio 分发）：
    from modules import hitl_mvp
    elif page == "🧪 HITL MVP":
        hitl_mvp.render()

用法二（独立运行调试）：
    streamlit run mvp/hitl_demo_streamlit.py
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_HTML_PATH = Path(__file__).resolve().parent.parent / "mvp" / "hitl-demo" / "类器官标注平台_HITL_MVP_单文件.html"

# 演示阶段 → URL hash（app.js 的 #auto= 驱动器）
_STAGES = {
    "总览": "",
    "盲读台": "blind",
    "揭示与分歧": "reveal",
    "仲裁台": "adj",
    "条件复核": "cosign",
    "定稿与闸门": "final",
    "闸门拦截演示": "blocked",
}

_IFRAME_HEIGHT = 1250  # 页面固有高度约 1200px；scrolling 兜底


def _load_html() -> str:
    if not _HTML_PATH.exists():
        st.error(f"未找到单文件版 HTML：{_HTML_PATH}")
        st.info("请确认 mvp/hitl-demo/ 目录已随仓库检出（该文件已提交至 main）。")
        return ""
    return _HTML_PATH.read_text(encoding="utf-8")


def _inject_stage(html: str, stage: str) -> str:
    """在 app.js 执行前把 #auto=hash 写入，使 iframe 内直接落在指定阶段。"""
    if not stage:
        return html
    snippet = f"<script>try{{location.hash='#auto={stage}'}}catch(e){{}}</script>"
    marker = "<body>"
    idx = html.find(marker)
    if idx < 0:
        return html + snippet
    insert_at = idx + len(marker)
    return html[:insert_at] + snippet + html[insert_at:]


def render() -> None:
    """app.py 页面分发调用入口。"""
    st.markdown("#### 🧪 HITL 标注流水线 MVP")
    st.caption(
        "盲读 → 揭示 → 仲裁 → 条件复核 → 导出闸门的可交互演示。"
        "指标为真实几何计算；契约对象字段名与 PRD v7 第七章一致。详见 `mvp/hitl-demo/README.md`。"
    )

    html = _load_html()
    if not html:
        return

    choice = st.selectbox("跳转到演示阶段", list(_STAGES.keys()), index=0)
    html = _inject_stage(html, _STAGES[choice])

    components.html(html, height=_IFRAME_HEIGHT, scrolling=True)

    with st.expander("实现边界（诚实声明）"):
        st.markdown(
            "- 样本图像为程序化合成（确定性种子），非真实显微图像\n"
            "- 专家 B / C 为脚本化模拟；指标（Dice/F1/HD95/α）为真实几何计算\n"
            "- 审计哈希为 FNV-1a（非密码学）；正式实现按 `analysis/audit_spec.py` 的 organoid-audit-v1\n"
            "- iframe 内运行，右上角「契约 JSON」抽屉、全部按钮交互均可用"
        )


if __name__ == "__main__":
    st.set_page_config(page_title="HITL MVP | 类器官标注平台", page_icon="🧪", layout="wide")
    render()
