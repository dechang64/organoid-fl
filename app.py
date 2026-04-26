"""
Organoid-FL: Federated Learning Platform for Organoid Image Analysis
=====================================================================
Streamlit-based research platform with interactive modules.

Architecture:
  - Rust HNSW VectorDB + gRPC (optional backend)
  - Python FL Engine (FedAvg)
  - Plotly interactive visualizations
  - SHA-256 Blockchain Audit Chain

Quick Start:
  pip install -r requirements.txt
  streamlit run app.py

Deployment:
  Docker:     docker compose up --build
  Cloud:      Push to GitHub → deploy on Streamlit Community Cloud
"""

import streamlit as st
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

# ── Page Config ──
st.set_page_config(
    page_title="Organoid-FL | Federated Learning for Organoid Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0;
        color: #94a3b8;
        font-size: 1rem;
    }
    .stMetric {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown("""
# 🧬 Organoid-FL

**Federated Learning Platform**
for Medical Organoid Image Analysis

---

**Tech Stack:**
- 🦀 Rust HNSW VectorDB
- 🧠 PyTorch FedAvg
- 🔗 gRPC Interface
- ⛓️ SHA-256 Audit Chain
- 📊 Plotly Visualizations

---

**Quick Start:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About This Platform**

Privacy-preserving AI for medical organoid analysis. Train models collaboratively across hospitals without sharing patient data.

**Unique Features:**
- 🔄 Interactive FL Training
- 🔍 HNSW Vector Search
- ⛓️ Blockchain Audit Trail
- 📈 Real-time Visualizations
- 🧪 Synthetic Data Generator
""")

# ── Navigation ──
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📁 Data Explorer",
        "🔄 FL Training",
        "🔍 Vector Search",
        "⛓️ Audit Chain",
        "📈 Model Analysis",
        "🔬 Research",
    ],
    label_visibility="collapsed",
)

# ── Route to Pages ──
if page == "🏠 Dashboard":
    from modules import dashboard
    dashboard.render()
elif page == "📁 Data Explorer":
    from modules import data_explorer
    data_explorer.render()
elif page == "🔄 FL Training":
    from modules import fl_training
    fl_training.render()
elif page == "🔍 Vector Search":
    from modules import vector_search
    vector_search.render()
elif page == "⛓️ Audit Chain":
    from modules import audit_chain
    audit_chain.render()
elif page == "📈 Model Analysis":
    from modules import model_analysis
    model_analysis.render()
elif page == "🔬 Research":
    from modules import research
    research.render()
