# ── modules/audit_chain.py ──
"""
Audit Chain Page
================
Browse the blockchain-style immutable audit log.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.audit_engine import AuditEngine
from visualization.charts import audit_timeline


def render():
    st.markdown(
        '<div class="main-header"><h1>⛓️ Audit Chain</h1>'
        '<p>Immutable blockchain-style log of all federated learning operations</p></div>',
        unsafe_allow_html=True,
    )

    # ── Initialize ──
    if "audit_engine" not in st.session_state:
        st.session_state["audit_engine"] = AuditEngine(max_blocks=1000)
        st.session_state["audit_blocks"] = 1
        st.session_state["audit_valid"] = True

    engine = st.session_state["audit_engine"]

    # ── Stats ──
    col1, col2, col3 = st.columns(3)
    stats = engine.get_stats()
    with col1:
        st.metric("Chain Length", stats["chain_length"])
    with col2:
        st.metric("Chain Valid", "✅ Yes" if stats["chain_valid"] else "❌ No")
    with col3:
        st.metric("Latest Hash", stats["latest_hash"])

    st.markdown("---")

    # ── Manual Operations ──
    with st.expander("🔧 Add Test Operations"):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Add Insert Block"):
                engine.append("insert", {"vectors": 10, "dimension": 512})
                st.session_state["audit_blocks"] = len(engine)
                st.rerun()
            if st.button("Add Search Block"):
                engine.append("search", {"k": 5, "results": 5})
                st.session_state["audit_blocks"] = len(engine)
                st.rerun()
        with col_b:
            if st.button("Add Delete Block"):
                engine.append("delete", {"vectors": 2})
                st.session_state["audit_blocks"] = len(engine)
                st.rerun()
            if st.button("Add FL Round Block"):
                engine.append("fl_round", {"round": len(engine), "clients": 3, "accuracy": 0.95})
                st.session_state["audit_blocks"] = len(engine)
                st.rerun()

    # ── Verify Chain ──
    if st.button("🔍 Verify Chain Integrity", use_container_width=True):
        valid = engine.verify_chain()
        st.session_state["audit_valid"] = valid
        if valid:
            st.success("✅ Chain integrity verified — all hashes match.")
        else:
            st.error("❌ Chain integrity broken — tampering detected!")

    st.markdown("---")

    # ── Chain Browser ──
    st.markdown("### 📋 Chain Browser")

    n_show = st.slider("Show last N blocks", 5, 50, 10)
    recent = engine.recent(n_show)

    if recent:
        df = engine.to_dataframe()
        st.dataframe(df.tail(n_show), use_container_width=True, hide_index=True)

        # Timeline visualization
        fig = audit_timeline(recent)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No blocks yet. Add test operations above.")

    # ── Methodology ──
    with st.expander("📖 Blockchain Audit Methodology"):
        st.markdown("""
        **SHA-256 Hash Chain**

        Each block contains:
        - `index`: Monotonically increasing block number
        - `timestamp`: ISO 8601 UTC timestamp
        - `operation`: Type of FL operation (insert/search/delete/fl_round)
        - `details`: Operation-specific metadata
        - `prev_hash`: SHA-256 hash of previous block
        - `hash`: SHA-256 hash of current block contents

        **Tamper Detection**: Any modification to a block changes its hash,
        breaking the chain link to the next block. Verification checks all links.

        **Compliance**: Provides HIPAA/GDPR-compatible audit trail for
        medical AI operations without exposing patient data.
        """)
