# ── utils/constants.py ──
"""Shared constants for Organoid-FL platform."""

# Organoid classification classes
ORGANOID_CLASSES = ["healthy", "early_stage", "late_stage"]

# Class display names and colors
CLASS_INFO = {
    "healthy": {"label": "Healthy", "color": "#22c55e", "emoji": "🟢"},
    "early_stage": {"label": "Early Stage", "color": "#f59e0b", "emoji": "🟡"},
    "late_stage": {"label": "Late Stage", "color": "#ef4444", "emoji": "🔴"},
}

# Model defaults
DEFAULT_INPUT_DIM = 512
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_CLASSES = 3

# FL defaults
DEFAULT_ROUNDS = 10
DEFAULT_CLIENTS = 3
DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOCAL_EPOCHS = 2

# HNSW defaults
HNSW_M = 16
HNSW_M0 = 32

# Research references
REFERENCES = {
    "McMahan 2017": "McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.",
    "Li 2020": "Li, T. et al. (2020). Federated Optimization in Heterogeneous Networks. MLSys.",
    "Kairouz 2021": "Kairouz, P. et al. (2021). Advances and Open Problems in Federated Learning. Foundations and Trends.",
    "Bonawitz 2019": "Bonawitz, K. et al. (2019). Towards Federated Learning at Scale: A System Design. MLSys.",
}

# Color palette for charts
COLORS = {
    "healthy": "#22c55e",
    "early_stage": "#f59e0b",
    "late_stage": "#ef4444",
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "bg": "#fafbfc",
    "grid": "#e5e7eb",
    "text": "#1f2937",
    "muted": "#6b7280",
}
