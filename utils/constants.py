# ── utils/constants.py ──
"""Shared constants for the Organoid-FL platform.

Label definitions now come from :mod:`utils.taxonomy` — the registry is the
single source of truth. Everything here is either derived from it or is a
genuinely non-label constant (model sizes, FL defaults, references).

Backward compatibility
----------------------
Legacy names (``ORGANOID_CLASSES``, ``CLASS_INFO``, ``DEFAULT_NUM_CLASSES``)
are preserved as derived shims so existing call sites keep importing. They
were previously hard-coded to a *placeholder* 3-class schema
(``healthy / early_stage / late_stage``) that matched no dataset in this
repository, which is why detectors silently mislabelled real 4-class data.

These shims now reflect the registry default — :data:`utils.taxonomy.DEFAULT_TAXONOMY`,
currently ``organoid_4class``. Code that genuinely needs the old placeholder
should ask for it explicitly; doing so raises a ``DeprecationWarning``.
"""

from __future__ import annotations

from typing import Any

from utils.taxonomy import (
    DEFAULT_TAXONOMY,
    Taxonomy,
    get_taxonomy,
    resolve_taxonomy,
)

__all__ = [
    "ACTIVE_TAXONOMY",
    "CLASS_INFO",
    "CLASS_KEYS",
    "COLORS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CLIENTS",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_LOCAL_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_ROUNDS",
    "HNSW_M",
    "HNSW_M0",
    "ORGANOID_CLASSES",
    "REFERENCES",
    "TASK_TO_TAXONOMY",
    "get_active_taxonomy",
    "set_active_taxonomy",
]


# ─────────────────────────────────────────────────────────────────────────────
# Active taxonomy
# ─────────────────────────────────────────────────────────────────────────────

#: Registry name of the schema the platform treats as canonical.
ACTIVE_TAXONOMY: str = DEFAULT_TAXONOMY

_ACTIVE_OVERRIDE: Taxonomy | None = None


def get_active_taxonomy() -> Taxonomy:
    """Return the taxonomy currently in force for the platform."""
    if _ACTIVE_OVERRIDE is not None:
        return _ACTIVE_OVERRIDE
    return get_taxonomy(ACTIVE_TAXONOMY)


def set_active_taxonomy(spec: Any) -> Taxonomy:
    """Switch the platform-wide taxonomy at runtime.

    Accepts anything :func:`utils.taxonomy.resolve_taxonomy` understands: a
    registered name, a ``Taxonomy``, a parsed ``data.yaml`` mapping, or a path
    to one. Returns the resolved taxonomy so callers can read ``n_classes``.
    """
    global _ACTIVE_OVERRIDE
    _ACTIVE_OVERRIDE = resolve_taxonomy(spec)
    return _ACTIVE_OVERRIDE


#: Task name -> preferred taxonomy name. Lets a caller ask "what schema should
#: the segmentation task use" without hard-coding a class list.
TASK_TO_TAXONOMY: dict[str, str] = {
    "detection": DEFAULT_TAXONOMY,
    "classification": DEFAULT_TAXONOMY,
    "segmentation": DEFAULT_TAXONOMY,
    "stage_classification": "developmental_4stage",
    "plate_phenotype": "multiorg_2class",
}


# ─────────────────────────────────────────────────────────────────────────────
# Derived label views (backward-compatible shims)
# ─────────────────────────────────────────────────────────────────────────────

#: Class keys of the active taxonomy, in declaration order.
#: NOTE: was ``["healthy", "early_stage", "late_stage"]`` — that schema matched
#: no dataset. Now derived, so it can never drift from the data again.
ORGANOID_CLASSES: list[str] = list(get_active_taxonomy().keys)

#: Alias for readers who think in terms of "class keys".
CLASS_KEYS: list[str] = ORGANOID_CLASSES

#: ``{key: {"label": ..., "color": ..., "id": ...}}`` for dashboards.
CLASS_INFO: dict[str, dict] = {
    c.key: {"label": c.label, "color": c.color, "id": c.id}
    for c in get_active_taxonomy().classes
}

#: Number of classes in the active taxonomy.
#: NOTE: was ``3``; the real datasets have 4, which is one more way the old
#: placeholder leaked into model configuration.
DEFAULT_NUM_CLASSES: int = get_active_taxonomy().n_classes


# ─────────────────────────────────────────────────────────────────────────────
# Model defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT_DIM = 512
DEFAULT_HIDDEN_DIM = 128

# ─────────────────────────────────────────────────────────────────────────────
# FL defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ROUNDS = 10
DEFAULT_CLIENTS = 3
DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOCAL_EPOCHS = 2

# ─────────────────────────────────────────────────────────────────────────────
# HNSW defaults
# ─────────────────────────────────────────────────────────────────────────────

HNSW_M = 16
HNSW_M0 = 32

# ─────────────────────────────────────────────────────────────────────────────
# Research references
# ─────────────────────────────────────────────────────────────────────────────

REFERENCES = {
    "McMahan 2017": "McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.",
    "Li 2020": "Li, T. et al. (2020). Federated Optimization in Heterogeneous Networks. MLSys.",
    "Kairouz 2021": "Kairouz, P. et al. (2021). Advances and Open Problems in Federated Learning. Foundations and Trends.",
    "Bonawitz 2019": "Bonawitz, K. et al. (2019). Towards Federated Learning at Scale: A System Design. MLSys.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for charts
# ─────────────────────────────────────────────────────────────────────────────

#: Per-class colours are registry-derived so charts, tables and detection
#: overlays agree. The named chrome colours below are label-independent.
COLORS: dict[str, str] = {
    **get_active_taxonomy().color_map,
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "bg": "#fafbfc",
    "grid": "#e5e7eb",
    "text": "#1f2937",
    "muted": "#6b7280",
}
