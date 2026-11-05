# ── utils/taxonomy.py ──
"""Taxonomy Registry — single source of truth for label schemas.

Motivation
----------
Before this module the repository carried three mutually contradictory label
schemas with no single authority:

    utils/constants.py:5        healthy / early_stage / late_stage   (3)
    analysis/detector.py:47     healthy / early_stage / late_stage   (3)
    data/*/data.yaml            organoid0 / organoid1 / organoid3 / spheroid (4)
    datasets/multiorg_balanced  Normal / Macros                      (2)

No dataset in the repository actually matched the schema the detector claimed,
so running the detector on real data silently mislabelled every `cls_id >= 3`
as the string ``class_3``.

This module replaces all of that with a registry. A taxonomy is an immutable,
versioned, ordered set of :class:`LabelClass` entries. Consumers ask the
registry for classes instead of hard-coding them.

Design rules
------------
* **Additive, never destructive.** Existing taxonomies stay registered; a new
  dataset means a new registered taxonomy, not a renumbering of an old one.
  This is what lets MultiOrg's 2-class and the intestinal 4-class coexist.
* **Legacy schemas are kept but flagged deprecated**, so old call sites keep
  working while emitting a :class:`DeprecationWarning` instead of silently
  returning wrong labels.
* **Registry is keyed by name, and names are resolved from many shapes** —
  string name, ``Taxonomy`` instance, a parsed ``data.yaml`` dict, or a path
  to one. See :func:`resolve_taxonomy`.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

__all__ = [
    "DEFAULT_TAXONOMY",
    "LEGACY_TAXONOMY",
    "DeprecatedTaxonomyWarning",
    "LabelClass",
    "Taxonomy",
    "TaxonomyError",
    "get_taxonomy",
    "list_taxonomies",
    "maybe_get_taxonomy",
    "register_taxonomy",
    "resolve_taxonomy",
    "taxonomy_from_data_yaml",
    "unregister_taxonomy",
]


class TaxonomyError(ValueError):
    """Raised for unknown names, malformed definitions, or id collisions."""


class DeprecatedTaxonomyWarning(DeprecationWarning):
    """Emitted when a deprecated taxonomy is requested by name."""


#: Default taxonomy for the platform. Chosen because it is the schema every
#: real dataset here actually uses; see :func:`_build_organoid_4class`.
DEFAULT_TAXONOMY = "organoid_4class"

#: The deprecated placeholder schema, kept resolvable for backward compatibility.
LEGACY_TAXONOMY = "organoid_health_3class"


# ─────────────────────────────────────────────────────────────────────────────
# LabelClass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelClass:
    """One label within a taxonomy.

    Attributes
    ----------
    id:
        Dense class index used by models and stored annotations. Must be
        unique within the taxonomy and non-negative.
    key:
        Stable machine identifier (lower snake case). This is what code and
        persisted records should reference — never the display ``label``.
    label:
        Human facing name.
    color:
        Hex colour used by every visualisation so dashboards stay consistent.
    parent:
        Optional ``key`` of a parent class. Used for lineage taxonomies such
        as developmental stages that can be collapsed into a coarser group.
    aliases:
        Alternative spellings that resolve back to this class, used to absorb
        vocabulary drift across datasets and legacy scripts.
    """

    id: int
    key: str
    label: str
    color: str = "#9ca3af"
    parent: Optional[str] = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id < 0:
            raise TaxonomyError(f"LabelClass.id must be a non-negative int, got {self.id!r}")
        if not self.key or not self.key.strip():
            raise TaxonomyError("LabelClass.key must be a non-empty string")
        if not isinstance(self.aliases, tuple):
            object.__setattr__(self, "aliases", tuple(self.aliases))

    @property
    def alias_set(self) -> frozenset:
        """``key`` plus every alias, lower-cased, for lookup."""
        return frozenset({self.key.lower(), *(a.lower() for a in self.aliases)})

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "label": self.label,
            "color": self.color,
            "parent": self.parent,
            "aliases": list(self.aliases),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Taxonomy:
    """An immutable, ordered, versioned label schema."""

    name: str
    classes: tuple[LabelClass, ...]
    version: str = "1.0"
    task: str = "detection"
    description: str = ""
    source: str = ""
    deprecated: bool = False
    superseded_by: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise TaxonomyError("Taxonomy.name must be a non-empty string")
        classes = tuple(self.classes)
        if not classes:
            raise TaxonomyError(f"Taxonomy {self.name!r} must declare at least one class")
        object.__setattr__(self, "classes", classes)

        ids = [c.id for c in classes]
        if len(set(ids)) != len(ids):
            raise TaxonomyError(f"Taxonomy {self.name!r} has duplicate class ids: {sorted(ids)}")

        keys = [c.key.lower() for c in classes]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise TaxonomyError(f"Taxonomy {self.name!r} has duplicate class keys: {dupes}")

        known = set(keys)
        for c in classes:
            if c.parent is not None and c.parent.lower() not in known:
                raise TaxonomyError(
                    f"Taxonomy {self.name!r}: class {c.key!r} references unknown parent {c.parent!r}"
                )

    # ── basic introspection ──────────────────────────────────────────────

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.classes)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.classes)

    @property
    def ids(self) -> tuple[int, ...]:
        """Class ids in declaration order (not necessarily ``0..n-1``)."""
        return tuple(c.id for c in self.classes)

    @property
    def is_dense(self) -> bool:
        """True when ids are exactly ``0..n-1``, i.e. safe as YOLO class indices."""
        return set(self.ids) == set(range(self.n_classes))

    @property
    def id_order(self) -> tuple[int, ...]:
        """Class ids sorted ascending — the index order a dense model expects."""
        return tuple(sorted(self.ids))

    # ── lookup maps ──────────────────────────────────────────────────────

    @property
    def id_to_class(self) -> dict[int, LabelClass]:
        return {c.id: c for c in self.classes}

    @property
    def key_to_class(self) -> dict[str, LabelClass]:
        return {c.key.lower(): c for c in self.classes}

    @property
    def key_to_id(self) -> dict[str, int]:
        return {c.key.lower(): c.id for c in self.classes}

    @property
    def id_to_key(self) -> dict[int, str]:
        return {c.id: c.key for c in self.classes}

    @property
    def color_map(self) -> dict[str, str]:
        """``{key: hex}`` — the canonical palette for this schema."""
        return {c.key: c.color for c in self.classes}

    @property
    def alias_map(self) -> dict[str, LabelClass]:
        """Every key and alias, lower-cased, mapped to its class."""
        out: dict[str, LabelClass] = {}
        for c in self.classes:
            for a in c.alias_set:
                out[a] = c
        return out

    # ── resolution ───────────────────────────────────────────────────────

    def resolve_key(self, name: Union[str, int]) -> str:
        """Resolve a class key, alias, label, or id to its canonical ``key``.

        Accepts ints (treated as ids) and strings (tried as key, alias, then
        case-insensitive label).
        """
        if isinstance(name, bool):
            raise TaxonomyError(f"Cannot resolve bool as a class in {self.name!r}")
        if isinstance(name, int):
            cls = self.id_to_class.get(name)
            if cls is None:
                raise TaxonomyError(
                    f"{self.name!r} has no class id {name}; valid ids are {self.ids}"
                )
            return cls.key

        needle = str(name).strip().lower()
        cls = self.alias_map.get(needle)
        if cls is None:
            for c in self.classes:
                if c.label.strip().lower() == needle:
                    cls = c
                    break
        if cls is None:
            raise TaxonomyError(
                f"{self.name!r} cannot resolve class {name!r}; "
                f"valid keys are {self.keys}"
            )
        return cls.key

    def resolve_id(self, name: Union[str, int]) -> int:
        """Resolve a class key/alias/label/id to its integer id."""
        return self.key_to_id[self.resolve_key(name).lower()]

    def try_resolve_key(self, name: Union[str, int]) -> Optional[str]:
        """Like :meth:`resolve_key` but returns ``None`` instead of raising."""
        try:
            return self.resolve_key(name)
        except TaxonomyError:
            return None

    def children_of(self, key: str) -> tuple[LabelClass, ...]:
        """Classes whose ``parent`` is ``key``."""
        target = key.lower()
        return tuple(c for c in self.classes if (c.parent or "").lower() == target)

    def roots(self) -> tuple[LabelClass, ...]:
        """Classes without a parent."""
        return tuple(c for c in self.classes if c.parent is None)

    # ── validation against models / datasets ─────────────────────────────

    def validate_index(self, class_id: int, *, allow_sparse: bool = False) -> int:
        """Validate a class index coming from a *dense* model output.

        Args:
            class_id: index produced by a model (0-based, contiguous).
            allow_sparse: when False (default) the taxonomy must be dense and
                ``class_id`` is matched directly against ``id``. When True the
                value is treated as a position into :attr:`id_order`.
        """
        if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < 0:
            raise TaxonomyError(f"class_id must be a non-negative int, got {class_id!r}")
        if class_id >= self.n_classes:
            raise TaxonomyError(
                f"class_id {class_id} out of range for {self.name!r} "
                f"({self.n_classes} classes)"
            )
        if allow_sparse:
            return self.id_order[class_id]
        if not self.is_dense:
            raise TaxonomyError(
                f"{self.name!r} has sparse class ids {self.ids}; "
                "pass allow_sparse=True to map positionally"
            )
        return class_id

    # ── serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "description": self.description,
            "source": self.source,
            "deprecated": self.deprecated,
            "superseded_by": self.superseded_by,
            "nc": self.n_classes,
            "names": [c.key for c in self.classes],
            "classes": [c.to_dict() for c in self.classes],
        }

    def to_data_yaml_dict(self) -> dict:
        """Emit a YOLO ``data.yaml``-shaped dict with a sparse-safe name list."""
        ordered = sorted(self.classes, key=lambda c: c.id)
        return {
            "nc": len(ordered),
            "names": {c.id: c.key for c in ordered},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Taxonomy":
        if payload.get("classes"):
            classes = tuple(
                LabelClass(
                    id=int(c["id"]),
                    key=str(c["key"]),
                    label=str(c.get("label", c["key"])),
                    color=str(c.get("color", "#9ca3af")),
                    parent=c.get("parent"),
                    aliases=tuple(c.get("aliases", ()) or ()),
                )
                for c in payload["classes"]
            )
        else:
            # Minimal form: just names
            classes = tuple(
                LabelClass(id=i, key=str(k), label=str(k))
                for i, k in enumerate(payload.get("names", ()))
            )
        return cls(
            name=str(payload["name"]),
            classes=classes,
            version=str(payload.get("version", "1.0")),
            task=str(payload.get("task", "detection")),
            description=str(payload.get("description", "")),
            source=str(payload.get("source", "")),
            deprecated=bool(payload.get("deprecated", False)),
            superseded_by=payload.get("superseded_by"),
        )

    def with_classes(self, classes: Iterable[LabelClass], **changes: Any) -> "Taxonomy":
        """Return a copy with a different class list — handy for deriving schemas."""
        return replace(self, classes=tuple(classes), **changes)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Taxonomy] = {}


def _normalise_registry_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def register_taxonomy(taxonomy: Taxonomy, *, overwrite: bool = False) -> Taxonomy:
    """Register ``taxonomy`` so it can be resolved by name.

    Raises:
        TaxonomyError: if the name exists and ``overwrite`` is False.
    """
    if not isinstance(taxonomy, Taxonomy):
        raise TaxonomyError(f"register_taxonomy expects a Taxonomy, got {type(taxonomy).__name__}")
    key = _normalise_registry_key(taxonomy.name)
    if key in _REGISTRY and not overwrite:
        raise TaxonomyError(
            f"Taxonomy {taxonomy.name!r} is already registered; "
            "pass overwrite=True to replace it"
        )
    _REGISTRY[key] = taxonomy
    return taxonomy


def unregister_taxonomy(name: str) -> None:
    """Remove a taxonomy. Raises :class:`TaxonomyError` if it is not present."""
    key = _normalise_registry_key(name)
    if key not in _REGISTRY:
        raise TaxonomyError(f"No taxonomy named {name!r} is registered")
    del _REGISTRY[key]


def maybe_get_taxonomy(name: str) -> Optional[Taxonomy]:
    """Return the taxonomy or ``None``. Never warns, never raises."""
    return _REGISTRY.get(_normalise_registry_key(name))


def get_taxonomy(name: str = DEFAULT_TAXONOMY) -> Taxonomy:
    """Return a registered taxonomy by name.

    Emits :class:`DeprecatedTaxonomyWarning` when the taxonomy is flagged
    deprecated, so legacy call sites surface instead of silently mislabelling.
    """
    key = _normalise_registry_key(name)
    taxonomy = _REGISTRY.get(key)
    if taxonomy is None:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise TaxonomyError(
            f"Unknown taxonomy {name!r}. Registered taxonomies: {available}"
        )
    if taxonomy.deprecated:
        hint = f" Use {taxonomy.superseded_by!r} instead." if taxonomy.superseded_by else ""
        warnings.warn(
            f"Taxonomy {taxonomy.name!r} is deprecated.{hint}",
            DeprecatedTaxonomyWarning,
            stacklevel=2,
        )
    return taxonomy


def list_taxonomies(*, include_deprecated: bool = True) -> tuple[str, ...]:
    """Names of every registered taxonomy, sorted."""
    names = [
        t.name
        for t in _REGISTRY.values()
        if include_deprecated or not t.deprecated
    ]
    return tuple(sorted(names))


# ─────────────────────────────────────────────────────────────────────────────
# Loading from an existing dataset descriptor
# ─────────────────────────────────────────────────────────────────────────────


def taxonomy_from_data_yaml(
    path: str,
    *,
    name: Optional[str] = None,
    task: str = "detection",
    register: bool = True,
) -> Taxonomy:
    """Build a :class:`Taxonomy` from a YOLO ``data.yaml``.

    Handles both the ``names`` list form and the ``{0: "a", 1: "b"}`` mapping
    form, and tolerates the missing ``nc`` seen in several files here.

    The inferred name defaults to the parent directory of ``path`` (or the
    ``path`` field inside the YAML), lower-snake-cased.
    """
    if not os.path.isfile(path):
        raise TaxonomyError(f"data.yaml not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        payload = _yaml_safe_load(fh.read(), path)

    if not isinstance(payload, Mapping):
        raise TaxonomyError(f"{path} does not contain a YAML mapping")

    raw_names = payload.get("names")
    if raw_names is None:
        raise TaxonomyError(f"{path} has no 'names' key")

    if isinstance(raw_names, Mapping):
        pairs = sorted(((int(k), str(v)) for k, v in raw_names.items()), key=lambda p: p[0])
    elif isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
        pairs = [(i, str(v)) for i, v in enumerate(raw_names)]
    else:
        raise TaxonomyError(f"{path}: unsupported 'names' value {raw_names!r}")

    declared_nc = payload.get("nc")
    if declared_nc is not None and int(declared_nc) != len(pairs):
        warnings.warn(
            f"{path}: nc={declared_nc} disagrees with {len(pairs)} names; using the names",
            stacklevel=2,
        )

    if name is None:
        base = payload.get("path") or os.path.dirname(path)
        name = _normalise_registry_key(os.path.basename(str(base).rstrip("/\\")) or "dataset")

    palette = _DEFAULT_PALETTE
    classes = tuple(
        LabelClass(
            id=cid,
            key=_normalise_registry_key(label),
            label=label,
            color=palette[i % len(palette)],
        )
        for i, (cid, label) in enumerate(pairs)
    )

    taxonomy = Taxonomy(
        name=name,
        classes=classes,
        version="1.0",
        task=task,
        description=f"Imported from {os.path.relpath(path)}",
        source=os.path.abspath(path),
    )
    if register:
        existing = maybe_get_taxonomy(taxonomy.name)
        if existing is None:
            register_taxonomy(taxonomy)
        elif existing.classes != taxonomy.classes:
            # Do not clobber a different schema registered under the same name.
            raise TaxonomyError(
                f"Taxonomy {taxonomy.name!r} already registered with different classes. "
                "Pass a distinct name= to register this dataset separately."
            )
    return taxonomy


def resolve_taxonomy(
    spec: Union[str, Taxonomy, Mapping[str, Any], None] = None,
    *,
    register_imported: bool = True,
) -> Taxonomy:
    """Resolve anything taxonomy-shaped into a :class:`Taxonomy`.

    Accepts, in order of preference:

    * ``None`` → :data:`DEFAULT_TAXONOMY`
    * a :class:`Taxonomy` (returned unchanged)
    * a registered name, e.g. ``"multiorg_2class"``
    * a path to an existing ``data.yaml`` (parsed and optionally registered)
    * a mapping — either a full taxonomy payload or a bare ``{"names": [...]}``
    """
    if spec is None:
        return get_taxonomy(DEFAULT_TAXONOMY)
    if isinstance(spec, Taxonomy):
        return spec

    if isinstance(spec, Mapping):
        payload = dict(spec)
        payload.setdefault("name", "anonymous")
        return Taxonomy.from_dict(payload)

    if isinstance(spec, str):
        registered = maybe_get_taxonomy(spec)
        if registered is not None:
            return get_taxonomy(registered.name)
        if os.path.isfile(spec):
            return taxonomy_from_data_yaml(spec, register=register_imported)
        raise TaxonomyError(
            f"Cannot resolve taxonomy from {spec!r}: not a registered name, "
            "not an existing file, and not a mapping."
        )

    raise TaxonomyError(f"Unsupported taxonomy spec: {type(spec).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Built-in taxonomies
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PALETTE = (
    "#2563eb", "#dc2626", "#16a34a", "#d97706",
    "#7c3aed", "#0891b2", "#db2777", "#65a30d",
)


def _build_organoid_4class() -> Taxonomy:
    """The schema every real dataset in this repository actually uses.

    Verified against ``data/organoid_patches/data.yaml``,
    ``data/intestinal_organoid/OrganoidDataset/data.yaml`` and all twelve
    ``fl_detect_results/*/*/data.yaml`` files.
    """
    return Taxonomy(
        name="organoid_4class",
        version="1.0",
        task="detection",
        description=(
            "Intestinal organoid morphology as shipped in the intestinal dataset. "
            "Ids are sparse (0,1,3,4) in the source YAML; remapped to dense 0..3 here "
            "while preserving the original tokens as aliases."
        ),
        source="data/organoid_patches/data.yaml",
        classes=(
            LabelClass(
                id=0, key="organoid0", label="Organoid 0",
                color="#2563eb", aliases=("cystic", "cyst_like", "class_0"),
            ),
            LabelClass(
                id=1, key="organoid1", label="Organoid 1",
                color="#dc2626", aliases=("early", "early_budding", "class_1"),
            ),
            LabelClass(
                id=2, key="organoid3", label="Organoid 3",
                color="#16a34a", aliases=("late", "late_budding", "budding", "class_2"),
            ),
            LabelClass(
                id=3, key="spheroid", label="Spheroid",
                color="#d97706", aliases=("sphere", "round", "class_3"),
            ),
        ),
    )


def _build_developmental_4stage() -> Taxonomy:
    """Developmental staging used by the annotation PRD's scientific definition."""
    return Taxonomy(
        name="developmental_4stage",
        version="1.0",
        task="detection",
        description=(
            "Canonical organoid developmental staging: cystic non-budding -> early "
            "budding -> late budding -> spheroid. This is the schema the annotation "
            "product reasons about; map dataset-specific tokens onto it via aliases."
        ),
        source="PRD v6 领域特征定义",
        classes=(
            LabelClass(
                id=0, key="cystic_non_budding", label="Cystic (non-budding)",
                color="#2563eb", aliases=("cystic", "cyst", "non_budding", "stage0"),
            ),
            LabelClass(
                id=1, key="early_budding", label="Early budding",
                color="#dc2626", parent="cystic_non_budding",
                aliases=("early", "early_bud", "stage1"),
            ),
            LabelClass(
                id=2, key="late_budding", label="Late budding",
                color="#16a34a", parent="early_budding",
                aliases=("late", "late_bud", "budding", "stage2"),
            ),
            LabelClass(
                id=3, key="spheroid", label="Spheroid",
                color="#d97706",
                aliases=("sphere", "round", "stage3"),
            ),
        ),
    )


def _build_multiorg_2class() -> Taxonomy:
    """MultiOrg plate phenotype schema (``datasets/multiorg_balanced``)."""
    return Taxonomy(
        name="multiorg_2class",
        version="1.0",
        task="detection",
        description="MultiOrg plate-level phenotype: Normal vs Macros.",
        source="datasets/multiorg_balanced/data.yaml",
        classes=(
            LabelClass(
                id=0, key="normal", label="Normal",
                color="#2563eb", aliases=("neg", "negative", "organoid"),
            ),
            LabelClass(
                id=1, key="macros", label="Macros",
                color="#dc2626", aliases=("macro", "pos", "positive", "mixed"),
            ),
        ),
    )


def _build_legacy_health_3class() -> Taxonomy:
    """Deprecated. Kept only so existing call sites resolve with a warning.

    No dataset in this repository ever used this schema — it was a placeholder
    that leaked into ``utils/constants.py`` and ``analysis/detector.py`` and
    caused real mislabelling on 4-class data.
    """
    return Taxonomy(
        name="organoid_health_3class",
        version="0.9",
        task="classification",
        description=(
            "DEPRECATED placeholder schema (healthy/early_stage/late_stage). "
            "Never matched any dataset in this repository. Superseded by developmental_4stage."
        ),
        source="utils/constants.py (pre-registry)",
        deprecated=True,
        superseded_by="developmental_4stage",
        classes=(
            LabelClass(
                id=0, key="healthy", label="Healthy",
                color="#16a34a", aliases=("normal", "viable"),
            ),
            LabelClass(
                id=1, key="early_stage", label="Early Stage",
                color="#d97706", aliases=("early", "stage_early"),
            ),
            LabelClass(
                id=2, key="late_stage", label="Late Stage",
                color="#dc2626", aliases=("late", "stage_late"),
            ),
        ),
    )


def _install_builtins() -> None:
    for factory in (
        _build_organoid_4class,
        _build_developmental_4stage,
        _build_multiorg_2class,
        _build_legacy_health_3class,
    ):
        taxonomy = factory()
        _REGISTRY[_normalise_registry_key(taxonomy.name)] = taxonomy


_install_builtins()


# ─────────────────────────────────────────────────────────────────────────────
# Minimal YAML loader
# ─────────────────────────────────────────────────────────────────────────────


def _yaml_safe_load(text: str, origin: str) -> Any:
    """Parse YAML, falling back to a tiny reader when PyYAML is unavailable.

    The fallback only needs to understand the two shapes these data.yaml files
    ever take, which keeps this module importable in a bare environment.
    """
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml_load(text, origin)


def _mini_yaml_load(text: str, origin: str) -> dict:
    """Parse the subset of YAML used by YOLO ``data.yaml`` files."""
    result: dict = {}
    current_list_key: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()

        if stripped.startswith("- "):
            if current_list_key is None:
                raise TaxonomyError(f"{origin}: list item outside of a key: {stripped!r}")
            result.setdefault(current_list_key, []).append(_coerce(stripped[2:].strip()))
            continue

        if ":" not in stripped:
            raise TaxonomyError(f"{origin}: cannot parse line {stripped!r}")

        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.strip()
        if not value:
            current_list_key = key
            result.setdefault(key, [])
        else:
            current_list_key = None
            result[key] = _coerce(value)

    return result


def _coerce(value: str) -> Any:
    value = value.strip().strip("'\"")
    try:
        return int(value)
    except ValueError:
        return value
