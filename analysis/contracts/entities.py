# ── analysis/contracts/entities.py ──
"""The nine entities of PRD 7.2, as immutable, strictly-serialised records.

Why a contract layer instead of extending ``analysis/schema.py``
----------------------------------------------------------------
``schema.py`` models *measurements* — a specimen, an image, an instance, a
track. Those objects exist before anyone annotates and survive after the task
closes. The nine objects here model the *review process*: who was asked, what
they produced, what the model suggested, who signed, who countersigned. Mixing
them into ``schema.py`` would give ``Instance`` a foreign key to a task, and the
first thing that happens to a measurement when a task is reopened is that
somebody starts mutating it.

So: separate package, separate lifetime, no mutable state. Every entity is a
frozen dataclass. There is not a single setter in this file, which is what makes
P-06 ("盲读版本在补充修订后内容哈希不变") true by construction rather than by
convention — a blind revision cannot be edited because there is no operation
that edits it; a supplement is a *new* ``Revision`` with a higher
``revision_no``.

Serialisation
-------------
``to_dict()`` and ``from_dict()`` are generic and driven by the dataclass type
hints, so there are no hand-written per-entity mappers to drift apart. Field
types are resolved once via :func:`typing.get_type_hints`, which means an enum
field round-trips through its string value and a ``tuple`` field comes back as a
tuple rather than a list.

``from_dict()`` is **strict**: unknown keys raise. A tolerant reader is how
schema drift becomes invisible — a producer that renames ``revision_no`` to
``rev_no`` should fail loudly on the first import, not silently produce
revisions all numbered zero because the field defaulted.

Validation philosophy
---------------------
Three kinds of rule, kept in three places:

* **Shape** (types, non-empty strings, ISO-8601 UTC timestamps) — here, in
  ``__post_init__``, so an illegal entity cannot exist at all.
* **Cross-entity** (a cosigner must not be the adjudicating chief; a model
  ranking must permute the sample's revisions) — module-level functions here,
  because they need two objects and a dataclass cannot see its neighbours.
* **Sequence** (which state may follow which) — ``state_machine``, because that
  is a table, not a property of one record.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from enum import Enum
from types import UnionType
from typing import (
    Any,
    ClassVar,
    Mapping,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from analysis.contracts.enums import (
    AssignmentState,
    CosignVerdict,
    Role,
    SampleState,
    Stratum,
)
from analysis.contracts.hashing import content_hash, is_content_hash
from analysis.contracts.templates import TemplateError, get_template, normalise_template_id

__all__ = [
    "Adjudication",
    "AnnotationProtocol",
    "Assignment",
    "ContractEntity",
    "ContractValidationError",
    "Cosign",
    "Disagreement",
    "ModelRank",
    "Revision",
    "StratumLabel",
    "Task",
    "validate_cosign_independence",
    "validate_model_rank",
]


class ContractValidationError(ValueError):
    """An entity (or a pair of entities) violates the frozen contract."""


_T = TypeVar("_T", bound="ContractEntity")


# ─────────────────────────────────────────────────────────────────────────────
# Primitive validators
# ─────────────────────────────────────────────────────────────────────────────


def _iso_now() -> str:
    """Current UTC time in the repository's canonical form (``…Z``, seconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_text(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{name} must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ContractValidationError(
            f"{name} must be a non-empty string. An empty value here is not a "
            "missing detail — it is a record that cannot answer the question it "
            "exists to answer (P-05)."
        )
    return value


def _require_enum(name: str, value: Any, enum: type[Enum]) -> Any:
    if isinstance(value, enum):
        return value
    if isinstance(value, str):
        try:
            return enum(value)
        except ValueError as exc:
            raise ContractValidationError(
                f"{name} = {value!r} is not a valid {enum.__name__}; "
                f"expected one of {[m.value for m in enum]}"
            ) from exc
    raise ContractValidationError(f"{name} must be a {enum.__name__}, got {type(value).__name__}")


def _parse_iso(name: str, value: Any) -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit UTC offset.

    A naive timestamp is rejected rather than assumed UTC. Assuming is how two
    nodes eight hours apart produce an audit chain whose ``revealed_at`` sits
    *before* its ``blind_choice_locked_at``, which reads as a P-09 violation in
    a review that has nothing to review.
    """
    text = _require_text(name, value)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ContractValidationError(
            f"{name} = {text!r} is not an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise ContractValidationError(
            f"{name} = {text!r} has no UTC offset. Timestamps must carry one "
            "(use the 'Z' suffix) so the ordering of lock and reveal is "
            "comparable across nodes."
        )
    return parsed.astimezone(timezone.utc)


def _require_iso(name: str, value: Any) -> str:
    _parse_iso(name, value)
    return value  # keep the caller's spelling; only the instant is validated


def _require_optional_iso(name: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    return _require_iso(name, value)


def _require_uint(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ContractValidationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_score(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ContractValidationError(
            f"{name} must lie in [0, 1], got {number}. A score outside the unit "
            "interval cannot be compared across samples or across metrics."
        )
    return number


# ─────────────────────────────────────────────────────────────────────────────
# Generic (de)serialisation
# ─────────────────────────────────────────────────────────────────────────────


def _unwrap_optional(hint: Any) -> tuple[Any, bool]:
    origin = get_origin(hint)
    if origin is Union or origin is UnionType:
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return hint, False


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    return value


def _decode(value: Any, hint: Any, path: str) -> Any:
    hint, optional = _unwrap_optional(hint)
    if value is None:
        if optional:
            return None
        raise ContractValidationError(f"{path} is required, got null")

    origin = get_origin(hint)
    if origin in (tuple, list):
        # ``tuple[str, ...]`` gives (str, Ellipsis); only the item type matters
        # here, since no contract field is a fixed-length heterogeneous tuple.
        args = get_args(hint)
        item_hint = args[0] if args else Any
        if not isinstance(value, (list, tuple)):
            raise ContractValidationError(f"{path} must be a JSON array, got {type(value).__name__}")
        items = [_decode(v, item_hint, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ContractValidationError(f"{path} must be a JSON object, got {type(value).__name__}")
        if not all(isinstance(k, str) for k in value):
            raise ContractValidationError(f"{path} has non-string keys; JSON object keys are strings")
        return dict(value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        return _require_enum(path, value, hint)
    if hint is str:
        return _require_text(path, value, allow_empty=True)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractValidationError(f"{path} must be an integer, got {type(value).__name__}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractValidationError(f"{path} must be a number, got {type(value).__name__}")
        return float(value)
    if hint is bool:
        if not isinstance(value, bool):
            raise ContractValidationError(f"{path} must be a boolean, got {type(value).__name__}")
        return value
    return value


@dataclass(frozen=True)
class ContractEntity:
    """Base for every record in PRD 7.2.

    Subclasses declare plain dataclass fields; serialisation, strictness and
    hashing come from here. ``ENTITY`` names the entity in exchange envelopes and
    in audit records, so renaming a Python class does not silently rename a
    persisted record type.
    """

    ENTITY: ClassVar[str] = "entity"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view. Enum members become their string values."""
        return {f.name: _encode(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls: type[_T], payload: Mapping[str, Any]) -> _T:
        """Rebuild from a JSON object, rejecting unknown and missing keys."""
        if not isinstance(payload, Mapping):
            raise ContractValidationError(
                f"{cls.__name__}.from_dict expects a mapping, got {type(payload).__name__}"
            )
        hints = get_type_hints(cls)
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ContractValidationError(
                f"{cls.__name__} has no field(s) {sorted(unknown)}. Unknown keys are "
                "rejected rather than ignored: a tolerant reader turns a rename "
                "into silently defaulted data. Known fields: " + str(sorted(known))
            )
        missing = sorted(known - set(payload))
        if missing:
            raise ContractValidationError(f"{cls.__name__} is missing required field(s) {missing}")
        kwargs = {name: _decode(payload[name], hints[name], f"{cls.__name__}.{name}") for name in known}
        return cls(**kwargs)

    def content_hash(self) -> str:
        """SHA-256 over the canonical encoding of :meth:`to_dict`."""
        return content_hash(self.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# 1. AnnotationProtocol
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnnotationProtocol(ContractEntity):
    """Versioned annotation rules (PRD 7.2, 7.4).

    ``hash`` covers the *rules only* — not ``id``, not ``version``, not
    ``effective_from``. Two consequences, both intentional:

    * Bumping ``version`` without changing a rule does not change the hash, so a
      reviewer can see that the version number moved and the semantics did not.
      Re-labelling a dataset "v2" to make it look newer is then visible in one
      column.
    * Changing a rule without bumping the version *does* change the hash, and
      ``__post_init__`` refuses to build the object with a stale hash. That is
      the check which keeps a taxonomy change from being applied to
      already-annotated samples under an old protocol name.

    ``touching_object_rule`` and ``overlap_split_rule`` are open-vocabulary
    strings; see ``contracts/enums.py`` on why they are not enumerated.
    """

    ENTITY: ClassVar[str] = "annotation_protocol"

    id: str
    version: str
    taxonomy_name: str
    touching_object_rule: str
    min_visible_boundary_um: float
    overlap_split_rule: str
    effective_from: str
    hash: str = ""

    def __post_init__(self) -> None:
        for name in ("id", "version", "taxonomy_name", "touching_object_rule", "overlap_split_rule"):
            _require_text(name, getattr(self, name))
        if isinstance(self.min_visible_boundary_um, bool) or not isinstance(
            self.min_visible_boundary_um, (int, float)
        ):
            raise ContractValidationError("min_visible_boundary_um must be a number")
        boundary = float(self.min_visible_boundary_um)
        if boundary < 0:
            raise ContractValidationError(
                f"min_visible_boundary_um must be non-negative, got {boundary}"
            )
        object.__setattr__(self, "min_visible_boundary_um", boundary)
        _require_iso("effective_from", self.effective_from)

        computed = self.rule_hash()
        if not self.hash:
            object.__setattr__(self, "hash", computed)
        elif self.hash != computed:
            raise ContractValidationError(
                f"AnnotationProtocol.hash = {self.hash[:16]}… does not match the rules "
                f"in this record ({computed[:16]}…). Either the rules were edited "
                "without recomputing the hash, or this is a different protocol "
                "claiming to be the same one. Both must fail here, before the "
                "protocol is attached to a task."
            )

    def rule_payload(self) -> dict[str, Any]:
        """The rule set the hash covers. Kept separate so it can be diffed."""
        return {
            "taxonomy_name": self.taxonomy_name,
            "touching_object_rule": self.touching_object_rule,
            "min_visible_boundary_um": self.min_visible_boundary_um,
            "overlap_split_rule": self.overlap_split_rule,
        }

    def rule_hash(self) -> str:
        """Hash of :meth:`rule_payload`."""
        return content_hash(self.rule_payload())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Task
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Task(ContractEntity):
    """The root of one workflow instance (PRD 7.2).

    ``image_scope`` may name more than one sample, while the state machine of
    PRD 4.8 is defined per sample. Rather than invent a tenth entity to hold
    per-sample state, ``state`` is the **aggregate**: the state of the
    least-advanced sample in scope, computed by
    :func:`analysis.contracts.state_machine.aggregate_task_state`. That keeps
    both PRD statements true — the state machine is sample-level, and the task
    still has the ``state`` field 7.2 requires — without adding a row type that
    the PRD does not define.
    """

    ENTITY: ClassVar[str] = "task"

    id: str
    node_id: str
    project_id: str
    template: str
    image_scope: tuple[str, ...]
    redundancy_n: int
    preannotation_policy: str
    protocol_id: str
    state: SampleState
    deadline: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("id", "node_id", "project_id", "protocol_id", "preannotation_policy"):
            _require_text(name, getattr(self, name))
        # Validated against the *registry*, not a closed enum: a template this
        # node has not registered cannot be projected or gated, so refusing it
        # here is what stops an unknown orchestration from being persisted.
        template_id = normalise_template_id(_require_text("template", self.template))
        try:
            get_template(template_id)
        except TemplateError as exc:
            raise ContractValidationError(str(exc)) from exc
        object.__setattr__(self, "template", template_id)
        object.__setattr__(self, "state", _require_enum("state", self.state, SampleState))

        scope = self.image_scope
        if isinstance(scope, str) or not isinstance(scope, (tuple, list)):
            raise ContractValidationError("image_scope must be a tuple of sample ids")
        scope = tuple(scope)
        if not scope:
            raise ContractValidationError(
                "image_scope must name at least one sample; a task over nothing "
                "can never leave DRAFT and would show up as permanently stuck"
            )
        for i, sample_id in enumerate(scope):
            _require_text(f"image_scope[{i}]", sample_id)
        if len(set(scope)) != len(scope):
            raise ContractValidationError(f"image_scope has duplicates: {sorted(scope)}")
        object.__setattr__(self, "image_scope", scope)

        _require_uint("redundancy_n", self.redundancy_n, minimum=1)
        if not 1 <= self.redundancy_n <= 5:
            raise ContractValidationError(
                f"redundancy_n must lie in 1..5 (D-07 allows 2..5 for research; "
                f"the teaching template uses 1), got {self.redundancy_n}"
            )
        _require_optional_iso("deadline", self.deadline)

    @property
    def is_multi_sample(self) -> bool:
        """True when the task spans more than one sample."""
        return len(self.image_scope) > 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Assignment
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Assignment(ContractEntity):
    """One person's obligation on one sample (PRD 7.2)."""

    ENTITY: ClassVar[str] = "assignment"

    id: str
    task_id: str
    sample_id: str
    assignee_id: str
    role: Role
    state: AssignmentState
    started_at: Optional[str] = None
    submitted_at: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("id", "task_id", "sample_id", "assignee_id"):
            _require_text(name, getattr(self, name))
        role = _require_enum("role", self.role, Role)
        if role not in (Role.DOMAIN_EXPERT, Role.ANNOTATOR):
            # A reviewer's or cosigner's output is a verdict, which lives on
            # Cosign / Adjudication. Letting those roles hold an Assignment too
            # would make "assigned + one revision" ambiguous for exactly the
            # people whose job is *not* to produce a redundant revision.
            raise ContractValidationError(
                f"Assignment.role must be DOMAIN_EXPERT or ANNOTATOR (the roles that "
                f"produce a Revision), got {role.value}"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "state", _require_enum("state", self.state, AssignmentState))
        _require_optional_iso("started_at", self.started_at)
        _require_optional_iso("submitted_at", self.submitted_at)
        if self.submitted_at is not None:
            if self.started_at is None:
                raise ContractValidationError("submitted_at is set but started_at is not")
            if _parse_iso("submitted_at", self.submitted_at) < _parse_iso("started_at", self.started_at):
                raise ContractValidationError(
                    "submitted_at precedes started_at; one of the two timestamps is "
                    "from the wrong clock"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Revision
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Revision(ContractEntity):
    """One complete annotation product (PRD 7.2, 7.3).

    ``payload_ref`` points at the annotation payload (an object-store key or a
    content-addressed blob id); the payload itself never enters this record, and
    masks stay out of metadata. ``payload_hash`` is the P-06 anchor: a
    supplement is a new ``Revision`` whose ``derived_from`` names its parent, so
    the blind read's hash is still verifiable after the supplement exists.

    ``derived_from`` is deliberately *not* ``parent_instance_id``. PRD 7.3 is
    explicit that instance lineage (budding, division) and revision lineage
    (draft 2 from draft 1) must never share a field, because the tracking module
    would then read a second revision as a new biological instance.
    """

    ENTITY: ClassVar[str] = "revision"

    id: str
    sample_id: str
    assignment_id: str
    author_id: str
    revision_no: int
    is_blind_read: bool
    payload_ref: str
    payload_hash: str
    created_at: str
    derived_from: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("id", "sample_id", "assignment_id", "author_id", "payload_ref"):
            _require_text(name, getattr(self, name))
        _require_uint("revision_no", self.revision_no, minimum=1)
        if not isinstance(self.is_blind_read, bool):
            raise ContractValidationError("is_blind_read must be a boolean")
        if not is_content_hash(self.payload_hash):
            raise ContractValidationError(
                f"payload_hash = {self.payload_hash!r} is not a lowercase hex SHA-256. "
                "Use analysis.contracts.hashing.content_hash to produce it."
            )
        _require_iso("created_at", self.created_at)
        if self.derived_from is not None:
            _require_text("derived_from", self.derived_from)
            if self.derived_from == self.id:
                raise ContractValidationError("Revision.derived_from cannot name itself")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Disagreement
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Disagreement(ContractEntity):
    """Disagreement analysis result (PRD 7.2).

    ``metric_snapshot`` is deliberately an open mapping: the metric set is
    versioned by the snapshot itself, not by this contract. ``computed_at``
    opens the window in which peer revisions become visible to the chief
    (PRD 3.2: "仅分歧分析后"), so its presence is a projection input, not just
    a report field.
    """

    ENTITY: ClassVar[str] = "disagreement"

    id: str
    sample_id: str
    metric_snapshot: Mapping[str, Any]
    regions_ref: str
    controversy_score: float
    computed_at: str

    def __post_init__(self) -> None:
        for name in ("id", "sample_id", "regions_ref"):
            _require_text(name, getattr(self, name))
        if not isinstance(self.metric_snapshot, Mapping):
            raise ContractValidationError("metric_snapshot must be a mapping")
        if not self.metric_snapshot:
            raise ContractValidationError(
                "metric_snapshot is empty. A disagreement record with no metric "
                "cannot support a threshold decision, and an empty snapshot "
                "passes every 'is it computed' check in the codebase."
            )
        _require_score("controversy_score", self.controversy_score)
        _require_iso("computed_at", self.computed_at)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ModelRank
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelRank(ContractEntity):
    """Model-proposed ordering of a sample's revisions (PRD 7.2, D-04).

    Advisory by construction: nothing in this record can select a revision. The
    projection layer is what stops it reaching a domain expert at all, since an
    ordering is enough to move a human choice — that is the anchoring effect
    D-03 exists to prevent, and the MVP audit found it leaking three separate
    ways (L1-1, L1-2, H-4).
    """

    ENTITY: ClassVar[str] = "model_rank"

    id: str
    sample_id: str
    model_id: str
    model_version: str
    ranking: tuple[str, ...]
    scores: Mapping[str, float]
    confidence: float
    produced_at: str

    def __post_init__(self) -> None:
        for name in ("id", "sample_id", "model_id", "model_version"):
            _require_text(name, getattr(self, name))
        ranking = self.ranking
        if isinstance(ranking, str) or not isinstance(ranking, (tuple, list)):
            raise ContractValidationError("ranking must be a tuple of revision ids")
        ranking = tuple(ranking)
        if not ranking:
            raise ContractValidationError("ranking must name at least one revision")
        if len(set(ranking)) != len(ranking):
            raise ContractValidationError(f"ranking repeats a revision id: {list(ranking)}")
        object.__setattr__(self, "ranking", ranking)
        if not isinstance(self.scores, Mapping):
            raise ContractValidationError("scores must be a mapping of revision id -> score")
        for key, value in self.scores.items():
            if not isinstance(key, str) or not key.strip():
                raise ContractValidationError(
                    f"scores has a non-string or empty key {key!r}; revision ids are strings"
                )
            _require_score(f"scores[{key}]", value)
        object.__setattr__(self, "scores", dict(self.scores))
        _require_score("confidence", self.confidence)
        _require_iso("produced_at", self.produced_at)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Adjudication
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Adjudication(ContractEntity):
    """The signed decision (PRD 7.2, P-03, P-05).

    ``signature_meaning`` is required and non-empty because P-05 asks the final
    label to answer "谁、何时、依据什么、是否采纳模型建议". A boolean
    ``adopted_model_rank`` would answer only the last question and answer it
    ambiguously ("adopted" can mean agreed with, not followed). The free-text
    meaning is what a publication's methods section can quote.
    """

    ENTITY: ClassVar[str] = "adjudication"

    id: str
    sample_id: str
    selected_revision_id: str
    selector_id: str
    reason_code: str
    reason_text: str
    evidence_ref: str
    signature_meaning: str
    decided_at: str

    def __post_init__(self) -> None:
        for name in (
            "id",
            "sample_id",
            "selected_revision_id",
            "selector_id",
            "evidence_ref",
            "signature_meaning",
        ):
            _require_text(name, getattr(self, name))
        # P-03: no reason, no adjudication. Both halves are required — a code
        # alone is unreadable to a reviewer, free text alone is unaggregatable.
        _require_text("reason_code", self.reason_code)
        _require_text("reason_text", self.reason_text)
        _require_iso("decided_at", self.decided_at)

    @property
    def adopts_model_rank(self) -> Optional[bool]:
        """True/False when the signature states it, ``None`` when it does not.

        Narrative convention, not parsed prose: the projection layer surfaces
        this so an operator can see how often the chief follows the model
        without re-reading every signature.
        """
        lowered = self.signature_meaning.lower()
        if "未采纳" in lowered or "not adopted" in lowered:
            return False
        if "采纳" in lowered or "adopted" in lowered:
            return True
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 8. Cosign
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cosign(ContractEntity):
    """Conditional second signature (PRD 7.2, D-09, P-09).

    The ordering invariants here *are* P-09. ``blind_choice`` is recorded with
    ``blind_choice_locked_at``, and ``revealed_at`` may not exist before that
    lock. A cosigner who can see the chief's conclusion before committing their
    own is not a second reader; they are a confirmation dialog. The MVP audit
    found exactly this defect (L1-1) and the fix was a UI gate — which is why
    the contract now refuses to construct the record out of order, so the gate
    cannot be forgotten in a second surface.
    """

    ENTITY: ClassVar[str] = "cosign"

    id: str
    sample_id: str
    adjudication_id: str
    trigger_rule: str
    cosigner_id: str
    blind_choice: Optional[str]
    blind_choice_locked_at: Optional[str]
    verdict: Optional[CosignVerdict]
    dissent_text: Optional[str]
    revealed_at: Optional[str]
    decided_at: Optional[str]

    def __post_init__(self) -> None:
        for name in ("id", "sample_id", "adjudication_id", "trigger_rule", "cosigner_id"):
            _require_text(name, getattr(self, name))
        _require_optional_iso("blind_choice_locked_at", self.blind_choice_locked_at)
        _require_optional_iso("revealed_at", self.revealed_at)
        _require_optional_iso("decided_at", self.decided_at)
        if self.verdict is not None:
            object.__setattr__(self, "verdict", _require_enum("verdict", self.verdict, CosignVerdict))

        if self.blind_choice is not None:
            _require_text("blind_choice", self.blind_choice)
        if (self.blind_choice is None) != (self.blind_choice_locked_at is None):
            raise ContractValidationError(
                "blind_choice and blind_choice_locked_at must be set together. A "
                "choice without a lock time cannot prove it preceded the reveal; a "
                "lock time without a choice records nothing."
            )
        if self.revealed_at is not None and self.blind_choice_locked_at is None:
            raise ContractValidationError(
                "revealed_at is set but no blind choice was locked (P-09). The "
                "cosigner must commit an independent choice before the chief's "
                "conclusion is shown. If this record is being built by replaying "
                "events, the events are out of order."
            )
        if self.revealed_at is not None and self.decided_at is not None:
            if _parse_iso("decided_at", self.decided_at) < _parse_iso("revealed_at", self.revealed_at):
                raise ContractValidationError("decided_at precedes revealed_at")
        if self.decided_at is not None and self.verdict is None:
            raise ContractValidationError("decided_at is set but verdict is empty")
        if self.verdict is CosignVerdict.REJECT:
            # Symmetric with P-03. A rejection with no stated dissent gives the
            # chief nothing to act on and the audit nothing to review.
            _require_text("dissent_text", self.dissent_text or "")


def validate_cosign_independence(adjudication: Adjudication, cosign: Cosign) -> None:
    """Reject a cosign whose signer is the adjudicating chief (PRD 3.1, D-09).

    Also checks that the two records describe the same sample and that the
    cosign points at *this* adjudication. Those are the two joins a replay from
    an envelope could get wrong without any single record looking invalid.
    """
    if cosign.cosigner_id == adjudication.selector_id:
        raise ContractValidationError(
            f"cosigner_id == selector_id ({adjudication.selector_id}); 复核专家不得与本案首席为"
            "同一人 (PRD 3.1). Self-cosign is not a second opinion."
        )
    if cosign.sample_id != adjudication.sample_id:
        raise ContractValidationError(
            f"Cosign.sample_id {cosign.sample_id!r} != Adjudication.sample_id "
            f"{adjudication.sample_id!r}"
        )
    if cosign.adjudication_id != adjudication.id:
        raise ContractValidationError(
            f"Cosign.adjudication_id {cosign.adjudication_id!r} does not name "
            f"adjudication {adjudication.id!r}"
        )


def validate_model_rank(rank: ModelRank, revisions: tuple[Revision, ...]) -> None:
    """A model ranking must permute exactly the sample's **blind** revisions.

    Blind revisions are the candidates: the ranking exists to order the N
    independent reads for the chief, not to order every later draft. Supplements
    (``revision_no >= 2``, ``is_blind_read=False``) are produced *after* the
    reveal and are not alternatives the chief chooses between, so requiring them
    in the ranking would make a ranking that was correct when produced
    permanently invalid — every supplement would look like a data error.

    Two failure modes this catches, both silent otherwise:

    * a ranking that names a revision from another sample — the classic
      cross-sample carry-over (MVP defect L2-1, where a global draft singleton
      moved geometry between samples);
    * a ranking that omits a blind revision — the UI then renders N-1 candidates
      and the missing one looks like it was judged unusable.
    """
    candidates = tuple(r for r in revisions if r.is_blind_read)
    sample_ids = {r.sample_id for r in revisions}
    if sample_ids and sample_ids != {rank.sample_id}:
        raise ContractValidationError(
            f"ModelRank.sample_id {rank.sample_id!r} but the supplied revisions belong "
            f"to {sorted(sample_ids)}"
        )
    ids = {r.id for r in candidates}
    if set(rank.ranking) != ids:
        missing = sorted(ids - set(rank.ranking))
        extra = sorted(set(rank.ranking) - ids)
        raise ContractValidationError(
            f"ranking is not a permutation of the sample's blind revisions "
            f"(missing={missing}, unknown={extra})"
        )
    unknown_scores = sorted(set(rank.scores) - ids)
    if unknown_scores:
        raise ContractValidationError(f"scores name unknown revision(s) {unknown_scores}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. StratumLabel
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StratumLabel(ContractEntity):
    """Difficulty stratum, assigned at ingestion (PRD 7.2, D-08).

    No default stratum, on purpose. D-08 requires that a missing label fail
    ingestion rather than default to "easy": defaulting silently puts the
    hardest cases into the easy stratum, the held-out set loses exactly the
    samples that carry the effect being measured, and nothing reports a
    problem because every row still has a stratum.
    """

    ENTITY: ClassVar[str] = "stratum_label"

    sample_id: str
    stratum: Stratum
    signal_snapshot: Mapping[str, Any]
    labeled_at: str
    rule_version: str

    def __post_init__(self) -> None:
        _require_text("sample_id", self.sample_id)
        object.__setattr__(self, "stratum", _require_enum("stratum", self.stratum, Stratum))
        if not isinstance(self.signal_snapshot, Mapping):
            raise ContractValidationError("signal_snapshot must be a mapping")
        if not self.signal_snapshot:
            raise ContractValidationError(
                "signal_snapshot is empty. A stratum without the signals that "
                "produced it cannot be re-adjudicated when the rule changes "
                "(Q-07 is still open, so this will happen)."
            )
        object.__setattr__(self, "signal_snapshot", dict(self.signal_snapshot))
        _require_iso("labeled_at", self.labeled_at)
        _require_text("rule_version", self.rule_version)
