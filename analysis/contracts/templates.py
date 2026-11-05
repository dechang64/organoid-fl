# ── analysis/contracts/templates.py ──
"""The two orchestration templates of PRD 2.3, expressed as configuration only.

Why this is a module and not an ``if template == "teaching"``
-------------------------------------------------------------
M0's exit criterion is stated as a sentence, not a metric: *"新增一个 template
只改配置与编排、不改数据模型"* (PRD 10.4). A sentence like that is unenforceable
unless the difference between the two templates has a single, inspectable
location. So every difference lives in a :class:`TemplateSpec` field, the state
machine reads those fields instead of branching on the template name, and
``tests/test_contracts_templates.py`` registers a *third* template from
configuration alone and runs the full state machine and exchange round-trip
through it. If adding that third template ever requires touching ``entities.py``
or ``state_machine.py``, that test is where it will hurt.

What is actually different
--------------------------
Not the data model: both templates produce the same ``Task``, ``Assignment``,
``Revision``, ``Disagreement``, ``ModelRank``, ``Adjudication``. What differs is
orchestration:

====================  ==========================  ==========================
field                 research                    teaching
====================  ==========================  ==========================
redundancy             2..5, default 3 (D-07)      1 (teacher replaces it)
blind read             required                    required
agreement              export gate (6.2)           scoring dimension only
cosign                 available, default off      not available
training export        allowed                     **blocked** (D-10/P-11)
revision author        DOMAIN_EXPERT               ANNOTATOR
signer                 CHIEF_EXPERT                REVIEWER
====================  ==========================  ==========================

The training-export row is the one that must never be runtime-configurable in
practice: D-10 says student annotations may not enter a training set, and a
setting an operator can flip is a setting an operator will flip. It stays a field
here — so the rule is visible in one place — but the gate
(``exchange.check_training_export``) refuses on the template value itself, and no
per-task override exists to defeat it.

``SYSTEM`` is never cast
------------------------
Several steps are automatic: reveal once all blind reads are in (4.8),
disagreement analysis after the last submission (P-02), gate evaluation after
signature, triggering cosign when the rule fires. Those cast *no* human role —
an empty ``frozenset`` means "the platform does this, no person may". Letting a
template name ``SYSTEM`` in a cast would let it claim a human step that no human
can perform, so :meth:`TemplateSpec.__post_init__` rejects it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar, Mapping

from analysis.contracts.enums import Role, SampleState, Template

__all__ = [
    "OPERATIONS",
    "RESEARCH_TEMPLATE",
    "TEACHING_TEMPLATE",
    "TEMPLATES",
    "TemplateError",
    "TemplateSpec",
    "get_template",
    "normalise_template_id",
    "register_template",
]


class TemplateError(ValueError):
    """A template specification is internally inconsistent or already taken."""


def _cast(overrides: Mapping[str, tuple[Role, ...]]) -> Mapping[str, frozenset[Role]]:
    """Build a complete cast table from a base plus overrides."""
    cast = {operation: frozenset() for operation in OPERATIONS}
    cast.update({operation: frozenset(members) for operation, members in overrides.items()})
    unknown = set(overrides) - set(OPERATIONS)
    if unknown:  # pragma: no cover - programming error, caught at import
        raise TemplateError(f"cast names unknown operation(s) {sorted(unknown)}")
    return cast


#: Every operation the state machine can be asked to perform. Every template
#: casts all of them — an absent key would raise ``KeyError`` inside a
#: transition, which reads as a bug rather than as "this template has no such
#: step". The declaration order follows the workflow.
OPERATIONS: tuple[str, ...] = (
    "publish",
    "begin_blind_read",
    "complete_reveal",
    "begin_review",
    "request_adjudication",
    "adjudicate",
    "request_cosign",
    "cosign_approve",
    "cosign_reject",
    "admit_to_dataset",
    "return_for_improvement",
    "reopen",
    "abort",
)

#: Operations only the platform performs. Named as a constant so a reviewer can
#: see at a glance which transitions nobody can trigger by hand.
SYSTEM_ONLY: frozenset[str] = frozenset(
    {
        "complete_reveal",
        "begin_review",
        "request_adjudication",
        "request_cosign",
        "admit_to_dataset",
        "return_for_improvement",
    }
)


@dataclass(frozen=True)
class TemplateSpec:
    """Everything that differs between two orchestration templates."""

    #: Persisted id of this template. A plain string, not the ``Template`` enum:
    #: the set is open so a peer node can register its own (see
    #: ``enums.Template``). Registration is what validates it.
    template: str
    label: str
    redundancy_range: tuple[int, int]
    default_redundancy: int
    blind_read_required: bool
    reveal_required: bool
    agreement_is_gate: bool
    cosign_available: bool
    training_export_allowed: bool
    cast: Mapping[str, frozenset[Role]]
    states: tuple[SampleState, ...]
    notes: str = ""

    #: Marker so a future subclass cannot quietly bypass __post_init__ checks.
    _KIND: ClassVar[str] = "template_spec"

    def __post_init__(self) -> None:
        if not isinstance(self.template, str) or not self.template.strip():
            raise TemplateError(
                f"template must be a non-empty string id, got {type(self.template).__name__}"
            )
        if not self.label.strip():
            raise TemplateError("label must be non-empty")

        low, high = self.redundancy_range
        if not 1 <= low <= high <= 5:
            raise TemplateError(
                f"redundancy_range must satisfy 1 <= low <= high <= 5 (D-07), got {self.redundancy_range}"
            )
        if not low <= self.default_redundancy <= high:
            raise TemplateError(
                f"default_redundancy {self.default_redundancy} is outside the "
                f"declared range {self.redundancy_range}"
            )

        missing = sorted(set(OPERATIONS) - set(self.cast))
        if missing:
            raise TemplateError(
                f"cast is missing operation(s) {missing}. Every operation must be cast "
                "explicitly, even to the empty set, so 'nobody may do this here' is a "
                "decision in the config rather than a KeyError in the machine."
            )
        for operation, members in self.cast.items():
            if not isinstance(members, frozenset):
                raise TemplateError(f"cast[{operation!r}] must be a frozenset of Role")
            for member in members:
                if not isinstance(member, Role):
                    raise TemplateError(f"cast[{operation!r}] contains {member!r}, not a Role")
                if not member.is_human:
                    raise TemplateError(
                        f"cast[{operation!r}] contains SYSTEM. Empty means 'the platform "
                        "does this'; naming SYSTEM in a cast would let a template claim a "
                        "human step no human can perform."
                    )
        for operation in SYSTEM_ONLY:
            if self.cast.get(operation):
                raise TemplateError(
                    f"{operation!r} is a platform step and must cast nobody, got "
                    f"{sorted(r.value for r in self.cast[operation])}"
                )

        if not self.states:
            raise TemplateError("states must list the reachable states")
        if self.states[0] is not SampleState.DRAFT:
            raise TemplateError("states must start at DRAFT")
        if len(set(self.states)) != len(self.states):
            raise TemplateError("states repeats a state")

        if not self.cosign_available:
            for operation in ("request_cosign", "cosign_approve", "cosign_reject"):
                if self.cast.get(operation):
                    raise TemplateError(
                        f"cosign_available is False but {operation!r} casts a role; "
                        "either enable cosign or leave the cast empty"
                    )
            if SampleState.AWAITING_COSIGN in self.states:
                raise TemplateError(
                    "cosign_available is False but AWAITING_COSIGN is reachable; a "
                    "state no edge can enter would show every sample as permanently stuck"
                )

    def allows_redundancy(self, n: int) -> bool:
        """True when ``n`` is a legal redundancy for this template."""
        low, high = self.redundancy_range
        return low <= n <= high

    def roles_for(self, operation: str) -> frozenset[Role]:
        """Who may perform ``operation`` under this template (empty = platform only)."""
        try:
            return self.cast[operation]
        except KeyError as exc:  # pragma: no cover - __post_init__ forbids this
            raise TemplateError(f"template {self.template!r} casts no role for {operation!r}") from exc

    def allows(self, operation: str, role: Role) -> bool:
        """True when ``role`` may perform ``operation`` here."""
        return role in self.cast.get(operation, frozenset())


_STAGES_RESEARCH: tuple[SampleState, ...] = (
    SampleState.DRAFT,
    SampleState.PUBLISHED,
    SampleState.IN_BLIND_READ,
    SampleState.AWAITING_REVEAL,
    SampleState.IN_REVIEW,
    SampleState.AWAITING_ADJUDICATION,
    SampleState.AWAITING_COSIGN,
    SampleState.ADJUDICATED,
    SampleState.NEEDS_IMPROVEMENT,
    SampleState.IN_DATASET,
    SampleState.REJECTED,
)

_STAGES_TEACHING = tuple(s for s in _STAGES_RESEARCH if s is not SampleState.AWAITING_COSIGN)

#: 科研生产模板 — four-level heavy flow (PRD 2.3, 4.8).
RESEARCH_TEMPLATE = TemplateSpec(
    template=Template.RESEARCH,
    label="科研生产",
    redundancy_range=(2, 5),
    default_redundancy=3,
    blind_read_required=True,
    reveal_required=True,
    agreement_is_gate=True,
    cosign_available=True,
    training_export_allowed=True,
    cast=_cast(
        {
            "publish": (Role.PROJECT_OWNER,),
            "begin_blind_read": (Role.DOMAIN_EXPERT,),
            "adjudicate": (Role.CHIEF_EXPERT,),
            "cosign_approve": (Role.COSIGNER,),
            "cosign_reject": (Role.COSIGNER,),
            "reopen": (Role.PROJECT_OWNER,),
            "abort": (Role.PROJECT_OWNER, Role.NODE_ADMIN),
        }
    ),
    states=_STAGES_RESEARCH,
    notes="四级重流程：盲读×N → 揭示 → 分歧 → 仲裁 → 条件复核。产出可辩护金标准。",
)

#: 教学实训模板 — two-level light flow (PRD 2.3, D-10).
TEACHING_TEMPLATE = TemplateSpec(
    template=Template.TEACHING,
    label="教学实训",
    redundancy_range=(1, 1),
    default_redundancy=1,
    blind_read_required=True,
    reveal_required=True,
    agreement_is_gate=False,
    cosign_available=False,
    training_export_allowed=False,
    cast=_cast(
        {
            "publish": (Role.PROJECT_OWNER,),
            "begin_blind_read": (Role.ANNOTATOR,),
            "adjudicate": (Role.REVIEWER,),
            "reopen": (Role.PROJECT_OWNER,),
            "abort": (Role.PROJECT_OWNER, Role.NODE_ADMIN),
        }
    ),
    states=_STAGES_TEACHING,
    notes="两级轻流程：学生初标 → 教师复核 → 评分与反馈。产物禁止进入训练集（D-10/P-11）。",
)

#: Registry. Keys are persisted ids, never display labels.
TEMPLATES: dict[str, TemplateSpec] = {
    Template.RESEARCH: RESEARCH_TEMPLATE,
    Template.TEACHING: TEACHING_TEMPLATE,
}


def normalise_template_id(template: Template | str) -> str:
    """Coerce a template reference to its persisted id.

    Needed because ``Template`` members compare equal to their values but a
    caller may hold either; everything downstream stores the plain string.
    """
    return template.value if isinstance(template, Enum) else str(template)


def get_template(template: Template | str) -> TemplateSpec:
    """Look up a registered template by id (or by ``Template`` member)."""
    key = normalise_template_id(template)
    try:
        return TEMPLATES[key]
    except KeyError as exc:
        raise TemplateError(
            f"{template!r} is not a registered template; known: {sorted(TEMPLATES)}"
        ) from exc


def register_template(spec: TemplateSpec, *, overwrite: bool = False) -> TemplateSpec:
    """Add a template to the registry.

    This is the M0 exit criterion in executable form: a new template arrives as a
    ``TemplateSpec`` and nothing else. ``overwrite`` is off by default for the
    same reason the taxonomy registry defaults it off — two modules registering
    the same name with different rules is a silent semantic fork.
    """
    if not isinstance(spec, TemplateSpec):
        raise TemplateError(f"register_template expects a TemplateSpec, got {type(spec).__name__}")
    key = normalise_template_id(spec.template)
    if key != spec.template:
        # Keep the registry and the record in the same representation, or a
        # later lookup by id silently misses.
        spec = replace(spec, template=key)
    existing = TEMPLATES.get(key)
    if existing is not None and not overwrite:
        if existing != spec:
            raise TemplateError(
                f"template {key!r} is already registered with different "
                "rules; pass overwrite=True only if that is intended"
            )
        return existing
    TEMPLATES[key] = spec
    return spec
