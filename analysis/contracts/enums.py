# ── analysis/contracts/enums.py ──
"""Closed vocabularies shared by every contract entity.

Why this module exists
----------------------
The interactive MVP carried its state names and role names as string literals
scattered through ``app.js``. That was acceptable for a demo and fatal for a
contract: a typo in one branch (``"AWAITING_REVEIW"``) does not raise, it simply
falls through to "unknown state", and the resulting audit record is
unverifiable. Every value that crosses a module boundary — and every value that
reaches the audit chain — is therefore an enum member here, defined once.

Scope discipline
----------------
This module imports nothing from the rest of the repository. The contract layer
must stay importable from a bare interpreter, from the future FastAPI service,
and from the test suite without dragging in torch, pandas or the import-order
guard those two need (see ``utils/import_order.py`` for why that matters).

What is *not* enumerated, and why
--------------------------------
Two vocabularies stay open on purpose:

* **Adjudication reason codes.** The PRD fixes the fields (7.2) but never fixes
  the code list, and a closed list invented here would silently reject a valid
  clinical reason in the first customer deployment. The contract therefore
  enforces *shape* — a non-empty code plus non-empty free text (P-03) — and
  leaves the vocabulary to the methodology owner. See ``REASON_CODE_SUGGESTIONS``
  for the seed list a UI should offer.
* **Protocol rule strings** (``touching_object_rule``, ``overlap_split_rule``).
  Same reasoning: the contract hashes them, so any change is detectable, but it
  does not pretend to know the domain's rule set.

The one place an open vocabulary would be unsafe is label identity, which is
already solved upstream by the taxonomy registry (``utils/taxonomy.py``) and is
referenced here by name rather than re-declared.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "REASON_CODE_SUGGESTIONS",
    "AssignmentState",
    "CosignVerdict",
    "EdgeKind",
    "ProjectionStage",
    "Role",
    "SampleState",
    "Stratum",
    "Template",
]


class Template(str, Enum):
    """The two templates the product ships (PRD 2.3, D-05/D-10).

    **Not** the type of ``Task.template``. That field is a validated string,
    because the template set has to be open: a federated node may register a
    template of its own, and an envelope arriving from a peer must be able to
    name it. A closed enum in the record would make the exchange format unable
    to represent a peer's orchestration without a code release — which is a
    federation problem dressed up as type safety. Registration in
    ``contracts.templates.TEMPLATES`` is what validates the value; these members
    are the two canonical ids, for readable code.

    Persisted in every ``Task`` and in every export manifest, because the export
    gate has to answer "where did this product come from" without consulting the
    task table (P-11).
    """

    RESEARCH = "research"
    TEACHING = "teaching"


class Role(str, Enum):
    """The nine roles of PRD 3.1, plus the platform's own actor.

    ``SYSTEM`` exists because several transitions are *automatic* — reveal once
    all blind reads are in (4.8), disagreement analysis after the third
    submission (P-02), gate evaluation after signature. An audit record for
    those transitions still needs an actor, and recording them under the
    operator who happened to be logged in would be a false attribution. The
    audit chain distinguishes ``SYSTEM`` from every human role precisely so a
    reviewer can tell "nobody pressed this" from "somebody did".
    """

    NODE_ADMIN = "node_admin"
    PROJECT_OWNER = "project_owner"
    DATA_MANAGER = "data_manager"
    DOMAIN_EXPERT = "domain_expert"
    CHIEF_EXPERT = "chief_expert"
    COSIGNER = "cosigner"
    REVIEWER = "reviewer"
    ANNOTATOR = "annotator"
    OBSERVER = "observer"
    SYSTEM = "system"

    @property
    def is_human(self) -> bool:
        """False only for :attr:`SYSTEM`, which has no session and no signature."""
        return self is not Role.SYSTEM


class SampleState(str, Enum):
    """The eleven states of PRD 4.8, in workflow order.

    **Spec deviation, deliberate.** PRD 4.8 says "十状态" and G-4 says "90 有序
    对穷举测试（12 合法 + 78 非法）", but the table in the same section uses
    ``IN_DATASET`` as the exit of ``ADJUDICATED`` without listing it as a state.
    Counting the states actually written down gives 10, counting the states
    actually *referenced* gives 11, and counting the transitions gives 12 legal
    edges — and 12 legal edges is only consistent with 11 states. The arithmetic
    in G-4 cannot be satisfied by the table it describes.

    This module takes the 11-state reading, because ``IN_DATASET`` carries
    semantics nothing else carries: it is the only state in which a sample may
    appear in an export manifest, and P-10 (stratum counts, absolute case
    numbers) is unanswerable without it. Collapsing it into ``ADJUDICATED``
    would make "signed" and "exportable" the same predicate, which is exactly
    the conflation that lets an unsigned sample reach a training set.

    Consequence: 11 states give 110 ordered pairs (n·(n-1)), not 90. The
    ``state_machine`` module publishes the real counts and the test suite asserts
    them, so the discrepancy is visible in CI rather than buried here. PRD 4.8
    and G-4 need a one-line correction; that is a documentation edit and does not
    change any behaviour.
    """

    DRAFT = "draft"
    PUBLISHED = "published"
    IN_BLIND_READ = "in_blind_read"
    AWAITING_REVEAL = "awaiting_reveal"
    IN_REVIEW = "in_review"
    AWAITING_ADJUDICATION = "awaiting_adjudication"
    AWAITING_COSIGN = "awaiting_cosign"
    ADJUDICATED = "adjudicated"
    IN_DATASET = "in_dataset"
    NEEDS_IMPROVEMENT = "needs_improvement"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        """True where the sample's life in this task ends.

        ``IN_DATASET`` and ``REJECTED`` are the only two: a rejected sample is
        never reopened (4.9 — "只能新开一轮（新 Task）"), and a dataset member is
        immutable, since mutating it would invalidate every exported snapshot
        hash that already references it.
        """
        return self in (SampleState.IN_DATASET, SampleState.REJECTED)

    @property
    def is_awaiting_human(self) -> bool:
        """True when the next move requires a person, not the platform."""
        return self in (
            SampleState.IN_BLIND_READ,
            SampleState.AWAITING_ADJUDICATION,
            SampleState.AWAITING_COSIGN,
        )


class AssignmentState(str, Enum):
    """Per-expert state of one assignment (PRD 4.8, assignment level).

    **Spec ambiguity, resolved.** 4.8 lists the chain as
    ``ASSIGNED → BLIND_DRAFTED → REVEALED → SUBMITTED → LOCKED`` but states in
    the same paragraph that ``REVEALED`` is granted when the *task* enters
    ``IN_REVIEW``, i.e. only once all N submissions are in. Those two statements
    cannot both hold. The declaration order of this enum is the literal PRD
    order; the transition table in ``state_machine`` implements
    ``BLIND_DRAFTED → SUBMITTED → REVEALED → LOCKED``, because submitting before
    revealing is the whole point of P-01. Enforcing the literal order would let
    an expert see the pre-annotation before they had committed a blind read.
    """

    ASSIGNED = "assigned"
    BLIND_DRAFTED = "blind_drafted"
    REVEALED = "revealed"
    SUBMITTED = "submitted"
    LOCKED = "locked"


class EdgeKind(str, Enum):
    """Why two states may be adjacent.

    Workflow edges are the twelve in PRD 4.8. ``ABORT`` covers administrative
    termination — 4.8 gives ``REJECTED`` the trigger "数据或协议问题 → 终止" but
    never says which states may reach it, so a strict reading leaves
    ``REJECTED`` unreachable, which would make the state a decoration. The two
    classes are kept distinct rather than merged into "legal transitions" so
    that a workflow change and a termination-policy change are separately
    reviewable, and so the PRD's "12 合法" count remains checkable.
    """

    WORKFLOW = "workflow"
    ABORT = "abort"


class ProjectionStage(str, Enum):
    """The coarse stage a projection is computed for (G-3).

    Derived from :class:`SampleState`, never supplied by a client. Three stages
    rather than eleven because the visibility rules change at exactly two
    moments: the reveal (pre-annotation becomes visible) and the signature
    (the majority disappears, one revision becomes *the* answer).
    """

    BLIND = "blind"
    REVEALED = "revealed"
    FINAL = "final"

    @classmethod
    def of(cls, state: SampleState) -> ProjectionStage:
        """Map a sample state onto the stage whose rules apply.

        ``NEEDS_IMPROVEMENT`` maps back to :attr:`BLIND`: the sample re-enters
        the blind-read loop (4.8), so everything the reveal made visible must
        be hidden again. Mapping it to :attr:`REVEALED` would leak the previous
        round's pre-annotations into the next round's blind read, which is the
        same class of defect as H-1/H-2 in the MVP audit.
        """
        if state in (
            SampleState.DRAFT,
            SampleState.PUBLISHED,
            SampleState.IN_BLIND_READ,
            SampleState.AWAITING_REVEAL,
            SampleState.NEEDS_IMPROVEMENT,
        ):
            return ProjectionStage.BLIND
        if state in (
            SampleState.IN_REVIEW,
            SampleState.AWAITING_ADJUDICATION,
            SampleState.AWAITING_COSIGN,
            SampleState.ADJUDICATED,
        ):
            return ProjectionStage.REVEALED
        return ProjectionStage.FINAL


class CosignVerdict(str, Enum):
    """Outcome of conditional cosign (PRD 4.8, D-09)."""

    APPROVE = "approve"
    REJECT = "reject"


class Stratum(str, Enum):
    """Difficulty stratum of a sample (PRD 7.2, D-08).

    Four values, and *no* default. ``StratumLabel`` therefore cannot be
    constructed without an explicit stratum, which is how "缺失则入湖失败而非
    默认归入容易层" is enforced structurally instead of by convention: there is
    no ``EASY`` fallback to fall into.
    """

    EASY = "easy"
    BORDERLINE = "borderline"
    LOW_CONTRAST = "low_contrast"
    CLUSTERED = "clustered"


#: Seed vocabulary a UI should offer for ``Adjudication.reason_code``.
#:
#: Deliberately *not* an enum. PRD 7.2 requires the field but no section fixes
#: its values, and this list is a starting point for the methodology owner
#: rather than a contract: ``Adjudication`` accepts any non-empty code, so
#: adding one here later is a UI change, not a migration.
REASON_CODE_SUGGESTIONS: tuple[str, ...] = (
    "majority_agreement",
    "boundary_quality",
    "protocol_application",
    "model_rank_considered",
    "chief_reselection",
    "other",
)
