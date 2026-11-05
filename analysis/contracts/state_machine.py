# ── analysis/contracts/state_machine.py ──
"""The sample-level state machine of PRD 4.8, as a data table.

Why a table instead of ``set_state()`` methods
----------------------------------------------
The MVP carried its transitions as branches inside view functions. That has two
consequences the audit found: the set of reachable states depended on which
buttons the UI had rendered (H-5: a domain expert could reach the arbitration
view because only the cosigner's view was guarded), and no test could enumerate
what *should* be impossible, because "impossible" was the absence of a code path
rather than a declared fact.

Here the machine is a literal list of edges. ``IllegalTransition`` is raised for
every ordered pair not in the list, which makes G-4's exhaustive test meaningful:
it enumerates all 110 ordered pairs and checks each against the table, so a new
state or a deleted edge cannot pass unnoticed.

The three kinds of "no", kept distinct
--------------------------------------
A rejected transition has three different causes and an operator needs to know
which one fired:

* :class:`IllegalTransition` — no such edge exists from this state (P-07);
* :class:`GuardRejected` — the edge exists but its precondition is false, and
  the message carries the evidence that was evaluated (§8.1 "复核：触发规则与其
  求值依据");
* :class:`ActorNotPermitted` — the edge and precondition are fine, this person
  may not perform the step under this template (G-2, the server-side check the
  MVP lacked).

Collapsing them into "conflict" is how a permission bug gets logged as a state
bug and investigated by the wrong person.

AWAITING_COSIGN is reachable but dormant
----------------------------------------
D-09 requires the state to exist in M0 while the trigger rule ships disabled. So
the two edges touching it are present in the table, but ``adjudicate`` requires
``not (cosign_available and cosign_enabled and cosign_triggered)``, and
``request_cosign`` requires all three. With ``cosign_enabled=False`` —
``TransitionFacts`` defaults it to False — no path leads in, and
``test_cosign_is_unreachable_with_defaults`` walks the graph to prove it rather
than asserting it in prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

from analysis.contracts.enums import AssignmentState, CosignVerdict, EdgeKind, Role, SampleState
from analysis.contracts.hashing import content_hash
from analysis.contracts.templates import TemplateSpec

__all__ = [
    "ABORT_EDGES",
    "ASSIGNMENT_EDGES",
    "GUARDS",
    "MACHINES",
    "WORKFLOW_EDGES",
    "ActorNotPermitted",
    "Edge",
    "GuardRejected",
    "IllegalTransition",
    "Machine",
    "TransitionError",
    "TransitionFacts",
    "TransitionResult",
    "advance_assignment",
    "aggregate_task_state",
    "assignment_machine_report",
    "sample_machine",
    "state_machine_report",
    "workflow_rank",
]


class TransitionError(ValueError):
    """Base class for every refused transition."""


class IllegalTransition(TransitionError):
    """No edge exists from the current state for this operation (P-07)."""


class GuardRejected(TransitionError):
    """The edge exists but its precondition evaluated false."""


class ActorNotPermitted(TransitionError):
    """The actor's role may not perform this operation under this template (G-2)."""


@dataclass(frozen=True)
class TransitionFacts:
    """Everything a guard is allowed to look at.

    A closed record rather than a ``**kwargs`` bag, because a guard that reads
    arbitrary context is a guard nobody can test. When a new guard needs a new
    fact, it appears here and every caller must pass it — which is the point.
    """

    template: TemplateSpec
    redundancy_n: int
    blind_reads_submitted: int = 0
    reveal_done: bool = False
    supplements_collected: bool = False
    disagreement_computed: bool = False
    #: D-09 trigger. Default False: the rule ships disabled (PRD 4.8, D-09).
    cosign_enabled: bool = False
    cosign_triggered: bool = False
    cosign_verdict: Optional[CosignVerdict] = None
    gate_passed: bool = False

    def evidence(self) -> dict[str, object]:
        """Primitive view of the facts, for the audit record and its hash."""
        return {
            "blind_reads_submitted": self.blind_reads_submitted,
            "cosign_enabled": self.cosign_enabled,
            "cosign_triggered": self.cosign_triggered,
            "cosign_verdict": None if self.cosign_verdict is None else self.cosign_verdict.value,
            "disagreement_computed": self.disagreement_computed,
            "gate_passed": self.gate_passed,
            "redundancy_n": self.redundancy_n,
            "reveal_done": self.reveal_done,
            "supplements_collected": self.supplements_collected,
            "template": self.template.template,
        }

    def evidence_hash(self) -> str:
        """Hash of :meth:`evidence`, so a reviewer can prove what was evaluated."""
        return content_hash(self.evidence())


# ─────────────────────────────────────────────────────────────────────────────
# Guards
# ─────────────────────────────────────────────────────────────────────────────

def _all_blind_reads_in(facts: TransitionFacts) -> bool:
    return facts.blind_reads_submitted >= facts.redundancy_n


def _always(_facts: TransitionFacts) -> bool:
    return True


def _cosign_fires(f: TransitionFacts) -> bool:
    """D-09: the sample hits the cosign trigger *and* the rule is switched on."""
    return f.template.cosign_available and f.cosign_enabled and f.cosign_triggered


GUARDS: Mapping[str, Callable[[TransitionFacts], bool]] = {
    "always": _always,
    "all_blind_reads_in": _all_blind_reads_in,
    "revealed": lambda f: f.reveal_done,
    # Disagreement is only computable with >= 2 revisions, so N=1 templates
    # (teaching) must not be blocked on it. This is the one place redundancy
    # count changes a guard rather than a threshold.
    "disagreement_ready": lambda f: f.disagreement_computed or f.redundancy_n < 2,
    "supplements_collected": lambda f: f.supplements_collected,
    "cosign_not_due": lambda f: not _cosign_fires(f),
    "cosign_due": _cosign_fires,
    "cosign_approved": lambda f: f.cosign_verdict is CosignVerdict.APPROVE,
    "cosign_rejected": lambda f: f.cosign_verdict is CosignVerdict.REJECT,
    "gate_passed": lambda f: f.gate_passed,
    "gate_failed": lambda f: not f.gate_passed,
}


@dataclass(frozen=True)
class Edge:
    """One declared transition."""

    src: SampleState
    dst: SampleState
    operation: str
    kind: EdgeKind
    guard: str
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# The table (PRD 4.8)
# ─────────────────────────────────────────────────────────────────────────────

WORKFLOW_EDGES: tuple[Edge, ...] = (
    Edge(
        SampleState.DRAFT,
        SampleState.PUBLISHED,
        "publish",
        EdgeKind.WORKFLOW,
        "always",
        "发布任务；Assignment 生成",
    ),
    Edge(
        SampleState.PUBLISHED,
        SampleState.IN_BLIND_READ,
        "begin_blind_read",
        EdgeKind.WORKFLOW,
        "always",
        "首位专家开始盲读",
    ),
    Edge(
        SampleState.IN_BLIND_READ,
        SampleState.AWAITING_REVEAL,
        "complete_reveal",
        EdgeKind.WORKFLOW,
        "all_blind_reads_in",
        "N 份盲读齐",
    ),
    Edge(
        SampleState.AWAITING_REVEAL,
        SampleState.IN_REVIEW,
        "begin_review",
        EdgeKind.WORKFLOW,
        "revealed",
        "揭示完成",
    ),
    Edge(
        SampleState.IN_REVIEW,
        SampleState.AWAITING_ADJUDICATION,
        "request_adjudication",
        EdgeKind.WORKFLOW,
        "supplements_collected",
        "分歧分析已出、补充修订收齐",
    ),
    Edge(
        SampleState.AWAITING_ADJUDICATION,
        SampleState.ADJUDICATED,
        "adjudicate",
        EdgeKind.WORKFLOW,
        "cosign_not_due",
        "未命中复核条件，首席签定稿",
    ),
    Edge(
        SampleState.AWAITING_ADJUDICATION,
        SampleState.AWAITING_COSIGN,
        "request_cosign",
        EdgeKind.WORKFLOW,
        "cosign_due",
        "命中复核条件（D-09，默认关闭）",
    ),
    Edge(
        SampleState.AWAITING_COSIGN,
        SampleState.ADJUDICATED,
        "cosign_approve",
        EdgeKind.WORKFLOW,
        "cosign_approved",
        "复核通过",
    ),
    Edge(
        SampleState.AWAITING_COSIGN,
        SampleState.AWAITING_ADJUDICATION,
        "cosign_reject",
        EdgeKind.WORKFLOW,
        "cosign_rejected",
        "复核否决，退回首席",
    ),
    Edge(
        SampleState.ADJUDICATED,
        SampleState.IN_DATASET,
        "admit_to_dataset",
        EdgeKind.WORKFLOW,
        "gate_passed",
        "门槛通过，入库",
    ),
    Edge(
        SampleState.ADJUDICATED,
        SampleState.NEEDS_IMPROVEMENT,
        "return_for_improvement",
        EdgeKind.WORKFLOW,
        "gate_failed",
        "门槛未通过",
    ),
    Edge(
        SampleState.NEEDS_IMPROVEMENT,
        SampleState.PUBLISHED,
        "reopen",
        EdgeKind.WORKFLOW,
        "always",
        "重开，重新进入盲读",
    ),
)

#: Administrative termination. PRD 4.8 gives REJECTED the trigger "数据或协议问题
#: → 终止" but names no source states; without these edges the state would be
#: unreachable and therefore decorative. The rule is deliberately blunt — any
#: state that is not itself terminal may be aborted — and it is a separate edge
#: class so a change to termination policy is reviewable on its own.
ABORT_EDGES: tuple[Edge, ...] = tuple(
    Edge(
        state,
        SampleState.REJECTED,
        "abort",
        EdgeKind.ABORT,
        "always",
        "数据或协议问题，行政终止",
    )
    for state in SampleState
    if not state.is_terminal
)


@dataclass(frozen=True)
class TransitionResult:
    """Outcome of an accepted transition."""

    state: SampleState
    audit: dict = field(default_factory=dict)


class Machine:
    """An immutable transition table with guard and role enforcement."""

    def __init__(self, name: str, states: tuple[SampleState, ...], edges: Iterable[Edge]) -> None:
        self.name = name
        self.states = tuple(states)
        self.edges = tuple(edges)
        table: dict[tuple[SampleState, str], Edge] = {}
        for edge in self.edges:
            key = (edge.src, edge.operation)
            if key in table:  # pragma: no cover - programming error
                raise TransitionError(
                    f"{name}: two edges for ({edge.src.value}, {edge.operation!r}); one "
                    "operation from one state must have exactly one destination, or "
                    "the machine is non-deterministic and replay cannot verify it"
                )
            if edge.guard not in GUARDS:  # pragma: no cover - programming error
                raise TransitionError(f"{name}: edge names unknown guard {edge.guard!r}")
            table[key] = edge
        self._table = table

    # ── introspection ────────────────────────────────────────────────

    def operations_from(self, state: SampleState) -> tuple[str, ...]:
        """Operations declared from ``state``, in table order."""
        return tuple(key[1] for key in self._table if key[0] is state)

    def legal_pairs(self) -> frozenset[tuple[SampleState, SampleState]]:
        """Every ``(src, dst)`` pair that some edge permits."""
        return frozenset((e.src, e.dst) for e in self.edges)

    def workflow_pairs(self) -> frozenset[tuple[SampleState, SampleState]]:
        return frozenset((e.src, e.dst) for e in self.edges if e.kind is EdgeKind.WORKFLOW)

    def abort_pairs(self) -> frozenset[tuple[SampleState, SampleState]]:
        return frozenset((e.src, e.dst) for e in self.edges if e.kind is EdgeKind.ABORT)

    def table_report(self) -> dict[str, int]:
        """Counts G-4 talks about. Computed, never asserted by hand."""
        n = len(self.states)
        ordered_pairs = n * (n - 1)  # self-loops are not transitions
        legal = len(self.legal_pairs())
        return {
            "states": n,
            "ordered_pairs": ordered_pairs,
            "workflow_edges": len(self.workflow_pairs()),
            "abort_edges": len(self.abort_pairs()),
            "legal_ordered_pairs": legal,
            "illegal_ordered_pairs": ordered_pairs - legal,
        }

    # ── the transition ───────────────────────────────────────────────

    def apply(
        self,
        state: SampleState,
        operation: str,
        facts: TransitionFacts,
        *,
        actor_role: Role,
        actor_id: str = "",
    ) -> TransitionResult:
        """Perform ``operation`` from ``state``, or raise.

        ``actor_role`` is checked here rather than in the caller because the MVP
        proved that a view-level guard is bypassable by anything that does not
        go through the view (G-2: the auto-driven demo executed arbitration with
        no UI at all).
        """
        edge = self._table.get((state, operation))
        if edge is None:
            allowed = self.operations_from(state)
            raise IllegalTransition(
                f"cannot {operation!r} from {state.value}: no such edge (P-07). "
                f"Allowed from here: {list(allowed) or 'nothing — terminal state'}"
            )

        cast = facts.template.roles_for(operation)
        if cast:
            if actor_role not in cast:
                raise ActorNotPermitted(
                    f"role {actor_role.value!r} may not {operation!r} under template "
                    f"{facts.template.template!r}; allowed: "
                    f"{sorted(r.value for r in cast)}"
                )
        elif actor_role is not Role.SYSTEM:
            raise ActorNotPermitted(
                f"{operation!r} is a platform step under template "
                f"{facts.template.template!r}; it cannot be triggered by "
                f"{actor_role.value!r} (no human may claim an automatic transition)"
            )

        guard = GUARDS[edge.guard]
        if not guard(facts):
            raise GuardRejected(
                f"{state.value} -/-> {edge.dst.value} via {operation!r}: guard "
                f"{edge.guard!r} evaluated false. Evidence: {facts.evidence()}"
            )

        audit = {
            "operation": operation,
            "kind": edge.kind.value,
            "from": state.value,
            "to": edge.dst.value,
            "guard": edge.guard,
            "guard_evidence_hash": facts.evidence_hash(),
            "template": facts.template.template,
            "actor_id": actor_id,
            "actor_role": actor_role.value,
            "machine": self.name,
        }
        return TransitionResult(state=edge.dst, audit=audit)


_SAMPLE_MACHINE = Machine(
    "sample",
    tuple(SampleState),
    WORKFLOW_EDGES + ABORT_EDGES,
)


def sample_machine() -> Machine:
    """The sample-level machine of PRD 4.8 (12 workflow edges + 9 abort edges)."""
    return _SAMPLE_MACHINE


#: Named machines, for a service that dispatches on a path segment.
MACHINES: Mapping[str, Machine] = {"sample": _SAMPLE_MACHINE}


# ─────────────────────────────────────────────────────────────────────────────
# Assignment-level machine (PRD 4.8, second half)
# ─────────────────────────────────────────────────────────────────────────────

#: ``(src, operation, dst, guard)``. The order is
#: ``BLIND_DRAFTED → SUBMITTED → REVEALED → LOCKED``, not the literal order in
#: which PRD 4.8 lists the five states — see ``enums.AssignmentState``. Submitting
#: before revealing is P-01 itself; the reverse would show a person the
#: pre-annotation while their "blind" read was still editable.
ASSIGNMENT_EDGES: tuple[tuple[AssignmentState, str, AssignmentState, str], ...] = (
    (AssignmentState.ASSIGNED, "save_draft", AssignmentState.BLIND_DRAFTED, "always"),
    (AssignmentState.BLIND_DRAFTED, "submit_blind", AssignmentState.SUBMITTED, "always"),
    (AssignmentState.SUBMITTED, "reveal", AssignmentState.REVEALED, "task_in_review"),
    (AssignmentState.REVEALED, "lock", AssignmentState.LOCKED, "sample_signed"),
)

_ASSIGNMENT_TABLE: Mapping[tuple[AssignmentState, str], tuple[AssignmentState, str]] = {
    (src, operation): (dst, guard) for src, operation, dst, guard in ASSIGNMENT_EDGES
}

#: Sample states from which the reveal may be granted ("REVEALED 只在任务进入
#: IN_REVIEW 时授予" — 4.8).
_REVEAL_PERMITTED_TASK_STATES: frozenset[SampleState] = frozenset(
    {
        SampleState.IN_REVIEW,
        SampleState.AWAITING_ADJUDICATION,
        SampleState.AWAITING_COSIGN,
        SampleState.ADJUDICATED,
        SampleState.NEEDS_IMPROVEMENT,
        SampleState.IN_DATASET,
    }
)

#: Sample states in which a revision is final and may be locked.
_SIGNED_TASK_STATES: frozenset[SampleState] = frozenset(
    {SampleState.ADJUDICATED, SampleState.NEEDS_IMPROVEMENT, SampleState.IN_DATASET}
)


def advance_assignment(
    state: AssignmentState,
    operation: str,
    *,
    task_state: SampleState,
) -> AssignmentState:
    """Advance one assignment, or raise.

    The two cross-machine guards are the reason this is not a plain table walk:
    an assignment may only be *revealed* once its task has entered review, and
    only *locked* once the sample is signed. Those are the two places where the
    assignment machine depends on the sample machine, and they are the two places
    a naive implementation gets wrong by advancing on the assignment's own clock.
    """
    key = (state, operation)
    if key not in _ASSIGNMENT_TABLE:
        allowed = [op for (src, op) in _ASSIGNMENT_TABLE if src is state]
        raise IllegalTransition(
            f"cannot {operation!r} an assignment in {state.value}: allowed from here "
            f"{allowed or 'nothing — locked assignments are immutable'}"
        )
    dst, guard = _ASSIGNMENT_TABLE[key]
    if guard == "task_in_review" and task_state not in _REVEAL_PERMITTED_TASK_STATES:
        raise GuardRejected(
            f"cannot reveal an assignment while its sample is {task_state.value}; "
            "reveal is granted only once all blind reads are in (PRD 4.8). Showing "
            "the pre-annotation earlier makes the 'blind' read non-blind."
        )
    if guard == "sample_signed" and task_state not in _SIGNED_TASK_STATES:
        raise GuardRejected(
            f"cannot lock an assignment while its sample is {task_state.value}; a "
            "revision may be locked only after the sample is signed"
        )
    return dst


def assignment_machine_report() -> dict[str, int]:
    """Counts for the assignment-level machine (5 states, 20 ordered pairs)."""
    n = len(AssignmentState)
    legal = len({(src, dst) for src, _op, dst, _g in ASSIGNMENT_EDGES})
    return {
        "states": n,
        "ordered_pairs": n * (n - 1),
        "legal_ordered_pairs": legal,
        "illegal_ordered_pairs": n * (n - 1) - legal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

_RANK: Mapping[SampleState, int] = {
    state: index
    for index, state in enumerate(
        (
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
    )
}


def workflow_rank(state: SampleState) -> int:
    """Depth of ``state`` in the workflow, for aggregation and for sorting."""
    return _RANK[state]


def aggregate_task_state(states: Iterable[SampleState]) -> SampleState:
    """The task's state is its least-advanced sample's state.

    ``Task.state`` has to be a single value (PRD 7.2) while the machine is
    sample-level (4.8), so a multi-sample task needs a projection. The minimum is
    the only defensible choice: a task is not "in review" until *every* sample is
    in review, and reporting the maximum would let one finished sample make a
    task look finished while three others had never been read. This mirrors how
    the export gate has to behave — a manifest is complete only when nothing in
    it is behind.
    """
    materialised = list(states)
    if not materialised:
        raise TransitionError("aggregate_task_state needs at least one sample state")
    return min(materialised, key=workflow_rank)


def state_machine_report() -> dict[str, int]:
    """Convenience wrapper around :meth:`Machine.table_report`."""
    return _SAMPLE_MACHINE.table_report()
