# ── tests/test_contracts_state_machine.py ──
"""State-machine tests: the exhaustive pair check (G-4/P-07) plus every guard.

The first three tests are the ones G-4 asks for. They spell the expected edges
out *literally* rather than deriving them from the module, because a test that
reads the table it is testing agrees with any table.
"""

from __future__ import annotations

import itertools

import pytest

from analysis.contracts.enums import AssignmentState, CosignVerdict, Role, SampleState
from analysis.contracts.hashing import content_hash
from analysis.contracts.state_machine import (
    ActorNotPermitted,
    GuardRejected,
    IllegalTransition,
    TransitionFacts,
    advance_assignment,
    aggregate_task_state,
    assignment_machine_report,
    sample_machine,
    state_machine_report,
    workflow_rank,
)
from analysis.contracts.templates import RESEARCH_TEMPLATE, TEACHING_TEMPLATE

M = sample_machine()

#: PRD 4.8, transcribed by hand. Twelve edges, in workflow order.
EXPECTED_WORKFLOW = frozenset(
    {
        (SampleState.DRAFT, SampleState.PUBLISHED),
        (SampleState.PUBLISHED, SampleState.IN_BLIND_READ),
        (SampleState.IN_BLIND_READ, SampleState.AWAITING_REVEAL),
        (SampleState.AWAITING_REVEAL, SampleState.IN_REVIEW),
        (SampleState.IN_REVIEW, SampleState.AWAITING_ADJUDICATION),
        (SampleState.AWAITING_ADJUDICATION, SampleState.ADJUDICATED),
        (SampleState.AWAITING_ADJUDICATION, SampleState.AWAITING_COSIGN),
        (SampleState.AWAITING_COSIGN, SampleState.ADJUDICATED),
        (SampleState.AWAITING_COSIGN, SampleState.AWAITING_ADJUDICATION),
        (SampleState.ADJUDICATED, SampleState.IN_DATASET),
        (SampleState.ADJUDICATED, SampleState.NEEDS_IMPROVEMENT),
        (SampleState.NEEDS_IMPROVEMENT, SampleState.PUBLISHED),
    }
)

#: Administrative termination: every non-terminal state may be aborted.
EXPECTED_ABORT = frozenset(
    (state, SampleState.REJECTED) for state in SampleState if not state.is_terminal
)


def facts(**overrides) -> TransitionFacts:
    """Facts with a healthy N=3 research task; overrides per test."""
    base = {"template": RESEARCH_TEMPLATE, "redundancy_n": 3}
    base.update(overrides)
    return TransitionFacts(**base)


def optimistic(*, cosign_enabled: bool = False) -> TransitionFacts:
    """Every precondition satisfied at once, used only for reachability walks."""
    return facts(
        blind_reads_submitted=3,
        reveal_done=True,
        supplements_collected=True,
        disagreement_computed=True,
        gate_passed=True,
        cosign_enabled=cosign_enabled,
        cosign_triggered=True,
        cosign_verdict=CosignVerdict.APPROVE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# G-4: the table, exhaustively
# ─────────────────────────────────────────────────────────────────────────────


def test_workflow_edges_match_prd_4_8_exactly():
    assert M.workflow_pairs() == EXPECTED_WORKFLOW


def test_abort_edges_cover_every_non_terminal_state():
    assert M.abort_pairs() == EXPECTED_ABORT


def test_every_ordered_pair_is_classified():
    """All n·(n-1) pairs, each either workflow, abort, or illegal — no gaps."""
    states = list(SampleState)
    workflow, abort = M.workflow_pairs(), M.abort_pairs()
    assert not (workflow & abort), "an edge cannot be both workflow and abort"

    for src, dst in itertools.permutations(states, 2):
        legal = (src, dst) in workflow or (src, dst) in abort
        expected = (src, dst) in EXPECTED_WORKFLOW or (src, dst) in EXPECTED_ABORT
        assert legal is expected, f"{src.value} -> {dst.value} classified as {legal}"

    illegal = [
        (a, b)
        for a, b in itertools.permutations(states, 2)
        if (a, b) not in workflow and (a, b) not in abort
    ]
    assert len(illegal) == 89


def test_self_transitions_are_not_edges():
    """A state machine with self-loops cannot distinguish 'unchanged' from 'advanced'."""
    for state in SampleState:
        assert (state, state) not in M.legal_pairs()


def test_counts_are_published_and_match_the_table():
    report = state_machine_report()
    assert report == {
        "states": 11,
        "ordered_pairs": 110,
        "workflow_edges": 12,
        "abort_edges": 9,
        "legal_ordered_pairs": 21,
        "illegal_ordered_pairs": 89,
    }


def test_prd_arithmetic_does_not_fit_the_table_it_describes():
    """Documents the 90/12/78 discrepancy so it cannot silently persist.

    PRD 4.8 says "十状态" and G-4 says "90 有序对（12 合法 + 78 非法）", but the
    table in 4.8 uses ``IN_DATASET`` as ``ADJUDICATED``'s exit without listing it
    as a state. Ten states give 90 pairs and only 11 legal edges (because
    ``ADJUDICATED -> IN_DATASET`` would be missing); twelve legal edges require
    eleven states, i.e. 110 pairs. Both PRD numbers cannot hold at once.

    The contract takes the 11-state reading (``enums.SampleState`` explains why
    ``IN_DATASET`` cannot be folded away). This test fails the day someone adds a
    state without revisiting the arithmetic — which is the point.
    """
    listed_in_prd = [s for s in SampleState if s is not SampleState.IN_DATASET]
    assert len(listed_in_prd) == 10
    assert 10 * 9 == 90  # G-4's pair count, consistent with 10 states
    assert len(list(SampleState)) == 11
    assert 11 * 10 == 110
    assert len(EXPECTED_WORKFLOW) == 12  # only consistent with 11 states


# ─────────────────────────────────────────────────────────────────────────────
# Reachability
# ─────────────────────────────────────────────────────────────────────────────


def _probes(*, cosign_enabled: bool) -> list[TransitionFacts]:
    """Every relevant fact combination, so a reachability walk is about *structure*.

    A single fact set cannot answer "is this state reachable": ``gate_passed``
    has to be true to reach ``IN_DATASET`` and false to reach
    ``NEEDS_IMPROVEMENT``, and no one object is both. Swapping in a probe per
    edge would let the test pass for a state that no run can actually get to;
    enumerating the small cross-product keeps the walk honest.
    """
    out = []
    for gate_passed in (True, False):
        for cosign_triggered in (True, False):
            for verdict in (CosignVerdict.APPROVE, CosignVerdict.REJECT):
                out.append(
                    facts(
                        blind_reads_submitted=3,
                        reveal_done=True,
                        supplements_collected=True,
                        disagreement_computed=True,
                        gate_passed=gate_passed,
                        cosign_enabled=cosign_enabled,
                        cosign_triggered=cosign_triggered,
                        cosign_verdict=verdict,
                    )
                )
    return out


def _reachable(probes: list[TransitionFacts]) -> set[SampleState]:
    """Walk the graph, following an edge if any probe satisfies its guard."""
    from analysis.contracts.state_machine import GUARDS

    seen = {SampleState.DRAFT}
    frontier = [SampleState.DRAFT]
    while frontier:
        state = frontier.pop()
        for edge in M.edges:
            if edge.src is not state:
                continue
            if not any(GUARDS[edge.guard](probe) for probe in probes):
                continue
            if edge.dst not in seen:
                seen.add(edge.dst)
                frontier.append(edge.dst)
    return seen


def test_cosign_state_is_unreachable_with_the_default_configuration():
    """D-09: the state exists in M0 while the trigger rule ships disabled."""
    reachable = _reachable(_probes(cosign_enabled=False))
    assert SampleState.AWAITING_COSIGN not in reachable
    assert SampleState.IN_DATASET in reachable


def test_cosign_state_becomes_reachable_once_the_rule_is_enabled():
    reachable = _reachable(_probes(cosign_enabled=True))
    assert SampleState.AWAITING_COSIGN in reachable
    assert SampleState.IN_DATASET in reachable


def test_everything_but_cosign_is_reachable_with_defaults():
    reachable = _reachable(_probes(cosign_enabled=False))
    expected = set(SampleState) - {SampleState.AWAITING_COSIGN}
    assert reachable == expected


# ─────────────────────────────────────────────────────────────────────────────
# The happy path, step by step
# ─────────────────────────────────────────────────────────────────────────────


def test_full_path_to_dataset():
    state = SampleState.DRAFT
    state = M.apply(state, "publish", facts(), actor_role=Role.PROJECT_OWNER).state
    assert state is SampleState.PUBLISHED

    state = M.apply(state, "begin_blind_read", facts(), actor_role=Role.DOMAIN_EXPERT).state
    assert state is SampleState.IN_BLIND_READ

    state = M.apply(
        state, "complete_reveal", facts(blind_reads_submitted=3), actor_role=Role.SYSTEM
    ).state
    assert state is SampleState.AWAITING_REVEAL

    state = M.apply(state, "begin_review", facts(reveal_done=True), actor_role=Role.SYSTEM).state
    assert state is SampleState.IN_REVIEW

    state = M.apply(
        state, "request_adjudication", facts(supplements_collected=True), actor_role=Role.SYSTEM
    ).state
    assert state is SampleState.AWAITING_ADJUDICATION

    state = M.apply(state, "adjudicate", facts(), actor_role=Role.CHIEF_EXPERT).state
    assert state is SampleState.ADJUDICATED

    state = M.apply(
        state, "admit_to_dataset", facts(gate_passed=True), actor_role=Role.SYSTEM
    ).state
    assert state is SampleState.IN_DATASET


def test_gate_failure_returns_for_improvement_then_reopens():
    state = SampleState.ADJUDICATED
    state = M.apply(
        state, "return_for_improvement", facts(gate_passed=False), actor_role=Role.SYSTEM
    ).state
    assert state is SampleState.NEEDS_IMPROVEMENT

    state = M.apply(state, "reopen", facts(), actor_role=Role.PROJECT_OWNER).state
    assert state is SampleState.PUBLISHED


def test_cosign_path_requires_a_verdict_before_signature():
    due = facts(cosign_enabled=True, cosign_triggered=True)
    state = M.apply(
        SampleState.AWAITING_ADJUDICATION, "request_cosign", due, actor_role=Role.SYSTEM
    ).state
    assert state is SampleState.AWAITING_COSIGN

    # No verdict yet: neither branch may fire.
    with pytest.raises(GuardRejected):
        M.apply(state, "cosign_approve", due, actor_role=Role.COSIGNER)
    with pytest.raises(GuardRejected):
        M.apply(state, "cosign_reject", due, actor_role=Role.COSIGNER)

    approved = facts(
        cosign_enabled=True, cosign_triggered=True, cosign_verdict=CosignVerdict.APPROVE
    )
    assert M.apply(state, "cosign_approve", approved, actor_role=Role.COSIGNER).state is (
        SampleState.ADJUDICATED
    )

    rejected = facts(
        cosign_enabled=True, cosign_triggered=True, cosign_verdict=CosignVerdict.REJECT
    )
    assert M.apply(state, "cosign_reject", rejected, actor_role=Role.COSIGNER).state is (
        SampleState.AWAITING_ADJUDICATION
    )


def test_adjudication_is_blocked_while_a_cosign_is_due():
    due = facts(cosign_enabled=True, cosign_triggered=True)
    with pytest.raises(GuardRejected, match="cosign_not_due"):
        M.apply(SampleState.AWAITING_ADJUDICATION, "adjudicate", due, actor_role=Role.CHIEF_EXPERT)


# ─────────────────────────────────────────────────────────────────────────────
# Illegal, rejected, and not-permitted — kept distinguishable
# ─────────────────────────────────────────────────────────────────────────────


def test_cannot_jump_states():
    with pytest.raises(IllegalTransition, match="no such edge"):
        M.apply(
            SampleState.DRAFT, "adjudicate", optimistic(), actor_role=Role.CHIEF_EXPERT
        )


def test_illegal_transition_lists_what_is_allowed():
    with pytest.raises(IllegalTransition) as excinfo:
        M.apply(SampleState.PUBLISHED, "adjudicate", optimistic(), actor_role=Role.CHIEF_EXPERT)
    assert "begin_blind_read" in str(excinfo.value)


def test_guard_rejection_carries_the_evidence_it_evaluated():
    """§8.1: 复核 — 触发规则与其求值依据. The message must show the inputs."""
    with pytest.raises(GuardRejected) as excinfo:
        M.apply(
            SampleState.IN_BLIND_READ,
            "complete_reveal",
            facts(blind_reads_submitted=2),
            actor_role=Role.SYSTEM,
        )
    message = str(excinfo.value)
    assert "blind_reads_submitted" in message
    assert "'blind_reads_submitted': 2" in message


def test_terminal_states_have_no_outgoing_operation():
    for state in (SampleState.IN_DATASET, SampleState.REJECTED):
        assert M.operations_from(state) == ()


def test_any_non_terminal_state_can_be_aborted():
    for state in SampleState:
        if state.is_terminal:
            continue
        result = M.apply(state, "abort", facts(), actor_role=Role.NODE_ADMIN, actor_id="admin_1")
        assert result.state is SampleState.REJECTED
    with pytest.raises(IllegalTransition):
        M.apply(SampleState.IN_DATASET, "abort", facts(), actor_role=Role.NODE_ADMIN)


# ─────────────────────────────────────────────────────────────────────────────
# Roles (G-2) — the server-side check the MVP did not have
# ─────────────────────────────────────────────────────────────────────────────


def test_domain_expert_cannot_adjudicate():
    with pytest.raises(ActorNotPermitted, match="may not 'adjudicate'"):
        M.apply(
            SampleState.AWAITING_ADJUDICATION,
            "adjudicate",
            facts(),
            actor_role=Role.DOMAIN_EXPERT,
        )


def test_platform_steps_cannot_be_triggered_by_a_human():
    """Reveal is automatic; letting an operator fire it would break P-01's timing."""
    with pytest.raises(ActorNotPermitted, match="platform step"):
        M.apply(
            SampleState.AWAITING_REVEAL,
            "begin_review",
            facts(reveal_done=True),
            actor_role=Role.PROJECT_OWNER,
        )


def test_system_cannot_perform_a_human_step():
    with pytest.raises(ActorNotPermitted):
        M.apply(SampleState.AWAITING_ADJUDICATION, "adjudicate", facts(), actor_role=Role.SYSTEM)


def test_teaching_template_recruits_annotator_and_reviewer():
    teaching = facts(template=TEACHING_TEMPLATE, redundancy_n=1, blind_reads_submitted=1)
    state = M.apply(
        SampleState.PUBLISHED, "begin_blind_read", teaching, actor_role=Role.ANNOTATOR
    ).state
    assert state is SampleState.IN_BLIND_READ
    # The same operation under the research template casts a domain expert only.
    with pytest.raises(ActorNotPermitted):
        M.apply(SampleState.PUBLISHED, "begin_blind_read", facts(), actor_role=Role.ANNOTATOR)


def test_teaching_template_requires_no_disagreement_for_n_equals_one():
    teaching = facts(
        template=TEACHING_TEMPLATE,
        redundancy_n=1,
        blind_reads_submitted=1,
        reveal_done=True,
        disagreement_computed=False,
    )
    state = M.apply(
        SampleState.AWAITING_REVEAL, "begin_review", teaching, actor_role=Role.SYSTEM
    ).state
    assert state is SampleState.IN_REVIEW


# ─────────────────────────────────────────────────────────────────────────────
# Audit record
# ─────────────────────────────────────────────────────────────────────────────


def test_audit_record_is_canonical_and_names_the_guard_evidence():
    f = facts(blind_reads_submitted=3)
    result = M.apply(
        SampleState.IN_BLIND_READ, "complete_reveal", f, actor_role=Role.SYSTEM, actor_id="node"
    )
    audit = result.audit

    assert audit["guard"] == "all_blind_reads_in"
    assert audit["guard_evidence_hash"] == f.evidence_hash()
    assert audit["from"] == "in_blind_read" and audit["to"] == "awaiting_reveal"
    assert audit["actor_role"] == "system"
    assert audit["kind"] == "workflow"
    # A block whose details cannot be canonicalised cannot be chained.
    assert len(content_hash(audit)) == 64


def test_evidence_hash_changes_when_the_basis_changes():
    a = facts(blind_reads_submitted=2)
    b = facts(blind_reads_submitted=3)
    assert a.evidence_hash() != b.evidence_hash()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def test_task_state_is_the_least_advanced_sample():
    assert (
        aggregate_task_state([SampleState.IN_DATASET, SampleState.IN_REVIEW, SampleState.DRAFT])
        is SampleState.DRAFT
    )
    assert aggregate_task_state([SampleState.IN_DATASET]) is SampleState.IN_DATASET


def test_workflow_rank_is_strictly_increasing_along_the_happy_path():
    path = [
        SampleState.DRAFT,
        SampleState.PUBLISHED,
        SampleState.IN_BLIND_READ,
        SampleState.AWAITING_REVEAL,
        SampleState.IN_REVIEW,
        SampleState.AWAITING_ADJUDICATION,
        SampleState.ADJUDICATED,
        SampleState.IN_DATASET,
    ]
    ranks = [workflow_rank(s) for s in path]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


# ─────────────────────────────────────────────────────────────────────────────
# Assignment-level machine
# ─────────────────────────────────────────────────────────────────────────────


def test_assignment_machine_counts():
    assert assignment_machine_report() == {
        "states": 5,
        "ordered_pairs": 20,
        "legal_ordered_pairs": 4,
        "illegal_ordered_pairs": 16,
    }


def test_assignment_submits_before_it_reveals():
    """P-01 at the assignment level: a reveal first would make the blind read non-blind."""
    state = AssignmentState.ASSIGNED
    state = advance_assignment(state, "save_draft", task_state=SampleState.IN_BLIND_READ)
    assert state is AssignmentState.BLIND_DRAFTED
    state = advance_assignment(state, "submit_blind", task_state=SampleState.IN_BLIND_READ)
    assert state is AssignmentState.SUBMITTED

    with pytest.raises(GuardRejected, match="blind reads are in"):
        advance_assignment(state, "reveal", task_state=SampleState.IN_BLIND_READ)

    state = advance_assignment(state, "reveal", task_state=SampleState.IN_REVIEW)
    assert state is AssignmentState.REVEALED


def test_assignment_locks_only_after_the_sample_is_signed():
    with pytest.raises(GuardRejected, match="signed"):
        advance_assignment(
            AssignmentState.REVEALED, "lock", task_state=SampleState.AWAITING_ADJUDICATION
        )
    assert (
        advance_assignment(AssignmentState.REVEALED, "lock", task_state=SampleState.ADJUDICATED)
        is AssignmentState.LOCKED
    )


def test_locked_assignment_is_immutable():
    with pytest.raises(IllegalTransition, match="nothing"):
        advance_assignment(
            AssignmentState.LOCKED, "save_draft", task_state=SampleState.IN_DATASET
        )
