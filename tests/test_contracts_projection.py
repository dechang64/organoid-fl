# ── tests/test_contracts_projection.py ──
"""Projection tests — the regression suite for the eleven MVP defects.

The envelopes here are built with *every* record present (adjudication, model
ranking, peer revisions) and then projected at an early stage. That is
deliberate: a test that hides a record by deleting it from the fixture proves
nothing about the projection. These fixtures hold the answers while the blind
read is in progress, which is exactly the situation the eleven defects were in.
"""

from __future__ import annotations

import itertools

from analysis.contracts.enums import ProjectionStage, Role, SampleState
from analysis.contracts.projection import (
    ProjectionContext,
    project_envelope,
    visibility_for,
)
from tests import contracts_factory as F

#: Keys a sample view is allowed to carry. Anything outside this set is a
#: projection bug by construction, because the projection only ever adds keys
#: from this list.
ALLOWED_SAMPLE_KEYS = {
    "sample_id",
    "state",
    "stage",
    "assignment",
    "revisions",
    "progress",
    "preannotation",
    "stratum",
    "model_rank",
    "disagreement",
    "adjudication",
    "cosign",
}


def project(envelope, role, viewer_id, *, cosign_locked=False):
    ctx = ProjectionContext(viewer_role=role, viewer_id=viewer_id, cosign_locked=cosign_locked)
    payload = project_envelope(envelope, ctx)
    sample = payload["samples"][0]
    assert set(sample) <= ALLOWED_SAMPLE_KEYS, set(sample) - ALLOWED_SAMPLE_KEYS
    return payload, sample


# ─────────────────────────────────────────────────────────────────────────────
# P-01: the blind read is genuinely blind
# ─────────────────────────────────────────────────────────────────────────────


def test_blind_read_hides_the_model_ranking_even_though_it_exists():
    envelope = F.build_envelope(state=SampleState.IN_BLIND_READ)
    assert envelope.model_ranks, "fixture must carry a ranking for this test to mean anything"

    _, sample = project(envelope, Role.DOMAIN_EXPERT, "expert_a")
    assert "model_rank" not in sample
    assert "preannotation" not in sample
    assert "disagreement" not in sample
    assert "adjudication" not in sample


def test_blind_read_hides_peer_revisions():
    envelope = F.build_envelope(state=SampleState.IN_BLIND_READ)
    _, sample = project(envelope, Role.DOMAIN_EXPERT, "expert_a")
    assert [r["author_id"] for r in sample["revisions"]] == ["expert_a"]


def test_blind_read_hides_submission_counts_from_everyone_who_can_annotate():
    """S-12: "2/3 submitted" tells an expert they are last. That is a leak."""
    envelope = F.build_envelope(state=SampleState.IN_BLIND_READ)
    for role in (Role.DOMAIN_EXPERT, Role.ANNOTATOR, Role.CHIEF_EXPERT, Role.COSIGNER):
        _, sample = project(envelope, role, "expert_a")
        assert "progress" not in sample, role


def test_blind_read_still_lets_the_owner_see_progress():
    """The project owner needs counts to chase deadlines and cannot annotate."""
    envelope = F.build_envelope(state=SampleState.IN_BLIND_READ)
    _, sample = project(envelope, Role.PROJECT_OWNER, "owner_1")
    assert sample["progress"] == {"submitted": 3, "required": 3}


# ─────────────────────────────────────────────────────────────────────────────
# Post-reveal: what changes, and what does not
# ─────────────────────────────────────────────────────────────────────────────


def test_reveal_shows_the_preannotation_flag_but_never_the_ranking():
    """D-03: reveal the pre-annotation; never the ordering, at any stage."""
    envelope = F.build_envelope(state=SampleState.IN_REVIEW)
    _, sample = project(envelope, Role.DOMAIN_EXPERT, "expert_a")
    assert sample["preannotation"] == {"available": True}
    assert "model_rank" not in sample


def test_domain_expert_never_sees_peers_even_after_reveal():
    """PRD 3.2 gives the domain expert an unconditional "否" for peer content."""
    envelope = F.build_envelope(state=SampleState.IN_REVIEW)
    _, sample = project(envelope, Role.DOMAIN_EXPERT, "expert_a")
    assert len(sample["revisions"]) == 1
    assert "disagreement" not in sample


def test_chief_sees_peers_and_ranking_only_after_disagreement_exists():
    """PRD 3.2: 首席 sees peer revisions 仅分歧分析后."""
    with_disagreement = F.build_envelope(state=SampleState.IN_REVIEW)
    _, sample = project(with_disagreement, Role.CHIEF_EXPERT, F.CHIEF)
    assert len(sample["revisions"]) == 3
    assert sample["model_rank"]["model_id"] == "organoid-seg"
    assert "disagreement" in sample

    without = F.build_envelope(state=SampleState.IN_REVIEW, with_disagreement=False)
    _, sample = project(without, Role.CHIEF_EXPERT, F.CHIEF)
    # The chief authors nothing, so with peer visibility withheld there is
    # nothing left to show — which is the correct outcome, not an empty bug.
    assert sample["revisions"] == []
    assert "model_rank" not in sample


def test_chief_also_lacks_model_rank_during_the_blind_read():
    envelope = F.build_envelope(state=SampleState.IN_BLIND_READ)
    _, sample = project(envelope, Role.CHIEF_EXPERT, F.CHIEF)
    assert "model_rank" not in sample


# ─────────────────────────────────────────────────────────────────────────────
# P-09: the cosigner's two phases
# ─────────────────────────────────────────────────────────────────────────────


def test_cosigner_before_lock_sees_no_adjudication_at_all():
    envelope = F.build_envelope(state=SampleState.AWAITING_COSIGN)
    _, sample = project(envelope, Role.COSIGNER, F.COSIGNER, cosign_locked=False)

    assert "adjudication" not in sample
    assert "model_rank" not in sample
    assert "preannotation" not in sample
    assert "progress" not in sample
    # They must still be able to see their own commitment.
    assert sample["cosign"]["blind_choice"] == "R-002"


def test_cosigner_after_lock_sees_the_selection_but_not_the_reasoning():
    envelope = F.build_envelope(state=SampleState.AWAITING_COSIGN)
    _, sample = project(envelope, Role.COSIGNER, F.COSIGNER, cosign_locked=True)

    adjudication = sample["adjudication"]
    assert adjudication["selected_revision_id"] == "R-002"
    assert "reason_text" not in adjudication
    assert "signature_meaning" not in adjudication
    # And still no ranking: P-09's independence requirement does not expire.
    assert "model_rank" not in sample


def test_cosigner_cannot_see_another_persons_cosign_record():
    """Another cosign would carry the same answer as the adjudication."""
    envelope = F.build_envelope(state=SampleState.AWAITING_COSIGN)
    _, sample = project(envelope, Role.COSIGNER, "someone_else", cosign_locked=True)
    assert "cosign" not in sample


# ─────────────────────────────────────────────────────────────────────────────
# Other roles
# ─────────────────────────────────────────────────────────────────────────────


def test_project_owner_sees_aggregates_not_answers():
    envelope = F.build_envelope(state=SampleState.IN_REVIEW)
    _, sample = project(envelope, Role.PROJECT_OWNER, "owner_1")
    assert len(sample["revisions"]) == 0
    assert "model_rank" not in sample
    assert "adjudication" in sample
    assert "reason_text" not in sample["adjudication"]


def test_observer_is_read_only_and_summary_only():
    envelope = F.build_envelope(state=SampleState.IN_DATASET)
    _, sample = project(envelope, Role.OBSERVER, "funder_1")
    assert "revisions" in sample and sample["revisions"] == []
    assert "model_rank" not in sample
    assert "adjudication" in sample


def test_stratum_is_withheld_from_annotators():
    """A difficulty label read before annotating is a hint about the answer."""
    envelope = F.build_envelope(state=SampleState.IN_REVIEW)
    for role in (Role.DOMAIN_EXPERT, Role.ANNOTATOR, Role.COSIGNER):
        _, sample = project(envelope, role, "expert_a")
        assert "stratum" not in sample, role
    _, sample = project(envelope, Role.CHIEF_EXPERT, F.CHIEF)
    assert sample["stratum"]["stratum"] == "borderline"


# ─────────────────────────────────────────────────────────────────────────────
# Table-driven: the whole policy, not one path
# ─────────────────────────────────────────────────────────────────────────────


ATTENTION = [r for r in Role if r is not Role.SYSTEM]


def test_no_role_ever_receives_the_ranking_unless_the_policy_says_so():
    """Exhaustive over role × state × lock: the absence is asserted, not assumed."""
    envelope = F.build_envelope(state=SampleState.IN_REVIEW)
    for role, state, locked in itertools.product(
        ATTENTION, list(SampleState), (False, True)
    ):
        env = F.build_envelope(state=state)
        payload = project_envelope(
            env, ProjectionContext(viewer_role=role, viewer_id="x", cosign_locked=locked)
        )
        sample = payload["samples"][0]
        expected = visibility_for(
            role,
            ProjectionStage.of(state),
            cosign_locked=locked,
            disagreement_present=bool(env.disagreements),
        )
        assert ("model_rank" in sample) is expected.model_rank, (role, state)
        assert ("progress" in sample) is expected.progress, (role, state)
        assert ("stratum" in sample) is expected.stratum, (role, state)
    assert envelope.model_ranks  # fixture sanity


def test_needs_improvement_returns_to_blind_visibility():
    """4.8: the sample re-enters the blind loop, so the reveal must be undone."""
    assert ProjectionStage.of(SampleState.NEEDS_IMPROVEMENT) is ProjectionStage.BLIND
    envelope = F.build_envelope(state=SampleState.NEEDS_IMPROVEMENT)
    _, sample = project(envelope, Role.DOMAIN_EXPERT, "expert_a")
    assert "preannotation" not in sample
    assert "model_rank" not in sample


def test_projection_never_echoes_internal_bookkeeping():
    envelope = F.build_envelope(state=SampleState.IN_REVIEW)
    payload, _sample = project(envelope, Role.CHIEF_EXPERT, F.CHIEF)
    flat = repr(payload)
    for forbidden in ("content_hash", "format_version", "node_id", "trigger_rule"):
        assert forbidden not in flat, forbidden
    assert set(payload) == {"stage", "viewer", "task", "samples"}
