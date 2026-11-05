# ── tests/test_contracts_entities.py ──
"""Contract-entity tests: shape, strictness, and the invariants that carry P-03/P-06/P-09.

The tests are grouped by the promise each one enforces, because a failing entity
test should tell you *which requirement broke*, not merely which field raised.
"""

from __future__ import annotations

import dataclasses

import pytest

from analysis.contracts.entities import (
    Adjudication,
    AnnotationProtocol,
    Assignment,
    ContractValidationError,
    Cosign,
    ModelRank,
    Revision,
    StratumLabel,
    Task,
    validate_cosign_independence,
    validate_model_rank,
)
from analysis.contracts.enums import AssignmentState, CosignVerdict, Role, SampleState, Stratum
from analysis.contracts.hashing import content_hash
from tests import contracts_factory as F


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip
# ─────────────────────────────────────────────────────────────────────────────


def _instances():
    return [
        F.protocol(),
        F.task(),
        F.assignments()[0],
        F.revisions()[0],
        F.disagreement(),
        F.model_rank(),
        F.adjudication(),
        F.cosign(),
        F.stratum(),
    ]


@pytest.mark.parametrize("entity", _instances(), ids=lambda e: e.ENTITY)
def test_every_entity_round_trips_through_its_own_dict(entity):
    """``from_dict(to_dict(x)) == x`` for all nine, including enum and tuple fields."""
    rebuilt = type(entity).from_dict(entity.to_dict())
    assert rebuilt == entity
    assert rebuilt.content_hash() == entity.content_hash()


def test_round_trip_preserves_tuple_shape():
    """A tuple field must come back as a tuple, not a list.

    ``Task.image_scope`` and ``ModelRank.ranking`` are tuples so they cannot be
    appended to in place. A reader that returns a list would let a caller mutate
    a "frozen" record's contents.
    """
    task = Task.from_dict(F.task().to_dict())
    assert isinstance(task.image_scope, tuple)
    rank = ModelRank.from_dict(F.model_rank().to_dict())
    assert isinstance(rank.ranking, tuple)


def test_entities_are_frozen():
    task = F.task()
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.state = SampleState.IN_DATASET  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Strictness
# ─────────────────────────────────────────────────────────────────────────────


def test_unknown_key_is_rejected():
    payload = F.task().to_dict()
    payload["redundancy"] = 3  # the plausible typo: missing _n
    with pytest.raises(ContractValidationError, match="no field"):
        Task.from_dict(payload)


def test_missing_key_is_rejected():
    payload = F.task().to_dict()
    del payload["redundancy_n"]
    with pytest.raises(ContractValidationError, match="missing required field"):
        Task.from_dict(payload)


def test_enum_accepts_string_and_rejects_unknown_value():
    task = Task.from_dict({**F.task().to_dict(), "state": "draft"})
    assert task.state is SampleState.DRAFT
    with pytest.raises(ContractValidationError, match="not a valid SampleState"):
        Task.from_dict({**F.task().to_dict(), "state": "drafted"})


# ─────────────────────────────────────────────────────────────────────────────
# AnnotationProtocol — hash integrity (feeds the audit chain)
# ─────────────────────────────────────────────────────────────────────────────


def test_protocol_hash_is_computed_when_omitted():
    assert F.protocol().hash


def test_protocol_rejects_rules_that_do_not_match_its_hash():
    """Editing a rule while keeping the hash must fail, not silently diverge.

    This is the check that stops a taxonomy change from being applied to already
    annotated samples under an old protocol name.
    """
    payload = F.protocol().to_dict()
    payload["min_visible_boundary_um"] = 5.0
    with pytest.raises(ContractValidationError, match="does not match the rules"):
        AnnotationProtocol.from_dict(payload)


def test_protocol_hash_covers_rules_but_not_version():
    """A version bump alone keeps the hash; a rule change alters it.

    Both halves matter: the first makes "same rules, new number" visible, the
    second makes "new rules, old number" impossible.
    """
    base = F.protocol()
    same_rules = F.protocol(version="1.1")
    changed_rule = F.protocol(overlap_split_rule="watershed_on_intensity")

    assert same_rules.hash == base.hash
    assert changed_rule.hash != base.hash


def test_protocol_rejects_unregistered_taxonomy():
    from analysis.contracts.protocol import ProtocolProblem, make_protocol

    with pytest.raises(ProtocolProblem, match="not registered"):
        make_protocol(
            id="proto-x",
            version="1.0",
            taxonomy_name="does_not_exist",
            touching_object_rule="split",
            min_visible_boundary_um=2.0,
            overlap_split_rule="watershed",
            effective_from="2026-10-01T00:00:00Z",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [0, 6, 9])
def test_task_rejects_redundancy_outside_1_to_5(n):
    with pytest.raises(ContractValidationError, match="redundancy_n"):
        F.task(redundancy_n=n)


def test_task_rejects_unknown_template():
    """The template must be registered: an unknown one cannot be projected or gated."""
    with pytest.raises(ContractValidationError, match="not a registered template"):
        F.task(template="special_snowflake")


def test_task_rejects_empty_and_duplicate_scope():
    with pytest.raises(ContractValidationError, match="at least one sample"):
        F.task(image_scope=())
    with pytest.raises(ContractValidationError, match="duplicates"):
        F.task(image_scope=("S-001", "S-001"))


def test_task_normalises_template_member_to_string():
    from analysis.contracts.enums import Template

    assert F.task(template=Template.RESEARCH).template == "research"


# ─────────────────────────────────────────────────────────────────────────────
# Assignment / Revision
# ─────────────────────────────────────────────────────────────────────────────


def test_assignment_rejects_a_signing_role():
    with pytest.raises(ContractValidationError, match="must be DOMAIN_EXPERT or ANNOTATOR"):
        Assignment(
            id="A-9",
            task_id=F.TASK_ID,
            sample_id=F.SAMPLE,
            assignee_id="chief_1",
            role=Role.CHIEF_EXPERT,
            state=AssignmentState.ASSIGNED,
        )


def test_assignment_rejects_submission_before_start():
    with pytest.raises(ContractValidationError, match="precedes started_at"):
        Assignment(
            id="A-9",
            task_id=F.TASK_ID,
            sample_id=F.SAMPLE,
            assignee_id="expert_a",
            role=Role.DOMAIN_EXPERT,
            state=AssignmentState.SUBMITTED,
            started_at=F.T2,
            submitted_at=F.T1,
        )


def test_revision_requires_a_sha256_payload_hash():
    payload = F.revisions()[0].to_dict()
    payload["payload_hash"] = "deadbeef"
    with pytest.raises(ContractValidationError, match="SHA-256"):
        Revision.from_dict(payload)


def test_revision_rejects_self_derivation():
    base = F.revisions()[0]
    with pytest.raises(ContractValidationError, match="cannot name itself"):
        dataclasses.replace(base, derived_from=base.id)


def test_revision_number_is_one_based():
    payload = F.revisions()[0].to_dict()
    payload["revision_no"] = 0
    with pytest.raises(ContractValidationError, match="revision_no"):
        Revision.from_dict(payload)


def test_revision_payload_hash_is_stable_under_key_reordering():
    """P-06 in miniature: the same content hashes the same regardless of key order."""
    a = {"sample_id": "S-001", "instances": 4, "author": "expert_a"}
    b = {"author": "expert_a", "instances": 4, "sample_id": "S-001"}
    assert content_hash(a) == content_hash(b)


# ─────────────────────────────────────────────────────────────────────────────
# Adjudication — P-03, P-05
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("field", ["reason_code", "reason_text", "signature_meaning", "evidence_ref"])
def test_adjudication_cannot_be_signed_without_its_evidence(field):
    payload = F.adjudication().to_dict()
    payload[field] = ""
    with pytest.raises(ContractValidationError, match=field):
        Adjudication.from_dict(payload)


def test_adjudication_reads_the_model_adoption_flag_from_the_signature():
    assert F.adjudication().adopts_model_rank is True
    assert dataclasses.replace(F.adjudication(), signature_meaning="未采纳模型排序，按边界重选。").adopts_model_rank is False
    assert dataclasses.replace(F.adjudication(), signature_meaning="按协议 4.2 执行。").adopts_model_rank is None


# ─────────────────────────────────────────────────────────────────────────────
# Cosign — P-09 ordering, enforced at construction
# ─────────────────────────────────────────────────────────────────────────────


def test_cosign_cannot_be_revealed_before_a_blind_choice_is_locked():
    """The defect L1-1 in record form: a reveal without a lock is not a second opinion."""
    payload = F.cosign().to_dict()
    payload["blind_choice"] = None
    payload["blind_choice_locked_at"] = None
    with pytest.raises(ContractValidationError, match="P-09"):
        Cosign.from_dict(payload)


def test_cosign_choice_and_lock_time_travel_together():
    payload = F.cosign().to_dict()
    payload["blind_choice_locked_at"] = None
    with pytest.raises(ContractValidationError, match="must be set together"):
        Cosign.from_dict(payload)


def test_cosign_rejection_requires_dissent_text():
    payload = F.cosign(verdict=CosignVerdict.REJECT).to_dict()
    payload["dissent_text"] = None
    with pytest.raises(ContractValidationError, match="dissent_text"):
        Cosign.from_dict(payload)


def test_cosign_cannot_be_decided_before_it_is_revealed():
    payload = F.cosign().to_dict()
    payload["revealed_at"] = F.T3
    payload["decided_at"] = F.T2
    with pytest.raises(ContractValidationError, match="precedes revealed_at"):
        Cosign.from_dict(payload)


def test_cosign_signer_must_not_be_the_adjudicating_chief():
    """PRD 3.1: 复核专家不得与本案首席为同一人. Self-cosign is a ritual, not a check."""
    adjudication = F.adjudication(selector="chief_1")
    with pytest.raises(ContractValidationError, match="同一人"):
        validate_cosign_independence(adjudication, F.cosign(cosigner="chief_1"))


def test_cosign_must_point_at_its_own_adjudication_and_sample():
    adjudication = F.adjudication()
    wrong_adjudication = dataclasses.replace(F.cosign(), adjudication_id="J-999")
    with pytest.raises(ContractValidationError, match="does not name"):
        validate_cosign_independence(adjudication, wrong_adjudication)

    wrong_sample = dataclasses.replace(F.cosign(), sample_id="S-002")
    with pytest.raises(ContractValidationError, match=r"Cosign\.sample_id"):
        validate_cosign_independence(adjudication, wrong_sample)


def test_consistent_cosign_pair_validates():
    validate_cosign_independence(F.adjudication(), F.cosign())


# ─────────────────────────────────────────────────────────────────────────────
# ModelRank — advisory, but internally exact
# ─────────────────────────────────────────────────────────────────────────────


def test_model_rank_must_permute_the_samples_revisions():
    revisions = F.revisions()
    with pytest.raises(ContractValidationError, match="not a permutation"):
        validate_model_rank(F.model_rank(revision_ids=("R-001", "R-002")), revisions)


def test_model_rank_rejects_a_foreign_sample():
    """Cross-sample carry-over (L2-1) expressed as a validation failure."""
    revisions = F.revisions()
    foreign = dataclasses.replace(F.model_rank(), sample_id="S-002")
    with pytest.raises(ContractValidationError, match="belong to"):
        validate_model_rank(foreign, revisions)


def test_model_rank_rejects_scores_for_unknown_revisions():
    rank = dataclasses.replace(F.model_rank(), scores={"R-001": 0.9, "R-404": 0.1})
    with pytest.raises(ContractValidationError, match="unknown revision"):
        validate_model_rank(rank, F.revisions())


def test_model_rank_rejects_duplicates_and_bad_scores():
    with pytest.raises(ContractValidationError, match="repeats"):
        F.model_rank(revision_ids=("R-001", "R-001", "R-003"))
    with pytest.raises(ContractValidationError, match=r"\[0, 1\]"):
        dataclasses.replace(F.model_rank(), confidence=1.4)


# ─────────────────────────────────────────────────────────────────────────────
# StratumLabel — no default stratum (D-08)
# ─────────────────────────────────────────────────────────────────────────────


def test_stratum_label_requires_an_explicit_stratum():
    """Constructing without a stratum must be impossible, not defaulted to easy."""
    with pytest.raises(TypeError):
        StratumLabel(  # type: ignore[call-arg]
            sample_id="S-001",
            signal_snapshot={"x": 1},
            labeled_at=F.T0,
            rule_version="v1",
        )


def test_stratum_label_requires_its_signals():
    with pytest.raises(ContractValidationError, match="signal_snapshot is empty"):
        StratumLabel(
            sample_id="S-001",
            stratum=Stratum.CLUSTERED,
            signal_snapshot={},
            labeled_at=F.T0,
            rule_version="v1",
        )


def test_stratum_rejects_unknown_value():
    payload = F.stratum().to_dict()
    payload["stratum"] = "very_hard"
    with pytest.raises(ContractValidationError, match="not a valid Stratum"):
        StratumLabel.from_dict(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Timestamps
# ─────────────────────────────────────────────────────────────────────────────


def test_naive_timestamp_is_rejected():
    """No offset means no comparable ordering, and P-09 is an ordering claim."""
    payload = F.revisions()[0].to_dict()
    payload["created_at"] = "2026-11-01T09:30:00"
    with pytest.raises(ContractValidationError, match="no UTC offset"):
        Revision.from_dict(payload)


def test_non_iso_timestamp_is_rejected():
    payload = F.revisions()[0].to_dict()
    payload["created_at"] = "01/11/2026 09:30"
    with pytest.raises(ContractValidationError, match="not an ISO-8601"):
        Revision.from_dict(payload)
