# ── tests/test_contracts_exchange.py ──
"""Exchange-format tests: integrity, strictness, and the P-11 export gate."""

from __future__ import annotations

import dataclasses
import json

import pytest

from analysis.contracts.entities import ContractValidationError, Revision
from analysis.contracts.enums import Role, SampleState
from analysis.contracts.exchange import (
    FORMAT_VERSION,
    ExchangeEnvelope,
    ExportRefused,
    check_training_export,
    load_envelope,
    summarise,
)
from analysis.contracts.hashing import content_hash
from tests import contracts_factory as F


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip and canonical bytes
# ─────────────────────────────────────────────────────────────────────────────


def test_round_trip_preserves_the_envelope():
    envelope = F.build_envelope()
    assert load_envelope(envelope.dump()) == envelope


def test_dump_is_canonical_and_reproducible():
    """Same records must produce the same bytes, in any process."""
    a = F.build_envelope().dump()
    b = F.build_envelope().dump()
    assert a == b
    # Keys are sorted by the canonical encoder, so a diff of two envelopes is
    # a diff of the data rather than a diff of dict insertion order.
    assert list(json.loads(a)) == sorted(json.loads(a))


def test_content_hash_covers_every_collection():
    base = F.build_envelope()
    changed = base.evolve(stratums=(dataclasses.replace(base.stratums[0], rule_version="v2"),))
    assert changed.content_hash != base.content_hash


def test_envelope_rejects_a_stale_content_hash():
    payload = F.build_envelope().to_dict()
    payload["protocol"]["version"] = "9.9"
    with pytest.raises(ContractValidationError, match="does not match"):
        ExchangeEnvelope.from_dict(payload)


def test_envelope_rejects_tampered_bytes_after_the_fact():
    payload = json.loads(F.build_envelope().dump())
    payload["revisions"][0]["payload_ref"] = "blob://someone/else"
    with pytest.raises(ContractValidationError, match="does not match"):
        load_envelope(json.dumps(payload))


# ─────────────────────────────────────────────────────────────────────────────
# Strictness
# ─────────────────────────────────────────────────────────────────────────────


def test_unknown_field_is_rejected():
    payload = F.build_envelope().to_dict()
    payload["extra_notes"] = "hello"
    with pytest.raises(ContractValidationError, match="unknown field"):
        ExchangeEnvelope.from_dict(payload)


def test_missing_collection_is_rejected():
    payload = F.build_envelope().to_dict()
    del payload["cosigns"]
    with pytest.raises(ContractValidationError, match="missing field"):
        ExchangeEnvelope.from_dict(payload)


def test_unknown_format_version_is_rejected():
    payload = F.build_envelope().to_dict()
    payload["format_version"] = "organoid-exchange-v2"
    with pytest.raises(ContractValidationError, match="unknown versions"):
        ExchangeEnvelope.from_dict(payload)


def test_envelope_template_must_agree_with_the_task():
    payload = F.build_envelope().to_dict()
    payload["template"] = "teaching"
    with pytest.raises(ContractValidationError, match="disagrees with task template"):
        ExchangeEnvelope.from_dict(payload)


def test_samples_must_equal_the_task_scope():
    with pytest.raises(ContractValidationError, match="image_scope"):
        F.build_envelope().evolve(samples=(F.SAMPLE, "S-002"))


def test_record_from_another_sample_is_rejected():
    """L2-1 as a validation failure: records cannot cross sample boundaries."""
    envelope = F.build_envelope()
    stray = dataclasses.replace(envelope.revisions[0], sample_id="S-002")
    with pytest.raises(ContractValidationError, match="outside this envelope"):
        envelope.evolve(revisions=(stray, *envelope.revisions[1:]))


def test_two_strata_for_one_sample_are_rejected():
    envelope = F.build_envelope()
    with pytest.raises(ContractValidationError, match="more than one StratumLabel"):
        envelope.evolve(stratums=(envelope.stratums[0], envelope.stratums[0]))


def test_duplicate_ids_are_rejected():
    envelope = F.build_envelope()
    with pytest.raises(ContractValidationError, match="duplicate id"):
        envelope.evolve(revisions=(envelope.revisions[0], envelope.revisions[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-entity verification
# ─────────────────────────────────────────────────────────────────────────────


def test_verification_catches_an_unknown_selected_revision():
    envelope = F.build_envelope()
    bad = dataclasses.replace(envelope.adjudications[0], selected_revision_id="R-999")
    patched = envelope.evolve(adjudications=(bad,))
    assert any("selects unknown revision" in p for p in patched.verify())


def test_verification_catches_a_self_cosign():
    envelope = F.build_envelope()
    bad = dataclasses.replace(envelope.cosigns[0], cosigner_id=F.CHIEF)
    patched = envelope.evolve(cosigns=(bad,))
    assert any("同一人" in p for p in patched.verify())


def test_verification_catches_a_revision_without_its_assignment():
    envelope = F.build_envelope()
    orphan = dataclasses.replace(envelope.revisions[0], assignment_id="A-404")
    patched = envelope.evolve(revisions=(orphan, *envelope.revisions[1:]))
    assert any("unknown assignment" in p for p in patched.verify())


def test_verification_catches_an_author_who_is_not_the_assignee():
    envelope = F.build_envelope()
    impostor = dataclasses.replace(envelope.revisions[0], author_id="expert_c")
    patched = envelope.evolve(revisions=(impostor, *envelope.revisions[1:]))
    assert any("author" in p for p in patched.verify())


# ─────────────────────────────────────────────────────────────────────────────
# P-06: a supplement does not disturb the blind read
# ─────────────────────────────────────────────────────────────────────────────


def test_a_supplement_leaves_the_blind_revision_hash_intact():
    """P-06: 盲读版本在补充修订后内容哈希不变."""
    envelope = F.build_envelope()
    blind = envelope.revisions[0]

    supplement = Revision(
        id="R-004",
        sample_id=F.SAMPLE,
        assignment_id=blind.assignment_id,
        author_id=blind.author_id,
        revision_no=2,
        is_blind_read=False,
        payload_ref="blob://revisions/R-004",
        payload_hash=content_hash({"sample_id": F.SAMPLE, "author": blind.author_id, "instances": 5}),
        created_at=F.T3,
        derived_from=blind.id,
    )
    extended = envelope.evolve(revisions=(*envelope.revisions, supplement))

    assert extended.revisions[0] == blind
    assert extended.revisions[0].payload_hash == blind.payload_hash
    assert extended.content_hash != envelope.content_hash
    assert extended.verify() == ()


# ─────────────────────────────────────────────────────────────────────────────
# P-11 / D-10: the export gate
# ─────────────────────────────────────────────────────────────────────────────


def test_research_products_may_train():
    check_training_export(F.build_envelope())


def test_teaching_products_are_refused():
    with pytest.raises(ExportRefused, match="may not export"):
        check_training_export(F.build_envelope(template="teaching", with_cosign=False))


def test_a_teaching_run_mislabelled_as_research_is_still_refused():
    """The template field is a claim; the assignments are evidence.

    This is the second, independent check: an import script that copies the
    wrong template constant would otherwise let student labels into a training
    set while every field looks well-formed.
    """
    disguised = F.build_envelope(
        template="research", assignment_roles=(Role.ANNOTATOR,) * 3
    )
    with pytest.raises(ExportRefused, match="trace to ANNOTATOR assignments"):
        check_training_export(disguised)


def test_disguised_envelope_still_passes_structural_verification():
    """The gate is a policy check, not a shape check — they must not be conflated."""
    disguised = F.build_envelope(template="research", assignment_roles=(Role.ANNOTATOR,) * 3)
    assert disguised.verify() == ()


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────


def test_summary_counts_every_collection():
    assert summarise(F.build_envelope()) == {
        "samples": 1,
        "assignments": 3,
        "revisions": 3,
        "disagreements": 1,
        "model_ranks": 1,
        "adjudications": 1,
        "cosigns": 1,
        "stratums": 1,
    }


def test_format_version_is_the_one_we_document():
    assert FORMAT_VERSION == "organoid-exchange-v1"
    assert F.build_envelope().format_version == FORMAT_VERSION


def test_exported_envelope_clears_a_needs_improvement_round():
    envelope = F.build_envelope(state=SampleState.NEEDS_IMPROVEMENT)
    assert envelope.verify() == ()
    assert load_envelope(envelope.dump()).task.state is SampleState.NEEDS_IMPROVEMENT
