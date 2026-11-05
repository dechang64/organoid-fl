# ── tests/test_contracts_templates.py ──
"""Template tests — including the M0 exit criterion, executed.

PRD 10.4 states the exit as: *"新增一个 template 只改配置与编排、不改数据模型"*.
``test_a_third_template_needs_configuration_only`` is that sentence turned into a
run: it registers a template that exists nowhere in the codebase, drives the same
state machine through it with a different cast, round-trips an exchange envelope
that carries its id, and asserts the entity list is byte-identical to the one
before. If a future template ever needs an entity change, that test fails first.
"""

from __future__ import annotations

import dataclasses

import pytest

from analysis.contracts import contract_report
from analysis.contracts.enums import Role, SampleState
from analysis.contracts.exchange import ExchangeEnvelope, load_envelope
from analysis.contracts.state_machine import TransitionFacts, sample_machine
from analysis.contracts.templates import (
    OPERATIONS,
    RESEARCH_TEMPLATE,
    SYSTEM_ONLY,
    TEACHING_TEMPLATE,
    TEMPLATES,
    TemplateError,
    TemplateSpec,
    get_template,
    register_template,
)
from tests import contracts_factory as F


# ─────────────────────────────────────────────────────────────────────────────
# The two shipped templates
# ─────────────────────────────────────────────────────────────────────────────


def test_research_template_matches_the_four_level_flow():
    assert RESEARCH_TEMPLATE.redundancy_range == (2, 5)
    assert RESEARCH_TEMPLATE.default_redundancy == 3
    assert RESEARCH_TEMPLATE.agreement_is_gate is True
    assert RESEARCH_TEMPLATE.cosign_available is True
    assert RESEARCH_TEMPLATE.training_export_allowed is True
    assert RESEARCH_TEMPLATE.allows("begin_blind_read", Role.DOMAIN_EXPERT)
    assert RESEARCH_TEMPLATE.allows("adjudicate", Role.CHIEF_EXPERT)
    assert not RESEARCH_TEMPLATE.allows("adjudicate", Role.REVIEWER)


def test_teaching_template_is_single_redundancy_and_cannot_train():
    assert TEACHING_TEMPLATE.redundancy_range == (1, 1)
    assert TEACHING_TEMPLATE.agreement_is_gate is False
    assert TEACHING_TEMPLATE.cosign_available is False
    assert TEACHING_TEMPLATE.training_export_allowed is False
    assert TEACHING_TEMPLATE.allows("begin_blind_read", Role.ANNOTATOR)
    assert TEACHING_TEMPLATE.allows("adjudicate", Role.REVIEWER)


def test_teaching_template_cannot_reach_the_cosign_state():
    """D-10 has no cosign; a state nothing can enter would look permanently stuck."""
    assert SampleState.AWAITING_COSIGN not in TEACHING_TEMPLATE.states
    assert SampleState.AWAITING_COSIGN in RESEARCH_TEMPLATE.states


def test_both_templates_cast_every_operation():
    for spec in (RESEARCH_TEMPLATE, TEACHING_TEMPLATE):
        assert set(spec.cast) == set(OPERATIONS)


def test_platform_steps_cast_nobody():
    for operation in SYSTEM_ONLY:
        assert RESEARCH_TEMPLATE.roles_for(operation) == frozenset()
        assert TEACHING_TEMPLATE.roles_for(operation) == frozenset()


def test_lookup_accepts_enum_member_and_string():
    from analysis.contracts.enums import Template

    assert get_template(Template.RESEARCH) is RESEARCH_TEMPLATE
    assert get_template("research") is RESEARCH_TEMPLATE
    with pytest.raises(TemplateError, match="not a registered template"):
        get_template("nonexistent")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def _spec(**overrides) -> TemplateSpec:
    base = {
        "template": "probe",
        "label": "探针",
        "redundancy_range": (2, 3),
        "default_redundancy": 2,
        "blind_read_required": True,
        "reveal_required": True,
        "agreement_is_gate": True,
        "cosign_available": False,
        "training_export_allowed": True,
        "cast": dict(TEACHING_TEMPLATE.cast),
        "states": TEACHING_TEMPLATE.states,
    }
    base.update(overrides)
    return TemplateSpec(**base)


def test_cast_may_not_claim_a_platform_step():
    with pytest.raises(TemplateError, match="platform step"):
        _spec(cast={**dict(TEACHING_TEMPLATE.cast), "begin_review": frozenset({Role.CHIEF_EXPERT})})


def test_cast_may_not_name_the_system_actor():
    with pytest.raises(TemplateError, match="contains SYSTEM"):
        _spec(cast={**dict(TEACHING_TEMPLATE.cast), "abort": frozenset({Role.SYSTEM})})


def test_every_operation_must_be_cast():
    partial = dict(TEACHING_TEMPLATE.cast)
    del partial["abort"]
    with pytest.raises(TemplateError, match="missing operation"):
        _spec(cast=partial)


def test_cosign_disabled_must_not_cast_cosign_roles():
    with pytest.raises(TemplateError, match="cosign_available is False"):
        _spec(cast={**dict(TEACHING_TEMPLATE.cast), "cosign_approve": frozenset({Role.COSIGNER})})


def test_redundancy_default_must_lie_in_range():
    with pytest.raises(TemplateError, match="outside the declared range"):
        _spec(default_redundancy=4)
    with pytest.raises(TemplateError, match="redundancy_range"):
        _spec(redundancy_range=(0, 9))


def test_states_must_start_at_draft_and_be_unique():
    with pytest.raises(TemplateError, match="start at DRAFT"):
        _spec(states=(SampleState.PUBLISHED, SampleState.DRAFT))
    with pytest.raises(TemplateError, match="repeats"):
        _spec(states=(SampleState.DRAFT, SampleState.DRAFT))


def test_registering_a_different_spec_under_a_taken_id_is_refused():
    with pytest.raises(TemplateError, match="already registered"):
        register_template(
            dataclasses.replace(TEACHING_TEMPLATE, default_redundancy=1, notes="different")
        )


def test_registering_the_identical_spec_is_idempotent():
    assert register_template(TEACHING_TEMPLATE) is TEACHING_TEMPLATE


# ─────────────────────────────────────────────────────────────────────────────
# The M0 exit criterion
# ─────────────────────────────────────────────────────────────────────────────


THIRD = TemplateSpec(
    template="two_reader_replay",
    label="双读回放",
    redundancy_range=(2, 2),
    default_redundancy=2,
    blind_read_required=True,
    reveal_required=True,
    agreement_is_gate=True,
    cosign_available=False,
    training_export_allowed=True,
    cast={
        **{operation: frozenset() for operation in OPERATIONS},
        "publish": frozenset({Role.PROJECT_OWNER}),
        # Two readers, and the signer is one of them. Neither PRD template has
        # this cast, so the machine really is being driven by configuration.
        "begin_blind_read": frozenset({Role.DOMAIN_EXPERT}),
        "adjudicate": frozenset({Role.DOMAIN_EXPERT}),
        "abort": frozenset({Role.NODE_ADMIN}),
    },
    states=tuple(s for s in RESEARCH_TEMPLATE.states if s is not SampleState.AWAITING_COSIGN),
    notes="M0 出口验证用：仅由配置声明，代码中不存在该模板的任何分支。",
)


def test_a_third_template_needs_configuration_only():
    """The M0 exit criterion, executed end to end."""
    entities_before = contract_report()["entities"]

    register_template(THIRD, overwrite=True)
    try:
        machine = sample_machine()
        spec = get_template("two_reader_replay")
        fact = TransitionFacts(
            template=spec,
            redundancy_n=2,
            blind_reads_submitted=2,
            reveal_done=True,
            supplements_collected=True,
            disagreement_computed=True,
            gate_passed=True,
        )

        # 1. The same state machine drives it, with this template's cast.
        state = SampleState.DRAFT
        for operation, role in (
            ("publish", Role.PROJECT_OWNER),
            ("begin_blind_read", Role.DOMAIN_EXPERT),
            ("complete_reveal", Role.SYSTEM),
            ("begin_review", Role.SYSTEM),
            ("request_adjudication", Role.SYSTEM),
            ("adjudicate", Role.DOMAIN_EXPERT),
            ("admit_to_dataset", Role.SYSTEM),
        ):
            state = machine.apply(state, operation, fact, actor_role=role).state
        assert state is SampleState.IN_DATASET

        # 2. Its records serialise through an unmodified envelope.
        two = F.revisions(blind=2)
        envelope = ExchangeEnvelope(
            format_version="organoid-exchange-v1",
            node_id=F.NODE,
            template="two_reader_replay",
            exported_at=F.T3,
            protocol=F.protocol(),
            task=F.task(template="two_reader_replay", redundancy_n=2),
            samples=(F.SAMPLE,),
            assignments=F.assignments()[:2],
            revisions=two,
            disagreements=(F.disagreement(),),
            model_ranks=(F.model_rank(revision_ids=("R-001", "R-002")),),
            adjudications=(F.adjudication(selected="R-001"),),
            cosigns=(),
            stratums=(F.stratum(),),
        )
        assert envelope.verify() == ()
        assert load_envelope(envelope.dump()) == envelope

        # 3. No entity changed. This is the assertion the criterion is about.
        assert contract_report()["entities"] == entities_before
    finally:
        TEMPLATES.pop("two_reader_replay", None)


def test_a_third_template_is_not_leaked_into_the_registry_by_other_tests():
    assert "two_reader_replay" not in TEMPLATES
