# ── tests/contracts_factory.py ──
"""Builders for contract test fixtures.

Not a test module (pytest collects ``test_*.py``), so this file is imported
explicitly. It exists because six test modules need the same "one plausible
sample, fully worked" envelope, and six hand-written copies would drift — the
second copy would inevitably use a different timestamp format and then a
projection test would fail for a reason that has nothing to do with projection.

Everything here is deterministic: fixed ids, fixed timestamps, no ``uuid4`` and
no ``now()``. A test that fails must fail identically on the next run.
"""

from __future__ import annotations

from typing import Optional, Sequence

from analysis.contracts.entities import (
    Adjudication,
    Assignment,
    Cosign,
    Disagreement,
    ModelRank,
    Revision,
    StratumLabel,
    Task,
)
from analysis.contracts.enums import (
    AssignmentState,
    CosignVerdict,
    Role,
    SampleState,
    Stratum,
    Template,
)
from analysis.contracts.exchange import ExchangeEnvelope
from analysis.contracts.hashing import content_hash
from analysis.contracts.protocol import make_protocol

SAMPLE = "S-001"
TASK_ID = "T-001"
NODE = "node-sz-01"
T0 = "2026-11-01T09:00:00Z"
T1 = "2026-11-01T09:30:00Z"
T2 = "2026-11-01T10:00:00Z"
T3 = "2026-11-01T11:00:00Z"

EXPERTS = ("expert_a", "expert_b", "expert_c")
CHIEF = "chief_1"
COSIGNER = "cosign_1"


def protocol(**overrides):
    """A valid protocol bound to the default registered taxonomy."""
    kwargs = {
        "id": "proto-organoid",
        "version": "1.0",
        "taxonomy_name": "organoid_4class",
        "touching_object_rule": "split_at_narrowest_neck",
        "min_visible_boundary_um": 2.5,
        "overlap_split_rule": "watershed_on_distance_transform",
        "effective_from": "2026-10-01T00:00:00Z",
    }
    kwargs.update(overrides)
    return make_protocol(**kwargs)


def task(*, state: SampleState = SampleState.IN_REVIEW, template: Template = Template.RESEARCH, **overrides):
    kwargs = {
        "id": TASK_ID,
        "node_id": NODE,
        "project_id": "P-001",
        "template": template,
        "image_scope": (SAMPLE,),
        "redundancy_n": 3 if template is Template.RESEARCH else 1,
        "preannotation_policy": "tiered_confidence",
        "protocol_id": "proto-organoid",
        "state": state,
        "deadline": "2026-11-15T00:00:00Z",
    }
    kwargs.update(overrides)
    return Task(**kwargs)


def assignments(
    *, template: Template = Template.RESEARCH, roles: Optional[Sequence[Role]] = None
) -> tuple[Assignment, ...]:
    people = EXPERTS if template is Template.RESEARCH else ("student_1",)
    if roles is None:
        roles = [Role.DOMAIN_EXPERT if template is Template.RESEARCH else Role.ANNOTATOR] * len(people)
    out = []
    for index, (person, role) in enumerate(zip(people, roles, strict=False), start=1):
        out.append(
            Assignment(
                id=f"A-{index:03d}",
                task_id=TASK_ID,
                sample_id=SAMPLE,
                assignee_id=person,
                role=role,
                state=AssignmentState.SUBMITTED,
                started_at=T0,
                submitted_at=T1,
            )
        )
    return tuple(out)


def revisions(*, template: Template = Template.RESEARCH, blind: int = 3) -> tuple[Revision, ...]:
    people = EXPERTS if template is Template.RESEARCH else ("student_1",)
    out = []
    for index, person in enumerate(people[:blind], start=1):
        payload = {"sample_id": SAMPLE, "author": person, "instances": index * 4}
        out.append(
            Revision(
                id=f"R-{index:03d}",
                sample_id=SAMPLE,
                assignment_id=f"A-{index:03d}",
                author_id=person,
                revision_no=1,
                is_blind_read=True,
                payload_ref=f"blob://revisions/R-{index:03d}",
                payload_hash=content_hash(payload),
                created_at=T1,
            )
        )
    return tuple(out)


def disagreement(score: float = 0.42) -> Disagreement:
    return Disagreement(
        id="D-001",
        sample_id=SAMPLE,
        metric_snapshot={
            "alpha_nominal": 0.64,
            "mean_dice": 0.81,
            "hd95_um": 12.5,
            "n_revisions": 3,
        },
        regions_ref="blob://disagreement/D-001",
        controversy_score=score,
        computed_at=T2,
    )


def model_rank(*, revision_ids: Sequence[str] = ("R-001", "R-002", "R-003")) -> ModelRank:
    return ModelRank(
        id="M-001",
        sample_id=SAMPLE,
        model_id="organoid-seg",
        model_version="0.9.3",
        ranking=tuple(revision_ids),
        scores={rid: round(0.9 - 0.1 * i, 2) for i, rid in enumerate(revision_ids)},
        confidence=0.71,
        produced_at=T2,
    )


def adjudication(*, selected: str = "R-002", selector: str = CHIEF) -> Adjudication:
    return Adjudication(
        id="J-001",
        sample_id=SAMPLE,
        selected_revision_id=selected,
        selector_id=selector,
        reason_code="boundary_quality",
        reason_text="R-002 separates the two touching bodies at the narrowest neck.",
        evidence_ref="blob://adjudication/J-001",
        signature_meaning="采纳模型排序建议；以边界质量为准。",
        decided_at=T3,
    )


def cosign(
    *, cosigner: str = COSIGNER, verdict: CosignVerdict = CosignVerdict.APPROVE
) -> Cosign:
    return Cosign(
        id="C-001",
        sample_id=SAMPLE,
        adjudication_id="J-001",
        trigger_rule="controversy_top_decile",
        cosigner_id=cosigner,
        blind_choice="R-002",
        blind_choice_locked_at=T2,
        verdict=verdict,
        dissent_text="boundary accepted but flag the cystic edge" if verdict is CosignVerdict.REJECT else None,
        revealed_at="2026-11-01T10:30:00Z",
        decided_at=T3,
    )


def stratum(value: Stratum = Stratum.BORDERLINE) -> StratumLabel:
    return StratumLabel(
        sample_id=SAMPLE,
        stratum=value,
        signal_snapshot={"model_confidence_p10": 0.31, "mean_gradient": 0.08},
        labeled_at=T0,
        rule_version="stratum-rule-v1",
    )


def build_envelope(
    *,
    template: Template = Template.RESEARCH,
    state: SampleState = SampleState.IN_REVIEW,
    with_disagreement: bool = True,
    with_rank: bool = True,
    with_adjudication: bool = True,
    with_cosign: bool = True,
    assignment_roles: Optional[Sequence[Role]] = None,
    blind: Optional[int] = None,
) -> ExchangeEnvelope:
    """A complete, self-consistent envelope for one sample.

    Defaults describe the interesting case: three blind reads in, disagreement
    computed, model ranking present, chief signed, cosign approved. Tests that
    care about an earlier stage pass ``state=...`` and drop the later records;
    tests that care about visibility pass a different role to the projection.
    """
    n = 3 if template is Template.RESEARCH else 1
    revs = revisions(template=template, blind=n if blind is None else blind)
    # The ranking must permute exactly the revisions that exist, and the
    # adjudication must select one of them, so both are derived rather than
    # fixed: a hardcoded three-way ranking would make every teaching-template
    # envelope fail verification for the wrong reason.
    rank = model_rank(revision_ids=tuple(r.id for r in revs))
    selected = "R-002" if len(revs) > 1 else revs[0].id
    return ExchangeEnvelope(
        format_version="organoid-exchange-v1",
        node_id=NODE,
        template=template,
        exported_at=T3,
        protocol=protocol(),
        task=task(state=state, template=template),
        samples=(SAMPLE,),
        assignments=assignments(template=template, roles=assignment_roles),
        revisions=revs,
        disagreements=(disagreement(),) if with_disagreement else (),
        model_ranks=(rank,) if with_rank else (),
        adjudications=(adjudication(selected=selected),) if with_adjudication else (),
        cosigns=(cosign(),) if with_cosign else (),
        stratums=(stratum(),),
    )
