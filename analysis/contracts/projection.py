# ── analysis/contracts/projection.py ──
"""Role- and stage-aware projection of contract records (G-3, P-01, P-09).

Why this file is the point of the whole contract layer
-----------------------------------------------------
The MVP audit found eleven defects. Every one of them had the same root cause,
stated in PRD 10.8: *"11 处泄露/污染里没有一处是流程设计错误，全部是投影只做在文案层、
没做在数据层"*. The screen said "hidden", the object still held the answer, and
anything that did not go through the screen — the JSON drawer (L1-2), the audit
chain (H-1/H-2), a role that was never guarded (H-5) — handed it over.

Redaction in the render layer can never fix that, because the render layer is one
of many readers. Projection has to happen where the response is built:

    project(records, viewer, stage) -> dict      # the only thing that ships

Everything in the returned dict was added by name. Nothing is removed from a copy
of the full record, so a new sensitive field added to an entity is invisible
until someone deliberately puts it in an allow-list here. That direction matters:
default-open redaction fails when the data grows, default-closed projection fails
visibly and immediately.

The visibility table
--------------------
Six capabilities, one row per (role, stage). Rationale for the non-obvious ones:

``model_rank``
    Never to a domain expert at any stage. D-03 exists because an ordering is
    enough to move a human judgement; showing it *after* the read is still
    anchoring, because the expert is asked to revise. Never to a cosigner either:
    P-09 requires an independent choice, and a ranking is an opinion they would
    be agreeing with.

``peer_revisions``
    PRD 3.2: a domain expert answers "no" for all time; the chief gets them only
    after disagreement analysis. The chief's row therefore also requires a
    disagreement record to exist, not merely a non-blind stage.

``progress``
    Submission counts before reveal are S-12, the time side channel: an expert
    who sees "2/3 submitted" knows they are last and that the protocol was
    applied by two others already. Hidden in ``BLIND`` for everyone except the
    project owner, who needs it to chase deadlines and cannot annotate.

``preannotation``
    The presence flag is itself a leak in ``BLIND`` — "a pre-annotation exists
    for this sample" tells an expert this sample was already modelled. The key
    is absent, not ``False``; ``False`` still confirms the platform asked.

``stratum``
    A difficulty label read before annotation is a hint about how hard the case
    is supposed to be. Owner, data manager, chief, and anyone after signature.

Nothing here is a security boundary by itself — G-2 puts authentication on the
server. What this module guarantees is that a *correctly authenticated* caller
still cannot obtain what their role does not need, which is the part the eleven
defects were about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

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
from analysis.contracts.enums import ProjectionStage, Role, SampleState
from analysis.contracts.exchange import ExchangeEnvelope

__all__ = [
    "ProjectionContext",
    "Visibility",
    "project_envelope",
    "visibility_for",
]


@dataclass(frozen=True)
class ProjectionContext:
    """Who is asking, for which stage, and what they have already committed.

    ``cosign_locked`` is the only viewer state that changes visibility, and it
    exists because P-09 is an ordering requirement, not a role requirement: the
    cosigner's role is identical before and after they commit their independent
    choice, but what they may see is not. Passing it in rather than deriving it
    from a timestamp keeps the decision auditable — the caller has to look up the
    lock, and a caller that forgets passes ``False`` and leaks nothing.

    There is deliberately **no** ``stage`` field. The stage is derived from the
    sample's own state (:meth:`ProjectionStage.of`), because a caller-supplied
    stage is a caller-supplied permission: one wrong argument and a blind read
    gets a revealed-stage payload. Deriving it makes that unrepresentable.
    """

    viewer_role: Role
    viewer_id: str
    cosign_locked: bool = False


@dataclass(frozen=True)
class Visibility:
    """Which of the six capabilities a projection may include."""

    model_rank: bool = False
    peer_revisions: bool = False
    adjudication: bool = False
    progress: bool = False
    preannotation: bool = False
    stratum: bool = False
    disposition: bool = False  #: reason_text / signature meaning on the decision


_NO_ENTITLEMENT = Visibility()

_RULES: Mapping[Role, Visibility] = {
    # Administrators operate the node; PRD 3.1 says they do not take part in
    # annotation judgement, and 3.2 grants them audit-trail access. They get
    # progress and strata (capacity planning) but never a model ranking, because
    # they never adjudicate and the ranking would only travel further.
    Role.NODE_ADMIN: Visibility(progress=True, stratum=True),
    Role.PROJECT_OWNER: Visibility(progress=True, stratum=True, adjudication=True),
    Role.DATA_MANAGER: Visibility(progress=True, stratum=True),
    Role.DOMAIN_EXPERT: Visibility(preannotation=True),
    Role.CHIEF_EXPERT: Visibility(
        model_rank=True,
        peer_revisions=True,
        adjudication=True,
        progress=True,
        preannotation=True,
        stratum=True,
        disposition=True,
    ),
    # The cosigner sees the adjudication only after locking their own choice;
    # ``visibility_for`` strips it below. Strata and progress are irrelevant to
    # their task and are withheld.
    Role.COSIGNER: Visibility(adjudication=True, preannotation=True, disposition=False),
    Role.REVIEWER: Visibility(
        model_rank=True,
        peer_revisions=True,
        adjudication=True,
        progress=True,
        preannotation=True,
        stratum=True,
        disposition=True,
    ),
    Role.ANNOTATOR: Visibility(preannotation=True),
    Role.OBSERVER: Visibility(adjudication=True),
    Role.SYSTEM: _NO_ENTITLEMENT,
}

#: Roles that may read peer revisions and model rankings at all. Everything else
#: is restricted to ``BLIND``-stage visibility even after the reveal, because
#: 3.2's "否" for a domain expert is not conditioned on the stage.
_PEER_PRIVILEGED: frozenset[Role] = frozenset({Role.CHIEF_EXPERT, Role.REVIEWER})

#: Roles that coordinate and never take part in the judgement. They may see
#: progress and strata before the reveal; everybody else may not.
_COORDINATORS: frozenset[Role] = frozenset(
    {Role.NODE_ADMIN, Role.PROJECT_OWNER, Role.DATA_MANAGER}
)


def visibility_for(
    role: Role,
    stage: ProjectionStage,
    *,
    cosign_locked: bool = False,
    disagreement_present: bool = False,
) -> Visibility:
    """Resolve the six capabilities for one viewer at one stage.

    Exposed (rather than inlined into :func:`project_envelope`) so the policy can
    be asserted directly. A test that checks "the response has no ranking" passes
    if the ranking section is merely empty because the sample has no ranking;
    a test against this function checks the rule.
    """
    base = _RULES.get(role, _NO_ENTITLEMENT)
    if stage is ProjectionStage.BLIND:
        # Pre-reveal, coordination facts go only to roles that cannot take part
        # in the judgement. A chief who learns "2/3 submitted" knows two
        # colleagues have already committed an interpretation, and that reads
        # into the arbitration even though no content was shown — the same
        # time-side-channel class as S-12.
        coordinator = role in _COORDINATORS
        return Visibility(
            progress=base.progress and coordinator,
            stratum=base.stratum and coordinator,
            adjudication=False,
            disposition=False,
        )

    peer = base.peer_revisions and role in _PEER_PRIVILEGED
    if role in _PEER_PRIVILEGED and not disagreement_present and stage is not ProjectionStage.FINAL:
        # PRD 3.2: the chief sees peer revisions only after disagreement analysis.
        # Without it there is no analysis to compare against, and the chief would
        # simply read three answers and pick one.
        peer = False
        rank = False
    else:
        rank = base.model_rank and role in _PEER_PRIVILEGED

    adjudication = base.adjudication
    disposition = base.disposition
    preannotation = base.preannotation
    if role is Role.COSIGNER and not cosign_locked:
        # P-09. Before the lock, the cosigner may know that a cosign is required
        # and nothing else — not the selection, not the reason, not the ranking,
        # and not whether a pre-annotation exists (that alone tells them the
        # sample was modelled, which is a hint about expected difficulty).
        adjudication = False
        disposition = False
        preannotation = False

    return Visibility(
        model_rank=rank,
        peer_revisions=peer,
        adjudication=adjudication,
        progress=base.progress,
        preannotation=preannotation,
        stratum=base.stratum,
        disposition=disposition,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────


def _task_view(task: Task, envelope: ExchangeEnvelope) -> dict[str, Any]:
    """Task header. No ``protocol_id`` pointers, no per-person data."""
    return {
        "id": task.id,
        "state": task.state.value,
        "template": task.template,
        "redundancy_n": task.redundancy_n,
        "deadline": task.deadline,
        "sample_count": len(envelope.samples),
        "protocol": {
            "id": envelope.protocol.id,
            "version": envelope.protocol.version,
            "hash": envelope.protocol.hash,
            "taxonomy_name": envelope.protocol.taxonomy_name,
            "touching_object_rule": envelope.protocol.touching_object_rule,
            "overlap_split_rule": envelope.protocol.overlap_split_rule,
            "min_visible_boundary_um": envelope.protocol.min_visible_boundary_um,
        },
    }


def _revision_view(revision: Revision) -> dict[str, Any]:
    return {
        "id": revision.id,
        "sample_id": revision.sample_id,
        "author_id": revision.author_id,
        "revision_no": revision.revision_no,
        "is_blind_read": revision.is_blind_read,
        "payload_ref": revision.payload_ref,
        "payload_hash": revision.payload_hash,
        "created_at": revision.created_at,
        "derived_from": revision.derived_from,
    }


def _assignment_view(assignment: Optional[Assignment]) -> Optional[dict[str, Any]]:
    if assignment is None:
        return None
    return {
        "id": assignment.id,
        "state": assignment.state.value,
        "role": assignment.role.value,
        "started_at": assignment.started_at,
        "submitted_at": assignment.submitted_at,
    }


def _model_rank_view(rank: ModelRank) -> dict[str, Any]:
    return {
        "model_id": rank.model_id,
        "model_version": rank.model_version,
        "ranking": list(rank.ranking),
        "scores": dict(rank.scores),
        "confidence": rank.confidence,
        "produced_at": rank.produced_at,
    }


def _disagreement_view(disagreement: Disagreement) -> dict[str, Any]:
    return {
        "id": disagreement.id,
        "metric_snapshot": dict(disagreement.metric_snapshot),
        "controversy_score": disagreement.controversy_score,
        "regions_ref": disagreement.regions_ref,
        "computed_at": disagreement.computed_at,
    }


def _adjudication_view(adjudication: Adjudication, *, with_disposition: bool) -> dict[str, Any]:
    view: dict[str, Any] = {
        "id": adjudication.id,
        "selected_revision_id": adjudication.selected_revision_id,
        "selector_id": adjudication.selector_id,
        "decided_at": adjudication.decided_at,
        "reason_code": adjudication.reason_code,
    }
    if with_disposition:
        # The reason text and signature meaning are the *reasoning*. A cosigner
        # who has not yet committed would be reading the chief's argument.
        view["reason_text"] = adjudication.reason_text
        view["signature_meaning"] = adjudication.signature_meaning
        view["evidence_ref"] = adjudication.evidence_ref
    return view


def _cosign_view(cosign: Cosign, *, viewer_id: str) -> Optional[dict[str, Any]]:
    """Only the viewer's own cosign record.

    Another person's cosign carries the same information as the adjudication
    (``blind_choice`` is usually the answer), so it is never projected.
    """
    if cosign.cosigner_id != viewer_id:
        return None
    return {
        "id": cosign.id,
        "trigger_rule": cosign.trigger_rule,
        "blind_choice": cosign.blind_choice,
        "blind_choice_locked_at": cosign.blind_choice_locked_at,
        "verdict": None if cosign.verdict is None else cosign.verdict.value,
        "dissent_text": cosign.dissent_text,
        "revealed_at": cosign.revealed_at,
        "decided_at": cosign.decided_at,
    }


def _stratum_view(label: StratumLabel) -> dict[str, Any]:
    return {
        "stratum": label.stratum.value,
        "rule_version": label.rule_version,
        "labeled_at": label.labeled_at,
    }


def project_envelope(envelope: ExchangeEnvelope, context: ProjectionContext) -> dict[str, Any]:
    """Build the only response shape a client is allowed to receive.

    The return value is a plain dict of JSON-safe primitives, ready to serialise.
    Sections that the viewer may not see are **absent**, not ``None``: a client
    that renders ``sample["model_rank"]`` on a sample where the key is missing
    fails loudly during development, while a client receiving ``None`` everywhere
    learns to treat ``None`` as "not allowed" and to render a placeholder — and
    one forgotten branch later the placeholder becomes the answer.
    """
    by_sample_revisions: dict[str, list[Revision]] = {}
    for revision in envelope.revisions:
        by_sample_revisions.setdefault(revision.sample_id, []).append(revision)

    by_sample: dict[str, dict[str, Any]] = {}
    for sample_id in envelope.samples:
        by_sample[sample_id] = {
            "sample_id": sample_id,
            "revisions": [],
        }

    for assignment in envelope.assignments:
        slot = by_sample.get(assignment.sample_id)
        if slot is not None and assignment.assignee_id == context.viewer_id:
            slot["assignment"] = _assignment_view(assignment)

    disagreement_by_sample = {d.sample_id: d for d in envelope.disagreements}
    rank_by_sample = {r.sample_id: r for r in envelope.model_ranks}
    adjudication_by_sample = {a.sample_id: a for a in envelope.adjudications}
    stratum_by_sample = {s.sample_id: s for s in envelope.stratums}
    cosign_by_sample: dict[str, Cosign] = {}
    for cosign in envelope.cosigns:
        if cosign.cosigner_id == context.viewer_id:
            cosign_by_sample[cosign.sample_id] = cosign

    sample_states = _sample_states(envelope)
    submitted_by_sample = _submitted_counts(envelope)

    samples: list[dict[str, Any]] = []
    for sample_id in envelope.samples:
        view = by_sample[sample_id]
        state = sample_states[sample_id]
        stage = ProjectionStage.of(state)
        disagreement = disagreement_by_sample.get(sample_id)
        visibility = visibility_for(
            context.viewer_role,
            stage,
            cosign_locked=context.cosign_locked,
            disagreement_present=disagreement is not None,
        )

        view["state"] = state.value
        view["stage"] = stage.value

        own = [r for r in by_sample_revisions.get(sample_id, ()) if r.author_id == context.viewer_id]
        if visibility.peer_revisions:
            visible_revisions = list(by_sample_revisions.get(sample_id, ()))
        else:
            visible_revisions = own
        view["revisions"] = [_revision_view(r) for r in visible_revisions]

        if visibility.progress:
            view["progress"] = {
                "submitted": submitted_by_sample.get(sample_id, 0),
                "required": envelope.task.redundancy_n,
            }
        if visibility.preannotation:
            view["preannotation"] = {"available": bool(rank_by_sample.get(sample_id))}
        if visibility.stratum and sample_id in stratum_by_sample:
            view["stratum"] = _stratum_view(stratum_by_sample[sample_id])
        if visibility.model_rank and sample_id in rank_by_sample:
            view["model_rank"] = _model_rank_view(rank_by_sample[sample_id])
        if visibility.peer_revisions and disagreement is not None:
            view["disagreement"] = _disagreement_view(disagreement)
        if visibility.adjudication and sample_id in adjudication_by_sample:
            view["adjudication"] = _adjudication_view(
                adjudication_by_sample[sample_id], with_disposition=visibility.disposition
            )
        if sample_id in cosign_by_sample:
            cosign_view = _cosign_view(cosign_by_sample[sample_id], viewer_id=context.viewer_id)
            if cosign_view is not None:
                view["cosign"] = cosign_view

        samples.append(view)

    return {
        "stage": ProjectionStage.of(envelope.task.state).value,
        "viewer": {"role": context.viewer_role.value, "id": context.viewer_id},
        "task": _task_view(envelope.task, envelope),
        "samples": samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# State and progress derivation
# ─────────────────────────────────────────────────────────────────────────────


def _sample_states(envelope: ExchangeEnvelope) -> Mapping[str, SampleState]:
    """Per-sample state.

    The envelope carries no per-sample state field — ``Task.state`` is the
    aggregate (see ``entities.Task``). For M0 every sample in an envelope shares
    the task's aggregate state, because an envelope is written at a point where
    the task is coherent. M1 persists per-sample states; when it does, this
    function is the single place that has to change.
    """
    return {sample_id: envelope.task.state for sample_id in envelope.samples}


def _submitted_counts(envelope: ExchangeEnvelope) -> Mapping[str, int]:
    """Distinct blind revisions per sample."""
    counts: dict[str, int] = {}
    for revision in envelope.revisions:
        if revision.is_blind_read:
            counts[revision.sample_id] = counts.get(revision.sample_id, 0) + 1
    return counts
