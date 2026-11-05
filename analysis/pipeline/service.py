# ── analysis/pipeline/service.py ──
"""The write path and the projected read path.

M1 (PRD 10.4) is "创建任务 → 盲读提交 → 落库 → 写审计 → 读回". M2 adds the two
things that make the revealed stage usable and auditable: disagreement analysis
triggered by the workflow itself (P-02), and a read path that hands every viewer
a *projected* payload rather than the raw rows (P-01). The supplement revision
(P-06) closes the same stage.

Assembled from the M0 contracts instead of beside them:

* every state change goes through ``analysis.contracts.state_machine``, so the
  role cast and the guards are enforced in the one place all callers share —
  the server-side role enforcement that G-2 demands and the MVP proved a view
  guard cannot provide;
* every change appends to the persistent audit chain, and the block carries
  identifiers and hashes only — the payload data lives in the entity rows the
  block points at. An audit record that duplicates the data it attests to
  becomes a second copy that can disagree;
* entities are stored immutably; a state advance writes a *new* frozen record
  (``overwrite=True`` on the aggregate Task only) and the audit chain records
  why.

The identity check is here, not in the machine: the machine verifies that a
role may perform an operation, and this service verifies that this *person*
holds that role *on this sample* through a real Assignment. A role without an
assignment is the "anyone can switch the role dropdown" defect the tester hit
in the MVP (G-2).

Why the read path lives here
----------------------------
``view_task`` is the only method a client should ever call to learn what a
sample contains. ``read_back`` exists for verifiers and returns everything on
purpose; giving a client that one is how the eleven MVP leaks happened, because
"the view filters it" is not a control. Here the projection is applied to the
data before it leaves the process, and the sections a viewer may not see are
absent from the response rather than blanked.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Mapping, Optional

from analysis.contracts.entities import (
    AnnotationProtocol,
    Assignment,
    ContractValidationError,
    Disagreement,
    Revision,
    StratumLabel,
    Task,
)
from analysis.contracts.enums import AssignmentState, Role, SampleState
from analysis.contracts.hashing import content_hash
from analysis.contracts.projection import ProjectionContext, project_task
from analysis.contracts.state_machine import (
    IllegalTransition,
    TransitionFacts,
    aggregate_task_state,
    advance_assignment,
    sample_machine,
)
from analysis.contracts.templates import get_template
from analysis.pipeline.chain import PersistentAuditChain
from analysis.pipeline.disagreement import (
    DisagreementError,
    compute_disagreement as analyse_disagreement,
)
from analysis.pipeline.geometry import CoderPayload, GeometryError
from analysis.pipeline.store import StoreError, SQLiteStore
from analysis.pipeline.stratum import StratumRuleError
from analysis.pipeline.stratum import label_stratum as build_stratum_label


class ServiceError(ValueError):
    """The service refused an operation before touching the machine."""


class AnnotationService:
    """Write path over one store. One service per store, one store per node."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store
        self.chain = PersistentAuditChain(store)
        self._machine = sample_machine()

    # ── task creation ────────────────────────────────────────────────

    def create_task(
        self,
        *,
        task_id: str,
        node_id: str,
        project_id: str,
        template: str,
        image_scope: Iterable[str],
        protocol_id: str,
        protocol: Optional[AnnotationProtocol] = None,
        preannotation_policy: str = "reveal_after_blind",
        redundancy_n: Optional[int] = None,
        deadline: Optional[str] = None,
        now: Optional[str] = None,
    ) -> Task:
        """Create a Task in DRAFT plus one DRAFT state row per sample.

        ``redundancy_n`` defaults to the template's default (3 for research,
        1 for teaching); D-07 lets a project lower it to the template's range
        floor, which the Task constructor validates.

        ``protocol`` is optional because M1 tasks were created without one, but a
        task without a stored protocol cannot be projected (``view_task``): the
        viewer would receive labels with no statement of the rules that produced
        them, and PRD 7.4 requires the rules to travel with the data.
        """
        spec = get_template(template)
        if protocol is not None and protocol.id != protocol_id:
            raise ServiceError(
                f"protocol.id {protocol.id!r} does not match protocol_id "
                f"{protocol_id!r}; storing both would let the row and the record "
                "name different rules"
            )
        task = Task(
            id=task_id,
            node_id=node_id,
            project_id=project_id,
            template=spec.template,
            image_scope=tuple(image_scope),
            redundancy_n=spec.default_redundancy if redundancy_n is None else redundancy_n,
            preannotation_policy=preannotation_policy,
            protocol_id=protocol_id,
            state=SampleState.DRAFT,
            deadline=deadline,
        )
        self.store.put(task)
        if protocol is not None:
            self.store.put(protocol)
        for sample_id in task.image_scope:
            self.store.set_sample_state(task_id, sample_id, SampleState.DRAFT.value, reveal_done=False)
        self.chain.append(
            "task_created",
            {
                "task_id": task.id,
                "template": task.template,
                "redundancy_n": task.redundancy_n,
                "samples": len(task.image_scope),
                "protocol_id": task.protocol_id,
            },
            timestamp=now,
        )
        return task

    # ── ingestion: the difficulty stratum ────────────────────────────

    def label_stratum(
        self,
        task_id: str,
        sample_id: str,
        *,
        signals: Mapping[str, Any],
        rule_name: str = "provisional_v1",
        allow_provisional: bool = False,
        now: Optional[str] = None,
    ) -> StratumLabel:
        """Record a sample's difficulty stratum at ingestion (D-08, Q-07).

        Allowed only before the sample is worked on. After the first blind read
        the stratum would be a label informed by the very difficulty it claims to
        predict, and PRD 4.8 is explicit that it cannot be reconstructed later —
        so it is written once, immutably, and a relabel needs a new decision.

        ``allow_provisional`` is the Q-07 gate: the only rule that ships is
        provisional, and using it is an explicit act rather than a default.
        """
        task = self._task(task_id)
        if sample_id not in task.image_scope:
            raise ServiceError(f"sample {sample_id!r} is not in task {task_id!r}")
        state = self._sample_state(task_id, sample_id)
        if state is not SampleState.DRAFT and state is not SampleState.PUBLISHED:
            raise ServiceError(
                f"sample {sample_id} is {state.value}; strata are assigned at ingestion, "
                "before anyone reads the sample. Labelling it now would use the "
                "annotation as evidence for the label it is supposed to test."
            )
        try:
            label = build_stratum_label(
                sample_id,
                signals,
                rule_name=rule_name,
                allow_provisional=allow_provisional,
                labeled_at=now,
            )
        except StratumRuleError as exc:
            raise ServiceError(str(exc)) from exc
        self.store.put(label, key=sample_id)
        self.chain.append(
            "stratum_labeled",
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "stratum": label.stratum.value,
                "rule_version": label.rule_version,
                "label_hash": label.content_hash(),
            },
            timestamp=now,
        )
        return label

    def stratum(self, sample_id: str) -> Optional[StratumLabel]:
        payload = self.store.get("stratum_label", sample_id)
        return None if payload is None else StratumLabel.from_dict(payload)

    # ── publish ──────────────────────────────────────────────────────

    def publish(
        self,
        task_id: str,
        *,
        actor_id: str,
        readers: Iterable[tuple[str, Role]],
        now: Optional[str] = None,
    ) -> Task:
        """Publish the task and mint the Assignments.

        ``readers`` is the cast of people who will do the blind reads, in
        order: exactly ``redundancy_n`` of them, each holding a role the
        template's ``begin_blind_read`` cast admits. Fewer than N readers would
        make the ``all_blind_reads_in`` guard unsatisfiable — the sample would
        hang in IN_BLIND_READ forever — so that is a creation-time error, not
        something to discover mid-workflow.
        """
        task = self._task(task_id)
        spec = get_template(task.template)
        reader_list = [(person, role) for person, role in readers]
        if len(reader_list) != task.redundancy_n:
            raise ServiceError(
                f"task {task_id} needs exactly {task.redundancy_n} readers, got "
                f"{len(reader_list)}; a shortfall makes all_blind_reads_in "
                "unsatisfiable and the sample would hang in IN_BLIND_READ"
            )
        admitted = spec.roles_for("begin_blind_read")
        for person, role in reader_list:
            if role not in admitted:
                raise ServiceError(
                    f"{person!r} holds {role.value!r}, which template "
                    f"{spec.template!r} does not cast for begin_blind_read; allowed: "
                    f"{sorted(r.value for r in admitted)}"
                )

        if spec.requires_stratum:
            unlabelled = [s for s in task.image_scope if self.store.get("stratum_label", s) is None]
            if unlabelled:
                raise ServiceError(
                    f"template {spec.template!r} requires a difficulty stratum for every "
                    f"sample, and {len(unlabelled)} sample(s) have none: "
                    f"{unlabelled[:5]}. D-08: ingestion fails rather than defaulting to "
                    "'easy', because the stratum keys the held-out split and cannot be "
                    "reconstructed after the work is done."
                )

        result = self._apply_per_sample(task, "publish", actor_id=actor_id, actor_role=Role.PROJECT_OWNER)
        for sample_id in task.image_scope:
            for person, role in reader_list:
                assignment = Assignment(
                    id=f"{task_id}:{sample_id}:{person}",
                    task_id=task_id,
                    sample_id=sample_id,
                    assignee_id=person,
                    role=role,
                    state=AssignmentState.ASSIGNED,
                    started_at=now,
                )
                self.store.put(assignment)
        self.chain.append(
            "task_published",
            {
                "task_id": task_id,
                "actor_id": actor_id,
                "readers": [person for person, _role in reader_list],
                "assignments": len(reader_list) * len(task.image_scope),
                "guard_evidence_hash": result.audit["guard_evidence_hash"],
            },
            timestamp=now,
        )
        return self._task(task_id)

    # ── the blind read ───────────────────────────────────────────────

    def save_draft(self, task_id: str, sample_id: str, *, actor_id: str, now: Optional[str] = None) -> Assignment:
        """Open the blind read: ASSIGNED → BLIND_DRAFTED, and the sample
        PUBLISHED → IN_BLIND_READ on the first reader's arrival."""
        task = self._task(task_id)
        assignment = self._assignment(task_id, sample_id, actor_id)
        if assignment.state is AssignmentState.ASSIGNED:
            new_state = advance_assignment(
                assignment.state, "save_draft", task_state=self._sample_state(task_id, sample_id)
            )
            assignment = self._replace_assignment(assignment, state=new_state)
            self._maybe_begin_blind_read(task, sample_id, actor_id=actor_id, now=now)
        return assignment

    def submit_blind(
        self,
        task_id: str,
        sample_id: str,
        *,
        actor_id: str,
        payload: bytes,
        now: Optional[str] = None,
    ) -> Revision:
        """Commit one blind read: revision + assignment advance + audit.

        An ASSIGNED reader is taken through ``save_draft`` first — submitting
        is drafting plus committing, and splitting them into two caller-visible
        steps would only invite a caller to forget the first one. The revision
        is written before the audit block, so a crash between the two leaves an
        unattested revision rather than an audit block attesting to nothing.
        """
        _ = self._task(task_id)
        assignment = self.save_draft(task_id, sample_id, actor_id=actor_id, now=now)
        if assignment.state is not AssignmentState.BLIND_DRAFTED:
            raise ServiceError(
                f"assignment {assignment.id} is {assignment.state.value}; a blind "
                "submission requires BLIND_DRAFTED (it may already be submitted)"
            )
        existing = self._revisions_for_assignment(assignment.id)
        revision_no = len(existing) + 1
        # Parsed before it becomes a Revision, not when somebody first reads the
        # sample: a payload the metric layer cannot measure is a submission that
        # will block adjudication later, and refusing it now is the only point
        # where the coder is still there to fix it.
        try:
            CoderPayload.parse(payload)
        except GeometryError as exc:
            raise ServiceError(
                f"payload for sample {sample_id} is not a measurable annotation: {exc}"
            ) from exc
        ref = self.store.put_blob(payload, stored_at=now or _now())
        revision = Revision(
            id=f"{assignment.id}:r{revision_no}",
            sample_id=sample_id,
            assignment_id=assignment.id,
            author_id=actor_id,
            revision_no=revision_no,
            is_blind_read=True,
            payload_ref=ref,
            payload_hash=ref[len("blob:") :],
            created_at=now or _now(),
        )
        self.store.put(revision)
        assignment = self._replace_assignment(
            assignment,
            state=advance_assignment(
                assignment.state, "submit_blind", task_state=self._sample_state(task_id, sample_id)
            ),
            submitted_at=now,
        )

        submitted = len(
            [a for a in self._assignments_for_sample(task_id, sample_id) if a.state is AssignmentState.SUBMITTED]
        )
        self.chain.append(
            "blind_submitted",
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "assignment_id": assignment.id,
                "revision_id": revision.id,
                "payload_hash": revision.payload_hash,
                "blind_reads_submitted": submitted,
                "redundancy_n": self._task(task_id).redundancy_n,
            },
            timestamp=now,
        )
        self._maybe_complete_reveal(task_id, sample_id, submitted=submitted, now=now)
        return revision

    def _maybe_begin_blind_read(self, task: Task, sample_id: str, *, actor_id: str, now: Optional[str]) -> None:
        state = self._sample_state(task_id=task.id, sample_id=sample_id)
        if state is not SampleState.PUBLISHED:
            return
        assignment = self._assignment(task.id, sample_id, actor_id)
        self._apply(
            task,
            sample_id,
            "begin_blind_read",
            actor_id=actor_id,
            actor_role=assignment.role,
            now=now,
        )

    def _maybe_complete_reveal(self, task_id: str, sample_id: str, *, submitted: int, now: Optional[str]) -> None:
        """Platform step: when the Nth blind read lands, the sample moves to
        AWAITING_REVEAL. Recorded under SYSTEM, never under the operator."""
        task = self._task(task_id)
        if submitted < task.redundancy_n:
            return
        state = self._sample_state(task_id, sample_id)
        if state is not SampleState.IN_BLIND_READ:
            return
        self._apply(
            task,
            sample_id,
            "complete_reveal",
            actor_id="",
            actor_role=Role.SYSTEM,
            now=now,
        )
        # P-02: the analysis is triggered by the workflow, not by somebody
        # remembering to run it. Computing it here (rather than at reveal) is what
        # makes the 60-second requirement achievable: at this instant all the
        # inputs exist, and nothing has to wait for a human to open a screen.
        # Nothing is exposed yet — AWAITING_REVEAL projects as the blind stage, so
        # no viewer can see the result before the reveal grants it.
        if task.redundancy_n >= 2:
            try:
                self.compute_disagreement(task_id, sample_id, now=now, triggered_by="auto")
            except (ServiceError, DisagreementError) as exc:
                # The submission is valid and stays; the analysis failed. Recorded
                # rather than raised, because unwinding would discard a committed
                # revision over a downstream problem — but not swallowed either:
                # ``reveal`` refuses until the analysis exists, so the failure
                # cannot reach a signed record.
                self.chain.append(
                    "disagreement_failed",
                    {"task_id": task_id, "sample_id": sample_id, "reason": str(exc)[:500]},
                    timestamp=now,
                )

    # ── reveal ───────────────────────────────────────────────────────

    def reveal(self, task_id: str, sample_id: str, *, now: Optional[str] = None) -> Task:
        """Grant the reveal: AWAITING_REVEAL → IN_REVIEW, assignments revealed.

        The sample must already be in AWAITING_REVEAL — a caller cannot reveal
        a sample whose blind reads are short, because no edge exists for it and
        the machine will say so (P-07). Assignment reveal runs *after* the
        sample enters IN_REVIEW, which is exactly the cross-machine guard in
        ``advance_assignment``.
        """
        task = self._task(task_id)
        state = self._sample_state(task_id, sample_id)
        if state is not SampleState.AWAITING_REVEAL:
            raise ServiceError(
                f"sample {sample_id} is {state.value}; reveal is granted only from "
                "AWAITING_REVEAL (P-07: there is no edge for anything else)"
            )
        if task.redundancy_n >= 2:
            try:
                self.compute_disagreement(task_id, sample_id, now=now, triggered_by="reveal")
            except (ServiceError, DisagreementError) as exc:
                raise ServiceError(
                    f"cannot reveal sample {sample_id}: the disagreement analysis "
                    f"cannot be produced ({exc}). PRD 4.6 enters IN_REVIEW with "
                    "「分歧分析已出」as its precondition, so revealing without it would "
                    "put the sample in a state the table says it cannot be in."
                ) from exc
        # The flag goes down *before* the transition, because the "revealed"
        # guard reads it from the facts: setting it after the transition would
        # be writing the evidence for a decision already taken.
        self.store.set_sample_state(task_id, sample_id, state.value, reveal_done=True)
        self._apply(task, sample_id, "begin_review", actor_id="", actor_role=Role.SYSTEM, now=now)
        for assignment in self._assignments_for_sample(task_id, sample_id):
            if assignment.state is AssignmentState.SUBMITTED:
                self._replace_assignment(
                    assignment,
                    state=advance_assignment(
                        assignment.state, "reveal", task_state=SampleState.IN_REVIEW
                    ),
                )
        revealed = [
            a for a in self._assignments_for_sample(task_id, sample_id)
            if a.state is AssignmentState.REVEALED
        ]
        self.chain.append(
            "reveal_granted",
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "revealed_assignments": len(revealed),
            },
            timestamp=now,
        )
        return self._task(task_id)

    # ── disagreement analysis (S3) ───────────────────────────────────

    def compute_disagreement(
        self,
        task_id: str,
        sample_id: str,
        *,
        now: Optional[str] = None,
        force: bool = False,
        triggered_by: str = "manual",
    ) -> Optional[Disagreement]:
        """Compute and persist the disagreement record for one sample (P-02).

        Runs on the *blind* revisions only. A supplement is written after its
        author has seen the others' work, so including it would break the
        independence that α and the pairwise metrics are defined on — the numbers
        would look like agreement while measuring something else. A supplement
        therefore does not change the metric set; it is recorded, and the
        adjudication view shows it next to the blind read it derives from.

        Returns the existing record when the input set has not changed, so the
        automation can call this without either duplicating audit events or
        needing a dirty flag.
        """
        _ = self._task(task_id)
        state = self._sample_state(task_id, sample_id)
        if state in (
            SampleState.DRAFT,
            SampleState.PUBLISHED,
            SampleState.IN_BLIND_READ,
            SampleState.NEEDS_IMPROVEMENT,
        ):
            raise ServiceError(
                f"sample {sample_id} is {state.value}; disagreement analysis compares a "
                "complete set of independent reads, and a partial set would report "
                "agreement between however many people happened to have submitted"
            )
        blind = [r for r in self._revisions_for_sample(task_id, sample_id) if r.is_blind_read]
        if len(blind) < 2:
            raise ServiceError(
                f"sample {sample_id} has {len(blind)} blind revision(s); at least two are "
                "needed to measure disagreement"
            )
        payloads = {r.author_id: self.store.get_blob(r.payload_ref) for r in blind}
        try:
            result = analyse_disagreement(sample_id, payloads)
        except DisagreementError as exc:
            raise ServiceError(str(exc)) from exc

        previous = self._disagreements_for_sample(task_id, sample_id)
        for record in previous:
            if record.metric_snapshot.get("revision_set_hash") == result.revision_set_hash and not force:
                return record

        regions_ref = self.store.put_blob(result.regions, stored_at=now or _now())
        snapshot = {
            **dict(result.snapshot),
            "revision_set_hash": result.revision_set_hash,
            "triggered_by": triggered_by,
        }
        record = Disagreement(
            id=f"{task_id}:{sample_id}:dis{len(previous) + 1}",
            sample_id=sample_id,
            metric_snapshot=snapshot,
            regions_ref=regions_ref,
            controversy_score=result.controversy_score,
            computed_at=now or _now(),
        )
        self.store.put(record)
        self.chain.append(
            "disagreement_computed",
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "disagreement_id": record.id,
                "coders": sorted(payloads),
                "alpha": snapshot["alpha"]["alpha"],
                "controversy_score": result.controversy_score,
                "disputed_share": snapshot["regions"]["disputed_share"],
                "revision_set_hash": result.revision_set_hash,
                "triggered_by": triggered_by,
                "computed_at": record.computed_at,
            },
            timestamp=now,
        )
        return record

    # ── supplement revision (P-06) ───────────────────────────────────

    def supplement(
        self,
        task_id: str,
        sample_id: str,
        *,
        actor_id: str,
        payload: bytes,
        now: Optional[str] = None,
    ) -> Revision:
        """Add the one supplement a coder may file after the reveal (PRD 4.5).

        The supplement is a *new* revision whose ``derived_from`` names the blind
        read. Nothing overwrites the blind read: P-06 requires its content hash to
        survive the supplement, and the only way to guarantee that structurally is
        to never write to it again.
        """
        self._task(task_id)
        state = self._sample_state(task_id, sample_id)
        if state is not SampleState.IN_REVIEW and state is not SampleState.AWAITING_ADJUDICATION:
            raise ServiceError(
                f"sample {sample_id} is {state.value}; a supplement is filed after the "
                "reveal (PRD 4.5), because its whole purpose is to be informed by it"
            )
        assignment = self._assignment(task_id, sample_id, actor_id)
        if assignment.state is not AssignmentState.REVEALED:
            raise ServiceError(
                f"assignment {assignment.id} is {assignment.state.value}; only a coder "
                "whose blind read has been revealed may supplement it"
            )
        existing = self._revisions_for_assignment(assignment.id)
        blind = [r for r in existing if r.is_blind_read]
        if not blind:
            raise ServiceError(f"assignment {assignment.id} has no blind revision to supplement")
        if any(not r.is_blind_read for r in existing):
            raise ServiceError(
                f"assignment {assignment.id} already has a supplement; PRD 4.5 allows one "
                "per coder, and a second one would be an unrecorded second round"
            )
        stamp = now or _now()
        ref = self.store.put_blob(payload, stored_at=stamp)
        revision = Revision(
            id=f"{assignment.id}:r{len(existing) + 1}",
            sample_id=sample_id,
            assignment_id=assignment.id,
            author_id=actor_id,
            revision_no=len(existing) + 1,
            is_blind_read=False,
            payload_ref=ref,
            payload_hash=ref[len("blob:") :],
            created_at=stamp,
            derived_from=blind[0].id,
        )
        self.store.put(revision)
        self.chain.append(
            "supplement_submitted",
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "assignment_id": assignment.id,
                "revision_id": revision.id,
                "derived_from": blind[0].id,
                # Both hashes in one block: P-06 is a claim about the pair, and a
                # verifier that has to look up two rows to check it will not.
                "blind_payload_hash": blind[0].payload_hash,
                "supplement_payload_hash": revision.payload_hash,
            },
            timestamp=now,
        )
        return revision

    # ── projected read path (P-01) ───────────────────────────────────

    def view_task(
        self,
        task_id: str,
        *,
        viewer_role: Role,
        viewer_id: str,
        cosign_locked: bool = False,
    ) -> dict[str, Any]:
        """What one viewer may see, as data rather than as a filtered view.

        The projection is applied here, before the response exists: sections the
        viewer is not entitled to are absent from the dict, not blanked. This is
        the fix for the defect class behind all eleven MVP leaks — the view
        rendering a field it received, having nothing to render because it was
        never sent, is the only version that survives a new client.
        """
        task = self._task(task_id)
        protocol_payload = self.store.get("annotation_protocol", task.protocol_id)
        if protocol_payload is None:
            raise ServiceError(
                f"task {task_id} has no stored protocol {task.protocol_id!r}, so a viewer "
                "cannot be told which rules produced the labels. Create the task with "
                "protocol=make_protocol(...) — PRD 7.4 requires the rules to travel with "
                "the data."
            )
        protocol = AnnotationProtocol.from_dict(protocol_payload)
        states = {
            sample_id: SampleState(row["state"])
            for sample_id, row in self.store.sample_states(task_id).items()
        }
        revisions = [
            r for r in self._all_revisions() if r.sample_id in task.image_scope
        ]
        submitted: dict[str, int] = {}
        for revision in revisions:
            if revision.is_blind_read:
                submitted[revision.sample_id] = submitted.get(revision.sample_id, 0) + 1

        return project_task(
            task=task,
            protocol=protocol,
            samples=list(task.image_scope),
            state_by_sample=states,
            submitted_by_sample=submitted,
            revisions=revisions,
            assignments=self._assignments_for_task(task_id),
            disagreements=[
                d for s in task.image_scope for d in self._disagreements_for_sample(task_id, s)
            ],
            model_ranks=[],
            adjudications=[],
            cosigns=[],
            stratums=[
                label
                for label in (self.stratum(s) for s in task.image_scope)
                if label is not None
            ],
            context=ProjectionContext(
                viewer_role=viewer_role, viewer_id=viewer_id, cosign_locked=cosign_locked
            ),
        )

    # ── metric matrix (PRD 6.1) ──────────────────────────────────────

    def metric_matrix(self, task_id: str) -> dict[str, Any]:
        """Per-sample metrics, one row each, plus a task-level summary.

        A row is emitted for every sample, including the ones with no
        disagreement record yet (their metric cells are ``None`` rather than
        zero): a matrix that silently omits unfinished samples reads as "these
        passed".
        """
        task = self._task(task_id)
        states = self.store.sample_states(task_id)
        rows: list[dict[str, Any]] = []
        for sample_id in task.image_scope:
            label = self.stratum(sample_id)
            records = self._disagreements_for_sample(task_id, sample_id)
            latest = records[-1] if records else None
            snapshot = {} if latest is None else dict(latest.metric_snapshot)
            pairs = snapshot.get("pairs", {})
            rows.append(
                {
                    "sample_id": sample_id,
                    "state": states.get(sample_id, {}).get("state"),
                    "stratum": None if label is None else label.stratum.value,
                    "stratum_rule": None if label is None else label.rule_version,
                    "coders": snapshot.get("coders", []),
                    "alpha": (snapshot.get("alpha") or {}).get("alpha"),
                    "alpha_ci": [
                        (snapshot.get("alpha") or {}).get("ci_low"),
                        (snapshot.get("alpha") or {}).get("ci_high"),
                    ],
                    "raw_agreement": ((snapshot.get("agreement") or {}).get("raw")),
                    "dice_mean": _mean_of(pairs.values(), "dice_mean"),
                    "detection_f1": _mean_of(pairs.values(), "detection_f1"),
                    "hd95_mean": _mean_of(pairs.values(), "hd95_mean"),
                    "controversy_score": None if latest is None else latest.controversy_score,
                    "disputed_share": (snapshot.get("regions") or {}).get("disputed_share"),
                    "metrics_version": snapshot.get("metrics_version"),
                    "disagreement_id": None if latest is None else latest.id,
                }
            )
        return {
            "task_id": task_id,
            "template": task.template,
            "redundancy_n": task.redundancy_n,
            "rows": rows,
            "summary": {
                "samples": len(rows),
                "with_disagreement": sum(1 for r in rows if r["disagreement_id"]),
                "alpha_mean": _mean_of(
                    [{"v": r["alpha"]} for r in rows if r["alpha"] is not None], "v"
                ),
                "dice_mean": _mean_of(
                    [{"v": r["dice_mean"]} for r in rows if r["dice_mean"] is not None], "v"
                ),
                "controversy_mean": _mean_of(
                    [{"v": r["controversy_score"]} for r in rows if r["controversy_score"] is not None],
                    "v",
                ),
            },
        }

    # ── read back ────────────────────────────────────────────────────

    def read_back(self, task_id: str) -> dict[str, Any]:
        """Everything a verifier needs, in one call.

        The two verify passes are the read path's contribution to M1's exit
        criteria: a store whose rows were edited outside the contract layer, or
        an audit chain that was reordered, fails here instead of feeding
        corrupted data to a training set. Entity-hash verification runs *first*
        — a row that no longer hashes to its stored value must not be handed
        to ``from_dict`` at all, because a tampered payload is untrusted input,
        not a parse error.
        """
        entities = self.store.verify_entity_hashes()
        if entities["corrupt"]:
            raise ServiceError(
                "entity rows no longer hash to their stored content_hash — edited "
                f"outside the contract layer: {entities['corrupt']}. Refusing to "
                "deserialise untrusted rows."
            )
        task = self._task(task_id)
        revisions = [Revision.from_dict(p) for p in self._payloads("revision")]
        mine = [r for r in revisions if r.sample_id in task.image_scope]
        blob_checks = {}
        for revision in mine:
            payload = self.store.get_blob(revision.payload_ref)
            blob_checks[revision.id] = content_hash({"bytes": payload.hex()}) == revision.payload_hash
        return {
            "task": task,
            "sample_states": self.store.sample_states(task_id),
            "assignments": self._assignments_for_task(task_id),
            "revisions": mine,
            "disagreements": [
                d for s in task.image_scope for d in self._disagreements_for_sample(task_id, s)
            ],
            "strata": [
                label
                for label in (self.stratum(s) for s in task.image_scope)
                if label is not None
            ],
            "payloads_intact": blob_checks,
            "audit": self.chain.verify(),
            "entities": entities,
        }

    def replay_sample_states(self, task_id: str) -> dict[str, str]:
        """Rebuild per-sample states from the audit chain alone.

        The chain is the system of record; the ``sample_states`` table is a
        cache of it. If the two disagree after a restart, the cache lost —
        that is what this comparison is for. Operations recognised here are
        exactly the ones this service writes.
        """
        task = self._task(task_id)
        states = {sample: SampleState.DRAFT for sample in task.image_scope}
        for block in self.chain.blocks():
            details = block["details"]
            op = block["operation"]
            if op == "task_created" and details.get("task_id") == task_id:
                continue
            if op == "task_published" and details.get("task_id") == task_id:
                for sample in task.image_scope:
                    states[sample] = SampleState.PUBLISHED
            elif op == "blind_submitted" and details.get("task_id") == task_id:
                sample = details["sample_id"]
                if states[sample] is SampleState.PUBLISHED:
                    states[sample] = SampleState.IN_BLIND_READ
                if details["blind_reads_submitted"] >= task.redundancy_n:
                    states[sample] = SampleState.AWAITING_REVEAL
            elif op == "reveal_granted" and details.get("task_id") == task_id:
                states[details["sample_id"]] = SampleState.IN_REVIEW
        return {sample: state.value for sample, state in states.items()}

    # ── internals ────────────────────────────────────────────────────

    def _task(self, task_id: str) -> Task:
        payload = self.store.get("task", task_id)
        if payload is None:
            raise ServiceError(f"no such task: {task_id!r}")
        return Task.from_dict(payload)

    def _payloads(self, kind: str) -> list[dict[str, Any]]:
        return [json.loads(row["payload"]) for row in self.store.iter_rows(kind)]

    def _assignment(self, task_id: str, sample_id: str, actor_id: str) -> Assignment:
        payload = self.store.get("assignment", f"{task_id}:{sample_id}:{actor_id}")
        if payload is None:
            raise ServiceError(
                f"{actor_id!r} holds no assignment on sample {sample_id!r} of task "
                f"{task_id!r}. Acting without an assignment is the role-dropdown "
                "defect (G-2); it fails here rather than at the view."
            )
        return Assignment.from_dict(payload)

    def _assignments_for_sample(self, task_id: str, sample_id: str) -> list[Assignment]:
        prefix = f"{task_id}:{sample_id}:"
        found = [
            Assignment.from_dict(p)
            for p in self._payloads("assignment")
            if p["id"].startswith(prefix)
        ]
        return sorted(found, key=lambda a: a.id)

    def _assignments_for_task(self, task_id: str) -> list[Assignment]:
        found = [
            Assignment.from_dict(p)
            for p in self._payloads("assignment")
            if p["id"].startswith(f"{task_id}:")
        ]
        return sorted(found, key=lambda a: a.id)

    def _revisions_for_assignment(self, assignment_id: str) -> list[Revision]:
        return [
            r
            for r in self._all_revisions()
            if r.assignment_id == assignment_id
        ]

    def _revisions_for_sample(self, task_id: str, sample_id: str) -> list[Revision]:
        return [
            r for r in self._all_revisions() if r.sample_id == sample_id
        ]

    def _all_revisions(self) -> list[Revision]:
        return sorted(
            (Revision.from_dict(p) for p in self._payloads("revision")),
            key=lambda r: (r.sample_id, r.assignment_id, r.revision_no),
        )

    def _disagreements_for_sample(self, _task_id: str, sample_id: str) -> list[Disagreement]:
        return sorted(
            (
                Disagreement.from_dict(p)
                for p in self._payloads("disagreement")
                if p.get("sample_id") == sample_id
            ),
            key=lambda d: (d.computed_at, d.id),
        )

    def _replace_assignment(
        self, assignment: Assignment, *, state: AssignmentState, submitted_at: Optional[str] = None
    ) -> Assignment:
        updated = Assignment(
            id=assignment.id,
            task_id=assignment.task_id,
            sample_id=assignment.sample_id,
            assignee_id=assignment.assignee_id,
            role=assignment.role,
            state=state,
            started_at=assignment.started_at,
            submitted_at=submitted_at or assignment.submitted_at,
        )
        self.store.put(updated, overwrite=True)
        return updated

    def _sample_state(self, task_id: str, sample_id: str) -> SampleState:
        states = self.store.sample_states(task_id)
        if sample_id not in states:
            raise ServiceError(f"no state row for sample {sample_id!r} of task {task_id!r}")
        return SampleState(states[sample_id]["state"])

    def _facts(self, task: Task, sample_id: str, *, submitted: Optional[int] = None) -> TransitionFacts:
        states = self.store.sample_states(task.id)
        row = states.get(sample_id, {})
        count = submitted
        if count is None:
            count = len(
                [
                    a
                    for a in self._assignments_for_sample(task.id, sample_id)
                    if a.state is AssignmentState.SUBMITTED
                ]
            )
        return TransitionFacts(
            template=get_template(task.template),
            redundancy_n=task.redundancy_n,
            blind_reads_submitted=count,
            reveal_done=bool(row.get("reveal_done")),
        )

    def _apply(
        self,
        task: Task,
        sample_id: str,
        operation: str,
        *,
        actor_id: str,
        actor_role: Role,
        now: Optional[str] = None,
    ):
        """Run one machine transition for one sample and persist everything it
        changes: the per-sample state, the aggregate Task, and one audit block
        per *event* (the transition and its admission are separate blocks —
        the machine's own audit dict is folded into the transition's block)."""
        result = self._machine.apply(
            self._sample_state(task.id, sample_id),
            operation,
            self._facts(task, sample_id),
            actor_role=actor_role,
            actor_id=actor_id,
        )
        reveal_done = self.store.sample_states(task.id).get(sample_id, {}).get("reveal_done", False)
        self.store.set_sample_state(task.id, sample_id, result.state.value, reveal_done=reveal_done)
        self._refresh_aggregate(task)
        if actor_role is not Role.SYSTEM:
            self.chain.append(
                f"transition_{operation}",
                {
                    "task_id": task.id,
                    "sample_id": sample_id,
                    "actor_id": actor_id,
                    "from": result.audit["from"],
                    "to": result.audit["to"],
                    "guard": result.audit["guard"],
                    "guard_evidence_hash": result.audit["guard_evidence_hash"],
                },
                timestamp=now,
            )
        return result

    def _apply_per_sample(self, task: Task, operation: str, *, actor_id: str, actor_role: Role):
        """One transition applied to every sample in scope (publish, abort)."""
        result = None
        for sample_id in task.image_scope:
            result = self._apply(task, sample_id, operation, actor_id=actor_id, actor_role=actor_role)
        return result

    def _refresh_aggregate(self, task: Task) -> Task:
        states = self.store.sample_states(task.id)
        aggregate = aggregate_task_state(SampleState(row["state"]) for row in states.values())
        if aggregate is task.state:
            return task
        updated = Task(
            id=task.id,
            node_id=task.node_id,
            project_id=task.project_id,
            template=task.template,
            image_scope=task.image_scope,
            redundancy_n=task.redundancy_n,
            preannotation_policy=task.preannotation_policy,
            protocol_id=task.protocol_id,
            state=aggregate,
            deadline=task.deadline,
        )
        self.store.put(updated, overwrite=True)
        return updated


def _mean_of(records: Iterable[Mapping[str, Any]], key: str) -> Optional[float]:
    """Mean of a metric column, skipping samples that have no value yet.

    Skipping rather than counting ``None`` as zero: an average that includes
    uncomputed samples reads as a bad score, and the fix somebody applies to a
    bad score is to look for a modelling problem that does not exist.
    """
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [  # the re-exports let a caller catch machine and storage errors
    # without importing two modules
    "AnnotationService",
    "ContractValidationError",
    "IllegalTransition",
    "ServiceError",
    "StoreError",
]
