# ── analysis/pipeline/service.py ──
"""The M1 write path: create task → publish → blind submit → reveal → read back.

This is the vertical slice PRD 10.4 defines for M1 ("写路径：创建任务 → 盲读提交
→ 落库 → 写审计 → 读回"), assembled from the M0 contracts instead of beside
them:

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
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from analysis.contracts.entities import (
    Assignment,
    ContractValidationError,
    Revision,
    Task,
)
from analysis.contracts.enums import AssignmentState, Role, SampleState
from analysis.contracts.hashing import content_hash
from analysis.contracts.state_machine import (
    IllegalTransition,
    TransitionFacts,
    aggregate_task_state,
    advance_assignment,
    sample_machine,
)
from analysis.contracts.templates import get_template
from analysis.pipeline.chain import PersistentAuditChain
from analysis.pipeline.store import StoreError, SQLiteStore


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
        preannotation_policy: str = "reveal_after_blind",
        redundancy_n: Optional[int] = None,
        deadline: Optional[str] = None,
        now: Optional[str] = None,
    ) -> Task:
        """Create a Task in DRAFT plus one DRAFT state row per sample.

        ``redundancy_n`` defaults to the template's default (3 for research,
        1 for teaching); D-07 lets a project lower it to the template's range
        floor, which the Task constructor validates.
        """
        spec = get_template(template)
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
        import json

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
        return [r for r in map(Revision.from_dict, self._payloads("revision")) if r.assignment_id == assignment_id]

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


def _now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [  # the re-exports let a caller catch machine and storage errors
    # without importing two modules
    "AnnotationService",
    "ContractValidationError",
    "IllegalTransition",
    "ServiceError",
    "StoreError",
]
