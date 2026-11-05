# ── analysis/contracts/exchange.py ──
"""The exchange envelope: one task's worth of records, and its integrity.

Why an envelope rather than a bag of records
--------------------------------------------
Three consumers need the same bytes: the canvas hands its output to the platform,
the platform hands a signed sample set to the exporter, and a node hands a
snapshot to a peer. If each defines "the records" separately, the third one will
disagree with the first two about whether ``Cosign`` is included, and the
disagreement will look like a data-loss bug rather than a schema bug.

So the envelope is closed: a fixed top-level set of fields, an explicit format
version, a content hash over everything except itself, and a strict reader.

Why the reader is strict
------------------------
``from_dict`` rejects unknown keys. During a format change the producing side
usually ships first; a tolerant reader then silently drops the new field and the
new information is lost *without an error*, which is distinguishable from "the
producer never sent it" only by reading both codebases. Failing loudly costs one
deployment; the alternative costs a migration nobody knows is needed.

The export gate is here, not in the exporter
--------------------------------------------
P-11 says ``template=teaching`` products must not reach a training set, and D-10
says it must be enforced by code rather than by convention. The check therefore
lives next to the data it inspects (:func:`check_training_export`) and refuses on
*two* independent facts:

1. the envelope's template, and
2. whether any revision traces back to an ``ANNOTATOR`` assignment.

The second exists because the first is a field somebody can set. A teaching run
mislabelled as research — a copy-paste in an import script is enough — would pass
check 1 and fail check 2, because the assignments still say who did the work.
Neither check alone is sufficient; together they close both the honest and the
accidental path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Mapping, Optional, Sequence

from analysis.contracts.entities import (
    Adjudication,
    AnnotationProtocol,
    Assignment,
    ContractEntity,
    ContractValidationError,
    Cosign,
    Disagreement,
    ModelRank,
    Revision,
    StratumLabel,
    Task,
    validate_cosign_independence,
    validate_model_rank,
)
from analysis.contracts.enums import Role
from analysis.contracts.hashing import canonical_json, content_hash, verify_content_hash
from analysis.contracts.templates import get_template, normalise_template_id

__all__ = [
    "FORMAT_VERSION",
    "ExchangeEnvelope",
    "ExportRefused",
    "check_training_export",
    "load_envelope",
    "summarise",
]

#: Bumped on any change to the envelope's shape or to the hash preimage.
FORMAT_VERSION = "organoid-exchange-v1"

#: Collections that carry ``sample_id`` and must therefore be checked against
#: ``samples``. Keeping the tuple in one place is what stops a new collection
#: from escaping the membership check.
_SAMPLE_SCOPED: tuple[str, ...] = (
    "assignments",
    "revisions",
    "disagreements",
    "model_ranks",
    "adjudications",
    "cosigns",
    "stratums",
)

_ENTITY_TYPES: Mapping[str, type[ContractEntity]] = {
    "assignments": Assignment,
    "revisions": Revision,
    "disagreements": Disagreement,
    "model_ranks": ModelRank,
    "adjudications": Adjudication,
    "cosigns": Cosign,
    "stratums": StratumLabel,
}


class ExportRefused(RuntimeError):
    """A training-set export was refused by the gate (P-11, D-10)."""


def _unique_ids(name: str, records: Sequence[ContractEntity]) -> None:
    ids = [r.id for r in records if hasattr(r, "id")]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ContractValidationError(f"{name} contains duplicate id(s) {duplicates}")


@dataclass(frozen=True)
class ExchangeEnvelope:
    """A complete, self-describing snapshot of one task.

    Every record type of PRD 7.2 appears exactly once as a collection. An empty
    collection is meaningful — a sample that has not yet been adjudicated has no
    adjudications — so the reader requires the key to be present rather than
    inferring absence from omission.
    """

    FORMAT: ClassVar[str] = FORMAT_VERSION

    format_version: str
    node_id: str
    template: str
    exported_at: str
    protocol: AnnotationProtocol
    task: Task
    samples: tuple[str, ...]
    assignments: tuple[Assignment, ...]
    revisions: tuple[Revision, ...]
    disagreements: tuple[Disagreement, ...]
    model_ranks: tuple[ModelRank, ...]
    adjudications: tuple[Adjudication, ...]
    cosigns: tuple[Cosign, ...]
    stratums: tuple[StratumLabel, ...]
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ContractValidationError(
                f"envelope format {self.format_version!r} is not {FORMAT_VERSION!r}. This "
                "reader refuses unknown versions rather than guessing at field meanings."
            )
        if not self.node_id.strip():
            raise ContractValidationError("node_id must be non-empty")
        object.__setattr__(self, "template", normalise_template_id(self.template))
        if not self.samples:
            raise ContractValidationError("an envelope with no samples carries no data")
        if self.task.template != self.template:
            raise ContractValidationError(
                f"envelope template {self.template!r} disagrees with task template "
                f"{self.task.template!r}; the gate reads the envelope field, so a "
                "mismatch here is a bypass waiting to happen"
            )
        if set(self.task.image_scope) != set(self.samples):
            raise ContractValidationError(
                "samples must equal task.image_scope: an envelope may hold records only "
                f"for the task's own samples (task has {sorted(self.task.image_scope)}, "
                f"envelope has {sorted(self.samples)})"
            )

        for name in _SAMPLE_SCOPED:
            records = getattr(self, name)
            _unique_ids(name, records)
            for record in records:
                if record.sample_id not in set(self.samples):
                    raise ContractValidationError(
                        f"{name}: record {getattr(record, 'id', '?')} names sample "
                        f"{record.sample_id!r}, which is outside this envelope. Records "
                        "from another sample are how cross-sample carry-over gets into a "
                        "dataset (MVP defect L2-1)."
                    )

        stratum_samples = [s.sample_id for s in self.stratums]
        if len(set(stratum_samples)) != len(stratum_samples):
            raise ContractValidationError(
                "a sample has more than one StratumLabel. D-08 assigns the stratum once, "
                "at ingestion; two labels for one sample means two rules ran, and the "
                "held-out sampling weights become ambiguous."
            )

        computed = self._compute_hash()
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ContractValidationError(
                "content_hash does not match the envelope contents; the bytes were "
                "modified after the hash was written"
            )

    # ── hashing ──────────────────────────────────────────────────────

    def payload(self) -> dict[str, Any]:
        """The hashed view: everything except ``content_hash`` itself."""
        return {
            "format_version": self.format_version,
            "node_id": self.node_id,
            "template": self.template,
            "exported_at": self.exported_at,
            "protocol": self.protocol.to_dict(),
            "task": self.task.to_dict(),
            "samples": list(self.samples),
            **{name: [r.to_dict() for r in getattr(self, name)] for name in _ENTITY_TYPES},
        }

    def _compute_hash(self) -> str:
        return content_hash(self.payload())

    # ── serialisation ────────────────────────────────────────────────

    def evolve(self, **changes: Any) -> ExchangeEnvelope:
        """Return a copy with ``changes`` applied and the hash recomputed.

        ``dataclasses.replace`` cannot be used directly: it carries the old
        ``content_hash`` over and ``__post_init__`` then (correctly) rejects the
        result. Producers that append a record — a supplement, a cosign verdict,
        a stratum label — need exactly this behaviour, so it is a method rather
        than a caveat in the docstring.
        """
        payload = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "content_hash"}
        payload.update(changes)
        return ExchangeEnvelope(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "content_hash": self.content_hash}

    def dump(self) -> bytes:
        """Canonical bytes, safe to write to disk or send to a peer."""
        return canonical_json(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExchangeEnvelope:
        if not isinstance(payload, Mapping):
            raise ContractValidationError(
                f"envelope must be a JSON object, got {type(payload).__name__}"
            )
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ContractValidationError(
                f"envelope has unknown field(s) {sorted(unknown)}; this reader is for "
                f"{FORMAT_VERSION} and will not drop a field it does not understand"
            )
        missing = sorted(known - set(payload) - {"content_hash"})
        if missing:
            raise ContractValidationError(f"envelope is missing field(s) {missing}")

        def records(name: str):
            items = payload[name]
            if not isinstance(items, (list, tuple)):
                raise ContractValidationError(f"{name} must be a JSON array")
            model = _ENTITY_TYPES[name]
            return tuple(model.from_dict(item) for item in items)

        return cls(
            format_version=str(payload["format_version"]),
            node_id=str(payload["node_id"]),
            template=str(payload["template"]),
            exported_at=str(payload["exported_at"]),
            protocol=AnnotationProtocol.from_dict(payload["protocol"]),
            task=Task.from_dict(payload["task"]),
            samples=tuple(str(s) for s in payload["samples"]),
            content_hash=str(payload.get("content_hash", "")),
            **{name: records(name) for name in _ENTITY_TYPES},
        )

    # ── cross-entity verification ───────────────────────────────────

    def verify(self) -> tuple[str, ...]:
        """Every cross-entity invariant that a single record cannot check.

        Returns problems instead of raising so an importer can report the full
        set: a malformed envelope usually has one root cause and ten symptoms,
        and fixing them one exception at a time is how an afternoon disappears.
        """
        problems: list[str] = []

        if not verify_content_hash(self.payload(), self.content_hash):
            problems.append("content_hash mismatch: envelope bytes do not match its hash")

        revision_ids = {r.id for r in self.revisions}
        assignment_by_id = {a.id: a for a in self.assignments}

        for revision in self.revisions:
            assignment = assignment_by_id.get(revision.assignment_id)
            if assignment is None:
                problems.append(
                    f"revision {revision.id} names unknown assignment {revision.assignment_id}"
                )
            elif assignment.sample_id != revision.sample_id:
                problems.append(
                    f"revision {revision.id} is on sample {revision.sample_id} but its "
                    f"assignment is on {assignment.sample_id}"
                )
            elif assignment.assignee_id != revision.author_id:
                problems.append(
                    f"revision {revision.id} author {revision.author_id} != assignment "
                    f"assignee {assignment.assignee_id}"
                )
            if revision.derived_from is not None and revision.derived_from not in revision_ids:
                problems.append(
                    f"revision {revision.id} derives from unknown revision {revision.derived_from}"
                )

        by_sample: dict[str, list[Revision]] = {}
        for revision in self.revisions:
            by_sample.setdefault(revision.sample_id, []).append(revision)
        for rank in self.model_ranks:
            try:
                validate_model_rank(rank, tuple(by_sample.get(rank.sample_id, ())))
            except ContractValidationError as exc:
                problems.append(str(exc))

        for adjudication in self.adjudications:
            if adjudication.selected_revision_id not in revision_ids:
                problems.append(
                    f"adjudication {adjudication.id} selects unknown revision "
                    f"{adjudication.selected_revision_id}"
                )
            if adjudication.sample_id not in set(self.samples):
                problems.append(f"adjudication {adjudication.id} is outside the sample scope")

        adjudication_by_id = {a.id: a for a in self.adjudications}
        for cosign in self.cosigns:
            adjudication = adjudication_by_id.get(cosign.adjudication_id)
            if adjudication is None:
                problems.append(
                    f"cosign {cosign.id} names unknown adjudication {cosign.adjudication_id}"
                )
                continue
            try:
                validate_cosign_independence(adjudication, cosign)
            except ContractValidationError as exc:
                problems.append(str(exc))

        return tuple(problems)


def load_envelope(data: bytes | str) -> ExchangeEnvelope:
    """Parse and verify canonical bytes produced by :meth:`ExchangeEnvelope.dump`."""
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"envelope is not valid JSON: {exc}") from exc
    envelope = ExchangeEnvelope.from_dict(payload)
    problems = envelope.verify()
    if problems:
        raise ContractValidationError(
            "envelope failed verification:\n  - " + "\n  - ".join(problems)
        )
    return envelope


def check_training_export(envelope: ExchangeEnvelope) -> None:
    """Refuse a training-set export that D-10 forbids (P-11).

    Raises:
        ExportRefused: naming which of the two independent checks fired.
    """
    spec = get_template(envelope.template)
    if not spec.training_export_allowed:
        raise ExportRefused(
            f"template {envelope.template!r} may not export to a training set "
            "(D-10/P-11). Student annotations are used for teaching feedback and "
            "pipeline validation only; their distribution bias would be learned "
            "directly by the next pre-annotation model."
        )

    annotator_assignments = {a.id for a in envelope.assignments if a.role is Role.ANNOTATOR}
    tainted = sorted(r.id for r in envelope.revisions if r.assignment_id in annotator_assignments)
    if tainted:
        raise ExportRefused(
            f"envelope claims template {envelope.template!r} but {len(tainted)} "
            f"revision(s) trace to ANNOTATOR assignments ({tainted[:5]}). A teaching run "
            "mislabelled as research is exactly what the second check exists to catch; "
            "the template field alone is not evidence of who did the work."
        )


def summarise(envelope: ExchangeEnvelope) -> Mapping[str, Optional[int]]:
    """Counts per collection, for logs and for the export manifest preview."""
    return {
        "samples": len(envelope.samples),
        "assignments": len(envelope.assignments),
        "revisions": len(envelope.revisions),
        "disagreements": len(envelope.disagreements),
        "model_ranks": len(envelope.model_ranks),
        "adjudications": len(envelope.adjudications),
        "cosigns": len(envelope.cosigns),
        "stratums": len(envelope.stratums),
    }
