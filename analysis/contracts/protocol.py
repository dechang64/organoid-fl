# ── analysis/contracts/protocol.py ──
"""Annotation-protocol construction and verification (PRD 7.2, C-05).

Why a factory instead of "just fill in the fields"
--------------------------------------------------
``AnnotationProtocol`` refuses to exist with a hash that does not match its
rules, which means every construction site has to compute the hash the same way.
A factory makes that the only easy path, and — more importantly — makes the
*taxonomy check* happen at construction rather than at first use.

The taxonomy check is the part that matters. ``taxonomy_name`` is a string, and
the platform already has a registry that is the single source of truth for label
identity (``utils/taxonomy.py``). A protocol that names a taxonomy nobody
registered is a protocol whose labels cannot be resolved at export time — which
is discovered days later, in a dataset someone is trying to train on. Checking
here turns that into a startup error.

Deliberately *not* checked
--------------------------
The rule strings themselves. ``touching_object_rule`` and
``overlap_split_rule`` are open vocabulary (see ``contracts/enums.py``): the
platform hashes them, so it can tell that they changed, but it does not claim to
know whether "split at the narrowest neck" is better than "split by watershed".
That judgement belongs to the methodology owner and has to be able to change
without a code release.
"""

from __future__ import annotations

from typing import Any, Mapping

from analysis.contracts.entities import AnnotationProtocol, ContractValidationError
from utils.taxonomy import list_taxonomies, maybe_get_taxonomy

__all__ = [
    "ProtocolProblem",
    "assert_taxonomy_registered",
    "make_protocol",
    "protocol_diff",
    "verify_protocol",
]


class ProtocolProblem(ContractValidationError):
    """A protocol cannot be built or no longer verifies."""


def assert_taxonomy_registered(name: str) -> None:
    """Raise :class:`ProtocolProblem` unless ``name`` is a registered taxonomy."""
    if maybe_get_taxonomy(name) is not None:
        return
    available = list(list_taxonomies())
    raise ProtocolProblem(
        f"taxonomy_name {name!r} is not registered. Available: {available}. A protocol "
        "must name a registered taxonomy, otherwise the labels it produces cannot be "
        "resolved at export time and the failure surfaces only when a dataset is "
        "being trained on — which is the worst possible moment."
    )


def make_protocol(
    *,
    id: str,
    version: str,
    taxonomy_name: str,
    touching_object_rule: str,
    min_visible_boundary_um: float,
    overlap_split_rule: str,
    effective_from: str,
) -> AnnotationProtocol:
    """Build a protocol, computing its rule hash.

    Raises:
        ProtocolProblem: if the taxonomy is unregistered or the rules are
            malformed (empty strings, negative boundary).
    """
    assert_taxonomy_registered(taxonomy_name)
    try:
        return AnnotationProtocol(
            id=id,
            version=version,
            taxonomy_name=taxonomy_name,
            touching_object_rule=touching_object_rule,
            min_visible_boundary_um=min_visible_boundary_um,
            overlap_split_rule=overlap_split_rule,
            effective_from=effective_from,
        )
    except ContractValidationError as exc:
        raise ProtocolProblem(str(exc)) from exc


def verify_protocol(protocol: AnnotationProtocol) -> tuple[str, ...]:
    """Re-check a protocol that was loaded from storage.

    Returns the problems found rather than raising, so an import can report all
    of them at once instead of failing on the first sample in a 5000-sample
    envelope.

    Two things can rot between write and read: the rule hash (someone edited the
    record) and the taxonomy registration (a registry was rebuilt without the
    older taxonomy). The second is not the importer's fault and must not be
    reported as tampering, hence a separate message.
    """
    problems: list[str] = []
    if protocol.rule_hash() != protocol.hash:
        problems.append(
            f"protocol {protocol.id!r} rule hash mismatch: stored {protocol.hash[:16]}…, "
            f"recomputed {protocol.rule_hash()[:16]}…"
        )
    try:
        assert_taxonomy_registered(protocol.taxonomy_name)
    except ProtocolProblem as exc:
        problems.append(str(exc))
    return tuple(problems)


def protocol_diff(before: AnnotationProtocol, after: AnnotationProtocol) -> Mapping[str, Any]:
    """Rule fields that changed between two protocols.

    Used by the UI to answer "why is this v3 different from v2" without a schema
    diff, and by the M1 migration to decide whether already-annotated samples
    must be re-read under the new rules. An empty result with a bumped version is
    a legitimate outcome (a relabelled release), which is exactly what
    ``AnnotationProtocol.rule_hash`` is designed to make visible.
    """
    keys = (
        "taxonomy_name",
        "touching_object_rule",
        "min_visible_boundary_um",
        "overlap_split_rule",
    )
    changes: dict[str, Any] = {}
    for key in keys:
        old = getattr(before, key)
        new = getattr(after, key)
        if old != new:
            changes[key] = {"from": old, "to": new}
    return changes
