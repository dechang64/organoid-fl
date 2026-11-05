# ── analysis/contracts/__init__.py ──
"""The frozen contract layer for the HITL annotation pipeline (M0).

What "frozen" means here
------------------------
Not "finished" — frozen in the sense PRD 10.4 uses: the shapes may be extended
additively, but changing one of them is a platform-wide semantic change that
requires a human decision (10.6: "数据模型与协议语义 — 否（只准复用）"). The point
of putting them in one package with one version string is that the blast radius
of such a change is visible: everything imports from here, so a rename shows up
in one diff instead of in eleven modules that each had their own copy of "the
shape of a revision".

What is in the package
----------------------
======================  ====================================================
module                  responsibility
======================  ====================================================
``enums``               closed vocabularies (states, roles, strata, templates)
``entities``            the nine records of PRD 7.2, immutable and strict
``protocol``            protocol construction, taxonomy binding, rule hash
``templates``           research / teaching as configuration only
``state_machine``       the 4.8 transition table, guards, role casting
``exchange``            the envelope, its integrity, and the P-11 export gate
``projection``          role+stage projection — the fix for the 11 defects
``hashing``             canonical JSON and content hashes
======================  ====================================================

Dependency order is a straight line — ``hashing`` → ``enums`` → ``entities`` →
``{protocol, templates}`` → ``state_machine`` → ``exchange`` → ``projection`` —
with no cycles and no third-party imports. That matters for two reasons: the
future REST service can import this package without torch, and the test suite can
exercise the contract without the pandas-before-torch dance that
``utils/import_order.py`` exists to enforce.

How this relates to the MVP
---------------------------
``mvp/hitl-demo/`` implements the same rules in JavaScript so the client can run
standalone. It is the *demonstration* of the requirements; this package is the
requirement. Where the two disagree, this one is right, and the disagreement is a
bug in the demo. G-3 says the demo's projection functions are a reference
implementation, not a security boundary — the same sentence applies to everything
in that directory.
"""

from __future__ import annotations

from analysis.contracts import (
    entities,
    enums,
    exchange,
    hashing,
    projection,
    protocol,
    state_machine,
    templates,
)

__all__ = [
    "CONTRACT_VERSION",
    "entities",
    "enums",
    "exchange",
    "hashing",
    "projection",
    "protocol",
    "state_machine",
    "templates",
]

#: Version of the *shape* contract, independent of the exchange format version.
#:
#: Bump when an entity gains/loses a field, when a state is added, or when a
#: projection rule changes. Persisted nowhere directly — the exchange envelope
#: carries ``organoid-exchange-v1`` instead — but reported by
#: ``contract_report()`` so a running service can be asked which contract it was
#: built against.
CONTRACT_VERSION = "organoid-contracts-v1"


def contract_report() -> dict:
    """Machine-readable summary of what this contract layer declares.

    Used by the M0 exit review and by the future service's ``/health`` endpoint:
    "which states exist, how many transitions are legal, which templates are
    registered, which entity types ship" answered from code rather than from a
    document that may be one revision behind.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "entities": sorted(
            cls.ENTITY
            for cls in (
                entities.AnnotationProtocol,
                entities.Task,
                entities.Assignment,
                entities.Revision,
                entities.Disagreement,
                entities.ModelRank,
                entities.Adjudication,
                entities.Cosign,
                entities.StratumLabel,
            )
        ),
        "sample_states": [s.value for s in enums.SampleState],
        "assignment_states": [s.value for s in enums.AssignmentState],
        "roles": [r.value for r in enums.Role],
        "templates": sorted(templates.TEMPLATES),
        "sample_machine": state_machine.state_machine_report(),
        "assignment_machine": state_machine.assignment_machine_report(),
        "exchange_format": exchange.FORMAT_VERSION,
    }
