# ── analysis/contracts/hashing.py ──
"""Content hashes for contract payloads, on the audit specification's rules.

Why not just reuse :func:`analysis.audit_spec.canonical_hash`
-------------------------------------------------------------
That function hashes a *chain block*: six fixed fields including ``prev_hash``,
``index`` and ``nonce``. A ``Revision`` payload is not a block — it has none of
those fields, and shoehorning it into the block shape would make a revision hash
depend on where the revision happened to sit in a chain.

What it does reuse is the part that makes a hash portable: the canonical JSON
encoding (sorted keys, compact separators, raw UTF-8, floats restricted to the
domain where Python, Rust and JavaScript agree on decimal notation). That
encoding lives in :mod:`analysis.audit_spec` and is imported rather than copied,
because a second copy is a second thing to get wrong — the original Python/Rust
divergence documented in that module came from exactly that.

Why a domain separator
----------------------
The preimage is ``b"organoid-content-v1\\n" + canonical_json``. Without the
prefix, a revision payload and an audit block that happened to serialise to the
same bytes would hash identically, and a ``payload_hash`` could be validated
against a block. The newline matters: without it, the prefix and a payload whose
first byte is ``{`` are still unambiguous, but with it the preimage is prefix-,
payload-separated for any future prefix that is itself a JSON string.

This is *not* a security boundary. It is an integrity check that lets P-06
("盲读版本在补充修订后内容哈希不变") and the export gate's snapshot hash be
computed identically in all three languages.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from analysis.audit_spec import SpecError, validate_details

__all__ = [
    "CONTENT_HASH_PREFIX",
    "HASH_PATTERN",
    "canonical_json",
    "content_hash",
    "is_content_hash",
    "verify_content_hash",
]

#: Version tag for the content-hash preimage. Bump only together with the
#: exchange format version, so a stored ``payload_hash`` always identifies the
#: rule that produced it.
CONTENT_HASH_PREFIX = b"organoid-content-v1\n"

#: A lowercase hex SHA-256.
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(payload: Any) -> str:
    """Serialise ``payload`` the one way all three languages agree on.

    Raises:
        SpecError: if the payload contains a type or a float outside the
            canonical domain. The message names the offending path, because a
            bare "cannot hash this" sends the caller hunting through a payload
            that may be thousands of instances deep.
    """
    try:
        validate_details(payload)
    except SpecError as exc:
        raise SpecError(
            f"payload cannot be hashed portably: {exc}. Store probabilities as "
            "basis points (0.95 -> 9500) and lengths as integer micrometres, or "
            "keep the exact text in a string field."
        ) from exc
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(payload: Any) -> str:
    """Lowercase hex SHA-256 of the canonical encoding of ``payload``.

    ``payload`` must already be JSON-shaped — enums converted to their values,
    tuples to lists, datetimes to ISO-8601 strings. :meth:`ContractEntity.to_dict`
    does that conversion, so hash ``entity.to_dict()`` rather than the entity.
    """
    return hashlib.sha256(CONTENT_HASH_PREFIX + canonical_json(payload).encode("utf-8")).hexdigest()


def is_content_hash(value: object) -> bool:
    """True when ``value`` looks like a hash this module produced."""
    return isinstance(value, str) and HASH_PATTERN.match(value) is not None


def verify_content_hash(payload: Any, expected: str) -> bool:
    """Constant-shape comparison against ``expected``.

    Uses :func:`hmac.compare_digest` semantics indirectly by comparing the
    recomputed digest to ``expected`` with ``==`` on equal-length hex strings.
    Timing is not a concern here: a ``payload_hash`` is an integrity check
    against corruption, and an attacker who can rewrite the payload can rewrite
    the hash field too. The signature is what stops that, and signatures live in
    the audit chain, not here.
    """
    return is_content_hash(expected) and content_hash(payload) == expected
