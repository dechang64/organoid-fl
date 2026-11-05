# ── analysis/audit_spec.py ──
"""Canonical audit-chain hash specification — shared by Python and Rust.

Why this module exists
----------------------
``analysis/audit_engine.py`` and ``src/blockchain.rs`` both claim to build a
tamper-evident SHA-256 audit chain, but they hashed different things:

    Python   sha256(json.dumps({index, timestamp, operation, details,
                                prev_hash}, sort_keys=True))

    Rust     sha256(format!("{}:{}:{:?}:{}:{}", index, timestamp,
                            operation, prev_hash, nonce))

These can never agree. The Python preimage is JSON with lexicographically
sorted keys and no ``nonce``; the Rust preimage is a colon-joined string whose
operation field is a Rust ``Debug`` rendering such as ``Insert { count: 10 }``.
A chain produced on one side therefore could not be verified on the other,
which defeats the point of an audit chain in a system that runs both.

The specification
-----------------
Version ``organoid-audit-v1``. The preimage is UTF-8 encoded compact JSON with
object keys sorted lexicographically and no insignificant whitespace:

    {"details":<obj>,"index":<u64>,"nonce":<u64>,"operation":"<str>",
     "prev_hash":"<str>","timestamp":"<str>"}

The digest is the lowercase hex SHA-256 of exactly those bytes.

Worked example (also a fixed test vector in
``tests/fixtures/audit_hash_vectors.json``)::

    index      = 1
    timestamp  = "2026-01-01T00:00:00Z"
    operation  = "insert"
    details    = {"count": 10}
    prev_hash  = "0"
    nonce      = 0

    preimage   = {"details":{"count":10},"index":1,"nonce":0,
                  "operation":"insert","prev_hash":"0",
                  "timestamp":"2026-01-01T00:00:00Z"}     (no line breaks)

Why JSON, and why these rules
-----------------------------
The format had to be reproducible from both languages without either side
depending on a particular library's defaults:

* **Sorted keys, no whitespace.** Python needs ``sort_keys=True`` and
  ``separators=(",", ":")``; its defaults emit ``", "`` and ``": "``, which
  would silently produce a different digest. Rust gets both for free because
  ``serde_json`` without the ``preserve_order`` feature stores objects in a
  ``BTreeMap``. This crate does not enable ``preserve_order``, and must not
  start, or Rust will begin preserving insertion order and the two sides
  diverge again.
* **Raw UTF-8, not ``\\uXXXX`` escapes.** Python defaults to ``ensure_ascii=True``,
  which escapes non-ASCII, while ``serde_json`` writes raw UTF-8. Callers must
  pass ``ensure_ascii=False``. This is handled here, so use
  :func:`canonical_preimage` rather than calling ``json.dumps`` directly.
* **Floats only inside a bounded domain.** Round-tripping a real number to text
  is the one place where two languages can silently disagree, because a
  language may switch to exponent notation at a different threshold —
  ``repr(1e-5)`` is ``'1e-05'`` in Python while Rust's ``{}`` prints
  ``0.00001``. Floats are therefore accepted only when they are finite and
  either exactly ``0.0`` or satisfy ``1e-4 <= |x| < 1e15``, the range in which
  Python's shortest-round-trip ``repr``, Rust's ``Display`` and JavaScript's
  ``Number.prototype.toString`` all agree on plain decimal notation. Values
  outside it are rejected, which forces the caller to rescale explicitly rather
  than silently produce an unverifiable digest. See :func:`validate_float`.

Compatibility
-------------
:func:`legacy_hash` reproduces the pre-v1 Python formula so that chains written
before this change can still be verified instead of silently failing. It is
provided for verification only — never for writing new blocks.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

__all__ = [
    "FLOAT_MAX_ABS",
    "FLOAT_MIN_ABS",
    "HASH_FIELDS",
    "SPEC_VERSION",
    "SpecError",
    "canonical_hash",
    "canonical_preimage",
    "legacy_hash",
    "validate_details",
    "validate_float",
]

#: Bumped whenever the preimage changes. Recorded alongside each block so a
#: verifier knows which rule to apply.
SPEC_VERSION = "organoid-audit-v1"

#: Field names that take part in the digest, in sorted order.
HASH_FIELDS = ("details", "index", "nonce", "operation", "prev_hash", "timestamp")

#: Largest value representable by Rust's ``u64``, used to keep ``index`` and
#: ``nonce`` in the shared integer domain.
_MAX_U64 = 2**64 - 1

#: Bounds of the cross-language-safe float domain, inclusive lower / exclusive
#: upper. Outside this range the three languages disagree about when to switch
#: to exponent notation, so a digest computed from such a value is not portable.
FLOAT_MIN_ABS = 1e-4
FLOAT_MAX_ABS = 1e15


class SpecError(ValueError):
    """Raised when a block cannot be canonicalised without ambiguity."""


def _check_uint(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(
            f"{name} must be a non-negative integer (a Rust u64), got "
            f"{type(value).__name__}: {value!r}"
        )
    if value < 0 or value > _MAX_U64:
        raise SpecError(f"{name} must be within u64 range [0, 2**64-1], got {value}")
    return value


def validate_float(value: float, path: str = "details") -> None:
    """Accept a float only inside the domain all target languages format alike.

    Raises:
        SpecError: for NaN, infinities, or magnitudes outside
            ``[FLOAT_MIN_ABS, FLOAT_MAX_ABS)`` (zero is allowed).
    """
    if value != value:  # NaN
        raise SpecError(f"{path} is NaN. NaN has no canonical JSON form.")
    if value in (float("inf"), float("-inf")):
        raise SpecError(f"{path} is infinite. Infinity has no canonical JSON form.")
    if value == 0.0:
        return
    magnitude = abs(value)
    if magnitude < FLOAT_MIN_ABS or magnitude >= FLOAT_MAX_ABS:
        raise SpecError(
            f"{path} = {value!r} is outside the canonical float domain "
            f"[{FLOAT_MIN_ABS:g}, {FLOAT_MAX_ABS:g}). Outside this range Python, "
            "Rust and JavaScript disagree about when to switch to exponent "
            "notation, so the digest would depend on the writing language "
            "rather than on the data. Rescale it explicitly — for a probability "
            "store basis points as an int (0.95 -> 9500), for a length store "
            "micrometres as an int — or store the exact text as a string."
        )


def validate_details(value: Any, path: str = "details") -> None:
    """Reject anything whose serialisation is not identical in both languages.

    Allowed: ``None``, ``bool``, ``int``, ``str``, floats within
    :data:`FLOAT_MIN_ABS`../:data:`FLOAT_MAX_ABS`, and lists/dicts of these with
    string keys. Rejected: out-of-domain floats and any other object.

    Raises:
        SpecError: with the offending path, so a caller can fix the payload
            rather than hunt through the chain for the bad block.
    """
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        _check_uint(path, value)
        return
    if isinstance(value, float):
        validate_float(value, path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SpecError(
                    f"{path} has a non-string key of type {type(key).__name__}; "
                    "JSON object keys are strings in both languages"
                )
            validate_details(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            validate_details(item, f"{path}[{i}]")
        return
    raise SpecError(
        f"{path} contains {type(value).__name__}, which has no defined "
        "canonical JSON form. Allowed types are None, bool, int, str, list "
        "and dict."
    )


def canonical_preimage(
    index: int,
    timestamp: str,
    operation: str,
    details: Mapping[str, Any] | None = None,
    prev_hash: str = "",
    nonce: int = 0,
) -> bytes:
    """Build the exact byte string that gets hashed.

    Both implementations must feed these bytes to SHA-256. Exposed separately
    from :func:`canonical_hash` so a discrepancy can be diagnosed: two sides
    that disagree can compare preimages instead of only digests.
    """
    _check_uint("index", index)
    _check_uint("nonce", nonce)
    if not isinstance(timestamp, str):
        raise SpecError(f"timestamp must be a string, got {type(timestamp).__name__}")
    if not isinstance(operation, str):
        raise SpecError(
            f"operation must be the operation *name* as a string, got "
            f"{type(operation).__name__}. The payload belongs in `details`; "
            "embedding a language-specific rendering of it here is what made "
            "the old Rust preimage unreproducible."
        )
    if not isinstance(prev_hash, str):
        raise SpecError(f"prev_hash must be a string, got {type(prev_hash).__name__}")

    payload = {
        "details": dict(details or {}),
        "index": index,
        "nonce": nonce,
        "operation": operation,
        "prev_hash": prev_hash,
        "timestamp": timestamp,
    }
    validate_details(payload["details"])

    # sort_keys + compact separators + ensure_ascii=False, matching serde_json.
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_hash(
    index: int,
    timestamp: str,
    operation: str,
    details: Mapping[str, Any] | None = None,
    prev_hash: str = "",
    nonce: int = 0,
) -> str:
    """Lowercase hex SHA-256 of :func:`canonical_preimage`."""
    return hashlib.sha256(
        canonical_preimage(index, timestamp, operation, details, prev_hash, nonce)
    ).hexdigest()


def legacy_hash(
    index: int,
    timestamp: str,
    operation: str,
    details: Mapping[str, Any] | None = None,
    prev_hash: str = "",
) -> str:
    """Reproduce the pre-v1 Python formula, for verifying old chains only.

    This is the original ``sha256(json.dumps(data, sort_keys=True))`` — note the
    default ``ensure_ascii=True`` and the default ``", "``/``": "`` separators,
    and the absence of ``nonce``. Kept so historical audit logs remain
    verifiable; do not use it when writing new blocks.
    """
    data = {
        "index": index,
        "timestamp": timestamp,
        "operation": operation,
        "details": dict(details or {}),
        "prev_hash": prev_hash,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _demo() -> None:  # pragma: no cover - manual smoke check
    args = (1, "2026-01-01T00:00:00Z", "insert", {"count": 10}, "0")
    print("preimage:", canonical_preimage(*args).decode())
    print("hash    :", canonical_hash(*args))
    try:
        canonical_hash(2, "t", "insert", {"scale": 0.5}, "0")
    except SpecError as exc:
        print("float rejected as intended:", exc)


if __name__ == "__main__":  # pragma: no cover
    _demo()
