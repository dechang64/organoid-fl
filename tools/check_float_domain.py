from __future__ import annotations
"""Verify the cross-language float domain claim.

The audit spec accepts floats only where the target languages agree on
plain-decimal, shortest-round-trip formatting.

Rust is not installed on this machine, so this script checks the Python side
thoroughly and uses Node/JavaScript as a *partial* third opinion. Two limits of
that comparison are deliberate and worth stating, because both look like bugs
until you know why:

1. JavaScript has a single number type, so ``String(1.0)`` is ``"1"`` while
   Python's ``json.dumps(1.0)`` is ``"1.0"``. Rust's ``serde_json`` emits
   ``"1.0"`` for an f64 — it uses ryu, which appends ``.0`` so a float is not
   mistaken for an integer. So for integral-valued floats Python and Rust agree
   and JavaScript is the outlier; those cases are excluded from the JS
   comparison rather than treated as failures.

2. Python switches to exponent notation below 1e-4, which is precisely why
   ``FLOAT_MIN_ABS`` is 1e-4. Values under that bound are out of domain on
   purpose and are checked for *rejection* instead of for formatting.

Run:  ./.venv/Scripts/python.exe tools/check_float_domain.py
"""

import json
import os
import random
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.audit_spec import FLOAT_MIN_ABS, SpecError, validate_float

# Resolve Node from PATH so this works on CI as well as on the dev machine,
# where it lives under the managed binaries directory.
NODE = shutil.which("node") or shutil.which("node.exe")


def python_form(x: float) -> str:
    """Exactly what json.dumps emits for this float."""
    return json.dumps(x)


def sample_in_domain(n=4000, seed=0):
    """Sample the domain uniformly in log space, staying strictly inside it."""
    rng = random.Random(seed)
    values = [FLOAT_MIN_ABS, 0.1, 0.5, 0.95, 0.123456, 2.675, 1234.5678]
    for _ in range(n):
        exp = rng.uniform(-4, 14.9)          # keep magnitude below 1e15
        mantissa = rng.uniform(1.0, 9.999)
        values.append(mantissa * (10**exp))
    return [v for v in values if _in_domain(v)]


def _in_domain(v: float) -> bool:
    try:
        validate_float(v)
        return True
    except SpecError:
        return False


def node_form(values):
    script = (
        "let s='';"
        "process.stdin.on('data',d=>s+=d);"
        "process.stdin.on('end',()=>{"
        "  const vals = JSON.parse(s);"
        "  process.stdout.write(JSON.stringify(vals.map(v => String(v))));"
        "});"
    )
    proc = subprocess.run(
        [NODE, "-e", script], input=json.dumps(values),
        capture_output=True, text=True, timeout=300, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def main() -> int:
    values = sample_in_domain()
    py = [python_form(v) for v in values]

    # 1. Python must never fall back to exponent notation inside the domain.
    exponent = [(v, s) for v, s in zip(values, py, strict=True) if "e" in s or "E" in s]

    # 2. Non-integral values should also agree with JavaScript, when Node is
    #    available. Without it, the Python-side checks still run — they are the
    #    ones that matter for the spec, since Rust and Python are the two
    #    implementations that must agree.
    non_integral = [(v, s) for v, s in zip(values, py, strict=True) if not float(v).is_integer()]
    js_mismatch: list[tuple[float, str, str]] = []
    if NODE:
        nd = dict(zip(
            [v for v, _ in non_integral],
            node_form([v for v, _ in non_integral]),
            strict=True,
        ))
        js_mismatch = [(v, s, nd[v]) for v, s in non_integral if nd[v] != s]

    # 3. Integral-valued floats must keep the trailing .0 that ryu produces, so
    #    they are never confused with an integer in the Rust preimage.
    integral = [
        (v, s) for v, s in zip(values, py, strict=True)
        if float(v).is_integer() and v != 0.0
    ]
    integral_lost_point = [(v, s) for v, s in integral if "." not in s]
    zero_form = python_form(0.0)

    print(f"sampled                       : {len(values)} in-domain values")
    print(f"python used exponent notation : {len(exponent)}")
    if NODE:
        print(f"non-integral vs node mismatch : {len(js_mismatch)} "
              f"(of {len(non_integral)} compared)")
    else:
        print("non-integral vs node mismatch : SKIPPED (node not on PATH)")
    print(f"integral floats losing '.0'   : {len(integral_lost_point)} "
          f"(of {len(integral)})")
    print(f"python form of 0.0            : {zero_form!r}")
    for v, a, b in js_mismatch[:5]:
        print(f"   {v!r}: python={a} node={b}")

    print("\nout-of-domain rejection (must all be True):")
    rejected = []
    for bad in (1e-5, 1e-9, 0 - 1e-5, 1e15, 1e20, float("inf"), float("-inf"),
                float("nan")):
        try:
            validate_float(bad)
            ok = False
        except SpecError:
            ok = True
        rejected.append((bad, ok))
        print(f"   {bad!s:>12} rejected={ok}")

    ok = (
        not exponent
        and not js_mismatch
        and not integral_lost_point
        and zero_form == "0.0"
        and all(flag for _, flag in rejected)
    )
    print("\nFLOAT DOMAIN CONSISTENT" if ok else "\nFLOAT DOMAIN PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
