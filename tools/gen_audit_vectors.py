"""Generate the shared cross-language hash vectors.

Run:  ./.venv/Scripts/python.exe tools/gen_audit_vectors.py
      ./.venv/Scripts/python.exe tools/gen_audit_vectors.py --check

Without arguments this (re)writes tests/fixtures/audit_hash_vectors.json.
With --check it recomputes everything and compares against the file on disk,
exiting non-zero on any difference. CI uses --check so a spec change cannot be
committed without regenerating the fixture that the Rust and Node
implementations are verified against.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.audit_spec import SPEC_VERSION, canonical_hash, canonical_preimage

VECTORS = [
    {
        "name": "genesis",
        "index": 0,
        "timestamp": "2026-01-01T00:00:00Z",
        "operation": "genesis",
        "details": {"message": "Organoid-FL Audit Chain initialized"},
        "prev_hash": "0",
        "nonce": 0,
    },
    {
        "name": "insert_counts",
        "index": 1,
        "timestamp": "2026-01-01T00:00:01Z",
        "operation": "insert",
        "details": {"count": 10},
        "prev_hash": "0",
        "nonce": 0,
    },
    {
        "name": "search_with_results",
        "index": 2,
        "timestamp": "2026-01-01T00:00:02Z",
        "operation": "search",
        "details": {"k": 5, "result_count": 5},
        "prev_hash": "3f786850e387550fdab836ed7e6dc881de23001b",
        "nonce": 0,
    },
    {
        "name": "nested_details_and_nonzero_nonce",
        "index": 7,
        "timestamp": "2026-01-01T00:00:07Z",
        "operation": "fl_round",
        "details": {
            "round": 3,
            "clients": [1, 2, 3],
            "gated": True,
            "note": None,
            "nested": {"b": [1, 2], "a": "z"},
        },
        "prev_hash": "a" * 64,
        "nonce": 42,
    },
    {
        "name": "unicode_payload",
        "index": 8,
        "timestamp": "2026-01-01T00:00:08Z",
        "operation": "label_change",
        "details": {"label": "类器官", "note": "smiley \U0001F600 and tab\t"},
        "prev_hash": "b" * 64,
        "nonce": 0,
    },
    {
        "name": "empty_details",
        "index": 9,
        "timestamp": "2026-01-01T00:00:09Z",
        "operation": "system",
        "details": {},
        "prev_hash": "c" * 64,
        "nonce": 0,
    },
    {
        "name": "metrics_with_in_domain_floats",
        "index": 10,
        "timestamp": "2026-01-01T00:00:10Z",
        "operation": "fl_round",
        "details": {"accuracy": 0.95, "loss": 0.123456, "lr": 0.001, "tiny": 0.0001},
        "prev_hash": "d" * 64,
        "nonce": 0,
    },
    {
        "name": "float_boundary_just_inside",
        "index": 11,
        "timestamp": "2026-01-01T00:00:11Z",
        "operation": "calibration",
        "details": {"pixel_size_um": 0.0001, "field_width_um": 99999999999.5},
        "prev_hash": "e" * 64,
        "nonce": 0,
    },
]

#: A linked chain, not just isolated digests: each vector's prev_hash is the
#: previous vector's sha256. M1's cross-language exit criterion is about the
#: *chain* ("写审计 → 读回" must reproduce under Node/Rust), so the fixture
#: pins linkage as well as digests. The operations mirror an M1 write path:
#: genesis → task_created → blind_submitted → blind_submitted.
CHAIN_VECTORS = [
    {
        "name": "chain_block_0_genesis",
        "index": 0,
        "timestamp": "2026-01-02T00:00:00Z",
        "operation": "genesis",
        "details": {"message": "annotation pipeline audit chain initialised"},
        "prev_hash": "0",
        "nonce": 0,
    },
    {
        "name": "chain_block_1_task_created",
        "index": 1,
        "timestamp": "2026-01-02T00:00:01Z",
        "operation": "task_created",
        "details": {"protocol_id": "proto-1", "redundancy_n": 3, "samples": 1, "task_id": "T-M1", "template": "research"},
        # prev_hash filled in below from the computed digest of block 0
    },
    {
        "name": "chain_block_2_blind_submitted_a",
        "index": 2,
        "timestamp": "2026-01-02T00:00:02Z",
        "operation": "blind_submitted",
        "details": {"assignment_id": "T-M1:S-001:expert_a", "blind_reads_submitted": 1, "payload_hash": "a" * 64, "redundancy_n": 3, "revision_id": "T-M1:S-001:expert_a:r1", "sample_id": "S-001", "task_id": "T-M1"},
    },
    {
        "name": "chain_block_3_blind_submitted_b",
        "index": 3,
        "timestamp": "2026-01-02T00:00:03Z",
        "operation": "blind_submitted",
        "details": {"assignment_id": "T-M1:S-001:expert_b", "blind_reads_submitted": 2, "payload_hash": "b" * 64, "redundancy_n": 3, "revision_id": "T-M1:S-001:expert_b:r1", "sample_id": "S-001", "task_id": "T-M1"},
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="verify the fixture on disk instead of rewriting it; exits non-zero "
             "on any difference (used by CI)",
    )
    args = parser.parse_args()

    out = []
    for spec in VECTORS:
        args_ = (spec["index"], spec["timestamp"], spec["operation"],
                 spec["details"], spec["prev_hash"], spec["nonce"])
        out.append({
            **spec,
            "preimage": canonical_preimage(*args_).decode("utf-8"),
            "sha256": canonical_hash(*args_),
        })

    # Link the chain: each block's prev_hash is the digest of the one before.
    chain_hashes: list[str] = []
    for spec in CHAIN_VECTORS:
        prev = chain_hashes[-1] if chain_hashes else "0"
        spec = {"nonce": 0, **spec, "prev_hash": prev}
        args_ = (spec["index"], spec["timestamp"], spec["operation"],
                 spec["details"], spec["prev_hash"], spec["nonce"])
        digest = canonical_hash(*args_)
        chain_hashes.append(digest)
        out.append({
            **spec,
            "preimage": canonical_preimage(*args_).decode("utf-8"),
            "sha256": digest,
        })

    payload = {
        "spec_version": SPEC_VERSION,
        "note": (
            "Shared cross-language audit-chain vectors. Both the Python and the "
            "Rust implementation must reproduce every sha256 exactly. "
            "Regenerate with tools/gen_audit_vectors.py only when the spec "
            "version changes."
        ),
        "vectors": out,
    }

    target = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "fixtures", "audit_hash_vectors.json",
    )

    if args.check:
        if not os.path.exists(target):
            print(f"FAIL: {target} does not exist; run without --check to create it")
            return 1
        with open(target, encoding="utf-8") as fh:
            on_disk = json.load(fh)

        problems: list[str] = []
        if on_disk.get("spec_version") != SPEC_VERSION:
            problems.append(
                f"spec_version on disk is {on_disk.get('spec_version')!r}, "
                f"code says {SPEC_VERSION!r}"
            )
        disk_by_name = {v.get("name"): v for v in on_disk.get("vectors", [])}
        for fresh in out:
            stale = disk_by_name.get(fresh["name"])
            if stale is None:
                problems.append(f"vector {fresh['name']!r} is missing from the fixture")
                continue
            for field in ("preimage", "sha256", "details", "nonce", "operation"):
                if stale.get(field) != fresh[field]:
                    problems.append(
                        f"vector {fresh['name']!r} field {field!r} differs:\n"
                        f"    on disk: {stale.get(field)!r}\n"
                        f"    computed: {fresh[field]!r}"
                    )
        extra = set(disk_by_name) - {v["name"] for v in out}
        if extra:
            problems.append(f"fixture contains unknown vectors: {sorted(extra)}")

        if problems:
            print("AUDIT VECTORS OUT OF DATE\n")
            for problem in problems:
                print(f"  - {problem}")
            print("\nRun tools/gen_audit_vectors.py to regenerate, then re-run the")
            print("Node check and `cargo test` so every language is re-verified.")
            return 1

        print(f"OK: {len(out)} vectors match {target} (spec {SPEC_VERSION})")
        return 0

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    for item in out:
        print(f"{item['name']:34s} {item['sha256']}")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
