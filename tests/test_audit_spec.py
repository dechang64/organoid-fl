# ── tests/test_audit_spec.py ──
"""Tests for the canonical audit-chain hash specification.

The point of these tests is cross-implementation agreement. A test that only
checks Python against itself would pass even if the preimage were defined in a
way no other language can reproduce — which is exactly how the original bug went
unnoticed: ``analysis/audit_engine.py`` and ``src/blockchain.rs`` each hashed
something plausible and self-consistent, and disagreed with each other.

So the assertions here are anchored to ``tests/fixtures/audit_hash_vectors.json``,
which is also consumed by ``tools/verify_audit_vectors.mjs`` (Node) and by the
Rust test in ``src/blockchain.rs``. If the fixture is regenerated, every
language must be re-run, not just this file.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.audit_engine import AuditBlock, AuditEngine
from analysis.audit_spec import (
    FLOAT_MAX_ABS,
    FLOAT_MIN_ABS,
    SPEC_VERSION,
    SpecError,
    canonical_hash,
    canonical_preimage,
    legacy_hash,
    validate_details,
    validate_float,
)

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures", "audit_hash_vectors.json",
)


def load_fixture():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


class TestSharedVectors(unittest.TestCase):
    """The Python implementation must reproduce every vector bit for bit."""

    def setUp(self):
        self.fixture = load_fixture()

    def test_fixture_declares_the_expected_spec_version(self):
        self.assertEqual(self.fixture["spec_version"], SPEC_VERSION)

    def test_there_are_vectors_to_check(self):
        self.assertGreaterEqual(len(self.fixture["vectors"]), 6)

    def test_every_preimage_matches(self):
        for v in self.fixture["vectors"]:
            with self.subTest(vector=v["name"]):
                actual = canonical_preimage(
                    v["index"], v["timestamp"], v["operation"],
                    v["details"], v["prev_hash"], v["nonce"],
                ).decode("utf-8")
                self.assertEqual(actual, v["preimage"])

    def test_every_digest_matches(self):
        for v in self.fixture["vectors"]:
            with self.subTest(vector=v["name"]):
                actual = canonical_hash(
                    v["index"], v["timestamp"], v["operation"],
                    v["details"], v["prev_hash"], v["nonce"],
                )
                self.assertEqual(actual, v["sha256"])
                self.assertEqual(len(actual), 64)
                self.assertEqual(actual, actual.lower())

    def test_fixture_covers_the_awkward_cases(self):
        """Guard the coverage itself: a future edit must not quietly drop the
        vectors that catch escaping, non-ASCII, nesting and empty payloads."""
        names = {v["name"] for v in self.fixture["vectors"]}
        for required in (
            "unicode_payload",                     # ensure_ascii=False
            "nested_details_and_nonzero_nonce",    # recursive key sorting
            "empty_details",                       # {} not omitted
            "metrics_with_in_domain_floats",       # float formatting
        ):
            self.assertIn(required, names)


class TestPreimageShape(unittest.TestCase):
    def test_preimage_is_compact_sorted_json(self):
        text = canonical_preimage(
            1, "2026-01-01T00:00:00Z", "insert", {"count": 10}, "0", 0
        ).decode("utf-8")
        self.assertNotIn(", ", text)
        self.assertNotIn(": ", text)
        self.assertNotIn("\n", text)
        keys = list(json.loads(text).keys())
        self.assertEqual(keys, sorted(keys))
        # Sorted order must put details first, timestamp last.
        self.assertTrue(text.startswith('{"details":'))
        self.assertTrue(text.endswith('}'))

    def test_non_ascii_is_not_escaped(self):
        """``ensure_ascii`` defaults to True in json.dumps and would escape this.

        serde_json writes raw UTF-8, so an escaped preimage would produce a
        digest the Rust side can never match. Reproducing the fixture vector is
        the real check; this asserts the mechanism directly so the failure
        message is actionable.
        """
        text = canonical_preimage(
            0, "t", "op", {"label": "类器官"}, "0", 0
        ).decode("utf-8")
        self.assertIn("类器官", text)
        self.assertNotIn("\\u", text)

    def test_preimage_bytes_are_utf8(self):
        raw = canonical_preimage(0, "t", "op", {"label": "类器官"}, "0", 0)
        self.assertIsInstance(raw, bytes)
        self.assertEqual(raw, raw.decode("utf-8").encode("utf-8"))

    def test_digest_changes_when_any_hashed_field_changes(self):
        base = (1, "2026-01-01T00:00:00Z", "insert", {"count": 10}, "0", 0)
        baseline = canonical_hash(*base)
        variants = {
            "index": (2, base[1], base[2], base[3], base[4], base[5]),
            "timestamp": (base[0], "2026-01-01T00:00:01Z", base[2], base[3], base[4], base[5]),
            "operation": (base[0], base[1], "delete", base[3], base[4], base[5]),
            "details": (base[0], base[1], base[2], {"count": 11}, base[4], base[5]),
            "prev_hash": (base[0], base[1], base[2], base[3], "1", base[5]),
            "nonce": (base[0], base[1], base[2], base[3], base[4], 1),
        }
        for field, args in variants.items():
            with self.subTest(field=field):
                self.assertNotEqual(canonical_hash(*args), baseline)

    def test_unsigned_digest_is_not_truncated(self):
        digest = canonical_hash(0, "t", "op", {}, "0", 0)
        self.assertEqual(len(digest), 64)


class TestValidation(unittest.TestCase):
    def test_u64_bounds(self):
        canonical_hash(2**64 - 1, "t", "op", {}, "0", 2**64 - 1)
        with self.assertRaises(SpecError):
            canonical_hash(2**64, "t", "op", {}, "0", 0)
        with self.assertRaises(SpecError):
            canonical_hash(-1, "t", "op", {}, "0", 0)

    def test_boolean_is_not_accepted_as_an_index(self):
        """``bool`` is a subclass of ``int`` in Python; True must not slip
        through as index 1 and produce a plausible-looking digest."""
        with self.assertRaises(SpecError):
            canonical_hash(True, "t", "op", {}, "0", 0)

    def test_operation_must_be_a_name_not_a_payload(self):
        with self.assertRaises(SpecError):
            canonical_hash(0, "t", {"insert": 3}, {}, "0", 0)

    def test_floats_outside_the_domain_are_rejected(self):
        for bad in (FLOAT_MIN_ABS / 10, FLOAT_MAX_ABS, FLOAT_MAX_ABS * 10,
                    float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=bad):
                with self.assertRaises(SpecError):
                    validate_float(bad)

    def test_floats_inside_the_domain_are_accepted(self):
        for good in (0.0, FLOAT_MIN_ABS, 0.5, 0.95, 1.0, FLOAT_MAX_ABS / 10):
            with self.subTest(value=good):
                validate_float(good)

    def test_nested_offending_value_reports_its_path(self):
        with self.assertRaises(SpecError) as ctx:
            validate_details({"metrics": {"loss": 1e-9}})
        self.assertIn("details.metrics.loss", str(ctx.exception))

    def test_non_string_keys_are_rejected(self):
        with self.assertRaises(SpecError):
            validate_details({1: "one"})

    def test_arbitrary_objects_are_rejected(self):
        with self.assertRaises(SpecError):
            validate_details({"when": object()})

    def test_lists_and_none_are_accepted(self):
        validate_details({"a": [1, "two", None, True], "b": {}})

    def test_append_rejects_an_unverifiable_payload(self):
        """The chain must refuse a block it cannot hash portably."""
        engine = AuditEngine()
        with self.assertRaises(SpecError):
            engine.append("fl_round", {"accuracy": 1e-9})
        self.assertEqual(len(engine), 1, "the bad block was added anyway")


class TestLegacyCompatibility(unittest.TestCase):
    def test_legacy_and_canonical_digests_differ(self):
        """Documents that the spec change was a real change, not a no-op."""
        args = (1, "2026-01-01T00:00:00Z", "insert", {"count": 10}, "0")
        self.assertNotEqual(legacy_hash(*args), canonical_hash(*args, 0))

    def test_a_legacy_block_still_verifies(self):
        block = AuditBlock(
            index=1,
            timestamp="2026-01-01T00:00:00Z",
            operation="insert",
            details={"count": 10},
            prev_hash="0",
        )
        block.hash = legacy_hash(
            block.index, block.timestamp, block.operation, block.details,
            block.prev_hash,
        )
        self.assertFalse(block.uses_current_spec())
        self.assertTrue(block.verify(), "a pre-unification block failed to verify")
        self.assertFalse(
            block.verify(allow_legacy=False),
            "a legacy block reported as canonical; the distinction matters when "
            "deciding whether an entry can be cross-checked against Rust",
        )

    def test_a_canonical_block_is_reported_as_canonical(self):
        block = AuditBlock(1, "2026-01-01T00:00:00Z", "insert", {"count": 10}, "0")
        block.hash = block.compute_hash()
        self.assertTrue(block.uses_current_spec())


class TestChainBehaviour(unittest.TestCase):
    def test_genesis_is_canonical(self):
        engine = AuditEngine()
        self.assertEqual(len(engine), 1)
        self.assertTrue(engine.chain[0].uses_current_spec())
        self.assertEqual(engine.chain[0].spec_version, SPEC_VERSION)

    def test_chain_links_and_verifies(self):
        engine = AuditEngine()
        for i in range(5):
            engine.append("insert", {"count": i})
        self.assertEqual(len(engine), 6)
        self.assertTrue(engine.verify_chain())
        self.assertTrue(all(b.uses_current_spec() for b in engine.chain))

    def test_tampering_is_detected(self):
        engine = AuditEngine()
        engine.append("insert", {"count": 10})
        engine.chain[1].details["count"] = 11
        self.assertFalse(engine.verify_chain())
        self.assertFalse(engine.chain[1].verify())

    def test_reordering_blocks_breaks_the_chain(self):
        engine = AuditEngine()
        engine.append("insert", {"count": 1})
        engine.append("insert", {"count": 2})
        engine.chain[1], engine.chain[2] = engine.chain[2], engine.chain[1]
        self.assertFalse(engine.verify_chain())

    def test_nonce_is_hashed(self):
        engine = AuditEngine()
        a = engine.append("op", {"x": 1}, nonce=0)
        b = engine.append("op", {"x": 1}, nonce=1)
        self.assertNotEqual(a.hash, b.hash)

    def test_stats_report_spec_versions(self):
        engine = AuditEngine()
        engine.append("op", {})
        stats = engine.get_stats()
        self.assertEqual(stats["current_spec"], 2)
        self.assertEqual(stats["legacy_spec"], 0)
        self.assertEqual(stats["expected_spec"], SPEC_VERSION)

    def test_a_mixed_chain_reports_both_spec_versions(self):
        engine = AuditEngine()
        engine.append("op", {})
        legacy = AuditBlock(
            index=2, timestamp="2026-01-01T00:00:00Z", operation="op",
            details={}, prev_hash=engine.chain[-1].hash,
        )
        legacy.hash = legacy_hash(
            legacy.index, legacy.timestamp, legacy.operation, legacy.details,
            legacy.prev_hash,
        )
        engine.chain.append(legacy)
        versions = engine.audit_spec_versions()
        self.assertEqual(versions["current_spec"], 2)
        self.assertEqual(versions["legacy_spec"], 1)

    def test_eviction_keeps_the_chain_verifiable(self):
        engine = AuditEngine(max_blocks=3)
        for i in range(6):
            engine.append("op", {"i": i})
        self.assertEqual(len(engine), 3)
        # The surviving tail is still internally consistent.
        for i in range(1, len(engine)):
            self.assertEqual(engine.chain[i].prev_hash, engine.chain[i - 1].hash)
            self.assertTrue(engine.chain[i].verify())

    def test_preimage_is_introspectable(self):
        """Two implementations that disagree need to compare bytes, not just
        digests, to find which field differs."""
        block = AuditBlock(1, "t", "op", {"a": 1}, "0")
        self.assertEqual(
            block.preimage().decode("utf-8"),
            canonical_preimage(1, "t", "op", {"a": 1}, "0", 0).decode("utf-8"),
        )

    def test_unicode_payload_survives_an_append(self):
        engine = AuditEngine()
        engine.append("label_change", {"label": "类器官", "note": "emoji \U0001F600"})
        self.assertTrue(engine.verify_chain())


if __name__ == "__main__":
    unittest.main(verbosity=2)
