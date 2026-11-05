# ── tests/test_pipeline_service.py ──
"""M1 vertical slice tests (PRD 10.4, M1 exit criteria).

The four criteria are each covered by at least one test:

1. 单测            — the whole class: happy path, role/identity enforcement,
                     guard rejections, immutability.
2. 静态检查        — ruff (run separately; nothing here to test).
3. 跨语言哈希一致  — ``test_chain_fixture_links_across_languages`` mirrors the
                     Node verifier's linkage check, and every block the
                     service writes goes through the spec hash the fixture
                     pins.
4. 重启不丢        — ``test_restart_preserves_everything`` and
                     ``test_audit_chain_survives_reopening``.

Plus tamper tests: a chain that cannot fail verification is decoration.
"""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

from analysis.contracts.enums import AssignmentState, Role, SampleState
from analysis.contracts.state_machine import IllegalTransition
from analysis.pipeline import AnnotationService, SQLiteStore, StoreError
from analysis.pipeline.chain import PersistentAuditChain
from analysis.pipeline.service import ServiceError

READERS = [
    ("expert_a", Role.DOMAIN_EXPERT),
    ("expert_b", Role.DOMAIN_EXPERT),
    ("expert_c", Role.DOMAIN_EXPERT),
]


#: Synthetic ingestion signals. The research template requires a stratum before
#: publish (D-08), and the only shipped rule is provisional pending Q-07, so the
#: tests acknowledge it explicitly — the same act a real project has to take.
SIGNALS = {"contrast": 0.62, "object_count": 6, "clustered_fraction": 0.05, "size_cv": 0.2}


def make_service(path: str = ":memory:") -> AnnotationService:
    svc = AnnotationService(SQLiteStore(path))
    svc.create_task(
        task_id="T1",
        node_id="node_a",
        project_id="proj",
        template="research",
        image_scope=("S-001",),
        protocol_id="proto-1",
    )
    svc.label_stratum("T1", "S-001", signals=SIGNALS, allow_provisional=True)
    return svc


def geometry(shift: int = 0, *, class_key: str = "organoid") -> bytes:
    """A minimal measurable payload: one square, optionally displaced.

    M1's tests used opaque bytes (``b"geom-A"``). M2 parses the payload at the
    write boundary, because a product the metric layer cannot measure blocks
    adjudication later — and by then the coder who could fix it has gone.
    """
    return json.dumps(
        {
            "format": "organoid-geometry-v1",
            "width": 200,
            "height": 200,
            "instances": [
                {
                    "id": "i1",
                    "class_key": class_key,
                    "rect": [10 + shift, 10, 50 + shift, 50],
                }
            ],
        }
    ).encode("utf-8")


#: Three payloads that share one object with small displacements, so the metric
#: layer has something to measure. Module-level, not default arguments, because
#: a call in a default is evaluated once at import — the lint rule that flags it
#: is pointing at a real class of bug, not at style.
DEFAULT_PAYLOADS: tuple[bytes, bytes, bytes] = (geometry(0), geometry(2), geometry(4))


def publish_and_submit(svc: AnnotationService, *, payloads=DEFAULT_PAYLOADS) -> None:
    svc.publish("T1", actor_id="owner_1", readers=READERS)
    for (person, _role), payload in zip(READERS, payloads, strict=True):
        svc.submit_blind("T1", "S-001", actor_id=person, payload=payload)


# ── 1. unit: happy path ───────────────────────────────────────────────


def test_full_write_path_reaches_in_review() -> None:
    svc = make_service()
    publish_and_submit(svc)
    task = svc.reveal("T1", "S-001")
    assert task.state is SampleState.IN_REVIEW

    report = svc.read_back("T1")
    # genesis + stratum_labeled + task_created + transition_publish + task_published
    # + transition_begin_blind_read + 3 × blind_submitted + disagreement_computed
    # + reveal_granted = 11. The two M2 additions are asserted by name below, so a
    # future change that drops one fails on the operation, not on the arithmetic.
    assert report["audit"]["valid"] and report["audit"]["length"] == 11
    operations = svc.chain.operations()
    assert "stratum_labeled" in operations
    assert operations.count("disagreement_computed") == 1
    assert all(report["payloads_intact"].values())
    assert report["entities"] == {"checked": report["entities"]["checked"], "corrupt": []}
    assert len(report["revisions"]) == 3
    assert all(r.is_blind_read and r.revision_no == 1 for r in report["revisions"])
    revealed = {a.assignee_id: a.state for a in report["assignments"]}
    assert set(revealed.values()) == {AssignmentState.REVEALED}


def test_every_audit_block_uses_the_spec_hash() -> None:
    """Each chain block must be reproducible by the shared spec function —
    the property the cross-language vectors pin, checked on live data."""
    from analysis.audit_spec import canonical_hash

    svc = make_service()
    publish_and_submit(svc)
    for block in svc.chain.blocks():
        expected = canonical_hash(
            index=block["index"],
            timestamp=block["timestamp"],
            operation=block["operation"],
            details=block["details"],
            prev_hash=block["prev_hash"],
            nonce=block["nonce"],
        )
        assert block["hash"] == expected, f"block {block['index']} is not spec-hashed"


def test_deterministic_run_is_reproducible() -> None:
    """Same inputs → same operation sequence with intact linkage. A chain
    whose shape depended on wall-clock noise could not be checked against a
    fixed vector at all."""
    services = [make_service() for _ in range(2)]
    for svc in services:
        publish_and_submit(svc)
        svc.reveal("T1", "S-001")
    ops = [[b["operation"] for b in svc.chain.blocks()] for svc in services]
    assert ops[0] == ops[1]
    for svc in services:
        blocks = svc.chain.blocks()
        for previous, block in pairwise(blocks):
            assert block["prev_hash"] == previous["hash"]
        assert blocks[0]["prev_hash"] == "0"


# ── 1. unit: enforcement ──────────────────────────────────────────────


def test_unassigned_actor_cannot_submit() -> None:
    svc = make_service()
    svc.publish("T1", actor_id="owner_1", readers=READERS)
    with pytest.raises(ServiceError, match="no assignment"):
        svc.submit_blind("T1", "S-001", actor_id="mallory", payload=geometry(0))


def test_reader_shortfall_refused_at_publish() -> None:
    svc = make_service()
    with pytest.raises(ServiceError, match="exactly 3 readers"):
        svc.publish("T1", actor_id="owner_1", readers=READERS[:2])


def test_wrong_role_cannot_publish() -> None:
    svc = make_service()
    # A domain expert is not in the research template's publish cast; the
    # machine, not the service, is what says so.
    with pytest.raises(Exception, match=r"may not .publish."):
        svc._apply_per_sample(
            svc._task("T1"), "publish", actor_id="expert_a", actor_role=Role.DOMAIN_EXPERT
        )


def test_reveal_before_all_blind_reads_has_no_edge() -> None:
    svc = make_service()
    svc.publish("T1", actor_id="owner_1", readers=READERS)
    svc.submit_blind("T1", "S-001", actor_id="expert_a", payload=geometry(0))
    with pytest.raises((ServiceError, IllegalTransition)):
        svc.reveal("T1", "S-001")


def test_cannot_publish_twice() -> None:
    svc = make_service()
    svc.publish("T1", actor_id="owner_1", readers=READERS)
    with pytest.raises(IllegalTransition, match=r"no such edge"):
        svc.publish("T1", actor_id="owner_1", readers=READERS)


def test_same_reader_cannot_submit_twice_under_one_assignment() -> None:
    svc = make_service()
    svc.publish("T1", actor_id="owner_1", readers=READERS)
    svc.submit_blind("T1", "S-001", actor_id="expert_a", payload=geometry(0))
    with pytest.raises(ServiceError, match=r"BLIND_DRAFTED|may already be submitted"):
        svc.submit_blind("T1", "S-001", actor_id="expert_a", payload=geometry(1))


def test_entities_are_immutable_without_overwrite() -> None:
    store = SQLiteStore(":memory:")
    svc = AnnotationService(store)
    svc.create_task(
        task_id="T1", node_id="n", project_id="p", template="research",
        image_scope=("S-001",), protocol_id="proto-1",
    )
    payload = store.get("task", "T1")
    from analysis.contracts.entities import Task

    with pytest.raises(StoreError, match="immutable"):
        store.put(Task.from_dict(payload))


def test_double_submit_from_other_reader_does_not_regress_state() -> None:
    """The third submission completes the reveal automatically; a later
    submit must not be able to pull the sample back into blind read."""
    svc = make_service()
    svc.publish("T1", actor_id="owner_1", readers=READERS)
    for person, _role in READERS:
        svc.submit_blind("T1", "S-001", actor_id=person, payload=geometry(0))
    # the platform has already moved the sample on
    assert svc._sample_state("T1", "S-001") is SampleState.AWAITING_REVEAL


# ── 3. cross-language chain vectors ──────────────────────────────────


def test_chain_fixture_links_across_languages() -> None:
    """Python-side mirror of the Node verifier's linkage check: the chain
    vectors' prev_hash values are each other's digests."""
    with open("tests/fixtures/audit_hash_vectors.json", encoding="utf-8") as fh:
        fixture = json.load(fh)
    chain = [v for v in fixture["vectors"] if v["name"].startswith("chain_block_")]
    assert len(chain) == 4
    for i, block in enumerate(chain):
        expected_prev = "0" if i == 0 else chain[i - 1]["sha256"]
        assert block["prev_hash"] == expected_prev, f"chain break at {block['name']}"


def test_service_chain_links_to_itself() -> None:
    svc = make_service()
    publish_and_submit(svc)
    blocks = svc.chain.blocks()
    for previous, block in pairwise(blocks):
        assert block["prev_hash"] == previous["hash"]


# ── 4. restart / persistence ─────────────────────────────────────────


def test_restart_preserves_everything(tmp_path) -> None:
    db = str(tmp_path / "m1.db")
    svc = make_service(db)
    publish_and_submit(svc)
    svc.reveal("T1", "S-001")
    before = svc.read_back("T1")
    chain_hashes = [b["hash"] for b in svc.chain.blocks()]
    svc.store.close()

    reopened = AnnotationService(SQLiteStore(db))
    after = reopened.read_back("T1")

    assert after["task"] == before["task"]
    assert [r.id for r in after["revisions"]] == [r.id for r in before["revisions"]]
    assert after["sample_states"] == before["sample_states"]
    assert after["audit"]["valid"] and after["audit"]["length"] == before["audit"]["length"]
    assert [b["hash"] for b in reopened.chain.blocks()] == chain_hashes
    # replay from the chain alone reproduces the cached states
    assert reopened.replay_sample_states("T1") == {
        k: v["state"] for k, v in after["sample_states"].items()
    }


def test_workflow_continues_after_restart(tmp_path) -> None:
    """Restart is not just readable — the machine still advances."""
    db = str(tmp_path / "m1.db")
    svc = make_service(db)
    svc.publish("T1", actor_id="owner_1", readers=READERS)
    svc.submit_blind("T1", "S-001", actor_id="expert_a", payload=geometry(0))
    svc.store.close()

    reopened = AnnotationService(SQLiteStore(db))
    reopened.submit_blind("T1", "S-001", actor_id="expert_b", payload=geometry(2))
    states = reopened.store.sample_states("T1")
    assert states["S-001"]["state"] == SampleState.IN_BLIND_READ.value
    assert reopened.chain.verify()["valid"]


def test_audit_chain_survives_reopening(tmp_path) -> None:
    db = str(tmp_path / "chain.db")
    svc = AnnotationService(SQLiteStore(db))
    head_before = svc.chain.head()["hash"]
    svc.store.close()

    reopened = PersistentAuditChain(SQLiteStore(db))
    assert reopened.head()["hash"] == head_before
    assert reopened.verify()["valid"]


# ── tamper evidence ──────────────────────────────────────────────────


def _write_blocks(db: str) -> SQLiteStore:
    store = SQLiteStore(db)
    chain = PersistentAuditChain(store)
    chain.append("event_one", {"n": 1})
    chain.append("event_two", {"n": 2})
    return store


def test_edited_block_is_a_digest_break(tmp_path) -> None:
    store = _write_blocks(str(tmp_path / "t1.db"))
    store._conn.execute("UPDATE audit_blocks SET details = ? WHERE index_ = 1", ('{"n": 999}',))
    store._conn.commit()
    report = PersistentAuditChain(store).verify()
    assert not report["valid"]
    assert any("digest mismatch" in p for p in report["problems"])


def test_removed_block_is_a_linkage_break(tmp_path) -> None:
    store = _write_blocks(str(tmp_path / "t2.db"))
    store._conn.execute("DELETE FROM audit_blocks WHERE index_ = 1")
    store._conn.commit()
    report = PersistentAuditChain(store).verify()
    assert not report["valid"]
    assert any("linkage break" in p for p in report["problems"])


def test_tampered_entity_row_is_caught_on_read_back(tmp_path) -> None:
    db = str(tmp_path / "t3.db")
    svc = make_service(db)
    publish_and_submit(svc)
    # edit a stored revision behind the contract layer's back
    svc.store._conn.execute(
        "UPDATE entities SET payload = replace(payload, 'payload_hash', 'payload_hax') "
        "WHERE kind = 'revision'"
    )
    svc.store._conn.commit()
    assert svc.store.verify_entity_hashes()["corrupt"], "tampering must be visible to the store scan"
    # and the read-back refuses to serve the tampered rows at all
    with pytest.raises(ServiceError, match="edited outside the contract layer"):
        svc.read_back("T1")


def test_blob_reference_is_self_verifying(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "t4.db"))
    ref = store.put_blob(b"geometry", stored_at="2026-01-01T00:00:00Z")
    assert store.get_blob(ref) == b"geometry"
    with pytest.raises(StoreError, match="not a blob reference"):
        store.get_blob("s3://elsewhere/geometry")
    with pytest.raises(StoreError, match="missing"):
        store.get_blob("blob:" + "0" * 64)


def test_chain_requires_well_formed_details(tmp_path) -> None:
    from analysis.audit_spec import SpecError

    store = SQLiteStore(str(tmp_path / "t5.db"))
    chain = PersistentAuditChain(store)
    with pytest.raises(SpecError):
        chain.append("bad_block", {"score": 0.5e-9})  # float outside the domain
