# ── analysis/pipeline/store.py ──
"""SQLite persistence for the M1 vertical slice (PRD 10.4, M1).

M0 froze the contracts; this module is where they stop being objects in
memory and become rows that survive a process restart — the third of M1's
four exit criteria ("重启不丢").

Design notes:

* One generic ``entities`` table, not one table per entity. The nine PRD 7.2
  records are schema-frozen by the contract layer (strict ``to_dict`` /
  ``from_dict``, unknown keys rejected), so the database does not need to
  re-state that schema — and a schema that exists in two places drifts in two
  places. Adding a contract field is then a code change, not a migration.
* Payloads (annotation geometry) live in a content-addressed ``blobs`` table,
  keyed by the SHA-256 that ``Revision.payload_hash`` already carries. The
  reference ``blob:<hash>`` is therefore self-verifying: copying a payload
  under the wrong hash is unrepresentable.
* The audit chain gets its own table with ``index`` as primary key, because it
  is append-only and verified as a sequence, not as individual rows.
* WAL mode, because the demo Streamlit process and a CLI verifier may hold the
  same file open at the same time.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterator, Mapping, Optional

from analysis.contracts.entities import ContractEntity
from analysis.contracts.hashing import verify_content_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    kind         TEXT NOT NULL,
    id           TEXT NOT NULL,
    payload      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (kind, id)
);
CREATE TABLE IF NOT EXISTS blobs (
    hash  TEXT PRIMARY KEY,
    bytes BLOB NOT NULL,
    stored_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_blocks (
    index_    INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    details   TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    nonce     INTEGER NOT NULL,
    hash      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_states (
    task_id   TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    state     TEXT NOT NULL,
    reveal_done INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, sample_id)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class StoreError(ValueError):
    """The store refused an operation that would corrupt its invariants."""


class SQLiteStore:
    """Durable home for contract entities, payload blobs and the audit chain."""

    def __init__(self, path: str) -> None:
        #: ``:memory:`` is allowed and used by the unit tests; every other path
        #: is what makes the restart test meaningful.
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.path = path

    # ── entities ─────────────────────────────────────────────────────

    def put(
        self, entity: ContractEntity, *, overwrite: bool = False, key: Optional[str] = None
    ) -> None:
        """Persist one contract entity, immutably by default.

        Immutability is the default on purpose: every entity in PRD 7.2 is a
        frozen dataclass, and the MVP defect L2-2 (a reject silently clearing
        state) is exactly what "update the row in place" enables. A caller that
        genuinely means to replace a record — the state-machine advance that
        rewrites a Task with a new aggregate state, for example — says so with
        ``overwrite=True`` and the audit chain says why.

        ``key`` overrides the row key for the one entity that has no ``id`` of its
        own: ``StratumLabel`` is 1:1 with a sample (PRD 7.2 lists no id field), so
        it is keyed by ``sample_id``. Defaulting to ``payload["id"]`` keeps every
        other entity's key derived from the record rather than from the caller.
        """
        payload = entity.to_dict()

        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        kind = entity.ENTITY
        entity_id = key if key is not None else str(payload.get("id", ""))
        if not entity_id:
            raise StoreError(
                f"{type(entity).__name__} has no id; pass key= to say what it is keyed by"
            )
        try:
            self._conn.execute(
                "INSERT INTO entities (kind, id, payload, content_hash) VALUES (?, ?, ?, ?)",
                (kind, entity_id, text, entity.content_hash()),
            )
        except sqlite3.IntegrityError as exc:
            if not overwrite:
                raise StoreError(
                    f"{kind} {entity_id!r} already exists; entities are immutable by "
                    "default. Pass overwrite=True if this is a state-machine advance."
                ) from exc
            self._conn.execute(
                "UPDATE entities SET payload = ?, content_hash = ? WHERE kind = ? AND id = ?",
                (text, entity.content_hash(), kind, entity_id),
            )
        self._conn.commit()

    def get(self, kind: str, entity_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT payload FROM entities WHERE kind = ? AND id = ?", (kind, entity_id)
        ).fetchone()
        import json

        return json.loads(row["payload"]) if row is not None else None

    def list_ids(self, kind: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT id FROM entities WHERE kind = ? ORDER BY id", (kind,)
        ).fetchall()
        return [row["id"] for row in rows]

    def count(self, kind: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM entities WHERE kind = ?", (kind,)
        ).fetchone()
        return int(row["n"])

    # ── blobs ────────────────────────────────────────────────────────

    def put_blob(self, payload: bytes, *, stored_at: str) -> str:
        """Content-address a payload; returns the ``blob:<hash>`` reference."""
        from analysis.contracts.hashing import content_hash

        digest = content_hash({"bytes": payload.hex()})
        existing = self._conn.execute(
            "SELECT hash FROM blobs WHERE hash = ?", (digest,)
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO blobs (hash, bytes, stored_at) VALUES (?, ?, ?)",
                (digest, sqlite3.Binary(payload), stored_at),
            )
            self._conn.commit()
        return f"blob:{digest}"

    def get_blob(self, ref: str) -> bytes:
        """Read back a payload by reference, refusing a malformed one."""
        if not ref.startswith("blob:"):
            raise StoreError(
                f"payload_ref {ref!r} is not a blob reference; refs are 'blob:<sha256>'"
            )
        digest = ref[len("blob:") :]
        row = self._conn.execute("SELECT bytes FROM blobs WHERE hash = ?", (digest,)).fetchone()
        if row is None:
            raise StoreError(f"blob {digest[:16]}… is missing from the store")
        return bytes(row["bytes"])

    # ── per-sample orchestration state ───────────────────────────────

    def set_sample_state(self, task_id: str, sample_id: str, state: str, *, reveal_done: bool) -> None:
        self._conn.execute(
            "INSERT INTO sample_states (task_id, sample_id, state, reveal_done) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(task_id, sample_id) DO UPDATE SET state = excluded.state, "
            "reveal_done = excluded.reveal_done",
            (task_id, sample_id, state, int(reveal_done)),
        )
        self._conn.commit()

    def sample_states(self, task_id: str) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT sample_id, state, reveal_done FROM sample_states WHERE task_id = ?",
            (task_id,),
        ).fetchall()
        return {
            row["sample_id"]: {"state": row["state"], "reveal_done": bool(row["reveal_done"])}
            for row in rows
        }

    # ── meta ─────────────────────────────────────────────────────────

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else None

    # ── lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def iter_rows(self, kind: str) -> Iterator[Mapping[str, Any]]:
        for row in self._conn.execute(
            "SELECT payload, content_hash FROM entities WHERE kind = ? ORDER BY id", (kind,)
        ):
            yield {"payload": row["payload"], "content_hash": row["content_hash"]}

    def verify_entity_hashes(self) -> dict[str, object]:
        """Recompute every stored entity's content hash.

        A database row whose payload no longer hashes to its stored value has
        been edited outside the contract layer — the exact thing the frozen
        entities exist to prevent. Cheap (one scan) and belongs in every
        read-back.
        """
        import json

        checked = 0
        bad: list[str] = []
        for row in self._conn.execute("SELECT kind, id, payload, content_hash FROM entities"):
            checked += 1
            payload = json.loads(row["payload"])
            if not verify_content_hash(payload, row["content_hash"]):
                bad.append(f"{row['kind']}/{row['id']}")
        return {"checked": checked, "corrupt": bad}
