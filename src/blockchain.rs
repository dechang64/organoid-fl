//! Blockchain-style immutable audit log for VectorDB operations.
//!
//! Each operation is recorded as a "block" with:
//! - Timestamp
//! - Operation type and details
//! - Previous block's hash (chain integrity)
//! - Current block's hash (SHA-256)
//!
//! This provides tamper-evident logging for compliance and auditability.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;

/// Types of operations that can be logged
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OperationType {
    /// Chain initialisation. Present as its own variant rather than reusing
    /// `System` because the canonical hash uses the operation *name*, and
    /// `Block::verify` recomputes that name from the stored operation — so the
    /// name has to be recoverable from the variant. An earlier version hashed
    /// genesis under the literal "genesis" while storing `System`, which made
    /// the genesis block fail its own integrity check.
    Genesis,
    Insert { count: usize },
    Search { k: usize, result_count: usize },
    Delete { count: usize },
    System { message: String },
}

impl OperationType {
    /// Canonical operation name used in the audit preimage.
    ///
    /// This must stay in step with `analysis/audit_spec.py`. The previous
    /// implementation hashed the `Debug` rendering of this enum instead
    /// (`Insert { count: 10 }`), which no other language can reproduce.
    pub fn name(&self) -> &'static str {
        match self {
            OperationType::Genesis => "genesis",
            OperationType::Insert { .. } => "insert",
            OperationType::Search { .. } => "search",
            OperationType::Delete { .. } => "delete",
            OperationType::System { .. } => "system",
        }
    }

    /// Canonical payload, hashed as the `details` field.
    ///
    /// Returned as a `serde_json::Value` so the encoder — not this method —
    /// decides key order. `serde_json` without the `preserve_order` feature
    /// backs objects with a `BTreeMap`, which sorts keys, matching the
    /// `sort_keys=True` used by the Python implementation. Enabling
    /// `preserve_order` would silently break cross-language verification.
    pub fn details(&self) -> serde_json::Value {
        match self {
            OperationType::Genesis => serde_json::json!({}),
            OperationType::Insert { count } => serde_json::json!({ "count": *count }),
            OperationType::Search { k, result_count } => {
                serde_json::json!({ "k": *k, "result_count": *result_count })
            }
            OperationType::Delete { count } => serde_json::json!({ "count": *count }),
            OperationType::System { message } => {
                serde_json::json!({ "message": message })
            }
        }
    }
}

/// A single block in the audit chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Block index (monotonically increasing)
    pub index: u64,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Operation type
    pub operation: OperationType,
    /// Hash of the previous block ("0" for genesis)
    pub prev_hash: String,
    /// Hash of this block
    pub hash: String,
    /// Optional nonce for future proof-of-work extension
    pub nonce: u64,
}

impl Block {
    /// Canonical audit preimage: compact JSON, object keys sorted.
    ///
    /// Field order in the source is irrelevant — the encoder sorts — but the
    /// field *set* is part of the specification and must match
    /// `HASH_FIELDS` in `analysis/audit_spec.py` exactly:
    /// `details`, `index`, `nonce`, `operation`, `prev_hash`, `timestamp`.
    ///
    /// `serde_json::to_string` emits compact JSON with no insignificant
    /// whitespace and writes non-ASCII as raw UTF-8, matching the Python side's
    /// `separators=(",", ":")` and `ensure_ascii=False`.
    fn canonical_preimage(
        index: u64,
        timestamp: &str,
        operation_name: &str,
        details: &serde_json::Value,
        prev_hash: &str,
        nonce: u64,
    ) -> String {
        let payload = serde_json::json!({
            "details": details,
            "index": index,
            "nonce": nonce,
            "operation": operation_name,
            "prev_hash": prev_hash,
            "timestamp": timestamp,
        });
        serde_json::to_string(&payload).expect("audit payload is always serialisable")
    }

    /// Compute the hash of this block's content.
    fn compute_hash(index: u64, timestamp: &str, operation: &OperationType, prev_hash: &str, nonce: u64) -> String {
        Self::compute_hash_parts(
            index,
            timestamp,
            operation.name(),
            &operation.details(),
            prev_hash,
            nonce,
        )
    }

    /// Hash an already-decomposed operation.
    fn compute_hash_parts(
        index: u64,
        timestamp: &str,
        operation_name: &str,
        details: &serde_json::Value,
        prev_hash: &str,
        nonce: u64,
    ) -> String {
        let data = Self::canonical_preimage(
            index, timestamp, operation_name, details, prev_hash, nonce,
        );
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Create the genesis block
    pub fn genesis() -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let operation = OperationType::Genesis;
        let hash = Self::compute_hash(0, &timestamp, &operation, "0", 0);
        Self {
            index: 0,
            timestamp,
            operation,
            prev_hash: "0".to_string(),
            hash,
            nonce: 0,
        }
    }

    /// Create a new block linked to the previous one
    pub fn new(index: u64, operation: OperationType, prev_hash: &str) -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let hash = Self::compute_hash(index, &timestamp, &operation, prev_hash, 0);
        Self {
            index,
            timestamp,
            operation,
            prev_hash: prev_hash.to_string(),
            hash,
            nonce: 0,
        }
    }

    /// Verify this block's hash integrity
    pub fn verify(&self) -> bool {
        let computed = Self::compute_hash(
            self.index,
            &self.timestamp,
            &self.operation,
            &self.prev_hash,
            self.nonce,
        );
        computed == self.hash
    }
}

/// The blockchain audit log
pub struct AuditChain {
    blocks: VecDeque<Block>,
    /// Maximum blocks to keep in memory
    max_blocks: usize,
    /// Optional file path for persistence
    log_path: Option<PathBuf>,
}

impl AuditChain {
    /// Create a new audit chain with a genesis block
    pub fn new(max_blocks: usize) -> Self {
        let mut chain = Self {
            blocks: VecDeque::with_capacity(max_blocks),
            max_blocks,
            log_path: None,
        };
        chain.blocks.push_back(Block::genesis());
        chain
    }

    /// Set the file path for persistent logging
    pub fn with_log_file(mut self, path: PathBuf) -> Self {
        self.log_path = Some(path);
        self
    }

    /// Append a new operation to the chain
    pub fn append(&mut self, operation: OperationType) -> &Block {
        let prev = self.blocks.back().expect("chain always has genesis");
        let index = prev.index + 1;
        let block = Block::new(index, operation, &prev.hash);

        // Persist to file if configured
        if let Some(ref path) = self.log_path {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(file, "{}", serde_json::to_string(&block).unwrap_or_default());
            }
        }

        // Evict oldest if over capacity
        if self.blocks.len() >= self.max_blocks {
            self.blocks.pop_front();
        }

        self.blocks.push_back(block);
        self.blocks.back().expect("just pushed")
    }

    /// Verify the integrity of the entire chain
    pub fn verify_chain(&self) -> bool {
        let blocks: Vec<&Block> = self.blocks.iter().collect();
        if blocks.is_empty() {
            return true;
        }

        // Verify genesis
        if !blocks[0].verify() {
            return false;
        }

        // Verify each subsequent block
        for window in blocks.windows(2) {
            if !window[1].verify() {
                return false;
            }
            if window[1].prev_hash != window[0].hash {
                return false;
            }
        }

        true
    }

    /// Get the latest N blocks (most recent first)
    pub fn recent(&self, n: usize) -> Vec<&Block> {
        self.blocks
            .iter()
            .rev()
            .take(n)
            .collect()
    }

    /// Get all blocks
    pub fn all(&self) -> &VecDeque<Block> {
        &self.blocks
    }

    /// Get the total number of blocks
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if chain only has genesis
    pub fn is_empty(&self) -> bool {
        self.blocks.len() <= 1
    }

    /// Get the latest block's hash
    pub fn latest_hash(&self) -> &str {
        &self.blocks.back().expect("chain always has genesis").hash
    }

    /// Load chain from a log file (appends to existing in-memory chain)
    pub fn load_from_file(&mut self, path: &PathBuf) -> Result<usize, String> {
        if !path.exists() {
            return Ok(0);
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read log file: {}", e))?;

        let mut count = 0;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(block) = serde_json::from_str::<Block>(line) {
                // Skip genesis if we already have one
                if block.index == 0 && !self.blocks.is_empty() {
                    continue;
                }
                self.blocks.push_back(block);
                count += 1;
            }
        }

        // Trim to max_blocks
        while self.blocks.len() > self.max_blocks {
            self.blocks.pop_front();
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_block() {
        let block = Block::genesis();
        assert_eq!(block.index, 0);
        assert_eq!(block.prev_hash, "0");
        assert!(block.verify());
    }

    #[test]
    fn test_chain_integrity() {
        let mut chain = AuditChain::new(100);
        chain.append(OperationType::Insert { count: 10 });
        chain.append(OperationType::Search { k: 5, result_count: 5 });
        chain.append(OperationType::Delete { count: 2 });

        assert!(chain.verify_chain());
        assert_eq!(chain.len(), 4); // genesis + 3 operations
    }

    #[test]
    fn test_recent_blocks() {
        let mut chain = AuditChain::new(100);
        for i in 0..10 {
            chain.append(OperationType::Insert { count: i });
        }

        let recent = chain.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].index, 10); // most recent
    }

    #[test]
    fn test_max_blocks_eviction() {
        let mut chain = AuditChain::new(5);
        for i in 0..10 {
            chain.append(OperationType::Insert { count: i });
        }

        // Should have at most 5 blocks
        assert!(chain.len() <= 5);
    }

    #[test]
    fn test_genesis_hashes_under_its_own_name() {
        // Regression guard: genesis used to store `System` while hashing the
        // literal "genesis", so verify() recomputed a different name and the
        // genesis block failed its own integrity check.
        let block = Block::genesis();
        assert_eq!(block.operation.name(), "genesis");
        assert!(block.verify());
    }

    /// Cross-language parity: reproduce every vector in the shared fixture.
    ///
    /// This is the test that would have caught the original bug. The Python
    /// implementation (`analysis/audit_spec.py`), this Rust implementation and
    /// `tools/verify_audit_vectors.mjs` all read the same
    /// `tests/fixtures/audit_hash_vectors.json`; if any side changes its
    /// preimage, its run fails rather than silently disagreeing.
    #[test]
    fn test_shared_audit_vectors_are_reproduced() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("audit_hash_vectors.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let fixture: serde_json::Value =
            serde_json::from_str(&raw).expect("fixture is not valid JSON");

        let vectors = fixture["vectors"]
            .as_array()
            .expect("fixture has no `vectors` array");
        assert!(!vectors.is_empty(), "fixture contains no vectors");

        for v in vectors {
            let name = v["name"].as_str().expect("vector has no name");
            let index = v["index"].as_u64().expect("index must fit u64");
            let nonce = v["nonce"].as_u64().unwrap_or(0);
            let timestamp = v["timestamp"].as_str().expect("timestamp must be a string");
            let operation = v["operation"].as_str().expect("operation must be a string");
            let prev_hash = v["prev_hash"].as_str().expect("prev_hash must be a string");

            // Round-trip through the canonical preimage and compare the text
            // first: when a digest mismatches, knowing whether the bytes or the
            // hash differed is what makes the failure diagnosable.
            let preimage = Block::canonical_preimage(
                index, timestamp, operation, &v["details"], prev_hash, nonce,
            );
            assert_eq!(
                preimage,
                v["preimage"].as_str().unwrap(),
                "preimage mismatch for vector `{name}`"
            );

            let digest = Block::compute_hash_parts(
                index, timestamp, operation, &v["details"], prev_hash, nonce,
            );
            assert_eq!(
                digest,
                v["sha256"].as_str().unwrap(),
                "hash mismatch for vector `{name}`"
            );
        }
    }

    /// The preimage must be compact, sorted-key JSON with raw UTF-8 — the
    /// three properties that make it portable. Each is asserted separately so a
    /// regression points at the encoder rather than at the whole spec.
    #[test]
    fn test_preimage_is_canonical_json() {
        let details = serde_json::json!({ "label": "类器官", "b": 1, "a": 2 });
        let preimage = Block::canonical_preimage(1, "t", "op", &details, "0", 0);

        assert!(!preimage.contains(", "), "preimage contains whitespace");
        assert!(!preimage.contains(": "), "preimage contains whitespace");
        assert!(!preimage.contains('\n'), "preimage contains a newline");
        assert!(
            preimage.contains("类器官"),
            "non-ASCII was escaped; serde_json writes raw UTF-8 and the Python \
             side uses ensure_ascii=False, so escaping here breaks parity"
        );
        assert!(
            preimage.starts_with("{\"details\":"),
            "keys are not sorted; got {preimage}"
        );
    }
}
