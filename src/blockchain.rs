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
    Insert { count: usize },
    Search { k: usize, result_count: usize },
    Delete { count: usize },
    System { message: String },
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
    /// Compute the hash of this block's content.
    fn compute_hash(index: u64, timestamp: &str, operation: &OperationType, prev_hash: &str, nonce: u64) -> String {
        let data = format!("{}:{}:{:?}:{}:{}", index, timestamp, operation, prev_hash, nonce);
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Create the genesis block
    pub fn genesis() -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let operation = OperationType::System {
            message: "Organoid VectorDB audit chain initialized".to_string(),
        };
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
}
