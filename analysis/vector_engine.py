# ── analysis/vector_engine.py ──
"""
Vector Search Engine
====================
In-memory vector search with cosine similarity (Python fallback).
Provides the same interface as the Rust HNSW VectorDB for demo purposes.
"""

import numpy as np
from typing import Optional
from collections import OrderedDict


class VectorEngine:
    """In-memory vector search engine.

    For production, replace with Rust HNSW VectorDB via gRPC.
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.vectors: OrderedDict[str, np.ndarray] = OrderedDict()
        self.metadata: dict[str, dict] = {}

    def insert(self, id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        """Insert a vector."""
        if len(vector) != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {len(vector)}")
        self.vectors[id] = vector.astype(np.float32)
        self.metadata[id] = metadata or {}

    def bulk_insert(self, ids: list[str], vectors: np.ndarray, metadata: Optional[list[dict]] = None) -> int:
        """Insert multiple vectors."""
        count = 0
        for i, vid in enumerate(ids):
            meta = metadata[i] if metadata else {}
            self.insert(vid, vectors[i], meta)
            count += 1
        return count

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Search for k nearest neighbors using cosine similarity.

        Returns:
            List of (id, similarity_score) sorted by descending similarity.
        """
        if len(self.vectors) == 0:
            return []

        query = query.astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        results = []
        for vid, vec in self.vectors.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            similarity = float(np.dot(query, vec) / (query_norm * vec_norm))
            results.append((vid, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by IDs."""
        count = 0
        for vid in ids:
            if vid in self.vectors:
                del self.vectors[vid]
                self.metadata.pop(vid, None)
                count += 1
        return count

    def __len__(self) -> int:
        return len(self.vectors)

    def get_stats(self) -> dict:
        return {
            "total_vectors": len(self),
            "dimension": self.dimension,
        }
