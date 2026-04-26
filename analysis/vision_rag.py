# ── analysis/vision_rag.py ──
"""
Vision RAG Engine
=================
Cross-hospital similar case retrieval using morphology features.

Pipeline:
  1. YOLO detects organoids → bounding boxes
  2. SAM2 segments → pixel masks
  3. Extract morphology metrics (circularity, area, eccentricity, etc.)
  4. Encode morphology → query vector
  5. HNSW search across federated hospitals
  6. Collect anonymized diagnostic reports (NOT images)
  7. Local LLM generates structured diagnostic report

Key Design:
  - Hospitals share morphology feature vectors, NEVER patient images
  - Diagnostic reports are anonymized before sharing
  - Vector search runs on aggregated feature index
  - LLM generation is entirely local (no data leaves the querying hospital)

This is NOT text RAG — it's Vision-to-Text RAG:
  Image → Morphology → Similar Cases → Reports → Diagnosis
"""

import numpy as np
import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Optional
from collections import OrderedDict

from analysis.vector_engine import VectorEngine


# ── Morphology Encoding ──

MORPHOLOGY_FEATURES = [
    "area",              # pixel count
    "perimeter",         # boundary length
    "circularity",       # 4π·area/perimeter²
    "solidity",          # area/convex_area
    "aspect_ratio",      # major/minor axis
    "eccentricity",      # 0=circle, 1=line
    "n_organoids",       # count per image
    "avg_area",          # mean organoid area
    "std_area",          # area variance
    "class_distribution", # [p_healthy, p_early, p_late]
]

# Normalization ranges (typical organoid microscopy)
MORPHOLOGY_RANGES = {
    "area": (100, 50000),
    "perimeter": (50, 2000),
    "circularity": (0.0, 1.0),
    "solidity": (0.5, 1.0),
    "aspect_ratio": (0.3, 5.0),
    "eccentricity": (0.0, 1.0),
    "n_organoids": (1, 20),
    "avg_area": (100, 30000),
    "std_area": (0, 15000),
}


def encode_morphology(morphology: dict) -> np.ndarray:
    """Encode morphology metrics into a normalized feature vector.

    Args:
        morphology: dict with keys from MORPHOLOGY_FEATURES

    Returns:
        Normalized float32 vector
    """
    vec = []
    for feat in MORPHOLOGY_FEATURES:
        if feat == "class_distribution":
            # Already normalized (probabilities)
            dist = morphology.get(feat, [0.33, 0.33, 0.34])
            vec.extend(dist)
        else:
            val = morphology.get(feat, 0.0)
            lo, hi = MORPHOLOGY_RANGES.get(feat, (0, 1))
            # Min-max normalization to [0, 1]
            normalized = (val - lo) / max(hi - lo, 1e-8)
            normalized = np.clip(normalized, 0, 1)
            vec.append(normalized)

    return np.array(vec, dtype=np.float32)


def decode_morphology(vec: np.ndarray) -> dict:
    """Decode a normalized feature vector back to morphology dict."""
    morphology = {}
    idx = 0
    for feat in MORPHOLOGY_FEATURES:
        if feat == "class_distribution":
            dist = vec[idx:idx + 3].tolist()
            # Re-normalize to sum to 1
            total = sum(dist) or 1
            morphology[feat] = [d / total for d in dist]
            idx += 3
        else:
            normalized = vec[idx]
            lo, hi = MORPHOLOGY_RANGES.get(feat, (0, 1))
            morphology[feat] = normalized * (hi - lo) + lo
            idx += 1
    return morphology


# ── Case Record ──

@dataclass
class CaseRecord:
    """An anonymized patient case record.

    Contains morphology features and diagnostic report,
    but NEVER the actual patient image.
    """
    case_id: str
    hospital_id: str
    morphology: dict                    # Morphology metrics
    morphology_vector: np.ndarray       # Encoded feature vector
    diagnosis: str                      # e.g., "healthy", "early_stage", "late_stage"
    confidence: float                   # Model confidence
    report: str                         # Anonymized diagnostic report
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("morphology_vector")  # Don't serialize vector
        return d

    @staticmethod
    def anonymize_id(raw_id: str) -> str:
        """Hash patient ID for anonymization."""
        return hashlib.sha256(raw_id.encode()).hexdigest()[:16]


# ── Vision RAG Engine ──

class VisionRAG:
    """Cross-hospital similar case retrieval engine.

    Workflow:
    1. Hospitals register cases (morphology + report, NO images)
    2. Query: encode new morphology → search similar cases
    3. Aggregate reports from top-k similar cases
    4. Generate structured diagnostic summary
    """

    def __init__(self, vector_dim: Optional[int] = None):
        # Dimension: 9 scalar features + 3 class distribution = 12
        self.feature_dim = vector_dim or len(MORPHOLOGY_FEATURES) + 2  # +2 for class_dist expansion
        self.vector_db = VectorEngine(dimension=self.feature_dim)
        self.cases: dict[str, CaseRecord] = {}
        self._case_counter = 0

    def register_case(
        self,
        morphology: dict,
        diagnosis: str,
        confidence: float,
        report: str,
        hospital_id: str = "hospital_1",
        case_id: Optional[str] = None,
    ) -> str:
        """Register a new anonymized case.

        Args:
            morphology: dict of morphology metrics
            diagnosis: classification label
            confidence: model confidence
            report: anonymized diagnostic text
            hospital_id: source hospital identifier
            case_id: optional custom ID

        Returns:
            case_id string
        """
        if case_id is None:
            self._case_counter += 1
            case_id = f"case_{self._case_counter:06d}"

        vec = encode_morphology(morphology)
        # Pad or truncate to match vector_db dimension
        if len(vec) < self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:len(vec)] = vec
            vec = padded
        else:
            vec = vec[:self.feature_dim]

        case = CaseRecord(
            case_id=case_id,
            hospital_id=hospital_id,
            morphology=morphology,
            morphology_vector=vec,
            diagnosis=diagnosis,
            confidence=confidence,
            report=report,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        self.cases[case_id] = case
        self.vector_db.insert(case_id, vec, {"diagnosis": diagnosis, "hospital": hospital_id})

        return case_id

    def query(
        self,
        morphology: dict,
        k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[tuple[str, float, CaseRecord]]:
        """Find similar cases across all hospitals.

        Args:
            morphology: query morphology metrics
            k: number of similar cases to retrieve
            min_similarity: minimum cosine similarity threshold

        Returns:
            List of (case_id, similarity, case_record) sorted by similarity
        """
        vec = encode_morphology(morphology)
        if len(vec) < self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:len(vec)] = vec
            vec = padded
        else:
            vec = vec[:self.feature_dim]

        results = self.vector_db.search(vec, k=k)

        similar = []
        for case_id, similarity in results:
            if similarity < min_similarity:
                continue
            if case_id in self.cases:
                similar.append((case_id, similarity, self.cases[case_id]))

        return similar

    def generate_report(
        self,
        query_morphology: dict,
        similar_cases: list[tuple[str, float, CaseRecord]],
    ) -> dict:
        """Generate a structured diagnostic summary from similar cases.

        This runs entirely LOCAL — no data leaves the querying hospital.

        Args:
            query_morphology: the query case morphology
            similar_cases: results from query()

        Returns:
            Structured report dict
        """
        if not similar_cases:
            return {
                "status": "no_similar_cases",
                "message": "No similar cases found in the federated database.",
                "recommendation": "Consult specialist review.",
            }

        # Aggregate statistics from similar cases
        diagnoses = [c.diagnosis for _, _, c in similar_cases]
        confidences = [c.confidence for _, _, c in similar_cases]
        hospitals = set(c.hospital_id for _, _, c in similar_cases)

        # Majority vote
        from collections import Counter
        diagnosis_counts = Counter(diagnoses)
        most_common = diagnosis_counts.most_common(1)[0]

        # Weighted by similarity
        weighted_diagnoses = {}
        for case_id, sim, case in similar_cases:
            d = case.diagnosis
            weighted_diagnoses[d] = weighted_diagnoses.get(d, 0) + sim

        top_diagnosis = max(weighted_diagnoses, key=weighted_diagnoses.get)

        # Build report
        report = {
            "status": "success",
            "query_morphology": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in query_morphology.items()
                if k != "class_distribution"
            },
            "similar_cases_found": len(similar_cases),
            "hospitals_consulted": len(hospitals),
            "top_diagnosis": top_diagnosis,
            "diagnosis_confidence": round(
                weighted_diagnoses[top_diagnosis] / sum(weighted_diagnoses.values()), 3
            ),
            "diagnosis_breakdown": dict(diagnosis_counts),
            "avg_model_confidence": round(np.mean(confidences), 3),
            "recommendations": self._generate_recommendations(
                query_morphology, top_diagnosis, similar_cases
            ),
            "similar_case_ids": [cid for cid, _, _ in similar_cases],
        }

        return report

    def _generate_recommendations(
        self,
        morphology: dict,
        diagnosis: str,
        similar_cases: list,
    ) -> list[str]:
        """Generate clinical recommendations based on morphology and similar cases."""
        recs = []

        circularity = morphology.get("circularity", 0.5)
        eccentricity = morphology.get("eccentricity", 0.0)
        solidity = morphology.get("solidity", 0.8)
        n_orgs = morphology.get("n_organoids", 1)

        if diagnosis == "late_stage":
            recs.append("⚠️ Late-stage morphology detected: consider drug susceptibility testing.")
            if eccentricity > 0.6:
                recs.append("High eccentricity suggests irregular growth — monitor closely.")
        elif diagnosis == "early_stage":
            recs.append("Early-stage morphology: continue current culture conditions.")
            if circularity < 0.5:
                recs.append("Low circularity may indicate early stress response.")
        else:
            recs.append("Healthy morphology confirmed: organoids suitable for downstream assays.")

        if solidity < 0.7:
            recs.append("Low solidity indicates concave boundaries — possible budding or fragmentation.")

        if n_orgs > 10:
            recs.append(f"High organoid count ({n_orgs}): verify seeding density is appropriate.")

        # Add insights from similar cases
        if similar_cases:
            top_reports = [c.report for _, _, c in similar_cases[:3] if c.report]
            if top_reports:
                recs.append(f"Based on {len(top_reports)} similar cases from federated database.")

        return recs

    def get_stats(self) -> dict:
        """Get database statistics."""
        hospital_counts = {}
        diagnosis_counts = {}
        for case in self.cases.values():
            hospital_counts[case.hospital_id] = hospital_counts.get(case.hospital_id, 0) + 1
            diagnosis_counts[case.diagnosis] = diagnosis_counts.get(case.diagnosis, 0) + 1

        return {
            "total_cases": len(self.cases),
            "hospitals": len(hospital_counts),
            "hospital_distribution": hospital_counts,
            "diagnosis_distribution": diagnosis_counts,
            "vector_db_size": len(self.vector_db),
        }

    def populate_demo(self, n_cases: int = 100, n_hospitals: int = 3, seed: int = 42):
        """Populate with synthetic demo cases for testing."""
        rng = np.random.RandomState(seed)
        diagnoses = ["healthy", "early_stage", "late_stage"]
        reports = {
            "healthy": "Organoid morphology consistent with healthy controls. Regular boundaries, uniform size distribution.",
            "early_stage": "Early morphological changes detected. Slightly irregular boundaries, mild size variation.",
            "late_stage": "Significant morphological changes. Highly irregular boundaries, heterogeneous size distribution.",
        }

        for i in range(n_cases):
            diag = rng.choice(diagnoses)
            morphology = {
                "area": rng.uniform(500, 20000),
                "perimeter": rng.uniform(100, 1000),
                "circularity": rng.uniform(0.3, 0.95) if diag == "healthy" else rng.uniform(0.1, 0.6),
                "solidity": rng.uniform(0.7, 0.98) if diag == "healthy" else rng.uniform(0.4, 0.8),
                "aspect_ratio": rng.uniform(0.8, 1.5) if diag == "healthy" else rng.uniform(0.5, 3.0),
                "eccentricity": rng.uniform(0.0, 0.4) if diag == "healthy" else rng.uniform(0.3, 0.9),
                "n_organoids": rng.randint(1, 15),
                "avg_area": rng.uniform(1000, 15000),
                "std_area": rng.uniform(500, 5000),
                "class_distribution": [
                    0.7 if diag == "healthy" else 0.1,
                    0.2 if diag == "healthy" else (0.3 if diag == "early_stage" else 0.1),
                    0.1 if diag == "healthy" else (0.6 if diag == "late_stage" else 0.3),
                ],
            }

            self.register_case(
                morphology=morphology,
                diagnosis=diag,
                confidence=rng.uniform(0.6, 0.99),
                report=reports[diag],
                hospital_id=f"hospital_{(i % n_hospitals) + 1}",
            )
