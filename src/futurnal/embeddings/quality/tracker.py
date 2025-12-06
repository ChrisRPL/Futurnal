"""Quality Metrics Tracker.

Tracks embedding quality over time with quality score computation.
Follows existing EmbeddingMetrics thread-safety patterns.

Computes quality scores using weighted combination of:
- Extraction confidence (40%)
- Embedding coherence (30%)
- Golden embedding similarity (20%)
- Temporal quality (10%, for events only)

Option B Compliance:
- Quality metrics tracked for experiential learning integration
- Re-embedding triggers based on quality thresholds
- Temporal quality for Event entities (Option B temporal-first)

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

from __future__ import annotations

import logging
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from futurnal.embeddings.quality.metrics import EmbeddingQualityMetrics
from futurnal.embeddings.quality.store import QualityMetricsStore
from futurnal.embeddings.quality.golden import GoldenEmbeddingsManager

if TYPE_CHECKING:
    from futurnal.embeddings.models import TemporalEmbeddingContext

logger = logging.getLogger(__name__)


class QualityMetricsTracker:
    """Tracks embedding quality over time.

    Thread-safe tracker for embedding quality metrics with:
    - Quality score computation using weighted combination
    - Low-quality embedding identification
    - Quality trend tracking
    - Integration with golden embeddings for reference comparison

    Quality Score Weights (from production plan):
    - Extraction confidence: 40% (what extraction pipeline reports)
    - Embedding coherence: 30% (vector quality: norm, validity, distribution)
    - Golden similarity: 20% (comparison to reference embeddings)
    - Temporal quality: 10% (for Event entities only)

    Example:
        store = QualityMetricsStore(db_path=Path("~/.futurnal/quality_metrics.db"))
        tracker = QualityMetricsTracker(store=store)

        # Record quality metrics
        metrics = tracker.record_embedding_quality(
            embedding_id="emb_123",
            entity_id="ent_456",
            entity_type="Person",
            embedding=embedding_vector,
            extraction_confidence=0.85,
            embedding_latency_ms=150.0,
            model_id="instructor-large-entity",
            vector_dimension=768,
        )

        print(f"Overall quality: {metrics.overall_quality_score:.2f}")

        # Find low-quality embeddings
        low_quality = tracker.identify_low_quality_embeddings(
            min_quality_score=0.6,
            limit=100,
        )
    """

    # Default weights from production plan
    DEFAULT_WEIGHTS = {
        "extraction": 0.4,
        "coherence": 0.3,
        "golden": 0.2,
        "temporal": 0.1,
    }

    def __init__(
        self,
        store: QualityMetricsStore,
        golden_manager: Optional[GoldenEmbeddingsManager] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize quality metrics tracker.

        Args:
            store: SQLite-backed store for persistence
            golden_manager: Optional manager for golden embeddings
            weights: Optional custom weights for quality computation
        """
        self._store = store
        self._golden_manager = golden_manager or GoldenEmbeddingsManager()
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._lock = Lock()

        # Validate weights sum to ~1.0
        weight_sum = sum(self._weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                f"Quality weights sum to {weight_sum:.3f}, expected 1.0. "
                "Weights will be normalized."
            )

        logger.info(
            f"Initialized QualityMetricsTracker with weights: {self._weights}"
        )

    def record_embedding_quality(
        self,
        embedding_id: str,
        entity_id: str,
        entity_type: str,
        embedding: np.ndarray,
        extraction_confidence: float,
        embedding_latency_ms: float,
        model_id: str,
        vector_dimension: int,
        temporal_context: Optional["TemporalEmbeddingContext"] = None,
    ) -> EmbeddingQualityMetrics:
        """Record quality metrics for new embedding.

        Computes all quality sub-scores and stores metrics.
        Thread-safe via internal lock.

        Args:
            embedding_id: Unique embedding identifier
            entity_id: PKG entity ID
            entity_type: Type of entity (Person, Event, etc.)
            embedding: The embedding vector
            extraction_confidence: Confidence from extraction pipeline
            embedding_latency_ms: Time to generate embedding
            model_id: Model used for embedding
            vector_dimension: Dimension of embedding vector
            temporal_context: Optional temporal context for Events

        Returns:
            EmbeddingQualityMetrics with computed quality scores
        """
        with self._lock:
            # Convert to numpy array if needed
            embedding = np.asarray(embedding, dtype=np.float32)

            # Compute coherence
            coherence = self._compute_coherence(embedding)

            # Compute golden similarity if golden embeddings available
            golden_similarity = None
            if entity_type in self._golden_manager.supported_types:
                golden_similarity = self._compute_golden_similarity(
                    embedding, entity_type
                )

            # Compute temporal quality for events
            temporal_accuracy = None
            causal_pattern_quality = None
            if entity_type == "Event" and temporal_context is not None:
                temporal_accuracy = self._compute_temporal_quality(
                    embedding, temporal_context
                )
                # Causal pattern quality based on causal chain presence
                if temporal_context.causal_chain:
                    causal_pattern_quality = min(1.0, 0.7 + len(temporal_context.causal_chain) * 0.1)

            # Compute overall quality score
            overall_score = self.compute_quality_score(
                embedding=embedding,
                entity_type=entity_type,
                extraction_confidence=extraction_confidence,
                coherence=coherence,
                golden_similarity=golden_similarity,
                temporal_accuracy=temporal_accuracy,
            )

            # Create metrics object
            metrics = EmbeddingQualityMetrics(
                embedding_id=embedding_id,
                entity_id=entity_id,
                extraction_confidence=extraction_confidence,
                embedding_coherence=coherence,
                embedding_similarity_score=golden_similarity,
                temporal_accuracy=temporal_accuracy,
                causal_pattern_quality=causal_pattern_quality,
                embedding_latency_ms=embedding_latency_ms,
                model_id=model_id,
                vector_dimension=vector_dimension,
                overall_quality_score=overall_score,
                created_at=datetime.utcnow(),
                last_validated=datetime.utcnow(),
            )

            # Persist to store
            self._store.insert(metrics)

            logger.debug(
                f"Recorded quality metrics for {embedding_id}: "
                f"overall={overall_score:.3f}, coherence={coherence:.3f}"
            )

            return metrics

    def compute_quality_score(
        self,
        embedding: np.ndarray,
        entity_type: str,
        extraction_confidence: float,
        coherence: Optional[float] = None,
        golden_similarity: Optional[float] = None,
        temporal_accuracy: Optional[float] = None,
    ) -> float:
        """Compute overall quality score with weighted combination.

        Weights (from production plan):
        - Extraction confidence: 40%
        - Embedding coherence: 30%
        - Golden similarity: 20%
        - Temporal quality: 10% (for events only)

        When components are missing, weights are renormalized among
        available components.

        Args:
            embedding: The embedding vector
            entity_type: Type of entity
            extraction_confidence: Confidence from extraction
            coherence: Optional pre-computed coherence score
            golden_similarity: Optional pre-computed golden similarity
            temporal_accuracy: Optional pre-computed temporal quality

        Returns:
            Weighted quality score between 0.0 and 1.0
        """
        scores = []
        weights = []

        # Component 1: Extraction confidence (always present)
        scores.append(extraction_confidence)
        weights.append(self._weights["extraction"])

        # Component 2: Coherence
        if coherence is not None:
            scores.append(coherence)
            weights.append(self._weights["coherence"])
        else:
            # Compute if not provided
            computed_coherence = self._compute_coherence(np.asarray(embedding))
            scores.append(computed_coherence)
            weights.append(self._weights["coherence"])

        # Component 3: Golden similarity
        if golden_similarity is not None:
            scores.append(golden_similarity)
            weights.append(self._weights["golden"])

        # Component 4: Temporal quality (for Events only)
        if entity_type == "Event" and temporal_accuracy is not None:
            scores.append(temporal_accuracy)
            weights.append(self._weights["temporal"])

        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in weights]
        else:
            # Fallback to equal weights
            normalized_weights = [1.0 / len(weights)] * len(weights)

        # Compute weighted sum
        quality_score = sum(s * w for s, w in zip(scores, normalized_weights))

        return min(1.0, max(0.0, quality_score))

    def _compute_coherence(self, embedding: np.ndarray) -> float:
        """Compute embedding coherence.

        Measures internal quality of embedding vector:
        1. Vector norm (should be ~1.0 if normalized)
        2. Validity (no NaN/Inf)
        3. Distribution (non-constant, reasonable variance)

        Args:
            embedding: The embedding vector

        Returns:
            Coherence score between 0.0 and 1.0
        """
        embedding = np.asarray(embedding, dtype=np.float32)

        # Check 1: Vector norm (should be ~1.0 for normalized embeddings)
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) < 0.01:
            norm_score = 1.0
        elif abs(norm - 1.0) < 0.1:
            norm_score = 0.8
        elif norm > 0:
            norm_score = 0.5
        else:
            norm_score = 0.0

        # Check 2: Validity (no NaN or Inf values)
        if np.isfinite(embedding).all():
            validity_score = 1.0
        else:
            validity_score = 0.0

        # Check 3: Distribution quality
        # Embedding should have reasonable variance (not all zeros or constant)
        std_dev = np.std(embedding)
        if std_dev < 0.001:
            distribution_score = 0.0  # Essentially constant
        elif std_dev < 0.01:
            distribution_score = 0.3
        elif std_dev < 0.1:
            distribution_score = 0.7
        else:
            distribution_score = min(1.0, std_dev * 5)  # Cap at reasonable level

        # Combined coherence score
        coherence = (norm_score + validity_score + distribution_score) / 3

        return coherence

    def _compute_golden_similarity(
        self,
        embedding: np.ndarray,
        entity_type: str,
    ) -> float:
        """Compute cosine similarity to golden reference embeddings.

        Returns max similarity among golden embeddings (best match).

        Args:
            embedding: The embedding vector
            entity_type: Type of entity

        Returns:
            Max cosine similarity to golden embeddings (0.0-1.0),
            or 0.7 (neutral) if no golden embeddings exist.
        """
        golden_embeddings = self._golden_manager.get_golden_embeddings(entity_type)

        if not golden_embeddings:
            return 0.7  # Neutral score when no golden embeddings available

        embedding = np.asarray(embedding, dtype=np.float32)

        # Compute cosine similarity to each golden embedding
        similarities = []
        for golden in golden_embeddings:
            golden = np.asarray(golden, dtype=np.float32)

            # Cosine similarity
            dot_product = np.dot(embedding, golden)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(golden)

            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                # Convert from [-1, 1] to [0, 1]
                similarity = (similarity + 1.0) / 2.0
                similarities.append(similarity)

        return max(similarities) if similarities else 0.7

    def _compute_temporal_quality(
        self,
        embedding: np.ndarray,
        temporal_context: "TemporalEmbeddingContext",
    ) -> float:
        """Compute temporal quality for event embeddings.

        Validates that temporal context is properly encoded by checking
        context completeness.

        Args:
            embedding: The embedding vector
            temporal_context: Temporal context from the event

        Returns:
            Temporal quality score between 0.0 and 1.0
        """
        # Base temporal quality (timestamp is required)
        base_score = 0.6

        # Boost for complete temporal context
        if temporal_context.duration is not None:
            base_score += 0.1

        if temporal_context.temporal_type is not None:
            base_score += 0.1

        if temporal_context.causal_chain:
            base_score += 0.1

        if temporal_context.event_sequence:
            base_score += 0.05

        if temporal_context.temporal_neighbors:
            base_score += 0.05

        return min(1.0, base_score)

    def identify_low_quality_embeddings(
        self,
        min_quality_score: float = 0.6,
        limit: int = 100,
    ) -> List[str]:
        """Identify entity IDs with embeddings below quality threshold.

        Queries the store for embeddings with low quality scores.
        Target precision: >90%

        Args:
            min_quality_score: Minimum acceptable quality score (default: 0.6)
            limit: Maximum number of entity IDs to return

        Returns:
            List of unique entity IDs needing re-embedding
        """
        return self._store.query_low_quality(min_quality_score, limit)

    def get_quality_summary(self, entity_id: str) -> Dict[str, Any]:
        """Get quality summary for an entity.

        Args:
            entity_id: PKG entity ID

        Returns:
            Dictionary with quality summary including latest scores and trend
        """
        metrics_list = self._store.get_by_entity_id(entity_id)

        if not metrics_list:
            return {
                "entity_id": entity_id,
                "has_metrics": False,
            }

        latest = metrics_list[-1]
        return {
            "entity_id": entity_id,
            "has_metrics": True,
            "latest_quality_score": latest.overall_quality_score,
            "extraction_confidence": latest.extraction_confidence,
            "embedding_coherence": latest.embedding_coherence,
            "golden_similarity": latest.embedding_similarity_score,
            "temporal_accuracy": latest.temporal_accuracy,
            "quality_trend": latest.quality_trend,
            "reembedding_count": latest.reembedding_count,
            "metrics_count": len(metrics_list),
            "last_validated": latest.last_validated.isoformat() if latest.last_validated else None,
        }

    def update_quality_trend(self, entity_id: str, lookback_days: int = 30) -> Optional[str]:
        """Update quality trend for an entity based on historical data.

        Args:
            entity_id: PKG entity ID
            lookback_days: Number of days to analyze

        Returns:
            Trend string ("improving", "stable", "degrading") or None
        """
        metrics = self._store.get_entity_history(entity_id, lookback_days)

        if len(metrics) < 2:
            return None  # Not enough data

        # Compute trend from extraction confidence over time
        qualities = [m.extraction_confidence for m in metrics]
        trend_value = qualities[-1] - qualities[0]

        if trend_value > 0.05:
            trend = "improving"
        elif trend_value < -0.05:
            trend = "degrading"
        else:
            trend = "stable"

        # Update the latest metrics record
        latest = metrics[-1]
        self._store.update(
            latest.embedding_id,
            {
                "quality_trend": trend,
                "last_validated": datetime.utcnow().isoformat(),
            },
        )

        return trend

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall quality tracking statistics.

        Returns:
            Dictionary with store statistics and golden embedding counts
        """
        store_stats = self._store.get_statistics()
        golden_stats = self._golden_manager.get_statistics()

        return {
            **store_stats,
            "golden_embeddings": golden_stats,
            "quality_weights": self._weights,
        }
