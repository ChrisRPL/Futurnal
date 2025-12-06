"""Embedding Quality Evolution Module.

Integrates embedding quality with experiential learning pipeline.
As extraction quality improves through Ghost→Animal evolution,
re-embed low-quality entities for continuous improvement.

Option B Compliance (Critical):
- Embedding quality improves over time via experiential learning
- Re-embed low-quality extractions when extraction pipeline improves
- Integration with TrainingFreeGRPO/WorldStateModel
- No model parameter updates (frozen Ghost models)

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from futurnal.embeddings.reembedding import ReembeddingProgress, ReembeddingService
    from futurnal.embeddings.quality.tracker import QualityMetricsTracker

logger = logging.getLogger(__name__)


class ExperientialLearningMonitor(Protocol):
    """Protocol for experiential learning monitor integration.

    Matches WorldStateModel interface from extraction.schema.experiential
    for integration with Ghost→Animal evolution.

    This protocol defines what EmbeddingQualityEvolution needs from
    the experiential learning system to trigger quality-based re-embedding.
    """

    def get_current_quality_metrics(self) -> Dict[str, float]:
        """Get current extraction quality metrics.

        Returns:
            Dictionary with keys like "precision", "recall", "f1"
        """
        ...

    def get_baseline_quality(self) -> Dict[str, float]:
        """Get baseline extraction quality metrics.

        Returns:
            Dictionary with baseline quality metrics for comparison
        """
        ...

    def assess_extraction_trajectory(
        self,
        recent_extractions: List[Any],
    ) -> Dict[str, float]:
        """Assess if extraction quality is improving.

        Args:
            recent_extractions: List of recent extraction results

        Returns:
            Dictionary with trajectory metrics including "improvement"
        """
        ...


class EmbeddingQualityEvolution:
    """Integrates embedding quality with experiential learning pipeline.

    Key Innovation (Option B):
    - Re-embed low-quality entities when extraction quality improves
    - Quality evolution without model parameter updates
    - Ghost→Animal evolution via experiential knowledge

    Workflow:
    1. Monitor extraction pipeline quality (via ExperientialLearningMonitor)
    2. When quality improves >5%, identify old low-quality embeddings
    3. Trigger re-embedding for those entities
    4. Track quality trends over time

    Example:
        evolution = EmbeddingQualityEvolution(
            quality_tracker=tracker,
            reembedding_service=reembedding_service,
            experiential_learning_monitor=world_state_model,
        )

        # Check for quality improvements and trigger re-embedding
        progress = evolution.monitor_extraction_quality_improvement()

        # Or manually trigger re-embedding for low quality
        progress = evolution.trigger_quality_based_reembedding(
            quality_threshold=0.7
        )
    """

    # Minimum extraction improvement to trigger re-embedding (5%)
    DEFAULT_IMPROVEMENT_THRESHOLD = 0.05

    # Default minimum quality score threshold
    DEFAULT_QUALITY_THRESHOLD = 0.7

    # Maximum entities to re-embed in one batch
    DEFAULT_MAX_REEMBEDDING = 1000

    # Default batch size for re-embedding operations
    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        quality_tracker: "QualityMetricsTracker",
        reembedding_service: "ReembeddingService",
        experiential_learning_monitor: Optional[ExperientialLearningMonitor] = None,
        improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    ) -> None:
        """Initialize quality evolution.

        Args:
            quality_tracker: Tracker for quality metrics
            reembedding_service: Service for triggering re-embedding
            experiential_learning_monitor: Optional WorldStateModel integration
            improvement_threshold: Minimum extraction improvement to trigger (default 5%)
            quality_threshold: Minimum quality score threshold (default 0.7)
        """
        self._quality_tracker = quality_tracker
        self._reembedding_service = reembedding_service
        self._learning_monitor = experiential_learning_monitor
        self._improvement_threshold = improvement_threshold
        self._quality_threshold = quality_threshold

        # Track baseline quality for comparison
        self._baseline_quality: Optional[Dict[str, float]] = None
        self._last_check: Optional[datetime] = None

        logger.info(
            f"Initialized EmbeddingQualityEvolution with "
            f"improvement_threshold={improvement_threshold:.1%}, "
            f"quality_threshold={quality_threshold:.2f}"
        )

    def monitor_extraction_quality_improvement(
        self,
    ) -> Optional["ReembeddingProgress"]:
        """Monitor extraction pipeline for quality improvements.

        When extraction quality improves significantly (>5%), trigger
        re-embedding of entities extracted with lower quality.

        Integration with TrainingFreeGRPO:
        - Monitors ExperientialLearningMonitor.get_current_quality_metrics()
        - Compares with baseline quality
        - Triggers re-embedding when precision improves

        Returns:
            ReembeddingProgress if re-embedding was triggered, None otherwise
        """
        if self._learning_monitor is None:
            logger.warning(
                "No experiential learning monitor configured, "
                "cannot monitor extraction quality improvement"
            )
            return None

        # Get current quality metrics
        try:
            current_quality = self._learning_monitor.get_current_quality_metrics()
        except Exception as e:
            logger.error(f"Failed to get current quality metrics: {e}")
            return None

        # Get or establish baseline
        if self._baseline_quality is None:
            try:
                self._baseline_quality = self._learning_monitor.get_baseline_quality()
            except Exception as e:
                logger.error(f"Failed to get baseline quality: {e}")
                self._baseline_quality = current_quality.copy()
                return None

        # Check for significant improvement
        current_precision = current_quality.get("precision", 0.0)
        baseline_precision = self._baseline_quality.get("precision", 0.0)
        improvement = current_precision - baseline_precision

        logger.debug(
            f"Quality check: current={current_precision:.3f}, "
            f"baseline={baseline_precision:.3f}, "
            f"improvement={improvement:.3f}"
        )

        if improvement > self._improvement_threshold:
            logger.info(
                f"Extraction quality improved by {improvement:.1%} "
                f"(threshold: {self._improvement_threshold:.1%}), "
                "triggering re-embedding"
            )

            # Identify entities extracted before improvement
            # Use baseline precision as quality threshold
            low_quality_entities = self._quality_tracker.identify_low_quality_embeddings(
                min_quality_score=baseline_precision,
                limit=self.DEFAULT_MAX_REEMBEDDING,
            )

            if not low_quality_entities:
                logger.info("No low-quality embeddings found for re-embedding")
                # Update baseline even if no re-embedding needed
                self._baseline_quality = current_quality.copy()
                self._last_check = datetime.utcnow()
                return None

            logger.info(
                f"Found {len(low_quality_entities)} entities for re-embedding"
            )

            # Trigger re-embedding
            try:
                progress = self._reembedding_service.trigger_reembedding(
                    entity_ids=low_quality_entities,
                    batch_size=self.DEFAULT_BATCH_SIZE,
                )

                # Update baseline after successful re-embedding trigger
                self._baseline_quality = current_quality.copy()
                self._last_check = datetime.utcnow()

                return progress

            except Exception as e:
                logger.error(f"Failed to trigger re-embedding: {e}")
                return None

        self._last_check = datetime.utcnow()
        return None

    def trigger_quality_based_reembedding(
        self,
        quality_threshold: Optional[float] = None,
        limit: int = DEFAULT_MAX_REEMBEDDING,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Optional["ReembeddingProgress"]:
        """Trigger re-embedding for low-quality embeddings.

        Continuous quality improvement mechanism that identifies and
        re-embeds embeddings below the quality threshold.

        Args:
            quality_threshold: Minimum quality score (default: self._quality_threshold)
            limit: Maximum entities to re-embed
            batch_size: Batch size for re-embedding

        Returns:
            ReembeddingProgress if re-embedding was triggered, None if no candidates
        """
        threshold = quality_threshold or self._quality_threshold

        # Identify low-quality embeddings
        low_quality_entities = self._quality_tracker.identify_low_quality_embeddings(
            min_quality_score=threshold,
            limit=limit,
        )

        if not low_quality_entities:
            logger.debug(
                f"No embeddings below quality threshold {threshold:.2f}"
            )
            return None

        logger.info(
            f"Found {len(low_quality_entities)} low-quality embeddings "
            f"(threshold: {threshold:.2f}), triggering re-embedding"
        )

        try:
            return self._reembedding_service.trigger_reembedding(
                entity_ids=low_quality_entities,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.error(f"Failed to trigger quality-based re-embedding: {e}")
            return None

    def measure_quality_trend(
        self,
        entity_id: str,
        lookback_days: int = 30,
    ) -> Literal["improving", "stable", "degrading"]:
        """Measure quality trend for entity embeddings.

        Analyzes historical quality metrics to determine if an entity's
        embedding quality is improving, stable, or degrading.

        Args:
            entity_id: PKG entity ID
            lookback_days: Number of days to analyze

        Returns:
            Trend: "improving", "stable", or "degrading"
        """
        # Delegate to quality tracker which has store access
        trend = self._quality_tracker.update_quality_trend(entity_id, lookback_days)

        if trend is None:
            return "stable"  # Not enough data

        return trend  # type: ignore

    def update_quality_trends_batch(
        self,
        limit: int = 1000,
        lookback_days: int = 30,
    ) -> Dict[str, int]:
        """Batch update quality trends for entities.

        Updates quality_trend field for entities with enough historical data.

        Args:
            limit: Maximum entities to update
            lookback_days: Days of history to analyze

        Returns:
            Dictionary with counts: {"improving": N, "stable": N, "degrading": N}
        """
        # Get all unique entity IDs
        store_stats = self._quality_tracker._store.get_statistics()

        # This is a simplified implementation - in production would iterate
        # through entity IDs more efficiently
        trend_counts = {
            "improving": 0,
            "stable": 0,
            "degrading": 0,
            "insufficient_data": 0,
        }

        logger.info(
            f"Batch updating quality trends for up to {limit} entities"
        )

        # Note: A full implementation would need a method to iterate through
        # entity IDs. For now, we log the intent and return empty counts.
        logger.debug(
            "Batch trend update not fully implemented - requires entity ID iteration"
        )

        return trend_counts

    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status and statistics.

        Returns:
            Dictionary with evolution status information
        """
        tracker_stats = self._quality_tracker.get_statistics()

        return {
            "improvement_threshold": self._improvement_threshold,
            "quality_threshold": self._quality_threshold,
            "baseline_quality": self._baseline_quality,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "has_learning_monitor": self._learning_monitor is not None,
            "tracker_statistics": tracker_stats,
        }

    def set_baseline_quality(self, baseline: Dict[str, float]) -> None:
        """Manually set baseline quality for comparison.

        Useful for initializing baseline without ExperientialLearningMonitor.

        Args:
            baseline: Dictionary with quality metrics (e.g., {"precision": 0.8})
        """
        self._baseline_quality = baseline.copy()
        logger.info(f"Manually set baseline quality: {baseline}")

    def reset_baseline(self) -> None:
        """Reset baseline quality to None.

        Next call to monitor_extraction_quality_improvement will
        establish a new baseline.
        """
        self._baseline_quality = None
        self._last_check = None
        logger.info("Reset quality baseline")
