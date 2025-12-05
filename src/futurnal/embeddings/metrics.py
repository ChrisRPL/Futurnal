"""Embedding Metrics for Multi-Model Architecture.

Provides thread-safe metrics collection for embedding service monitoring.
Tracks per-model latency, throughput, and error rates.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/02-multi-model-architecture.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetrics:
    """Metrics for a single model.

    Tracks embedding generation statistics for performance monitoring
    and optimization decisions.

    Attributes:
        model_id: Identifier of the model
        total_embeddings: Total number of embeddings generated
        total_latency_ms: Cumulative latency in milliseconds
        latencies: Recent latency values for percentile calculations
        errors: Number of errors encountered
        last_used: Timestamp of last successful use
    """

    model_id: str
    total_embeddings: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    errors: int = 0
    last_used: Optional[datetime] = None

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds.

        Returns:
            Average latency, or 0.0 if no embeddings generated
        """
        if self.total_embeddings == 0:
            return 0.0
        return self.total_latency_ms / self.total_embeddings

    @property
    def p50_latency_ms(self) -> float:
        """Calculate 50th percentile (median) latency.

        Returns:
            P50 latency, or average if insufficient data
        """
        if len(self.latencies) < 10:
            return self.avg_latency_ms
        sorted_latencies = sorted(self.latencies)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx]

    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency.

        Returns:
            P95 latency, or average if insufficient data
        """
        if len(self.latencies) < 20:
            return self.avg_latency_ms
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99_latency_ms(self) -> float:
        """Calculate 99th percentile latency.

        Returns:
            P99 latency, or P95 if insufficient data
        """
        if len(self.latencies) < 100:
            return self.p95_latency_ms
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]

    @property
    def error_rate(self) -> float:
        """Calculate error rate.

        Returns:
            Error rate as fraction (0.0 to 1.0)
        """
        total = self.total_embeddings + self.errors
        if total == 0:
            return 0.0
        return self.errors / total

    @property
    def success_rate(self) -> float:
        """Calculate success rate.

        Returns:
            Success rate as fraction (0.0 to 1.0)
        """
        return 1.0 - self.error_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of metrics
        """
        return {
            "model_id": self.model_id,
            "total_embeddings": self.total_embeddings,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


class EmbeddingMetrics:
    """Thread-safe metrics collector for embedding service.

    Tracks per-model and per-entity-type statistics for monitoring,
    alerting, and optimization decisions.

    Features:
        - Per-model latency statistics (avg, p50, p95, p99)
        - Entity type distribution tracking
        - Error rate monitoring
        - Memory-bounded latency history
        - Thread-safe operations

    Example:
        metrics = EmbeddingMetrics()

        # Record embedding
        metrics.record_embedding(
            model_id="instructor-large-entity",
            entity_type="Person",
            latency_ms=145.3,
            vector_dimension=768,
            success=True,
        )

        # Get summary
        summary = metrics.get_summary()
        print(f"Total embeddings: {summary['total_embeddings']}")
        print(f"Error rate: {summary['error_rate']:.2%}")
    """

    def __init__(self, max_latency_history: int = 1000) -> None:
        """Initialize the metrics collector.

        Args:
            max_latency_history: Maximum number of latency values to retain
                for percentile calculations (default: 1000)
        """
        self._model_metrics: Dict[str, ModelMetrics] = {}
        self._entity_type_counts: Dict[str, int] = {}
        self._dimension_counts: Dict[int, int] = {}
        self._lock = Lock()
        self._max_history = max_latency_history
        self._start_time = datetime.utcnow()
        self._total_batch_operations = 0

    def record_embedding(
        self,
        model_id: str,
        entity_type: str,
        latency_ms: float,
        vector_dimension: int,
        success: bool = True,
    ) -> None:
        """Record an embedding operation.

        Thread-safe recording of embedding generation metrics.

        Args:
            model_id: ID of the model used
            entity_type: Type of entity embedded
            latency_ms: Time taken in milliseconds
            vector_dimension: Dimension of output embedding
            success: Whether the operation succeeded
        """
        with self._lock:
            # Initialize model metrics if needed
            if model_id not in self._model_metrics:
                self._model_metrics[model_id] = ModelMetrics(model_id=model_id)

            metrics = self._model_metrics[model_id]

            if success:
                metrics.total_embeddings += 1
                metrics.total_latency_ms += latency_ms
                metrics.last_used = datetime.utcnow()

                # Maintain bounded latency history
                if len(metrics.latencies) >= self._max_history:
                    # Remove oldest half to avoid frequent resizing
                    metrics.latencies = metrics.latencies[-self._max_history // 2 :]
                metrics.latencies.append(latency_ms)
            else:
                metrics.errors += 1

            # Entity type counts
            self._entity_type_counts[entity_type] = (
                self._entity_type_counts.get(entity_type, 0) + 1
            )

            # Dimension counts
            self._dimension_counts[vector_dimension] = (
                self._dimension_counts.get(vector_dimension, 0) + 1
            )

    def record_batch_operation(self, batch_size: int) -> None:
        """Record a batch embedding operation.

        Args:
            batch_size: Number of embeddings in the batch
        """
        with self._lock:
            self._total_batch_operations += 1

    def get_model_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Get metrics for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            ModelMetrics if found, None otherwise
        """
        with self._lock:
            return self._model_metrics.get(model_id)

    def get_all_model_metrics(self) -> Dict[str, ModelMetrics]:
        """Get metrics for all models.

        Returns:
            Dict mapping model_id to ModelMetrics
        """
        with self._lock:
            return dict(self._model_metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary.

        Returns:
            Dictionary with summary statistics including:
            - total_embeddings: Total successful embeddings
            - total_errors: Total errors
            - error_rate: Overall error rate
            - entity_type_distribution: Counts per entity type
            - dimension_distribution: Counts per embedding dimension
            - model_metrics: Per-model statistics
            - uptime_seconds: Time since metrics collection started
            - batch_operations: Number of batch operations
        """
        with self._lock:
            total_embeddings = sum(
                m.total_embeddings for m in self._model_metrics.values()
            )
            total_errors = sum(m.errors for m in self._model_metrics.values())

            return {
                "total_embeddings": total_embeddings,
                "total_errors": total_errors,
                "error_rate": total_errors / max(total_embeddings + total_errors, 1),
                "entity_type_distribution": dict(self._entity_type_counts),
                "dimension_distribution": dict(self._dimension_counts),
                "model_metrics": {
                    model_id: m.to_dict() for model_id, m in self._model_metrics.items()
                },
                "uptime_seconds": (
                    datetime.utcnow() - self._start_time
                ).total_seconds(),
                "batch_operations": self._total_batch_operations,
            }

    def get_latency_summary(self) -> Dict[str, Dict[str, float]]:
        """Get latency summary across all models.

        Returns:
            Dict mapping model_id to latency statistics
        """
        with self._lock:
            return {
                model_id: {
                    "avg_ms": m.avg_latency_ms,
                    "p50_ms": m.p50_latency_ms,
                    "p95_ms": m.p95_latency_ms,
                    "p99_ms": m.p99_latency_ms,
                }
                for model_id, m in self._model_metrics.items()
            }

    def reset(self) -> None:
        """Reset all metrics.

        Use with caution - clears all collected data.
        """
        with self._lock:
            self._model_metrics.clear()
            self._entity_type_counts.clear()
            self._dimension_counts.clear()
            self._start_time = datetime.utcnow()
            self._total_batch_operations = 0

    @property
    def uptime_seconds(self) -> float:
        """Get time since metrics collection started."""
        return (datetime.utcnow() - self._start_time).total_seconds()

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        return (
            f"EmbeddingMetrics("
            f"embeddings={summary['total_embeddings']}, "
            f"errors={summary['total_errors']}, "
            f"models={len(self._model_metrics)})"
        )
