"""Embedding Performance Profiler.

Profiles embedding performance and identifies bottlenecks.
Tracks latency, throughput, memory usage with recommendations.

Performance Targets (from production plan):
- <2s single embedding latency
- >100/min batch throughput
- Memory within limits

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Performance target thresholds (from production plan)
LATENCY_TARGET_P95_MS = 2000  # <2s target
LATENCY_WARNING_P95_MS = 1500  # Warning level
LATENCY_TARGET_P99_MS = 5000  # P99 acceptable

THROUGHPUT_TARGET_PER_MIN = 100  # >100/min target
THROUGHPUT_WARNING_PER_MIN = 30  # Warning level

MEMORY_WARNING_MB = 2000  # 2GB warning threshold
MEMORY_TARGET_MB = 4000  # 4GB target max


@dataclass
class PerformanceMetric:
    """Single performance measurement.

    Records timing, memory, and context for one embedding operation.
    """

    model_id: str
    entity_type: str
    content_length: int
    latency_ms: float
    memory_mb: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "entity_type": self.entity_type,
            "content_length": self.content_length,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "timestamp": self.timestamp.isoformat(),
        }


class EmbeddingPerformanceProfiler:
    """Profiles embedding performance and identifies bottlenecks.

    Thread-safe profiler that tracks:
    - Latency (avg, p50, p95, p99)
    - Throughput (embeddings per minute)
    - Memory usage
    - Per-model and per-entity-type breakdowns

    Performance Targets:
    - <2s single embedding latency (P95)
    - >100/min batch throughput
    - Memory within target limits

    Example:
        profiler = EmbeddingPerformanceProfiler()

        # Record metrics
        profiler.profile_embedding_request(
            model_id="instructor-large-entity",
            entity_type="Person",
            content_length=500,
            latency_ms=150.0,
            memory_mb=800.0,
        )

        # Get performance report
        report = profiler.generate_performance_report()
        print(f"P95 latency: {report['p95_latency_ms']:.0f}ms")

        # Check for bottlenecks
        bottlenecks = profiler.identify_performance_bottlenecks()
        for recommendation in bottlenecks:
            print(f"  - {recommendation}")
    """

    # Maximum metrics to retain in memory
    DEFAULT_MAX_HISTORY = 10000

    def __init__(
        self,
        max_metrics_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        """Initialize profiler with bounded history.

        Args:
            max_metrics_history: Maximum metrics to retain (default: 10000)
        """
        self._metrics: List[PerformanceMetric] = []
        self._max_history = max_metrics_history
        self._lock = Lock()

        logger.info(
            f"Initialized EmbeddingPerformanceProfiler with "
            f"max_history={max_metrics_history}"
        )

    def profile_embedding_request(
        self,
        model_id: str,
        entity_type: str,
        content_length: int,
        latency_ms: float,
        memory_mb: Optional[float] = None,
    ) -> None:
        """Record performance metrics for embedding request.

        Thread-safe metric recording with automatic history pruning.

        Args:
            model_id: ID of the embedding model
            entity_type: Type of entity embedded
            content_length: Length of input content
            latency_ms: Time taken in milliseconds
            memory_mb: Optional memory usage in MB (auto-detected if None)
        """
        if memory_mb is None:
            memory_mb = self._get_current_memory_mb()

        metric = PerformanceMetric(
            model_id=model_id,
            entity_type=entity_type,
            content_length=content_length,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
        )

        with self._lock:
            self._metrics.append(metric)

            # Prune old metrics if over capacity
            if len(self._metrics) > self._max_history:
                # Keep most recent half
                self._metrics = self._metrics[-self._max_history // 2 :]

        logger.debug(
            f"Profiled embedding: model={model_id}, type={entity_type}, "
            f"latency={latency_ms:.0f}ms, memory={memory_mb:.0f}MB"
        )

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Returns metrics matching production plan targets.

        Returns:
            Dictionary with:
            - avg_latency_ms: Average latency
            - p50_latency_ms: Median latency
            - p95_latency_ms: 95th percentile latency
            - p99_latency_ms: 99th percentile latency
            - max_latency_ms: Maximum latency
            - avg_memory_mb: Average memory usage
            - max_memory_mb: Maximum memory usage
            - throughput_per_minute: Embeddings per minute
            - total_requests: Total requests recorded
            - by_model: Per-model statistics
            - by_entity_type: Per-entity-type statistics
        """
        with self._lock:
            if not self._metrics:
                return {
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "avg_memory_mb": 0.0,
                    "max_memory_mb": 0.0,
                    "throughput_per_minute": 0.0,
                    "total_requests": 0,
                    "by_model": {},
                    "by_entity_type": {},
                }

            latencies = [m.latency_ms for m in self._metrics]
            memory_values = [m.memory_mb for m in self._metrics]

            # Calculate time span for throughput
            time_span_seconds = (
                self._metrics[-1].timestamp - self._metrics[0].timestamp
            ).total_seconds()
            time_span_minutes = max(time_span_seconds / 60, 0.01)  # Avoid division by zero

            return {
                "avg_latency_ms": float(np.mean(latencies)),
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "max_latency_ms": float(max(latencies)),
                "avg_memory_mb": float(np.mean(memory_values)),
                "max_memory_mb": float(max(memory_values)),
                "throughput_per_minute": len(self._metrics) / time_span_minutes,
                "total_requests": len(self._metrics),
                "by_model": self._get_by_model_stats(),
                "by_entity_type": self._get_by_entity_type_stats(),
                "time_span_minutes": time_span_minutes,
            }

    def identify_performance_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks with recommendations.

        Checks against production targets:
        - <2s single embedding (P95)
        - >100/min batch throughput

        Returns:
            List of recommendation strings
        """
        report = self.generate_performance_report()
        recommendations = []

        if report["total_requests"] == 0:
            return ["No performance data collected yet"]

        # Latency checks
        p95_latency = report.get("p95_latency_ms", 0)
        p99_latency = report.get("p99_latency_ms", 0)

        if p95_latency > LATENCY_TARGET_P95_MS:
            recommendations.append(
                f"CRITICAL: P95 latency ({p95_latency:.0f}ms) exceeds 2s target - "
                "consider model quantization or lighter model variant"
            )
        elif p95_latency > LATENCY_WARNING_P95_MS:
            recommendations.append(
                f"WARNING: P95 latency ({p95_latency:.0f}ms) approaching 2s target - "
                "monitor closely"
            )

        if p99_latency > LATENCY_TARGET_P99_MS:
            recommendations.append(
                f"WARNING: P99 latency ({p99_latency:.0f}ms) exceeds 5s - "
                "investigate slow outlier requests"
            )

        # Memory checks
        avg_memory = report.get("avg_memory_mb", 0)
        max_memory = report.get("max_memory_mb", 0)

        if avg_memory > MEMORY_WARNING_MB:
            recommendations.append(
                f"WARNING: Average memory ({avg_memory:.0f}MB) exceeds {MEMORY_WARNING_MB}MB - "
                "enable model unloading between batches"
            )

        if max_memory > MEMORY_TARGET_MB:
            recommendations.append(
                f"CRITICAL: Peak memory ({max_memory:.0f}MB) exceeds {MEMORY_TARGET_MB}MB - "
                "reduce batch size or use quantized models"
            )

        # Throughput checks
        throughput = report.get("throughput_per_minute", 0)

        if throughput < THROUGHPUT_WARNING_PER_MIN:
            recommendations.append(
                f"WARNING: Low throughput ({throughput:.1f}/min) below {THROUGHPUT_WARNING_PER_MIN}/min - "
                "increase batch size or parallelize"
            )
        elif throughput < THROUGHPUT_TARGET_PER_MIN:
            recommendations.append(
                f"INFO: Throughput ({throughput:.1f}/min) below target {THROUGHPUT_TARGET_PER_MIN}/min - "
                "optimize batch processing"
            )

        # Model-specific recommendations
        by_model = report.get("by_model", {})
        for model_id, stats in by_model.items():
            model_avg = stats.get("avg_latency_ms", 0)
            if model_avg > LATENCY_WARNING_P95_MS:
                recommendations.append(
                    f"WARNING: Model '{model_id}' has high latency ({model_avg:.0f}ms) - "
                    "consider lighter variant"
                )

        # Entity type recommendations
        by_type = report.get("by_entity_type", {})
        for entity_type, stats in by_type.items():
            type_avg = stats.get("avg_latency_ms", 0)
            if type_avg > LATENCY_WARNING_P95_MS:
                recommendations.append(
                    f"INFO: Entity type '{entity_type}' has high latency ({type_avg:.0f}ms)"
                )

        if not recommendations:
            recommendations.append(
                f"All metrics within targets - P95={p95_latency:.0f}ms, "
                f"throughput={throughput:.1f}/min"
            )

        return recommendations

    def _get_by_model_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics grouped by model.

        Returns:
            Dictionary mapping model_id to statistics
        """
        by_model: Dict[str, List[float]] = {}
        for m in self._metrics:
            if m.model_id not in by_model:
                by_model[m.model_id] = []
            by_model[m.model_id].append(m.latency_ms)

        return {
            model_id: {
                "avg_latency_ms": float(np.mean(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "count": len(latencies),
            }
            for model_id, latencies in by_model.items()
        }

    def _get_by_entity_type_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics grouped by entity type.

        Returns:
            Dictionary mapping entity_type to statistics
        """
        by_type: Dict[str, List[float]] = {}
        for m in self._metrics:
            if m.entity_type not in by_type:
                by_type[m.entity_type] = []
            by_type[m.entity_type].append(m.latency_ms)

        return {
            entity_type: {
                "avg_latency_ms": float(np.mean(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "count": len(latencies),
            }
            for entity_type, latencies in by_type.items()
        }

    def _get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB.

        Returns:
            RSS memory usage in megabytes
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            logger.debug("psutil not available for memory profiling")
            return 0.0
        except Exception as e:
            logger.debug(f"Failed to get memory usage: {e}")
            return 0.0

    def get_recent_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get most recent performance metrics.

        Args:
            count: Number of recent metrics to return

        Returns:
            List of metric dictionaries
        """
        with self._lock:
            recent = self._metrics[-count:]
            return [m.to_dict() for m in recent]

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            count = len(self._metrics)
            self._metrics.clear()

        logger.info(f"Reset performance profiler, cleared {count} metrics")

    def get_metric_count(self) -> int:
        """Get number of metrics currently stored.

        Returns:
            Number of metrics in history
        """
        with self._lock:
            return len(self._metrics)
