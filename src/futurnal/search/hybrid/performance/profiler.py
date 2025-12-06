"""Performance Profiler for Hybrid Search.

Provides comprehensive performance monitoring and bottleneck detection
for query execution.

Key Features:
- Latency tracking by component, strategy, and intent
- Percentile calculations (p50, p95, p99)
- Bottleneck identification with recommendations
- Metrics export for monitoring integration

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md

Option B Compliance:
- Performance gates: P95 <1s validated for production
- Supports experiential learning feedback loops
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Performance targets from production plan
LATENCY_TARGET_P95_MS = 1000  # <1s for 95% of queries
LATENCY_TARGET_P50_MS = 200   # <200ms median
CACHE_HIT_RATE_TARGET = 0.60  # >60% cache hit rate


@dataclass
class LatencyBreakdown:
    """Detailed latency breakdown for a query."""

    query_id: str
    total_ms: float
    components: Dict[str, float]  # Component -> latency
    timestamp: datetime
    strategy: str
    intent_type: str
    cache_hits: List[str]
    cache_misses: List[str]


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""

    timestamp: datetime
    p50_latency: float
    p95_latency: float
    p99_latency: float
    cache_hit_rate: float
    queries_per_second: float
    error_rate: float


class PerformanceProfiler:
    """Performance monitoring and profiling.

    Tracks:
    - Latency breakdown by component
    - Cache effectiveness
    - Query throughput
    - Error rates

    Integration Points:
    - QueryPlanOptimizer: Provides historical data for planning
    - SearchQualityFeedback: Correlates performance with quality
    - Monitoring/Alerting: Exports metrics

    Example:
        >>> profiler = PerformanceProfiler()
        >>> profiler.record_query(
        ...     query_id="q123",
        ...     total_ms=150,
        ...     components={"embed_query": 20, "vector_search": 80, "rank_results": 50},
        ...     strategy="vector",
        ...     intent_type="factual",
        ...     cache_hits=["embedding"],
        ...     cache_misses=["query_result"]
        ... )
        >>> report = profiler.export_metrics()
    """

    def __init__(
        self,
        retention_hours: int = 24,
        snapshot_interval_minutes: int = 5,
        max_records: int = 10000,
    ) -> None:
        """Initialize performance profiler.

        Args:
            retention_hours: How long to retain records
            snapshot_interval_minutes: Interval between snapshots
            max_records: Maximum records to keep
        """
        self.retention_hours = retention_hours
        self.snapshot_interval = snapshot_interval_minutes
        self.max_records = max_records

        self.latency_records: List[LatencyBreakdown] = []
        self.snapshots: List[PerformanceSnapshot] = []

        # Aggregated statistics
        self._latency_by_strategy: Dict[str, List[float]] = defaultdict(list)
        self._latency_by_intent: Dict[str, List[float]] = defaultdict(list)
        self._latency_by_component: Dict[str, List[float]] = defaultdict(list)

    def record_query(
        self,
        query_id: str,
        total_ms: float,
        components: Dict[str, float],
        strategy: str,
        intent_type: str,
        cache_hits: Optional[List[str]] = None,
        cache_misses: Optional[List[str]] = None,
    ) -> None:
        """Record latency breakdown for a query.

        Args:
            query_id: Unique query identifier
            total_ms: Total query latency in milliseconds
            components: Latency breakdown by component
            strategy: Retrieval strategy used
            intent_type: Query intent classification
            cache_hits: List of cache layers that hit
            cache_misses: List of cache layers that missed
        """
        record = LatencyBreakdown(
            query_id=query_id,
            total_ms=total_ms,
            components=components,
            timestamp=datetime.utcnow(),
            strategy=strategy,
            intent_type=intent_type,
            cache_hits=cache_hits or [],
            cache_misses=cache_misses or [],
        )

        self.latency_records.append(record)

        # Update aggregates
        self._latency_by_strategy[strategy].append(total_ms)
        self._latency_by_intent[intent_type].append(total_ms)
        for component, latency in components.items():
            self._latency_by_component[component].append(latency)

        # Cleanup old records periodically
        if len(self.latency_records) % 100 == 0:
            self._cleanup_old_records()

    def get_avg_latency(
        self,
        strategy: Optional[str] = None,
        intent_type: Optional[str] = None,
    ) -> Optional[float]:
        """Get average latency, optionally filtered.

        Args:
            strategy: Optional strategy filter
            intent_type: Optional intent filter

        Returns:
            Average latency in ms or None if no data
        """
        if strategy and strategy in self._latency_by_strategy:
            values = self._latency_by_strategy[strategy]
            return statistics.mean(values) if values else None

        if intent_type and intent_type in self._latency_by_intent:
            values = self._latency_by_intent[intent_type]
            return statistics.mean(values) if values else None

        if self.latency_records:
            return statistics.mean(r.total_ms for r in self.latency_records)

        return None

    def get_percentile_latency(self, percentile: float) -> Optional[float]:
        """Get latency at specified percentile.

        Args:
            percentile: Percentile (0-100)

        Returns:
            Latency in ms at percentile or None
        """
        if not self.latency_records:
            return None

        latencies = sorted(r.total_ms for r in self.latency_records)
        index = int(len(latencies) * percentile / 100)
        return latencies[min(index, len(latencies) - 1)]

    def get_component_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics by component.

        Returns:
            Dict mapping component to statistics
        """
        breakdown: Dict[str, Dict[str, float]] = {}

        for component, latencies in self._latency_by_component.items():
            if latencies:
                breakdown[component] = {
                    "mean": statistics.mean(latencies),
                    "p50": statistics.median(latencies),
                    "p95": self._percentile(latencies, 95),
                    "p99": self._percentile(latencies, 99),
                    "min": min(latencies),
                    "max": max(latencies),
                }

        return breakdown

    def identify_bottlenecks(
        self,
        threshold_ratio: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks.

        Args:
            threshold_ratio: Component is bottleneck if it takes
                           more than this ratio of total latency

        Returns:
            List of bottleneck components with recommendations
        """
        bottlenecks: List[Dict[str, Any]] = []
        breakdown = self.get_component_breakdown()

        avg_total = self.get_avg_latency() or 1.0

        for component, stats in breakdown.items():
            ratio = stats["mean"] / avg_total
            if ratio > threshold_ratio:
                bottlenecks.append(
                    {
                        "component": component,
                        "mean_latency_ms": stats["mean"],
                        "ratio_of_total": ratio,
                        "recommendation": self._get_recommendation(component, stats),
                    }
                )

        return sorted(bottlenecks, key=lambda x: x["ratio_of_total"], reverse=True)

    def _get_recommendation(
        self,
        component: str,
        stats: Dict[str, float],
    ) -> str:
        """Get optimization recommendation for bottleneck.

        Args:
            component: Component name
            stats: Component statistics

        Returns:
            Recommendation string
        """
        recommendations = {
            "embed_query": "Consider caching embeddings or using faster embedding model",
            "vector_search": "Optimize vector index or reduce search scope",
            "graph_search": "Add graph indexes or limit traversal depth",
            "temporal_filter": "Pre-compute temporal indexes",
            "llm_intent": "Enable LLM inference caching or use faster model",
            "rank_results": "Simplify ranking function or reduce result set",
            "fuse_results": "Optimize fusion algorithm or reduce candidate count",
            "traverse_causes": "Limit causal traversal depth",
            "traverse_effects": "Limit effect traversal depth",
        }

        return recommendations.get(
            component,
            f"Review {component} implementation for optimization opportunities",
        )

    def create_snapshot(self) -> PerformanceSnapshot:
        """Create point-in-time performance snapshot.

        Returns:
            PerformanceSnapshot with current metrics
        """
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            p50_latency=self.get_percentile_latency(50) or 0.0,
            p95_latency=self.get_percentile_latency(95) or 0.0,
            p99_latency=self.get_percentile_latency(99) or 0.0,
            cache_hit_rate=self._calculate_cache_hit_rate(),
            queries_per_second=self._calculate_qps(),
            error_rate=0.0,  # TODO: Implement error tracking
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent queries.

        Returns:
            Cache hit rate (0-1)
        """
        if not self.latency_records:
            return 0.0

        total_cache_ops = 0
        total_hits = 0

        for record in self.latency_records[-100:]:  # Last 100 queries
            total_cache_ops += len(record.cache_hits) + len(record.cache_misses)
            total_hits += len(record.cache_hits)

        return total_hits / total_cache_ops if total_cache_ops > 0 else 0.0

    def _calculate_qps(self) -> float:
        """Calculate queries per second.

        Returns:
            QPS rate
        """
        if len(self.latency_records) < 2:
            return 0.0

        recent = self.latency_records[-100:]
        time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()

        return len(recent) / time_span if time_span > 0 else 0.0

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values.

        Args:
            values: List of values
            percentile: Percentile (0-100)

        Returns:
            Value at percentile
        """
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _cleanup_old_records(self) -> None:
        """Remove records older than retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)

        # Remove old records
        self.latency_records = [
            r for r in self.latency_records if r.timestamp > cutoff
        ]

        # Enforce max records
        if len(self.latency_records) > self.max_records:
            self.latency_records = self.latency_records[-self.max_records:]

        # Rebuild aggregates periodically
        if len(self.latency_records) % 1000 == 0:
            self._rebuild_aggregates()

    def _rebuild_aggregates(self) -> None:
        """Rebuild aggregate statistics from records."""
        self._latency_by_strategy = defaultdict(list)
        self._latency_by_intent = defaultdict(list)
        self._latency_by_component = defaultdict(list)

        for record in self.latency_records:
            self._latency_by_strategy[record.strategy].append(record.total_ms)
            self._latency_by_intent[record.intent_type].append(record.total_ms)
            for component, latency in record.components.items():
                self._latency_by_component[component].append(latency)

    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for monitoring integration.

        Returns:
            Dict with all performance metrics
        """
        return {
            "latency": {
                "p50": self.get_percentile_latency(50),
                "p95": self.get_percentile_latency(95),
                "p99": self.get_percentile_latency(99),
                "mean": self.get_avg_latency(),
            },
            "cache": {
                "hit_rate": self._calculate_cache_hit_rate(),
            },
            "throughput": {
                "qps": self._calculate_qps(),
            },
            "bottlenecks": self.identify_bottlenecks(),
            "by_strategy": {
                strategy: statistics.mean(latencies)
                for strategy, latencies in self._latency_by_strategy.items()
                if latencies
            },
            "by_component": self.get_component_breakdown(),
            "targets": {
                "p95_target_ms": LATENCY_TARGET_P95_MS,
                "p50_target_ms": LATENCY_TARGET_P50_MS,
                "cache_hit_target": CACHE_HIT_RATE_TARGET,
                "p95_meets_target": (self.get_percentile_latency(95) or 0) < LATENCY_TARGET_P95_MS,
                "p50_meets_target": (self.get_percentile_latency(50) or 0) < LATENCY_TARGET_P50_MS,
                "cache_meets_target": self._calculate_cache_hit_rate() > CACHE_HIT_RATE_TARGET,
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.latency_records = []
        self.snapshots = []
        self._latency_by_strategy = defaultdict(list)
        self._latency_by_intent = defaultdict(list)
        self._latency_by_component = defaultdict(list)
        logger.info("PerformanceProfiler reset")

    def get_record_count(self) -> int:
        """Get number of latency records.

        Returns:
            Number of records
        """
        return len(self.latency_records)
