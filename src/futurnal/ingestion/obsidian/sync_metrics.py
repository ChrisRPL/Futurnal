"""Comprehensive sync performance monitoring and metrics for Obsidian vaults.

This module provides detailed performance monitoring, metrics collection,
and reporting capabilities for Obsidian vault synchronization operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of sync metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class SyncPhase(Enum):
    """Sync operation phases for detailed tracking."""
    FILE_SCAN = "file_scan"
    CHANGE_DETECTION = "change_detection"
    EVENT_PROCESSING = "event_processing"
    BATCH_CREATION = "batch_creation"
    BATCH_PROCESSING = "batch_processing"
    JOB_QUEUING = "job_queuing"
    PATH_CHANGE_HANDLING = "path_change_handling"


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimerContext:
    """Context manager for timing operations."""
    metric_name: str
    collector: 'SyncMetricsCollector'
    labels: Dict[str, str] = field(default_factory=dict)
    start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.metric_name, duration, self.labels)


@dataclass
class SyncMetricsSummary:
    """Summary of sync metrics over a time period."""
    vault_id: str
    time_period_start: datetime
    time_period_end: datetime

    # Event metrics
    total_events_processed: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_priority: Dict[str, int] = field(default_factory=dict)

    # Performance metrics
    average_batch_size: float = 0.0
    average_batch_processing_time: float = 0.0
    average_event_processing_time: float = 0.0

    # Throughput metrics
    events_per_second: float = 0.0
    batches_per_minute: float = 0.0

    # Change detection metrics
    path_changes_detected: int = 0
    content_changes_detected: int = 0
    change_detection_time: float = 0.0

    # Error metrics
    failed_events: int = 0
    failed_batches: int = 0
    error_rate: float = 0.0

    # Queue metrics
    average_queue_depth: float = 0.0
    max_queue_depth: int = 0
    backpressure_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class SyncMetricsCollector:
    """Collects and manages sync performance metrics."""

    def __init__(
        self,
        *,
        max_history_size: int = 10000,
        metric_retention_hours: int = 24,
        enable_detailed_timing: bool = True,
        enable_histogram_metrics: bool = True,
    ):
        self.max_history_size = max_history_size
        self.metric_retention_hours = metric_retention_hours
        self.enable_detailed_timing = enable_detailed_timing
        self.enable_histogram_metrics = enable_histogram_metrics

        # Metric storage
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))

        # Labeled metrics
        self._labeled_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._labeled_gauges: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Event tracking
        self._event_history: deque = deque(maxlen=max_history_size)
        self._batch_history: deque = deque(maxlen=max_history_size)
        self._error_history: deque = deque(maxlen=max_history_size)

        # Active timers
        self._active_timers: Dict[str, float] = {}

        # Aggregation cache
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl_seconds = 60  # Cache for 1 minute

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self._counters[name] += value

        if labels:
            label_key = self._serialize_labels(labels)
            self._labeled_counters[name][label_key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        self._gauges[name] = value

        if labels:
            label_key = self._serialize_labels(labels)
            self._labeled_gauges[name][label_key] = value

    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric."""
        metric_value = MetricValue(
            value=duration,
            labels=labels or {}
        )
        self._timers[name].append(metric_value)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        if not self.enable_histogram_metrics:
            return

        metric_value = MetricValue(
            value=value,
            labels=labels or {}
        )
        self._histograms[name].append(metric_value)

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        if not self.enable_detailed_timing:
            return
        self._active_timers[name] = time.time()

    def stop_timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Stop a named timer and record the duration."""
        if not self.enable_detailed_timing or name not in self._active_timers:
            return None

        start_time = self._active_timers.pop(name)
        duration = time.time() - start_time
        self.record_timer(name, duration, labels)
        return duration

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> TimerContext:
        """Create a timer context manager."""
        return TimerContext(name, self, labels or {})

    def record_event(self, event_type: str, vault_id: str, **metadata) -> None:
        """Record a sync event."""
        event_data = {
            "type": event_type,
            "vault_id": vault_id,
            "timestamp": datetime.utcnow(),
            "metadata": metadata
        }
        self._event_history.append(event_data)

        # Update counters
        self.increment_counter("sync_events_total")
        self.increment_counter("sync_events_by_type", labels={"type": event_type})
        self.increment_counter("sync_events_by_vault", labels={"vault_id": vault_id})

    def record_batch(self, batch_id: str, vault_id: str, event_count: int, processing_time: float, **metadata) -> None:
        """Record a batch processing event."""
        batch_data = {
            "batch_id": batch_id,
            "vault_id": vault_id,
            "event_count": event_count,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow(),
            "metadata": metadata
        }
        self._batch_history.append(batch_data)

        # Update metrics
        self.increment_counter("sync_batches_total")
        self.increment_counter("sync_batches_by_vault", labels={"vault_id": vault_id})
        self.record_timer("batch_processing_time", processing_time, {"vault_id": vault_id})
        self.record_histogram("batch_size", event_count, {"vault_id": vault_id})

    def record_error(self, error_type: str, vault_id: str, error_message: str, **metadata) -> None:
        """Record a sync error."""
        error_data = {
            "type": error_type,
            "vault_id": vault_id,
            "message": error_message,
            "timestamp": datetime.utcnow(),
            "metadata": metadata
        }
        self._error_history.append(error_data)

        # Update counters
        self.increment_counter("sync_errors_total")
        self.increment_counter("sync_errors_by_type", labels={"type": error_type})
        self.increment_counter("sync_errors_by_vault", labels={"vault_id": vault_id})

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get counter value."""
        if labels:
            label_key = self._serialize_labels(labels)
            return self._labeled_counters[name].get(label_key, 0)
        return self._counters.get(name, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        if labels:
            label_key = self._serialize_labels(labels)
            return self._labeled_gauges[name].get(label_key)
        return self._gauges.get(name)

    def get_timer_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        values = self._timers.get(name, [])

        if labels:
            # Filter by labels
            label_key = self._serialize_labels(labels)
            values = [v for v in values if self._serialize_labels(v.labels) == label_key]

        if not values:
            return {}

        durations = [v.value for v in values]
        return {
            "count": len(durations),
            "sum": sum(durations),
            "avg": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "p50": self._percentile(durations, 0.5),
            "p90": self._percentile(durations, 0.9),
            "p95": self._percentile(durations, 0.95),
            "p99": self._percentile(durations, 0.99),
        }

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self._histograms.get(name, [])

        if labels:
            label_key = self._serialize_labels(labels)
            values = [v for v in values if self._serialize_labels(v.labels) == label_key]

        if not values:
            return {}

        data = [v.value for v in values]
        return {
            "count": len(data),
            "sum": sum(data),
            "avg": sum(data) / len(data),
            "min": min(data),
            "max": max(data),
            "p50": self._percentile(data, 0.5),
            "p90": self._percentile(data, 0.9),
            "p95": self._percentile(data, 0.95),
            "p99": self._percentile(data, 0.99),
        }

    def generate_summary(self, vault_id: str, hours: int = 1) -> SyncMetricsSummary:
        """Generate a metrics summary for the specified time period."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Filter events by time period
        events = [e for e in self._event_history
                 if e["timestamp"] >= start_time and e["vault_id"] == vault_id]

        batches = [b for b in self._batch_history
                  if b["timestamp"] >= start_time and b["vault_id"] == vault_id]

        errors = [e for e in self._error_history
                 if e["timestamp"] >= start_time and e["vault_id"] == vault_id]

        # Calculate metrics
        summary = SyncMetricsSummary(
            vault_id=vault_id,
            time_period_start=start_time,
            time_period_end=end_time
        )

        # Event metrics
        summary.total_events_processed = len(events)
        summary.events_by_type = {}
        summary.events_by_priority = {}

        for event in events:
            event_type = event["type"]
            summary.events_by_type[event_type] = summary.events_by_type.get(event_type, 0) + 1

        # Batch metrics
        if batches:
            total_events = sum(b["event_count"] for b in batches)
            total_processing_time = sum(b["processing_time"] for b in batches)

            summary.average_batch_size = total_events / len(batches)
            summary.average_batch_processing_time = total_processing_time / len(batches)

            if total_events > 0:
                summary.average_event_processing_time = total_processing_time / total_events

        # Throughput metrics
        period_hours = hours
        if period_hours > 0:
            summary.events_per_second = summary.total_events_processed / (period_hours * 3600)
            summary.batches_per_minute = len(batches) / (period_hours * 60)

        # Error metrics
        summary.failed_events = len(errors)
        if summary.total_events_processed > 0:
            summary.error_rate = summary.failed_events / summary.total_events_processed

        # Change detection metrics (from metadata)
        path_changes = sum(1 for e in events if e.get("metadata", {}).get("path_changes", 0) > 0)
        content_changes = sum(1 for e in events if e.get("metadata", {}).get("content_changes", 0) > 0)

        summary.path_changes_detected = path_changes
        summary.content_changes_detected = content_changes

        return summary

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "labeled_counters": {k: dict(v) for k, v in self._labeled_counters.items()},
            "labeled_gauges": {k: dict(v) for k, v in self._labeled_gauges.items()},
            "timer_stats": {name: self.get_timer_stats(name) for name in self._timers.keys()},
            "histogram_stats": {name: self.get_histogram_stats(name) for name in self._histograms.keys()},
            "last_updated": datetime.utcnow().isoformat(),
        }

    def export_metrics(self, file_path: Path) -> None:
        """Export metrics to a JSON file."""
        metrics_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "metrics": self.get_all_metrics(),
            "recent_events": list(self._event_history)[-100:],  # Last 100 events
            "recent_batches": list(self._batch_history)[-100:],  # Last 100 batches
            "recent_errors": list(self._error_history)[-50:],   # Last 50 errors
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def cleanup_old_metrics(self) -> None:
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metric_retention_hours)

        # Clean up event history
        self._event_history = deque(
            [e for e in self._event_history if e["timestamp"] >= cutoff_time],
            maxlen=self.max_history_size
        )

        # Clean up batch history
        self._batch_history = deque(
            [b for b in self._batch_history if b["timestamp"] >= cutoff_time],
            maxlen=self.max_history_size
        )

        # Clean up error history
        self._error_history = deque(
            [e for e in self._error_history if e["timestamp"] >= cutoff_time],
            maxlen=self.max_history_size
        )

        # Clean up timer and histogram data
        for timer_deque in self._timers.values():
            while timer_deque and timer_deque[0].timestamp < cutoff_time:
                timer_deque.popleft()

        for hist_deque in self._histograms.values():
            while hist_deque and hist_deque[0].timestamp < cutoff_time:
                hist_deque.popleft()

        logger.debug(f"Cleaned up metrics older than {cutoff_time}")

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        self._counters.clear()
        self._gauges.clear()
        self._timers.clear()
        self._histograms.clear()
        self._labeled_counters.clear()
        self._labeled_gauges.clear()
        self._event_history.clear()
        self._batch_history.clear()
        self._error_history.clear()
        self._active_timers.clear()
        self._cache.clear()

    def _serialize_labels(self, labels: Dict[str, str]) -> str:
        """Serialize labels for use as dictionary keys."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(percentile * (len(sorted_data) - 1))
        return sorted_data[index]


class SyncMetricsReporter:
    """Reports sync metrics in various formats."""

    def __init__(self, collector: SyncMetricsCollector):
        self.collector = collector

    def generate_text_report(self, vault_id: str, hours: int = 1) -> str:
        """Generate a human-readable text report."""
        summary = self.collector.generate_summary(vault_id, hours)

        report = f"""
Sync Metrics Report for Vault: {vault_id}
Period: {summary.time_period_start.strftime('%Y-%m-%d %H:%M')} - {summary.time_period_end.strftime('%Y-%m-%d %H:%M')}

Event Processing:
- Total Events: {summary.total_events_processed}
- Events per Second: {summary.events_per_second:.2f}
- Average Event Processing Time: {summary.average_event_processing_time:.3f}s

Batch Processing:
- Batches per Minute: {summary.batches_per_minute:.2f}
- Average Batch Size: {summary.average_batch_size:.1f}
- Average Batch Processing Time: {summary.average_batch_processing_time:.3f}s

Change Detection:
- Path Changes: {summary.path_changes_detected}
- Content Changes: {summary.content_changes_detected}

Error Metrics:
- Failed Events: {summary.failed_events}
- Error Rate: {summary.error_rate:.1%}
        """.strip()

        return report

    def generate_prometheus_metrics(self) -> str:
        """Generate metrics in Prometheus format."""
        lines = []

        # Counters
        for name, value in self.collector._counters.items():
            metric_name = f"futurnal_sync_{name.replace('-', '_')}"
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")

        # Labeled counters
        for name, label_dict in self.collector._labeled_counters.items():
            metric_name = f"futurnal_sync_{name.replace('-', '_')}"
            lines.append(f"# TYPE {metric_name} counter")
            for label_key, value in label_dict.items():
                labels = "{" + label_key.replace("=", '="').replace(",", '",') + '"}'
                lines.append(f"{metric_name}{labels} {value}")

        # Gauges
        for name, value in self.collector._gauges.items():
            metric_name = f"futurnal_sync_{name.replace('-', '_')}"
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")

        return "\n".join(lines)


def create_metrics_collector(**config) -> SyncMetricsCollector:
    """Factory function to create a metrics collector with sensible defaults."""
    return SyncMetricsCollector(**config)


def create_metrics_reporter(collector: SyncMetricsCollector) -> SyncMetricsReporter:
    """Factory function to create a metrics reporter."""
    return SyncMetricsReporter(collector)