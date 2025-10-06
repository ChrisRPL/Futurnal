"""Metrics collection system for IMAP sync operations.

This module provides comprehensive metrics tracking for IMAP connector quality gates,
measuring reliability, performance, accuracy, privacy, and integration metrics as defined in
``docs/phase-1/imap-connector-production-plan/10-quality-gates-testing.md``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImapSyncMetricsSummary:
    """Aggregated summary of IMAP sync metrics over a time window."""

    mailbox_id: str
    time_window_hours: int
    generated_at: datetime

    # Reliability metrics
    total_sync_attempts: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    sync_failure_rate: float = 0.0  # Target: <0.5%

    total_connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    connection_failure_rate: float = 0.0  # Target: <1%

    total_parse_attempts: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    parse_failure_rate: float = 0.0  # Target: <2%

    # Performance metrics
    detection_times_seconds: List[float] = field(default_factory=list)
    average_detection_time_seconds: float = 0.0  # Target: <300s (5 min)
    max_detection_time_seconds: float = 0.0

    total_messages_processed: int = 0
    total_processing_duration_seconds: float = 0.0
    messages_per_second: float = 0.0  # Target: ≥1.0

    average_sync_latency_seconds: float = 0.0  # Target: <30s per folder
    sync_latencies: List[float] = field(default_factory=list)

    # Accuracy metrics
    thread_reconstruction_attempts: int = 0
    thread_reconstruction_correct: int = 0
    thread_reconstruction_accuracy: float = 0.0  # Target: ≥95%

    attachment_extraction_attempts: int = 0
    attachment_extraction_successful: int = 0
    attachment_extraction_accuracy: float = 0.0  # Target: ≥98%

    # Privacy metrics
    pii_leak_count: int = 0  # Target: 0 (zero tolerance)
    consent_checks_performed: int = 0
    consent_checks_granted: int = 0
    consent_coverage: float = 0.0  # Target: 100%

    # Integration metrics
    element_sink_attempts: int = 0
    element_sink_successes: int = 0
    element_sink_success_rate: float = 0.0  # Target: ≥99%

    state_persistence_attempts: int = 0
    state_persistence_successes: int = 0
    state_persistence_success_rate: float = 0.0  # Target: 100%

    def calculate_rates(self) -> None:
        """Calculate all rate-based metrics from raw counts."""
        # Reliability rates
        if self.total_sync_attempts > 0:
            self.sync_failure_rate = self.failed_syncs / self.total_sync_attempts

        if self.total_connection_attempts > 0:
            self.connection_failure_rate = (
                self.failed_connections / self.total_connection_attempts
            )

        if self.total_parse_attempts > 0:
            self.parse_failure_rate = self.failed_parses / self.total_parse_attempts

        # Performance metrics
        if self.detection_times_seconds:
            self.average_detection_time_seconds = sum(
                self.detection_times_seconds
            ) / len(self.detection_times_seconds)
            self.max_detection_time_seconds = max(self.detection_times_seconds)

        if self.total_processing_duration_seconds > 0 and self.total_messages_processed > 0:
            self.messages_per_second = (
                self.total_messages_processed / self.total_processing_duration_seconds
            )

        if self.sync_latencies:
            self.average_sync_latency_seconds = sum(self.sync_latencies) / len(
                self.sync_latencies
            )

        # Accuracy metrics
        if self.thread_reconstruction_attempts > 0:
            self.thread_reconstruction_accuracy = (
                self.thread_reconstruction_correct / self.thread_reconstruction_attempts
            )

        if self.attachment_extraction_attempts > 0:
            self.attachment_extraction_accuracy = (
                self.attachment_extraction_successful
                / self.attachment_extraction_attempts
            )

        # Privacy metrics
        if self.consent_checks_performed > 0:
            self.consent_coverage = (
                self.consent_checks_granted / self.consent_checks_performed
            )

        # Integration metrics
        if self.element_sink_attempts > 0:
            self.element_sink_success_rate = (
                self.element_sink_successes / self.element_sink_attempts
            )

        if self.state_persistence_attempts > 0:
            self.state_persistence_success_rate = (
                self.state_persistence_successes / self.state_persistence_attempts
            )


class ImapSyncMetricsCollector:
    """Thread-safe metrics collector for IMAP sync operations."""

    def __init__(self):
        """Initialize metrics collector with empty state."""
        self._lock = Lock()
        self._counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._timings: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # ============================================================================
    # Reliability Metrics
    # ============================================================================

    def record_sync_attempt(self, mailbox_id: str, success: bool) -> None:
        """Record a sync attempt outcome."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["sync_attempts"] += 1
            if success:
                self._counters[labels_key]["sync_successes"] += 1
            else:
                self._counters[labels_key]["sync_failures"] += 1

    def record_connection_attempt(self, mailbox_id: str, success: bool) -> None:
        """Record a connection attempt outcome."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["connection_attempts"] += 1
            if success:
                self._counters[labels_key]["connection_successes"] += 1
            else:
                self._counters[labels_key]["connection_failures"] += 1

    def record_parse_attempt(self, mailbox_id: str, success: bool) -> None:
        """Record an email parse attempt outcome."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["parse_attempts"] += 1
            if success:
                self._counters[labels_key]["parse_successes"] += 1
            else:
                self._counters[labels_key]["parse_failures"] += 1

    # ============================================================================
    # Performance Metrics
    # ============================================================================

    def record_detection_time(self, mailbox_id: str, seconds: float) -> None:
        """Record time to detect a new message (for IDLE/polling)."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._timings[labels_key]["detection_times"].append(seconds)

    def record_message_processing(
        self, mailbox_id: str, message_count: int, duration_seconds: float
    ) -> None:
        """Record message processing throughput."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["messages_processed"] += message_count
            self._timings[labels_key]["processing_durations"].append(duration_seconds)

    def record_sync_latency(self, mailbox_id: str, folder: str, seconds: float) -> None:
        """Record sync latency for a specific folder."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._timings[labels_key]["sync_latencies"].append(seconds)
            self._events[labels_key].append(
                {
                    "type": "sync_latency",
                    "folder": folder,
                    "latency": seconds,
                    "timestamp": datetime.utcnow(),
                }
            )

    # ============================================================================
    # Accuracy Metrics
    # ============================================================================

    def record_thread_reconstruction(
        self, mailbox_id: str, success: bool, correct: bool = True
    ) -> None:
        """Record thread reconstruction attempt and accuracy.

        Args:
            mailbox_id: Mailbox identifier
            success: Whether reconstruction completed without errors
            correct: Whether reconstruction matched ground truth (if available)
        """
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            if success:
                self._counters[labels_key]["thread_reconstruction_attempts"] += 1
                if correct:
                    self._counters[labels_key]["thread_reconstruction_correct"] += 1

    def record_attachment_extraction(self, mailbox_id: str, success: bool) -> None:
        """Record attachment extraction attempt outcome."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["attachment_extraction_attempts"] += 1
            if success:
                self._counters[labels_key]["attachment_extraction_successful"] += 1

    # ============================================================================
    # Privacy Metrics
    # ============================================================================

    def record_pii_leak(self, mailbox_id: str, details: str = "") -> None:
        """Record a PII leak incident (critical security event).

        Args:
            mailbox_id: Mailbox identifier
            details: Description of the leak (redacted)
        """
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["pii_leaks"] += 1
            logger.critical(
                f"PII LEAK DETECTED for mailbox {mailbox_id}",
                extra={"mailbox_id": mailbox_id, "details": details},
            )

    def record_consent_check(self, mailbox_id: str, granted: bool) -> None:
        """Record consent check outcome."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["consent_checks"] += 1
            if granted:
                self._counters[labels_key]["consent_granted"] += 1

    # ============================================================================
    # Integration Metrics
    # ============================================================================

    def record_element_sink(self, mailbox_id: str, success: bool) -> None:
        """Record element sink operation outcome."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["element_sink_attempts"] += 1
            if success:
                self._counters[labels_key]["element_sink_successes"] += 1

    def record_state_persistence(self, mailbox_id: str, success: bool) -> None:
        """Record state persistence operation outcome."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            self._counters[labels_key]["state_persistence_attempts"] += 1
            if success:
                self._counters[labels_key]["state_persistence_successes"] += 1

    # ============================================================================
    # Query and Aggregation
    # ============================================================================

    def get_counter(self, metric_name: str, mailbox_id: str) -> int:
        """Get current value of a counter metric."""
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            return self._counters[labels_key].get(metric_name, 0)

    def generate_summary(
        self, mailbox_id: str, hours: int = 1
    ) -> ImapSyncMetricsSummary:
        """Generate aggregated metrics summary for time window.

        Args:
            mailbox_id: Mailbox to summarize
            hours: Time window in hours (currently returns all-time metrics)

        Returns:
            Aggregated metrics summary
        """
        with self._lock:
            labels_key = f"mailbox:{mailbox_id}"
            counters = self._counters[labels_key]
            timings = self._timings[labels_key]

            summary = ImapSyncMetricsSummary(
                mailbox_id=mailbox_id,
                time_window_hours=hours,
                generated_at=datetime.utcnow(),
                # Reliability
                total_sync_attempts=counters.get("sync_attempts", 0),
                successful_syncs=counters.get("sync_successes", 0),
                failed_syncs=counters.get("sync_failures", 0),
                total_connection_attempts=counters.get("connection_attempts", 0),
                successful_connections=counters.get("connection_successes", 0),
                failed_connections=counters.get("connection_failures", 0),
                total_parse_attempts=counters.get("parse_attempts", 0),
                successful_parses=counters.get("parse_successes", 0),
                failed_parses=counters.get("parse_failures", 0),
                # Performance
                detection_times_seconds=timings.get("detection_times", []).copy(),
                total_messages_processed=counters.get("messages_processed", 0),
                total_processing_duration_seconds=sum(
                    timings.get("processing_durations", [])
                ),
                sync_latencies=timings.get("sync_latencies", []).copy(),
                # Accuracy
                thread_reconstruction_attempts=counters.get(
                    "thread_reconstruction_attempts", 0
                ),
                thread_reconstruction_correct=counters.get(
                    "thread_reconstruction_correct", 0
                ),
                attachment_extraction_attempts=counters.get(
                    "attachment_extraction_attempts", 0
                ),
                attachment_extraction_successful=counters.get(
                    "attachment_extraction_successful", 0
                ),
                # Privacy
                pii_leak_count=counters.get("pii_leaks", 0),
                consent_checks_performed=counters.get("consent_checks", 0),
                consent_checks_granted=counters.get("consent_granted", 0),
                # Integration
                element_sink_attempts=counters.get("element_sink_attempts", 0),
                element_sink_successes=counters.get("element_sink_successes", 0),
                state_persistence_attempts=counters.get("state_persistence_attempts", 0),
                state_persistence_successes=counters.get(
                    "state_persistence_successes", 0
                ),
            )

            # Calculate all derived metrics
            summary.calculate_rates()

            return summary

    def reset_metrics(self, mailbox_id: Optional[str] = None) -> None:
        """Reset metrics for a specific mailbox or all mailboxes.

        Args:
            mailbox_id: Mailbox to reset, or None to reset all
        """
        with self._lock:
            if mailbox_id:
                labels_key = f"mailbox:{mailbox_id}"
                self._counters.pop(labels_key, None)
                self._timings.pop(labels_key, None)
                self._events.pop(labels_key, None)
            else:
                self._counters.clear()
                self._timings.clear()
                self._events.clear()


# Global metrics collector instance
_global_metrics_collector: Optional[ImapSyncMetricsCollector] = None


def get_global_metrics_collector() -> ImapSyncMetricsCollector:
    """Get or create global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = ImapSyncMetricsCollector()
    return _global_metrics_collector
