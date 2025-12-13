"""Privacy anomaly detection with sliding window analysis.

This module provides real-time anomaly detection for privacy-sensitive events
using sliding window analysis and configurable thresholds.

Features:
- Sliding window event aggregation
- Multiple anomaly types (consent violations, unusual activity, etc.)
- Configurable thresholds and time windows
- Background monitoring thread
- Alert generation for detected anomalies

Anomaly Types:
- CONSENT_VIOLATION: Failed consent checks exceeding threshold
- UNUSUAL_DATA_VOLUME: Data volume spike above baseline
- UNEXPECTED_ACTIVITY: Activity outside scheduled times
- ESCALATION_WITHOUT_CONSENT: Escalation events without prior consent
- RAPID_CONSENT_CHANGES: Too many consent changes in short period

Privacy-First Design (Option B):
- All detection is local-only
- No external data transmission
- Alerts stored locally with audit trail

Usage:
    >>> from futurnal.privacy.anomaly_detector import AnomalyDetector, AnomalyConfig
    >>> config = AnomalyConfig(consent_violation_threshold=5, window_minutes=10)
    >>> detector = AnomalyDetector(config)
    >>> detector.start()
    >>> # ... events are recorded via audit logger ...
    >>> detector.check_for_anomalies()
    >>> detector.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .audit import AuditLogger

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of privacy anomalies that can be detected."""

    CONSENT_VIOLATION = "consent_violation"
    UNUSUAL_DATA_VOLUME = "unusual_data_volume"
    UNEXPECTED_ACTIVITY = "unexpected_activity"
    ESCALATION_WITHOUT_CONSENT = "escalation_without_consent"
    RAPID_CONSENT_CHANGES = "rapid_consent_changes"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Anomaly:
    """Represents a detected privacy anomaly.

    Attributes:
        anomaly_type: Type of anomaly detected
        severity: Severity level
        message: Human-readable description
        detected_at: Timestamp of detection
        source: Source that triggered the anomaly (if applicable)
        metadata: Additional context about the anomaly
    """

    anomaly_type: AnomalyType
    severity: AnomalySeverity
    message: str
    detected_at: datetime
    source: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Convert anomaly to dictionary for storage/serialization."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "detected_at": self.detected_at.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection thresholds.

    Attributes:
        window_minutes: Time window for event aggregation (default: 10 min)
        consent_violation_threshold: Max failed consent checks before alert
        data_volume_spike_factor: Multiplier above baseline for volume alert
        max_consent_changes_per_hour: Max consent changes before alert
        activity_schedule: Optional dict of allowed activity hours per source
        check_interval_seconds: How often to run anomaly checks
    """

    window_minutes: int = 10
    consent_violation_threshold: int = 5
    data_volume_spike_factor: float = 3.0
    max_consent_changes_per_hour: int = 10
    activity_schedule: Dict[str, List[tuple]] = field(default_factory=dict)
    check_interval_seconds: int = 60

    def __post_init__(self):
        if self.window_minutes < 1:
            raise ValueError("window_minutes must be at least 1")
        if self.consent_violation_threshold < 1:
            raise ValueError("consent_violation_threshold must be at least 1")
        if self.data_volume_spike_factor < 1.0:
            raise ValueError("data_volume_spike_factor must be at least 1.0")


@dataclass
class EventWindow:
    """Sliding window for event tracking.

    Attributes:
        events: List of (timestamp, event_type, metadata) tuples
        window_minutes: Window duration
    """

    window_minutes: int = 10
    events: List[tuple] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_event(self, event_type: str, metadata: Optional[Dict] = None) -> None:
        """Add an event to the window."""
        with self._lock:
            self.events.append((datetime.utcnow(), event_type, metadata or {}))
            self._prune()

    def _prune(self) -> None:
        """Remove events outside the window (must hold lock)."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        self.events = [(ts, et, md) for ts, et, md in self.events if ts > cutoff]

    def count(self, event_type: Optional[str] = None) -> int:
        """Count events in window, optionally filtered by type."""
        with self._lock:
            self._prune()
            if event_type is None:
                return len(self.events)
            return sum(1 for _, et, _ in self.events if et == event_type)

    def count_by_source(self, source: str) -> int:
        """Count events from a specific source."""
        with self._lock:
            self._prune()
            return sum(
                1 for _, _, md in self.events
                if md.get("source") == source
            )

    def get_events(self, event_type: Optional[str] = None) -> List[tuple]:
        """Get all events in window, optionally filtered by type."""
        with self._lock:
            self._prune()
            if event_type is None:
                return list(self.events)
            return [(ts, et, md) for ts, et, md in self.events if et == event_type]

    def sum_metadata_field(self, event_type: str, field_name: str) -> float:
        """Sum a numeric metadata field for events of a given type."""
        with self._lock:
            self._prune()
            total = 0.0
            for _, et, md in self.events:
                if et == event_type and field_name in md:
                    try:
                        total += float(md[field_name])
                    except (ValueError, TypeError):
                        pass
            return total


class AnomalyDetector:
    """Detects privacy anomalies from audit log patterns.

    Monitors audit events in a sliding window and detects anomalous patterns
    such as consent violations, unusual data volumes, and unexpected activity.

    Example:
        >>> detector = AnomalyDetector(AnomalyConfig())
        >>> detector.start()
        >>> # Record events
        >>> detector.record_consent_violation("source_a")
        >>> detector.record_data_processed("source_a", bytes_processed=1000000)
        >>> # Check for anomalies
        >>> anomalies = detector.check_for_anomalies()
        >>> for a in anomalies:
        ...     print(f"{a.severity.value}: {a.message}")
    """

    def __init__(
        self,
        config: Optional[AnomalyConfig] = None,
        audit_logger: Optional["AuditLogger"] = None,
        on_anomaly: Optional[Callable[[Anomaly], None]] = None,
    ):
        """Initialize anomaly detector.

        Args:
            config: Detection configuration (uses defaults if None)
            audit_logger: Optional audit logger for logging anomalies
            on_anomaly: Optional callback for anomaly notifications
        """
        self.config = config or AnomalyConfig()
        self.audit_logger = audit_logger
        self.on_anomaly = on_anomaly

        # Event windows
        self._consent_violations = EventWindow(window_minutes=self.config.window_minutes)
        self._consent_changes = EventWindow(window_minutes=60)  # Fixed 1 hour for consent changes
        self._data_events = EventWindow(window_minutes=self.config.window_minutes)
        self._activity_events = EventWindow(window_minutes=self.config.window_minutes)
        self._escalation_events = EventWindow(window_minutes=self.config.window_minutes)

        # Baseline tracking
        self._data_baseline: Dict[str, float] = {}  # source -> avg bytes per window
        self._baseline_samples: Dict[str, List[float]] = defaultdict(list)
        self._baseline_lock = threading.Lock()

        # Background monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Detected anomalies (recent)
        self._recent_anomalies: List[Anomaly] = []
        self._anomaly_lock = threading.Lock()

        # Track alerted anomalies to avoid duplicates
        self._alerted_keys: Set[str] = set()

    def start(self) -> None:
        """Start background anomaly monitoring."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="AnomalyDetector",
        )
        self._monitor_thread.start()
        logger.info("Anomaly detector started")

    def stop(self) -> None:
        """Stop background anomaly monitoring."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Anomaly detector stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.wait(self.config.check_interval_seconds):
            try:
                self.check_for_anomalies()
            except Exception as e:
                logger.error(f"Error in anomaly monitoring: {e}")

    def record_consent_violation(
        self,
        source: str,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Record a consent violation event.

        Args:
            source: Data source that was denied
            scope: Consent scope that was missing
            reason: Reason for violation
        """
        self._consent_violations.add_event(
            "consent_violation",
            {"source": source, "scope": scope, "reason": reason},
        )

    def record_consent_change(
        self,
        source: str,
        scope: str,
        action: str,  # "grant" or "revoke"
    ) -> None:
        """Record a consent change event.

        Args:
            source: Data source
            scope: Consent scope
            action: "grant" or "revoke"
        """
        self._consent_changes.add_event(
            "consent_change",
            {"source": source, "scope": scope, "action": action},
        )

    def record_data_processed(
        self,
        source: str,
        bytes_processed: int,
        files_processed: int = 0,
    ) -> None:
        """Record data processing event.

        Args:
            source: Data source
            bytes_processed: Number of bytes processed
            files_processed: Number of files processed
        """
        self._data_events.add_event(
            "data_processed",
            {
                "source": source,
                "bytes": bytes_processed,
                "files": files_processed,
            },
        )

    def record_activity(
        self,
        source: str,
        action: str,
    ) -> None:
        """Record source activity event.

        Args:
            source: Data source
            action: Type of activity
        """
        self._activity_events.add_event(
            "activity",
            {"source": source, "action": action},
        )

    def record_escalation(
        self,
        source: str,
        escalation_type: str,
        had_consent: bool,
    ) -> None:
        """Record an escalation event.

        Args:
            source: Data source
            escalation_type: Type of escalation (e.g., "cloud_processing")
            had_consent: Whether consent was granted for escalation
        """
        self._escalation_events.add_event(
            "escalation",
            {
                "source": source,
                "type": escalation_type,
                "had_consent": had_consent,
            },
        )

    def check_for_anomalies(self) -> List[Anomaly]:
        """Run all anomaly checks and return detected anomalies.

        Returns:
            List of newly detected anomalies
        """
        new_anomalies = []

        # Check consent violations
        anomaly = self._check_consent_violations()
        if anomaly:
            new_anomalies.append(anomaly)

        # Check rapid consent changes
        anomaly = self._check_rapid_consent_changes()
        if anomaly:
            new_anomalies.append(anomaly)

        # Check data volume spikes
        anomalies = self._check_data_volume_spikes()
        new_anomalies.extend(anomalies)

        # Check unexpected activity
        anomalies = self._check_unexpected_activity()
        new_anomalies.extend(anomalies)

        # Check escalations without consent
        anomaly = self._check_escalation_consent()
        if anomaly:
            new_anomalies.append(anomaly)

        # Store and notify
        for anomaly in new_anomalies:
            self._store_anomaly(anomaly)
            self._notify_anomaly(anomaly)

        return new_anomalies

    def _check_consent_violations(self) -> Optional[Anomaly]:
        """Check for excessive consent violations."""
        count = self._consent_violations.count("consent_violation")

        if count >= self.config.consent_violation_threshold:
            key = f"consent_violation_{datetime.utcnow().strftime('%Y%m%d%H')}"
            if key in self._alerted_keys:
                return None
            self._alerted_keys.add(key)

            # Get affected sources
            events = self._consent_violations.get_events("consent_violation")
            sources = set(e[2].get("source") for e in events if e[2].get("source"))

            return Anomaly(
                anomaly_type=AnomalyType.CONSENT_VIOLATION,
                severity=AnomalySeverity.WARNING,
                message=f"High consent violation rate: {count} violations in {self.config.window_minutes} minutes",
                detected_at=datetime.utcnow(),
                metadata={
                    "violation_count": count,
                    "threshold": self.config.consent_violation_threshold,
                    "affected_sources": list(sources),
                    "window_minutes": self.config.window_minutes,
                },
            )

        return None

    def _check_rapid_consent_changes(self) -> Optional[Anomaly]:
        """Check for unusually rapid consent changes."""
        count = self._consent_changes.count("consent_change")

        if count > self.config.max_consent_changes_per_hour:
            key = f"rapid_consent_{datetime.utcnow().strftime('%Y%m%d%H')}"
            if key in self._alerted_keys:
                return None
            self._alerted_keys.add(key)

            return Anomaly(
                anomaly_type=AnomalyType.RAPID_CONSENT_CHANGES,
                severity=AnomalySeverity.INFO,
                message=f"Rapid consent changes detected: {count} changes in 1 hour",
                detected_at=datetime.utcnow(),
                metadata={
                    "change_count": count,
                    "threshold": self.config.max_consent_changes_per_hour,
                },
            )

        return None

    def _check_data_volume_spikes(self) -> List[Anomaly]:
        """Check for unusual data volume spikes per source."""
        anomalies = []

        # Get current volumes by source
        events = self._data_events.get_events("data_processed")
        source_volumes: Dict[str, float] = defaultdict(float)

        for _, _, md in events:
            source = md.get("source")
            bytes_val = md.get("bytes", 0)
            if source:
                source_volumes[source] += bytes_val

        # Update baselines and check for spikes
        with self._baseline_lock:
            for source, volume in source_volumes.items():
                # Update baseline samples
                self._baseline_samples[source].append(volume)
                # Keep last 10 samples for baseline
                if len(self._baseline_samples[source]) > 10:
                    self._baseline_samples[source] = self._baseline_samples[source][-10:]

                # Calculate baseline (average of samples)
                if len(self._baseline_samples[source]) >= 3:
                    baseline = sum(self._baseline_samples[source][:-1]) / (len(self._baseline_samples[source]) - 1)
                    self._data_baseline[source] = baseline

                    # Check for spike
                    if baseline > 0 and volume > baseline * self.config.data_volume_spike_factor:
                        key = f"volume_spike_{source}_{datetime.utcnow().strftime('%Y%m%d%H')}"
                        if key not in self._alerted_keys:
                            self._alerted_keys.add(key)
                            anomalies.append(Anomaly(
                                anomaly_type=AnomalyType.UNUSUAL_DATA_VOLUME,
                                severity=AnomalySeverity.WARNING,
                                message=f"Unusual data volume from {source}: {volume/1024/1024:.1f}MB (baseline: {baseline/1024/1024:.1f}MB)",
                                detected_at=datetime.utcnow(),
                                source=source,
                                metadata={
                                    "current_bytes": volume,
                                    "baseline_bytes": baseline,
                                    "spike_factor": volume / baseline if baseline > 0 else 0,
                                },
                            ))

        return anomalies

    def _check_unexpected_activity(self) -> List[Anomaly]:
        """Check for activity outside scheduled times."""
        anomalies = []

        if not self.config.activity_schedule:
            return anomalies

        events = self._activity_events.get_events("activity")
        now = datetime.utcnow()
        current_hour = now.hour

        for _, _, md in events:
            source = md.get("source")
            if not source or source not in self.config.activity_schedule:
                continue

            allowed_hours = self.config.activity_schedule[source]
            is_allowed = False

            for start_hour, end_hour in allowed_hours:
                if start_hour <= current_hour < end_hour:
                    is_allowed = True
                    break

            if not is_allowed:
                key = f"unexpected_activity_{source}_{datetime.utcnow().strftime('%Y%m%d%H')}"
                if key not in self._alerted_keys:
                    self._alerted_keys.add(key)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.UNEXPECTED_ACTIVITY,
                        severity=AnomalySeverity.INFO,
                        message=f"Activity from {source} outside scheduled hours (current: {current_hour}:00)",
                        detected_at=datetime.utcnow(),
                        source=source,
                        metadata={
                            "current_hour": current_hour,
                            "allowed_hours": allowed_hours,
                        },
                    ))

        return anomalies

    def _check_escalation_consent(self) -> Optional[Anomaly]:
        """Check for escalations without proper consent."""
        events = self._escalation_events.get_events("escalation")

        unconsented = [e for e in events if not e[2].get("had_consent")]

        if unconsented:
            key = f"unconsented_escalation_{datetime.utcnow().strftime('%Y%m%d%H')}"
            if key in self._alerted_keys:
                return None
            self._alerted_keys.add(key)

            sources = set(e[2].get("source") for e in unconsented if e[2].get("source"))
            types = set(e[2].get("type") for e in unconsented if e[2].get("type"))

            return Anomaly(
                anomaly_type=AnomalyType.ESCALATION_WITHOUT_CONSENT,
                severity=AnomalySeverity.CRITICAL,
                message=f"Escalation attempted without consent: {len(unconsented)} events",
                detected_at=datetime.utcnow(),
                metadata={
                    "event_count": len(unconsented),
                    "affected_sources": list(sources),
                    "escalation_types": list(types),
                },
            )

        return None

    def _store_anomaly(self, anomaly: Anomaly) -> None:
        """Store anomaly in recent list."""
        with self._anomaly_lock:
            self._recent_anomalies.append(anomaly)
            # Keep only last 100 anomalies
            if len(self._recent_anomalies) > 100:
                self._recent_anomalies = self._recent_anomalies[-100:]

    def _notify_anomaly(self, anomaly: Anomaly) -> None:
        """Notify of detected anomaly via callback and audit log."""
        # Log to audit trail
        if self.audit_logger:
            from .audit import AuditEvent

            self.audit_logger.record(
                AuditEvent(
                    job_id=f"anomaly_{int(datetime.utcnow().timestamp())}",
                    source="anomaly_detector",
                    action=f"anomaly:{anomaly.anomaly_type.value}",
                    status=anomaly.severity.value,
                    timestamp=anomaly.detected_at,
                    metadata=anomaly.to_dict(),
                )
            )

        # Call notification callback
        if self.on_anomaly:
            try:
                self.on_anomaly(anomaly)
            except Exception as e:
                logger.error(f"Error in anomaly callback: {e}")

        # Log at appropriate level
        if anomaly.severity == AnomalySeverity.CRITICAL:
            logger.error(f"ANOMALY DETECTED: {anomaly.message}")
        elif anomaly.severity == AnomalySeverity.WARNING:
            logger.warning(f"ANOMALY DETECTED: {anomaly.message}")
        else:
            logger.info(f"ANOMALY DETECTED: {anomaly.message}")

    def get_recent_anomalies(
        self,
        anomaly_type: Optional[AnomalyType] = None,
        since: Optional[datetime] = None,
    ) -> List[Anomaly]:
        """Get recent anomalies, optionally filtered.

        Args:
            anomaly_type: Filter by anomaly type
            since: Only return anomalies after this time

        Returns:
            List of anomalies matching criteria
        """
        with self._anomaly_lock:
            anomalies = list(self._recent_anomalies)

        if anomaly_type:
            anomalies = [a for a in anomalies if a.anomaly_type == anomaly_type]

        if since:
            anomalies = [a for a in anomalies if a.detected_at > since]

        return anomalies

    def clear_recent_anomalies(self) -> None:
        """Clear the recent anomalies list."""
        with self._anomaly_lock:
            self._recent_anomalies.clear()

    def reset_alert_tracking(self) -> None:
        """Reset alert tracking to allow re-alerting on same conditions."""
        self._alerted_keys.clear()


def create_anomaly_detector(
    audit_logger: Optional["AuditLogger"] = None,
    config: Optional[AnomalyConfig] = None,
    on_anomaly: Optional[Callable[[Anomaly], None]] = None,
) -> AnomalyDetector:
    """Create an anomaly detector with standard configuration.

    Args:
        audit_logger: Optional audit logger for anomaly logging
        config: Optional configuration (uses defaults if None)
        on_anomaly: Optional callback for anomaly notifications

    Returns:
        Configured AnomalyDetector instance
    """
    return AnomalyDetector(
        config=config or AnomalyConfig(),
        audit_logger=audit_logger,
        on_anomaly=on_anomaly,
    )


__all__ = [
    "AnomalyDetector",
    "AnomalyConfig",
    "Anomaly",
    "AnomalyType",
    "AnomalySeverity",
    "EventWindow",
    "create_anomaly_detector",
]
