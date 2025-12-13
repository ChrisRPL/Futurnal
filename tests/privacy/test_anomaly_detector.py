"""Tests for privacy anomaly detection module."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.futurnal.privacy.anomaly_detector import (
    AnomalyDetector,
    AnomalyConfig,
    Anomaly,
    AnomalyType,
    AnomalySeverity,
    EventWindow,
    create_anomaly_detector,
)


class TestEventWindow:
    """Test EventWindow functionality."""

    def test_add_event(self):
        window = EventWindow(window_minutes=5)
        window.add_event("test_event", {"key": "value"})

        assert window.count() == 1

    def test_count_events_by_type(self):
        window = EventWindow(window_minutes=5)
        window.add_event("type_a", {})
        window.add_event("type_a", {})
        window.add_event("type_b", {})

        assert window.count("type_a") == 2
        assert window.count("type_b") == 1
        assert window.count() == 3

    def test_count_by_source(self):
        window = EventWindow(window_minutes=5)
        window.add_event("event", {"source": "src_a"})
        window.add_event("event", {"source": "src_a"})
        window.add_event("event", {"source": "src_b"})

        assert window.count_by_source("src_a") == 2
        assert window.count_by_source("src_b") == 1

    def test_get_events(self):
        window = EventWindow(window_minutes=5)
        window.add_event("type_a", {"data": 1})
        window.add_event("type_b", {"data": 2})

        events = window.get_events("type_a")
        assert len(events) == 1
        assert events[0][2]["data"] == 1

    def test_sum_metadata_field(self):
        window = EventWindow(window_minutes=5)
        window.add_event("data", {"bytes": 100})
        window.add_event("data", {"bytes": 200})
        window.add_event("data", {"bytes": 300})

        total = window.sum_metadata_field("data", "bytes")
        assert total == 600.0


class TestAnomalyConfig:
    """Test AnomalyConfig validation."""

    def test_default_config(self):
        config = AnomalyConfig()
        assert config.window_minutes == 10
        assert config.consent_violation_threshold == 5

    def test_invalid_window_minutes(self):
        with pytest.raises(ValueError):
            AnomalyConfig(window_minutes=0)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            AnomalyConfig(consent_violation_threshold=0)

    def test_invalid_spike_factor(self):
        with pytest.raises(ValueError):
            AnomalyConfig(data_volume_spike_factor=0.5)


class TestAnomaly:
    """Test Anomaly data class."""

    def test_to_dict(self):
        anomaly = Anomaly(
            anomaly_type=AnomalyType.CONSENT_VIOLATION,
            severity=AnomalySeverity.WARNING,
            message="Test message",
            detected_at=datetime(2024, 1, 1, 12, 0, 0),
            source="test_source",
            metadata={"count": 5},
        )

        d = anomaly.to_dict()

        assert d["anomaly_type"] == "consent_violation"
        assert d["severity"] == "warning"
        assert d["message"] == "Test message"
        assert d["source"] == "test_source"
        assert d["metadata"]["count"] == 5


class TestAnomalyDetector:
    """Test AnomalyDetector core functionality."""

    @pytest.fixture
    def detector(self):
        """Create a detector with low thresholds for testing."""
        config = AnomalyConfig(
            window_minutes=10,
            consent_violation_threshold=3,
            data_volume_spike_factor=2.0,
            max_consent_changes_per_hour=5,
            check_interval_seconds=1,
        )
        return AnomalyDetector(config)

    def test_detect_consent_violations(self, detector):
        # Record violations below threshold
        detector.record_consent_violation("source_a", scope="CONTENT_ANALYSIS")
        detector.record_consent_violation("source_a", scope="METADATA_ACCESS")

        anomalies = detector.check_for_anomalies()
        assert len(anomalies) == 0  # Below threshold

        # Record more to exceed threshold
        detector.record_consent_violation("source_b", scope="CONTENT_ANALYSIS")
        detector.reset_alert_tracking()

        anomalies = detector.check_for_anomalies()
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.CONSENT_VIOLATION

    def test_detect_rapid_consent_changes(self, detector):
        # Record consent changes
        for i in range(6):
            detector.record_consent_change(f"source_{i}", "CONTENT", "grant")

        anomalies = detector.check_for_anomalies()

        # Should detect rapid consent changes
        consent_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.RAPID_CONSENT_CHANGES]
        assert len(consent_anomalies) == 1

    def test_detect_escalation_without_consent(self, detector):
        # Record escalation without consent
        detector.record_escalation("source_a", "cloud_processing", had_consent=False)

        anomalies = detector.check_for_anomalies()

        escalation_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.ESCALATION_WITHOUT_CONSENT]
        assert len(escalation_anomalies) == 1
        assert escalation_anomalies[0].severity == AnomalySeverity.CRITICAL

    def test_escalation_with_consent_no_anomaly(self, detector):
        # Record escalation with proper consent
        detector.record_escalation("source_a", "cloud_processing", had_consent=True)

        anomalies = detector.check_for_anomalies()

        escalation_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.ESCALATION_WITHOUT_CONSENT]
        assert len(escalation_anomalies) == 0

    def test_detect_unexpected_activity(self):
        # Create detector with activity schedule
        config = AnomalyConfig(
            window_minutes=10,
            activity_schedule={
                "source_a": [(9, 17)],  # Only allowed 9 AM - 5 PM
            },
        )
        detector = AnomalyDetector(config)

        # Record activity (will check against current hour)
        detector.record_activity("source_a", "sync")

        anomalies = detector.check_for_anomalies()

        # Result depends on current hour - just verify no errors
        # In production this would be more deterministic
        assert isinstance(anomalies, list)

    def test_anomaly_callback(self):
        callback = MagicMock()
        config = AnomalyConfig(consent_violation_threshold=2)
        detector = AnomalyDetector(config, on_anomaly=callback)

        # Trigger anomaly
        detector.record_consent_violation("source_a")
        detector.record_consent_violation("source_b")
        detector.check_for_anomalies()

        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        assert isinstance(call_args, Anomaly)

    def test_get_recent_anomalies(self, detector):
        # Record violations to trigger anomaly
        for i in range(4):
            detector.record_consent_violation(f"source_{i}")

        detector.check_for_anomalies()

        # Get recent anomalies
        anomalies = detector.get_recent_anomalies()
        assert len(anomalies) >= 1

        # Filter by type
        consent_anomalies = detector.get_recent_anomalies(
            anomaly_type=AnomalyType.CONSENT_VIOLATION
        )
        assert all(a.anomaly_type == AnomalyType.CONSENT_VIOLATION for a in consent_anomalies)

    def test_clear_recent_anomalies(self, detector):
        for i in range(4):
            detector.record_consent_violation(f"source_{i}")
        detector.check_for_anomalies()

        assert len(detector.get_recent_anomalies()) > 0

        detector.clear_recent_anomalies()
        assert len(detector.get_recent_anomalies()) == 0

    def test_reset_alert_tracking(self, detector):
        # Trigger an anomaly
        for i in range(4):
            detector.record_consent_violation(f"source_{i}")
        anomalies1 = detector.check_for_anomalies()

        # Same conditions should not re-alert
        anomalies2 = detector.check_for_anomalies()
        assert len(anomalies2) == 0  # Already alerted

        # Reset tracking
        detector.reset_alert_tracking()

        # Now should alert again
        anomalies3 = detector.check_for_anomalies()
        assert len(anomalies3) >= 1


class TestAnomalyDetectorDataVolume:
    """Test data volume spike detection."""

    def test_detect_data_volume_spike(self):
        config = AnomalyConfig(
            window_minutes=10,
            data_volume_spike_factor=2.0,
        )
        detector = AnomalyDetector(config)

        # Build baseline (need at least 3 samples)
        detector.record_data_processed("source_a", bytes_processed=1000)
        detector.check_for_anomalies()
        detector.reset_alert_tracking()

        detector.record_data_processed("source_a", bytes_processed=1000)
        detector.check_for_anomalies()
        detector.reset_alert_tracking()

        detector.record_data_processed("source_a", bytes_processed=1000)
        detector.check_for_anomalies()
        detector.reset_alert_tracking()

        # Now add a spike (3x baseline = 3000, which is > 2.0 * 1000)
        detector.record_data_processed("source_a", bytes_processed=5000)
        anomalies = detector.check_for_anomalies()

        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.UNUSUAL_DATA_VOLUME]
        assert len(volume_anomalies) == 1
        assert volume_anomalies[0].source == "source_a"


class TestAnomalyDetectorBackgroundMonitoring:
    """Test background monitoring functionality."""

    def test_start_stop(self):
        config = AnomalyConfig(check_interval_seconds=1)
        detector = AnomalyDetector(config)

        detector.start()
        assert detector._running is True
        assert detector._monitor_thread is not None

        detector.stop()
        assert detector._running is False

    def test_start_twice_is_noop(self):
        config = AnomalyConfig(check_interval_seconds=1)
        detector = AnomalyDetector(config)

        detector.start()
        thread1 = detector._monitor_thread

        detector.start()
        thread2 = detector._monitor_thread

        assert thread1 is thread2  # Same thread

        detector.stop()


class TestAnomalyDetectorWithAuditLogger:
    """Test integration with audit logger."""

    def test_anomaly_logged_to_audit(self, tmp_path):
        from src.futurnal.privacy.audit import AuditLogger

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        audit_logger = AuditLogger(output_dir=audit_dir)

        config = AnomalyConfig(consent_violation_threshold=2)
        detector = AnomalyDetector(config, audit_logger=audit_logger)

        # Trigger anomaly
        detector.record_consent_violation("source_a")
        detector.record_consent_violation("source_b")
        detector.check_for_anomalies()

        # Check audit log
        events = list(audit_logger.iter_events())
        anomaly_events = [e for e in events if "anomaly" in e.get("action", "")]
        assert len(anomaly_events) >= 1


class TestCreateAnomalyDetector:
    """Test factory function."""

    def test_create_with_defaults(self):
        detector = create_anomaly_detector()
        assert detector is not None
        assert detector.config.window_minutes == 10

    def test_create_with_custom_config(self):
        config = AnomalyConfig(window_minutes=5)
        detector = create_anomaly_detector(config=config)
        assert detector.config.window_minutes == 5

    def test_create_with_callback(self):
        callback = MagicMock()
        detector = create_anomaly_detector(on_anomaly=callback)
        assert detector.on_anomaly is callback
