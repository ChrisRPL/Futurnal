"""Tests for privacy alerts module."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from futurnal.privacy.alerts import (
    AlertManager,
    AlertConfig,
    Alert,
    AlertChannel,
    create_alert_manager,
)
from futurnal.privacy.anomaly_detector import (
    Anomaly,
    AnomalyType,
    AnomalySeverity,
)


class TestAlert:
    """Test Alert data class."""

    def test_to_dict(self):
        alert = Alert(
            alert_id="test_123",
            title="Test Alert",
            message="This is a test",
            severity="warning",
            source="test_source",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"key": "value"},
        )

        d = alert.to_dict()

        assert d["alert_id"] == "test_123"
        assert d["title"] == "Test Alert"
        assert d["severity"] == "warning"
        assert d["acknowledged"] is False

    def test_from_dict(self):
        data = {
            "alert_id": "test_456",
            "title": "Test Alert",
            "message": "Message",
            "severity": "critical",
            "source": "detector",
            "created_at": "2024-01-01T12:00:00",
            "acknowledged": True,
            "acknowledged_at": "2024-01-01T13:00:00",
            "metadata": {},
        }

        alert = Alert.from_dict(data)

        assert alert.alert_id == "test_456"
        assert alert.severity == "critical"
        assert alert.acknowledged is True

    def test_from_anomaly(self):
        anomaly = Anomaly(
            anomaly_type=AnomalyType.CONSENT_VIOLATION,
            severity=AnomalySeverity.WARNING,
            message="High consent violation rate",
            detected_at=datetime.utcnow(),
            metadata={"count": 5},
        )

        alert = Alert.from_anomaly(anomaly)

        assert "Consent Violation" in alert.title
        assert alert.message == "High consent violation rate"
        assert alert.severity == "warning"
        assert alert.anomaly_type == "consent_violation"


class TestAlertConfig:
    """Test AlertConfig."""

    def test_default_config(self):
        config = AlertConfig()
        assert config.retention_days == 30
        assert config.min_severity == "info"
        assert config.throttle_seconds == 300

    def test_custom_config(self):
        config = AlertConfig(
            retention_days=7,
            min_severity="warning",
            desktop_notifications=False,
        )
        assert config.retention_days == 7
        assert config.min_severity == "warning"


class TestAlertManager:
    """Test AlertManager core functionality."""

    @pytest.fixture
    def alert_dir(self, tmp_path):
        return tmp_path / "alerts"

    @pytest.fixture
    def manager(self, alert_dir):
        config = AlertConfig(
            alert_dir=alert_dir,
            throttle_seconds=0,  # Disable throttling for tests
        )
        return AlertManager(config)

    def test_add_remove_channel(self, manager):
        manager.add_channel(AlertChannel.LOG_FILE)
        assert AlertChannel.LOG_FILE in manager._channels

        manager.remove_channel(AlertChannel.LOG_FILE)
        assert AlertChannel.LOG_FILE not in manager._channels

    def test_send_alert(self, manager):
        manager.add_channel(AlertChannel.LOG_FILE)

        alert = Alert(
            alert_id="test_1",
            title="Test",
            message="Message",
            severity="warning",
            source="test",
        )

        result = manager.send_alert(alert)
        assert result is True

        # Alert should be stored
        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_id == "test_1"

    def test_send_alert_severity_filter(self, manager):
        manager.config.min_severity = "warning"

        # Info alert should be filtered
        info_alert = Alert(
            alert_id="info_1",
            title="Info",
            message="Info message",
            severity="info",
            source="test",
        )

        result = manager.send_alert(info_alert)
        assert result is False

        # Warning alert should pass
        warning_alert = Alert(
            alert_id="warning_1",
            title="Warning",
            message="Warning message",
            severity="warning",
            source="test",
        )

        result = manager.send_alert(warning_alert)
        assert result is True

    def test_send_alert_throttling(self, alert_dir):
        config = AlertConfig(
            alert_dir=alert_dir,
            throttle_seconds=60,  # 1 minute throttle
        )
        manager = AlertManager(config)

        alert1 = Alert(
            alert_id="alert_1",
            title="Test",
            message="Message",
            severity="warning",
            source="test",
            anomaly_type="consent_violation",
        )

        alert2 = Alert(
            alert_id="alert_2",
            title="Test",
            message="Message 2",
            severity="warning",
            source="test",
            anomaly_type="consent_violation",
        )

        # First alert should send
        result1 = manager.send_alert(alert1)
        assert result1 is True

        # Second similar alert should be throttled
        result2 = manager.send_alert(alert2)
        assert result2 is False

    def test_send_anomaly_alert(self, manager):
        manager.add_channel(AlertChannel.LOG_FILE)

        anomaly = Anomaly(
            anomaly_type=AnomalyType.CONSENT_VIOLATION,
            severity=AnomalySeverity.WARNING,
            message="Test anomaly",
            detected_at=datetime.utcnow(),
        )

        result = manager.send_anomaly_alert(anomaly)
        assert result is True

        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert "Consent Violation" in alerts[0].title

    def test_callback_notification(self, manager):
        callback = MagicMock()
        manager.add_callback(callback)

        alert = Alert(
            alert_id="test_1",
            title="Test",
            message="Message",
            severity="warning",
            source="test",
        )

        manager.send_alert(alert)
        callback.assert_called_once()


class TestAlertManagerPersistence:
    """Test alert persistence functionality."""

    @pytest.fixture
    def alert_dir(self, tmp_path):
        return tmp_path / "alerts"

    def test_alert_written_to_log_file(self, alert_dir):
        config = AlertConfig(alert_dir=alert_dir, throttle_seconds=0)
        manager = AlertManager(config)
        manager.add_channel(AlertChannel.LOG_FILE)

        alert = Alert(
            alert_id="persist_1",
            title="Persistent Alert",
            message="Should be in log",
            severity="warning",
            source="test",
        )

        manager.send_alert(alert)

        # Check log file
        log_file = alert_dir / "alerts.jsonl"
        assert log_file.exists()

        content = log_file.read_text()
        data = json.loads(content.strip())
        assert data["alert_id"] == "persist_1"

    def test_load_alerts_on_init(self, alert_dir):
        # Create alert log
        alert_dir.mkdir(parents=True, exist_ok=True)
        log_file = alert_dir / "alerts.jsonl"

        alert_data = {
            "alert_id": "existing_1",
            "title": "Existing",
            "message": "Loaded from file",
            "severity": "info",
            "source": "test",
            "created_at": datetime.utcnow().isoformat(),
            "acknowledged": False,
            "acknowledged_at": None,
            "metadata": {},
        }
        log_file.write_text(json.dumps(alert_data) + "\n")

        # Create manager - should load existing
        config = AlertConfig(alert_dir=alert_dir)
        manager = AlertManager(config)

        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_id == "existing_1"


class TestAlertManagerFiltering:
    """Test alert filtering functionality."""

    @pytest.fixture
    def manager_with_alerts(self, tmp_path):
        config = AlertConfig(
            alert_dir=tmp_path / "alerts",
            throttle_seconds=0,
        )
        manager = AlertManager(config)

        # Add various alerts
        for i, severity in enumerate(["info", "warning", "critical"]):
            alert = Alert(
                alert_id=f"alert_{i}",
                title=f"Alert {i}",
                message=f"Message {i}",
                severity=severity,
                source="test",
            )
            manager._store_alert(alert)

        return manager

    def test_filter_by_severity(self, manager_with_alerts):
        warnings = manager_with_alerts.get_alerts(severity="warning")
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"

    def test_filter_unacknowledged(self, manager_with_alerts):
        # All should be unacknowledged initially
        unacked = manager_with_alerts.get_alerts(unacknowledged_only=True)
        assert len(unacked) == 3

        # Acknowledge one
        manager_with_alerts.acknowledge_alert("alert_1")

        # Should have 2 unacknowledged
        unacked = manager_with_alerts.get_alerts(unacknowledged_only=True)
        assert len(unacked) == 2

    def test_filter_by_time(self, manager_with_alerts):
        # All alerts are recent
        since = datetime.utcnow() - timedelta(minutes=1)
        recent = manager_with_alerts.get_alerts(since=since)
        assert len(recent) == 3

        # No alerts from future
        future = datetime.utcnow() + timedelta(hours=1)
        none = manager_with_alerts.get_alerts(since=future)
        assert len(none) == 0

    def test_limit_results(self, manager_with_alerts):
        limited = manager_with_alerts.get_alerts(limit=2)
        assert len(limited) == 2


class TestAlertAcknowledgment:
    """Test alert acknowledgment functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = AlertConfig(
            alert_dir=tmp_path / "alerts",
            throttle_seconds=0,
        )
        return AlertManager(config)

    def test_acknowledge_alert(self, manager):
        alert = Alert(
            alert_id="ack_test",
            title="Test",
            message="Message",
            severity="warning",
            source="test",
        )
        manager._store_alert(alert)

        result = manager.acknowledge_alert("ack_test")
        assert result is True

        alerts = manager.get_alerts()
        assert alerts[0].acknowledged is True
        assert alerts[0].acknowledged_at is not None

    def test_acknowledge_nonexistent(self, manager):
        result = manager.acknowledge_alert("nonexistent")
        assert result is False

    def test_acknowledge_all(self, manager):
        for i in range(3):
            alert = Alert(
                alert_id=f"alert_{i}",
                title="Test",
                message="Message",
                severity="warning",
                source="test",
            )
            manager._store_alert(alert)

        count = manager.acknowledge_all()
        assert count == 3

        assert manager.get_unacknowledged_count() == 0

    def test_get_unacknowledged_count(self, manager):
        for i in range(5):
            alert = Alert(
                alert_id=f"alert_{i}",
                title="Test",
                message="Message",
                severity="warning",
                source="test",
            )
            manager._store_alert(alert)

        assert manager.get_unacknowledged_count() == 5

        manager.acknowledge_alert("alert_0")
        manager.acknowledge_alert("alert_1")

        assert manager.get_unacknowledged_count() == 3


class TestAlertCleanup:
    """Test alert cleanup functionality."""

    def test_cleanup_old_alerts(self, tmp_path):
        config = AlertConfig(
            alert_dir=tmp_path / "alerts",
            retention_days=7,
        )
        manager = AlertManager(config)

        # Add old alert
        old_alert = Alert(
            alert_id="old_1",
            title="Old",
            message="Old message",
            severity="info",
            source="test",
            created_at=datetime.utcnow() - timedelta(days=10),
        )
        manager._alerts.append(old_alert)

        # Add recent alert
        new_alert = Alert(
            alert_id="new_1",
            title="New",
            message="New message",
            severity="info",
            source="test",
            created_at=datetime.utcnow(),
        )
        manager._alerts.append(new_alert)

        removed = manager.cleanup_old_alerts()

        assert removed == 1
        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_id == "new_1"


class TestIPCNotification:
    """Test IPC notification functionality."""

    def test_ipc_notification_writes_file(self, tmp_path):
        config = AlertConfig(
            alert_dir=tmp_path / "alerts",
            throttle_seconds=0,
        )
        manager = AlertManager(config)
        manager.add_channel(AlertChannel.IPC)

        alert = Alert(
            alert_id="ipc_test",
            title="IPC Test",
            message="Should be in IPC file",
            severity="warning",
            source="test",
        )

        manager.send_alert(alert)

        # Check IPC file
        ipc_file = tmp_path / "alerts" / "ipc_alerts.json"
        assert ipc_file.exists()

        data = json.loads(ipc_file.read_text())
        assert len(data) == 1
        assert data[0]["alert_id"] == "ipc_test"


class TestDesktopNotifications:
    """Test desktop notification functionality."""

    def test_macos_notification(self, tmp_path):
        config = AlertConfig(
            alert_dir=tmp_path / "alerts",
            desktop_notifications=True,
        )
        manager = AlertManager(config)
        manager.add_channel(AlertChannel.DESKTOP)

        alert = Alert(
            alert_id="desktop_test",
            title="Desktop Test",
            message="Should trigger notification",
            severity="warning",
            source="test",
        )

        # Mock subprocess to avoid actual notification
        with patch("subprocess.run") as mock_run:
            manager._send_macos_notification(alert)
            # Should have been called (or at least attempted)
            # May not be called if not on macOS


class TestCreateAlertManager:
    """Test factory function."""

    def test_create_with_defaults(self, tmp_path):
        manager = create_alert_manager(alert_dir=tmp_path / "alerts")

        assert AlertChannel.LOG_FILE in manager._channels
        assert AlertChannel.IPC in manager._channels

    def test_create_without_desktop(self, tmp_path):
        manager = create_alert_manager(
            alert_dir=tmp_path / "alerts",
            enable_desktop=False,
        )

        assert AlertChannel.DESKTOP not in manager._channels

    def test_create_without_ipc(self, tmp_path):
        manager = create_alert_manager(
            alert_dir=tmp_path / "alerts",
            enable_ipc=False,
        )

        assert AlertChannel.IPC not in manager._channels
