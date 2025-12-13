"""Local notification system for privacy alerts.

This module provides a local-only alert system for privacy anomalies and events.
It supports multiple notification channels including desktop notifications,
log files, and IPC to desktop applications.

Features:
- Desktop notifications (platform-specific)
- Alert log persistence
- IPC notification to desktop app
- Alert severity filtering
- Alert acknowledgment tracking

Privacy-First Design (Option B):
- All alerts processed locally
- No external notification services
- Alert history stored locally with configurable retention

Usage:
    >>> from futurnal.privacy.alerts import AlertManager, AlertChannel
    >>> manager = AlertManager()
    >>> manager.add_channel(AlertChannel.DESKTOP)
    >>> manager.add_channel(AlertChannel.LOG_FILE)
    >>> manager.send_alert(anomaly)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .anomaly_detector import Anomaly, AnomalySeverity

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Available alert notification channels."""

    DESKTOP = "desktop"  # OS-level desktop notification
    LOG_FILE = "log_file"  # Alert log file
    IPC = "ipc"  # Inter-process communication to desktop app
    CALLBACK = "callback"  # Custom callback function


@dataclass
class AlertConfig:
    """Configuration for alert manager.

    Attributes:
        alert_dir: Directory for alert log files
        retention_days: Days to retain alert history
        min_severity: Minimum severity to trigger alerts
        desktop_notifications: Enable desktop notifications
        ipc_socket_path: Path to IPC socket for desktop app
        throttle_seconds: Minimum time between duplicate alerts
    """

    alert_dir: Optional[Path] = None
    retention_days: int = 30
    min_severity: str = "info"  # info, warning, critical
    desktop_notifications: bool = True
    ipc_socket_path: Optional[Path] = None
    throttle_seconds: int = 300  # 5 minutes default


@dataclass
class Alert:
    """Represents a privacy alert to be sent.

    Attributes:
        alert_id: Unique identifier
        title: Alert title
        message: Alert body message
        severity: Alert severity level
        source: Source of the alert (e.g., anomaly detector)
        anomaly_type: Type of anomaly if from detector
        created_at: Timestamp of alert creation
        acknowledged: Whether alert has been acknowledged
        acknowledged_at: When alert was acknowledged
        metadata: Additional alert context
    """

    alert_id: str
    title: str
    message: str
    severity: str
    source: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    anomaly_type: Optional[str] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Convert alert to dictionary for storage."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "source": self.source,
            "anomaly_type": self.anomaly_type,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Alert":
        """Create Alert from dictionary."""
        return cls(
            alert_id=str(data["alert_id"]),
            title=str(data["title"]),
            message=str(data["message"]),
            severity=str(data["severity"]),
            source=str(data["source"]),
            anomaly_type=data.get("anomaly_type"),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            acknowledged=bool(data.get("acknowledged", False)),
            acknowledged_at=datetime.fromisoformat(str(data["acknowledged_at"])) if data.get("acknowledged_at") else None,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_anomaly(cls, anomaly: "Anomaly") -> "Alert":
        """Create Alert from an Anomaly object."""
        import uuid

        severity_map = {
            "info": "info",
            "warning": "warning",
            "critical": "critical",
        }

        return cls(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            title=f"Privacy Alert: {anomaly.anomaly_type.value.replace('_', ' ').title()}",
            message=anomaly.message,
            severity=severity_map.get(anomaly.severity.value, "info"),
            source="anomaly_detector",
            anomaly_type=anomaly.anomaly_type.value,
            created_at=anomaly.detected_at,
            metadata=anomaly.metadata,
        )


class AlertManager:
    """Manages privacy alert notifications across multiple channels.

    Example:
        >>> manager = AlertManager(AlertConfig(alert_dir=Path("~/.futurnal/alerts")))
        >>> manager.add_channel(AlertChannel.DESKTOP)
        >>> manager.add_channel(AlertChannel.LOG_FILE)
        >>> manager.send_alert(alert)
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """Initialize alert manager.

        Args:
            config: Alert configuration (uses defaults if None)
        """
        self.config = config or AlertConfig()
        self._channels: Set[AlertChannel] = set()
        self._callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

        # Alert history
        self._alerts: List[Alert] = []
        self._alert_lock = threading.Lock()

        # Throttling
        self._last_alert_times: Dict[str, datetime] = {}

        # Initialize alert directory
        if self.config.alert_dir:
            self.config.alert_dir.mkdir(parents=True, exist_ok=True)
            self._load_alerts()

    def add_channel(self, channel: AlertChannel) -> None:
        """Add a notification channel.

        Args:
            channel: Channel to add
        """
        with self._lock:
            self._channels.add(channel)

    def remove_channel(self, channel: AlertChannel) -> None:
        """Remove a notification channel.

        Args:
            channel: Channel to remove
        """
        with self._lock:
            self._channels.discard(channel)

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback function for alerts.

        Args:
            callback: Function to call with alert
        """
        with self._lock:
            self._callbacks.append(callback)

    def send_alert(self, alert: Alert) -> bool:
        """Send an alert through all configured channels.

        Args:
            alert: Alert to send

        Returns:
            True if alert was sent (not throttled)
        """
        # Check severity threshold
        severity_order = ["info", "warning", "critical"]
        if severity_order.index(alert.severity) < severity_order.index(self.config.min_severity):
            return False

        # Check throttling
        throttle_key = f"{alert.anomaly_type}:{alert.source}"
        now = datetime.utcnow()

        with self._lock:
            last_time = self._last_alert_times.get(throttle_key)
            if last_time and (now - last_time).total_seconds() < self.config.throttle_seconds:
                logger.debug(f"Alert throttled: {throttle_key}")
                return False
            self._last_alert_times[throttle_key] = now

        # Store alert
        self._store_alert(alert)

        # Send to channels
        channels = list(self._channels)
        callbacks = list(self._callbacks)

        for channel in channels:
            try:
                self._send_to_channel(channel, alert)
            except Exception as e:
                logger.error(f"Failed to send alert to {channel.value}: {e}")

        for callback in callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        return True

    def send_anomaly_alert(self, anomaly: "Anomaly") -> bool:
        """Create and send an alert from an anomaly.

        Args:
            anomaly: Anomaly to create alert from

        Returns:
            True if alert was sent
        """
        alert = Alert.from_anomaly(anomaly)
        return self.send_alert(alert)

    def _send_to_channel(self, channel: AlertChannel, alert: Alert) -> None:
        """Send alert to a specific channel."""
        if channel == AlertChannel.DESKTOP:
            self._send_desktop_notification(alert)
        elif channel == AlertChannel.LOG_FILE:
            self._send_to_log_file(alert)
        elif channel == AlertChannel.IPC:
            self._send_ipc_notification(alert)

    def _send_desktop_notification(self, alert: Alert) -> None:
        """Send OS-level desktop notification."""
        if not self.config.desktop_notifications:
            return

        try:
            # Try platform-specific notification
            import platform
            system = platform.system()

            if system == "Darwin":  # macOS
                self._send_macos_notification(alert)
            elif system == "Linux":
                self._send_linux_notification(alert)
            elif system == "Windows":
                self._send_windows_notification(alert)
            else:
                logger.warning(f"Desktop notifications not supported on {system}")

        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")

    def _send_macos_notification(self, alert: Alert) -> None:
        """Send notification on macOS using osascript."""
        import subprocess

        title = alert.title.replace('"', '\\"')
        message = alert.message.replace('"', '\\"')

        script = f'''display notification "{message}" with title "{title}"'''

        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            logger.debug("macOS notification timed out")
        except subprocess.CalledProcessError as e:
            logger.debug(f"macOS notification failed: {e}")

    def _send_linux_notification(self, alert: Alert) -> None:
        """Send notification on Linux using notify-send."""
        import subprocess

        urgency_map = {
            "info": "low",
            "warning": "normal",
            "critical": "critical",
        }
        urgency = urgency_map.get(alert.severity, "normal")

        try:
            subprocess.run(
                [
                    "notify-send",
                    "-u", urgency,
                    alert.title,
                    alert.message,
                ],
                check=True,
                capture_output=True,
                timeout=5,
            )
        except FileNotFoundError:
            logger.debug("notify-send not available")
        except subprocess.TimeoutExpired:
            logger.debug("Linux notification timed out")
        except subprocess.CalledProcessError as e:
            logger.debug(f"Linux notification failed: {e}")

    def _send_windows_notification(self, alert: Alert) -> None:
        """Send notification on Windows using toast notification."""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                alert.title,
                alert.message,
                duration=5,
                threaded=True,
            )
        except ImportError:
            logger.debug("win10toast not available for Windows notifications")
        except Exception as e:
            logger.debug(f"Windows notification failed: {e}")

    def _send_to_log_file(self, alert: Alert) -> None:
        """Write alert to log file."""
        if not self.config.alert_dir:
            return

        log_file = self.config.alert_dir / "alerts.jsonl"

        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write alert to log: {e}")

    def _send_ipc_notification(self, alert: Alert) -> None:
        """Send alert via IPC to desktop application.

        Uses a simple JSON file approach for cross-platform compatibility.
        Desktop app polls this file for new alerts.
        """
        if not self.config.ipc_socket_path:
            # Default to alerts dir
            ipc_dir = self.config.alert_dir or Path.home() / ".futurnal" / "alerts"
            ipc_dir.mkdir(parents=True, exist_ok=True)
            ipc_file = ipc_dir / "ipc_alerts.json"
        else:
            ipc_file = self.config.ipc_socket_path

        try:
            # Read existing alerts
            existing = []
            if ipc_file.exists():
                try:
                    existing = json.loads(ipc_file.read_text())
                except json.JSONDecodeError:
                    existing = []

            # Add new alert
            existing.append(alert.to_dict())

            # Keep only recent alerts (last 50)
            existing = existing[-50:]

            # Write back
            ipc_file.write_text(json.dumps(existing, indent=2))

        except Exception as e:
            logger.debug(f"IPC notification failed: {e}")

    def _store_alert(self, alert: Alert) -> None:
        """Store alert in memory and persist."""
        with self._alert_lock:
            self._alerts.append(alert)
            # Keep only recent alerts in memory
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-1000:]

    def _load_alerts(self) -> None:
        """Load alerts from log file."""
        if not self.config.alert_dir:
            return

        log_file = self.config.alert_dir / "alerts.jsonl"
        if not log_file.exists():
            return

        try:
            with log_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            alert = Alert.from_dict(data)
                            self._alerts.append(alert)
                        except (json.JSONDecodeError, KeyError):
                            continue

            # Keep only recent alerts
            cutoff = datetime.utcnow() - timedelta(days=self.config.retention_days)
            self._alerts = [a for a in self._alerts if a.created_at > cutoff]

        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")

    def get_alerts(
        self,
        severity: Optional[str] = None,
        unacknowledged_only: bool = False,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alerts with optional filtering.

        Args:
            severity: Filter by severity level
            unacknowledged_only: Only return unacknowledged alerts
            since: Only return alerts after this time
            limit: Maximum number of alerts to return

        Returns:
            List of matching alerts (most recent first)
        """
        with self._alert_lock:
            alerts = list(self._alerts)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        if since:
            alerts = [a for a in alerts if a.created_at > since]

        # Sort by created_at descending
        alerts.sort(key=lambda a: a.created_at, reverse=True)

        return alerts[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged.

        Args:
            alert_id: ID of alert to acknowledge

        Returns:
            True if alert was found and acknowledged
        """
        with self._alert_lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_at = datetime.utcnow()
                    return True
        return False

    def acknowledge_all(self) -> int:
        """Acknowledge all unacknowledged alerts.

        Returns:
            Number of alerts acknowledged
        """
        count = 0
        now = datetime.utcnow()

        with self._alert_lock:
            for alert in self._alerts:
                if not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_at = now
                    count += 1

        return count

    def get_unacknowledged_count(self) -> int:
        """Get count of unacknowledged alerts."""
        with self._alert_lock:
            return sum(1 for a in self._alerts if not a.acknowledged)

    def cleanup_old_alerts(self) -> int:
        """Remove alerts older than retention period.

        Returns:
            Number of alerts removed
        """
        cutoff = datetime.utcnow() - timedelta(days=self.config.retention_days)

        with self._alert_lock:
            original_count = len(self._alerts)
            self._alerts = [a for a in self._alerts if a.created_at > cutoff]
            removed = original_count - len(self._alerts)

        # Also clean up log file
        if self.config.alert_dir and removed > 0:
            self._rewrite_alert_log()

        return removed

    def _rewrite_alert_log(self) -> None:
        """Rewrite alert log with current alerts."""
        if not self.config.alert_dir:
            return

        log_file = self.config.alert_dir / "alerts.jsonl"

        try:
            with self._alert_lock:
                alerts = list(self._alerts)

            with log_file.open("w", encoding="utf-8") as f:
                for alert in alerts:
                    f.write(json.dumps(alert.to_dict()) + "\n")

        except Exception as e:
            logger.error(f"Failed to rewrite alert log: {e}")


def create_alert_manager(
    alert_dir: Optional[Path] = None,
    enable_desktop: bool = True,
    enable_ipc: bool = True,
) -> AlertManager:
    """Create an alert manager with standard configuration.

    Args:
        alert_dir: Directory for alert storage (default: ~/.futurnal/alerts)
        enable_desktop: Enable desktop notifications
        enable_ipc: Enable IPC notifications to desktop app

    Returns:
        Configured AlertManager instance
    """
    if alert_dir is None:
        alert_dir = Path.home() / ".futurnal" / "alerts"

    config = AlertConfig(
        alert_dir=alert_dir,
        desktop_notifications=enable_desktop,
    )

    manager = AlertManager(config)

    if enable_desktop:
        manager.add_channel(AlertChannel.DESKTOP)

    manager.add_channel(AlertChannel.LOG_FILE)

    if enable_ipc:
        manager.add_channel(AlertChannel.IPC)

    return manager


__all__ = [
    "AlertManager",
    "AlertConfig",
    "Alert",
    "AlertChannel",
    "create_alert_manager",
]
