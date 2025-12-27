"""Notification configuration and preferences.

Phase 2D: User notification preferences including:
- Notification frequency
- Do-not-disturb schedules
- Channel preferences
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationFrequency(str, Enum):
    """How often to batch and deliver notifications."""

    IMMEDIATE = "immediate"  # Deliver as soon as available
    HOURLY = "hourly"  # Batch per hour
    DAILY = "daily"  # One batch per day
    WEEKLY = "weekly"  # One batch per week
    MANUAL = "manual"  # Only on user request


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""

    URGENT = "urgent"  # Always notify immediately
    HIGH = "high"  # Bypass hourly batching
    NORMAL = "normal"  # Standard batching
    LOW = "low"  # Only in daily/weekly digests


@dataclass
class DoNotDisturbSchedule:
    """Do-not-disturb time window.

    Attributes:
        enabled: Whether DND is enabled
        start_time: Start of DND period (HH:MM)
        end_time: End of DND period (HH:MM)
        days: Days of week to apply (0=Monday, 6=Sunday)
    """

    enabled: bool = True
    start_time: str = "22:00"  # 10 PM
    end_time: str = "08:00"  # 8 AM
    days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])  # All days

    def is_dnd_active(self) -> bool:
        """Check if DND is currently active."""
        if not self.enabled:
            return False

        from datetime import datetime

        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()

        if current_day not in self.days:
            return False

        try:
            start = time.fromisoformat(self.start_time)
            end = time.fromisoformat(self.end_time)

            # Handle overnight DND (e.g., 22:00 to 08:00)
            if start > end:
                return current_time >= start or current_time <= end
            else:
                return start <= current_time <= end
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "days": self.days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoNotDisturbSchedule":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            start_time=data.get("start_time", "22:00"),
            end_time=data.get("end_time", "08:00"),
            days=data.get("days", [0, 1, 2, 3, 4, 5, 6]),
        )


@dataclass
class ChannelPreferences:
    """Preferences for notification channels.

    Attributes:
        dashboard_enabled: Show in-app dashboard notifications
        desktop_enabled: Show native OS notifications
        min_priority_desktop: Minimum priority for desktop notifications
    """

    dashboard_enabled: bool = True
    desktop_enabled: bool = True
    min_priority_desktop: NotificationPriority = NotificationPriority.HIGH

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dashboard_enabled": self.dashboard_enabled,
            "desktop_enabled": self.desktop_enabled,
            "min_priority_desktop": self.min_priority_desktop.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelPreferences":
        """Create from dictionary."""
        return cls(
            dashboard_enabled=data.get("dashboard_enabled", True),
            desktop_enabled=data.get("desktop_enabled", True),
            min_priority_desktop=NotificationPriority(
                data.get("min_priority_desktop", "high")
            ),
        )


@dataclass
class NotificationPreferences:
    """User notification preferences.

    Attributes:
        frequency: Batching frequency
        dnd_schedule: Do-not-disturb configuration
        channels: Channel-specific preferences
        max_daily_notifications: Maximum notifications per day
        min_insight_confidence: Minimum confidence for notifications
    """

    frequency: NotificationFrequency = NotificationFrequency.DAILY
    dnd_schedule: DoNotDisturbSchedule = field(default_factory=DoNotDisturbSchedule)
    channels: ChannelPreferences = field(default_factory=ChannelPreferences)
    max_daily_notifications: int = 10
    min_insight_confidence: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frequency": self.frequency.value,
            "dnd_schedule": self.dnd_schedule.to_dict(),
            "channels": self.channels.to_dict(),
            "max_daily_notifications": self.max_daily_notifications,
            "min_insight_confidence": self.min_insight_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationPreferences":
        """Create from dictionary."""
        return cls(
            frequency=NotificationFrequency(data.get("frequency", "daily")),
            dnd_schedule=DoNotDisturbSchedule.from_dict(data.get("dnd_schedule", {})),
            channels=ChannelPreferences.from_dict(data.get("channels", {})),
            max_daily_notifications=data.get("max_daily_notifications", 10),
            min_insight_confidence=data.get("min_insight_confidence", 0.6),
        )


@dataclass
class NotificationConfig:
    """Notification system configuration.

    Stores preferences and notification history in:
    ~/.futurnal/notifications/config.json

    Attributes:
        preferences: User preferences
        last_notification_time: Timestamp of last notification sent
        notifications_today: Count of notifications sent today
    """

    preferences: NotificationPreferences = field(default_factory=NotificationPreferences)
    last_notification_time: Optional[str] = None
    notifications_today: int = 0
    last_reset_date: Optional[str] = None

    DEFAULT_CONFIG_PATH = "~/.futurnal/notifications/config.json"

    def __init__(self, config_path: Optional[str] = None):
        """Initialize notification config.

        Args:
            config_path: Path to config file
        """
        self._config_path = Path(
            os.path.expanduser(config_path or self.DEFAULT_CONFIG_PATH)
        )
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        self.preferences = NotificationPreferences()
        self.last_notification_time = None
        self.notifications_today = 0
        self.last_reset_date = None

        self._load()
        self._reset_daily_count_if_needed()

        logger.info(
            f"NotificationConfig initialized "
            f"(frequency={self.preferences.frequency.value}, path={self._config_path})"
        )

    def _load(self) -> None:
        """Load config from file."""
        if not self._config_path.exists():
            return

        try:
            data = json.loads(self._config_path.read_text())
            self.preferences = NotificationPreferences.from_dict(
                data.get("preferences", {})
            )
            self.last_notification_time = data.get("last_notification_time")
            self.notifications_today = data.get("notifications_today", 0)
            self.last_reset_date = data.get("last_reset_date")
        except Exception as e:
            logger.warning(f"Failed to load notification config: {e}")

    def _save(self) -> None:
        """Save config to file."""
        try:
            data = {
                "preferences": self.preferences.to_dict(),
                "last_notification_time": self.last_notification_time,
                "notifications_today": self.notifications_today,
                "last_reset_date": self.last_reset_date,
            }
            self._config_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved notification config to {self._config_path}")
        except Exception as e:
            logger.warning(f"Failed to save notification config: {e}")

    def _reset_daily_count_if_needed(self) -> None:
        """Reset daily notification count if it's a new day."""
        from datetime import date

        today = date.today().isoformat()
        if self.last_reset_date != today:
            self.notifications_today = 0
            self.last_reset_date = today
            self._save()

    def update_preferences(self, **kwargs) -> None:
        """Update preferences with provided values.

        Args:
            **kwargs: Preference fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.preferences, key):
                setattr(self.preferences, key, value)
        self._save()

    def record_notification_sent(self) -> None:
        """Record that a notification was sent."""
        from datetime import datetime

        self.last_notification_time = datetime.utcnow().isoformat()
        self.notifications_today += 1
        self._save()

    def can_send_notification(self, priority: NotificationPriority) -> bool:
        """Check if a notification can be sent now.

        Args:
            priority: Priority of the notification

        Returns:
            True if notification can be sent
        """
        # Urgent always goes through
        if priority == NotificationPriority.URGENT:
            return True

        # Check DND (except for urgent)
        if self.preferences.dnd_schedule.is_dnd_active():
            return False

        # Check daily limit
        if self.notifications_today >= self.preferences.max_daily_notifications:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "preferences": self.preferences.to_dict(),
            "last_notification_time": self.last_notification_time,
            "notifications_today": self.notifications_today,
            "last_reset_date": self.last_reset_date,
            "dnd_active": self.preferences.dnd_schedule.is_dnd_active(),
        }


# Global instance
_notification_config: Optional[NotificationConfig] = None


def get_notification_config() -> NotificationConfig:
    """Get the default notification config singleton."""
    global _notification_config
    if _notification_config is None:
        _notification_config = NotificationConfig()
    return _notification_config
