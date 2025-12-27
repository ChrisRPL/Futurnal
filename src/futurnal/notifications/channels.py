"""Notification delivery channels.

Phase 2D: Channel implementations for delivering insights:
- Dashboard: In-app notifications
- Desktop: Native OS notifications via Tauri
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Notification:
    """A notification to be delivered.

    Attributes:
        notification_id: Unique identifier
        title: Notification title
        body: Notification content
        insight_id: Associated insight ID
        priority: Notification priority
        created_at: When notification was created
        delivered_at: When notification was delivered (if any)
        read: Whether notification has been read
        action_url: Optional URL to open on click
        metadata: Additional metadata
    """

    notification_id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    body: str = ""
    insight_id: Optional[str] = None
    priority: str = "normal"
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None
    read: bool = False
    action_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "notification_id": self.notification_id,
            "title": self.title,
            "body": self.body,
            "insight_id": self.insight_id,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "read": self.read,
            "action_url": self.action_url,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Notification":
        """Create from dictionary."""
        return cls(
            notification_id=data.get("notification_id", str(uuid4())),
            title=data.get("title", ""),
            body=data.get("body", ""),
            insight_id=data.get("insight_id"),
            priority=data.get("priority", "normal"),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            delivered_at=datetime.fromisoformat(data["delivered_at"])
            if data.get("delivered_at")
            else None,
            read=data.get("read", False),
            action_url=data.get("action_url"),
            metadata=data.get("metadata", {}),
        )


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    def deliver(self, notification: Notification) -> bool:
        """Deliver a notification through this channel.

        Args:
            notification: The notification to deliver

        Returns:
            True if delivery was successful
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this channel is available for delivery.

        Returns:
            True if channel can deliver notifications
        """
        pass


class DashboardChannel(NotificationChannel):
    """In-app dashboard notifications.

    Stores notifications in a JSON file for the frontend to display.
    """

    DEFAULT_NOTIFICATIONS_PATH = "~/.futurnal/notifications/dashboard.json"
    MAX_NOTIFICATIONS = 100

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize dashboard channel.

        Args:
            storage_path: Path to store notifications
        """
        self._storage_path = Path(
            os.path.expanduser(storage_path or self.DEFAULT_NOTIFICATIONS_PATH)
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._notifications: List[Notification] = []
        self._load()

        logger.info(
            f"DashboardChannel initialized ({len(self._notifications)} notifications)"
        )

    def _load(self) -> None:
        """Load notifications from storage."""
        if not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text())
            self._notifications = [Notification.from_dict(n) for n in data]
        except Exception as e:
            logger.warning(f"Failed to load dashboard notifications: {e}")

    def _save(self) -> None:
        """Save notifications to storage."""
        try:
            # Limit to max notifications (keep most recent)
            if len(self._notifications) > self.MAX_NOTIFICATIONS:
                self._notifications = self._notifications[-self.MAX_NOTIFICATIONS :]

            data = [n.to_dict() for n in self._notifications]
            self._storage_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved {len(self._notifications)} dashboard notifications")
        except Exception as e:
            logger.warning(f"Failed to save dashboard notifications: {e}")

    def deliver(self, notification: Notification) -> bool:
        """Add notification to dashboard.

        Args:
            notification: The notification to add

        Returns:
            True (dashboard delivery always succeeds)
        """
        notification.delivered_at = datetime.utcnow()
        self._notifications.append(notification)
        self._save()

        logger.info(f"Delivered notification to dashboard: {notification.title[:50]}")
        return True

    def is_available(self) -> bool:
        """Dashboard is always available."""
        return True

    def get_unread(self) -> List[Notification]:
        """Get unread notifications.

        Returns:
            List of unread notifications
        """
        return [n for n in self._notifications if not n.read]

    def get_all(self, limit: int = 50) -> List[Notification]:
        """Get all notifications.

        Args:
            limit: Maximum number to return

        Returns:
            List of notifications (most recent first)
        """
        return list(reversed(self._notifications[-limit:]))

    def mark_read(self, notification_id: str) -> bool:
        """Mark a notification as read.

        Args:
            notification_id: The notification ID

        Returns:
            True if notification was found and marked
        """
        for n in self._notifications:
            if n.notification_id == notification_id:
                n.read = True
                self._save()
                return True
        return False

    def clear_all(self) -> int:
        """Clear all notifications.

        Returns:
            Number of notifications cleared
        """
        count = len(self._notifications)
        self._notifications = []
        self._save()
        return count


class DesktopChannel(NotificationChannel):
    """Native OS notifications via Tauri.

    Sends notifications through Tauri's notification API.
    This channel requires the Tauri app to be running.
    """

    def __init__(self):
        """Initialize desktop channel."""
        logger.info("DesktopChannel initialized")

    def deliver(self, notification: Notification) -> bool:
        """Trigger desktop notification.

        This creates a marker file that Tauri can watch for.
        In practice, the frontend should call the Tauri notification API directly.

        Args:
            notification: The notification to send

        Returns:
            True if marker file was created
        """
        try:
            # Create a pending notification file for Tauri to pick up
            pending_dir = Path(os.path.expanduser("~/.futurnal/notifications/pending"))
            pending_dir.mkdir(parents=True, exist_ok=True)

            pending_file = pending_dir / f"{notification.notification_id}.json"
            pending_file.write_text(json.dumps(notification.to_dict(), indent=2))

            logger.info(f"Created pending desktop notification: {notification.title[:50]}")
            return True

        except Exception as e:
            logger.warning(f"Failed to create desktop notification: {e}")
            return False

    def is_available(self) -> bool:
        """Check if desktop notifications are available.

        Returns:
            True if Tauri app appears to be running
        """
        # Simple check: if pending directory exists, assume Tauri might be watching
        pending_dir = Path(os.path.expanduser("~/.futurnal/notifications/pending"))
        return pending_dir.exists()

    def clear_pending(self) -> int:
        """Clear pending desktop notifications.

        Returns:
            Number of pending notifications cleared
        """
        pending_dir = Path(os.path.expanduser("~/.futurnal/notifications/pending"))
        if not pending_dir.exists():
            return 0

        count = 0
        for f in pending_dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass

        return count
