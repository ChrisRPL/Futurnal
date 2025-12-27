"""Notification system for proactive insight delivery.

Phase 2D: Intelligent Notification System

This module provides the notification infrastructure for delivering
insights to users at appropriate times.
"""

from futurnal.notifications.service import NotificationService, get_notification_service
from futurnal.notifications.channels import (
    NotificationChannel,
    DashboardChannel,
    DesktopChannel,
)
from futurnal.notifications.config import (
    NotificationConfig,
    NotificationPreferences,
    get_notification_config,
)

__all__ = [
    "NotificationService",
    "get_notification_service",
    "NotificationChannel",
    "DashboardChannel",
    "DesktopChannel",
    "NotificationConfig",
    "NotificationPreferences",
    "get_notification_config",
]
