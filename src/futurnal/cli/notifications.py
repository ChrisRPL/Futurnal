"""Notification CLI commands for Phase 2D.

Provides commands for:
- Viewing notification preferences
- Updating notification settings
- Viewing notification history
- Managing do-not-disturb schedules
"""

import json
import logging
from typing import Optional

from typer import Typer, Option, Argument

logger = logging.getLogger(__name__)

notifications_app = Typer(help="Notification preferences and history commands")


def _get_notification_service():
    """Get NotificationService singleton."""
    try:
        from futurnal.notifications.service import get_notification_service
        return get_notification_service()
    except ImportError as e:
        logger.warning(f"Notification service not available: {e}")
        return None


def _get_notification_config():
    """Get NotificationConfig singleton."""
    try:
        from futurnal.notifications.config import get_notification_config
        return get_notification_config()
    except ImportError as e:
        logger.warning(f"Notification config not available: {e}")
        return None


# ============================================================================
# Preferences Commands
# ============================================================================

@notifications_app.command("preferences")
def get_preferences(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get current notification preferences.

    Examples:
        futurnal notifications preferences
        futurnal notifications preferences --json
    """
    try:
        config = _get_notification_config()
        if config is None:
            raise RuntimeError("Notification system not available")

        prefs = config.preferences
        data = {
            "success": True,
            "frequency": prefs.frequency.value,
            "maxDailyNotifications": prefs.max_daily_notifications,
            "minInsightConfidence": prefs.min_insight_confidence,
            "dndSchedule": {
                "enabled": prefs.dnd_schedule.enabled,
                "startTime": prefs.dnd_schedule.start_time,
                "endTime": prefs.dnd_schedule.end_time,
                "days": prefs.dnd_schedule.days,
                "isActive": prefs.dnd_schedule.is_dnd_active(),
            },
            "channels": {
                "dashboardEnabled": prefs.channels.dashboard_enabled,
                "desktopEnabled": prefs.channels.desktop_enabled,
                "minPriorityDesktop": prefs.channels.min_priority_desktop.value,
            },
        }

        if output_json:
            print(json.dumps(data))
        else:
            print("\nNotification Preferences")
            print("-" * 40)
            print(f"Frequency: {prefs.frequency.value}")
            print(f"Max Daily: {prefs.max_daily_notifications}")
            print(f"Min Confidence: {prefs.min_insight_confidence:.0%}")
            print()
            print("Do Not Disturb:")
            dnd = prefs.dnd_schedule
            status = "ACTIVE" if dnd.is_dnd_active() else "inactive"
            print(f"  Status: {status}")
            print(f"  Hours: {dnd.start_time} - {dnd.end_time}")
            print(f"  Enabled: {dnd.enabled}")
            print()
            print("Channels:")
            print(f"  Dashboard: {'enabled' if prefs.channels.dashboard_enabled else 'disabled'}")
            print(f"  Desktop: {'enabled' if prefs.channels.desktop_enabled else 'disabled'}")

    except Exception as e:
        logger.error(f"Get preferences failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


@notifications_app.command("set-frequency")
def set_frequency(
    frequency: str = Argument(..., help="Frequency: immediate, hourly, daily, weekly, manual"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Set notification frequency.

    Examples:
        futurnal notifications set-frequency daily
        futurnal notifications set-frequency hourly --json
    """
    try:
        from futurnal.notifications.config import NotificationFrequency

        # Validate frequency
        try:
            freq_enum = NotificationFrequency(frequency)
        except ValueError:
            valid = ", ".join(f.value for f in NotificationFrequency)
            raise ValueError(f"Invalid frequency '{frequency}'. Valid: {valid}")

        config = _get_notification_config()
        if config is None:
            raise RuntimeError("Notification system not available")

        config.preferences.frequency = freq_enum
        config._save()

        if output_json:
            print(json.dumps({
                "success": True,
                "frequency": freq_enum.value,
            }))
        else:
            print(f"Notification frequency set to: {freq_enum.value}")

    except Exception as e:
        logger.error(f"Set frequency failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


@notifications_app.command("set-dnd")
def set_dnd(
    enabled: bool = Option(None, "--enabled/--disabled", help="Enable or disable DND"),
    start: Optional[str] = Option(None, "--start", help="Start time (HH:MM)"),
    end: Optional[str] = Option(None, "--end", help="End time (HH:MM)"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Set do-not-disturb schedule.

    Examples:
        futurnal notifications set-dnd --start 22:00 --end 08:00
        futurnal notifications set-dnd --disabled
        futurnal notifications set-dnd --enabled --json
    """
    try:
        config = _get_notification_config()
        if config is None:
            raise RuntimeError("Notification system not available")

        dnd = config.preferences.dnd_schedule

        if enabled is not None:
            dnd.enabled = enabled

        if start is not None:
            dnd.start_time = start

        if end is not None:
            dnd.end_time = end

        config._save()

        if output_json:
            print(json.dumps({
                "success": True,
                "dndSchedule": dnd.to_dict(),
                "isActive": dnd.is_dnd_active(),
            }))
        else:
            status = "ACTIVE" if dnd.is_dnd_active() else "inactive"
            print(f"DND schedule updated: {dnd.start_time} - {dnd.end_time}")
            print(f"Enabled: {dnd.enabled}")
            print(f"Current status: {status}")

    except Exception as e:
        logger.error(f"Set DND failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


@notifications_app.command("set-channels")
def set_channels(
    dashboard: Optional[bool] = Option(None, "--dashboard/--no-dashboard", help="Enable/disable dashboard"),
    desktop: Optional[bool] = Option(None, "--desktop/--no-desktop", help="Enable/disable desktop"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Enable or disable notification channels.

    Examples:
        futurnal notifications set-channels --desktop
        futurnal notifications set-channels --no-desktop --dashboard
    """
    try:
        config = _get_notification_config()
        if config is None:
            raise RuntimeError("Notification system not available")

        channels = config.preferences.channels

        if dashboard is not None:
            channels.dashboard_enabled = dashboard

        if desktop is not None:
            channels.desktop_enabled = desktop

        config._save()

        if output_json:
            print(json.dumps({
                "success": True,
                "channels": channels.to_dict(),
            }))
        else:
            print("Notification channels updated:")
            print(f"  Dashboard: {'enabled' if channels.dashboard_enabled else 'disabled'}")
            print(f"  Desktop: {'enabled' if channels.desktop_enabled else 'disabled'}")

    except Exception as e:
        logger.error(f"Set channels failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


# ============================================================================
# History Commands
# ============================================================================

@notifications_app.command("history")
def get_history(
    limit: int = Option(20, "--limit", "-n", help="Maximum notifications to return"),
    unread_only: bool = Option(False, "--unread", "-u", help="Only show unread"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get notification history.

    Examples:
        futurnal notifications history
        futurnal notifications history --unread --limit 10
        futurnal notifications history --json
    """
    try:
        service = _get_notification_service()
        if service is None:
            raise RuntimeError("Notification service not available")

        if unread_only:
            notifications = [n.to_dict() for n in service.dashboard.get_unread()]
        else:
            notifications = service.get_dashboard_notifications(limit)

        if output_json:
            print(json.dumps({
                "success": True,
                "notifications": notifications,
                "totalCount": len(notifications),
                "unreadCount": service.get_unread_count(),
            }))
        else:
            unread = service.get_unread_count()
            print(f"\nNotification History ({len(notifications)} shown, {unread} unread)")
            print("-" * 60)

            for n in notifications:
                read_marker = "  " if n.get("read") else "●"
                title = n.get("title", "")[:50]
                created = n.get("created_at", "")[:10]
                print(f"{read_marker} [{created}] {title}")

    except Exception as e:
        logger.error(f"Get history failed: {e}")
        if output_json:
            print(json.dumps({
                "success": False,
                "notifications": [],
                "totalCount": 0,
                "unreadCount": 0,
                "error": str(e),
            }))
        else:
            print(f"Error: {e}")


@notifications_app.command("mark-read")
def mark_read(
    notification_id: str = Argument(..., help="Notification ID to mark as read"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Mark a notification as read.

    Examples:
        futurnal notifications mark-read <notification_id>
    """
    try:
        service = _get_notification_service()
        if service is None:
            raise RuntimeError("Notification service not available")

        success = service.mark_notification_read(notification_id)

        if output_json:
            print(json.dumps({
                "success": success,
                "notificationId": notification_id,
            }))
        else:
            if success:
                print(f"Marked notification {notification_id} as read")
            else:
                print(f"Notification {notification_id} not found")

    except Exception as e:
        logger.error(f"Mark read failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


@notifications_app.command("clear")
def clear_notifications(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Clear all notifications.

    Examples:
        futurnal notifications clear
        futurnal notifications clear --json
    """
    try:
        service = _get_notification_service()
        if service is None:
            raise RuntimeError("Notification service not available")

        count = service.dashboard.clear_all()

        if output_json:
            print(json.dumps({
                "success": True,
                "clearedCount": count,
            }))
        else:
            print(f"Cleared {count} notifications")

    except Exception as e:
        logger.error(f"Clear failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "clearedCount": 0, "error": str(e)}))
        else:
            print(f"Error: {e}")


# ============================================================================
# Status Commands
# ============================================================================

@notifications_app.command("status")
def get_status(
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get notification system status.

    Examples:
        futurnal notifications status
        futurnal notifications status --json
    """
    try:
        service = _get_notification_service()
        if service is None:
            raise RuntimeError("Notification service not available")

        status = service.get_status()
        status["success"] = True

        if output_json:
            print(json.dumps(status))
        else:
            print("\nNotification Status")
            print("-" * 40)
            print(f"Pending insights: {status['pending_insights']}")
            print(f"Unread notifications: {status['unread_notifications']}")
            print(f"Sent today: {status['notifications_today']}/{status['max_daily']}")
            print(f"Frequency: {status['frequency']}")
            print(f"DND active: {status['dnd_active']}")
            print(f"Desktop: {'enabled' if status['desktop_enabled'] else 'disabled'}")
            print(f"Dashboard: {'enabled' if status['dashboard_enabled'] else 'disabled'}")

    except Exception as e:
        logger.error(f"Get status failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}")


@notifications_app.command("deliver")
def deliver_pending(
    force: bool = Option(False, "--force", "-f", help="Bypass DND and frequency checks"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Deliver pending insight notifications.

    Examples:
        futurnal notifications deliver
        futurnal notifications deliver --force
    """
    try:
        service = _get_notification_service()
        if service is None:
            raise RuntimeError("Notification service not available")

        delivered = service.deliver_pending(force=force)

        if output_json:
            print(json.dumps({
                "success": True,
                "deliveredCount": delivered,
            }))
        else:
            print(f"Delivered {delivered} insight notifications")

    except Exception as e:
        logger.error(f"Deliver failed: {e}")
        if output_json:
            print(json.dumps({"success": False, "deliveredCount": 0, "error": str(e)}))
        else:
            print(f"Error: {e}")
