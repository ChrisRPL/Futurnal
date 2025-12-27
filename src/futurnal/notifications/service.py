"""Notification service for proactive insight delivery.

Phase 2D: Intelligent Notification System

Key features:
- Multi-channel delivery (dashboard, desktop)
- Do-not-disturb scheduling
- Insight batching and prioritization
- Daily limits and frequency controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from futurnal.notifications.channels import (
    DashboardChannel,
    DesktopChannel,
    Notification,
    NotificationChannel,
)
from futurnal.notifications.config import (
    NotificationConfig,
    NotificationFrequency,
    NotificationPriority,
    get_notification_config,
)

logger = logging.getLogger(__name__)


@dataclass
class InsightBatch:
    """A batch of insights for notification.

    Attributes:
        insights: List of insight dictionaries
        priority: Batch priority (highest insight priority)
        created_at: When batch was created
    """

    insights: List[Dict[str, Any]]
    priority: NotificationPriority = NotificationPriority.NORMAL
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class NotificationService:
    """Service for delivering insights to users proactively.

    Handles:
    - Batching insights based on frequency preferences
    - Respecting DND schedules
    - Multi-channel delivery (dashboard + desktop)
    - Daily notification limits

    Usage:
        service = NotificationService()
        service.schedule_insight(insight)
        service.deliver_pending()
    """

    def __init__(
        self,
        config: Optional[NotificationConfig] = None,
        dashboard_channel: Optional[DashboardChannel] = None,
        desktop_channel: Optional[DesktopChannel] = None,
    ):
        """Initialize notification service.

        Args:
            config: Notification configuration
            dashboard_channel: Dashboard delivery channel
            desktop_channel: Desktop notification channel
        """
        self.config = config or get_notification_config()
        self.dashboard = dashboard_channel or DashboardChannel()
        self.desktop = desktop_channel or DesktopChannel()

        # Pending insights waiting for delivery
        self._pending_insights: List[Dict[str, Any]] = []
        self._last_batch_time: Optional[datetime] = None

        logger.info(
            f"NotificationService initialized "
            f"(frequency={self.config.preferences.frequency.value}, "
            f"channels=dashboard+desktop)"
        )

    def schedule_insight(
        self,
        insight: Dict[str, Any],
        priority: Optional[NotificationPriority] = None,
    ) -> bool:
        """Schedule an insight for notification.

        Args:
            insight: Insight dictionary to notify about
            priority: Override priority (defaults based on insight confidence)

        Returns:
            True if insight was scheduled
        """
        # Determine priority from insight if not provided
        if priority is None:
            confidence = insight.get("confidence", 0.5)
            if confidence >= 0.9:
                priority = NotificationPriority.HIGH
            elif confidence >= 0.7:
                priority = NotificationPriority.NORMAL
            else:
                priority = NotificationPriority.LOW

        # Check if insight meets minimum confidence
        if insight.get("confidence", 0) < self.config.preferences.min_insight_confidence:
            logger.debug(f"Insight below confidence threshold: {insight.get('insightId')}")
            return False

        # Add to pending
        self._pending_insights.append({
            **insight,
            "_notification_priority": priority.value,
            "_scheduled_at": datetime.utcnow().isoformat(),
        })

        logger.info(
            f"Scheduled insight for notification: {insight.get('title', '')[:50]} "
            f"(priority={priority.value})"
        )

        # Check if we should deliver immediately
        if (
            self.config.preferences.frequency == NotificationFrequency.IMMEDIATE
            or priority == NotificationPriority.URGENT
        ):
            self.deliver_pending()

        return True

    def should_deliver_now(self) -> bool:
        """Check if pending insights should be delivered now.

        Considers:
        - Frequency setting
        - DND schedule
        - Daily limits
        - Time since last batch

        Returns:
            True if delivery should proceed
        """
        if not self._pending_insights:
            return False

        # Check DND
        if self.config.preferences.dnd_schedule.is_dnd_active():
            logger.debug("Delivery blocked: DND active")
            return False

        # Check daily limit
        if self.config.notifications_today >= self.config.preferences.max_daily_notifications:
            logger.debug("Delivery blocked: daily limit reached")
            return False

        # Check frequency
        freq = self.config.preferences.frequency

        if freq == NotificationFrequency.IMMEDIATE:
            return True

        if freq == NotificationFrequency.MANUAL:
            return False

        # Calculate time since last batch
        if self._last_batch_time is None:
            return True

        elapsed = datetime.utcnow() - self._last_batch_time

        if freq == NotificationFrequency.HOURLY:
            return elapsed >= timedelta(hours=1)

        if freq == NotificationFrequency.DAILY:
            return elapsed >= timedelta(hours=24)

        if freq == NotificationFrequency.WEEKLY:
            return elapsed >= timedelta(days=7)

        return True

    def deliver_pending(self, force: bool = False) -> int:
        """Deliver pending insights to notification channels.

        Args:
            force: Bypass DND and frequency checks

        Returns:
            Number of insights delivered
        """
        if not self._pending_insights:
            return 0

        # Check if we should deliver
        if not force and not self.should_deliver_now():
            logger.debug("Delivery deferred (frequency/DND check)")
            return 0

        # Group insights into batches
        batches = self._create_batches()

        delivered = 0
        for batch in batches:
            if self._deliver_batch(batch):
                delivered += len(batch.insights)

        # Clear delivered insights
        self._pending_insights = []
        self._last_batch_time = datetime.utcnow()

        logger.info(f"Delivered {delivered} insights in {len(batches)} batch(es)")
        return delivered

    def _create_batches(self) -> List[InsightBatch]:
        """Group pending insights into batches.

        Returns:
            List of InsightBatch objects
        """
        if not self._pending_insights:
            return []

        # Sort by priority
        priority_order = {
            "urgent": 0,
            "high": 1,
            "normal": 2,
            "low": 3,
        }
        sorted_insights = sorted(
            self._pending_insights,
            key=lambda i: priority_order.get(i.get("_notification_priority", "normal"), 2),
        )

        # For immediate/hourly: one batch per priority level
        # For daily/weekly: single digest batch
        freq = self.config.preferences.frequency

        if freq in (NotificationFrequency.DAILY, NotificationFrequency.WEEKLY):
            # Single digest batch
            return [InsightBatch(
                insights=sorted_insights,
                priority=NotificationPriority(
                    sorted_insights[0].get("_notification_priority", "normal")
                ),
            )]

        # Batch by priority
        batches = []
        current_priority = None
        current_batch = []

        for insight in sorted_insights:
            priority = insight.get("_notification_priority", "normal")
            if priority != current_priority:
                if current_batch:
                    batches.append(InsightBatch(
                        insights=current_batch,
                        priority=NotificationPriority(current_priority),
                    ))
                current_priority = priority
                current_batch = [insight]
            else:
                current_batch.append(insight)

        if current_batch:
            batches.append(InsightBatch(
                insights=current_batch,
                priority=NotificationPriority(current_priority),
            ))

        return batches

    def _deliver_batch(self, batch: InsightBatch) -> bool:
        """Deliver a batch of insights.

        Args:
            batch: The batch to deliver

        Returns:
            True if delivery succeeded
        """
        if not batch.insights:
            return False

        # Create notification from batch
        if len(batch.insights) == 1:
            insight = batch.insights[0]
            notification = Notification(
                title=insight.get("title", "New Insight"),
                body=insight.get("description", "")[:200],
                insight_id=insight.get("insightId"),
                priority=batch.priority.value,
                metadata={"batch_size": 1},
            )
        else:
            # Digest notification
            notification = Notification(
                title=f"{len(batch.insights)} New Insights",
                body=self._format_digest(batch.insights),
                priority=batch.priority.value,
                metadata={
                    "batch_size": len(batch.insights),
                    "insight_ids": [i.get("insightId") for i in batch.insights],
                },
            )

        # Deliver to dashboard (always)
        dashboard_ok = self.dashboard.deliver(notification)

        # Deliver to desktop (if priority meets threshold)
        desktop_ok = False
        if self.config.preferences.channels.desktop_enabled:
            min_priority = self.config.preferences.channels.min_priority_desktop
            priority_order = ["urgent", "high", "normal", "low"]

            if priority_order.index(batch.priority.value) <= priority_order.index(min_priority.value):
                desktop_ok = self.desktop.deliver(notification)

        # Record notification sent
        if dashboard_ok or desktop_ok:
            self.config.record_notification_sent()

        return dashboard_ok or desktop_ok

    def _format_digest(self, insights: List[Dict[str, Any]]) -> str:
        """Format multiple insights as a digest.

        Args:
            insights: List of insights

        Returns:
            Formatted digest string
        """
        lines = []
        for insight in insights[:5]:  # Limit to 5
            title = insight.get("title", "Untitled")[:40]
            confidence = insight.get("confidence", 0) * 100
            lines.append(f"• {title} ({confidence:.0f}%)")

        if len(insights) > 5:
            lines.append(f"... and {len(insights) - 5} more")

        return "\n".join(lines)

    def get_pending_count(self) -> int:
        """Get number of pending insights.

        Returns:
            Count of pending insights
        """
        return len(self._pending_insights)

    def get_dashboard_notifications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get dashboard notifications.

        Args:
            limit: Maximum number to return

        Returns:
            List of notification dictionaries
        """
        return [n.to_dict() for n in self.dashboard.get_all(limit)]

    def get_unread_count(self) -> int:
        """Get count of unread dashboard notifications.

        Returns:
            Unread count
        """
        return len(self.dashboard.get_unread())

    def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a dashboard notification as read.

        Args:
            notification_id: The notification ID

        Returns:
            True if marked successfully
        """
        return self.dashboard.mark_read(notification_id)

    def get_status(self) -> Dict[str, Any]:
        """Get notification service status.

        Returns:
            Status dictionary
        """
        return {
            "pending_insights": len(self._pending_insights),
            "unread_notifications": self.get_unread_count(),
            "notifications_today": self.config.notifications_today,
            "max_daily": self.config.preferences.max_daily_notifications,
            "frequency": self.config.preferences.frequency.value,
            "dnd_active": self.config.preferences.dnd_schedule.is_dnd_active(),
            "desktop_enabled": self.config.preferences.channels.desktop_enabled,
            "dashboard_enabled": self.config.preferences.channels.dashboard_enabled,
        }


# Global instance
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get the default notification service singleton."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
