"""IMAP orchestrator integration utilities.

This module provides helpers for registering IMAP mailboxes with the
IngestionOrchestrator for scheduled sync operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from futurnal.orchestrator.scheduler import IngestionOrchestrator, SourceRegistration

from futurnal.orchestrator.models import JobPriority, JobType

from .descriptor import ImapMailboxDescriptor

logger = logging.getLogger(__name__)


class ImapSourceRegistration:
    """Helper for registering IMAP mailboxes with the orchestrator."""

    @staticmethod
    def register_mailbox(
        orchestrator: IngestionOrchestrator,
        mailbox_descriptor: ImapMailboxDescriptor,
        schedule: str = "@interval",
        interval_seconds: int = 300,  # 5 minutes default
        priority: JobPriority = JobPriority.NORMAL,
    ) -> SourceRegistration:
        """Register an IMAP mailbox with the orchestrator for scheduled sync.

        This creates a SourceRegistration that schedules periodic sync jobs
        via APScheduler. Each sync job will call ImapEmailConnector.sync_mailbox()
        via the orchestrator's job queue.

        Args:
            orchestrator: IngestionOrchestrator instance
            mailbox_descriptor: Mailbox descriptor with sync configuration
            schedule: Schedule type ("@interval", "@manual", or cron expression)
            interval_seconds: Interval in seconds for "@interval" schedule
            priority: Job priority (LOW, NORMAL, HIGH)

        Returns:
            SourceRegistration that was registered with the orchestrator

        Example:
            >>> orchestrator = IngestionOrchestrator(...)
            >>> descriptor = mailbox_registry.get("mailbox-id")
            >>> registration = ImapSourceRegistration.register_mailbox(
            ...     orchestrator,
            ...     descriptor,
            ...     interval_seconds=300,  # Sync every 5 minutes
            ... )
        """
        from futurnal.orchestrator.scheduler import SourceRegistration

        # Create LocalIngestionSource for orchestrator compatibility
        local_source = mailbox_descriptor.to_local_source(
            workspace_root=orchestrator._workspace_dir,
            max_workers=None,  # Use orchestrator default
            max_files_per_batch=None,
            schedule=schedule,
            priority=priority.name.lower(),
        )

        # Create SourceRegistration
        registration = SourceRegistration(
            source=local_source,
            schedule=schedule,
            interval_seconds=interval_seconds if schedule == "@interval" else None,
            priority=priority,
            paused=False,
        )

        # Register with orchestrator
        orchestrator.register_source(registration)

        logger.info(
            f"Registered IMAP mailbox with orchestrator",
            extra={
                "mailbox_id": mailbox_descriptor.id,
                "email": mailbox_descriptor.email_address,
                "schedule": schedule,
                "interval_seconds": interval_seconds,
                "priority": priority.name,
            }
        )

        return registration

    @staticmethod
    def unregister_mailbox(
        orchestrator: IngestionOrchestrator,
        mailbox_descriptor: ImapMailboxDescriptor,
    ) -> None:
        """Unregister an IMAP mailbox from the orchestrator.

        Stops scheduled sync jobs for the mailbox.

        Args:
            orchestrator: IngestionOrchestrator instance
            mailbox_descriptor: Mailbox descriptor to unregister
        """
        source_name = mailbox_descriptor.name or f"imap-{mailbox_descriptor.id[:8]}"

        if source_name in orchestrator._sources:
            del orchestrator._sources[source_name]

            # Remove APScheduler jobs
            job_id_patterns = [
                f"interval-{source_name}",
                f"schedule-{source_name}",
                f"fallback-{source_name}",
            ]

            for job_id in job_id_patterns:
                try:
                    orchestrator._scheduler.remove_job(job_id)
                except Exception:
                    pass  # Job may not exist

            logger.info(
                f"Unregistered IMAP mailbox from orchestrator",
                extra={
                    "mailbox_id": mailbox_descriptor.id,
                    "email": mailbox_descriptor.email_address,
                }
            )


def create_imap_job_payload(mailbox_id: str, trigger: str = "schedule") -> dict:
    """Create job payload for IMAP sync job.

    Args:
        mailbox_id: Mailbox identifier
        trigger: Trigger type (schedule, manual, watcher, retry)

    Returns:
        Job payload dictionary
    """
    return {
        "mailbox_id": mailbox_id,
        "trigger": trigger,
    }


__all__ = [
    "ImapSourceRegistration",
    "create_imap_job_payload",
]
