"""GitHub Repository Connector - integrates sync with Futurnal pipeline.

This module provides the main connector class that orchestrates GitHub repository
synchronization and integrates with the Futurnal ingestion pipeline through
ElementSink for PKG ingestion.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

from .descriptor import GitHubRepositoryDescriptor
from .sync_models import FileContent, SyncResult, SyncState, SyncStrategy
from .sync_orchestrator import GitHubSyncOrchestrator
from ...privacy.audit import AuditEvent, AuditLogger
from ...privacy.consent import ConsentRegistry, ConsentRequiredError

logger = logging.getLogger(__name__)


@runtime_checkable
class ElementSink(Protocol):
    """Sink interface for handling parsed elements."""

    def handle(self, element: dict) -> None:
        ...

    def handle_deletion(self, element: dict) -> None:  # pragma: no cover - optional
        ...


@dataclass
class GitHubRepositoryConnector:
    """Connector for GitHub repository ingestion into Futurnal pipeline.

    This connector orchestrates GitHub repository synchronization and processes
    synced files through the Futurnal pipeline via ElementSink integration.
    """

    descriptor: GitHubRepositoryDescriptor
    sync_orchestrator: GitHubSyncOrchestrator
    element_sink: Optional[ElementSink] = None
    audit_logger: Optional[AuditLogger] = None
    consent_registry: Optional[ConsentRegistry] = None

    def __post_init__(self) -> None:
        """Initialize connector."""
        # Validate descriptor
        if not self.descriptor:
            raise ValueError("Repository descriptor is required")

        if not self.sync_orchestrator:
            raise ValueError("Sync orchestrator is required")

    async def sync(
        self,
        *,
        strategy: Optional[SyncStrategy] = None,
        job_id: Optional[str] = None,
    ) -> SyncResult:
        """Synchronize repository and process files.

        Args:
            strategy: Optional sync strategy (uses defaults if not provided)
            job_id: Optional job identifier for tracking

        Returns:
            SyncResult with sync statistics

        Raises:
            ConsentRequiredError: If required consent is not granted
        """
        active_job = job_id or uuid.uuid4().hex

        logger.info(
            f"Starting GitHub repository sync: {self.descriptor.full_name} "
            f"(job: {active_job})"
        )

        # Check consent if registry provided
        if self.consent_registry:
            self._check_consent(active_job)

        # Emit audit event for sync start
        if self.audit_logger:
            self._emit_audit_event(
                job_id=active_job,
                action="repo_sync_started",
                status="info",
            )

        try:
            # Perform sync via orchestrator
            result = await self.sync_orchestrator.sync_repository(
                descriptor=self.descriptor,
                strategy=strategy,
            )

            # Process synced files if ElementSink provided
            if self.element_sink and result.is_success():
                await self._process_synced_files(result, job_id=active_job)

            # Emit completion audit event
            if self.audit_logger:
                self._emit_audit_event(
                    job_id=active_job,
                    action="repo_sync_completed",
                    status="success",
                    metadata={
                        "files_synced": result.files_synced,
                        "bytes_synced": result.bytes_synced,
                        "duration_seconds": result.duration_seconds,
                    },
                )

            logger.info(
                f"Sync completed: {result.files_synced} files, "
                f"{result.bytes_synced_mb:.2f} MB"
            )

            return result

        except ConsentRequiredError:
            # Re-raise consent errors
            raise

        except Exception as e:
            logger.error(f"Sync failed: {e}", exc_info=True)

            # Emit failure audit event
            if self.audit_logger:
                self._emit_audit_event(
                    job_id=active_job,
                    action="repo_sync_failed",
                    status="error",
                    metadata={"error": str(e)},
                )

            raise

    async def _process_synced_files(
        self,
        result: SyncResult,
        *,
        job_id: str,
    ) -> None:
        """Process synced files through ElementSink.

        Args:
            result: Sync result with file information
            job_id: Job identifier
        """
        if not self.element_sink:
            return

        logger.info(f"Processing {result.files_synced} files through pipeline")

        # Note: In a full implementation, we would retrieve the actual FileContent
        # objects from the sync result and process each one. For now, this is a
        # placeholder that shows the integration pattern.

        # The actual file processing would happen in the sync implementations
        # (GraphQLRepositorySync, GitCloneRepositorySync) which would need to
        # be extended to store FileContent objects that we can then process here.

        logger.debug(
            f"File processing complete (job: {job_id}). "
            f"In production, this would process actual FileContent objects."
        )

    def _check_consent(self, job_id: str) -> None:
        """Check if required consent is granted.

        Args:
            job_id: Job identifier

        Raises:
            ConsentRequiredError: If consent not granted
        """
        if not self.consent_registry:
            return

        required_scopes = self.descriptor.get_required_consent_scopes()

        for scope in required_scopes:
            if not self.consent_registry.has_consent(scope):
                logger.warning(
                    f"Consent not granted for scope: {scope} (job: {job_id})"
                )
                raise ConsentRequiredError(
                    f"Consent required for GitHub repository access: {scope}"
                )

    def _emit_audit_event(
        self,
        *,
        job_id: str,
        action: str,
        status: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Emit audit event.

        Args:
            job_id: Job identifier
            action: Action being performed
            status: Status (info, success, error)
            metadata: Optional additional metadata
        """
        if not self.audit_logger:
            return

        # Build redaction policy from descriptor
        policy = self.descriptor.build_redaction_policy(allow_plaintext=False)

        # Redact repository identifier
        redacted_identifier = policy.apply(self.descriptor.full_name)

        event_metadata = {
            "repo_id": self.descriptor.id,
            "owner": self.descriptor.owner,
            "repo": self.descriptor.repo,
            "full_name": self.descriptor.full_name,
            "sync_mode": self.descriptor.sync_mode.value,
            "visibility": self.descriptor.visibility.value,
        }

        if metadata:
            event_metadata.update(metadata)

        event = AuditEvent(
            job_id=job_id,
            source="github_repository_connector",
            action=action,
            status=status,
            timestamp=datetime.now(timezone.utc),
            redacted_path=redacted_identifier.redacted,
            path_hash=redacted_identifier.path_hash,
            metadata=event_metadata,
        )

        try:
            self.audit_logger.record(event)
        except Exception as e:
            # Don't fail operations due to audit logging issues
            logger.error(f"Failed to emit audit event: {e}")

    def get_sync_status(self) -> Optional[SyncState]:
        """Get current sync status.

        Returns:
            SyncState if exists, None otherwise
        """
        return self.sync_orchestrator.get_sync_status(self.descriptor.id)

    def get_statistics(self) -> dict:
        """Get sync statistics.

        Returns:
            Dictionary of statistics
        """
        return self.sync_orchestrator.state_manager.get_statistics(
            self.descriptor.id
        )


__all__ = [
    "ElementSink",
    "GitHubRepositoryConnector",
]
