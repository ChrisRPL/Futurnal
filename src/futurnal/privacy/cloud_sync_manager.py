"""Cloud sync consent manager for Firebase PKG metadata backup.

This module provides a high-level manager for cloud sync consent operations,
coordinating with ConsentRegistry for storage and AuditLogger for tracking.

Key responsibilities:
- Enforce consent checks before any sync operation
- Coordinate grant/revoke operations with audit logging
- Provide status queries for UI display
- Handle revocation with immediate propagation
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Callable, Awaitable

from .consent import ConsentRecord, ConsentRegistry, ConsentRequiredError
from .cloud_consent import (
    CloudSyncScope,
    CloudSyncConsentStatus,
    CloudSyncAuditEntry,
    CLOUD_SYNC_SOURCE,
    get_required_scopes,
)

if TYPE_CHECKING:
    from .audit import AuditLogger

logger = logging.getLogger(__name__)


# Alias for more descriptive error in cloud sync context
class CloudSyncConsentRequiredError(ConsentRequiredError):
    """Raised when cloud sync consent is required but not granted."""
    pass


class CloudSyncConsentManager:
    """Manager for cloud sync consent operations.

    This class provides a unified interface for managing cloud sync consent,
    coordinating with ConsentRegistry for persistence and AuditLogger for
    audit trail.

    Attributes:
        consent_registry: Registry for consent storage
        audit_logger: Logger for audit events
        on_revocation_callback: Optional callback when consent is revoked
    """

    def __init__(
        self,
        consent_registry: ConsentRegistry,
        audit_logger: Optional["AuditLogger"] = None,
        on_revocation_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """Initialize cloud sync consent manager.

        Args:
            consent_registry: ConsentRegistry instance for storage
            audit_logger: Optional AuditLogger for audit trail
            on_revocation_callback: Optional async callback when consent is revoked
                                    (used to trigger cloud data deletion)
        """
        self.consent_registry = consent_registry
        self.audit_logger = audit_logger
        self.on_revocation_callback = on_revocation_callback

    def get_status(self) -> CloudSyncConsentStatus:
        """Get current cloud sync consent status.

        Returns:
            CloudSyncConsentStatus with current consent state
        """
        granted_scopes: List[str] = []
        granted_at: Optional[datetime] = None

        for scope in CloudSyncScope:
            record = self.consent_registry.get(
                source=CLOUD_SYNC_SOURCE,
                scope=scope.value,
            )
            if record and record.is_active():
                granted_scopes.append(scope.value)
                # Track earliest grant time
                if granted_at is None or record.timestamp < granted_at:
                    granted_at = record.timestamp

        has_consent = len(granted_scopes) > 0
        has_required = CloudSyncScope.PKG_METADATA_BACKUP.value in granted_scopes

        return CloudSyncConsentStatus(
            has_consent=has_consent and has_required,
            granted_scopes=granted_scopes,
            granted_at=granted_at,
            is_syncing=False,  # Actual sync status managed by desktop app
        )

    def grant_consent(
        self,
        scopes: List[CloudSyncScope],
        *,
        operator: Optional[str] = None,
    ) -> CloudSyncConsentStatus:
        """Grant consent for specified cloud sync scopes.

        The PKG_METADATA_BACKUP scope is required for any sync operation.
        If not included in the scopes list, it will be added automatically.

        Args:
            scopes: List of scopes to grant consent for
            operator: Optional operator identifier (e.g., user email)

        Returns:
            Updated CloudSyncConsentStatus

        Raises:
            ValueError: If empty scope list provided
        """
        if not scopes:
            raise ValueError("At least one scope must be provided")

        # Ensure required scope is included
        required_scope = CloudSyncScope.PKG_METADATA_BACKUP
        if required_scope not in scopes:
            scopes = [required_scope] + list(scopes)

        job_id = f"cloud_sync_grant_{uuid.uuid4().hex[:8]}"

        for scope in scopes:
            # Grant consent (permanent, no expiration)
            record = self.consent_registry.grant(
                source=CLOUD_SYNC_SOURCE,
                scope=scope.value,
                operator=operator,
                duration_hours=None,  # Permanent until revoked
            )

            # Log audit event
            self._log_consent_event(
                job_id=job_id,
                scope=scope,
                granted=True,
                operator=operator,
                record=record,
            )

            logger.info(
                f"Cloud sync consent granted: scope={scope.value}, operator={operator}"
            )

        return self.get_status()

    def revoke_consent(
        self,
        *,
        operator: Optional[str] = None,
        scopes: Optional[List[CloudSyncScope]] = None,
    ) -> CloudSyncConsentStatus:
        """Revoke cloud sync consent.

        By default, revokes ALL cloud sync scopes. Optionally can revoke
        specific scopes only.

        IMPORTANT: Revocation should trigger cloud data deletion via the
        on_revocation_callback.

        Args:
            operator: Optional operator identifier
            scopes: Optional list of specific scopes to revoke (default: all)

        Returns:
            Updated CloudSyncConsentStatus
        """
        scopes_to_revoke = scopes if scopes else list(CloudSyncScope)
        job_id = f"cloud_sync_revoke_{uuid.uuid4().hex[:8]}"

        for scope in scopes_to_revoke:
            # Revoke consent
            record = self.consent_registry.revoke(
                source=CLOUD_SYNC_SOURCE,
                scope=scope.value,
                operator=operator,
            )

            # Log audit event
            self._log_consent_event(
                job_id=job_id,
                scope=scope,
                granted=False,
                operator=operator,
                record=record,
            )

            logger.info(
                f"Cloud sync consent revoked: scope={scope.value}, operator={operator}"
            )

        # Log data deletion event (will be executed by callback)
        if self.audit_logger:
            self._log_sync_event(
                action="data_deletion_requested",
                success=True,
                metadata={"reason": "consent_revoked", "operator": operator},
            )

        return self.get_status()

    def require_consent(
        self,
        scope: CloudSyncScope = CloudSyncScope.PKG_METADATA_BACKUP,
    ) -> ConsentRecord:
        """Require consent for a sync operation.

        This should be called before any sync operation to enforce consent.

        Args:
            scope: The scope to check (default: PKG_METADATA_BACKUP)

        Returns:
            Active ConsentRecord if consent is granted

        Raises:
            ConsentRequiredError: If consent is not granted or expired
        """
        return self.consent_registry.require(
            source=CLOUD_SYNC_SOURCE,
            scope=scope.value,
        )

    def has_consent(
        self,
        scope: CloudSyncScope = CloudSyncScope.PKG_METADATA_BACKUP,
    ) -> bool:
        """Check if consent is granted for a scope.

        Non-raising version of require_consent.

        Args:
            scope: The scope to check

        Returns:
            True if consent is granted and active
        """
        try:
            self.require_consent(scope)
            return True
        except ConsentRequiredError:
            return False

    def has_scope(
        self,
        scope: CloudSyncScope,
    ) -> bool:
        """Check if a specific scope is granted.

        Alias for has_consent for better readability.

        Args:
            scope: The scope to check

        Returns:
            True if consent is granted and active for the scope
        """
        return self.has_consent(scope)

    def log_sync_started(
        self,
        *,
        nodes_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log that a sync operation has started.

        Args:
            nodes_count: Number of nodes to be synced
            metadata: Additional context
        """
        self._log_sync_event(
            action="sync_started",
            nodes_affected=nodes_count,
            success=True,
            metadata=metadata or {},
        )

    def log_sync_completed(
        self,
        *,
        nodes_synced: int = 0,
        duration_ms: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log that a sync operation completed successfully.

        Args:
            nodes_synced: Number of nodes that were synced
            duration_ms: Duration in milliseconds
            metadata: Additional context
        """
        meta = metadata or {}
        if duration_ms is not None:
            meta["duration_ms"] = duration_ms

        self._log_sync_event(
            action="sync_completed",
            nodes_affected=nodes_synced,
            success=True,
            metadata=meta,
        )

    def log_sync_failed(
        self,
        *,
        error_message: str,
        nodes_affected: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log that a sync operation failed.

        Args:
            error_message: Description of the error
            nodes_affected: Number of nodes that were affected
            metadata: Additional context
        """
        self._log_sync_event(
            action="sync_failed",
            nodes_affected=nodes_affected,
            success=False,
            error_message=error_message,
            metadata=metadata or {},
        )

    def log_data_deleted(
        self,
        *,
        nodes_deleted: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log that cloud data was deleted (after revocation).

        Args:
            nodes_deleted: Number of nodes deleted from cloud
            metadata: Additional context
        """
        self._log_sync_event(
            action="data_deleted",
            nodes_affected=nodes_deleted,
            success=True,
            metadata=metadata or {},
        )

    def _log_consent_event(
        self,
        *,
        job_id: str,
        scope: CloudSyncScope,
        granted: bool,
        operator: Optional[str],
        record: ConsentRecord,
    ) -> None:
        """Log a consent grant/revoke event to audit log."""
        if self.audit_logger is None:
            return

        self.audit_logger.record_consent_event(
            job_id=job_id,
            source=CLOUD_SYNC_SOURCE,
            scope=scope.value,
            granted=granted,
            operator=operator,
            token_hash=record.token_hash,
        )

    def _log_sync_event(
        self,
        *,
        action: str,
        nodes_affected: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a sync operation event to audit log."""
        if self.audit_logger is None:
            return

        from .audit import AuditEvent

        event = AuditEvent(
            job_id=f"cloud_sync_{uuid.uuid4().hex[:8]}",
            source=CLOUD_SYNC_SOURCE,
            action=action,
            status="success" if success else "failed",
            timestamp=datetime.utcnow(),
            metadata={
                "nodes_affected": nodes_affected,
                **(metadata or {}),
                **({"error": error_message} if error_message else {}),
            },
        )
        self.audit_logger.record(event)


def create_cloud_sync_manager(
    consent_dir: Optional[Path] = None,
    audit_dir: Optional[Path] = None,
    *,
    encryption_manager: Optional["EncryptionManager"] = None,
) -> CloudSyncConsentManager:
    """Factory function to create a CloudSyncConsentManager.

    Creates ConsentRegistry and AuditLogger instances with default paths
    if not provided.

    Args:
        consent_dir: Directory for consent storage (default: ~/.futurnal/consent)
        audit_dir: Directory for audit logs (default: ~/.futurnal/audit)
        encryption_manager: Optional encryption manager

    Returns:
        Configured CloudSyncConsentManager
    """
    from .audit import AuditLogger

    # Use default Futurnal directories
    base_dir = Path.home() / ".futurnal"
    consent_dir = consent_dir or base_dir / "consent"
    audit_dir = audit_dir or base_dir / "audit"

    consent_registry = ConsentRegistry(
        consent_dir,
        encryption_manager=encryption_manager,
    )

    audit_logger = AuditLogger(
        output_dir=audit_dir,
        encryption_manager=encryption_manager,
    )

    return CloudSyncConsentManager(
        consent_registry=consent_registry,
        audit_logger=audit_logger,
    )
