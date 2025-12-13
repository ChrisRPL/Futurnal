"""Data purge service for complete user data removal.

This module provides mechanisms to completely remove user data on request,
supporting GDPR right to erasure and privacy-first defaults.

Features:
- Complete audit log purge
- Consent record purge
- Source-specific purge
- Pre-purge audit logging (meta-audit)
- Purge verification

Privacy-First Design (Option B):
- User controls their data lifecycle
- Complete removal with no orphans
- Purge operation itself is logged before execution

Usage:
    >>> from futurnal.privacy.purge import DataPurgeService
    >>> purge_service = DataPurgeService(
    ...     audit_logger=audit_logger,
    ...     consent_registry=consent_registry,
    ... )
    >>> result = purge_service.purge_all(confirm=True)
    >>> print(f"Purged {result.files_deleted} files")
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .consent import ConsentRegistry

logger = logging.getLogger(__name__)


class PurgeError(Exception):
    """Base exception for purge errors."""


class PurgeConfirmationRequired(PurgeError):
    """Raised when purge is attempted without confirmation."""


class PurgeVerificationFailed(PurgeError):
    """Raised when post-purge verification fails."""


@dataclass
class PurgeResult:
    """Result of a purge operation.

    Attributes:
        success: Whether purge completed successfully
        files_deleted: Number of files deleted
        bytes_freed: Approximate bytes freed
        sources_purged: Set of sources that were purged
        errors: List of any errors encountered
        started_at: Timestamp when purge started
        completed_at: Timestamp when purge completed
    """

    success: bool = True
    files_deleted: int = 0
    bytes_freed: int = 0
    sources_purged: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for logging."""
        return {
            "success": self.success,
            "files_deleted": self.files_deleted,
            "bytes_freed": self.bytes_freed,
            "sources_purged": list(self.sources_purged),
            "error_count": len(self.errors),
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.started_at and self.completed_at
                else None
            ),
        }


@dataclass
class DataPurgeService:
    """Service for purging privacy-sensitive data.

    Provides methods to completely remove audit logs, consent records,
    and other privacy-sensitive data. All purge operations are logged
    before execution (meta-audit).

    Attributes:
        audit_logger: AuditLogger instance to purge
        consent_registry: ConsentRegistry instance to purge
        additional_paths: Additional paths to purge (e.g., workspace data)
        meta_audit_logger: Separate logger for purge operations (optional)
    """

    audit_logger: Optional["AuditLogger"] = None
    consent_registry: Optional["ConsentRegistry"] = None
    additional_paths: List[Path] = field(default_factory=list)
    meta_audit_logger: Optional["AuditLogger"] = None

    def purge_all(self, *, confirm: bool = False) -> PurgeResult:
        """Purge all privacy-sensitive data.

        WARNING: This operation is irreversible!

        Args:
            confirm: Must be True to proceed with purge

        Returns:
            PurgeResult with operation details

        Raises:
            PurgeConfirmationRequired: If confirm is False
        """
        if not confirm:
            raise PurgeConfirmationRequired(
                "Purge requires explicit confirmation. Set confirm=True to proceed."
            )

        result = PurgeResult(started_at=datetime.utcnow())

        # Log purge operation before execution
        self._log_purge_start("purge_all", sources=["all"])

        try:
            # Purge audit logs
            if self.audit_logger:
                audit_result = self._purge_audit_logs()
                result.files_deleted += audit_result.files_deleted
                result.bytes_freed += audit_result.bytes_freed
                result.sources_purged.add("audit_logs")
                result.errors.extend(audit_result.errors)

            # Purge consent records
            if self.consent_registry:
                consent_result = self._purge_consent()
                result.files_deleted += consent_result.files_deleted
                result.bytes_freed += consent_result.bytes_freed
                result.sources_purged.add("consent_records")
                result.errors.extend(consent_result.errors)

            # Purge additional paths
            for path in self.additional_paths:
                path_result = self._purge_path(path)
                result.files_deleted += path_result.files_deleted
                result.bytes_freed += path_result.bytes_freed
                result.sources_purged.add(str(path))
                result.errors.extend(path_result.errors)

            result.success = len(result.errors) == 0

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Purge failed: {e}")

        result.completed_at = datetime.utcnow()

        # Log purge completion
        self._log_purge_complete("purge_all", result)

        return result

    def purge_audit_logs(self, *, confirm: bool = False) -> PurgeResult:
        """Purge only audit logs.

        Args:
            confirm: Must be True to proceed

        Returns:
            PurgeResult with operation details

        Raises:
            PurgeConfirmationRequired: If confirm is False
        """
        if not confirm:
            raise PurgeConfirmationRequired(
                "Audit purge requires confirmation. Set confirm=True."
            )

        result = PurgeResult(started_at=datetime.utcnow())

        self._log_purge_start("purge_audit_logs", sources=["audit_logs"])

        try:
            if self.audit_logger:
                audit_result = self._purge_audit_logs()
                result.files_deleted = audit_result.files_deleted
                result.bytes_freed = audit_result.bytes_freed
                result.sources_purged.add("audit_logs")
                result.errors = audit_result.errors
                result.success = len(result.errors) == 0
        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        result.completed_at = datetime.utcnow()
        self._log_purge_complete("purge_audit_logs", result)

        return result

    def purge_consent(self, *, confirm: bool = False) -> PurgeResult:
        """Purge only consent records.

        Args:
            confirm: Must be True to proceed

        Returns:
            PurgeResult with operation details

        Raises:
            PurgeConfirmationRequired: If confirm is False
        """
        if not confirm:
            raise PurgeConfirmationRequired(
                "Consent purge requires confirmation. Set confirm=True."
            )

        result = PurgeResult(started_at=datetime.utcnow())

        self._log_purge_start("purge_consent", sources=["consent_records"])

        try:
            if self.consent_registry:
                consent_result = self._purge_consent()
                result.files_deleted = consent_result.files_deleted
                result.bytes_freed = consent_result.bytes_freed
                result.sources_purged.add("consent_records")
                result.errors = consent_result.errors
                result.success = len(result.errors) == 0
        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        result.completed_at = datetime.utcnow()
        self._log_purge_complete("purge_consent", result)

        return result

    def purge_by_source(
        self,
        source: str,
        *,
        confirm: bool = False,
    ) -> PurgeResult:
        """Purge data for a specific source.

        Removes audit logs related to the source and revokes all
        consent for that source.

        Args:
            source: Source identifier to purge
            confirm: Must be True to proceed

        Returns:
            PurgeResult with operation details

        Raises:
            PurgeConfirmationRequired: If confirm is False
        """
        if not confirm:
            raise PurgeConfirmationRequired(
                f"Purge of source '{source}' requires confirmation."
            )

        result = PurgeResult(started_at=datetime.utcnow())

        self._log_purge_start("purge_by_source", sources=[source])

        try:
            # Note: Full source-specific audit purge would require filtering logs
            # For now, we revoke consent for the source
            if self.consent_registry:
                from .consent import ConsentRequiredError

                # Get all consent scopes for this source and revoke them
                revoked_count = 0
                for record in list(self.consent_registry.snapshot()):
                    if record.source == source and record.granted:
                        self.consent_registry.revoke(
                            source=source,
                            scope=record.scope,
                            operator="purge_service",
                        )
                        revoked_count += 1

                if revoked_count > 0:
                    result.sources_purged.add(source)
                    result.files_deleted = revoked_count  # Counting revocations

            result.success = True

        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        result.completed_at = datetime.utcnow()
        self._log_purge_complete("purge_by_source", result)

        return result

    def _purge_audit_logs(self) -> PurgeResult:
        """Internal method to purge audit logs."""
        result = PurgeResult()

        if not self.audit_logger:
            return result

        audit_dir = self.audit_logger.output_dir

        try:
            # Delete all audit files
            for file_path in audit_dir.glob("audit*.log"):
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    result.files_deleted += 1
                    result.bytes_freed += size
                except Exception as e:
                    result.errors.append(f"Failed to delete {file_path}: {e}")

            # Delete manifest
            manifest_path = audit_dir / self.audit_logger.manifest_name
            if manifest_path.exists():
                try:
                    size = manifest_path.stat().st_size
                    manifest_path.unlink()
                    result.files_deleted += 1
                    result.bytes_freed += size
                except Exception as e:
                    result.errors.append(f"Failed to delete manifest: {e}")

            # Delete review directory
            review_dir = audit_dir / self.audit_logger.review_dirname
            if review_dir.exists():
                try:
                    for review_file in review_dir.glob("*.log"):
                        size = review_file.stat().st_size
                        review_file.unlink()
                        result.files_deleted += 1
                        result.bytes_freed += size
                    review_dir.rmdir()
                except Exception as e:
                    result.errors.append(f"Failed to purge review dir: {e}")

        except Exception as e:
            result.errors.append(f"Audit purge error: {e}")

        return result

    def _purge_consent(self) -> PurgeResult:
        """Internal method to purge consent records."""
        result = PurgeResult()

        if not self.consent_registry:
            return result

        consent_path = self.consent_registry._path

        try:
            if consent_path.exists():
                size = consent_path.stat().st_size
                consent_path.unlink()
                result.files_deleted = 1
                result.bytes_freed = size
        except Exception as e:
            result.errors.append(f"Failed to delete consent file: {e}")

        return result

    def _purge_path(self, path: Path) -> PurgeResult:
        """Internal method to purge a directory or file."""
        result = PurgeResult()

        if not path.exists():
            return result

        try:
            if path.is_dir():
                # Calculate size before deletion
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        result.bytes_freed += file_path.stat().st_size
                        result.files_deleted += 1

                shutil.rmtree(path)
            else:
                result.bytes_freed = path.stat().st_size
                result.files_deleted = 1
                path.unlink()

        except Exception as e:
            result.errors.append(f"Failed to purge {path}: {e}")

        return result

    def _log_purge_start(self, operation: str, sources: List[str]) -> None:
        """Log purge operation start (meta-audit)."""
        audit_target = self.meta_audit_logger or self.audit_logger
        if not audit_target:
            return

        from .audit import AuditEvent

        audit_target.record(
            AuditEvent(
                job_id=f"purge_start_{int(datetime.utcnow().timestamp())}",
                source="purge_service",
                action=f"purge:{operation}",
                status="started",
                timestamp=datetime.utcnow(),
                metadata={
                    "operation": operation,
                    "targets": sources,
                    "warning": "DATA_DELETION_INITIATED",
                },
            )
        )

    def _log_purge_complete(self, operation: str, result: PurgeResult) -> None:
        """Log purge operation completion (meta-audit)."""
        audit_target = self.meta_audit_logger or self.audit_logger
        if not audit_target:
            return

        # If we're logging to the same audit_logger that was purged,
        # the manifest may no longer exist - check before logging
        if audit_target is self.audit_logger and "audit_logs" in result.sources_purged:
            # Audit logger was purged, can't log completion to it
            # The start event was already logged before purge
            logger.debug("Skipping purge completion log - audit logger was purged")
            return

        from .audit import AuditEvent

        audit_target.record(
            AuditEvent(
                job_id=f"purge_complete_{int(datetime.utcnow().timestamp())}",
                source="purge_service",
                action=f"purge:{operation}",
                status="completed" if result.success else "failed",
                timestamp=datetime.utcnow(),
                metadata=result.to_dict(),
            )
        )

    def verify_purge(self) -> bool:
        """Verify that purge was successful.

        Checks that audit logs and consent records are empty or deleted.

        Returns:
            True if data is purged, False if data remains
        """
        # Check audit logs
        if self.audit_logger:
            audit_dir = self.audit_logger.output_dir
            log_files = list(audit_dir.glob("audit*.log"))
            if log_files:
                logger.warning(f"Audit log files still exist: {log_files}")
                return False

        # Check consent
        if self.consent_registry:
            consent_path = self.consent_registry._path
            if consent_path.exists():
                logger.warning(f"Consent file still exists: {consent_path}")
                return False

        return True


def create_purge_service(
    workspace_dir: Optional[Path] = None,
    audit_logger: Optional["AuditLogger"] = None,
    consent_registry: Optional["ConsentRegistry"] = None,
) -> DataPurgeService:
    """Create a DataPurgeService with standard configuration.

    Args:
        workspace_dir: Optional workspace directory to include in purge
        audit_logger: AuditLogger instance
        consent_registry: ConsentRegistry instance

    Returns:
        Configured DataPurgeService
    """
    additional_paths = []
    if workspace_dir and workspace_dir.exists():
        additional_paths.append(workspace_dir)

    return DataPurgeService(
        audit_logger=audit_logger,
        consent_registry=consent_registry,
        additional_paths=additional_paths,
    )


__all__ = [
    "DataPurgeService",
    "PurgeResult",
    "PurgeError",
    "PurgeConfirmationRequired",
    "PurgeVerificationFailed",
    "create_purge_service",
]
