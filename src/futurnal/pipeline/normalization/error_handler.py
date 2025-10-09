"""Normalization error handling and quarantine integration.

This module provides normalization-specific error classification, retry policy
mapping, and integration with the orchestrator's quarantine system. Ensures
failed documents are tracked with detailed diagnostics for operator review and
recovery.

Key Features:
- Normalization-specific error classification (13 error types)
- Retry policy selection per failure type
- Integration with QuarantineStore for persistence
- Privacy-aware error logging with path redaction
- Detailed diagnostic metadata capture
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from ...orchestrator.quarantine import QuarantineReason, QuarantineStore
from ...orchestrator.models import IngestionJob, JobPriority, JobType

logger = logging.getLogger(__name__)


class NormalizationErrorType(str, Enum):
    """Classification of normalization errors with retry policy implications.

    Categories:
    - Format errors: Unlikely to succeed on retry (structural issues)
    - Processing errors: May succeed on retry (transient processing failures)
    - Resource errors: Transient, should retry with backoff
    - Privacy errors: Never retry (access control violations)
    """

    # Format errors (unlikely to succeed on retry)
    UNSUPPORTED_FORMAT = "unsupported_format"
    MALFORMED_CONTENT = "malformed_content"
    CORRUPTED_FILE = "corrupted_file"

    # Processing errors (may succeed on retry)
    UNSTRUCTURED_PARSE_ERROR = "unstructured_parse_error"
    CHUNKING_FAILURE = "chunking_failure"
    ENRICHMENT_FAILURE = "enrichment_failure"

    # Resource errors (transient)
    MEMORY_EXHAUSTED = "memory_exhausted"
    DISK_FULL = "disk_full"
    FILE_ACCESS_DENIED = "file_access_denied"

    # Privacy errors (never retry)
    ENCRYPTION_DETECTED = "encryption_detected"
    PERMISSION_DENIED = "permission_denied"


class NormalizationErrorHandler:
    """Handler for normalization errors with quarantine integration.

    Bridges normalization pipeline failures to the orchestrator's quarantine
    system, providing detailed diagnostics and retry policy selection.

    Example:
        >>> handler = NormalizationErrorHandler(quarantine_store)
        >>> await handler.handle_error(
        ...     file_path=Path("/path/to/doc.pdf"),
        ...     source_id="local_123",
        ...     source_type="local_files",
        ...     error=ValueError("Malformed PDF structure"),
        ...     metadata={"format": "pdf", "stage": "unstructured_parse"}
        ... )
    """

    def __init__(self, quarantine_store: QuarantineStore):
        """Initialize error handler.

        Args:
            quarantine_store: QuarantineStore instance for persistence
        """
        self.quarantine_store = quarantine_store

    async def handle_error(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        error: Exception,
        metadata: Optional[dict] = None,
    ) -> None:
        """Handle normalization error and route to quarantine.

        Classifies error, determines retry policy, maps to quarantine reason,
        and persists to quarantine store with detailed diagnostics.

        Args:
            file_path: Path to failed document
            source_id: Connector-specific identifier
            source_type: Source type (e.g., "obsidian_vault", "local_files")
            error: Exception that caused normalization failure
            metadata: Additional context (format, stage, file_size, etc.)
        """
        # Step 1: Classify error for detailed diagnostics
        error_type = self._classify_error(error)

        # Step 2: Determine retry policy
        retry_policy = self._get_retry_policy(error_type)

        # Step 3: Map to quarantine reason for orchestrator compatibility
        quarantine_reason = self._map_to_quarantine_reason(error_type)

        # Step 4: Build diagnostic metadata
        diagnostic_metadata = self._build_diagnostic_metadata(
            error_type=error_type,
            retry_policy=retry_policy,
            file_path=file_path,
            original_metadata=metadata,
            error=error,
        )

        # Step 5: Create synthetic IngestionJob for quarantine
        # Note: Normalization failures aren't true "jobs", but we reuse the
        # job-based quarantine interface by creating a synthetic job
        job_id = f"norm_{source_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Map source_type to JobType, fallback to LOCAL_FILES
        job_type_map = {
            "local_files": JobType.LOCAL_FILES,
            "obsidian_vault": JobType.OBSIDIAN_VAULT,
            "imap_mailbox": JobType.IMAP_MAILBOX,
            "github_repository": JobType.GITHUB_REPOSITORY,
        }
        job_type = job_type_map.get(source_type, JobType.LOCAL_FILES)

        synthetic_job = IngestionJob(
            job_id=job_id,
            job_type=job_type,
            payload={
                "source_id": source_id,
                "source_type": source_type,
                "file_path": str(file_path),
                "normalization_failure": True,
            },
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.now(timezone.utc),
        )

        # Step 6: Capture traceback for debugging
        error_traceback = "".join(traceback.format_exception(
            type(error), error, error.__traceback__
        ))

        # Step 7: Persist to quarantine
        self.quarantine_store.quarantine(
            job=synthetic_job,
            reason=quarantine_reason,
            error_message=str(error),
            error_traceback=error_traceback,
            metadata=diagnostic_metadata,
        )

        logger.error(
            f"Normalization failure quarantined: {file_path.name}",
            extra={
                "error_type": error_type.value,
                "retry_policy": retry_policy,
                "quarantine_reason": quarantine_reason.value,
                "file_name": file_path.name,
            },
        )

    def _classify_error(self, error: Exception) -> NormalizationErrorType:
        """Classify error for retry policy selection.

        Uses multi-tier approach:
        1. Exception type (most reliable)
        2. Exception attributes (e.g., status_code)
        3. Error message pattern matching
        4. Fallback to MALFORMED_CONTENT

        Args:
            error: Exception to classify

        Returns:
            NormalizationErrorType classification
        """
        error_name = type(error).__name__
        error_msg = str(error).lower()

        # Tier 1: Exception type
        if isinstance(error, PermissionError):
            return NormalizationErrorType.PERMISSION_DENIED
        elif isinstance(error, MemoryError):
            return NormalizationErrorType.MEMORY_EXHAUSTED
        elif isinstance(error, OSError):
            # OSError can indicate various file access issues
            if "disk" in error_msg or "space" in error_msg:
                return NormalizationErrorType.DISK_FULL
            elif "permission" in error_msg or "access" in error_msg:
                return NormalizationErrorType.FILE_ACCESS_DENIED

        # Tier 2: Error message pattern matching (order matters)

        # Unsupported format
        if any(pattern in error_msg for pattern in [
            "unsupported format",
            "format not supported",
            "unknown format",
            "invalid file type",
        ]):
            return NormalizationErrorType.UNSUPPORTED_FORMAT

        # Corruption/malformed (check before parse to avoid false positives)
        if any(pattern in error_msg for pattern in [
            "corrupted",
            "corrupt",
            "damaged",
            "truncated",
            "incomplete file",
        ]):
            return NormalizationErrorType.CORRUPTED_FILE

        # Unstructured.io specific errors
        if any(pattern in error_msg for pattern in [
            "unstructured",
            "partition",
            "element extraction",
        ]):
            return NormalizationErrorType.UNSTRUCTURED_PARSE_ERROR

        # Chunking failures
        if any(pattern in error_msg for pattern in [
            "chunk",
            "chunking",
            "split",
            "splitting",
        ]):
            return NormalizationErrorType.CHUNKING_FAILURE

        # Enrichment failures
        if any(pattern in error_msg for pattern in [
            "enrichment",
            "metadata extraction",
            "language detection",
            "classification",
        ]):
            return NormalizationErrorType.ENRICHMENT_FAILURE

        # Memory issues
        if any(pattern in error_msg for pattern in [
            "memory",
            "out of memory",
            "memory exhausted",
            "allocation failed",
        ]):
            return NormalizationErrorType.MEMORY_EXHAUSTED

        # Disk space issues
        if any(pattern in error_msg for pattern in [
            "disk full",
            "no space",
            "out of space",
            "disk space",
        ]):
            return NormalizationErrorType.DISK_FULL

        # Permission/access issues
        if any(pattern in error_msg for pattern in [
            "permission denied",
            "access denied",
            "forbidden",
            "unauthorized",
            "not permitted",
        ]):
            return NormalizationErrorType.PERMISSION_DENIED

        # Encryption detection
        if any(pattern in error_msg for pattern in [
            "encrypted",
            "encryption",
            "password protected",
            "locked",
        ]):
            return NormalizationErrorType.ENCRYPTION_DETECTED

        # Generic parse errors
        if any(pattern in error_msg for pattern in [
            "parse",
            "parsing",
            "decode",
            "decoding",
            "malformed",
            "invalid syntax",
        ]):
            return NormalizationErrorType.MALFORMED_CONTENT

        # Fallback
        return NormalizationErrorType.MALFORMED_CONTENT

    def _get_retry_policy(self, error_type: NormalizationErrorType) -> str:
        """Get retry policy for error type.

        Policies:
        - retry_with_backoff: Transient resource issues
        - never_retry: Permanent failures (privacy, unsupported format)
        - retry_once: Unknown/transient processing issues

        Args:
            error_type: Classified error type

        Returns:
            Retry policy string
        """
        # Resource errors: Retry with exponential backoff
        if error_type in [
            NormalizationErrorType.MEMORY_EXHAUSTED,
            NormalizationErrorType.DISK_FULL,
            NormalizationErrorType.FILE_ACCESS_DENIED,
        ]:
            return "retry_with_backoff"

        # Privacy and format errors: Never retry
        elif error_type in [
            NormalizationErrorType.ENCRYPTION_DETECTED,
            NormalizationErrorType.PERMISSION_DENIED,
            NormalizationErrorType.UNSUPPORTED_FORMAT,
        ]:
            return "never_retry"

        # Processing errors: Retry once (may be transient)
        else:
            return "retry_once"

    def _map_to_quarantine_reason(
        self, error_type: NormalizationErrorType
    ) -> QuarantineReason:
        """Map NormalizationErrorType to QuarantineReason.

        Enables integration with orchestrator's quarantine system which uses
        the more general QuarantineReason taxonomy.

        Args:
            error_type: Normalization-specific error type

        Returns:
            Corresponding QuarantineReason
        """
        mapping = {
            # Format errors → PARSE_ERROR or INVALID_STATE
            NormalizationErrorType.UNSUPPORTED_FORMAT: QuarantineReason.PARSE_ERROR,
            NormalizationErrorType.MALFORMED_CONTENT: QuarantineReason.PARSE_ERROR,
            NormalizationErrorType.CORRUPTED_FILE: QuarantineReason.INVALID_STATE,

            # Processing errors → PARSE_ERROR or CONNECTOR_ERROR
            NormalizationErrorType.UNSTRUCTURED_PARSE_ERROR: QuarantineReason.PARSE_ERROR,
            NormalizationErrorType.CHUNKING_FAILURE: QuarantineReason.CONNECTOR_ERROR,
            NormalizationErrorType.ENRICHMENT_FAILURE: QuarantineReason.CONNECTOR_ERROR,

            # Resource errors → RESOURCE_EXHAUSTED
            NormalizationErrorType.MEMORY_EXHAUSTED: QuarantineReason.RESOURCE_EXHAUSTED,
            NormalizationErrorType.DISK_FULL: QuarantineReason.RESOURCE_EXHAUSTED,
            NormalizationErrorType.FILE_ACCESS_DENIED: QuarantineReason.PERMISSION_DENIED,

            # Privacy errors → PERMISSION_DENIED
            NormalizationErrorType.ENCRYPTION_DETECTED: QuarantineReason.PERMISSION_DENIED,
            NormalizationErrorType.PERMISSION_DENIED: QuarantineReason.PERMISSION_DENIED,
        }

        return mapping.get(error_type, QuarantineReason.UNKNOWN)

    def _build_diagnostic_metadata(
        self,
        *,
        error_type: NormalizationErrorType,
        retry_policy: str,
        file_path: Path,
        original_metadata: Optional[dict],
        error: Exception,
    ) -> dict:
        """Build comprehensive diagnostic metadata for quarantine.

        Captures context necessary for operators to diagnose and resolve
        normalization failures. Privacy-aware (no file content).

        Args:
            error_type: Classified error type
            retry_policy: Retry policy string
            file_path: Path to failed file
            original_metadata: Original metadata from service
            error: Exception instance

        Returns:
            Diagnostic metadata dictionary
        """
        # Start with original metadata if provided
        metadata = dict(original_metadata) if original_metadata else {}

        # Add normalization-specific diagnostics
        metadata.update({
            "normalization_failure": True,
            "error_type": error_type.value,
            "error_class": type(error).__name__,
            "retry_policy": retry_policy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
        })

        # Add file size if available (privacy-safe)
        try:
            if file_path.exists():
                metadata["file_size_bytes"] = file_path.stat().st_size
        except Exception:
            # Ignore stat errors
            pass

        # Add processing stage if available
        if original_metadata and "stage" in original_metadata:
            metadata["failure_stage"] = original_metadata["stage"]

        return metadata
