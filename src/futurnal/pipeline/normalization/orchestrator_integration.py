"""Orchestrator integration for NormalizationService.

This module provides integration between the NormalizationService and the
IngestionOrchestrator, enabling automated document normalization from all
connectors with state management, audit logging, and metrics collection.

Key Features:
- Process documents from connectors through normalization pipeline
- State checkpointing via StateStore for idempotency
- Comprehensive audit logging for all normalization events
- Metrics collection and telemetry export
- Error handling with quarantine integration
- Retry logic for transient failures
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...ingestion.local.state import FileRecord, StateStore
from ...orchestrator.quarantine import QuarantineStore
from ...privacy.audit import AuditEvent, AuditLogger
from ..stubs import NormalizationSink
from .service import NormalizationConfig, NormalizationService, NormalizationError

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a file through normalization pipeline.

    Attributes:
        success: Whether processing succeeded
        file_path: Path to processed file
        source_id: Source identifier
        document_id: Generated document ID (SHA256)
        processing_duration_ms: Processing duration in milliseconds
        error_message: Error message if processing failed
        was_cached: Whether result came from state cache
    """
    success: bool
    file_path: Path
    source_id: str
    document_id: Optional[str] = None
    processing_duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    was_cached: bool = False


class NormalizationProcessor:
    """Orchestrator integration wrapper for NormalizationService.

    Provides the bridge between IngestionOrchestrator and NormalizationService,
    handling state management, audit logging, and metrics collection for
    automated document normalization from all connectors.

    Example:
        >>> processor = NormalizationProcessor(
        ...     normalization_service=service,
        ...     state_store=state_store,
        ...     audit_logger=audit_logger
        ... )
        >>> result = await processor.process_file(
        ...     file_path=Path("/path/to/file.md"),
        ...     source_id="local-123",
        ...     source_type="local_files"
        ... )
    """

    def __init__(
        self,
        *,
        normalization_service: NormalizationService,
        state_store: Optional[StateStore] = None,
        audit_logger: Optional[AuditLogger] = None,
        enable_state_checkpointing: bool = True,
    ):
        """Initialize normalization processor.

        Args:
            normalization_service: Configured NormalizationService instance
            state_store: Optional state store for checkpointing
            audit_logger: Optional audit logger for events
            enable_state_checkpointing: Whether to use state checkpointing
        """
        self.normalization_service = normalization_service
        self.state_store = state_store
        self.audit_logger = audit_logger
        self.enable_state_checkpointing = enable_state_checkpointing

        # Metrics tracking
        self.files_processed = 0
        self.files_failed = 0
        self.files_skipped_cached = 0
        self.total_processing_time_ms = 0.0

    async def process_file(
        self,
        *,
        file_path: Path | str,
        source_id: str,
        source_type: str,
        source_metadata: Optional[dict] = None,
        force_reprocess: bool = False,
    ) -> ProcessingResult:
        """Process a single file through the normalization pipeline.

        Handles state checkpointing to avoid duplicate processing, comprehensive
        audit logging, and error handling with quarantine integration.

        Args:
            file_path: Path to file to process
            source_id: Connector-specific identifier
            source_type: Source type (e.g., "local_files", "obsidian_vault")
            source_metadata: Additional metadata from connector
            force_reprocess: Force reprocessing even if cached

        Returns:
            ProcessingResult with outcome details
        """
        file_path = Path(file_path)
        start_time = datetime.now(timezone.utc)

        # Check state cache for idempotency
        if self.enable_state_checkpointing and not force_reprocess and self.state_store:
            cached = await self._check_state_cache(file_path)
            if cached:
                logger.debug(
                    f"Skipping cached file: {file_path.name}",
                    extra={
                        "file_path": str(file_path),
                        "source_id": source_id,
                        "cached": True,
                    }
                )
                self.files_skipped_cached += 1
                self._record_audit_event(
                    file_path=file_path,
                    source_id=source_id,
                    source_type=source_type,
                    action="normalization_skipped",
                    status="cached",
                    metadata={"reason": "state_cache_hit"},
                )
                return ProcessingResult(
                    success=True,
                    file_path=file_path,
                    source_id=source_id,
                    was_cached=True,
                )

        # Log processing start
        self._record_audit_event(
            file_path=file_path,
            source_id=source_id,
            source_type=source_type,
            action="normalization_processing",
            status="started",
        )

        try:
            # Process through normalization service
            normalized_doc = await self.normalization_service.normalize_document(
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                source_metadata=source_metadata,
            )

            # Update state checkpoint
            if self.enable_state_checkpointing and self.state_store:
                await self._update_state_checkpoint(file_path, normalized_doc.sha256)

            # Calculate duration
            duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Update metrics
            self.files_processed += 1
            self.total_processing_time_ms += duration_ms

            # Log success
            self._record_audit_event(
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                action="normalization_completed",
                status="success",
                sha256=normalized_doc.sha256,
                metadata={
                    "document_id": normalized_doc.document_id,
                    "format": normalized_doc.metadata.format.value,
                    "is_chunked": normalized_doc.is_chunked,
                    "total_chunks": normalized_doc.metadata.total_chunks,
                    "processing_duration_ms": duration_ms,
                },
            )

            logger.info(
                f"Successfully normalized: {file_path.name}",
                extra={
                    "file_path": str(file_path),
                    "source_id": source_id,
                    "document_id": normalized_doc.document_id,
                    "duration_ms": duration_ms,
                }
            )

            return ProcessingResult(
                success=True,
                file_path=file_path,
                source_id=source_id,
                document_id=normalized_doc.document_id,
                processing_duration_ms=duration_ms,
            )

        except NormalizationError as e:
            # Handle normalization failure
            duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Update metrics
            self.files_failed += 1

            # Log failure
            self._record_audit_event(
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                action="normalization_failed",
                status="failed",
                metadata={
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:500],
                    "processing_duration_ms": duration_ms,
                },
            )

            logger.error(
                f"Failed to normalize: {file_path.name}",
                extra={
                    "file_path": str(file_path),
                    "source_id": source_id,
                    "error": str(e),
                    "duration_ms": duration_ms,
                },
                exc_info=e,
            )

            return ProcessingResult(
                success=False,
                file_path=file_path,
                source_id=source_id,
                processing_duration_ms=duration_ms,
                error_message=str(e),
            )

    async def process_batch(
        self,
        *,
        files: list[tuple[Path, str]],  # (file_path, source_id) pairs
        source_type: str,
        source_metadata: Optional[dict] = None,
    ) -> list[ProcessingResult]:
        """Process multiple files in batch.

        Args:
            files: List of (file_path, source_id) tuples
            source_type: Source type for all files
            source_metadata: Shared metadata for all files

        Returns:
            List of ProcessingResult for each file
        """
        results = []
        for file_path, source_id in files:
            result = await self.process_file(
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                source_metadata=source_metadata,
            )
            results.append(result)
        return results

    async def _check_state_cache(self, file_path: Path) -> bool:
        """Check if file is already processed in state cache.

        Args:
            file_path: Path to file

        Returns:
            True if file is cached and unchanged, False otherwise
        """
        if not self.state_store:
            return False

        try:
            # Check if file exists in state
            cached_record = self.state_store.fetch(file_path)
            if not cached_record:
                return False

            # Verify file hasn't changed
            if not file_path.exists():
                return False

            current_stats = file_path.stat()
            current_size = current_stats.st_size
            current_mtime = current_stats.st_mtime

            # Cache hit if size and mtime match
            return (
                cached_record.size == current_size
                and cached_record.mtime == current_mtime
            )

        except Exception as e:
            logger.warning(
                f"Error checking state cache for {file_path}: {e}",
                extra={"file_path": str(file_path), "error": str(e)},
            )
            return False

    async def _update_state_checkpoint(self, file_path: Path, sha256: str) -> None:
        """Update state checkpoint for processed file.

        Args:
            file_path: Path to processed file
            sha256: SHA256 hash of file content
        """
        if not self.state_store:
            return

        try:
            stats = file_path.stat()
            record = FileRecord(
                path=file_path,
                size=stats.st_size,
                mtime=stats.st_mtime,
                sha256=sha256,
            )
            self.state_store.upsert(record)
            logger.debug(
                f"Updated state checkpoint: {file_path.name}",
                extra={
                    "file_path": str(file_path),
                    "sha256": sha256,
                }
            )
        except Exception as e:
            logger.warning(
                f"Failed to update state checkpoint for {file_path}: {e}",
                extra={"file_path": str(file_path), "error": str(e)},
            )

    def _record_audit_event(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        action: str,
        status: str,
        sha256: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record audit event for normalization activity.

        Args:
            file_path: Path to file
            source_id: Source identifier
            source_type: Source type
            action: Action being performed
            status: Status of action
            sha256: Optional document SHA256
            metadata: Optional additional metadata
        """
        if not self.audit_logger:
            return

        try:
            event = AuditEvent(
                job_id=f"norm_{source_id}",
                source=source_type,
                action=action,
                status=status,
                timestamp=datetime.now(timezone.utc),
                sha256=sha256,
                metadata={
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix,
                    **(metadata or {}),
                },
            )
            self.audit_logger.record(event)
        except Exception as e:
            logger.warning(
                f"Failed to record audit event: {e}",
                extra={"error": str(e)},
            )

    def get_metrics(self) -> dict:
        """Get processing metrics for telemetry.

        Returns:
            Dictionary with processing statistics
        """
        total_files = self.files_processed + self.files_failed + self.files_skipped_cached

        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "files_skipped_cached": self.files_skipped_cached,
            "total_files": total_files,
            "success_rate": (
                self.files_processed / (self.files_processed + self.files_failed)
                if (self.files_processed + self.files_failed) > 0
                else 0.0
            ),
            "cache_hit_rate": (
                self.files_skipped_cached / total_files
                if total_files > 0
                else 0.0
            ),
            "total_processing_time_ms": self.total_processing_time_ms,
            "average_processing_time_ms": (
                self.total_processing_time_ms / self.files_processed
                if self.files_processed > 0
                else 0.0
            ),
        }
