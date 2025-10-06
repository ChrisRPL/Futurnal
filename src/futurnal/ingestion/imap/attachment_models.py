"""Email attachment models with processing state tracking.

This module extends the basic AttachmentMetadata from email_parser with
full attachment processing lifecycle support, including:
- Content storage with hash-based deduplication
- Processing status tracking through Unstructured.io pipeline
- Privacy-aware metadata (no content in model, only references)
- Integration with quarantine system for failed processing

Design Philosophy:
- Separation: EmailParser extracts metadata only; AttachmentExtractor handles content
- Privacy-first: Never store content in model, only storage paths and hashes
- Deduplication: Content-hash (SHA256) based storage to save space
- Resilience: Comprehensive status tracking with retry support

Ghost→Animal Evolution:
- **Phase 1 (CURRENT)**: Extract attachment content for Unstructured.io processing
- **Phase 2 (FUTURE)**: Intelligent format detection and adaptive processing
- **Phase 3 (FUTURE)**: Causal understanding of attachment patterns in communication
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class AttachmentProcessingStatus(str, Enum):
    """Processing status for email attachments.

    Lifecycle:
    - PENDING: Extracted but not yet processed
    - PROCESSING: Currently being processed through Unstructured.io
    - COMPLETED: Successfully processed, elements extracted
    - FAILED: Processing failed (temporary, can retry)
    - SKIPPED: Intentionally skipped (too large, unsupported format)
    - QUARANTINED: Permanently failed, moved to quarantine
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    QUARANTINED = "quarantined"


class EmailAttachment(BaseModel):
    """Email attachment with full processing lifecycle metadata.

    Extends AttachmentMetadata from EmailParser with:
    - Content storage reference (not content itself)
    - Processing status and timestamps
    - Content hash for deduplication
    - Privacy-aware design (no content in model)

    This model represents the full lifecycle of an attachment from extraction
    through processing to PKG integration.
    """

    # Identity
    attachment_id: str = Field(..., description="Generated UUID for this attachment")
    message_id: str = Field(..., description="Parent email Message-ID")
    part_id: str = Field(..., description="MIME part identifier")

    # Metadata (from original AttachmentMetadata)
    filename: str = Field(..., description="Attachment filename")
    content_type: str = Field(..., description="MIME content type")
    size_bytes: int = Field(..., ge=0, description="Attachment size in bytes")
    is_inline: bool = Field(default=False, description="True if inline attachment")
    content_id: Optional[str] = Field(
        default=None, description="Content-ID for inline images"
    )

    # Content reference (privacy-first: no content stored)
    content_hash: str = Field(
        default="", description="SHA256 hash of content (for deduplication)"
    )
    storage_path: Optional[Path] = Field(
        default=None, description="Local storage path (if extracted)"
    )

    # Processing status
    processing_status: AttachmentProcessingStatus = Field(
        default=AttachmentProcessingStatus.PENDING,
        description="Current processing status"
    )
    processed_at: Optional[datetime] = Field(
        default=None, description="Timestamp when processing completed"
    )
    extraction_elements: int = Field(
        default=0, ge=0, description="Number of elements extracted by Unstructured.io"
    )
    processing_error: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )

    # Privacy classification
    contains_sensitive: bool = Field(
        default=False, description="True if sensitive content detected"
    )

    # Provenance
    extracted_at: datetime = Field(..., description="Timestamp when attachment extracted")
    mailbox_id: str = Field(..., description="Source mailbox identifier")

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, value: str) -> str:  # type: ignore[override]
        """Validate content hash is valid SHA256 or empty."""
        if not value:
            return value

        if len(value) != 64:
            raise ValueError(f"Invalid SHA256 hash length: {len(value)}, expected 64")

        if not all(c in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"Invalid SHA256 hash characters: {value}")

        return value.lower()

    @field_validator("storage_path")
    @classmethod
    def _validate_storage_path(cls, value: Optional[Path]) -> Optional[Path]:  # type: ignore[override]
        """Validate storage path is absolute if provided."""
        if value is None:
            return value

        if not value.is_absolute():
            raise ValueError(f"Storage path must be absolute: {value}")

        return value

    @property
    def is_processed(self) -> bool:
        """Check if attachment has been successfully processed."""
        return self.processing_status == AttachmentProcessingStatus.COMPLETED

    @property
    def is_pending(self) -> bool:
        """Check if attachment is pending processing."""
        return self.processing_status == AttachmentProcessingStatus.PENDING

    @property
    def is_failed(self) -> bool:
        """Check if attachment processing failed."""
        return self.processing_status in (
            AttachmentProcessingStatus.FAILED,
            AttachmentProcessingStatus.QUARANTINED,
        )

    @property
    def is_skipped(self) -> bool:
        """Check if attachment was intentionally skipped."""
        return self.processing_status == AttachmentProcessingStatus.SKIPPED

    @property
    def has_content(self) -> bool:
        """Check if attachment content has been extracted and stored."""
        return self.storage_path is not None and self.storage_path.exists()

    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    def mark_processing(self) -> None:
        """Mark attachment as currently being processed."""
        self.processing_status = AttachmentProcessingStatus.PROCESSING

    def mark_completed(self, extraction_elements: int = 0) -> None:
        """Mark attachment as successfully processed.

        Args:
            extraction_elements: Number of elements extracted
        """
        self.processing_status = AttachmentProcessingStatus.COMPLETED
        self.processed_at = datetime.utcnow()
        self.extraction_elements = extraction_elements
        self.processing_error = None

    def mark_failed(self, error: str) -> None:
        """Mark attachment as failed processing.

        Args:
            error: Error message
        """
        self.processing_status = AttachmentProcessingStatus.FAILED
        self.processing_error = error

    def mark_skipped(self, reason: str) -> None:
        """Mark attachment as skipped.

        Args:
            reason: Reason for skipping
        """
        self.processing_status = AttachmentProcessingStatus.SKIPPED
        self.processing_error = reason

    def mark_quarantined(self, reason: str) -> None:
        """Mark attachment as quarantined (permanent failure).

        Args:
            reason: Reason for quarantine
        """
        self.processing_status = AttachmentProcessingStatus.QUARANTINED
        self.processing_error = reason

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        """Compute SHA256 hash of attachment content.

        Args:
            content: Raw attachment content bytes

        Returns:
            Lowercase hexadecimal SHA256 hash
        """
        return hashlib.sha256(content).hexdigest()


__all__ = [
    "AttachmentProcessingStatus",
    "EmailAttachment",
]
