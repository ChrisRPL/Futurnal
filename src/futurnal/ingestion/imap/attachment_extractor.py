"""Attachment extraction from email MIME parts with filtering and storage.

This module handles extraction of actual attachment content from email messages,
applying size and format filtering, and storing content with hash-based deduplication.

Key Features:
- Extract attachments from multipart MIME messages
- Size filtering (<50MB default, configurable)
- Format filtering (supported extensions only)
- Content-hash based deduplication (SHA256)
- Privacy-aware audit logging (no content in logs)
- Quarantine support for unsupported/oversized attachments

Design:
- Separation: Works with EmailParser metadata, handles content separately
- Privacy: Audit logs without content exposure
- Resilience: Handles missing filenames, corrupted content, etc.

Integration:
- Input: Raw email message bytes + metadata from EmailParser
- Output: EmailAttachment objects with storage references
- Storage: Configurable local directory with hash-based filenames
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from email import message_from_bytes
from email.policy import default as email_policy
from pathlib import Path
from typing import Any, List, Optional, Set

from .attachment_models import EmailAttachment, AttachmentProcessingStatus
from ...privacy.audit import AuditEvent, AuditLogger

logger = logging.getLogger(__name__)


class AttachmentExtractor:
    """Extract and store email attachments with filtering and deduplication.

    This class handles the full attachment extraction lifecycle:
    1. Extract attachments from MIME multipart messages
    2. Apply size and format filtering
    3. Compute content hash for deduplication
    4. Store content with hash-based filenames
    5. Emit privacy-aware audit events

    Privacy-First Design:
    - Never logs attachment content
    - Audit events contain only metadata (size, type, filename)
    - Storage path anonymization in logs
    """

    # Default supported extensions (aligned with Unstructured.io capabilities)
    SUPPORTED_EXTENSIONS = {
        # Documents
        '.pdf', '.doc', '.docx', '.txt', '.rtf',
        '.odt', '.pages',
        # Spreadsheets
        '.xls', '.xlsx', '.csv', '.numbers',
        # Presentations
        '.ppt', '.pptx', '.key',
        # Images (OCR capable)
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
        # Web & Markup
        '.html', '.htm', '.md', '.xml',
        # Archives (for future support)
        '.zip', '.tar', '.gz',
    }

    def __init__(
        self,
        *,
        max_size_bytes: int = 50 * 1024 * 1024,  # 50MB default
        storage_dir: Path,
        supported_extensions: Optional[Set[str]] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """Initialize attachment extractor.

        Args:
            max_size_bytes: Maximum attachment size to extract (default 50MB)
            storage_dir: Directory for storing attachment content
            supported_extensions: Set of supported file extensions (default: SUPPORTED_EXTENSIONS)
            audit_logger: Optional audit logger for privacy-compliant event recording
        """
        self.max_size_bytes = max_size_bytes
        self.storage_dir = Path(storage_dir)
        self.supported_extensions = supported_extensions or self.SUPPORTED_EXTENSIONS
        self.audit_logger = audit_logger

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"AttachmentExtractor initialized",
            extra={
                "max_size_mb": max_size_bytes / (1024 * 1024),
                "storage_dir": str(storage_dir),
                "supported_extensions_count": len(self.supported_extensions),
            }
        )

    def extract_attachments(
        self,
        raw_message: bytes,
        message_id: str,
        mailbox_id: str,
    ) -> List[EmailAttachment]:
        """Extract all attachments from email message.

        Args:
            raw_message: Raw RFC822/MIME message bytes
            message_id: Email Message-ID
            mailbox_id: Source mailbox identifier

        Returns:
            List of EmailAttachment objects (including skipped attachments)

        Note:
            Skipped attachments (too large, unsupported) are included with
            status=SKIPPED for auditing purposes.
        """
        try:
            msg = message_from_bytes(raw_message, policy=email_policy)
        except Exception as e:
            logger.error(
                f"Failed to parse email message for attachment extraction: {e}",
                extra={"message_id": message_id, "error": str(e)}
            )
            return []

        attachments = []

        if not msg.is_multipart():
            return attachments

        for part in msg.walk():
            if part.get_content_disposition() in ['attachment', 'inline']:
                attachment = self._extract_part(part, message_id, mailbox_id)
                if attachment:
                    attachments.append(attachment)
                    self._log_extraction_event(attachment)

        logger.info(
            f"Extracted {len(attachments)} attachments from email",
            extra={
                "message_id": message_id,
                "attachment_count": len(attachments),
                "mailbox_id": mailbox_id,
            }
        )

        return attachments

    def _extract_part(
        self,
        part: Any,
        message_id: str,
        mailbox_id: str,
    ) -> Optional[EmailAttachment]:
        """Extract single attachment part.

        Args:
            part: MIME part object
            message_id: Email Message-ID
            mailbox_id: Source mailbox identifier

        Returns:
            EmailAttachment object or None if extraction failed
        """
        filename = part.get_filename()
        if not filename:
            logger.debug(
                "Skipping attachment without filename",
                extra={"message_id": message_id}
            )
            return None

        # Get content
        try:
            content = part.get_payload(decode=True)
        except Exception as e:
            logger.warning(
                f"Failed to decode attachment content: {e}",
                extra={"message_id": message_id, "filename": filename, "error": str(e)}
            )
            return None

        if not content:
            logger.debug(
                f"Skipping empty attachment: {filename}",
                extra={"message_id": message_id}
            )
            return None

        size_bytes = len(content)

        # Check size limit
        if size_bytes > self.max_size_bytes:
            logger.info(
                f"Skipping large attachment: {filename}",
                extra={
                    "message_id": message_id,
                    "size_bytes": size_bytes,
                    "max_size_bytes": self.max_size_bytes,
                    "size_mb": size_bytes / (1024 * 1024),
                }
            )
            return self._create_skipped_attachment(
                filename, part, message_id, mailbox_id, size_bytes, "too_large"
            )

        # Check supported format
        extension = Path(filename).suffix.lower()
        if extension not in self.supported_extensions:
            logger.debug(
                f"Skipping unsupported attachment: {filename}",
                extra={
                    "message_id": message_id,
                    "extension": extension,
                    "filename": filename,
                }
            )
            return self._create_skipped_attachment(
                filename, part, message_id, mailbox_id, size_bytes, "unsupported_format"
            )

        # Calculate content hash
        content_hash = EmailAttachment.compute_content_hash(content)

        # Store attachment (with deduplication)
        try:
            storage_path = self._store_attachment(content, content_hash, filename)
        except Exception as e:
            logger.error(
                f"Failed to store attachment: {e}",
                extra={
                    "message_id": message_id,
                    "filename": filename,
                    "error": str(e)
                }
            )
            return self._create_failed_attachment(
                filename, part, message_id, mailbox_id, size_bytes, str(e)
            )

        # Create attachment record
        attachment = EmailAttachment(
            attachment_id=str(uuid.uuid4()),
            message_id=message_id,
            part_id=part.get('Content-ID', '').strip('<>') or str(uuid.uuid4()),
            filename=filename,
            content_type=part.get_content_type(),
            size_bytes=size_bytes,
            is_inline=part.get_content_disposition() == 'inline',
            content_id=part.get('Content-ID', '').strip('<>') or None,
            content_hash=content_hash,
            storage_path=storage_path,
            processing_status=AttachmentProcessingStatus.PENDING,
            extracted_at=datetime.utcnow(),
            mailbox_id=mailbox_id,
        )

        return attachment

    def _store_attachment(
        self,
        content: bytes,
        content_hash: str,
        filename: str,
    ) -> Path:
        """Store attachment content to disk with deduplication.

        Uses content hash as filename to automatically deduplicate identical
        attachments across different emails.

        Args:
            content: Raw attachment content bytes
            content_hash: SHA256 hash of content
            filename: Original filename (for extension)

        Returns:
            Path to stored file

        Raises:
            IOError: If storage fails
        """
        # Use content hash as filename with original extension
        extension = Path(filename).suffix.lower()
        storage_filename = f"{content_hash}{extension}"
        storage_path = self.storage_dir / storage_filename

        # Deduplication: only write if file doesn't exist
        if not storage_path.exists():
            try:
                storage_path.write_bytes(content)
                logger.debug(
                    f"Stored attachment content",
                    extra={
                        "content_hash": content_hash[:16],  # Truncate for logging
                        "size_bytes": len(content),
                        "extension": extension,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Failed to write attachment to storage: {e}",
                    extra={
                        "content_hash": content_hash[:16],
                        "storage_path": str(storage_path),
                        "error": str(e)
                    }
                )
                raise
        else:
            logger.debug(
                f"Attachment already exists (deduplicated)",
                extra={"content_hash": content_hash[:16]}
            )

        return storage_path

    def _create_skipped_attachment(
        self,
        filename: str,
        part: Any,
        message_id: str,
        mailbox_id: str,
        size_bytes: int,
        skip_reason: str,
    ) -> EmailAttachment:
        """Create attachment record for skipped attachment.

        Args:
            filename: Attachment filename
            part: MIME part object
            message_id: Email Message-ID
            mailbox_id: Source mailbox identifier
            size_bytes: Attachment size
            skip_reason: Reason for skipping

        Returns:
            EmailAttachment with status=SKIPPED
        """
        attachment = EmailAttachment(
            attachment_id=str(uuid.uuid4()),
            message_id=message_id,
            part_id=part.get('Content-ID', '').strip('<>') or str(uuid.uuid4()),
            filename=filename,
            content_type=part.get_content_type(),
            size_bytes=size_bytes,
            is_inline=False,
            content_id=None,
            content_hash="",
            storage_path=None,
            processing_status=AttachmentProcessingStatus.SKIPPED,
            processing_error=skip_reason,
            extracted_at=datetime.utcnow(),
            mailbox_id=mailbox_id,
        )

        return attachment

    def _create_failed_attachment(
        self,
        filename: str,
        part: Any,
        message_id: str,
        mailbox_id: str,
        size_bytes: int,
        error: str,
    ) -> EmailAttachment:
        """Create attachment record for failed extraction.

        Args:
            filename: Attachment filename
            part: MIME part object
            message_id: Email Message-ID
            mailbox_id: Source mailbox identifier
            size_bytes: Attachment size
            error: Error message

        Returns:
            EmailAttachment with status=FAILED
        """
        attachment = EmailAttachment(
            attachment_id=str(uuid.uuid4()),
            message_id=message_id,
            part_id=part.get('Content-ID', '').strip('<>') or str(uuid.uuid4()),
            filename=filename,
            content_type=part.get_content_type(),
            size_bytes=size_bytes,
            is_inline=False,
            content_id=None,
            content_hash="",
            storage_path=None,
            processing_status=AttachmentProcessingStatus.FAILED,
            processing_error=error,
            extracted_at=datetime.utcnow(),
            mailbox_id=mailbox_id,
        )

        return attachment

    def _log_extraction_event(self, attachment: EmailAttachment) -> None:
        """Log attachment extraction event (privacy-aware).

        Args:
            attachment: Extracted attachment
        """
        if not self.audit_logger:
            return

        self.audit_logger.record(
            AuditEvent(
                job_id=f"attachment_extract_{attachment.attachment_id}",
                source="imap_attachment_extractor",
                action="attachment_extracted",
                status="success" if not attachment.is_failed else "failed",
                timestamp=datetime.utcnow(),
                metadata={
                    "attachment_id": attachment.attachment_id,
                    "message_id_hash": attachment.message_id[:16],  # Truncate for privacy
                    "attachment_ext": Path(attachment.filename).suffix.lower(),
                    "content_type": attachment.content_type,
                    "size_bytes": attachment.size_bytes,
                    "size_mb": attachment.size_mb,
                    "is_inline": attachment.is_inline,
                    "processing_status": attachment.processing_status.value,
                    "has_storage": attachment.has_content,
                    "mailbox_id": attachment.mailbox_id,
                },
            )
        )


__all__ = [
    "AttachmentExtractor",
]
