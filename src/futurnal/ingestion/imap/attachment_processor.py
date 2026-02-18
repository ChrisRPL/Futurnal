"""Attachment processing through Unstructured.io pipeline.

This module handles processing of extracted email attachments through Unstructured.io
to extract text, tables, and structured content for Ghost learning.

Key Features:
- Process attachments through Unstructured.io partition()
- OCR support for images with configurable languages
- Timeout enforcement (60s default)
- Element enrichment with attachment metadata
- Comprehensive error handling with quarantine support

Integration:
- Input: EmailAttachment objects with storage paths
- Processing: Unstructured.io partition() with async timeout
- Output: List of enriched element dictionaries
- Metadata: Adds source_attachment_id, source_message_id to elements

Ghost Learning:
The extracted elements feed into the Ghost's learning pipeline, enabling it to
understand attachment content (PDFs, documents, images) while respecting privacy
and resource constraints.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .attachment_models import EmailAttachment, AttachmentProcessingStatus
from ...privacy.audit import AuditEvent, AuditLogger

logger = logging.getLogger(__name__)


class AttachmentProcessor:
    """Process attachments through Unstructured.io extraction pipeline.

    This class handles the full processing lifecycle:
    1. Validate attachment is ready for processing
    2. Process file through Unstructured.io with timeout
    3. Enrich extracted elements with attachment metadata
    4. Update attachment status and statistics
    5. Handle errors with appropriate status updates

    Privacy & Resource Management:
    - Processing timeout enforced (60s default)
    - No attachment content in logs
    - Failed processing tracked with error details
    - Supports retry for temporary failures
    """

    def __init__(
        self,
        *,
        ocr_languages: str = "eng",
        processing_timeout: int = 60,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """Initialize attachment processor.

        Args:
            ocr_languages: OCR language codes (e.g., "eng", "eng+fra")
            processing_timeout: Maximum processing time in seconds
            audit_logger: Optional audit logger for privacy-compliant event recording
        """
        self.ocr_languages = ocr_languages
        self.processing_timeout = processing_timeout
        self.audit_logger = audit_logger

        logger.info(
            f"AttachmentProcessor initialized",
            extra={
                "ocr_languages": ocr_languages,
                "processing_timeout_sec": processing_timeout,
            }
        )

    async def process_attachment(
        self,
        attachment: EmailAttachment,
    ) -> List[Dict[str, Any]]:
        """Process attachment through Unstructured.io.

        Args:
            attachment: EmailAttachment to process

        Returns:
            List of enriched element dictionaries from Unstructured.io

        Note:
            - Updates attachment.processing_status in-place
            - Returns empty list on failure (status updated to FAILED)
            - Enforces timeout (attachment marked FAILED if timeout)
        """
        # Validate attachment is ready for processing
        if attachment.processing_status != AttachmentProcessingStatus.PENDING:
            logger.debug(
                f"Skipping attachment with status {attachment.processing_status}",
                extra={
                    "attachment_id": attachment.attachment_id,
                    "status": attachment.processing_status.value,
                }
            )
            return []

        if not attachment.storage_path or not attachment.storage_path.exists():
            error_msg = f"Attachment file not found: {attachment.filename}"
            logger.error(
                error_msg,
                extra={
                    "attachment_id": attachment.attachment_id,
                    "storage_path": str(attachment.storage_path) if attachment.storage_path else None,
                }
            )
            attachment.mark_failed(error_msg)
            self._log_processing_event(attachment, success=False)
            return []

        # Mark as processing
        attachment.mark_processing()

        try:
            # Process with Unstructured.io with timeout
            elements = await self._process_with_unstructured(attachment)

            # Mark as completed
            attachment.mark_completed(extraction_elements=len(elements))

            logger.info(
                f"Successfully processed attachment",
                extra={
                    "attachment_id": attachment.attachment_id,
                    "attachment_filename": attachment.filename,
                    "element_count": len(elements),
                }
            )

            self._log_processing_event(attachment, success=True)
            return elements

        except asyncio.TimeoutError:
            error_msg = f"Processing timeout ({self.processing_timeout}s)"
            logger.error(
                error_msg,
                extra={
                    "attachment_id": attachment.attachment_id,
                    "attachment_filename": attachment.filename,
                    "timeout_sec": self.processing_timeout,
                }
            )
            attachment.mark_failed(error_msg)
            self._log_processing_event(attachment, success=False)
            return []

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(
                error_msg,
                extra={
                    "attachment_id": attachment.attachment_id,
                    "attachment_filename": attachment.filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            attachment.mark_failed(error_msg)
            self._log_processing_event(attachment, success=False)
            return []

    async def _process_with_unstructured(
        self,
        attachment: EmailAttachment,
    ) -> List[Dict[str, Any]]:
        """Process file with Unstructured.io partition().

        Args:
            attachment: EmailAttachment with storage path

        Returns:
            List of enriched element dictionaries

        Raises:
            asyncio.TimeoutError: If processing exceeds timeout
            Exception: On Unstructured.io processing errors
        """
        from unstructured.partition.auto import partition

        # Process with timeout
        elements = await asyncio.wait_for(
            asyncio.to_thread(
                partition,
                filename=str(attachment.storage_path),
                languages=[self.ocr_languages],
            ),
            timeout=self.processing_timeout,
        )

        # Some tests/mock setups provide async partition callables.
        if asyncio.iscoroutine(elements):
            elements = await asyncio.wait_for(
                elements,
                timeout=self.processing_timeout,
            )

        # Convert to dicts and enrich with metadata
        element_dicts = []
        for element in elements:
            element_dict = element.to_dict()

            # Enrich with attachment metadata for PKG integration
            if 'metadata' not in element_dict:
                element_dict['metadata'] = {}

            element_dict['metadata']['source_attachment_id'] = attachment.attachment_id
            element_dict['metadata']['source_message_id'] = attachment.message_id
            element_dict['metadata']['attachment_filename'] = attachment.filename
            element_dict['metadata']['attachment_content_type'] = attachment.content_type
            element_dict['metadata']['attachment_size_bytes'] = attachment.size_bytes

            element_dicts.append(element_dict)

        return element_dicts

    def _log_processing_event(
        self,
        attachment: EmailAttachment,
        success: bool
    ) -> None:
        """Log attachment processing event (privacy-aware).

        Args:
            attachment: Processed attachment
            success: True if processing succeeded
        """
        if not self.audit_logger:
            return

        self.audit_logger.record(
            AuditEvent(
                job_id=f"attachment_process_{attachment.attachment_id}",
                source="imap_attachment_processor",
                action="attachment_processed",
                status="success" if success else "failed",
                timestamp=datetime.utcnow(),
                metadata={
                    "attachment_id": attachment.attachment_id,
                    "message_id_hash": attachment.message_id[:16],  # Truncate for privacy
                    "attachment_ext": attachment.filename.split('.')[-1] if '.' in attachment.filename else "",
                    "content_type": attachment.content_type,
                    "size_bytes": attachment.size_bytes,
                    "size_mb": attachment.size_mb,
                    "processing_status": attachment.processing_status.value,
                    "extraction_elements": attachment.extraction_elements,
                    "processing_error": attachment.processing_error if not success else None,
                    "mailbox_id": attachment.mailbox_id,
                },
            )
        )


__all__ = [
    "AttachmentProcessor",
]
