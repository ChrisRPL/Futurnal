"""Semantic triple extraction from email attachments for PKG construction.

This module extracts structured semantic relationships from EmailAttachment objects
to populate the Personal Knowledge Graph (PKG). Attachment triples capture:
- Attachment entities and their properties (filename, type, size)
- Relationship to parent email
- Processing status and provenance
- Content type and format metadata

These triples enable the Ghost to understand attachment patterns in communication,
identify frequently shared document types, and build context around email content.

Ghost→Animal Evolution:
- **Phase 1 (CURRENT)**: Metadata-based triple extraction from attachment properties
- **Phase 2 (FUTURE)**: Content-based relationship extraction from Unstructured.io elements
- **Phase 3 (FUTURE)**: Pattern recognition in attachment sharing behavior
"""

from __future__ import annotations

import logging
from typing import List

from .attachment_models import EmailAttachment
from ...pipeline.triples import SemanticTriple

logger = logging.getLogger(__name__)


def extract_attachment_triples(
    attachment: EmailAttachment,
    email_message_id: str,
) -> List[SemanticTriple]:
    """Extract semantic triples from email attachment metadata.

    Generates PKG triples representing:
    - Attachment entity with properties (filename, content type, size)
    - Link to parent email (partOfEmail relationship)
    - Processing status and metadata
    - Content format information

    Args:
        attachment: EmailAttachment object
        email_message_id: Parent email Message-ID

    Returns:
        List of SemanticTriple objects for PKG storage

    Example triples generated:
        (attachment:abc123, rdf:type, futurnal:EmailAttachment)
        (attachment:abc123, attachment:partOfEmail, email:xyz789)
        (attachment:abc123, attachment:filename, "document.pdf")
        (attachment:abc123, attachment:contentType, "application/pdf")
        (attachment:abc123, attachment:sizeBytes, "1024000")
        (attachment:abc123, attachment:processingStatus, "completed")
        (email:xyz789, email:hasAttachment, attachment:abc123)
    """
    triples = []

    # Create URIs
    attachment_uri = _create_attachment_uri(attachment.attachment_id)
    email_uri = _create_email_uri(email_message_id)
    source_element_id = attachment.attachment_id

    # Attachment type triple
    triples.append(
        SemanticTriple(
            subject=attachment_uri,
            predicate="rdf:type",
            object="futurnal:EmailAttachment",
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    # Link to parent email (bidirectional)
    triples.append(
        SemanticTriple(
            subject=attachment_uri,
            predicate="attachment:partOfEmail",
            object=email_uri,
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=email_uri,
            predicate="email:hasAttachment",
            object=attachment_uri,
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    # Filename triple
    triples.append(
        SemanticTriple(
            subject=attachment_uri,
            predicate="attachment:filename",
            object=attachment.filename,
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    # Content type triple
    triples.append(
        SemanticTriple(
            subject=attachment_uri,
            predicate="attachment:contentType",
            object=attachment.content_type,
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    # Size triple
    triples.append(
        SemanticTriple(
            subject=attachment_uri,
            predicate="attachment:sizeBytes",
            object=str(attachment.size_bytes),
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    # Inline flag
    if attachment.is_inline:
        triples.append(
            SemanticTriple(
                subject=attachment_uri,
                predicate="attachment:isInline",
                object="true",
                source_element_id=source_element_id,
                extraction_method="attachment_metadata",
            )
        )

        # Content-ID for inline images
        if attachment.content_id:
            triples.append(
                SemanticTriple(
                    subject=attachment_uri,
                    predicate="attachment:contentId",
                    object=attachment.content_id,
                    source_element_id=source_element_id,
                    extraction_method="attachment_metadata",
                )
            )

    # Processing status triple
    triples.append(
        SemanticTriple(
            subject=attachment_uri,
            predicate="attachment:processingStatus",
            object=attachment.processing_status.value,
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    # Processing metadata (if processed)
    if attachment.is_processed:
        triples.append(
            SemanticTriple(
                subject=attachment_uri,
                predicate="attachment:extractionElements",
                object=str(attachment.extraction_elements),
                source_element_id=source_element_id,
                extraction_method="attachment_metadata",
            )
        )

        if attachment.processed_at:
            triples.append(
                SemanticTriple(
                    subject=attachment_uri,
                    predicate="attachment:processedAt",
                    object=attachment.processed_at.isoformat(),
                    source_element_id=source_element_id,
                    extraction_method="attachment_metadata",
                )
            )

    # Content hash (for deduplication tracking)
    if attachment.content_hash:
        triples.append(
            SemanticTriple(
                subject=attachment_uri,
                predicate="attachment:contentHash",
                object=attachment.content_hash,
                source_element_id=source_element_id,
                extraction_method="attachment_metadata",
            )
        )

    # Temporal metadata
    triples.append(
        SemanticTriple(
            subject=attachment_uri,
            predicate="attachment:extractedAt",
            object=attachment.extracted_at.isoformat(),
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    # Provenance
    triples.append(
        SemanticTriple(
            subject=attachment_uri,
            predicate="attachment:mailboxId",
            object=attachment.mailbox_id,
            source_element_id=source_element_id,
            extraction_method="attachment_metadata",
        )
    )

    # Error metadata (if failed or skipped)
    if attachment.is_failed or attachment.is_skipped:
        if attachment.processing_error:
            triples.append(
                SemanticTriple(
                    subject=attachment_uri,
                    predicate="attachment:processingError",
                    object=attachment.processing_error,
                    source_element_id=source_element_id,
                    extraction_method="attachment_metadata",
                )
            )

    return triples


def _create_attachment_uri(attachment_id: str) -> str:
    """Create a URI for an email attachment.

    Args:
        attachment_id: Attachment UUID

    Returns:
        Attachment entity URI (e.g., "futurnal:attachment/abc-123-def")
    """
    return f"futurnal:attachment/{attachment_id}"


def _create_email_uri(message_id: str) -> str:
    """Create a URI for an email message.

    Args:
        message_id: Email Message-ID

    Returns:
        Email entity URI (e.g., "futurnal:email/abc123@example.com")
    """
    # Clean message ID and create URI
    clean_id = message_id.replace("<", "").replace(">", "").replace(" ", "_")
    return f"futurnal:email/{clean_id}"


__all__ = ["extract_attachment_triples"]
