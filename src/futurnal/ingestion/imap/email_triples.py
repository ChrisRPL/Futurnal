"""Semantic triple extraction from email metadata for PKG construction.

This module extracts structured semantic relationships from EmailMessage objects
to populate the Personal Knowledge Graph (PKG). Email triples capture:
- Email entities and their properties (subject, date, participants)
- Person entities and their metadata (display names)
- Threading relationships for conversation reconstruction
- Temporal relationships for experiential timeline

These triples form the foundation for the Ghost's understanding of email
conversations, enabling pattern detection, participant analysis, and causal
exploration in Phase 2 and beyond.

Ghost→Animal Evolution:
- **Phase 1 (CURRENT)**: Metadata-based triple extraction from headers
- **Phase 2 (FUTURE)**: Content-based relationship extraction with LLM
- **Phase 3 (FUTURE)**: Causal relationship inference from email patterns
"""

from __future__ import annotations

import logging
from typing import List

from .email_parser import EmailMessage, EmailAddress
from ...pipeline.triples import SemanticTriple

logger = logging.getLogger(__name__)


def extract_email_triples(email_message: EmailMessage) -> List[SemanticTriple]:
    """Extract semantic triples from email metadata.

    Generates PKG triples representing:
    - Email entity with properties (subject, date, folder)
    - Sender and recipient person entities
    - Threading relationships (in-reply-to, references)
    - Participant relationships (from, to, cc)
    - Temporal metadata (sent date, retrieved date)

    Args:
        email_message: Parsed email message

    Returns:
        List of SemanticTriple objects for PKG storage

    Example triples generated:
        (email:abc123, rdf:type, futurnal:Email)
        (email:abc123, email:subject, "Meeting Notes")
        (email:abc123, email:from, person:john@example.com)
        (email:abc123, email:to, person:jane@example.com)
        (email:abc123, email:sentDate, "2024-01-15T10:30:00Z")
        (email:abc123, email:inReplyTo, email:xyz789)
        (person:john@example.com, person:displayName, "John Doe")
    """
    triples = []

    # Create email URI
    email_uri = _create_email_uri(email_message.message_id)
    source_element_id = email_message.message_id

    # Email type triple
    triples.append(
        SemanticTriple(
            subject=email_uri,
            predicate="rdf:type",
            object="futurnal:Email",
            source_element_id=source_element_id,
            source_path=f"{email_message.mailbox_id}/{email_message.folder}",
            extraction_method="email_metadata",
        )
    )

    # Subject triple
    if email_message.subject:
        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate="email:subject",
                object=email_message.subject,
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

    # Sender triples
    triples.extend(
        _extract_sender_triples(
            email_uri, email_message.from_address, source_element_id
        )
    )

    # Recipient triples
    triples.extend(
        _extract_recipient_triples(
            email_uri, email_message.to_addresses, "email:to", source_element_id
        )
    )

    triples.extend(
        _extract_recipient_triples(
            email_uri, email_message.cc_addresses, "email:cc", source_element_id
        )
    )

    # Date triple
    triples.append(
        SemanticTriple(
            subject=email_uri,
            predicate="email:sentDate",
            object=email_message.date.isoformat(),
            source_element_id=source_element_id,
            extraction_method="email_metadata",
        )
    )

    # Folder triple
    triples.append(
        SemanticTriple(
            subject=email_uri,
            predicate="email:folder",
            object=email_message.folder,
            source_element_id=source_element_id,
            extraction_method="email_metadata",
        )
    )

    # Threading triples
    triples.extend(_extract_threading_triples(email_uri, email_message, source_element_id))

    # Attachment triples
    if email_message.has_attachments:
        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate="email:hasAttachments",
                object="true",
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate="email:attachmentCount",
                object=str(len(email_message.attachments)),
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

    # Privacy classification triple
    if email_message.contains_sensitive_keywords:
        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate="email:containsSensitiveKeywords",
                object="true",
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

    return triples


def _extract_sender_triples(
    email_uri: str, from_address: EmailAddress, source_element_id: str
) -> List[SemanticTriple]:
    """Extract sender-related triples.

    Args:
        email_uri: Email entity URI
        from_address: Sender email address
        source_element_id: Source message ID

    Returns:
        List of sender triples
    """
    triples = []
    person_uri = _create_person_uri(from_address.address)

    # Email from person
    triples.append(
        SemanticTriple(
            subject=email_uri,
            predicate="email:from",
            object=person_uri,
            source_element_id=source_element_id,
            extraction_method="email_metadata",
        )
    )

    # Person type
    triples.append(
        SemanticTriple(
            subject=person_uri,
            predicate="rdf:type",
            object="futurnal:Person",
            source_element_id=source_element_id,
            extraction_method="email_metadata",
        )
    )

    # Person email address
    triples.append(
        SemanticTriple(
            subject=person_uri,
            predicate="person:emailAddress",
            object=from_address.address,
            source_element_id=source_element_id,
            extraction_method="email_metadata",
        )
    )

    # Person display name (if available)
    if from_address.display_name:
        triples.append(
            SemanticTriple(
                subject=person_uri,
                predicate="person:displayName",
                object=from_address.display_name,
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

    return triples


def _extract_recipient_triples(
    email_uri: str,
    addresses: List[EmailAddress],
    predicate: str,
    source_element_id: str,
) -> List[SemanticTriple]:
    """Extract recipient-related triples.

    Args:
        email_uri: Email entity URI
        addresses: List of recipient addresses
        predicate: Relationship predicate (email:to or email:cc)
        source_element_id: Source message ID

    Returns:
        List of recipient triples
    """
    triples = []

    for addr in addresses:
        person_uri = _create_person_uri(addr.address)

        # Email to/cc person
        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate=predicate,
                object=person_uri,
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

        # Person type
        triples.append(
            SemanticTriple(
                subject=person_uri,
                predicate="rdf:type",
                object="futurnal:Person",
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

        # Person email address
        triples.append(
            SemanticTriple(
                subject=person_uri,
                predicate="person:emailAddress",
                object=addr.address,
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

        # Person display name (if available)
        if addr.display_name:
            triples.append(
                SemanticTriple(
                    subject=person_uri,
                    predicate="person:displayName",
                    object=addr.display_name,
                    source_element_id=source_element_id,
                    extraction_method="email_metadata",
                )
            )

    return triples


def _extract_threading_triples(
    email_uri: str, email_message: EmailMessage, source_element_id: str
) -> List[SemanticTriple]:
    """Extract email threading triples.

    Args:
        email_uri: Email entity URI
        email_message: Email message
        source_element_id: Source message ID

    Returns:
        List of threading triples
    """
    triples = []

    # In-Reply-To triple (direct parent in conversation)
    if email_message.in_reply_to:
        parent_uri = _create_email_uri(email_message.in_reply_to)

        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate="email:inReplyTo",
                object=parent_uri,
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

    # References triples (full thread chain)
    for i, ref_msg_id in enumerate(email_message.references):
        ref_uri = _create_email_uri(ref_msg_id)

        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate="email:references",
                object=ref_uri,
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

        # Add thread order metadata for temporal sequencing
        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate=f"email:threadPosition",
                object=str(i),
                source_element_id=source_element_id,
                extraction_method="email_metadata",
            )
        )

    return triples


def _create_email_uri(message_id: str) -> str:
    """Create a URI for an email message.

    Args:
        message_id: Email Message-ID

    Returns:
        Email entity URI (e.g., "email:abc123@example.com")
    """
    # Clean message ID and create URI
    clean_id = message_id.replace("<", "").replace(">", "").replace(" ", "_")
    return f"futurnal:email/{clean_id}"


def _create_person_uri(email_address: str) -> str:
    """Create a URI for a person based on email address.

    Args:
        email_address: Email address

    Returns:
        Person entity URI (e.g., "person:john@example.com")
    """
    clean_addr = email_address.lower().replace(" ", "_")
    return f"futurnal:person/{clean_addr}"


__all__ = ["extract_email_triples"]
