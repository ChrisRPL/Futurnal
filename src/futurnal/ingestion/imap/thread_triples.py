"""Semantic triple extraction from email threads for PKG construction.

Extracts structured semantic relationships from reconstructed email threads
to populate the Personal Knowledge Graph (PKG). Thread triples capture:
- Thread entities and their properties (subject, participants, temporal data)
- Message-to-thread relationships for conversation context
- Participant relationships and roles in conversations
- Conversation flow (parent-child message relationships)
- Thread metadata for pattern detection and causal analysis

These triples enable the Ghost's understanding of communication patterns,
relationship dynamics, and conversational context evolution, laying the
foundation for Phase 2 correlation detection and Phase 3 causal inference.

Ghost→Animal Evolution:
- **Phase 1 (CURRENT)**: Metadata-based thread triple extraction
- **Phase 2 (FUTURE)**: Behavioral pattern extraction (response patterns, participation dynamics)
- **Phase 3 (FUTURE)**: Causal relationship inference from conversation patterns
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .thread_models import EmailThread
from .thread_reconstructor import ThreadReconstructor
from ...pipeline.triples import SemanticTriple

logger = logging.getLogger(__name__)


def extract_thread_triples(
    thread: EmailThread, reconstructor: ThreadReconstructor
) -> List[SemanticTriple]:
    """Extract semantic triples from reconstructed email thread.

    Generates PKG triples representing:
    - Thread entity with properties (subject, counts, dates, structure)
    - Message-to-thread relationships for all messages in conversation
    - Participant relationships with roles (initiator, recipient, participant)
    - Conversation flow via parent-child message relationships
    - Thread metadata for temporal and structural analysis

    Args:
        thread: Reconstructed EmailThread
        reconstructor: ThreadReconstructor with message graph

    Returns:
        List of SemanticTriple objects for PKG storage

    Example triples generated:
        (thread:abc123, rdf:type, futurnal:EmailThread)
        (thread:abc123, thread:subject, "Project Discussion")
        (thread:abc123, thread:messageCount, "5")
        (email:msg1, email:partOfThread, thread:abc123)
        (person:john@example.com, person:participatedIn, thread:abc123)
        (person:john@example.com, person:threadRole, "initiator")
        (email:msg2, email:inReplyTo, email:msg1)
    """
    triples = []

    # Create thread URI
    thread_uri = _create_thread_uri(thread.thread_id)
    source_element_id = thread.thread_id

    # ========================================================================
    # Thread Entity Triples
    # ========================================================================

    # Thread type
    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="rdf:type",
            object="futurnal:EmailThread",
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    # Thread subject
    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="thread:subject",
            object=thread.subject,
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    # Thread counts
    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="thread:messageCount",
            object=str(thread.message_count),
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="thread:participantCount",
            object=str(thread.participant_count),
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    # Thread temporal metadata
    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="thread:startDate",
            object=thread.start_date.isoformat(),
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="thread:lastMessageDate",
            object=thread.last_message_date.isoformat(),
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="thread:durationDays",
            object=str(round(thread.duration_days, 2)),
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    # Thread structure metadata
    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="thread:depth",
            object=str(thread.depth),
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    triples.append(
        SemanticTriple(
            subject=thread_uri,
            predicate="thread:branchCount",
            object=str(thread.branch_count),
            source_element_id=source_element_id,
            extraction_method="thread_reconstruction",
        )
    )

    # Thread response time analytics
    if thread.average_response_time_minutes > 0:
        triples.append(
            SemanticTriple(
                subject=thread_uri,
                predicate="thread:averageResponseTimeMinutes",
                object=str(round(thread.average_response_time_minutes, 2)),
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

        triples.append(
            SemanticTriple(
                subject=thread_uri,
                predicate="thread:totalResponseTimeMinutes",
                object=str(round(thread.total_response_time_minutes, 2)),
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

    # Thread characteristics (boolean properties)
    if thread.has_attachments:
        triples.append(
            SemanticTriple(
                subject=thread_uri,
                predicate="thread:hasAttachments",
                object="true",
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

    # ========================================================================
    # Message-to-Thread Relationship Triples
    # ========================================================================

    for message_id in thread.message_ids:
        email_uri = _create_email_uri(message_id)

        # Link message to thread
        triples.append(
            SemanticTriple(
                subject=email_uri,
                predicate="email:partOfThread",
                object=thread_uri,
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

    # ========================================================================
    # Participant Relationship Triples
    # ========================================================================

    for participant in thread.participants:
        person_uri = _create_person_uri(participant.email_address)

        # Person participated in thread
        triples.append(
            SemanticTriple(
                subject=person_uri,
                predicate="person:participatedIn",
                object=thread_uri,
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

        # Person role in thread
        triples.append(
            SemanticTriple(
                subject=person_uri,
                predicate="person:threadRole",
                object=participant.role.value,
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

        # Participant message count in thread
        triples.append(
            SemanticTriple(
                subject=person_uri,
                predicate="person:threadMessageCount",
                object=str(participant.message_count),
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

        # Participant temporal engagement
        triples.append(
            SemanticTriple(
                subject=person_uri,
                predicate="person:firstThreadMessageDate",
                object=participant.first_message_date.isoformat(),
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

        triples.append(
            SemanticTriple(
                subject=person_uri,
                predicate="person:lastThreadMessageDate",
                object=participant.last_message_date.isoformat(),
                source_element_id=source_element_id,
                extraction_method="thread_reconstruction",
            )
        )

    # ========================================================================
    # Conversation Flow Triples (Parent-Child Relationships)
    # ========================================================================

    for message_id in thread.message_ids:
        if message_id not in reconstructor.message_graph:
            continue

        node = reconstructor.message_graph[message_id]

        # Add parent-child relationship if parent exists
        if node.parent_message_id:
            email_uri = _create_email_uri(message_id)
            parent_uri = _create_email_uri(node.parent_message_id)

            triples.append(
                SemanticTriple(
                    subject=email_uri,
                    predicate="email:inReplyTo",
                    object=parent_uri,
                    source_element_id=source_element_id,
                    extraction_method="thread_reconstruction",
                )
            )

            # Add depth metadata for this message in thread
            triples.append(
                SemanticTriple(
                    subject=email_uri,
                    predicate="email:threadDepth",
                    object=str(node.depth),
                    source_element_id=source_element_id,
                    extraction_method="thread_reconstruction",
                )
            )

    logger.debug(
        f"Extracted {len(triples)} triples from thread {thread.thread_id}",
        extra={
            "thread_id": thread.thread_id,
            "triple_count": len(triples),
            "message_count": thread.message_count,
            "participant_count": thread.participant_count,
        },
    )

    return triples


def _create_thread_uri(thread_id: str) -> str:
    """Create a URI for an email thread.

    Args:
        thread_id: Thread identifier (typically root Message-ID)

    Returns:
        Thread entity URI (e.g., "futurnal:thread/abc123@example.com")
    """
    # Clean thread ID and create URI
    clean_id = thread_id.replace("<", "").replace(">", "").replace(" ", "_")
    return f"futurnal:thread/{clean_id}"


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


def _create_person_uri(email_address: str) -> str:
    """Create a URI for a person based on email address.

    Args:
        email_address: Email address

    Returns:
        Person entity URI (e.g., "futurnal:person/john@example.com")
    """
    clean_addr = email_address.lower().replace(" ", "_")
    return f"futurnal:person/{clean_addr}"


__all__ = ["extract_thread_triples"]
