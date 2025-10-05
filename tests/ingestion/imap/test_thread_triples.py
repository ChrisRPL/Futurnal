"""Comprehensive tests for thread triple extraction.

Tests cover:
- Thread entity triple generation
- Thread metadata triples (subject, counts, dates, structure)
- Message-to-thread relationship triples
- Participant relationship triples
- Conversation flow triples (parent-child)
- URI creation and formatting
- Integration with ThreadReconstructor
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from futurnal.ingestion.imap.email_parser import EmailAddress, EmailMessage
from futurnal.ingestion.imap.thread_models import (
    EmailThread,
    ParticipantRole,
    ThreadParticipant,
    ThreadNode,
)
from futurnal.ingestion.imap.thread_reconstructor import ThreadReconstructor
from futurnal.ingestion.imap.thread_triples import extract_thread_triples


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def reconstructor():
    """Create a fresh ThreadReconstructor instance."""
    return ThreadReconstructor()


def create_email_message(
    message_id: str,
    subject: str = "Test Subject",
    from_addr: str = "sender@example.com",
    date: datetime | None = None,
    in_reply_to: str | None = None,
) -> EmailMessage:
    """Helper to create test EmailMessage."""
    if date is None:
        date = datetime.utcnow()

    return EmailMessage(
        message_id=message_id,
        uid=1,
        folder="INBOX",
        subject=subject,
        from_address=EmailAddress(address=from_addr),
        date=date,
        in_reply_to=in_reply_to,
        size_bytes=1000,
        retrieved_at=datetime.utcnow(),
        mailbox_id="test_mailbox",
    )


def create_simple_thread() -> EmailThread:
    """Helper to create a simple test thread."""
    now = datetime.utcnow()
    participant = ThreadParticipant(
        email_address="alice@example.com",
        role=ParticipantRole.INITIATOR,
        message_count=2,
        first_message_date=now,
        last_message_date=now + timedelta(minutes=10),
    )

    return EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        message_ids=["msg1@example.com", "msg2@example.com"],
        message_count=2,
        participants=[participant],
        participant_count=1,
        subject="Test Discussion",
        start_date=now,
        last_message_date=now + timedelta(minutes=10),
        duration_days=0.007,  # ~10 minutes
        depth=1,
        branch_count=0,
        mailbox_id="test_mailbox",
        reconstructed_at=now,
    )


# ============================================================================
# Thread Entity Triple Tests
# ============================================================================


def test_thread_type_triple(reconstructor):
    """Test thread type triple generation."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # Find type triple
    type_triple = next(t for t in triples if t.predicate == "rdf:type")
    assert type_triple.object == "futurnal:EmailThread"
    assert "thread1@example.com" in type_triple.subject


def test_thread_subject_triple(reconstructor):
    """Test thread subject triple."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    subject_triple = next(t for t in triples if t.predicate == "thread:subject")
    assert subject_triple.object == "Test Discussion"


def test_thread_message_count_triple(reconstructor):
    """Test thread message count triple."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    count_triple = next(t for t in triples if t.predicate == "thread:messageCount")
    assert count_triple.object == "2"


def test_thread_participant_count_triple(reconstructor):
    """Test thread participant count triple."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    count_triple = next(t for t in triples if t.predicate == "thread:participantCount")
    assert count_triple.object == "1"


# ============================================================================
# Thread Temporal Metadata Tests
# ============================================================================


def test_thread_temporal_triples(reconstructor):
    """Test thread temporal metadata triples."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # Check start date
    start_triple = next(t for t in triples if t.predicate == "thread:startDate")
    assert start_triple.object  # Should have ISO format date

    # Check last message date
    last_triple = next(t for t in triples if t.predicate == "thread:lastMessageDate")
    assert last_triple.object

    # Check duration
    duration_triple = next(t for t in triples if t.predicate == "thread:durationDays")
    assert float(duration_triple.object) >= 0


# ============================================================================
# Thread Structure Metadata Tests
# ============================================================================


def test_thread_depth_triple(reconstructor):
    """Test thread depth triple."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    depth_triple = next(t for t in triples if t.predicate == "thread:depth")
    assert depth_triple.object == "1"


def test_thread_branch_count_triple(reconstructor):
    """Test thread branch count triple."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    branch_triple = next(t for t in triples if t.predicate == "thread:branchCount")
    assert branch_triple.object == "0"


# ============================================================================
# Response Time Triple Tests
# ============================================================================


def test_thread_response_time_triples(reconstructor):
    """Test response time triples when present."""
    thread = create_simple_thread()
    thread.average_response_time_minutes = 15.5
    thread.total_response_time_minutes = 31.0

    triples = extract_thread_triples(thread, reconstructor)

    avg_triple = next(t for t in triples if t.predicate == "thread:averageResponseTimeMinutes")
    assert avg_triple.object == "15.5"

    total_triple = next(t for t in triples if t.predicate == "thread:totalResponseTimeMinutes")
    assert total_triple.object == "31.0"


def test_thread_no_response_time_triples_when_zero(reconstructor):
    """Test no response time triples when zero."""
    thread = create_simple_thread()
    thread.average_response_time_minutes = 0.0
    thread.total_response_time_minutes = 0.0

    triples = extract_thread_triples(thread, reconstructor)

    # Should not have response time triples
    response_triples = [
        t for t in triples
        if "ResponseTime" in t.predicate
    ]
    assert len(response_triples) == 0


# ============================================================================
# Thread Characteristics Tests
# ============================================================================


def test_thread_has_attachments_triple(reconstructor):
    """Test has_attachments triple when true."""
    thread = create_simple_thread()
    thread.has_attachments = True

    triples = extract_thread_triples(thread, reconstructor)

    attachment_triple = next(
        t for t in triples if t.predicate == "thread:hasAttachments"
    )
    assert attachment_triple.object == "true"


def test_thread_no_attachments_triple_when_false(reconstructor):
    """Test no attachments triple when false."""
    thread = create_simple_thread()
    thread.has_attachments = False

    triples = extract_thread_triples(thread, reconstructor)

    # Should not have attachments triple
    attachment_triples = [
        t for t in triples if t.predicate == "thread:hasAttachments"
    ]
    assert len(attachment_triples) == 0


# ============================================================================
# Message-to-Thread Relationship Tests
# ============================================================================


def test_message_to_thread_relationships(reconstructor):
    """Test message-to-thread relationship triples."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # Find all partOfThread triples
    part_of_triples = [t for t in triples if t.predicate == "email:partOfThread"]

    # Should have one for each message
    assert len(part_of_triples) == 2

    # Check they point to correct thread
    for triple in part_of_triples:
        assert "thread1@example.com" in triple.object


# ============================================================================
# Participant Relationship Tests
# ============================================================================


def test_participant_relationships(reconstructor):
    """Test participant relationship triples."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # Find participatedIn triple
    participated_triple = next(
        t for t in triples if t.predicate == "person:participatedIn"
    )
    assert "alice@example.com" in participated_triple.subject
    assert "thread1@example.com" in participated_triple.object


def test_participant_role_triple(reconstructor):
    """Test participant role triple."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    role_triple = next(t for t in triples if t.predicate == "person:threadRole")
    assert role_triple.object == "initiator"


def test_participant_message_count_triple(reconstructor):
    """Test participant message count triple."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    count_triple = next(
        t for t in triples if t.predicate == "person:threadMessageCount"
    )
    assert count_triple.object == "2"


def test_participant_temporal_triples(reconstructor):
    """Test participant temporal engagement triples."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # First message date
    first_triple = next(
        t for t in triples if t.predicate == "person:firstThreadMessageDate"
    )
    assert first_triple.object  # Should have ISO date

    # Last message date
    last_triple = next(
        t for t in triples if t.predicate == "person:lastThreadMessageDate"
    )
    assert last_triple.object


def test_multiple_participants(reconstructor):
    """Test triples for thread with multiple participants."""
    now = datetime.utcnow()
    participants = [
        ThreadParticipant(
            email_address="alice@example.com",
            role=ParticipantRole.INITIATOR,
            message_count=1,
            first_message_date=now,
            last_message_date=now,
        ),
        ThreadParticipant(
            email_address="bob@example.com",
            role=ParticipantRole.PARTICIPANT,
            message_count=1,
            first_message_date=now + timedelta(minutes=10),
            last_message_date=now + timedelta(minutes=10),
        ),
    ]

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        message_ids=["msg1@example.com", "msg2@example.com"],
        message_count=2,
        participants=participants,
        participant_count=2,
        subject="Multi-participant",
        start_date=now,
        last_message_date=now + timedelta(minutes=10),
        mailbox_id="test_mailbox",
        reconstructed_at=now,
    )

    triples = extract_thread_triples(thread, reconstructor)

    # Should have participated triples for both
    participated_triples = [
        t for t in triples if t.predicate == "person:participatedIn"
    ]
    assert len(participated_triples) == 2


# ============================================================================
# Conversation Flow Tests
# ============================================================================


def test_conversation_flow_triples(reconstructor):
    """Test parent-child conversation flow triples."""
    # Build actual thread with reconstructor
    msg1 = create_email_message("msg1@example.com", date=datetime.utcnow())
    msg2 = create_email_message(
        "msg2@example.com",
        in_reply_to="msg1@example.com",
        date=datetime.utcnow() + timedelta(minutes=10),
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)

    threads = reconstructor.reconstruct_threads()
    thread = threads[0]

    triples = extract_thread_triples(thread, reconstructor)

    # Find inReplyTo triple
    reply_triples = [t for t in triples if t.predicate == "email:inReplyTo"]
    assert len(reply_triples) == 1

    reply_triple = reply_triples[0]
    assert "msg2@example.com" in reply_triple.subject
    assert "msg1@example.com" in reply_triple.object


def test_thread_depth_on_messages(reconstructor):
    """Test thread depth metadata on individual messages."""
    # Build 3-level thread
    base_time = datetime.utcnow()
    msg1 = create_email_message("msg1@example.com", date=base_time)
    msg2 = create_email_message(
        "msg2@example.com",
        in_reply_to="msg1@example.com",
        date=base_time + timedelta(minutes=10),
    )
    msg3 = create_email_message(
        "msg3@example.com",
        in_reply_to="msg2@example.com",
        date=base_time + timedelta(minutes=20),
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)
    reconstructor.add_message(msg3)

    threads = reconstructor.reconstruct_threads()
    thread = threads[0]

    triples = extract_thread_triples(thread, reconstructor)

    # Find threadDepth triples
    depth_triples = [t for t in triples if t.predicate == "email:threadDepth"]

    # Should have depth for messages with parents (msg2, msg3)
    assert len(depth_triples) == 2


# ============================================================================
# URI Creation Tests
# ============================================================================


def test_thread_uri_creation(reconstructor):
    """Test thread URI creation and formatting."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # Check URI format
    thread_uri = triples[0].subject
    assert thread_uri.startswith("futurnal:thread/")
    assert "thread1@example.com" in thread_uri or "thread1" in thread_uri


def test_email_uri_creation(reconstructor):
    """Test email URI creation in triples."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # Find message relationship triple
    msg_triple = next(t for t in triples if t.predicate == "email:partOfThread")

    # Check email URI format
    assert msg_triple.subject.startswith("futurnal:email/")


def test_person_uri_creation(reconstructor):
    """Test person URI creation in triples."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # Find person relationship triple
    person_triple = next(t for t in triples if t.predicate == "person:participatedIn")

    # Check person URI format
    assert person_triple.subject.startswith("futurnal:person/")
    assert "alice@example.com" in person_triple.subject


# ============================================================================
# Extraction Method Tests
# ============================================================================


def test_extraction_method_metadata(reconstructor):
    """Test all triples have correct extraction_method."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # All should have thread_reconstruction method
    for triple in triples:
        assert triple.extraction_method == "thread_reconstruction"


def test_source_element_id_metadata(reconstructor):
    """Test all triples have source_element_id."""
    thread = create_simple_thread()
    triples = extract_thread_triples(thread, reconstructor)

    # All should have thread_id as source
    for triple in triples:
        assert triple.source_element_id == thread.thread_id


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_thread_reconstruction_to_triples(reconstructor):
    """Test complete flow from messages to thread to triples."""
    base_time = datetime(2024, 1, 1, 10, 0)

    # Create 3-message thread
    msg1 = create_email_message(
        "msg1@example.com",
        subject="Discussion",
        from_addr="alice@example.com",
        date=base_time,
    )
    msg2 = create_email_message(
        "msg2@example.com",
        subject="Re: Discussion",
        from_addr="bob@example.com",
        in_reply_to="msg1@example.com",
        date=base_time + timedelta(minutes=30),
    )
    msg3 = create_email_message(
        "msg3@example.com",
        subject="Re: Discussion",
        from_addr="alice@example.com",
        in_reply_to="msg2@example.com",
        date=base_time + timedelta(minutes=45),
    )

    # Add to reconstructor
    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)
    reconstructor.add_message(msg3)

    # Reconstruct threads
    threads = reconstructor.reconstruct_threads()
    assert len(threads) == 1
    thread = threads[0]

    # Calculate response times
    messages = {
        msg1.message_id: msg1,
        msg2.message_id: msg2,
        msg3.message_id: msg3,
    }
    reconstructor.calculate_response_times(thread, messages)

    # Extract triples
    triples = extract_thread_triples(thread, reconstructor)

    # Verify we have comprehensive triples
    assert len(triples) > 20  # Should have many triples

    # Verify key triple types exist
    predicates = {t.predicate for t in triples}
    assert "rdf:type" in predicates
    assert "thread:subject" in predicates
    assert "thread:messageCount" in predicates
    assert "person:participatedIn" in predicates
    assert "email:partOfThread" in predicates
    assert "email:inReplyTo" in predicates


def test_empty_thread_triples(reconstructor):
    """Test triple extraction from empty thread."""
    now = datetime.utcnow()
    thread = EmailThread(
        thread_id="empty@example.com",
        root_message_id="empty@example.com",
        subject="Empty",
        start_date=now,
        last_message_date=now,
        mailbox_id="test_mailbox",
        reconstructed_at=now,
    )

    triples = extract_thread_triples(thread, reconstructor)

    # Should still have basic thread metadata triples
    assert len(triples) > 0

    # Should have type triple
    type_triples = [t for t in triples if t.predicate == "rdf:type"]
    assert len(type_triples) == 1
