"""Comprehensive tests for email thread reconstruction.

Tests cover:
- Simple thread reconstruction (2-3 messages)
- Complex branching threads
- Deep threads (>10 levels)
- Out-of-order message handling
- Missing parent handling (orphans)
- References fallback when In-Reply-To missing
- Subject normalization edge cases
- Response time calculations
- Duplicate message handling
- Participant role identification
- Thread statistics calculation
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from futurnal.ingestion.imap.email_parser import EmailAddress, EmailMessage
from futurnal.ingestion.imap.thread_models import ParticipantRole
from futurnal.ingestion.imap.thread_reconstructor import ThreadReconstructor


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
    references: list[str] | None = None,
    uid: int = 1,
) -> EmailMessage:
    """Helper to create test EmailMessage."""
    if date is None:
        date = datetime.utcnow()

    return EmailMessage(
        message_id=message_id,
        uid=uid,
        folder="INBOX",
        subject=subject,
        from_address=EmailAddress(address=from_addr),
        date=date,
        in_reply_to=in_reply_to,
        references=references or [],
        size_bytes=1000,
        retrieved_at=datetime.utcnow(),
        mailbox_id="test_mailbox",
    )


# ============================================================================
# Basic Thread Reconstruction Tests
# ============================================================================


def test_add_single_message(reconstructor):
    """Test adding a single message to graph."""
    msg = create_email_message("msg1@example.com")
    reconstructor.add_message(msg)

    assert len(reconstructor.message_graph) == 1
    assert "msg1@example.com" in reconstructor.message_graph

    node = reconstructor.message_graph["msg1@example.com"]
    assert node.message_id == "msg1@example.com"
    assert node.parent_message_id is None
    assert node.is_root is True


def test_add_two_message_thread(reconstructor):
    """Test simple two-message thread (parent and reply)."""
    msg1 = create_email_message("msg1@example.com", subject="Original")
    msg2 = create_email_message(
        "msg2@example.com",
        subject="Re: Original",
        in_reply_to="msg1@example.com",
        date=datetime.utcnow() + timedelta(minutes=10),
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)

    assert len(reconstructor.message_graph) == 2

    # Check parent node
    parent = reconstructor.message_graph["msg1@example.com"]
    assert parent.is_root is True
    assert "msg2@example.com" in parent.children

    # Check child node
    child = reconstructor.message_graph["msg2@example.com"]
    assert child.parent_message_id == "msg1@example.com"
    assert child.is_root is False


def test_three_message_linear_thread(reconstructor):
    """Test linear three-message thread (A -> B -> C)."""
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

    # Check structure
    assert len(reconstructor.message_graph) == 3
    assert "msg2@example.com" in reconstructor.message_graph["msg1@example.com"].children
    assert "msg3@example.com" in reconstructor.message_graph["msg2@example.com"].children


def test_reconstruct_single_thread(reconstructor):
    """Test reconstructing threads from graph."""
    msg1 = create_email_message("msg1@example.com", subject="Discussion")
    msg2 = create_email_message(
        "msg2@example.com", subject="Re: Discussion", in_reply_to="msg1@example.com"
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)

    threads = reconstructor.reconstruct_threads()

    assert len(threads) == 1
    thread = threads[0]
    assert thread.thread_id == "msg1@example.com"
    assert thread.root_message_id == "msg1@example.com"
    assert thread.message_count == 2
    assert set(thread.message_ids) == {"msg1@example.com", "msg2@example.com"}


# ============================================================================
# Branching Thread Tests
# ============================================================================


def test_branching_thread(reconstructor):
    """Test thread with branching (A -> B, A -> C)."""
    base_time = datetime.utcnow()

    msg1 = create_email_message("msg1@example.com", date=base_time)
    msg2 = create_email_message(
        "msg2@example.com",
        in_reply_to="msg1@example.com",
        date=base_time + timedelta(minutes=10),
    )
    msg3 = create_email_message(
        "msg3@example.com",
        in_reply_to="msg1@example.com",
        date=base_time + timedelta(minutes=15),
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)
    reconstructor.add_message(msg3)

    # Parent should have two children
    parent = reconstructor.message_graph["msg1@example.com"]
    assert len(parent.children) == 2
    assert "msg2@example.com" in parent.children
    assert "msg3@example.com" in parent.children

    threads = reconstructor.reconstruct_threads()
    assert len(threads) == 1
    thread = threads[0]
    assert thread.branch_count >= 1  # At least one branch


def test_complex_branching_thread(reconstructor):
    """Test complex branching structure (tree with multiple branches)."""
    base_time = datetime.utcnow()

    # Create tree: msg1 -> [msg2 -> msg4, msg3 -> msg5]
    msg1 = create_email_message("msg1@example.com", date=base_time)
    msg2 = create_email_message(
        "msg2@example.com", in_reply_to="msg1@example.com", date=base_time + timedelta(minutes=10)
    )
    msg3 = create_email_message(
        "msg3@example.com", in_reply_to="msg1@example.com", date=base_time + timedelta(minutes=12)
    )
    msg4 = create_email_message(
        "msg4@example.com", in_reply_to="msg2@example.com", date=base_time + timedelta(minutes=20)
    )
    msg5 = create_email_message(
        "msg5@example.com", in_reply_to="msg3@example.com", date=base_time + timedelta(minutes=22)
    )

    for msg in [msg1, msg2, msg3, msg4, msg5]:
        reconstructor.add_message(msg)

    threads = reconstructor.reconstruct_threads()
    assert len(threads) == 1
    thread = threads[0]
    assert thread.message_count == 5
    assert thread.branch_count >= 1


# ============================================================================
# Deep Thread Tests
# ============================================================================


def test_deep_thread(reconstructor):
    """Test deep thread (10+ levels)."""
    base_time = datetime.utcnow()
    messages = []

    # Create 12-level deep thread
    for i in range(12):
        parent_id = f"msg{i}@example.com" if i > 0 else None
        msg = create_email_message(
            f"msg{i+1}@example.com",
            in_reply_to=parent_id,
            date=base_time + timedelta(minutes=i * 10),
        )
        messages.append(msg)
        reconstructor.add_message(msg)

    threads = reconstructor.reconstruct_threads()
    assert len(threads) == 1
    thread = threads[0]
    assert thread.message_count == 12
    assert thread.depth >= 11
    assert thread.is_deep_thread is True


# ============================================================================
# Out-of-Order Message Tests
# ============================================================================


def test_out_of_order_messages(reconstructor):
    """Test adding messages out of chronological order."""
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

    # Add in reverse order
    reconstructor.add_message(msg3)
    reconstructor.add_message(msg2)
    reconstructor.add_message(msg1)

    threads = reconstructor.reconstruct_threads()
    assert len(threads) == 1
    thread = threads[0]
    assert thread.message_count == 3
    assert thread.root_message_id == "msg1@example.com"


def test_child_arrives_before_parent(reconstructor):
    """Test child message arriving before parent."""
    msg2 = create_email_message("msg2@example.com", in_reply_to="msg1@example.com")
    msg1 = create_email_message("msg1@example.com")

    # Add child first
    reconstructor.add_message(msg2)
    assert len(reconstructor.message_graph) == 1

    # Add parent later - should link to orphan
    reconstructor.add_message(msg1)

    parent = reconstructor.message_graph["msg1@example.com"]
    assert "msg2@example.com" in parent.children


# ============================================================================
# References Header Tests
# ============================================================================


def test_link_via_references_header(reconstructor):
    """Test linking via References header when In-Reply-To missing."""
    msg1 = create_email_message("msg1@example.com")
    msg2 = create_email_message(
        "msg2@example.com",
        in_reply_to=None,  # No In-Reply-To
        references=["msg1@example.com"],  # But has References
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)

    child = reconstructor.message_graph["msg2@example.com"]
    assert child.parent_message_id == "msg1@example.com"

    parent = reconstructor.message_graph["msg1@example.com"]
    assert "msg2@example.com" in parent.children


def test_references_with_multiple_ids(reconstructor):
    """Test References with multiple Message-IDs (chooses last)."""
    msg1 = create_email_message("msg1@example.com")
    msg2 = create_email_message(
        "msg2@example.com", in_reply_to="msg1@example.com"
    )
    msg3 = create_email_message(
        "msg3@example.com",
        in_reply_to=None,
        references=["msg1@example.com", "msg2@example.com"],  # Last is immediate parent
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)
    reconstructor.add_message(msg3)

    child = reconstructor.message_graph["msg3@example.com"]
    assert child.parent_message_id == "msg2@example.com"


# ============================================================================
# Orphan Message Tests
# ============================================================================


def test_orphan_message_no_parent(reconstructor):
    """Test message with In-Reply-To but parent never arrives."""
    msg = create_email_message("msg2@example.com", in_reply_to="msg1@example.com")
    reconstructor.add_message(msg)

    # Message should be added but parent link not established
    assert "msg2@example.com" in reconstructor.message_graph
    node = reconstructor.message_graph["msg2@example.com"]
    assert node.parent_message_id == "msg1@example.com"  # Recorded but not linked


def test_handle_orphan_with_placeholder(reconstructor):
    """Test creating placeholder parent for orphan."""
    msg = create_email_message("msg2@example.com", in_reply_to="msg1@example.com")
    reconstructor.add_message(msg)

    # Create placeholder for orphan
    reconstructor.handle_orphan_message(msg)

    assert "msg1@example.com" in reconstructor.message_graph
    placeholder = reconstructor.message_graph["msg1@example.com"]
    assert placeholder.from_address == "unknown@unknown"
    assert placeholder.is_root is True


# ============================================================================
# Subject Normalization Tests
# ============================================================================


def test_subject_normalization_re(reconstructor):
    """Test normalizing Re: prefix."""
    assert reconstructor._normalize_subject("Re: Subject") == "Subject"
    assert reconstructor._normalize_subject("RE: Subject") == "Subject"
    assert reconstructor._normalize_subject("re: Subject") == "Subject"


def test_subject_normalization_fwd(reconstructor):
    """Test normalizing Fwd: prefix."""
    assert reconstructor._normalize_subject("Fwd: Subject") == "Subject"
    assert reconstructor._normalize_subject("FWD: Subject") == "Subject"
    assert reconstructor._normalize_subject("fwd: Subject") == "Subject"


def test_subject_normalization_multiple_prefixes(reconstructor):
    """Test normalizing multiple Re:/Fwd: prefixes."""
    assert reconstructor._normalize_subject("Re: Re: Subject") == "Subject"
    assert reconstructor._normalize_subject("Re: Fwd: Re: Subject") == "Subject"


def test_subject_normalization_gmail_style(reconstructor):
    """Test normalizing Gmail-style Re[2]: prefix."""
    assert reconstructor._normalize_subject("Re[2]: Subject") == "Subject"
    assert reconstructor._normalize_subject("Re[10]: Subject") == "Subject"


def test_subject_normalization_empty(reconstructor):
    """Test normalizing empty subject."""
    assert reconstructor._normalize_subject("") == ""
    assert reconstructor._normalize_subject(None) == ""


# ============================================================================
# Response Time Tests
# ============================================================================


def test_calculate_response_times(reconstructor):
    """Test response time calculation."""
    base_time = datetime(2024, 1, 1, 10, 0)

    msg1 = create_email_message("msg1@example.com", date=base_time)
    msg2 = create_email_message(
        "msg2@example.com",
        in_reply_to="msg1@example.com",
        date=base_time + timedelta(minutes=30),
    )
    msg3 = create_email_message(
        "msg3@example.com",
        in_reply_to="msg2@example.com",
        date=base_time + timedelta(minutes=45),
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)
    reconstructor.add_message(msg3)

    threads = reconstructor.reconstruct_threads()
    thread = threads[0]

    messages = {
        msg1.message_id: msg1,
        msg2.message_id: msg2,
        msg3.message_id: msg3,
    }

    reconstructor.calculate_response_times(thread, messages)

    # 30 min + 15 min = 45 min total, 22.5 min average
    assert thread.total_response_time_minutes == 45.0
    assert thread.average_response_time_minutes == 22.5


def test_response_times_no_replies(reconstructor):
    """Test response time calculation for single message."""
    msg = create_email_message("msg1@example.com")
    reconstructor.add_message(msg)

    threads = reconstructor.reconstruct_threads()
    thread = threads[0]

    reconstructor.calculate_response_times(thread, {msg.message_id: msg})

    assert thread.total_response_time_minutes == 0.0
    assert thread.average_response_time_minutes == 0.0


# ============================================================================
# Participant Tests
# ============================================================================


def test_participant_role_initiator(reconstructor):
    """Test initiator role assignment."""
    msg = create_email_message("msg1@example.com", from_addr="alice@example.com")
    reconstructor.add_message(msg)

    threads = reconstructor.reconstruct_threads()
    thread = threads[0]

    assert len(thread.participants) == 1
    participant = thread.participants[0]
    assert participant.email_address == "alice@example.com"
    assert participant.role == ParticipantRole.INITIATOR


def test_participant_role_participant(reconstructor):
    """Test participant role assignment for replier."""
    msg1 = create_email_message("msg1@example.com", from_addr="alice@example.com")
    msg2 = create_email_message(
        "msg2@example.com",
        from_addr="bob@example.com",
        in_reply_to="msg1@example.com",
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)

    threads = reconstructor.reconstruct_threads()
    thread = threads[0]

    assert len(thread.participants) == 2

    # Find Bob's participant record
    bob = next(p for p in thread.participants if p.email_address == "bob@example.com")
    assert bob.role == ParticipantRole.PARTICIPANT


def test_participant_message_counts(reconstructor):
    """Test participant message count tracking."""
    msg1 = create_email_message("msg1@example.com", from_addr="alice@example.com")
    msg2 = create_email_message(
        "msg2@example.com",
        from_addr="alice@example.com",
        in_reply_to="msg1@example.com",
    )
    msg3 = create_email_message(
        "msg3@example.com",
        from_addr="bob@example.com",
        in_reply_to="msg1@example.com",
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)
    reconstructor.add_message(msg3)

    threads = reconstructor.reconstruct_threads()
    thread = threads[0]

    alice = next(p for p in thread.participants if p.email_address == "alice@example.com")
    assert alice.message_count == 2

    bob = next(p for p in thread.participants if p.email_address == "bob@example.com")
    assert bob.message_count == 1


# ============================================================================
# Thread Statistics Tests
# ============================================================================


def test_get_thread_statistics(reconstructor):
    """Test overall threading statistics."""
    base_time = datetime.utcnow()

    # Create two threads
    # Thread 1: 3 messages
    msg1 = create_email_message("msg1@example.com", date=base_time)
    msg2 = create_email_message(
        "msg2@example.com", in_reply_to="msg1@example.com", date=base_time + timedelta(hours=1)
    )
    msg3 = create_email_message(
        "msg3@example.com", in_reply_to="msg2@example.com", date=base_time + timedelta(hours=2)
    )

    # Thread 2: 2 messages
    msg4 = create_email_message("msg4@example.com", date=base_time)
    msg5 = create_email_message(
        "msg5@example.com", in_reply_to="msg4@example.com", date=base_time + timedelta(hours=1)
    )

    for msg in [msg1, msg2, msg3, msg4, msg5]:
        reconstructor.add_message(msg)

    threads = reconstructor.reconstruct_threads()
    stats = reconstructor.get_thread_statistics()

    assert stats["total_threads"] == 2
    assert stats["total_messages"] == 5
    assert stats["avg_messages_per_thread"] == 2.5


def test_get_thread_for_message(reconstructor):
    """Test finding thread for specific message."""
    msg1 = create_email_message("msg1@example.com")
    msg2 = create_email_message("msg2@example.com", in_reply_to="msg1@example.com")

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)
    reconstructor.reconstruct_threads()

    thread = reconstructor.get_thread_for_message("msg2@example.com")
    assert thread is not None
    assert thread.root_message_id == "msg1@example.com"
    assert "msg2@example.com" in thread.message_ids


def test_get_thread_for_nonexistent_message(reconstructor):
    """Test finding thread for non-existent message."""
    msg = create_email_message("msg1@example.com")
    reconstructor.add_message(msg)
    reconstructor.reconstruct_threads()

    thread = reconstructor.get_thread_for_message("nonexistent@example.com")
    assert thread is None


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_reconstructor(reconstructor):
    """Test reconstructing with no messages."""
    threads = reconstructor.reconstruct_threads()
    assert len(threads) == 0

    stats = reconstructor.get_thread_statistics()
    assert stats["total_threads"] == 0
    assert stats["total_messages"] == 0


def test_multiple_root_threads(reconstructor):
    """Test multiple independent threads."""
    msg1 = create_email_message("msg1@example.com", subject="Thread 1")
    msg2 = create_email_message("msg2@example.com", subject="Thread 2")
    msg3 = create_email_message("msg3@example.com", subject="Thread 3")

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)
    reconstructor.add_message(msg3)

    threads = reconstructor.reconstruct_threads()
    assert len(threads) == 3


def test_duplicate_message_handling(reconstructor):
    """Test adding same message twice."""
    msg1 = create_email_message("msg1@example.com")

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg1)  # Add again

    # Should only be in graph once
    assert len(reconstructor.message_graph) == 1


def test_thread_duration_calculation(reconstructor):
    """Test thread duration calculation."""
    start = datetime(2024, 1, 1, 10, 0)
    end = datetime(2024, 1, 5, 10, 0)  # 4 days later

    msg1 = create_email_message("msg1@example.com", date=start)
    msg2 = create_email_message(
        "msg2@example.com", in_reply_to="msg1@example.com", date=end
    )

    reconstructor.add_message(msg1)
    reconstructor.add_message(msg2)

    threads = reconstructor.reconstruct_threads()
    thread = threads[0]

    assert thread.duration_days == 4.0
