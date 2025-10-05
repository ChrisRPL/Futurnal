"""Comprehensive tests for email thread data models.

Tests cover:
- ParticipantRole enum validation
- ThreadParticipant model validation and computed properties
- ThreadNode model validation and tree properties
- EmailThread model validation and analytics properties
- Field constraints and defaults
- Edge cases and boundary conditions
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from futurnal.ingestion.imap.thread_models import (
    EmailThread,
    ParticipantRole,
    ThreadNode,
    ThreadParticipant,
)


# ============================================================================
# ParticipantRole Tests
# ============================================================================


def test_participant_role_values():
    """Test ParticipantRole enum has expected values."""
    assert ParticipantRole.INITIATOR == "initiator"
    assert ParticipantRole.PRIMARY_RECIPIENT == "primary_recipient"
    assert ParticipantRole.PARTICIPANT == "participant"
    assert ParticipantRole.CC_RECIPIENT == "cc_recipient"
    assert ParticipantRole.OBSERVER == "observer"


def test_participant_role_from_string():
    """Test creating ParticipantRole from string."""
    role = ParticipantRole("initiator")
    assert role == ParticipantRole.INITIATOR


# ============================================================================
# ThreadParticipant Tests
# ============================================================================


def test_thread_participant_basic():
    """Test basic ThreadParticipant creation."""
    now = datetime.utcnow()
    participant = ThreadParticipant(
        email_address="john@example.com",
        display_name="John Doe",
        role=ParticipantRole.INITIATOR,
        message_count=3,
        first_message_date=now,
        last_message_date=now + timedelta(hours=2),
    )

    assert participant.email_address == "john@example.com"
    assert participant.display_name == "John Doe"
    assert participant.role == ParticipantRole.INITIATOR
    assert participant.message_count == 3
    assert participant.first_message_date == now
    assert participant.last_message_date == now + timedelta(hours=2)


def test_thread_participant_no_display_name():
    """Test ThreadParticipant without display name."""
    now = datetime.utcnow()
    participant = ThreadParticipant(
        email_address="john@example.com",
        role=ParticipantRole.PARTICIPANT,
        message_count=1,
        first_message_date=now,
        last_message_date=now,
    )

    assert participant.display_name is None


def test_thread_participant_participation_duration():
    """Test participation_duration_days property."""
    start = datetime(2024, 1, 1, 10, 0)
    end = datetime(2024, 1, 3, 10, 0)  # 2 days later

    participant = ThreadParticipant(
        email_address="john@example.com",
        role=ParticipantRole.PARTICIPANT,
        message_count=5,
        first_message_date=start,
        last_message_date=end,
    )

    assert participant.participation_duration_days == 2.0


def test_thread_participant_zero_duration():
    """Test participation duration when all messages same time."""
    now = datetime.utcnow()
    participant = ThreadParticipant(
        email_address="john@example.com",
        role=ParticipantRole.INITIATOR,
        message_count=1,
        first_message_date=now,
        last_message_date=now,
    )

    assert participant.participation_duration_days == 0.0


def test_thread_participant_message_count_validation():
    """Test message_count cannot be negative."""
    now = datetime.utcnow()

    with pytest.raises(ValueError):
        ThreadParticipant(
            email_address="john@example.com",
            role=ParticipantRole.PARTICIPANT,
            message_count=-1,  # Invalid
            first_message_date=now,
            last_message_date=now,
        )


# ============================================================================
# ThreadNode Tests
# ============================================================================


def test_thread_node_root():
    """Test root ThreadNode (no parent)."""
    now = datetime.utcnow()
    node = ThreadNode(
        message_id="msg1@example.com",
        parent_message_id=None,
        date=now,
        from_address="alice@example.com",
        subject="Thread Subject",
    )

    assert node.message_id == "msg1@example.com"
    assert node.parent_message_id is None
    assert node.is_root is True
    assert node.is_leaf is True  # No children yet
    assert node.depth == 0
    assert len(node.children) == 0


def test_thread_node_with_parent():
    """Test ThreadNode with parent."""
    now = datetime.utcnow()
    node = ThreadNode(
        message_id="msg2@example.com",
        parent_message_id="msg1@example.com",
        date=now,
        from_address="bob@example.com",
        subject="Re: Thread Subject",
    )

    assert node.parent_message_id == "msg1@example.com"
    assert node.is_root is False


def test_thread_node_with_children():
    """Test ThreadNode with children."""
    now = datetime.utcnow()
    node = ThreadNode(
        message_id="msg1@example.com",
        parent_message_id=None,
        children=["msg2@example.com", "msg3@example.com"],
        date=now,
        from_address="alice@example.com",
        subject="Thread Subject",
    )

    assert node.is_leaf is False
    assert node.child_count == 2
    assert "msg2@example.com" in node.children


def test_thread_node_depth_validation():
    """Test depth cannot be negative."""
    now = datetime.utcnow()

    with pytest.raises(ValueError):
        ThreadNode(
            message_id="msg1@example.com",
            depth=-1,  # Invalid
            date=now,
            from_address="alice@example.com",
            subject="Subject",
        )


# ============================================================================
# EmailThread Tests
# ============================================================================


def test_email_thread_basic():
    """Test basic EmailThread creation."""
    start = datetime(2024, 1, 1, 10, 0)
    end = datetime(2024, 1, 2, 10, 0)

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        message_ids=["msg1@example.com", "msg2@example.com"],
        message_count=2,
        participants=[],
        participant_count=2,
        subject="Project Discussion",
        start_date=start,
        last_message_date=end,
        duration_days=1.0,
        depth=1,
        mailbox_id="mailbox1",
        reconstructed_at=datetime.utcnow(),
    )

    assert thread.thread_id == "thread1@example.com"
    assert thread.root_message_id == "msg1@example.com"
    assert thread.message_count == 2
    assert thread.subject == "Project Discussion"
    assert thread.duration_days == 1.0


def test_email_thread_single_participant():
    """Test thread with single participant."""
    now = datetime.utcnow()
    participant = ThreadParticipant(
        email_address="alice@example.com",
        role=ParticipantRole.INITIATOR,
        message_count=1,
        first_message_date=now,
        last_message_date=now,
    )

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        participants=[participant],
        participant_count=1,
        subject="Solo Thread",
        start_date=now,
        last_message_date=now,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    assert thread.is_multi_participant is False


def test_email_thread_multi_participant():
    """Test thread with multiple participants."""
    now = datetime.utcnow()
    participants = [
        ThreadParticipant(
            email_address="alice@example.com",
            role=ParticipantRole.INITIATOR,
            message_count=2,
            first_message_date=now,
            last_message_date=now,
        ),
        ThreadParticipant(
            email_address="bob@example.com",
            role=ParticipantRole.PARTICIPANT,
            message_count=1,
            first_message_date=now,
            last_message_date=now,
        ),
    ]

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        participants=participants,
        participant_count=2,
        subject="Team Discussion",
        start_date=now,
        last_message_date=now,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    assert thread.is_multi_participant is True


def test_email_thread_is_long_thread():
    """Test is_long_thread property (>10 messages)."""
    now = datetime.utcnow()
    message_ids = [f"msg{i}@example.com" for i in range(15)]

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg0@example.com",
        message_ids=message_ids,
        message_count=15,
        subject="Long Discussion",
        start_date=now,
        last_message_date=now + timedelta(days=5),
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    assert thread.is_long_thread is True


def test_email_thread_is_deep_thread():
    """Test is_deep_thread property (>5 levels)."""
    now = datetime.utcnow()

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        depth=7,  # Deep thread
        subject="Deep Discussion",
        start_date=now,
        last_message_date=now,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    assert thread.is_deep_thread is True


def test_email_thread_is_branching():
    """Test is_branching_thread property."""
    now = datetime.utcnow()

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        branch_count=3,  # Has branches
        subject="Branching Discussion",
        start_date=now,
        last_message_date=now,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    assert thread.is_branching_thread is True


def test_email_thread_messages_per_day():
    """Test messages_per_day calculation."""
    start = datetime(2024, 1, 1, 10, 0)
    end = datetime(2024, 1, 5, 10, 0)  # 4 days

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        message_count=8,  # 8 messages over 4 days
        subject="Discussion",
        start_date=start,
        last_message_date=end,
        duration_days=4.0,
        mailbox_id="mailbox1",
        reconstructed_at=datetime.utcnow(),
    )

    assert thread.messages_per_day == 2.0


def test_email_thread_messages_per_day_zero_duration():
    """Test messages_per_day when duration is zero."""
    now = datetime.utcnow()

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        message_count=5,
        subject="Instant Thread",
        start_date=now,
        last_message_date=now,
        duration_days=0.0,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    # Should return message count when duration is 0
    assert thread.messages_per_day == 5.0


def test_email_thread_response_times():
    """Test response time fields."""
    now = datetime.utcnow()

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        subject="Discussion",
        start_date=now,
        last_message_date=now,
        total_response_time_minutes=120.0,
        average_response_time_minutes=30.0,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    assert thread.total_response_time_minutes == 120.0
    assert thread.average_response_time_minutes == 30.0


def test_email_thread_has_attachments():
    """Test has_attachments flag."""
    now = datetime.utcnow()

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        subject="Discussion",
        start_date=now,
        last_message_date=now,
        has_attachments=True,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    assert thread.has_attachments is True


def test_email_thread_subject_variations():
    """Test subject_variations field."""
    now = datetime.utcnow()

    variations = [
        "Project Discussion",
        "Re: Project Discussion",
        "Re: Project Discussion - Updated",
    ]

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        subject="Project Discussion",
        subject_variations=variations,
        start_date=now,
        last_message_date=now,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    assert len(thread.subject_variations) == 3
    assert "Re: Project Discussion - Updated" in thread.subject_variations


def test_email_thread_defaults():
    """Test EmailThread with default values."""
    now = datetime.utcnow()

    thread = EmailThread(
        thread_id="thread1@example.com",
        root_message_id="msg1@example.com",
        subject="Test",
        start_date=now,
        last_message_date=now,
        mailbox_id="mailbox1",
        reconstructed_at=now,
    )

    # Check defaults
    assert thread.message_count == 0
    assert thread.participant_count == 0
    assert thread.duration_days == 0.0
    assert thread.depth == 0
    assert thread.branch_count == 0
    assert thread.total_response_time_minutes == 0.0
    assert thread.average_response_time_minutes == 0.0
    assert thread.has_attachments is False
    assert len(thread.message_ids) == 0
    assert len(thread.participants) == 0
    assert len(thread.subject_variations) == 0


def test_email_thread_negative_counts_validation():
    """Test that negative counts are rejected."""
    now = datetime.utcnow()

    with pytest.raises(ValueError):
        EmailThread(
            thread_id="thread1@example.com",
            root_message_id="msg1@example.com",
            message_count=-1,  # Invalid
            subject="Test",
            start_date=now,
            last_message_date=now,
            mailbox_id="mailbox1",
            reconstructed_at=now,
        )


def test_email_thread_negative_duration_validation():
    """Test that negative duration is rejected."""
    now = datetime.utcnow()

    with pytest.raises(ValueError):
        EmailThread(
            thread_id="thread1@example.com",
            root_message_id="msg1@example.com",
            subject="Test",
            start_date=now,
            last_message_date=now,
            duration_days=-1.0,  # Invalid
            mailbox_id="mailbox1",
            reconstructed_at=now,
        )
