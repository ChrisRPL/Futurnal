"""Comprehensive tests for subject evolution tracking.

Tests cover:
- Subject variation detection
- Subject normalization (Re:/Fwd: removal, case, whitespace)
- Subject change point identification
- Change type classification
- Evolution summary statistics
- Chronological message sorting
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from futurnal.ingestion.imap.email_parser import EmailAddress, EmailMessage
from futurnal.ingestion.imap.subject_evolution import SubjectEvolutionTracker
from futurnal.ingestion.imap.thread_models import EmailThread


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tracker():
    """Create a fresh SubjectEvolutionTracker instance."""
    return SubjectEvolutionTracker()


def create_email_message(
    message_id: str,
    subject: str,
    date: datetime | None = None,
) -> EmailMessage:
    """Helper to create test EmailMessage."""
    if date is None:
        date = datetime.utcnow()

    return EmailMessage(
        message_id=message_id,
        uid=1,
        folder="INBOX",
        subject=subject,
        from_address=EmailAddress(address="sender@example.com"),
        date=date,
        size_bytes=1000,
        retrieved_at=datetime.utcnow(),
        mailbox_id="test_mailbox",
    )


def create_thread(message_ids: list[str], thread_id: str = "thread1") -> EmailThread:
    """Helper to create test EmailThread."""
    return EmailThread(
        thread_id=thread_id,
        root_message_id=message_ids[0] if message_ids else "root",
        message_ids=message_ids,
        message_count=len(message_ids),
        subject="Test Thread",
        start_date=datetime.utcnow(),
        last_message_date=datetime.utcnow(),
        mailbox_id="test_mailbox",
        reconstructed_at=datetime.utcnow(),
    )


# ============================================================================
# Subject Normalization Tests
# ============================================================================


def test_normalize_subject_re_prefix(tracker):
    """Test normalizing Re: prefix."""
    assert tracker._normalize_subject("Re: Subject") == "subject"
    assert tracker._normalize_subject("RE: Subject") == "subject"
    assert tracker._normalize_subject("re: Subject") == "subject"


def test_normalize_subject_fwd_prefix(tracker):
    """Test normalizing Fwd: prefix."""
    assert tracker._normalize_subject("Fwd: Subject") == "subject"
    assert tracker._normalize_subject("FWD: Subject") == "subject"
    assert tracker._normalize_subject("fwd: Subject") == "subject"


def test_normalize_subject_multiple_prefixes(tracker):
    """Test normalizing multiple Re:/Fwd: prefixes."""
    assert tracker._normalize_subject("Re: Re: Subject") == "subject"
    assert tracker._normalize_subject("Re: Fwd: Subject") == "subject"
    assert tracker._normalize_subject("Fwd: Re: Re: Subject") == "subject"


def test_normalize_subject_gmail_style(tracker):
    """Test normalizing Gmail-style Re[N]: prefix."""
    assert tracker._normalize_subject("Re[2]: Subject") == "subject"
    assert tracker._normalize_subject("Re[10]: Subject") == "subject"


def test_normalize_subject_lowercase(tracker):
    """Test subject is converted to lowercase."""
    assert tracker._normalize_subject("UPPERCASE SUBJECT") == "uppercase subject"
    assert tracker._normalize_subject("MiXeD CaSe") == "mixed case"


def test_normalize_subject_whitespace(tracker):
    """Test whitespace normalization."""
    assert tracker._normalize_subject("Subject   with   spaces") == "subject with spaces"
    assert tracker._normalize_subject("  Subject  ") == "subject"


def test_normalize_subject_empty(tracker):
    """Test normalizing empty subject."""
    assert tracker._normalize_subject("") == ""
    assert tracker._normalize_subject("   ") == ""


# ============================================================================
# Subject Variation Tests
# ============================================================================


def test_analyze_subject_evolution_no_variations(tracker):
    """Test thread with single subject."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg1": create_email_message("msg1", "Discussion", date=base_time),
        "msg2": create_email_message("msg2", "Re: Discussion", date=base_time + timedelta(minutes=10)),
        "msg3": create_email_message("msg3", "Re: Discussion", date=base_time + timedelta(minutes=20)),
    }

    thread = create_thread(["msg1", "msg2", "msg3"])
    variations = tracker.analyze_subject_evolution(thread, messages)

    # Should only have one variation (all same subject after normalization)
    assert len(variations) == 1
    assert variations[0] == "Discussion"


def test_analyze_subject_evolution_with_variations(tracker):
    """Test thread with multiple subject variations."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg1": create_email_message("msg1", "Original Topic", date=base_time),
        "msg2": create_email_message("msg2", "Re: Original Topic", date=base_time + timedelta(minutes=10)),
        "msg3": create_email_message("msg3", "Re: Original Topic - Updated", date=base_time + timedelta(minutes=20)),
        "msg4": create_email_message("msg4", "New Topic Direction", date=base_time + timedelta(minutes=30)),
    }

    thread = create_thread(["msg1", "msg2", "msg3", "msg4"])
    variations = tracker.analyze_subject_evolution(thread, messages)

    assert len(variations) == 3
    assert "Original Topic" in variations
    assert "Re: Original Topic - Updated" in variations
    assert "New Topic Direction" in variations


def test_analyze_subject_evolution_chronological_order(tracker):
    """Test variations are detected in chronological order."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg1": create_email_message("msg1", "Topic A", date=base_time),
        "msg2": create_email_message("msg2", "Topic B", date=base_time + timedelta(minutes=10)),
        "msg3": create_email_message("msg3", "Topic C", date=base_time + timedelta(minutes=20)),
    }

    thread = create_thread(["msg1", "msg2", "msg3"])
    variations = tracker.analyze_subject_evolution(thread, messages)

    # Should be in chronological order
    assert variations == ["Topic A", "Topic B", "Topic C"]


# ============================================================================
# Subject Change Detection Tests
# ============================================================================


def test_identify_subject_changes_no_changes(tracker):
    """Test thread with no subject changes."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg1": create_email_message("msg1", "Same Subject", date=base_time),
        "msg2": create_email_message("msg2", "Re: Same Subject", date=base_time + timedelta(minutes=10)),
    }

    thread = create_thread(["msg1", "msg2"])
    changes = tracker.identify_subject_changes(thread, messages)

    assert len(changes) == 0


def test_identify_subject_changes_single_change(tracker):
    """Test thread with one subject change."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg1": create_email_message("msg1", "Original", date=base_time),
        "msg2": create_email_message("msg2", "Changed Topic", date=base_time + timedelta(minutes=10)),
    }

    thread = create_thread(["msg1", "msg2"])
    changes = tracker.identify_subject_changes(thread, messages)

    assert len(changes) == 1
    change = changes[0]
    assert change["message_id"] == "msg2"
    assert change["previous_subject"] == "original"
    assert change["new_subject"] == "changed topic"


def test_identify_subject_changes_multiple(tracker):
    """Test thread with multiple subject changes."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg1": create_email_message("msg1", "Topic A", date=base_time),
        "msg2": create_email_message("msg2", "Topic B", date=base_time + timedelta(minutes=10)),
        "msg3": create_email_message("msg3", "Topic C", date=base_time + timedelta(minutes=20)),
    }

    thread = create_thread(["msg1", "msg2", "msg3"])
    changes = tracker.identify_subject_changes(thread, messages)

    assert len(changes) == 2
    assert changes[0]["message_id"] == "msg2"
    assert changes[1]["message_id"] == "msg3"


# ============================================================================
# Change Classification Tests
# ============================================================================


def test_classify_subject_change_minor_edit(tracker):
    """Test classification of minor edits (high similarity)."""
    # High overlap: 4 common words, 5 total words
    prev = "project discussion meeting notes"
    new = "project discussion meeting notes"

    change_type = tracker._classify_subject_change(prev, new)
    # Actually same, but test with tiny variation
    prev = "project discussion meeting"
    new = "project discussion meeting notes"
    change_type = tracker._classify_subject_change(prev, new)
    assert change_type == "minor_edit"


def test_classify_subject_change_topic_shift(tracker):
    """Test classification of topic shift (medium similarity)."""
    # Medium overlap: 2 common words (project, meeting), 5 total words
    prev = "project meeting summary"
    new = "project meeting agenda updates"

    change_type = tracker._classify_subject_change(prev, new)
    assert change_type == "topic_shift"


def test_classify_subject_change_branch(tracker):
    """Test classification of branch (low similarity)."""
    prev = "project discussion"
    new = "budget review meeting"

    change_type = tracker._classify_subject_change(prev, new)
    assert change_type == "branch"


# ============================================================================
# Evolution Summary Tests
# ============================================================================


def test_get_subject_evolution_summary_stable(tracker):
    """Test evolution summary for stable thread (no changes)."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg1": create_email_message("msg1", "Stable Topic", date=base_time),
        "msg2": create_email_message("msg2", "Re: Stable Topic", date=base_time + timedelta(minutes=10)),
        "msg3": create_email_message("msg3", "Re: Stable Topic", date=base_time + timedelta(minutes=20)),
    }

    thread = create_thread(["msg1", "msg2", "msg3"])
    summary = tracker.get_subject_evolution_summary(thread, messages)

    assert summary["total_variations"] == 1
    assert summary["subject_changes"] == 0
    assert summary["change_frequency"] == 0.0
    assert summary["subject_stability"] == 1.0
    assert summary["most_common_subject"] == "stable topic"


def test_get_subject_evolution_summary_volatile(tracker):
    """Test evolution summary for volatile thread (many changes)."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg1": create_email_message("msg1", "Topic A", date=base_time),
        "msg2": create_email_message("msg2", "Topic B", date=base_time + timedelta(minutes=10)),
        "msg3": create_email_message("msg3", "Topic C", date=base_time + timedelta(minutes=20)),
        "msg4": create_email_message("msg4", "Topic D", date=base_time + timedelta(minutes=30)),
    }

    thread = create_thread(["msg1", "msg2", "msg3", "msg4"])
    summary = tracker.get_subject_evolution_summary(thread, messages)

    assert summary["total_variations"] == 4
    assert summary["subject_changes"] == 3
    assert summary["change_frequency"] == 0.75  # 3 changes / 4 messages
    assert summary["subject_stability"] == 0.0  # Low stability


def test_get_subject_evolution_summary_single_message(tracker):
    """Test evolution summary for single-message thread."""
    messages = {
        "msg1": create_email_message("msg1", "Single Message", date=datetime.utcnow()),
    }

    thread = create_thread(["msg1"])
    summary = tracker.get_subject_evolution_summary(thread, messages)

    assert summary["total_variations"] == 1
    assert summary["subject_changes"] == 0
    assert summary["subject_stability"] == 1.0


# ============================================================================
# Message Sorting Tests
# ============================================================================


def test_sort_messages_chronologically(tracker):
    """Test chronological sorting of messages."""
    base_time = datetime(2024, 1, 1, 10, 0)

    messages = {
        "msg2": create_email_message("msg2", "B", date=base_time + timedelta(minutes=10)),
        "msg1": create_email_message("msg1", "A", date=base_time),
        "msg3": create_email_message("msg3", "C", date=base_time + timedelta(minutes=20)),
    }

    # Messages in thread are unsorted
    thread = create_thread(["msg2", "msg1", "msg3"])

    sorted_ids = tracker._sort_messages_chronologically(thread, messages)

    # Should be sorted by date
    assert sorted_ids == ["msg1", "msg2", "msg3"]


def test_sort_messages_same_timestamp(tracker):
    """Test sorting when messages have same timestamp."""
    now = datetime.utcnow()

    messages = {
        "msg1": create_email_message("msg1", "A", date=now),
        "msg2": create_email_message("msg2", "B", date=now),
        "msg3": create_email_message("msg3", "C", date=now),
    }

    thread = create_thread(["msg1", "msg2", "msg3"])
    sorted_ids = tracker._sort_messages_chronologically(thread, messages)

    # Should maintain stable sort (original order preserved for equal dates)
    assert len(sorted_ids) == 3


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_thread(tracker):
    """Test evolution analysis on empty thread."""
    thread = create_thread([])
    messages = {}

    variations = tracker.analyze_subject_evolution(thread, messages)
    changes = tracker.identify_subject_changes(thread, messages)

    assert len(variations) == 0
    assert len(changes) == 0


def test_missing_message_in_dict(tracker):
    """Test handling when message ID not in messages dict."""
    thread = create_thread(["msg1", "msg2", "msg3"])
    messages = {
        "msg1": create_email_message("msg1", "Topic", date=datetime.utcnow()),
        # msg2 and msg3 missing
    }

    variations = tracker.analyze_subject_evolution(thread, messages)

    # Should only process msg1
    assert len(variations) == 1


def test_none_subject_handling(tracker):
    """Test handling None subject."""
    messages = {
        "msg1": EmailMessage(
            message_id="msg1",
            uid=1,
            folder="INBOX",
            subject=None,  # None subject
            from_address=EmailAddress(address="sender@example.com"),
            date=datetime.utcnow(),
            size_bytes=1000,
            retrieved_at=datetime.utcnow(),
            mailbox_id="test_mailbox",
        ),
    }

    thread = create_thread(["msg1"])
    variations = tracker.analyze_subject_evolution(thread, messages)

    # Should handle None gracefully
    assert len(variations) == 0 or variations[0] == ""
