"""Tests for MockImapServer infrastructure validation."""

from __future__ import annotations

import pytest

from tests.ingestion.imap.conftest import MockImapServer


@pytest.mark.integration
def test_mock_server_initialization():
    """Test mock server initializes with correct defaults."""
    server = MockImapServer(provider="generic")

    assert server.provider == "generic"
    assert server.uidvalidity == 1
    assert server.next_uid == 1
    assert server.modseq == 1
    assert b"IMAP4rev1" in server.capabilities
    assert b"IDLE" in server.capabilities


@pytest.mark.integration
def test_mock_server_add_message(sample_email_message: bytes):
    """Test adding messages to mock server."""
    server = MockImapServer()
    server.add_message(1, sample_email_message)

    assert 1 in server.messages
    assert server.messages[1] == sample_email_message
    assert server.next_uid == 2


@pytest.mark.integration
def test_mock_server_simulate_new_message(sample_email_message: bytes):
    """Test simulating new message arrival."""
    server = MockImapServer()
    uid = server.simulate_new_message(sample_email_message)

    assert uid == 1
    assert uid in server.messages
    assert server.next_uid == 2
    assert server.modseq == 2


@pytest.mark.integration
def test_mock_server_simulate_deletion(sample_email_message: bytes):
    """Test simulating message deletion."""
    server = MockImapServer()
    uid = server.simulate_new_message(sample_email_message)

    assert uid in server.messages
    server.simulate_deletion(uid)
    assert uid not in server.messages
    assert server.modseq == 3  # Incremented on deletion


@pytest.mark.integration
def test_mock_server_uidvalidity_change(sample_email_message: bytes):
    """Test UIDVALIDITY change clears mailbox."""
    server = MockImapServer()
    server.add_message(1, sample_email_message)
    server.add_message(2, sample_email_message)

    old_uidvalidity = server.uidvalidity
    server.change_uidvalidity()

    assert server.uidvalidity == old_uidvalidity + 1
    assert len(server.messages) == 0
    assert server.next_uid == 1


@pytest.mark.integration
def test_mock_server_select_folder():
    """Test SELECT command simulation."""
    server = MockImapServer()
    server.add_message(1, b"message1")
    server.add_message(2, b"message2")

    select_info = server.select_folder("INBOX")

    assert select_info[b"UIDVALIDITY"] == server.uidvalidity
    assert select_info[b"EXISTS"] == 2
    assert select_info[b"UIDNEXT"] == 3
    assert select_info[b"HIGHESTMODSEQ"] == server.modseq
    assert server.selected_folder == "INBOX"


@pytest.mark.integration
def test_mock_server_search():
    """Test SEARCH command simulation."""
    server = MockImapServer()
    server.add_message(1, b"message1")
    server.add_message(2, b"message2")
    server.add_message(5, b"message5")

    results = server.search(["ALL"])
    assert results == [1, 2, 5]


@pytest.mark.integration
def test_mock_server_fetch(sample_email_message: bytes):
    """Test FETCH command simulation."""
    server = MockImapServer()
    server.add_message(1, sample_email_message)

    fetch_results = server.fetch([1], ["RFC822", "FLAGS"])

    assert 1 in fetch_results
    assert fetch_results[1][b"UID"] == 1
    assert fetch_results[1][b"RFC822"] == sample_email_message
    assert b"FLAGS" in fetch_results[1]


@pytest.mark.integration
@pytest.mark.provider_gmail
def test_mock_gmail_server_labels():
    """Test Gmail-specific label support."""
    server = MockImapServer(provider="gmail")
    server.add_message(1, b"message", labels=["\\Inbox", "\\Important", "Work"])

    assert 1 in server.gmail_labels
    assert server.gmail_labels[1] == ["\\Inbox", "\\Important", "Work"]

    fetch_results = server.fetch([1], ["X-GM-LABELS"])
    assert b"X-GM-LABELS" in fetch_results[1]


@pytest.mark.integration
@pytest.mark.provider_gmail
def test_mock_gmail_server_thread_id():
    """Test Gmail-specific thread ID support."""
    server = MockImapServer(provider="gmail")
    server.add_message(1, b"message", thread_id="1234567890abcdef")

    assert 1 in server.gmail_thread_ids
    assert server.gmail_thread_ids[1] == "1234567890abcdef"


@pytest.mark.integration
@pytest.mark.provider_gmail
def test_mock_gmail_server_folders():
    """Test Gmail-specific folder structure."""
    server = MockImapServer(provider="gmail")
    folders = server.list_folders()

    assert "INBOX" in folders
    assert "[Gmail]/All Mail" in folders
    assert "[Gmail]/Sent Mail" in folders
    assert "[Gmail]/Drafts" in folders


@pytest.mark.integration
@pytest.mark.provider_office365
def test_mock_office365_server_folders():
    """Test Office365-specific folder structure."""
    server = MockImapServer(provider="office365")
    folders = server.list_folders()

    assert "INBOX" in folders
    assert "Sent Items" in folders
    assert "Deleted Items" in folders
    assert "Archive" in folders


@pytest.mark.integration
def test_mock_server_capabilities_customization():
    """Test custom capabilities configuration."""
    custom_caps = [b"IMAP4rev1", b"IDLE", b"QRESYNC"]
    server = MockImapServer(capabilities=custom_caps)

    caps = server.get_capabilities()
    assert caps == custom_caps
    assert b"QRESYNC" in caps


@pytest.mark.integration
def test_mock_server_idle_check():
    """Test IDLE command simulation."""
    server = MockImapServer()
    events = server.idle_check(timeout=0.1)

    # Default: no events
    assert events == []


@pytest.mark.integration
def test_mock_server_noop():
    """Test NOOP command simulation."""
    server = MockImapServer()
    # Should not raise
    server.noop()
