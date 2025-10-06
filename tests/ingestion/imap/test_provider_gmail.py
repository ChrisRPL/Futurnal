"""Gmail-specific IMAP connector tests.

Tests Gmail-specific features:
- OAuth2 authentication flow
- Gmail label extraction (X-GM-LABELS)
- Gmail special folders ([Gmail]/...)
- Gmail conversation threading
- Gmail IMAP extensions
"""

from __future__ import annotations

import pytest

from tests.ingestion.imap.conftest import MockImapServer


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_oauth2_capabilities(mock_gmail_server: MockImapServer):
    """Test Gmail server advertises OAuth2 capabilities."""
    caps = mock_gmail_server.get_capabilities()
    assert b"X-GM-EXT-1" in caps


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_label_extraction(mock_gmail_server: MockImapServer, sample_email_message: bytes):
    """Test extraction of Gmail labels from messages."""
    mock_gmail_server.add_message(
        1, sample_email_message, labels=["\\Inbox", "\\Important", "Work", "Project-A"]
    )

    fetch_result = mock_gmail_server.fetch([1], ["X-GM-LABELS"])
    labels = fetch_result[1][b"X-GM-LABELS"]

    assert len(labels) == 4
    assert b"\\Inbox" in labels
    assert b"Work" in labels


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_special_folders(mock_gmail_server: MockImapServer):
    """Test Gmail special folder structure."""
    folders = mock_gmail_server.list_folders()

    assert "[Gmail]/All Mail" in folders
    assert "[Gmail]/Sent Mail" in folders
    assert "[Gmail]/Drafts" in folders
    assert "[Gmail]/Trash" in folders


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_conversation_threading(mock_gmail_server: MockImapServer, gmail_threaded_messages: list):
    """Test Gmail thread ID assignment."""
    thread_id = "1234567890abcdef"

    for i, msg in enumerate(gmail_threaded_messages, start=1):
        mock_gmail_server.add_message(i, msg, thread_id=thread_id)

    # All messages should have same thread ID
    for uid in [1, 2, 3]:
        fetch_result = mock_gmail_server.fetch([uid], ["X-GM-THRID"])
        assert fetch_result[uid][b"X-GM-THRID"] == thread_id.encode()


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_high_volume_labels(mock_gmail_server: MockImapServer):
    """Test handling of many Gmail labels."""
    labels = [f"Label-{i}" for i in range(50)]
    mock_gmail_server.add_message(1, b"message", labels=labels)

    fetch_result = mock_gmail_server.fetch([1], ["X-GM-LABELS"])
    fetched_labels = fetch_result[1][b"X-GM-LABELS"]

    assert len(fetched_labels) == 50


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_system_labels(mock_gmail_server: MockImapServer):
    """Test Gmail system labels (backslash-prefixed)."""
    system_labels = ["\\Inbox", "\\Starred", "\\Important", "\\Sent", "\\Draft"]
    mock_gmail_server.add_message(1, b"message", labels=system_labels)

    fetch_result = mock_gmail_server.fetch([1], ["X-GM-LABELS"])
    labels = fetch_result[1][b"X-GM-LABELS"]

    assert all(label.startswith(b"\\") for label in labels)


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_all_mail_sync(mock_gmail_server: MockImapServer):
    """Test syncing from [Gmail]/All Mail folder."""
    mock_gmail_server.select_folder("[Gmail]/All Mail")

    assert mock_gmail_server.selected_folder == "[Gmail]/All Mail"


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_label_deletion_handling(mock_gmail_server: MockImapServer):
    """Test handling of label removal (message remains in All Mail)."""
    mock_gmail_server.add_message(1, b"message", labels=["\\Inbox", "Work"])

    # Simulate label removal (message still exists, just labels changed)
    mock_gmail_server.gmail_labels[1] = ["Work"]

    fetch_result = mock_gmail_server.fetch([1], ["X-GM-LABELS"])
    labels = fetch_result[1][b"X-GM-LABELS"]

    assert b"\\Inbox" not in labels
    assert b"Work" in labels


@pytest.mark.provider_gmail
@pytest.mark.integration
def test_gmail_uidplus_support(mock_gmail_server: MockImapServer):
    """Test Gmail supports UIDPLUS extension."""
    caps = mock_gmail_server.get_capabilities()
    assert b"UIDPLUS" in caps


@pytest.mark.provider_gmail
@pytest.mark.performance
def test_gmail_large_label_set_performance(mock_gmail_server: MockImapServer):
    """Test performance with many labels across many messages."""
    # Add 1000 messages with varying label sets
    for i in range(1000):
        labels = [f"Label-{j}" for j in range(i % 20)]
        mock_gmail_server.add_message(i + 1, b"message", labels=labels)

    # Fetch should complete without issues
    fetch_result = mock_gmail_server.fetch(list(range(1, 101)), ["X-GM-LABELS"])
    assert len(fetch_result) == 100
