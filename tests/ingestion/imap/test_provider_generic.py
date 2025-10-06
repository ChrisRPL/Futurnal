"""Generic IMAP server tests.

Tests standard IMAP4rev1 compliance and basic features:
- App password authentication
- Standard folder structure
- Basic IMAP4rev1 commands
- Fallback strategies for limited capabilities
"""

from __future__ import annotations

import pytest

from tests.ingestion.imap.conftest import MockImapServer


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_imap4rev1_compliance(mock_imap_server: MockImapServer):
    """Test basic IMAP4rev1 compliance."""
    caps = mock_imap_server.get_capabilities()
    assert b"IMAP4rev1" in caps


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_standard_folders(mock_imap_server: MockImapServer):
    """Test standard folder structure."""
    folders = mock_imap_server.list_folders()

    assert "INBOX" in folders
    assert "Sent" in folders
    assert "Drafts" in folders
    assert "Trash" in folders


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_basic_select(mock_imap_server: MockImapServer):
    """Test basic SELECT command."""
    select_info = mock_imap_server.select_folder("INBOX")

    assert b"UIDVALIDITY" in select_info
    assert b"EXISTS" in select_info
    assert b"UIDNEXT" in select_info


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_search_all(mock_imap_server: MockImapServer):
    """Test SEARCH ALL command."""
    mock_imap_server.add_message(1, b"message1")
    mock_imap_server.add_message(2, b"message2")

    results = mock_imap_server.search(["ALL"])
    assert len(results) == 2


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_fetch_rfc822(mock_imap_server: MockImapServer, sample_email_message: bytes):
    """Test FETCH RFC822 command."""
    mock_imap_server.add_message(1, sample_email_message)

    fetch_result = mock_imap_server.fetch([1], ["RFC822"])
    assert fetch_result[1][b"RFC822"] == sample_email_message


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_fetch_flags(mock_imap_server: MockImapServer):
    """Test FETCH FLAGS command."""
    mock_imap_server.add_message(1, b"message")

    fetch_result = mock_imap_server.fetch([1], ["FLAGS"])
    assert b"FLAGS" in fetch_result[1]


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_idle_capability(mock_imap_server: MockImapServer):
    """Test IDLE capability support."""
    caps = mock_imap_server.get_capabilities()
    assert b"IDLE" in caps


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_condstore_capability(mock_imap_server: MockImapServer):
    """Test CONDSTORE capability support."""
    caps = mock_imap_server.get_capabilities()
    assert b"CONDSTORE" in caps


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_uidvalidity_stability(mock_imap_server: MockImapServer):
    """Test UIDVALIDITY remains stable across selects."""
    select1 = mock_imap_server.select_folder("INBOX")
    uidvalidity1 = select1[b"UIDVALIDITY"]

    select2 = mock_imap_server.select_folder("INBOX")
    uidvalidity2 = select2[b"UIDVALIDITY"]

    assert uidvalidity1 == uidvalidity2


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_message_deletion(mock_imap_server: MockImapServer):
    """Test message deletion handling."""
    mock_imap_server.add_message(1, b"message")
    assert 1 in mock_imap_server.messages

    mock_imap_server.simulate_deletion(1)
    assert 1 not in mock_imap_server.messages


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_incremental_sync_support(mock_imap_server: MockImapServer):
    """Test UID-based incremental sync."""
    # Add initial messages
    mock_imap_server.add_message(1, b"message1")
    mock_imap_server.add_message(2, b"message2")

    select_info = mock_imap_server.select_folder("INBOX")
    uidnext = select_info[b"UIDNEXT"]

    # Add new message
    new_uid = mock_imap_server.simulate_new_message(b"message3")

    assert new_uid >= uidnext


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_noop_keepalive(mock_imap_server: MockImapServer):
    """Test NOOP command for connection keepalive."""
    # Should not raise
    mock_imap_server.noop()


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_limited_capabilities_fallback():
    """Test fallback when advanced capabilities unavailable."""
    # Server without IDLE/CONDSTORE
    limited_server = MockImapServer(capabilities=[b"IMAP4rev1"])

    caps = limited_server.get_capabilities()
    assert b"IMAP4rev1" in caps
    assert b"IDLE" not in caps
    assert b"CONDSTORE" not in caps

    # Should still support basic operations
    select_info = limited_server.select_folder("INBOX")
    assert b"UIDVALIDITY" in select_info


@pytest.mark.provider_generic
@pytest.mark.integration
def test_generic_folder_case_sensitivity(mock_imap_server: MockImapServer):
    """Test folder name case handling."""
    # INBOX is case-insensitive per RFC 3501
    mock_imap_server.select_folder("INBOX")
    assert mock_imap_server.selected_folder == "INBOX"
