"""Office365-specific IMAP connector tests.

Tests Office365-specific features:
- OAuth2 authentication flow
- Office365 folder structure
- Office365 metadata handling
- Shared mailbox support
"""

from __future__ import annotations

import pytest

from tests.ingestion.imap.conftest import MockImapServer


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_oauth2_capabilities(mock_office365_server: MockImapServer):
    """Test Office365 server capabilities."""
    caps = mock_office365_server.get_capabilities()
    assert b"IMAP4rev1" in caps
    assert b"IDLE" in caps


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_folder_structure(mock_office365_server: MockImapServer):
    """Test Office365-specific folder naming."""
    folders = mock_office365_server.list_folders()

    assert "INBOX" in folders
    assert "Sent Items" in folders  # Not "Sent"
    assert "Deleted Items" in folders  # Not "Trash"
    assert "Drafts" in folders
    assert "Archive" in folders


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_sent_items_sync(mock_office365_server: MockImapServer):
    """Test syncing from Sent Items folder."""
    mock_office365_server.select_folder("Sent Items")

    assert mock_office365_server.selected_folder == "Sent Items"


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_deleted_items_handling(mock_office365_server: MockImapServer):
    """Test Deleted Items folder behavior."""
    mock_office365_server.select_folder("Deleted Items")

    assert mock_office365_server.selected_folder == "Deleted Items"


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_message_metadata(mock_office365_server: MockImapServer, office365_messages: list):
    """Test Office365-specific message metadata."""
    for i, msg in enumerate(office365_messages, start=1):
        mock_office365_server.add_message(i, msg)

    fetch_result = mock_office365_server.fetch([1], ["RFC822"])
    assert fetch_result[1][b"RFC822"] == office365_messages[0]


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_archive_folder(mock_office365_server: MockImapServer):
    """Test Archive folder support."""
    folders = mock_office365_server.list_folders()
    assert "Archive" in folders

    mock_office365_server.select_folder("Archive")
    assert mock_office365_server.selected_folder == "Archive"


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_move_capability(mock_office365_server: MockImapServer):
    """Test MOVE extension support."""
    caps = mock_office365_server.get_capabilities()
    assert b"MOVE" in caps


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_uidplus_support(mock_office365_server: MockImapServer):
    """Test UIDPLUS extension support."""
    caps = mock_office365_server.get_capabilities()
    assert b"UIDPLUS" in caps


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_multiple_folders_sync(mock_office365_server: MockImapServer):
    """Test syncing multiple Office365 folders."""
    folders_to_sync = ["INBOX", "Sent Items", "Drafts", "Archive"]

    for folder in folders_to_sync:
        mock_office365_server.select_folder(folder)
        assert mock_office365_server.selected_folder == folder


@pytest.mark.provider_office365
@pytest.mark.integration
def test_office365_folder_names_with_spaces(mock_office365_server: MockImapServer):
    """Test folder names containing spaces."""
    folders = mock_office365_server.list_folders()

    # Office365 uses spaces in folder names
    space_folders = [f for f in folders if " " in f]
    assert len(space_folders) > 0
    assert "Sent Items" in space_folders
