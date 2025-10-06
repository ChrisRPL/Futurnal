"""Tests for IMAP sync engine."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from futurnal.ingestion.imap.sync_engine import ImapSyncEngine
from futurnal.ingestion.imap.sync_state import ImapSyncState, ImapSyncStateStore


@pytest.fixture
def state_store(tmp_path: Path) -> ImapSyncStateStore:
    """Create temporary state store."""
    db_path = tmp_path / "sync_state.db"
    store = ImapSyncStateStore(db_path)
    yield store
    store.close()


@pytest.fixture
def mock_descriptor():
    """Create mock mailbox descriptor."""
    descriptor = Mock()
    descriptor.id = "test-mailbox"
    descriptor.email_address = "test@example.com"
    return descriptor


@pytest.fixture
def mock_connection_pool(mock_descriptor):
    """Create mock connection pool."""
    pool = Mock()
    pool.descriptor = mock_descriptor
    return pool


@pytest.fixture
def mock_imap_client():
    """Create mock IMAP client."""
    client = Mock()
    client.select_folder.return_value = {
        b"UIDVALIDITY": 12345,
        b"EXISTS": 10,
        b"HIGHESTMODSEQ": 100,
    }
    client.capabilities.return_value = {b"IDLE", b"CONDSTORE"}
    client.search.return_value = []
    client.fetch.return_value = {}
    return client


@pytest.fixture
def mock_connection(mock_imap_client):
    """Create mock connection."""
    connection = Mock()
    connection.connect.return_value.__enter__ = Mock(return_value=mock_imap_client)
    connection.connect.return_value.__exit__ = Mock(return_value=None)
    return connection


@pytest.fixture
def sync_engine(mock_connection_pool, state_store):
    """Create sync engine for testing."""
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )
    return engine


# ============================================================================
# Sync Engine Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sync_folder_first_sync(
    sync_engine: ImapSyncEngine, mock_connection_pool, mock_connection, mock_imap_client
):
    """Test first sync of a folder."""
    mock_connection_pool.acquire.return_value.__aenter__ = AsyncMock(
        return_value=mock_connection
    )
    mock_connection_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock search to return some UIDs
    mock_imap_client.search.return_value = [1, 2, 3, 4, 5]

    result = await sync_engine.sync_folder("INBOX")

    assert len(result.new_messages) == 5
    assert result.new_messages == [1, 2, 3, 4, 5]
    assert len(result.updated_messages) == 0
    assert len(result.deleted_messages) == 0


@pytest.mark.asyncio
async def test_sync_folder_incremental_uid(
    sync_engine: ImapSyncEngine,
    mock_connection_pool,
    mock_connection,
    mock_imap_client,
    state_store: ImapSyncStateStore,
):
    """Test incremental UID-based sync."""
    # Set up existing state
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=5,
        last_exists_count=5,
        last_sync_time=now,
    )
    state_store.upsert(state)

    mock_connection_pool.acquire.return_value.__aenter__ = AsyncMock(
        return_value=mock_connection
    )
    mock_connection_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock search to return new UIDs
    mock_imap_client.search.return_value = [6, 7, 8]

    result = await sync_engine.sync_folder("INBOX")

    assert len(result.new_messages) == 3
    assert result.new_messages == [6, 7, 8]


@pytest.mark.asyncio
async def test_sync_folder_uidvalidity_change(
    sync_engine: ImapSyncEngine,
    mock_connection_pool,
    mock_connection,
    mock_imap_client,
    state_store: ImapSyncStateStore,
):
    """Test full resync when UIDVALIDITY changes."""
    # Set up existing state with different UIDVALIDITY
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=99999,  # Different from mock
        last_synced_uid=5,
        last_sync_time=now,
    )
    state_store.upsert(state)

    mock_connection_pool.acquire.return_value.__aenter__ = AsyncMock(
        return_value=mock_connection
    )
    mock_connection_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock search to return all UIDs (full resync)
    mock_imap_client.search.return_value = [1, 2, 3, 4, 5]

    result = await sync_engine.sync_folder("INBOX")

    # Should treat all messages as new
    assert len(result.new_messages) == 5
    assert result.new_messages == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_sync_folder_deletion_detection(
    sync_engine: ImapSyncEngine,
    mock_connection_pool,
    mock_connection,
    mock_imap_client,
    state_store: ImapSyncStateStore,
):
    """Test deletion detection via EXISTS count."""
    # Set up existing state
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=10,
        last_exists_count=10,  # Previously had 10 messages
        last_sync_time=now,
    )
    state_store.upsert(state)

    mock_connection_pool.acquire.return_value.__aenter__ = AsyncMock(
        return_value=mock_connection
    )
    mock_connection_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock EXISTS to show fewer messages
    mock_imap_client.select_folder.return_value[b"EXISTS"] = 7

    # Mock search for new UIDs (none)
    mock_imap_client.search.side_effect = [
        [],  # No new messages
        [1, 2, 3, 4, 5, 6, 7],  # All current UIDs
    ]

    result = await sync_engine.sync_folder("INBOX")

    # Should detect deletions
    assert len(result.deleted_messages) > 0


@pytest.mark.asyncio
async def test_sync_folder_modseq_sync(
    sync_engine: ImapSyncEngine,
    mock_connection_pool,
    mock_connection,
    mock_imap_client,
    state_store: ImapSyncStateStore,
):
    """Test MODSEQ-based sync."""
    # Set up existing state with MODSEQ support
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=5,
        highest_modseq=50,
        last_exists_count=5,
        last_sync_time=now,
        supports_modseq=True,
    )
    state_store.upsert(state)

    mock_connection_pool.acquire.return_value.__aenter__ = AsyncMock(
        return_value=mock_connection
    )
    mock_connection_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock MODSEQ search
    mock_imap_client.search.return_value = [6, 7]
    mock_imap_client.fetch.return_value = {
        6: {b"MODSEQ": 51, b"FLAGS": []},
        7: {b"MODSEQ": 52, b"FLAGS": []},
    }

    result = await sync_engine.sync_folder("INBOX")

    # Should find new messages
    assert len(result.new_messages) == 2
    assert result.new_messages == [6, 7]


@pytest.mark.asyncio
async def test_sync_folder_capability_detection(
    sync_engine: ImapSyncEngine,
    mock_connection_pool,
    mock_connection,
    mock_imap_client,
    state_store: ImapSyncStateStore,
):
    """Test server capability detection."""
    # Set up initial state with matching UIDVALIDITY to avoid resync path
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,  # Match mock_imap_client UIDVALIDITY
        last_synced_uid=0,
        last_exists_count=0,
        last_sync_time=now,
    )
    state_store.upsert(state)

    mock_connection_pool.acquire.return_value.__aenter__ = AsyncMock(
        return_value=mock_connection
    )
    mock_connection_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock capabilities
    mock_imap_client.capabilities.return_value = {b"IDLE", b"CONDSTORE", b"QRESYNC"}
    mock_imap_client.search.return_value = []

    await sync_engine.sync_folder("INBOX")

    # Check state was updated with capabilities
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state is not None
    assert state.supports_idle is True
    assert state.supports_modseq is True
    assert state.supports_qresync is True


@pytest.mark.asyncio
async def test_sync_folder_error_handling(
    sync_engine: ImapSyncEngine, mock_connection_pool, mock_connection, mock_imap_client
):
    """Test error handling during sync."""
    mock_connection_pool.acquire.return_value.__aenter__ = AsyncMock(
        return_value=mock_connection
    )
    mock_connection_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock search to raise an exception
    mock_imap_client.search.side_effect = Exception("Connection lost")

    with pytest.raises(Exception):
        await sync_engine.sync_folder("INBOX")


@pytest.mark.asyncio
async def test_sync_folder_statistics_update(
    sync_engine: ImapSyncEngine,
    mock_connection_pool,
    mock_connection,
    mock_imap_client,
    state_store: ImapSyncStateStore,
):
    """Test sync statistics are updated."""
    mock_connection_pool.acquire.return_value.__aenter__ = AsyncMock(
        return_value=mock_connection
    )
    mock_connection_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock search to return UIDs
    mock_imap_client.search.return_value = [1, 2, 3]

    await sync_engine.sync_folder("INBOX")

    # Check statistics
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state is not None
    assert state.total_syncs == 1
    assert state.messages_synced == 3
