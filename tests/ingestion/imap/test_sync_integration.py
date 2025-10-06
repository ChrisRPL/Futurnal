"""Integration tests for IMAP sync workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from futurnal.ingestion.imap.idle_monitor import IdleMonitor
from futurnal.ingestion.imap.noop_poller import NoopPoller
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
def mock_imap_client():
    """Create mock IMAP client for integration tests."""
    client = Mock()
    client.select_folder.return_value = {
        b"UIDVALIDITY": 12345,
        b"EXISTS": 0,
        b"HIGHESTMODSEQ": 0,
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
    connection.metrics = Mock()
    connection.metrics.record_idle_renewal = Mock()
    connection._log_event = Mock()
    return connection


@pytest.fixture
def mock_connection_pool(mock_descriptor, mock_connection):
    """Create mock connection pool."""
    pool = Mock()
    pool.descriptor = mock_descriptor
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_full_sync_workflow(
    mock_connection_pool, mock_imap_client, state_store
):
    """Test complete sync workflow from initial to incremental."""
    # Create sync engine
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )

    # Initial sync - no messages
    mock_imap_client.search.return_value = []
    result1 = await engine.sync_folder("INBOX")
    assert len(result1.new_messages) == 0

    # Verify state was created
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state is not None
    assert state.uidvalidity == 12345
    assert state.total_syncs == 1

    # Second sync - new messages arrive
    mock_imap_client.select_folder.return_value[b"EXISTS"] = 3
    mock_imap_client.search.return_value = [1, 2, 3]
    result2 = await engine.sync_folder("INBOX")
    assert len(result2.new_messages) == 3

    # Verify state was updated
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state is not None
    assert state.last_synced_uid == 3
    assert state.message_count == 3
    assert state.total_syncs == 2
    assert state.messages_synced == 3


@pytest.mark.asyncio
async def test_uidvalidity_resync_workflow(
    mock_connection_pool, mock_imap_client, state_store
):
    """Test UIDVALIDITY change triggers full resync."""
    # Create sync engine
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )

    # Initial sync
    mock_imap_client.search.return_value = [1, 2, 3]
    await engine.sync_folder("INBOX")

    state = state_store.fetch("test-mailbox", "INBOX")
    assert state.uidvalidity == 12345
    assert state.last_synced_uid == 3

    # UIDVALIDITY changes (e.g., mailbox migration)
    mock_imap_client.select_folder.return_value[b"UIDVALIDITY"] = 99999
    mock_imap_client.search.return_value = [1, 2, 3, 4, 5]

    result = await engine.sync_folder("INBOX")

    # Should perform full resync
    assert len(result.new_messages) == 5

    # State should be reset with new UIDVALIDITY
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state.uidvalidity == 99999
    assert state.last_synced_uid == 5


@pytest.mark.asyncio
async def test_modseq_incremental_sync(
    mock_connection_pool, mock_imap_client, state_store
):
    """Test MODSEQ-based incremental sync."""
    # Set up initial state with MODSEQ support
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=5,
        highest_modseq=100,
        last_exists_count=5,
        last_sync_time=now,
        supports_modseq=True,
    )
    state_store.upsert(state)

    # Create sync engine
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )

    # Mock MODSEQ search for changed messages
    mock_imap_client.select_folder.return_value[b"HIGHESTMODSEQ"] = 105
    mock_imap_client.search.return_value = [6, 7]
    mock_imap_client.fetch.return_value = {
        6: {b"MODSEQ": 101, b"FLAGS": []},
        7: {b"MODSEQ": 102, b"FLAGS": []},
    }

    result = await engine.sync_folder("INBOX")

    # Should find new messages
    assert len(result.new_messages) == 2
    assert 6 in result.new_messages
    assert 7 in result.new_messages

    # State should be updated
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state.last_synced_uid == 7
    assert state.highest_modseq == 105


@pytest.mark.asyncio
async def test_deletion_detection_workflow(
    mock_connection_pool, mock_imap_client, state_store
):
    """Test message deletion detection."""
    # Set up initial state
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=10,
        last_exists_count=10,
        last_sync_time=now,
    )
    state_store.upsert(state)

    # Create sync engine
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )

    # Mock deletion: EXISTS decreased and some UIDs missing
    mock_imap_client.select_folder.return_value[b"EXISTS"] = 7
    mock_imap_client.search.side_effect = [
        [],  # No new messages
        [1, 2, 3, 4, 5, 6, 7],  # Current UIDs (8, 9, 10 deleted)
    ]

    result = await engine.sync_folder("INBOX")

    # Should detect deletions
    assert len(result.deleted_messages) > 0
    deleted_set = set(result.deleted_messages)
    assert 8 in deleted_set or 9 in deleted_set or 10 in deleted_set


@pytest.mark.asyncio
async def test_multi_folder_sync(mock_connection_pool, mock_imap_client, state_store):
    """Test syncing multiple folders."""
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )

    folders = ["INBOX", "Sent", "Drafts"]
    results = {}

    for folder in folders:
        # Mock different message counts per folder
        mock_imap_client.select_folder.return_value[b"UIDVALIDITY"] = 12345
        mock_imap_client.search.return_value = list(range(1, len(folder) + 1))

        results[folder] = await engine.sync_folder(folder)

    # Verify all folders were synced
    for folder in folders:
        state = state_store.fetch("test-mailbox", folder)
        assert state is not None
        assert state.folder == folder
        assert state.total_syncs == 1


@pytest.mark.asyncio
async def test_sync_statistics_accumulation(
    mock_connection_pool, mock_imap_client, state_store
):
    """Test sync statistics accumulate correctly."""
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )

    # First sync
    mock_imap_client.search.return_value = [1, 2, 3]
    await engine.sync_folder("INBOX")

    # Second sync
    mock_imap_client.search.return_value = [4, 5]
    await engine.sync_folder("INBOX")

    # Third sync
    mock_imap_client.search.return_value = [6]
    await engine.sync_folder("INBOX")

    # Check accumulated statistics
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state is not None
    assert state.total_syncs == 3
    assert state.messages_synced == 6  # 3 + 2 + 1


@pytest.mark.asyncio
async def test_idle_to_sync_workflow(
    mock_connection, mock_imap_client, mock_connection_pool, state_store
):
    """Test IDLE monitor triggering sync."""
    # Create sync engine
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )

    # Track sync calls
    sync_called = False

    async def sync_callback(result):
        nonlocal sync_called
        sync_called = True
        # Trigger actual sync
        await engine.sync_folder("INBOX")

    # Mock IDLE to detect changes
    mock_imap_client.idle_check.return_value = [(b"1 EXISTS", b"")]
    mock_imap_client.search.return_value = [1]

    # Create and start IDLE monitor
    monitor = IdleMonitor(
        connection=mock_connection,
        folder="INBOX",
        callback=sync_callback,
        renewal_interval=0.1,
    )

    await monitor.start()
    await asyncio.sleep(0.2)  # Wait for IDLE cycle
    await monitor.stop()

    # Sync should have been triggered
    assert sync_called

    # State should be updated
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state is not None


@pytest.mark.asyncio
async def test_noop_poller_workflow(mock_connection_pool, mock_imap_client, state_store):
    """Test NOOP poller performing regular syncs."""
    engine = ImapSyncEngine(
        connection_pool=mock_connection_pool,
        state_store=state_store,
    )

    mock_descriptor = mock_connection_pool.descriptor

    # Mock sync results
    mock_imap_client.search.return_value = [1, 2, 3]

    # Create and start NOOP poller
    poller = NoopPoller(
        sync_engine=engine,
        mailbox_descriptor=mock_descriptor,
        folder="INBOX",
        poll_interval=0.1,
    )

    await poller.start()
    await asyncio.sleep(0.3)  # Wait for multiple polls
    await poller.stop()

    # State should be updated
    state = state_store.fetch("test-mailbox", "INBOX")
    assert state is not None
    assert state.total_syncs >= 1


import asyncio
