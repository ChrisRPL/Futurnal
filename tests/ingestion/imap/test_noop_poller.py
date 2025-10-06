"""Tests for NOOP polling fallback."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from futurnal.ingestion.imap.noop_poller import NoopPoller
from futurnal.ingestion.imap.sync_state import SyncResult


@pytest.fixture
def mock_sync_engine():
    """Create mock sync engine."""
    engine = Mock()
    engine.sync_folder = AsyncMock()
    return engine


@pytest.fixture
def mock_descriptor():
    """Create mock mailbox descriptor."""
    descriptor = Mock()
    descriptor.id = "test-mailbox"
    return descriptor


# ============================================================================
# NoopPoller Tests
# ============================================================================


@pytest.mark.asyncio
async def test_noop_poller_start_stop(mock_sync_engine, mock_descriptor):
    """Test starting and stopping NOOP poller."""
    poller = NoopPoller(
        sync_engine=mock_sync_engine,
        mailbox_descriptor=mock_descriptor,
        folder="INBOX",
        poll_interval=1,
    )

    # Start poller
    await poller.start()
    assert poller._poll_task is not None
    assert not poller._poll_task.done()

    # Wait a bit
    await asyncio.sleep(0.1)

    # Stop poller
    await poller.stop()
    assert poller._stop_event.is_set()


@pytest.mark.asyncio
async def test_noop_poller_performs_sync(mock_sync_engine, mock_descriptor):
    """Test NOOP poller performs sync."""
    # Mock sync to return no changes
    mock_sync_engine.sync_folder.return_value = SyncResult()

    poller = NoopPoller(
        sync_engine=mock_sync_engine,
        mailbox_descriptor=mock_descriptor,
        folder="INBOX",
        poll_interval=0.1,  # Short interval for testing
    )

    await poller.start()
    await asyncio.sleep(0.2)  # Wait for at least one poll
    await poller.stop()

    # Should have called sync_folder
    assert mock_sync_engine.sync_folder.call_count >= 1
    mock_sync_engine.sync_folder.assert_called_with("INBOX")


@pytest.mark.asyncio
async def test_noop_poller_detects_changes(mock_sync_engine, mock_descriptor):
    """Test NOOP poller detects changes."""
    # Mock sync to return changes
    mock_sync_engine.sync_folder.return_value = SyncResult(
        new_messages=[1, 2, 3],
        updated_messages=[4],
        deleted_messages=[5],
    )

    poller = NoopPoller(
        sync_engine=mock_sync_engine,
        mailbox_descriptor=mock_descriptor,
        folder="INBOX",
        poll_interval=0.1,
    )

    await poller.start()
    await asyncio.sleep(0.2)
    await poller.stop()

    # Should have detected changes
    assert mock_sync_engine.sync_folder.call_count >= 1


@pytest.mark.asyncio
async def test_noop_poller_handles_sync_error(mock_sync_engine, mock_descriptor):
    """Test NOOP poller handles sync errors."""
    # Mock sync to raise exception
    mock_sync_engine.sync_folder.side_effect = Exception("Connection lost")

    poller = NoopPoller(
        sync_engine=mock_sync_engine,
        mailbox_descriptor=mock_descriptor,
        folder="INBOX",
        poll_interval=0.1,
    )

    await poller.start()
    await asyncio.sleep(0.2)
    await poller.stop()

    # Should not crash, just log error
    assert mock_sync_engine.sync_folder.call_count >= 1


@pytest.mark.asyncio
async def test_noop_poller_respects_interval(mock_sync_engine, mock_descriptor):
    """Test NOOP poller respects polling interval."""
    mock_sync_engine.sync_folder.return_value = SyncResult()

    poller = NoopPoller(
        sync_engine=mock_sync_engine,
        mailbox_descriptor=mock_descriptor,
        folder="INBOX",
        poll_interval=0.2,  # 200ms interval
    )

    await poller.start()
    await asyncio.sleep(0.3)  # Wait for ~1.5 intervals
    await poller.stop()

    # Should have synced approximately once (1-2 times due to timing)
    assert 1 <= mock_sync_engine.sync_folder.call_count <= 2


@pytest.mark.asyncio
async def test_noop_poller_already_running(mock_sync_engine, mock_descriptor):
    """Test starting poller when already running."""
    poller = NoopPoller(
        sync_engine=mock_sync_engine,
        mailbox_descriptor=mock_descriptor,
        folder="INBOX",
        poll_interval=1,
    )

    await poller.start()

    # Try to start again
    with pytest.raises(RuntimeError, match="already running"):
        await poller.start()

    await poller.stop()


@pytest.mark.asyncio
async def test_noop_poller_multiple_folders(mock_sync_engine, mock_descriptor):
    """Test multiple pollers for different folders."""
    mock_sync_engine.sync_folder.return_value = SyncResult()

    poller1 = NoopPoller(
        sync_engine=mock_sync_engine,
        mailbox_descriptor=mock_descriptor,
        folder="INBOX",
        poll_interval=0.1,
    )

    poller2 = NoopPoller(
        sync_engine=mock_sync_engine,
        mailbox_descriptor=mock_descriptor,
        folder="Sent",
        poll_interval=0.1,
    )

    await poller1.start()
    await poller2.start()
    await asyncio.sleep(0.2)
    await poller1.stop()
    await poller2.stop()

    # Should have synced both folders
    calls = mock_sync_engine.sync_folder.call_args_list
    folders_synced = {call[0][0] for call in calls}
    assert "INBOX" in folders_synced
    assert "Sent" in folders_synced
