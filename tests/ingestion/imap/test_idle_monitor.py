"""Tests for IMAP IDLE monitoring."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from futurnal.ingestion.imap.idle_monitor import IdleMonitor
from futurnal.ingestion.imap.sync_state import SyncResult


@pytest.fixture
def mock_imap_client():
    """Create mock IMAP client."""
    client = Mock()
    client.select_folder.return_value = {}
    client.idle.return_value = None
    client.idle_check.return_value = []
    client.idle_done.return_value = None
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
async def callback():
    """Create mock callback."""
    return AsyncMock()


# ============================================================================
# IdleMonitor Tests
# ============================================================================


@pytest.mark.asyncio
async def test_idle_monitor_start_stop(mock_connection, callback):
    """Test starting and stopping IDLE monitor."""
    monitor = IdleMonitor(
        connection=mock_connection,
        folder="INBOX",
        callback=callback,
        renewal_interval=1,  # Short interval for testing
    )

    # Start monitor
    await monitor.start()
    assert monitor._monitor_task is not None
    assert not monitor._monitor_task.done()

    # Wait a bit
    await asyncio.sleep(0.1)

    # Stop monitor
    await monitor.stop()
    assert monitor._stop_event.is_set()


@pytest.mark.asyncio
async def test_idle_monitor_detects_exists(
    mock_connection, mock_imap_client, callback
):
    """Test IDLE monitor detects EXISTS responses."""
    # Mock IDLE response with EXISTS
    mock_imap_client.idle_check.return_value = [(b"3 EXISTS", b"")]

    monitor = IdleMonitor(
        connection=mock_connection,
        folder="INBOX",
        callback=callback,
        renewal_interval=0.1,  # Very short for testing
    )

    await monitor.start()
    await asyncio.sleep(0.2)  # Wait for IDLE cycle
    await monitor.stop()

    # Callback should have been called
    assert callback.call_count >= 1


@pytest.mark.asyncio
async def test_idle_monitor_detects_expunge(
    mock_connection, mock_imap_client, callback
):
    """Test IDLE monitor detects EXPUNGE responses."""
    # Mock IDLE response with EXPUNGE
    mock_imap_client.idle_check.return_value = [(b"2 EXPUNGE", b"")]

    monitor = IdleMonitor(
        connection=mock_connection,
        folder="INBOX",
        callback=callback,
        renewal_interval=0.1,
    )

    await monitor.start()
    await asyncio.sleep(0.2)
    await monitor.stop()

    # Callback should have been called
    assert callback.call_count >= 1


@pytest.mark.asyncio
async def test_idle_monitor_detects_fetch(
    mock_connection, mock_imap_client, callback
):
    """Test IDLE monitor detects FETCH responses."""
    # Mock IDLE response with FETCH
    mock_imap_client.idle_check.return_value = [(b"1 FETCH (FLAGS (\\Seen))", b"")]

    monitor = IdleMonitor(
        connection=mock_connection,
        folder="INBOX",
        callback=callback,
        renewal_interval=0.1,
    )

    await monitor.start()
    await asyncio.sleep(0.2)
    await monitor.stop()

    # Callback should have been called
    assert callback.call_count >= 1


@pytest.mark.asyncio
async def test_idle_monitor_no_changes(mock_connection, mock_imap_client, callback):
    """Test IDLE monitor with no changes."""
    # Mock IDLE response with no changes
    mock_imap_client.idle_check.return_value = []

    monitor = IdleMonitor(
        connection=mock_connection,
        folder="INBOX",
        callback=callback,
        renewal_interval=0.1,
    )

    await monitor.start()
    await asyncio.sleep(0.2)
    await monitor.stop()

    # Callback should not have been called
    assert callback.call_count == 0


@pytest.mark.asyncio
async def test_idle_monitor_renewal(mock_connection, mock_imap_client, callback):
    """Test IDLE renewal after timeout."""
    # Mock IDLE to timeout (no responses)
    mock_imap_client.idle_check.return_value = []

    monitor = IdleMonitor(
        connection=mock_connection,
        folder="INBOX",
        callback=callback,
        renewal_interval=0.1,
    )

    await monitor.start()
    await asyncio.sleep(0.3)  # Wait for multiple renewals
    await monitor.stop()

    # Should have renewed multiple times
    assert mock_connection.metrics.record_idle_renewal.call_count >= 2


@pytest.mark.asyncio
async def test_idle_monitor_already_running(mock_connection, callback):
    """Test starting monitor when already running."""
    monitor = IdleMonitor(
        connection=mock_connection,
        folder="INBOX",
        callback=callback,
        renewal_interval=1,
    )

    await monitor.start()

    # Try to start again
    with pytest.raises(RuntimeError, match="already running"):
        await monitor.start()

    await monitor.stop()


def test_idle_monitor_detect_changes_empty():
    """Test change detection with empty responses."""
    monitor = IdleMonitor(
        connection=Mock(),
        folder="INBOX",
        callback=AsyncMock(),
    )

    assert monitor._detect_changes([]) is False


def test_idle_monitor_detect_changes_exists():
    """Test change detection with EXISTS."""
    monitor = IdleMonitor(
        connection=Mock(),
        folder="INBOX",
        callback=AsyncMock(),
    )

    responses = [(b"3 EXISTS", b"")]
    assert monitor._detect_changes(responses) is True


def test_idle_monitor_detect_changes_expunge():
    """Test change detection with EXPUNGE."""
    monitor = IdleMonitor(
        connection=Mock(),
        folder="INBOX",
        callback=AsyncMock(),
    )

    responses = [(b"2 EXPUNGE", b"")]
    assert monitor._detect_changes(responses) is True


def test_idle_monitor_detect_changes_fetch():
    """Test change detection with FETCH."""
    monitor = IdleMonitor(
        connection=Mock(),
        folder="INBOX",
        callback=AsyncMock(),
    )

    responses = [(b"1 FETCH (FLAGS (\\Seen))", b"")]
    assert monitor._detect_changes(responses) is True


def test_idle_monitor_detect_changes_invalid():
    """Test change detection with invalid responses."""
    monitor = IdleMonitor(
        connection=Mock(),
        folder="INBOX",
        callback=AsyncMock(),
    )

    # Invalid response format
    responses = ["invalid", None, 123]
    assert monitor._detect_changes(responses) is False
