"""Tests for IMAP sync state models and persistence."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from futurnal.ingestion.imap.sync_state import (
    ImapSyncState,
    ImapSyncStateStore,
    SyncResult,
)


# ============================================================================
# SyncResult Tests
# ============================================================================


def test_sync_result_defaults():
    """Test SyncResult with default values."""
    result = SyncResult()

    assert result.new_messages == []
    assert result.updated_messages == []
    assert result.deleted_messages == []
    assert result.sync_duration_seconds == 0.0
    assert result.errors == []
    assert result.has_changes is False
    assert result.total_changes == 0


def test_sync_result_with_changes():
    """Test SyncResult with changes."""
    result = SyncResult(
        new_messages=[1, 2, 3],
        updated_messages=[4, 5],
        deleted_messages=[6],
        sync_duration_seconds=1.5,
    )

    assert len(result.new_messages) == 3
    assert len(result.updated_messages) == 2
    assert len(result.deleted_messages) == 1
    assert result.has_changes is True
    assert result.total_changes == 6
    assert result.sync_duration_seconds == 1.5


def test_sync_result_with_errors():
    """Test SyncResult with errors."""
    result = SyncResult(
        errors=["Connection timeout", "MODSEQ not supported"],
    )

    assert len(result.errors) == 2
    assert result.has_changes is False


# ============================================================================
# ImapSyncState Tests
# ============================================================================


def test_imap_sync_state_basic():
    """Test basic ImapSyncState creation."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=100,
        highest_modseq=500,
        last_sync_time=now,
    )

    assert state.mailbox_id == "test-mailbox"
    assert state.folder == "INBOX"
    assert state.uidvalidity == 12345
    assert state.last_synced_uid == 100
    assert state.highest_modseq == 500
    assert state.last_sync_time == now


def test_imap_sync_state_defaults():
    """Test ImapSyncState with default values."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_sync_time=now,
    )

    assert state.last_synced_uid == 0
    assert state.highest_modseq is None
    assert state.message_count == 0
    assert state.last_exists_count == 0
    assert state.total_syncs == 0
    assert state.messages_synced == 0
    assert state.messages_updated == 0
    assert state.messages_deleted == 0
    assert state.sync_errors == 0
    assert state.supports_idle is False
    assert state.supports_modseq is False
    assert state.supports_qresync is False


def test_imap_sync_state_capabilities():
    """Test ImapSyncState with server capabilities."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_sync_time=now,
        supports_idle=True,
        supports_modseq=True,
        supports_qresync=False,
    )

    assert state.supports_idle is True
    assert state.supports_modseq is True
    assert state.supports_qresync is False


def test_imap_sync_state_statistics():
    """Test ImapSyncState with statistics."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="test-mailbox",
        folder="INBOX",
        uidvalidity=12345,
        last_sync_time=now,
        total_syncs=5,
        messages_synced=100,
        messages_updated=20,
        messages_deleted=5,
        sync_errors=2,
    )

    assert state.total_syncs == 5
    assert state.messages_synced == 100
    assert state.messages_updated == 20
    assert state.messages_deleted == 5
    assert state.sync_errors == 2


def test_imap_sync_state_validation():
    """Test ImapSyncState field validation."""
    now = datetime.utcnow()

    # Negative last_synced_uid should fail
    with pytest.raises(ValueError):
        ImapSyncState(
            mailbox_id="test-mailbox",
            folder="INBOX",
            uidvalidity=12345,
            last_synced_uid=-1,
            last_sync_time=now,
        )

    # Negative message_count should fail
    with pytest.raises(ValueError):
        ImapSyncState(
            mailbox_id="test-mailbox",
            folder="INBOX",
            uidvalidity=12345,
            last_sync_time=now,
            message_count=-1,
        )


# ============================================================================
# ImapSyncStateStore Tests
# ============================================================================


@pytest.fixture
def state_store(tmp_path: Path) -> ImapSyncStateStore:
    """Create temporary state store."""
    db_path = tmp_path / "sync_state.db"
    store = ImapSyncStateStore(db_path)
    yield store
    store.close()


def test_state_store_initialization(state_store: ImapSyncStateStore):
    """Test state store initialization."""
    # Should be able to iterate (empty initially)
    states = list(state_store.iter_all())
    assert len(states) == 0


def test_state_store_upsert_and_fetch(state_store: ImapSyncStateStore):
    """Test upserting and fetching state."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="mailbox1",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=100,
        last_sync_time=now,
    )

    # Upsert state
    state_store.upsert(state)

    # Fetch state
    fetched = state_store.fetch("mailbox1", "INBOX")
    assert fetched is not None
    assert fetched.mailbox_id == "mailbox1"
    assert fetched.folder == "INBOX"
    assert fetched.uidvalidity == 12345
    assert fetched.last_synced_uid == 100


def test_state_store_upsert_update(state_store: ImapSyncStateStore):
    """Test updating existing state."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="mailbox1",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=100,
        last_sync_time=now,
    )

    # Insert
    state_store.upsert(state)

    # Update
    state.last_synced_uid = 200
    state.total_syncs = 1
    state_store.upsert(state)

    # Verify update
    fetched = state_store.fetch("mailbox1", "INBOX")
    assert fetched is not None
    assert fetched.last_synced_uid == 200
    assert fetched.total_syncs == 1


def test_state_store_fetch_nonexistent(state_store: ImapSyncStateStore):
    """Test fetching non-existent state."""
    fetched = state_store.fetch("nonexistent", "INBOX")
    assert fetched is None


def test_state_store_remove(state_store: ImapSyncStateStore):
    """Test removing state."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="mailbox1",
        folder="INBOX",
        uidvalidity=12345,
        last_sync_time=now,
    )

    # Insert
    state_store.upsert(state)
    assert state_store.fetch("mailbox1", "INBOX") is not None

    # Remove
    state_store.remove("mailbox1", "INBOX")
    assert state_store.fetch("mailbox1", "INBOX") is None


def test_state_store_list_by_mailbox(state_store: ImapSyncStateStore):
    """Test listing states by mailbox."""
    now = datetime.utcnow()

    # Insert multiple states for same mailbox
    for folder in ["INBOX", "Sent", "Drafts"]:
        state = ImapSyncState(
            mailbox_id="mailbox1",
            folder=folder,
            uidvalidity=12345,
            last_sync_time=now,
        )
        state_store.upsert(state)

    # Insert state for different mailbox
    state = ImapSyncState(
        mailbox_id="mailbox2",
        folder="INBOX",
        uidvalidity=67890,
        last_sync_time=now,
    )
    state_store.upsert(state)

    # List by mailbox
    states = list(state_store.list_by_mailbox("mailbox1"))
    assert len(states) == 3
    assert all(s.mailbox_id == "mailbox1" for s in states)
    assert {s.folder for s in states} == {"INBOX", "Sent", "Drafts"}


def test_state_store_iter_all(state_store: ImapSyncStateStore):
    """Test iterating all states."""
    now = datetime.utcnow()

    # Insert multiple states
    for mailbox_id in ["mailbox1", "mailbox2"]:
        for folder in ["INBOX", "Sent"]:
            state = ImapSyncState(
                mailbox_id=mailbox_id,
                folder=folder,
                uidvalidity=12345,
                last_sync_time=now,
            )
            state_store.upsert(state)

    # Iterate all
    states = list(state_store.iter_all())
    assert len(states) == 4


def test_state_store_modseq_persistence(state_store: ImapSyncStateStore):
    """Test MODSEQ value persistence."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="mailbox1",
        folder="INBOX",
        uidvalidity=12345,
        last_synced_uid=100,
        highest_modseq=500,
        last_sync_time=now,
        supports_modseq=True,
    )

    state_store.upsert(state)
    fetched = state_store.fetch("mailbox1", "INBOX")

    assert fetched is not None
    assert fetched.highest_modseq == 500
    assert fetched.supports_modseq is True


def test_state_store_capabilities_persistence(state_store: ImapSyncStateStore):
    """Test server capabilities persistence."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="mailbox1",
        folder="INBOX",
        uidvalidity=12345,
        last_sync_time=now,
        supports_idle=True,
        supports_modseq=True,
        supports_qresync=False,
    )

    state_store.upsert(state)
    fetched = state_store.fetch("mailbox1", "INBOX")

    assert fetched is not None
    assert fetched.supports_idle is True
    assert fetched.supports_modseq is True
    assert fetched.supports_qresync is False


def test_state_store_statistics_persistence(state_store: ImapSyncStateStore):
    """Test statistics persistence."""
    now = datetime.utcnow()
    state = ImapSyncState(
        mailbox_id="mailbox1",
        folder="INBOX",
        uidvalidity=12345,
        last_sync_time=now,
        total_syncs=10,
        messages_synced=500,
        messages_updated=50,
        messages_deleted=10,
        sync_errors=2,
    )

    state_store.upsert(state)
    fetched = state_store.fetch("mailbox1", "INBOX")

    assert fetched is not None
    assert fetched.total_syncs == 10
    assert fetched.messages_synced == 500
    assert fetched.messages_updated == 50
    assert fetched.messages_deleted == 10
    assert fetched.sync_errors == 2
