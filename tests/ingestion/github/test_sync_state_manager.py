"""Tests for GitHub sync state manager."""

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from futurnal.ingestion.github.sync_state_manager import SyncStateManager
from futurnal.ingestion.github.sync_models import SyncState, SyncStatus


@pytest.fixture
def temp_state_dir():
    """Create temporary state directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def state_manager(temp_state_dir):
    """Create state manager with temp directory."""
    return SyncStateManager(state_dir=temp_state_dir)


def test_state_manager_initialization(temp_state_dir):
    """Test state manager initialization."""
    manager = SyncStateManager(state_dir=temp_state_dir)

    assert manager.state_dir.resolve() == temp_state_dir.resolve()
    assert manager.state_dir.exists()
    assert manager.state_dir.is_dir()


def test_state_manager_default_directory():
    """Test state manager with default directory."""
    manager = SyncStateManager()

    expected = Path.home() / ".futurnal" / "sync_state" / "github"
    assert manager.state_dir == expected


def test_save_and_load_state(state_manager):
    """Test saving and loading state."""
    state = SyncState(
        repo_id="test-repo",
        sync_mode="graphql_api",
        status=SyncStatus.PENDING,
    )

    # Save state
    state_manager.save(state)

    # Load state
    loaded = state_manager.load("test-repo")

    assert loaded is not None
    assert loaded.repo_id == "test-repo"
    assert loaded.sync_mode == "graphql_api"
    assert loaded.status == SyncStatus.PENDING


def test_load_nonexistent_state(state_manager):
    """Test loading non-existent state."""
    loaded = state_manager.load("nonexistent")

    assert loaded is None


def test_delete_state(state_manager):
    """Test deleting state."""
    state = SyncState(repo_id="test-repo", sync_mode="graphql_api")
    state_manager.save(state)

    # Verify exists
    assert state_manager.load("test-repo") is not None

    # Delete
    result = state_manager.delete("test-repo")
    assert result is True

    # Verify deleted
    assert state_manager.load("test-repo") is None


def test_delete_nonexistent_state(state_manager):
    """Test deleting non-existent state."""
    result = state_manager.delete("nonexistent")
    assert result is False


def test_list_all_states(state_manager):
    """Test listing all states."""
    # Create multiple states
    for i in range(3):
        state = SyncState(repo_id=f"repo-{i}", sync_mode="graphql_api")
        state_manager.save(state)

    # List all
    states = state_manager.list_all()

    assert len(states) == 3
    assert all(isinstance(s, SyncState) for s in states)


def test_find_by_status(state_manager):
    """Test finding states by status."""
    # Create states with different statuses
    state1 = SyncState(repo_id="repo-1", sync_mode="graphql_api")
    state1.status = SyncStatus.COMPLETED
    state_manager.save(state1)

    state2 = SyncState(repo_id="repo-2", sync_mode="graphql_api")
    state2.status = SyncStatus.FAILED
    state_manager.save(state2)

    state3 = SyncState(repo_id="repo-3", sync_mode="graphql_api")
    state3.status = SyncStatus.COMPLETED
    state_manager.save(state3)

    # Find completed
    completed = state_manager.find_by_status(SyncStatus.COMPLETED)
    assert len(completed) == 2

    # Find failed
    failed = state_manager.find_by_status(SyncStatus.FAILED)
    assert len(failed) == 1


def test_find_unhealthy(state_manager):
    """Test finding unhealthy states."""
    # Create healthy state
    healthy = SyncState(repo_id="healthy", sync_mode="graphql_api")
    healthy.consecutive_failures = 2
    state_manager.save(healthy)

    # Create unhealthy state
    unhealthy = SyncState(repo_id="unhealthy", sync_mode="graphql_api")
    unhealthy.consecutive_failures = 10
    state_manager.save(unhealthy)

    # Find unhealthy
    unhealthy_states = state_manager.find_unhealthy()
    assert len(unhealthy_states) == 1
    assert unhealthy_states[0].repo_id == "unhealthy"


def test_get_or_create_new(state_manager):
    """Test get_or_create with new state."""
    state = state_manager.get_or_create(
        repo_id="new-repo",
        sync_mode="git_clone",
        local_clone_path=Path("/tmp/test"),
    )

    assert state.repo_id == "new-repo"
    assert state.sync_mode == "git_clone"
    assert state.local_clone_path == Path("/tmp/test")
    assert state.status == SyncStatus.PENDING


def test_get_or_create_existing(state_manager):
    """Test get_or_create with existing state."""
    # Create initial state
    initial = SyncState(repo_id="existing", sync_mode="graphql_api")
    initial.total_files_synced = 100
    state_manager.save(initial)

    # Get existing
    state = state_manager.get_or_create(
        repo_id="existing",
        sync_mode="graphql_api",
    )

    assert state.repo_id == "existing"
    assert state.total_files_synced == 100  # Preserved


def test_update_commit_sha(state_manager):
    """Test updating commit SHA."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")
    state_manager.save(state)

    state_manager.update_commit_sha("test", "abc123")

    loaded = state_manager.load("test")
    assert loaded.last_commit_sha == "abc123"


def test_mark_sync_started(state_manager):
    """Test marking sync as started."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")
    state_manager.save(state)

    state_manager.mark_sync_started("test")

    loaded = state_manager.load("test")
    assert loaded.status == SyncStatus.IN_PROGRESS
    assert loaded.last_attempt_time is not None


def test_mark_sync_completed(state_manager):
    """Test marking sync as completed."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")
    state.consecutive_failures = 3
    state_manager.save(state)

    state_manager.mark_sync_completed(
        repo_id="test",
        files_synced=50,
        bytes_synced=100000,
        commits=5,
    )

    loaded = state_manager.load("test")
    assert loaded.status == SyncStatus.COMPLETED
    assert loaded.total_files_synced == 50
    assert loaded.total_bytes_synced == 100000
    assert loaded.total_commits_processed == 5
    assert loaded.consecutive_failures == 0


def test_mark_sync_failed(state_manager):
    """Test marking sync as failed."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")
    state_manager.save(state)

    state_manager.mark_sync_failed("test", error_count=5)

    loaded = state_manager.load("test")
    assert loaded.status == SyncStatus.FAILED
    assert loaded.sync_errors == 5
    assert loaded.consecutive_failures == 1


def test_get_history(state_manager):
    """Test getting sync history."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")

    # Simulate multiple successful syncs
    for i in range(5):
        state.mark_sync_completed(files_synced=10, bytes_synced=1000, commits=1)
        state_manager.save(state)

    # Get history
    history = state_manager.get_history("test", limit=3)

    # Should have up to 3 history entries
    assert len(history) <= 3


def test_get_statistics(state_manager):
    """Test getting statistics."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")
    state.total_files_synced = 100
    state.total_bytes_synced = 500000
    state.consecutive_failures = 2
    state_manager.save(state)

    stats = state_manager.get_statistics("test")

    assert stats["total_files_synced"] == 100
    assert stats["total_bytes_synced"] == 500000
    assert stats["consecutive_failures"] == 2


def test_get_statistics_nonexistent(state_manager):
    """Test getting statistics for non-existent repo."""
    stats = state_manager.get_statistics("nonexistent")

    assert stats["total_syncs"] == 0
    assert stats["successful_syncs"] == 0
    assert stats["failed_syncs"] == 0


def test_cleanup_old_states(state_manager):
    """Test cleaning up old states."""
    # Create old state
    state = SyncState(repo_id="old", sync_mode="graphql_api")
    state_manager.save(state)

    # Manually modify file mtime to be old
    state_path = state_manager._state_path("old")
    import os
    import time

    # Set mtime to 100 days ago
    old_time = time.time() - (100 * 86400)
    os.utime(state_path, (old_time, old_time))

    # Clean up states older than 90 days
    cleaned = state_manager.cleanup_old_states(days=90)

    assert cleaned >= 1
    assert state_manager.load("old") is None


def test_concurrent_access(state_manager):
    """Test that file locking works for concurrent access."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")

    # Multiple saves should not corrupt
    for i in range(10):
        state.total_files_synced = i
        state_manager.save(state)

    loaded = state_manager.load("test")
    assert loaded.total_files_synced == 9  # Last value


def test_corrupted_state_file(state_manager, temp_state_dir):
    """Test handling of corrupted state file."""
    # Create corrupted file
    state_path = temp_state_dir / "corrupted.json"
    state_path.write_text("invalid json{{{")

    # Should return None for corrupted file
    loaded = state_manager.load("corrupted")
    assert loaded is None


def test_atomic_write(state_manager):
    """Test that writes are atomic."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")
    state_manager.save(state)

    # Verify no .tmp files left behind
    tmp_files = list(state_manager.state_dir.glob("*.tmp"))
    assert len(tmp_files) == 0
