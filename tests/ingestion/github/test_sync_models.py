"""Tests for GitHub sync data models."""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from futurnal.ingestion.github.sync_models import (
    BranchSyncState,
    DiskSpaceEstimate,
    FileContent,
    FileEntry,
    SyncResult,
    SyncState,
    SyncStatus,
    SyncStrategy,
)


# ---------------------------------------------------------------------------
# SyncStrategy Tests
# ---------------------------------------------------------------------------


def test_sync_strategy_defaults():
    """Test SyncStrategy default values."""
    strategy = SyncStrategy()

    assert strategy.branches == ["main"]
    assert strategy.include_all_branches is False
    assert strategy.include_tags is False
    assert strategy.file_patterns == []
    assert ".git/" in strategy.exclude_patterns
    assert "node_modules/" in strategy.exclude_patterns
    assert strategy.max_file_size_mb == 10
    assert strategy.fetch_file_content is True
    assert strategy.batch_size == 10
    assert strategy.clone_depth is None
    assert strategy.use_sparse_checkout is False
    assert strategy.clone_submodules is False
    assert strategy.single_branch is True


def test_sync_strategy_custom_values():
    """Test SyncStrategy with custom values."""
    strategy = SyncStrategy(
        branches=["develop", "feature/test"],
        include_all_branches=True,
        include_tags=True,
        file_patterns=["*.py", "*.md"],
        exclude_patterns=["tests/*"],
        max_file_size_mb=50,
        fetch_file_content=False,
        batch_size=20,
        clone_depth=100,
        use_sparse_checkout=True,
        clone_submodules=True,
        single_branch=False,
    )

    assert strategy.branches == ["develop", "feature/test"]
    assert strategy.include_all_branches is True
    assert strategy.include_tags is True
    assert strategy.file_patterns == ["*.py", "*.md"]
    assert strategy.exclude_patterns == ["tests/*"]
    assert strategy.max_file_size_mb == 50
    assert strategy.fetch_file_content is False
    assert strategy.batch_size == 20
    assert strategy.clone_depth == 100
    assert strategy.use_sparse_checkout is True
    assert strategy.clone_submodules is True
    assert strategy.single_branch is False


def test_sync_strategy_branch_validation():
    """Test branch name validation."""
    # Empty branches should default to main
    strategy = SyncStrategy(branches=[])
    assert strategy.branches == ["main"]

    # Whitespace should be stripped
    strategy = SyncStrategy(branches=["  develop  ", "main"])
    assert strategy.branches == ["develop", "main"]


def test_sync_strategy_pattern_validation():
    """Test pattern validation strips whitespace."""
    strategy = SyncStrategy(
        file_patterns=["  *.py  ", "*.md"],
        exclude_patterns=["  tests/*  ", "*.pyc"],
    )

    assert strategy.file_patterns == ["*.py", "*.md"]
    assert strategy.exclude_patterns == ["tests/*", "*.pyc"]


# ---------------------------------------------------------------------------
# FileEntry Tests
# ---------------------------------------------------------------------------


def test_file_entry_basic():
    """Test basic FileEntry creation."""
    entry = FileEntry(
        path="src/main.py",
        name="main.py",
        type="blob",
        mode="100644",
        size=1024,
        is_binary=False,
        sha="abc123",
    )

    assert entry.path == "src/main.py"
    assert entry.name == "main.py"
    assert entry.type == "blob"
    assert entry.size == 1024
    assert entry.is_binary is False
    assert entry.sha == "abc123"


def test_file_entry_size_mb():
    """Test size_mb property."""
    entry = FileEntry(
        path="test.txt",
        name="test.txt",
        type="blob",
        size=1048576,  # 1 MB
    )

    assert entry.size_mb == 1.0


def test_file_entry_should_skip():
    """Test should_skip logic."""
    # Binary file should be skipped
    binary_entry = FileEntry(
        path="test.bin",
        name="test.bin",
        type="blob",
        size=100,
        is_binary=True,
    )
    assert binary_entry.should_skip(max_size_mb=10) is True

    # Large file should be skipped
    large_entry = FileEntry(
        path="large.txt",
        name="large.txt",
        type="blob",
        size=20 * 1024 * 1024,  # 20 MB
        is_binary=False,
    )
    assert large_entry.should_skip(max_size_mb=10) is True

    # Normal file should not be skipped
    normal_entry = FileEntry(
        path="normal.txt",
        name="normal.txt",
        type="blob",
        size=1024,
        is_binary=False,
    )
    assert normal_entry.should_skip(max_size_mb=10) is False


# ---------------------------------------------------------------------------
# FileContent Tests
# ---------------------------------------------------------------------------


def test_file_content_basic():
    """Test FileContent creation."""
    content = FileContent(
        path="README.md",
        content="# Test Project",
        size=14,
        sha="def456",
        encoding="utf-8",
        is_binary=False,
    )

    assert content.path == "README.md"
    assert content.content == "# Test Project"
    assert content.size == 14
    assert content.sha == "def456"
    assert content.encoding == "utf-8"
    assert content.is_binary is False


# ---------------------------------------------------------------------------
# BranchSyncState Tests
# ---------------------------------------------------------------------------


def test_branch_sync_state_basic():
    """Test BranchSyncState creation."""
    now = datetime.now(timezone.utc)

    state = BranchSyncState(
        branch_name="main",
        last_commit_sha="abc123",
        last_sync_time=now,
        file_count=100,
        total_bytes=50000,
        sync_errors=0,
    )

    assert state.branch_name == "main"
    assert state.last_commit_sha == "abc123"
    assert state.last_sync_time == now
    assert state.file_count == 100
    assert state.total_bytes == 50000
    assert state.sync_errors == 0


def test_branch_sync_state_timezone_aware():
    """Test timezone awareness enforcement."""
    naive_dt = datetime.now()
    state = BranchSyncState(
        branch_name="main",
        last_commit_sha="abc123",
        last_sync_time=naive_dt,
    )

    # Should add UTC timezone
    assert state.last_sync_time.tzinfo is not None


# ---------------------------------------------------------------------------
# SyncState Tests
# ---------------------------------------------------------------------------


def test_sync_state_creation():
    """Test SyncState creation with defaults."""
    state = SyncState(
        repo_id="test-repo-id",
        sync_mode="graphql_api",
    )

    assert state.repo_id == "test-repo-id"
    assert state.sync_mode == "graphql_api"
    assert state.status == SyncStatus.PENDING
    assert state.created_at.tzinfo is not None
    assert state.last_sync_time is None
    assert state.total_files_synced == 0
    assert state.total_bytes_synced == 0
    assert state.consecutive_failures == 0


def test_sync_state_mark_sync_started():
    """Test marking sync as started."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")

    state.mark_sync_started()

    assert state.status == SyncStatus.IN_PROGRESS
    assert state.last_attempt_time is not None


def test_sync_state_mark_sync_completed():
    """Test marking sync as completed."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")
    state.consecutive_failures = 2

    state.mark_sync_completed(
        files_synced=50,
        bytes_synced=100000,
        commits=5,
    )

    assert state.status == SyncStatus.COMPLETED
    assert state.last_sync_time is not None
    assert state.total_files_synced == 50
    assert state.total_bytes_synced == 100000
    assert state.total_commits_processed == 5
    assert state.consecutive_failures == 0  # Reset on success


def test_sync_state_mark_sync_failed():
    """Test marking sync as failed."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")

    state.mark_sync_failed(error_count=3)

    assert state.status == SyncStatus.FAILED
    assert state.sync_errors == 3
    assert state.consecutive_failures == 1


def test_sync_state_update_branch_state():
    """Test updating branch state."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")

    state.update_branch_state(
        branch_name="main",
        commit_sha="abc123",
        file_count=100,
        bytes_synced=50000,
        errors=0,
    )

    assert "main" in state.branch_states
    branch_state = state.branch_states["main"]
    assert branch_state.last_commit_sha == "abc123"
    assert branch_state.file_count == 100
    assert branch_state.total_bytes == 50000
    assert state.last_commit_sha == "abc123"  # Updated global SHA


def test_sync_state_is_healthy():
    """Test health check logic."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")

    # Healthy with no failures
    assert state.is_healthy() is True

    # Still healthy with few failures
    state.consecutive_failures = 3
    assert state.is_healthy() is True

    # Unhealthy with many failures
    state.consecutive_failures = 5
    assert state.is_healthy() is False


def test_sync_state_total_bytes_mb():
    """Test bytes to MB conversion."""
    state = SyncState(repo_id="test", sync_mode="graphql_api")
    state.total_bytes_synced = 10485760  # 10 MB

    assert state.total_bytes_mb == 10.0


# ---------------------------------------------------------------------------
# SyncResult Tests
# ---------------------------------------------------------------------------


def test_sync_result_basic():
    """Test SyncResult creation."""
    started = datetime.now(timezone.utc)
    completed = started + timedelta(seconds=30)

    result = SyncResult(
        repo_id="test-repo",
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        files_synced=100,
        files_skipped=10,
        files_failed=2,
        bytes_synced=500000,
        commits_processed=5,
        started_at=started,
        completed_at=completed,
        duration_seconds=30.0,
        branches_synced=["main"],
    )

    assert result.repo_id == "test-repo"
    assert result.sync_mode == "graphql_api"
    assert result.status == SyncStatus.COMPLETED
    assert result.files_synced == 100
    assert result.files_failed == 2
    assert result.duration_seconds == 30.0


def test_sync_result_bytes_synced_mb():
    """Test MB conversion property."""
    result = SyncResult(
        repo_id="test",
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        bytes_synced=10485760,  # 10 MB
        started_at=datetime.now(timezone.utc),
    )

    assert result.bytes_synced_mb == 10.0


def test_sync_result_success_rate():
    """Test success rate calculation."""
    result = SyncResult(
        repo_id="test",
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        files_synced=90,
        files_failed=10,
        started_at=datetime.now(timezone.utc),
    )

    assert result.success_rate == 90.0


def test_sync_result_is_success():
    """Test success check."""
    # Complete success
    success_result = SyncResult(
        repo_id="test",
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        files_synced=100,
        files_failed=0,
        started_at=datetime.now(timezone.utc),
    )
    assert success_result.is_success() is True

    # Partial failure
    partial_result = SyncResult(
        repo_id="test",
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        files_synced=90,
        files_failed=10,
        started_at=datetime.now(timezone.utc),
    )
    assert partial_result.is_success() is False


def test_sync_result_has_partial_success():
    """Test partial success check."""
    # Partial success (some files synced, some failed)
    partial_result = SyncResult(
        repo_id="test",
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        files_synced=90,
        files_failed=10,
        started_at=datetime.now(timezone.utc),
    )
    assert partial_result.has_partial_success() is True

    # Complete failure
    failure_result = SyncResult(
        repo_id="test",
        sync_mode="graphql_api",
        status=SyncStatus.FAILED,
        files_synced=0,
        files_failed=100,
        started_at=datetime.now(timezone.utc),
    )
    assert failure_result.has_partial_success() is False


# ---------------------------------------------------------------------------
# DiskSpaceEstimate Tests
# ---------------------------------------------------------------------------


def test_disk_space_estimate_basic():
    """Test DiskSpaceEstimate creation."""
    estimate = DiskSpaceEstimate(
        repo_id="test-repo",
        sync_mode="git_clone",
        estimated_size_bytes=1073741824,  # 1 GB
        available_space_bytes=10737418240,  # 10 GB
        is_sufficient=True,
        buffer_percentage=0.2,
    )

    assert estimate.repo_id == "test-repo"
    assert estimate.estimated_size_gb == 1.0
    assert estimate.available_space_gb == 10.0
    assert estimate.is_sufficient is True


def test_disk_space_estimate_conversions():
    """Test size conversion properties."""
    estimate = DiskSpaceEstimate(
        repo_id="test",
        sync_mode="git_clone",
        estimated_size_bytes=524288000,  # ~500 MB
        available_space_bytes=5368709120,  # ~5 GB
        is_sufficient=True,
    )

    assert 499.0 < estimate.estimated_size_mb < 501.0
    assert 4.9 < estimate.available_space_gb < 5.1


def test_disk_space_estimate_space_after_sync():
    """Test space calculation after sync."""
    estimate = DiskSpaceEstimate(
        repo_id="test",
        sync_mode="git_clone",
        estimated_size_bytes=1073741824,  # 1 GB
        available_space_bytes=5368709120,  # 5 GB
        is_sufficient=True,
    )

    space_after = estimate.space_after_sync_bytes()
    assert space_after == (5368709120 - 1073741824)

    space_after_gb = estimate.space_after_sync_gb()
    assert 3.9 < space_after_gb < 4.1
