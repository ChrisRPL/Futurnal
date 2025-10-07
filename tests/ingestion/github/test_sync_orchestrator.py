"""Tests for GitHub sync orchestrator."""

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from futurnal.ingestion.github.sync_orchestrator import GitHubSyncOrchestrator
from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    SyncMode,
    VisibilityType,
    Provenance,
)
from futurnal.ingestion.github.sync_models import (
    SyncResult,
    SyncStatus,
    SyncStrategy,
)
from futurnal.ingestion.github.sync_state_manager import SyncStateManager


@pytest.fixture
def temp_dirs():
    """Create temporary directories."""
    with tempfile.TemporaryDirectory() as state_dir, \
         tempfile.TemporaryDirectory() as clone_dir, \
         tempfile.TemporaryDirectory() as workspace_dir:
        yield {
            "state": Path(state_dir),
            "clone": Path(clone_dir),
            "workspace": Path(workspace_dir),
        }


@pytest.fixture
def mock_credential_manager():
    """Create mock credential manager."""
    return MagicMock()


@pytest.fixture
def mock_api_client_manager():
    """Create mock API client manager."""
    return MagicMock()


@pytest.fixture
def orchestrator(mock_credential_manager, mock_api_client_manager, temp_dirs):
    """Create orchestrator instance."""
    state_manager = SyncStateManager(state_dir=temp_dirs["state"])

    return GitHubSyncOrchestrator(
        credential_manager=mock_credential_manager,
        api_client_manager=mock_api_client_manager,
        state_manager=state_manager,
        clone_base_dir=temp_dirs["clone"],
        workspace_dir=temp_dirs["workspace"],
    )


@pytest.fixture
def test_descriptor():
    """Create test repository descriptor."""
    return GitHubRepositoryDescriptor(
        id="test-repo-id",
        owner="testowner",
        repo="testrepo",
        full_name="testowner/testrepo",
        visibility=VisibilityType.PUBLIC,
        credential_id="test-cred",
        sync_mode=SyncMode.GRAPHQL_API,
        provenance=Provenance(
            os_user="test",
            machine_id_hash="test123",
            tool_version="1.0.0",
        ),
    )


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


def test_orchestrator_initialization(orchestrator, temp_dirs):
    """Test orchestrator initialization."""
    assert orchestrator.credential_manager is not None
    assert orchestrator.api_client_manager is not None
    assert orchestrator.state_manager is not None
    assert orchestrator.clone_base_dir == temp_dirs["clone"]
    assert orchestrator.workspace_dir == temp_dirs["workspace"]


def test_orchestrator_default_directories():
    """Test orchestrator with default directories."""
    mock_cred = MagicMock()
    mock_api = MagicMock()

    orch = GitHubSyncOrchestrator(
        credential_manager=mock_cred,
        api_client_manager=mock_api,
    )

    assert orch.state_manager is not None
    assert orch.clone_base_dir.name == "github"
    assert orch.workspace_dir.name == "github"


# ---------------------------------------------------------------------------
# Mode Recommendation Tests
# ---------------------------------------------------------------------------


def test_recommend_sync_mode_large_repo(orchestrator):
    """Test recommending GraphQL for large repos."""
    mode = orchestrator.recommend_sync_mode(
        repo_size_kb=10000000,  # 10 GB
        file_count=1000,
        available_disk_gb=15.0,
    )

    assert mode == SyncMode.GRAPHQL_API  # Too large for disk


def test_recommend_sync_mode_small_repo(orchestrator):
    """Test recommending Git Clone for small repos."""
    mode = orchestrator.recommend_sync_mode(
        repo_size_kb=50000,  # 50 MB
        file_count=100,
        available_disk_gb=100.0,
    )

    assert mode == SyncMode.GIT_CLONE  # Small repo


def test_recommend_sync_mode_many_files(orchestrator):
    """Test recommending Git Clone for repos with many files."""
    mode = orchestrator.recommend_sync_mode(
        repo_size_kb=500000,  # 500 MB
        file_count=15000,  # Many files
        available_disk_gb=50.0,
    )

    assert mode == SyncMode.GIT_CLONE  # Many files = clone better


# ---------------------------------------------------------------------------
# Strategy Building Tests
# ---------------------------------------------------------------------------


def test_build_default_strategy(orchestrator, test_descriptor):
    """Test building default strategy from descriptor."""
    strategy = orchestrator._build_default_strategy(test_descriptor)

    assert isinstance(strategy, SyncStrategy)
    assert strategy.branches == ["main", "master"]
    assert strategy.batch_size == 10


def test_build_default_strategy_with_paths(orchestrator):
    """Test strategy building with custom paths."""
    descriptor = GitHubRepositoryDescriptor(
        id="test",
        owner="owner",
        repo="repo",
        full_name="owner/repo",
        visibility=VisibilityType.PUBLIC,
        credential_id="cred",
        include_paths=["src/*"],
        exclude_paths=["tests/*"],
        provenance=Provenance(
            os_user="test",
            machine_id_hash="test",
            tool_version="1.0.0",
        ),
    )

    strategy = orchestrator._build_default_strategy(descriptor)

    assert strategy.file_patterns == ["src/*"]
    assert strategy.exclude_patterns == ["tests/*"]


# ---------------------------------------------------------------------------
# Sync Tests with Mocking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_graphql_mode(orchestrator, test_descriptor):
    """Test syncing in GraphQL mode."""
    # Mock successful GraphQL sync
    mock_result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        started_at=datetime.now(timezone.utc),
    )

    with patch.object(orchestrator, "_sync_graphql", return_value=mock_result):
        result = await orchestrator.sync_repository(
            descriptor=test_descriptor,
            strategy=SyncStrategy(branches=["main"]),
        )

    assert result.status == SyncStatus.COMPLETED
    assert result.files_synced == 50


@pytest.mark.asyncio
async def test_sync_git_clone_mode(orchestrator):
    """Test syncing in Git Clone mode."""
    descriptor = GitHubRepositoryDescriptor(
        id="test",
        owner="owner",
        repo="repo",
        full_name="owner/repo",
        visibility=VisibilityType.PUBLIC,
        credential_id="cred",
        sync_mode=SyncMode.GIT_CLONE,
        provenance=Provenance(
            os_user="test",
            machine_id_hash="test",
            tool_version="1.0.0",
        ),
    )

    mock_result = SyncResult(
        repo_id=descriptor.id,
        sync_mode="git_clone",
        status=SyncStatus.COMPLETED,
        files_synced=100,
        bytes_synced=500000,
        started_at=datetime.now(timezone.utc),
    )

    with patch.object(orchestrator, "_sync_git_clone", return_value=mock_result):
        result = await orchestrator.sync_repository(
            descriptor=descriptor,
            strategy=SyncStrategy(branches=["main"]),
        )

    assert result.status == SyncStatus.COMPLETED
    assert result.files_synced == 100


@pytest.mark.asyncio
async def test_sync_with_force_mode(orchestrator, test_descriptor):
    """Test syncing with forced mode override."""
    mock_result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode="git_clone",
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        started_at=datetime.now(timezone.utc),
    )

    with patch.object(orchestrator, "_sync_git_clone", return_value=mock_result):
        # Descriptor has GRAPHQL_API but we force GIT_CLONE
        result = await orchestrator.sync_repository(
            descriptor=test_descriptor,
            strategy=SyncStrategy(branches=["main"]),
            force_mode=SyncMode.GIT_CLONE,
        )

    assert result.sync_mode == "git_clone"


# ---------------------------------------------------------------------------
# State Management Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_updates_state_on_success(orchestrator, test_descriptor):
    """Test that successful sync updates state."""
    mock_result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        commits_processed=5,
        started_at=datetime.now(timezone.utc),
    )

    with patch.object(orchestrator, "_sync_graphql", return_value=mock_result):
        await orchestrator.sync_repository(
            descriptor=test_descriptor,
            strategy=SyncStrategy(branches=["main"]),
        )

    # Check state was updated
    state = orchestrator.state_manager.load(test_descriptor.id)
    assert state is not None
    assert state.status == SyncStatus.COMPLETED
    assert state.total_files_synced == 50


@pytest.mark.asyncio
async def test_sync_updates_state_on_failure(orchestrator, test_descriptor):
    """Test that failed sync updates state."""
    mock_result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode="graphql_api",
        status=SyncStatus.FAILED,
        files_failed=10,
        error_message="Test error",
        started_at=datetime.now(timezone.utc),
    )

    with patch.object(orchestrator, "_sync_graphql", return_value=mock_result):
        await orchestrator.sync_repository(
            descriptor=test_descriptor,
            strategy=SyncStrategy(branches=["main"]),
        )

    # Check state was updated
    state = orchestrator.state_manager.load(test_descriptor.id)
    assert state is not None
    assert state.status == SyncStatus.FAILED
    assert state.consecutive_failures == 1


# ---------------------------------------------------------------------------
# Query Methods Tests
# ---------------------------------------------------------------------------


def test_get_sync_status(orchestrator, test_descriptor):
    """Test getting sync status."""
    # Create state
    from futurnal.ingestion.github.sync_models import SyncState

    state = SyncState(
        repo_id=test_descriptor.id,
        sync_mode="graphql_api",
    )
    orchestrator.state_manager.save(state)

    # Query status
    retrieved = orchestrator.get_sync_status(test_descriptor.id)

    assert retrieved is not None
    assert retrieved.repo_id == test_descriptor.id


def test_list_all_syncs(orchestrator):
    """Test listing all syncs."""
    from futurnal.ingestion.github.sync_models import SyncState

    # Create multiple states
    for i in range(3):
        state = SyncState(repo_id=f"repo-{i}", sync_mode="graphql_api")
        orchestrator.state_manager.save(state)

    syncs = orchestrator.list_all_syncs()

    assert len(syncs) == 3


def test_cleanup_old_syncs(orchestrator):
    """Test cleaning up old syncs."""
    from futurnal.ingestion.github.sync_models import SyncState
    import time
    import os

    # Create old state
    state = SyncState(repo_id="old", sync_mode="graphql_api")
    orchestrator.state_manager.save(state)

    # Make it old
    state_path = orchestrator.state_manager._state_path("old")
    old_time = time.time() - (100 * 86400)
    os.utime(state_path, (old_time, old_time))

    cleaned = orchestrator.cleanup_old_syncs(days=90)

    assert cleaned >= 1
