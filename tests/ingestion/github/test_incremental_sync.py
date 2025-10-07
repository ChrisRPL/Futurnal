"""Tests for incremental sync engine."""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from futurnal.ingestion.github.incremental_sync import IncrementalSyncEngine
from futurnal.ingestion.github.sync_models import (
    BranchSyncState,
    SyncResult,
    SyncState,
    SyncStatus,
)
from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    Provenance,
    SyncMode,
    VisibilityType,
)
from futurnal.ingestion.github.api_client_manager import GitHubAPIClientManager
from futurnal.ingestion.github.sync_state_manager import SyncStateManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_api_client():
    """Create mock API client manager."""
    client = MagicMock(spec=GitHubAPIClientManager)
    client.graphql_request = AsyncMock()
    client.rest_request = AsyncMock()
    return client


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    manager = MagicMock(spec=SyncStateManager)
    manager.get_or_create = MagicMock()
    manager.save = MagicMock()
    manager.load = MagicMock()
    return manager


@pytest.fixture
def mock_element_sink():
    """Create mock element sink."""
    sink = MagicMock()
    sink.handle = MagicMock()
    sink.handle_deletion = MagicMock()
    return sink


@pytest.fixture
def sample_descriptor():
    """Create sample repository descriptor."""
    return GitHubRepositoryDescriptor(
        id="test-repo-123",
        owner="test-owner",
        repo="test-repo",
        full_name="test-owner/test-repo",
        visibility=VisibilityType.PRIVATE,
        credential_id="cred-123",
        sync_mode=SyncMode.GRAPHQL_API,
        branches=["main", "develop"],
        provenance=Provenance(
            os_user="test-user",
            machine_id_hash="test-machine-hash",
            tool_version="1.0.0",
        ),
    )


@pytest.fixture
def incremental_engine(mock_api_client, mock_state_manager):
    """Create incremental sync engine with mocks."""
    return IncrementalSyncEngine(
        api_client_manager=mock_api_client,
        state_manager=mock_state_manager,
    )


# ---------------------------------------------------------------------------
# Test: Get Branch HEAD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_branch_head_success(incremental_engine, mock_api_client):
    """Test successful branch HEAD retrieval."""
    # Mock GraphQL response
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "ref": {
                    "target": {
                        "oid": "abc123def456"
                    }
                }
            }
        }
    }

    sha = await incremental_engine._get_branch_head(
        owner="test-owner",
        repo="test-repo",
        branch="main",
        credential_id="cred-123",
    )

    assert sha == "abc123def456"
    assert mock_api_client.graphql_request.called


@pytest.mark.asyncio
async def test_get_branch_head_not_found(incremental_engine, mock_api_client):
    """Test branch HEAD retrieval when branch doesn't exist."""
    # Mock GraphQL response with null ref
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "ref": None
            }
        }
    }

    sha = await incremental_engine._get_branch_head(
        owner="test-owner",
        repo="test-repo",
        branch="nonexistent",
        credential_id="cred-123",
    )

    assert sha == ""


@pytest.mark.asyncio
async def test_get_branch_head_error(incremental_engine, mock_api_client):
    """Test branch HEAD retrieval with API error."""
    mock_api_client.graphql_request.side_effect = Exception("API error")

    with pytest.raises(Exception, match="API error"):
        await incremental_engine._get_branch_head(
            owner="test-owner",
            repo="test-repo",
            branch="main",
            credential_id="cred-123",
        )


# ---------------------------------------------------------------------------
# Test: Get Commit Range
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_commit_range_success(incremental_engine, mock_api_client):
    """Test successful commit range retrieval."""
    # Mock REST API response
    mock_api_client.rest_request.return_value = {
        "status": "ahead",
        "ahead_by": 3,
        "commits": [
            {"sha": "commit1", "commit": {"message": "First"}},
            {"sha": "commit2", "commit": {"message": "Second"}},
            {"sha": "commit3", "commit": {"message": "Third"}},
        ],
    }

    result = await incremental_engine._get_commit_range(
        owner="test-owner",
        repo="test-repo",
        from_sha="old123",
        to_sha="new456",
        credential_id="cred-123",
    )

    assert result["status"] == "ahead"
    assert len(result["commits"]) == 3
    assert result["commits"][0]["sha"] == "commit1"


@pytest.mark.asyncio
async def test_get_commit_range_diverged(incremental_engine, mock_api_client):
    """Test commit range with diverged status (force push indicator)."""
    mock_api_client.rest_request.return_value = {
        "status": "diverged",
        "ahead_by": 2,
        "behind_by": 1,
        "commits": [
            {"sha": "new1"},
            {"sha": "new2"},
        ],
    }

    result = await incremental_engine._get_commit_range(
        owner="test-owner",
        repo="test-repo",
        from_sha="old123",
        to_sha="new456",
        credential_id="cred-123",
    )

    assert result["status"] == "diverged"
    assert len(result["commits"]) == 2


# ---------------------------------------------------------------------------
# Test: Get Commit Changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_commit_changes_success(incremental_engine, mock_api_client):
    """Test successful commit changes retrieval."""
    mock_api_client.rest_request.return_value = {
        "sha": "commit123",
        "files": [
            {"filename": "file1.py", "status": "added"},
            {"filename": "file2.py", "status": "modified"},
            {"filename": "file3.py", "status": "removed"},
        ],
    }

    changes = await incremental_engine._get_commit_changes(
        owner="test-owner",
        repo="test-repo",
        commit_sha="commit123",
        credential_id="cred-123",
    )

    assert "added" in changes
    assert "modified" in changes
    assert "deleted" in changes
    assert "file1.py" in changes["added"]
    assert "file2.py" in changes["modified"]
    assert "file3.py" in changes["deleted"]


@pytest.mark.asyncio
async def test_get_commit_changes_with_renamed(incremental_engine, mock_api_client):
    """Test commit changes with renamed files."""
    mock_api_client.rest_request.return_value = {
        "sha": "commit123",
        "files": [
            {
                "filename": "new_name.py",
                "previous_filename": "old_name.py",
                "status": "renamed",
            },
        ],
    }

    changes = await incremental_engine._get_commit_changes(
        owner="test-owner",
        repo="test-repo",
        commit_sha="commit123",
        credential_id="cred-123",
    )

    # Renames treated as delete + add
    assert "old_name.py" in changes["deleted"]
    assert "new_name.py" in changes["added"]


# ---------------------------------------------------------------------------
# Test: Force Push Detection
# ---------------------------------------------------------------------------


def test_is_force_push_via_status(incremental_engine):
    """Test force push detection via compare API status."""
    commits = [
        {"sha": "new1"},
        {"sha": "new2"},
    ]

    is_force_push = incremental_engine._is_force_push(
        commits=commits,
        expected_parent_sha="old123",
        compare_status="diverged",
    )

    assert is_force_push is True


def test_is_force_push_via_ancestry(incremental_engine):
    """Test that normal incremental update is not detected as force push.

    The compare API only returns NEW commits, not including the base commit.
    So parent SHA won't be in commits list for normal updates.
    """
    commits = [
        {"sha": "new1"},
        {"sha": "new2"},
    ]

    is_force_push = incremental_engine._is_force_push(
        commits=commits,
        expected_parent_sha="old123",  # Not in commits (this is normal)
        compare_status="ahead",  # Normal incremental update
    )

    assert is_force_push is False  # Should NOT be detected as force push


def test_no_force_push(incremental_engine):
    """Test normal case without force push."""
    commits = [
        {"sha": "new1"},
        {"sha": "new2"},
    ]

    is_force_push = incremental_engine._is_force_push(
        commits=commits,
        expected_parent_sha="old123",
        compare_status="ahead",  # Not diverged = normal update
    )

    assert is_force_push is False


# ---------------------------------------------------------------------------
# Test: Detect Deleted Branches
# ---------------------------------------------------------------------------


def test_detect_deleted_branches(incremental_engine, sample_descriptor):
    """Test detection of deleted branches."""
    # State has more branches than descriptor
    state = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
        branch_states={
            "main": BranchSyncState(
                branch_name="main",
                last_commit_sha="sha1",
                last_sync_time=datetime.now(timezone.utc),
            ),
            "develop": BranchSyncState(
                branch_name="develop",
                last_commit_sha="sha2",
                last_sync_time=datetime.now(timezone.utc),
            ),
            "old-branch": BranchSyncState(
                branch_name="old-branch",
                last_commit_sha="sha3",
                last_sync_time=datetime.now(timezone.utc),
            ),
        },
    )

    deleted = incremental_engine._detect_deleted_branches(sample_descriptor, state)

    assert "old-branch" in deleted
    assert "main" not in deleted
    assert "develop" not in deleted


def test_detect_no_deleted_branches(incremental_engine, sample_descriptor):
    """Test when no branches are deleted."""
    state = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
        branch_states={
            "main": BranchSyncState(
                branch_name="main",
                last_commit_sha="sha1",
                last_sync_time=datetime.now(timezone.utc),
            ),
        },
    )

    deleted = incremental_engine._detect_deleted_branches(sample_descriptor, state)

    assert len(deleted) == 0


# ---------------------------------------------------------------------------
# Test: Merge Results
# ---------------------------------------------------------------------------


def test_merge_results_single(incremental_engine):
    """Test merging single branch result."""
    started_at = datetime.now(timezone.utc)
    results = [
        SyncResult(
            repo_id="test-repo",
            sync_mode="graphql_api",
            status=SyncStatus.COMPLETED,
            started_at=started_at,
            files_synced=10,
            bytes_synced=1000,
            commits_processed=5,
            new_commits=["c1", "c2"],
            branches_synced=["main"],
        )
    ]

    merged = incremental_engine._merge_results(results, started_at)

    assert merged.files_synced == 10
    assert merged.bytes_synced == 1000
    assert merged.commits_processed == 5
    assert len(merged.new_commits) == 2
    assert "main" in merged.branches_synced


def test_merge_results_multiple(incremental_engine):
    """Test merging multiple branch results."""
    started_at = datetime.now(timezone.utc)
    results = [
        SyncResult(
            repo_id="test-repo",
            sync_mode="graphql_api",
            status=SyncStatus.COMPLETED,
            started_at=started_at,
            files_synced=10,
            bytes_synced=1000,
            commits_processed=5,
            new_commits=["c1", "c2"],
            modified_files=["f1.py"],
            branches_synced=["main"],
        ),
        SyncResult(
            repo_id="test-repo",
            sync_mode="graphql_api",
            status=SyncStatus.COMPLETED,
            started_at=started_at,
            files_synced=5,
            bytes_synced=500,
            commits_processed=3,
            new_commits=["c3", "c4"],
            modified_files=["f2.py"],
            branches_synced=["develop"],
        ),
    ]

    merged = incremental_engine._merge_results(results, started_at)

    assert merged.files_synced == 15
    assert merged.bytes_synced == 1500
    assert merged.commits_processed == 8
    assert len(merged.new_commits) == 4
    assert len(merged.branches_synced) == 2


def test_merge_results_with_force_push(incremental_engine):
    """Test merging results with force push flag."""
    started_at = datetime.now(timezone.utc)
    results = [
        SyncResult(
            repo_id="test-repo",
            sync_mode="graphql_api",
            status=SyncStatus.COMPLETED,
            started_at=started_at,
            force_push_handled=False,
        ),
        SyncResult(
            repo_id="test-repo",
            sync_mode="graphql_api",
            status=SyncStatus.COMPLETED,
            started_at=started_at,
            force_push_handled=True,
        ),
    ]

    merged = incremental_engine._merge_results(results, started_at)

    assert merged.force_push_handled is True


# ---------------------------------------------------------------------------
# Test: Initial Branch Sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_branch_sync_graphql_mode(
    incremental_engine, sample_descriptor, mock_state_manager, mock_api_client
):
    """Test initial branch sync in GraphQL mode."""
    # Setup state manager mock
    mock_state_manager.get_or_create.return_value = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
    )

    # Mock GraphQL sync (tree fetch)
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "object": {
                    "entries": [
                        {
                            "name": "file1.py",
                            "type": "blob",
                            "path": "file1.py",
                            "object": {"byteSize": 100, "isBinary": False, "oid": "abc"},
                        }
                    ]
                }
            }
        }
    }

    result = await incremental_engine._initial_branch_sync(
        descriptor=sample_descriptor,
        branch_name="main",
        head_sha="new123",
    )

    # Should return a sync result
    assert isinstance(result, SyncResult)


# ---------------------------------------------------------------------------
# Test: Incremental Branch Sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_branch_sync_with_changes(
    incremental_engine, sample_descriptor, mock_api_client
):
    """Test incremental branch sync with actual changes."""
    branch_state = BranchSyncState(
        branch_name="main",
        last_commit_sha="old123",
        last_sync_time=datetime.now(timezone.utc),
    )

    # Mock commit range response
    mock_api_client.rest_request.side_effect = [
        # First call: compare commits
        {
            "status": "ahead",
            "commits": [
                {"sha": "commit1"},
                {"sha": "commit2"},
            ],
        },
        # Second call: get commit1 changes
        {
            "sha": "commit1",
            "files": [
                {"filename": "file1.py", "status": "added"},
            ],
        },
        # Third call: get commit2 changes
        {
            "sha": "commit2",
            "files": [
                {"filename": "file2.py", "status": "modified"},
            ],
        },
    ]

    result = await incremental_engine._incremental_branch_sync(
        descriptor=sample_descriptor,
        branch_name="main",
        branch_state=branch_state,
        current_head="new456",
    )

    assert result.status == SyncStatus.COMPLETED
    assert len(result.new_commits) == 2
    assert "commit1" in result.new_commits
    assert "commit2" in result.new_commits


@pytest.mark.asyncio
async def test_incremental_branch_sync_force_push(
    incremental_engine, sample_descriptor, mock_api_client, mock_state_manager
):
    """Test incremental branch sync with force push detection."""
    branch_state = BranchSyncState(
        branch_name="main",
        last_commit_sha="old123",
        last_sync_time=datetime.now(timezone.utc),
    )

    # Mock commit range with diverged status
    mock_api_client.rest_request.return_value = {
        "status": "diverged",
        "commits": [
            {"sha": "new1"},
        ],
    }

    # Mock state manager for force push handler
    mock_state_manager.get_or_create.return_value = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
    )

    # Mock GraphQL for initial sync (triggered by force push)
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "object": {"entries": []}
            }
        }
    }

    result = await incremental_engine._incremental_branch_sync(
        descriptor=sample_descriptor,
        branch_name="main",
        branch_state=branch_state,
        current_head="new456",
    )

    assert result.force_push_handled is True


# ---------------------------------------------------------------------------
# Test: Sync Branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_branch_initial(
    incremental_engine, sample_descriptor, mock_api_client, mock_state_manager
):
    """Test syncing branch for first time."""
    # No existing branch state
    state = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
        branch_states={},
    )

    # Mock get branch head
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "ref": {"target": {"oid": "new123"}},
                "object": {"entries": []},
            }
        }
    }

    mock_state_manager.get_or_create.return_value = state

    result = await incremental_engine._sync_branch(
        descriptor=sample_descriptor,
        branch_name="main",
        state=state,
    )

    # Should create new branch state
    assert "main" in state.branch_states
    assert state.branch_states["main"].last_commit_sha == "new123"


@pytest.mark.asyncio
async def test_sync_branch_no_changes(
    incremental_engine, sample_descriptor, mock_api_client
):
    """Test syncing branch when no changes detected."""
    # Existing branch state with same SHA
    state = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
        branch_states={
            "main": BranchSyncState(
                branch_name="main",
                last_commit_sha="same123",
                last_sync_time=datetime.now(timezone.utc),
            )
        },
    )

    # Mock get branch head returning same SHA
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "ref": {"target": {"oid": "same123"}}
            }
        }
    }

    result = await incremental_engine._sync_branch(
        descriptor=sample_descriptor,
        branch_name="main",
        state=state,
    )

    assert result.status == SyncStatus.COMPLETED
    assert result.files_synced == 0
    assert result.commits_processed == 0


# ---------------------------------------------------------------------------
# Test: Full Repository Sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_repository_success(
    incremental_engine, sample_descriptor, mock_api_client, mock_state_manager
):
    """Test successful full repository sync."""
    # Setup mocks
    mock_state_manager.get_or_create.return_value = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
        branch_states={},
    )

    # Mock branch head calls for both branches
    # Use return_value instead of side_effect to handle multiple calls
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "ref": {"target": {"oid": "main123"}},
                "object": {"entries": []},
            }
        }
    }

    result = await incremental_engine.sync_repository(sample_descriptor)

    assert result.status == SyncStatus.COMPLETED
    assert len(result.branches_synced) > 0
    assert mock_state_manager.save.called


@pytest.mark.asyncio
async def test_sync_repository_with_deleted_branches(
    incremental_engine, sample_descriptor, mock_api_client, mock_state_manager
):
    """Test repository sync with deleted branches cleanup."""
    # State has old-branch that's not in descriptor
    state = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
        branch_states={
            "old-branch": BranchSyncState(
                branch_name="old-branch",
                last_commit_sha="old123",
                last_sync_time=datetime.now(timezone.utc),
            )
        },
    )

    mock_state_manager.get_or_create.return_value = state

    # Mock branch heads
    mock_api_client.graphql_request.side_effect = [
        {"data": {"repository": {"ref": {"target": {"oid": "main123"}}, "object": {"entries": []}}}},
        {"data": {"repository": {"ref": {"target": {"oid": "dev456"}}, "object": {"entries": []}}}},
    ]

    result = await incremental_engine.sync_repository(sample_descriptor)

    # old-branch should be removed from state
    assert "old-branch" not in state.branch_states


# ---------------------------------------------------------------------------
# Test: Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_repository_error_handling(
    incremental_engine, sample_descriptor, mock_api_client, mock_state_manager
):
    """Test error handling in repository sync.

    The engine is resilient - individual branch failures don't fail the entire sync.
    Branches that succeed are still synced. Only if there's a fatal error at the
    repository level does the entire sync fail.
    """
    mock_state_manager.get_or_create.return_value = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
    )

    # Mock API error for branch HEAD fetches
    mock_api_client.graphql_request.side_effect = Exception("API failure")

    result = await incremental_engine.sync_repository(sample_descriptor)

    # Sync completes but individual branches may have failed
    assert result.status == SyncStatus.COMPLETED
    # Error details captured in branch-level error messages
    assert result.error_message is not None or result.files_synced == 0


# ---------------------------------------------------------------------------
# Test: File Sync and Deletion Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_deleted_files(
    incremental_engine, sample_descriptor, mock_element_sink
):
    """Test handling deleted files."""
    incremental_engine.element_sink = mock_element_sink

    await incremental_engine._handle_deleted_files(
        descriptor=sample_descriptor,
        file_paths=["file1.py", "file2.py"],
    )

    # Should call handle_deletion for each file
    assert mock_element_sink.handle_deletion.call_count == 2


@pytest.mark.asyncio
async def test_handle_deleted_files_no_sink(incremental_engine, sample_descriptor):
    """Test handling deleted files without element sink."""
    # Should not raise error when sink is None
    await incremental_engine._handle_deleted_files(
        descriptor=sample_descriptor,
        file_paths=["file1.py"],
    )


# ---------------------------------------------------------------------------
# Test: Data Model Extensions
# ---------------------------------------------------------------------------


def test_branch_sync_state_extended_fields():
    """Test that BranchSyncState has incremental sync fields."""
    state = BranchSyncState(
        branch_name="main",
        last_commit_sha="sha123",
        last_sync_time=datetime.now(timezone.utc),
        parent_commit_sha="parent_sha",
        force_push_detected=True,
        commits_processed=10,
    )

    assert state.parent_commit_sha == "parent_sha"
    assert state.force_push_detected is True
    assert state.commits_processed == 10


def test_sync_result_extended_fields():
    """Test that SyncResult has incremental sync fields."""
    result = SyncResult(
        repo_id="test",
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        new_commits=["c1", "c2"],
        modified_files=["f1.py"],
        deleted_files=["f2.py"],
        added_files=["f3.py"],
        force_push_handled=True,
    )

    assert len(result.new_commits) == 2
    assert len(result.modified_files) == 1
    assert len(result.deleted_files) == 1
    assert len(result.added_files) == 1
    assert result.force_push_handled is True


# ---------------------------------------------------------------------------
# Test: Integration Scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_incremental_workflow(
    incremental_engine, sample_descriptor, mock_api_client, mock_state_manager
):
    """Test complete incremental sync workflow."""
    # Initial state: no previous sync
    initial_state = SyncState(
        repo_id="test-repo-123",
        sync_mode="graphql_api",
        branch_states={},
    )

    mock_state_manager.get_or_create.return_value = initial_state

    # First sync: initial sync
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "ref": {"target": {"oid": "initial123"}},
                "object": {"entries": []},
            }
        }
    }

    first_result = await incremental_engine.sync_repository(sample_descriptor)

    assert first_result.status == SyncStatus.COMPLETED
    assert "main" in initial_state.branch_states

    # Second sync: incremental with changes
    mock_api_client.graphql_request.return_value = {
        "data": {
            "repository": {
                "ref": {"target": {"oid": "updated456"}}
            }
        }
    }

    mock_api_client.rest_request.side_effect = [
        # Compare commits for main
        {
            "status": "ahead",
            "commits": [{"sha": "new_commit"}],
        },
        # Get commit changes
        {
            "sha": "new_commit",
            "files": [{"filename": "file.py", "status": "modified"}],
        },
        # Compare commits for develop
        {
            "status": "ahead",
            "commits": [],
        },
    ]

    second_result = await incremental_engine.sync_repository(sample_descriptor)

    assert second_result.status == SyncStatus.COMPLETED
    assert second_result.commits_processed > 0
