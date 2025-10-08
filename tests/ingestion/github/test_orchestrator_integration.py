"""Tests for GitHub orchestrator integration.

This module tests the integration between GitHub Repository Connector
and the IngestionOrchestrator, including job execution, element delivery,
and health monitoring.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from futurnal.ingestion.github.orchestrator_integration import (
    GitHubConnectorManager,
    WebhookEvent,
)
from futurnal.ingestion.github.sync_models import SyncResult, SyncStatus
from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    SyncMode,
    VisibilityType,
)
from datetime import datetime, timezone


@pytest.fixture
def workspace_dir(tmp_path):
    """Create temporary workspace directory."""
    return tmp_path / "workspace"


@pytest.fixture
def mock_credential_manager():
    """Mock GitHub credential manager."""
    return Mock()


@pytest.fixture
def mock_api_client_manager():
    """Mock GitHub API client manager."""
    return Mock()


@pytest.fixture
def mock_element_sink():
    """Mock element sink."""
    sink = Mock()
    sink.handle = Mock()
    sink.handle_deletion = Mock()
    return sink


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger."""
    return Mock()


@pytest.fixture
def mock_consent_registry():
    """Mock consent registry."""
    return Mock()


@pytest.fixture
def manager(
    workspace_dir,
    mock_credential_manager,
    mock_api_client_manager,
    mock_element_sink,
    mock_audit_logger,
    mock_consent_registry,
):
    """Create GitHubConnectorManager instance."""
    return GitHubConnectorManager(
        workspace_dir=workspace_dir,
        credential_manager=mock_credential_manager,
        api_client_manager=mock_api_client_manager,
        element_sink=mock_element_sink,
        audit_logger=mock_audit_logger,
        consent_registry=mock_consent_registry,
    )


@pytest.fixture
def sample_descriptor():
    """Create sample repository descriptor."""
    return GitHubRepositoryDescriptor.from_registration(
        owner="testowner",
        repo="testrepo",
        credential_id="test-cred",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GRAPHQL_API,
    )


def test_manager_initialization(manager, workspace_dir):
    """Test manager initialization."""
    assert manager.workspace_dir == workspace_dir
    assert workspace_dir.exists()
    assert manager.registry is not None
    assert manager.state_manager is not None
    assert manager.sync_orchestrator is not None
    assert manager.quarantine_handler is not None


def test_manager_directories_created(manager, workspace_dir):
    """Test that required directories are created."""
    assert (workspace_dir / "sources" / "github").exists()
    assert (workspace_dir / "github").exists()
    assert (workspace_dir / "quarantine" / "github").exists()


def test_list_repositories_empty(manager):
    """Test listing repositories when none registered."""
    repos = manager.list_repositories()
    assert repos == []


def test_get_repository_not_found(manager):
    """Test getting nonexistent repository."""
    repo = manager.get_repository("nonexistent-id")
    assert repo is None


def test_get_sync_status_not_found(manager):
    """Test getting sync status for nonexistent repository."""
    status = manager.get_sync_status("nonexistent-id")
    assert status is None


@pytest.mark.asyncio
async def test_sync_repository_not_found(manager):
    """Test syncing nonexistent repository."""
    result = await manager.sync_repository(
        repo_id="nonexistent-id",
        job_id="test-job",
    )

    # Should return failed result
    assert result.status == SyncStatus.FAILED
    assert "not found" in result.error_message.lower()


@pytest.mark.asyncio
async def test_sync_repository_success(manager, sample_descriptor):
    """Test successful repository sync."""
    # Register repository
    manager.registry.add_or_update(sample_descriptor)

    # Mock successful sync
    mock_result = SyncResult(
        repo_id=sample_descriptor.id,
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        files_synced=10,
        bytes_synced=1024,
    )

    with patch.object(
        manager.sync_orchestrator,
        "sync_repository",
        new=AsyncMock(return_value=mock_result),
    ):
        result = await manager.sync_repository(
            repo_id=sample_descriptor.id,
            job_id="test-job",
        )

    assert result.status == SyncStatus.COMPLETED
    assert result.files_synced == 10
    assert result.bytes_synced == 1024


@pytest.mark.asyncio
async def test_sync_repository_with_element_sink(manager, sample_descriptor, mock_element_sink):
    """Test repository sync with element processing."""
    # Register repository
    manager.registry.add_or_update(sample_descriptor)

    # Mock successful sync with deletions
    mock_result = SyncResult(
        repo_id=sample_descriptor.id,
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        files_synced=5,
        bytes_synced=512,
        deleted_files=["old_file.py", "removed.txt"],
    )

    with patch.object(
        manager.sync_orchestrator,
        "sync_repository",
        new=AsyncMock(return_value=mock_result),
    ):
        result = await manager.sync_repository(
            repo_id=sample_descriptor.id,
            job_id="test-job",
        )

    # Verify element sink was called for deletions
    assert result.status == SyncStatus.COMPLETED
    assert mock_element_sink.handle_deletion.call_count == 2


@pytest.mark.asyncio
async def test_sync_repository_exception(manager, sample_descriptor):
    """Test repository sync with exception."""
    # Register repository
    manager.registry.add_or_update(sample_descriptor)

    # Mock sync exception
    with patch.object(
        manager.sync_orchestrator,
        "sync_repository",
        new=AsyncMock(side_effect=RuntimeError("Sync failed")),
    ):
        result = await manager.sync_repository(
            repo_id=sample_descriptor.id,
            job_id="test-job",
        )

    assert result.status == SyncStatus.FAILED
    assert "Sync failed" in result.error_message


@pytest.mark.asyncio
async def test_process_webhook_event_repo_not_found(manager):
    """Test processing webhook for unregistered repository."""
    event = WebhookEvent(
        event_id="evt-123",
        event_type="push",
        repository="unknown/repo",
        timestamp=datetime.now(timezone.utc),
        payload={},
    )

    # Should not raise exception
    await manager.process_webhook_event(event, job_id="test-job")


@pytest.mark.asyncio
async def test_process_webhook_event_push(manager, sample_descriptor):
    """Test processing push webhook event."""
    # Register repository
    manager.registry.add_or_update(sample_descriptor)

    event = WebhookEvent(
        event_id="evt-123",
        event_type="push",
        repository=sample_descriptor.full_name,
        timestamp=datetime.now(timezone.utc),
        payload={"ref": "refs/heads/main"},
    )

    # Mock successful sync
    mock_result = SyncResult(
        repo_id=sample_descriptor.id,
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )

    with patch.object(
        manager.sync_orchestrator,
        "sync_repository",
        new=AsyncMock(return_value=mock_result),
    ):
        await manager.process_webhook_event(event, job_id="test-job")

    # Verify sync was triggered
    # (In actual implementation, would verify sync_orchestrator.sync_repository was called)


@pytest.mark.asyncio
async def test_process_webhook_event_ignored_type(manager, sample_descriptor):
    """Test processing ignored webhook event type."""
    # Register repository
    manager.registry.add_or_update(sample_descriptor)

    event = WebhookEvent(
        event_id="evt-123",
        event_type="star",  # Ignored event type
        repository=sample_descriptor.full_name,
        timestamp=datetime.now(timezone.utc),
        payload={},
    )

    # Should not trigger sync
    with patch.object(
        manager,
        "sync_repository",
        new=AsyncMock(),
    ) as mock_sync:
        await manager.process_webhook_event(event, job_id="test-job")
        mock_sync.assert_not_called()


def test_get_health_status(manager):
    """Test health status generation."""
    health = manager.get_health_status()

    assert "status" in health
    assert "checks" in health
    assert "metrics" in health
    assert health["status"] in ["healthy", "degraded"]


def test_get_health_status_with_repositories(manager, sample_descriptor):
    """Test health status with registered repositories."""
    # Register repository
    manager.registry.add_or_update(sample_descriptor)

    health = manager.get_health_status()

    assert health["checks"]["repository_registry"] == "pass"
    assert health["metrics"]["registered_repositories"] == 1
    assert "quarantined_files" in health["metrics"]


def test_get_statistics_empty(manager):
    """Test statistics when empty."""
    stats = manager.get_statistics()

    assert stats["registered_repositories"] == 0
    assert stats["total_files_synced"] == 0
    assert stats["total_bytes_synced"] == 0
    assert "quarantine_statistics" in stats


def test_get_statistics_with_data(manager, sample_descriptor):
    """Test statistics with data."""
    # Register repository
    manager.registry.add_or_update(sample_descriptor)

    # Mock state manager to return state with data
    mock_state = Mock()
    mock_state.files_synced = 100
    mock_state.bytes_synced = 1024 * 1024
    mock_state.sync_errors = 2

    with patch.object(
        manager.state_manager,
        "load",
        return_value=mock_state,
    ):
        stats = manager.get_statistics()

    assert stats["registered_repositories"] == 1
    assert stats["total_files_synced"] == 100
    assert stats["total_bytes_synced"] == 1024 * 1024
    assert stats["total_bytes_synced_mb"] == 1.0
    assert stats["total_sync_errors"] == 2


def test_webhook_event_creation():
    """Test WebhookEvent creation."""
    event = WebhookEvent(
        event_id="test-event-123",
        event_type="push",
        repository="owner/repo",
        timestamp=datetime.now(timezone.utc),
        payload={"ref": "refs/heads/main", "commits": []},
    )

    assert event.event_id == "test-event-123"
    assert event.event_type == "push"
    assert event.repository == "owner/repo"
    assert isinstance(event.timestamp, datetime)
    assert "ref" in event.payload


@pytest.mark.asyncio
async def test_element_sink_deletion_error_handling(manager, sample_descriptor, mock_element_sink):
    """Test error handling in element deletion."""
    # Register repository
    manager.registry.add_or_update(sample_descriptor)

    # Mock element sink to raise exception
    mock_element_sink.handle_deletion.side_effect = RuntimeError("Deletion failed")

    # Mock successful sync with deletions
    mock_result = SyncResult(
        repo_id=sample_descriptor.id,
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        deleted_files=["file.py"],
    )

    with patch.object(
        manager.sync_orchestrator,
        "sync_repository",
        new=AsyncMock(return_value=mock_result),
    ):
        # Should not raise exception
        result = await manager.sync_repository(
            repo_id=sample_descriptor.id,
            job_id="test-job",
        )

    # Sync should still succeed despite deletion error
    assert result.status == SyncStatus.COMPLETED


def test_manager_component_initialization(manager):
    """Test that all manager components are properly initialized."""
    assert manager.credential_manager is not None
    assert manager.api_client_manager is not None
    assert manager.element_sink is not None
    assert manager.audit_logger is not None
    assert manager.consent_registry is not None
    assert manager.registry is not None
    assert manager.state_manager is not None
    assert manager.sync_orchestrator is not None
    assert manager.incremental_sync is not None
    assert manager.quarantine_handler is not None
    assert manager.issue_normalizer is not None
    assert manager.pr_normalizer is not None


def test_create_connector(manager, sample_descriptor):
    """Test connector creation."""
    connector = manager._create_connector(sample_descriptor)

    assert connector.descriptor == sample_descriptor
    assert connector.sync_orchestrator == manager.sync_orchestrator
    assert connector.element_sink == manager.element_sink
    assert connector.audit_logger == manager.audit_logger
    assert connector.consent_registry == manager.consent_registry
