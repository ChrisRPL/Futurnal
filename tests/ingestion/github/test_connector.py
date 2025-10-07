"""Tests for GitHub Repository Connector."""

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from futurnal.ingestion.github.connector import (
    ElementSink,
    GitHubRepositoryConnector,
)
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
from futurnal.ingestion.github.sync_orchestrator import GitHubSyncOrchestrator
from futurnal.privacy.audit import AuditEvent, AuditLogger
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_sync_orchestrator():
    """Create mock sync orchestrator."""
    orchestrator = MagicMock(spec=GitHubSyncOrchestrator)
    orchestrator.sync_repository = AsyncMock()
    orchestrator.get_sync_status = MagicMock(return_value=None)
    orchestrator.state_manager = MagicMock()
    orchestrator.state_manager.get_statistics = MagicMock(
        return_value={"total_syncs": 0}
    )
    return orchestrator


@pytest.fixture
def mock_element_sink():
    """Create mock element sink."""
    sink = MagicMock(spec=ElementSink)
    sink.handle = MagicMock()
    sink.handle_deletion = MagicMock()
    return sink


@pytest.fixture
def mock_audit_logger():
    """Create mock audit logger."""
    logger = MagicMock(spec=AuditLogger)
    logger.record = MagicMock()
    return logger


@pytest.fixture
def mock_consent_registry():
    """Create mock consent registry."""
    registry = MagicMock(spec=ConsentRegistry)
    registry.has_consent = MagicMock(return_value=True)
    return registry


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


@pytest.fixture
def test_strategy():
    """Create test sync strategy."""
    return SyncStrategy(
        branches=["main"],
        file_patterns=[],
        exclude_patterns=[".git/", "*.pyc"],
        max_file_size_mb=10,
        fetch_file_content=True,
        batch_size=10,
    )


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


def test_connector_initialization(test_descriptor, mock_sync_orchestrator):
    """Test connector initialization."""
    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
    )

    assert connector.descriptor == test_descriptor
    assert connector.sync_orchestrator == mock_sync_orchestrator
    assert connector.element_sink is None
    assert connector.audit_logger is None
    assert connector.consent_registry is None


def test_connector_with_all_components(
    test_descriptor,
    mock_sync_orchestrator,
    mock_element_sink,
    mock_audit_logger,
    mock_consent_registry,
):
    """Test connector initialization with all components."""
    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
        element_sink=mock_element_sink,
        audit_logger=mock_audit_logger,
        consent_registry=mock_consent_registry,
    )

    assert connector.element_sink == mock_element_sink
    assert connector.audit_logger == mock_audit_logger
    assert connector.consent_registry == mock_consent_registry


def test_connector_validation_missing_descriptor(mock_sync_orchestrator):
    """Test connector validation fails without descriptor."""
    with pytest.raises(ValueError, match="Repository descriptor is required"):
        GitHubRepositoryConnector(
            descriptor=None,
            sync_orchestrator=mock_sync_orchestrator,
        )


def test_connector_validation_missing_orchestrator(test_descriptor):
    """Test connector validation fails without orchestrator."""
    with pytest.raises(ValueError, match="Sync orchestrator is required"):
        GitHubRepositoryConnector(
            descriptor=test_descriptor,
            sync_orchestrator=None,
        )


# ---------------------------------------------------------------------------
# Sync Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_basic(
    test_descriptor,
    mock_sync_orchestrator,
    test_strategy,
):
    """Test basic sync operation."""
    # Mock successful sync result
    result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode=test_descriptor.sync_mode.value,
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        started_at=datetime.now(timezone.utc),
        duration_seconds=10.5,
    )
    mock_sync_orchestrator.sync_repository.return_value = result

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
    )

    sync_result = await connector.sync(strategy=test_strategy)

    assert sync_result.status == SyncStatus.COMPLETED
    assert sync_result.files_synced == 50
    assert sync_result.bytes_synced == 100000
    mock_sync_orchestrator.sync_repository.assert_called_once()


@pytest.mark.asyncio
async def test_sync_with_audit_logging(
    test_descriptor,
    mock_sync_orchestrator,
    mock_audit_logger,
    test_strategy,
):
    """Test sync with audit logging."""
    result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode=test_descriptor.sync_mode.value,
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        started_at=datetime.now(timezone.utc),
        duration_seconds=10.5,
    )
    mock_sync_orchestrator.sync_repository.return_value = result

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
        audit_logger=mock_audit_logger,
    )

    await connector.sync(strategy=test_strategy)

    # Should emit 2 audit events: start and completion
    assert mock_audit_logger.record.call_count == 2

    # Check start event
    start_event = mock_audit_logger.record.call_args_list[0][0][0]
    assert start_event.action == "repo_sync_started"
    assert start_event.status == "info"

    # Check completion event
    completion_event = mock_audit_logger.record.call_args_list[1][0][0]
    assert completion_event.action == "repo_sync_completed"
    assert completion_event.status == "success"
    assert completion_event.metadata["files_synced"] == 50


@pytest.mark.asyncio
async def test_sync_with_consent_check(
    test_descriptor,
    mock_sync_orchestrator,
    mock_consent_registry,
    test_strategy,
):
    """Test sync with consent checking."""
    result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode=test_descriptor.sync_mode.value,
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        started_at=datetime.now(timezone.utc),
    )
    mock_sync_orchestrator.sync_repository.return_value = result

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
        consent_registry=mock_consent_registry,
    )

    await connector.sync(strategy=test_strategy)

    # Should check consent for required scopes
    assert mock_consent_registry.has_consent.called


@pytest.mark.asyncio
async def test_sync_consent_not_granted(
    test_descriptor,
    mock_sync_orchestrator,
    mock_consent_registry,
    test_strategy,
):
    """Test sync fails when consent not granted."""
    mock_consent_registry.has_consent.return_value = False

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
        consent_registry=mock_consent_registry,
    )

    with pytest.raises(ConsentRequiredError):
        await connector.sync(strategy=test_strategy)


@pytest.mark.asyncio
async def test_sync_with_element_sink(
    test_descriptor,
    mock_sync_orchestrator,
    mock_element_sink,
    test_strategy,
):
    """Test sync with element sink processing."""
    result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode=test_descriptor.sync_mode.value,
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        started_at=datetime.now(timezone.utc),
    )
    mock_sync_orchestrator.sync_repository.return_value = result

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
        element_sink=mock_element_sink,
    )

    await connector.sync(strategy=test_strategy)

    # ElementSink processing is called for successful syncs
    # (Note: actual file processing logic is a placeholder in connector.py)


@pytest.mark.asyncio
async def test_sync_with_custom_job_id(
    test_descriptor,
    mock_sync_orchestrator,
    test_strategy,
):
    """Test sync with custom job ID."""
    result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode=test_descriptor.sync_mode.value,
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        started_at=datetime.now(timezone.utc),
    )
    mock_sync_orchestrator.sync_repository.return_value = result

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
    )

    custom_job_id = "custom-job-123"
    await connector.sync(strategy=test_strategy, job_id=custom_job_id)

    # Job ID is used internally
    mock_sync_orchestrator.sync_repository.assert_called_once()


@pytest.mark.asyncio
async def test_sync_failure(
    test_descriptor,
    mock_sync_orchestrator,
    mock_audit_logger,
    test_strategy,
):
    """Test sync failure handling."""
    mock_sync_orchestrator.sync_repository.side_effect = Exception("Sync failed")

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
        audit_logger=mock_audit_logger,
    )

    with pytest.raises(Exception, match="Sync failed"):
        await connector.sync(strategy=test_strategy)

    # Should emit failure audit event
    assert mock_audit_logger.record.call_count == 2  # start + failure

    failure_event = mock_audit_logger.record.call_args_list[1][0][0]
    assert failure_event.action == "repo_sync_failed"
    assert failure_event.status == "error"
    assert "error" in failure_event.metadata


# ---------------------------------------------------------------------------
# Status and Statistics Tests
# ---------------------------------------------------------------------------


def test_get_sync_status(test_descriptor, mock_sync_orchestrator):
    """Test getting sync status."""
    from futurnal.ingestion.github.sync_models import SyncState

    mock_state = SyncState(
        repo_id=test_descriptor.id,
        sync_mode=test_descriptor.sync_mode.value,
    )
    mock_sync_orchestrator.get_sync_status.return_value = mock_state

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
    )

    status = connector.get_sync_status()

    assert status == mock_state
    mock_sync_orchestrator.get_sync_status.assert_called_once_with(test_descriptor.id)


def test_get_sync_status_none(test_descriptor, mock_sync_orchestrator):
    """Test getting sync status when none exists."""
    mock_sync_orchestrator.get_sync_status.return_value = None

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
    )

    status = connector.get_sync_status()

    assert status is None


def test_get_statistics(test_descriptor, mock_sync_orchestrator):
    """Test getting sync statistics."""
    mock_stats = {
        "total_syncs": 10,
        "successful_syncs": 8,
        "failed_syncs": 2,
        "total_files_synced": 500,
        "total_bytes_synced": 1000000,
    }
    mock_sync_orchestrator.state_manager.get_statistics.return_value = mock_stats

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
    )

    stats = connector.get_statistics()

    assert stats == mock_stats
    mock_sync_orchestrator.state_manager.get_statistics.assert_called_once_with(
        test_descriptor.id
    )


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_sync_workflow(
    test_descriptor,
    mock_sync_orchestrator,
    mock_element_sink,
    mock_audit_logger,
    mock_consent_registry,
    test_strategy,
):
    """Test complete sync workflow with all components."""
    result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode=test_descriptor.sync_mode.value,
        status=SyncStatus.COMPLETED,
        files_synced=100,
        bytes_synced=500000,
        started_at=datetime.now(timezone.utc),
        duration_seconds=15.0,
    )
    mock_sync_orchestrator.sync_repository.return_value = result

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
        element_sink=mock_element_sink,
        audit_logger=mock_audit_logger,
        consent_registry=mock_consent_registry,
    )

    # Run sync
    sync_result = await connector.sync(strategy=test_strategy)

    # Verify result
    assert sync_result.status == SyncStatus.COMPLETED
    assert sync_result.files_synced == 100
    assert sync_result.bytes_synced == 500000

    # Verify consent was checked
    assert mock_consent_registry.has_consent.called

    # Verify audit events
    assert mock_audit_logger.record.call_count == 2

    # Verify orchestrator was called
    mock_sync_orchestrator.sync_repository.assert_called_once()


@pytest.mark.asyncio
async def test_sync_with_audit_failure(
    test_descriptor,
    mock_sync_orchestrator,
    mock_audit_logger,
    test_strategy,
):
    """Test sync continues even if audit logging fails."""
    result = SyncResult(
        repo_id=test_descriptor.id,
        sync_mode=test_descriptor.sync_mode.value,
        status=SyncStatus.COMPLETED,
        files_synced=50,
        bytes_synced=100000,
        started_at=datetime.now(timezone.utc),
    )
    mock_sync_orchestrator.sync_repository.return_value = result
    mock_audit_logger.record.side_effect = Exception("Audit logging failed")

    connector = GitHubRepositoryConnector(
        descriptor=test_descriptor,
        sync_orchestrator=mock_sync_orchestrator,
        audit_logger=mock_audit_logger,
    )

    # Sync should succeed despite audit logging failure
    sync_result = await connector.sync(strategy=test_strategy)

    assert sync_result.status == SyncStatus.COMPLETED
    assert sync_result.files_synced == 50
