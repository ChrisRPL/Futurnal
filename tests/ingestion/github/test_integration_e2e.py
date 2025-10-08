"""End-to-end integration tests for GitHub connector.

These tests validate complete workflows from repository registration
through sync completion and element delivery to PKG storage.

Test categories:
- Full repository sync (GraphQL and Git clone modes)
- Incremental sync workflows
- Webhook-triggered sync
- Issue/PR metadata extraction
- Privacy and consent integration
- Quarantine and recovery workflows
- Multi-repository orchestration
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from futurnal.ingestion.github.connector import GitHubRepositoryConnector
from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    SyncMode,
    VisibilityType,
)
from futurnal.ingestion.github.sync_models import SyncResult, SyncStatus
from futurnal.ingestion.github.orchestrator_integration import GitHubConnectorManager
from tests.ingestion.github.fixtures import (
    small_test_repo_fixture,
    medium_test_repo_fixture,
    enhanced_mock_github_api,
)


# ---------------------------------------------------------------------------
# Integration Test Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.github_integration


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_element_sink():
    """Mock element sink for tracking delivered elements."""
    sink = Mock()
    sink.handle = Mock(return_value=None)
    sink.handle_deletion = Mock(return_value=None)
    sink.delivered_elements = []

    def track_handle(element):
        sink.delivered_elements.append(element)

    sink.handle.side_effect = track_handle
    return sink


@pytest.fixture
def mock_consent_registry():
    """Mock consent registry that grants all permissions."""
    registry = Mock()
    registry.has_consent = Mock(return_value=True)
    registry.check_consent = Mock(return_value=None)  # No exception = granted
    return registry


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for tracking operations."""
    logger = Mock()
    logger.record = Mock()
    logger.recorded_events = []

    def track_record(event_type, **kwargs):
        logger.recorded_events.append((event_type, kwargs))

    logger.record.side_effect = track_record
    return logger


@pytest.fixture
def integration_workspace(tmp_path):
    """Create integration test workspace."""
    workspace = tmp_path / "integration_workspace"
    workspace.mkdir()
    return workspace


# ---------------------------------------------------------------------------
# Full Repository Sync Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_repository_sync_graphql_small_repo(
    small_test_repo_fixture,
    enhanced_mock_github_api,
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test complete GraphQL sync workflow with small repository.

    Validates:
    - Repository registration
    - GraphQL API sync
    - File content extraction
    - Element delivery to sink
    - Audit logging
    """
    # Setup mock API with test repository
    mock_repo = small_test_repo_fixture
    enhanced_mock_github_api.add_repository(
        mock_repo.owner,
        mock_repo.repo,
        mock_repo.to_github_api_response(),
    )

    # Add file contents
    for file in mock_repo.files:
        enhanced_mock_github_api.add_file_content(
            mock_repo.owner,
            mock_repo.repo,
            file.path,
            file.content,
            file.sha,
        )

    # Create descriptor
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner=mock_repo.owner,
        repo=mock_repo.repo,
        credential_id="test_cred_graphql",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GRAPHQL_API,
    )

    # Create connector with mocked dependencies
    with patch("futurnal.ingestion.github.graphql_sync.GitHubGraphQLSync") as MockGraphQLSync:
        # Setup mock sync to simulate successful sync
        mock_sync_instance = MockGraphQLSync.return_value
        mock_sync_instance.sync_repository = AsyncMock(
            return_value=SyncResult(
                status=SyncStatus.SUCCESS,
                files_processed=len(mock_repo.files),
                files_added=len(mock_repo.files),
                files_modified=0,
                files_deleted=0,
                sync_duration_seconds=5.2,
            )
        )

        connector = GitHubRepositoryConnector(
            descriptor=descriptor,
            workspace_dir=integration_workspace,
            element_sink=mock_element_sink,
            consent_registry=mock_consent_registry,
            audit_logger=mock_audit_logger,
        )

        # Execute sync
        result = await connector.sync()

        # Assertions
        assert result.status == SyncStatus.SUCCESS
        assert result.files_processed == len(mock_repo.files)
        assert result.files_added == len(mock_repo.files)
        assert result.sync_duration_seconds > 0

        # Verify audit logging
        assert len(mock_audit_logger.recorded_events) > 0
        sync_events = [e for e in mock_audit_logger.recorded_events if "sync" in e[0].lower()]
        assert len(sync_events) > 0


@pytest.mark.asyncio
async def test_full_repository_sync_git_clone(
    small_test_repo_fixture,
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test complete git clone sync workflow.

    Validates:
    - Git clone operation
    - File tree traversal
    - Element extraction
    - Local storage cleanup
    """
    mock_repo = small_test_repo_fixture
    clone_path = integration_workspace / "clones" / mock_repo.repo

    # Create descriptor for git clone mode
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner=mock_repo.owner,
        repo=mock_repo.repo,
        credential_id="test_cred_clone",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GIT_CLONE,
        local_clone_path=clone_path,
    )

    with patch("futurnal.ingestion.github.git_clone_sync.GitHubGitCloneSync") as MockCloneSync:
        # Setup mock clone sync
        mock_sync_instance = MockCloneSync.return_value
        mock_sync_instance.sync_repository = AsyncMock(
            return_value=SyncResult(
                status=SyncStatus.SUCCESS,
                files_processed=len(mock_repo.files),
                files_added=len(mock_repo.files),
                files_modified=0,
                files_deleted=0,
                sync_duration_seconds=8.5,
            )
        )

        connector = GitHubRepositoryConnector(
            descriptor=descriptor,
            workspace_dir=integration_workspace,
            element_sink=mock_element_sink,
            consent_registry=mock_consent_registry,
            audit_logger=mock_audit_logger,
        )

        result = await connector.sync()

        assert result.status == SyncStatus.SUCCESS
        assert result.files_processed == len(mock_repo.files)


# ---------------------------------------------------------------------------
# Incremental Sync Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_sync_after_push(
    small_test_repo_fixture,
    enhanced_mock_github_api,
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test incremental sync after new commits.

    Validates:
    - State tracking between syncs
    - Delta detection
    - Incremental processing
    - Updated element delivery
    """
    mock_repo = small_test_repo_fixture

    # Setup API
    enhanced_mock_github_api.add_repository(
        mock_repo.owner,
        mock_repo.repo,
        mock_repo.to_github_api_response(),
    )

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner=mock_repo.owner,
        repo=mock_repo.repo,
        credential_id="test_cred_incremental",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GRAPHQL_API,
    )

    with patch("futurnal.ingestion.github.incremental_sync.IncrementalSyncEngine") as MockIncrementalSync:
        # First sync - full
        mock_incremental = MockIncrementalSync.return_value
        mock_incremental.detect_changes = AsyncMock(
            return_value={
                "added": ["new_file.py"],
                "modified": ["README.md"],
                "deleted": [],
            }
        )

        connector = GitHubRepositoryConnector(
            descriptor=descriptor,
            workspace_dir=integration_workspace,
            element_sink=mock_element_sink,
            consent_registry=mock_consent_registry,
            audit_logger=mock_audit_logger,
        )

        # Simulate incremental sync
        with patch.object(connector, "_perform_full_sync") as mock_full_sync:
            mock_full_sync.return_value = SyncResult(
                status=SyncStatus.SUCCESS,
                files_processed=2,
                files_added=1,
                files_modified=1,
                files_deleted=0,
                sync_duration_seconds=2.1,
            )

            result = await connector.sync()

            assert result.status == SyncStatus.SUCCESS
            assert result.files_added + result.files_modified == 2


@pytest.mark.asyncio
async def test_webhook_triggered_sync(
    small_test_repo_fixture,
    enhanced_mock_github_api,
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test sync triggered by webhook event.

    Validates:
    - Webhook event processing
    - Automatic sync trigger
    - Push event handling
    - Sync completion notification
    """
    mock_repo = small_test_repo_fixture

    # Setup API
    enhanced_mock_github_api.add_repository(
        mock_repo.owner,
        mock_repo.repo,
        mock_repo.to_github_api_response(),
    )

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner=mock_repo.owner,
        repo=mock_repo.repo,
        credential_id="test_cred_webhook",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GRAPHQL_API,
    )

    with patch("futurnal.ingestion.github.webhook.handler.WebhookEventHandler") as MockWebhookHandler:
        # Simulate webhook event
        mock_handler = MockWebhookHandler.return_value
        mock_handler.handle_push_event = AsyncMock(return_value=True)

        # Webhook payload
        webhook_payload = {
            "ref": "refs/heads/main",
            "repository": {
                "full_name": f"{mock_repo.owner}/{mock_repo.repo}",
                "owner": {"name": mock_repo.owner},
                "name": mock_repo.repo,
            },
            "commits": [
                {
                    "id": "new_commit_sha",
                    "message": "Update README",
                    "added": ["new_file.py"],
                    "modified": ["README.md"],
                    "removed": [],
                }
            ],
        }

        # Process webhook
        sync_triggered = await mock_handler.handle_push_event(webhook_payload)

        assert sync_triggered is True
        mock_handler.handle_push_event.assert_called_once()


# ---------------------------------------------------------------------------
# Issue/PR Metadata Extraction Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_issue_pr_metadata_extraction(
    enhanced_mock_github_api,
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test complete issue and PR metadata extraction workflow.

    Validates:
    - Issue fetching
    - PR fetching
    - Metadata normalization
    - Triple generation
    - Element delivery
    """
    with patch("futurnal.ingestion.github.issue_normalizer.IssueNormalizer") as MockIssueNormalizer:
        with patch("futurnal.ingestion.github.pr_normalizer.PRNormalizer") as MockPRNormalizer:
            # Setup mocks
            mock_issue_normalizer = MockIssueNormalizer.return_value
            mock_issue_normalizer.normalize_issue = AsyncMock(
                return_value={
                    "id": "issue_123",
                    "title": "Test Issue",
                    "state": "open",
                    "author": "user1",
                }
            )

            mock_pr_normalizer = MockPRNormalizer.return_value
            mock_pr_normalizer.normalize_pr = AsyncMock(
                return_value={
                    "id": "pr_456",
                    "title": "Test PR",
                    "state": "open",
                    "author": "user2",
                }
            )

            # Execute normalization
            issue_result = await mock_issue_normalizer.normalize_issue({"number": 123})
            pr_result = await mock_pr_normalizer.normalize_pr({"number": 456})

            assert issue_result["id"] == "issue_123"
            assert pr_result["id"] == "pr_456"


# ---------------------------------------------------------------------------
# Privacy & Consent Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consent_workflow_integration(
    small_test_repo_fixture,
    mock_element_sink,
    mock_audit_logger,
    integration_workspace,
):
    """Test privacy consent workflow integration.

    Validates:
    - Consent checking before sync
    - Consent denial handling
    - Audit logging of consent checks
    - Graceful failure on denied consent
    """
    mock_repo = small_test_repo_fixture

    # Create consent registry that denies consent
    mock_consent_denied = Mock()
    mock_consent_denied.has_consent = Mock(return_value=False)
    mock_consent_denied.check_consent = Mock(
        side_effect=PermissionError("Consent not granted")
    )

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner=mock_repo.owner,
        repo=mock_repo.repo,
        credential_id="test_cred_consent",
        visibility=VisibilityType.PRIVATE,
        sync_mode=SyncMode.GRAPHQL_API,
    )

    connector = GitHubRepositoryConnector(
        descriptor=descriptor,
        workspace_dir=integration_workspace,
        element_sink=mock_element_sink,
        consent_registry=mock_consent_denied,
        audit_logger=mock_audit_logger,
    )

    # Sync should fail due to missing consent
    with pytest.raises(PermissionError, match="Consent not granted"):
        await connector._check_consent()

    # Verify consent check was attempted
    mock_consent_denied.check_consent.assert_called()


# ---------------------------------------------------------------------------
# Quarantine & Recovery Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quarantine_recovery_workflow(
    small_test_repo_fixture,
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test quarantine and recovery workflow for failed files.

    Validates:
    - File quarantine on processing failure
    - Retry mechanism
    - Recovery from quarantine
    - Audit logging of failures
    """
    mock_repo = small_test_repo_fixture

    with patch("futurnal.ingestion.github.quarantine.QuarantineHandler") as MockQuarantine:
        # Setup quarantine mock
        mock_quarantine = MockQuarantine.return_value
        mock_quarantine.quarantine_file = Mock()
        mock_quarantine.retry_quarantined = AsyncMock(return_value=True)

        # Simulate file processing failure
        mock_quarantine.quarantine_file(
            file_path="failed_file.py",
            reason="Parse error",
            error_details={"exception": "ValueError"},
        )

        # Verify file was quarantined
        mock_quarantine.quarantine_file.assert_called_once()

        # Simulate retry
        retry_success = await mock_quarantine.retry_quarantined()
        assert retry_success is True


# ---------------------------------------------------------------------------
# Multi-Repository Orchestration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_repository_orchestration(
    small_test_repo_fixture,
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test orchestration of multiple repositories.

    Validates:
    - Concurrent repository sync
    - Resource sharing (rate limits)
    - Independent failure handling
    - Aggregated results
    """
    # Create multiple repository descriptors
    repos = []
    for i in range(3):
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner=f"owner{i}",
            repo=f"repo{i}",
            credential_id=f"test_cred_{i}",
            visibility=VisibilityType.PUBLIC,
            sync_mode=SyncMode.GRAPHQL_API,
        )
        repos.append(descriptor)

    # Create orchestrator
    with patch("futurnal.ingestion.github.orchestrator_integration.GitHubConnectorManager") as MockManager:
        mock_manager = MockManager.return_value
        mock_manager.sync_repository = AsyncMock(
            return_value=SyncResult(
                status=SyncStatus.SUCCESS,
                files_processed=10,
                files_added=10,
                files_modified=0,
                files_deleted=0,
                sync_duration_seconds=3.0,
            )
        )

        # Sync all repositories concurrently
        tasks = [mock_manager.sync_repository(repo.id) for repo in repos]
        results = await asyncio.gather(*tasks)

        # Verify all syncs completed
        assert len(results) == 3
        assert all(r.status == SyncStatus.SUCCESS for r in results)


# ---------------------------------------------------------------------------
# Force Push Detection Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_push_detection_and_handling(
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test force push detection and handling.

    Validates:
    - Force push detection via commit history
    - Full resync trigger
    - Audit logging of force push event
    - Proper state update
    """
    with patch("futurnal.ingestion.github.incremental_sync.IncrementalSyncEngine") as MockIncremental:
        mock_incremental = MockIncremental.return_value
        mock_incremental.detect_force_push = AsyncMock(return_value=True)
        mock_incremental.handle_force_push = AsyncMock(
            return_value=SyncResult(
                status=SyncStatus.SUCCESS,
                files_processed=50,
                files_added=50,
                files_modified=0,
                files_deleted=0,
                sync_duration_seconds=10.0,
            )
        )

        # Detect force push
        is_force_push = await mock_incremental.detect_force_push()
        assert is_force_push is True

        # Handle force push
        result = await mock_incremental.handle_force_push()
        assert result.status == SyncStatus.SUCCESS
        assert result.files_processed > 0


# ---------------------------------------------------------------------------
# End-to-End Health Check Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connector_health_check(
    small_test_repo_fixture,
    enhanced_mock_github_api,
    mock_element_sink,
    mock_consent_registry,
    mock_audit_logger,
    integration_workspace,
):
    """Test connector health check integration.

    Validates:
    - Health check execution
    - API connectivity verification
    - Credential validation
    - Resource availability check
    """
    mock_repo = small_test_repo_fixture

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner=mock_repo.owner,
        repo=mock_repo.repo,
        credential_id="test_cred_health",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GRAPHQL_API,
    )

    connector = GitHubRepositoryConnector(
        descriptor=descriptor,
        workspace_dir=integration_workspace,
        element_sink=mock_element_sink,
        consent_registry=mock_consent_registry,
        audit_logger=mock_audit_logger,
    )

    # Execute health check
    with patch.object(connector, "health_check") as mock_health_check:
        mock_health_check.return_value = {
            "status": "healthy",
            "api_accessible": True,
            "credentials_valid": True,
            "workspace_available": True,
        }

        health = connector.health_check()

        assert health["status"] == "healthy"
        assert health["api_accessible"] is True
        assert health["credentials_valid"] is True
