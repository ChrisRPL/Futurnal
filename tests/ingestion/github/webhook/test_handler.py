"""Tests for webhook event handler."""

import pytest
from datetime import datetime, timezone

from futurnal.ingestion.github.webhook.handler import WebhookEventHandler
from futurnal.ingestion.github.webhook.models import WebhookEvent, WebhookEventType


# ---------------------------------------------------------------------------
# Event Handler Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_push_event(
    sample_push_payload,
    sample_repo_descriptor,
    mock_sync_engine,
    mock_issue_normalizer,
    mock_pr_normalizer,
    mock_api_client_manager,
    mock_state_manager,
    mock_repository_registry,
    mock_audit_logger,
):
    """Test handling push event."""
    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    event = WebhookEvent(
        event_id="test-push",
        event_type=WebhookEventType.PUSH,
        repository="octocat/Hello-World",
        timestamp=datetime.now(timezone.utc),
        payload=sample_push_payload,
    )

    await handler.handle_event(event)

    # Verify sync engine was called
    mock_sync_engine.sync_repository.assert_called_once()


@pytest.mark.asyncio
async def test_handler_pull_request_event(
    sample_pr_payload,
    sample_repo_descriptor,
    mock_sync_engine,
    mock_issue_normalizer,
    mock_pr_normalizer,
    mock_api_client_manager,
    mock_state_manager,
    mock_repository_registry,
    mock_audit_logger,
):
    """Test handling pull request event."""
    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    event = WebhookEvent(
        event_id="test-pr",
        event_type=WebhookEventType.PULL_REQUEST,
        repository="octocat/Hello-World",
        timestamp=datetime.now(timezone.utc),
        payload=sample_pr_payload,
    )

    await handler.handle_event(event)

    # Verify PR normalizer was called
    mock_pr_normalizer.normalize_pull_request.assert_called_once()


@pytest.mark.asyncio
async def test_handler_issue_event(
    sample_issue_payload,
    sample_repo_descriptor,
    mock_sync_engine,
    mock_issue_normalizer,
    mock_pr_normalizer,
    mock_api_client_manager,
    mock_state_manager,
    mock_repository_registry,
    mock_audit_logger,
):
    """Test handling issue event."""
    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    event = WebhookEvent(
        event_id="test-issue",
        event_type=WebhookEventType.ISSUES,
        repository="octocat/Hello-World",
        timestamp=datetime.now(timezone.utc),
        payload=sample_issue_payload,
    )

    await handler.handle_event(event)

    # Verify issue normalizer was called
    mock_issue_normalizer.normalize_issue.assert_called_once()


@pytest.mark.asyncio
async def test_handler_repository_not_found(
    sample_push_payload,
    mock_sync_engine,
    mock_issue_normalizer,
    mock_pr_normalizer,
    mock_api_client_manager,
    mock_state_manager,
    mock_audit_logger,
):
    """Test handling event for unregistered repository."""
    from unittest.mock import Mock
    from futurnal.ingestion.github.descriptor import RepositoryRegistry
    from futurnal.privacy.audit import AuditLogger
    from pathlib import Path
    import tempfile

    # Create registry with no repositories
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_logger = AuditLogger(Path(tmpdir) / "audit")
        empty_registry = RepositoryRegistry(audit_logger=audit_logger)
        empty_registry.list = Mock(return_value=[])

        handler = WebhookEventHandler(
            sync_engine=mock_sync_engine,
            issue_normalizer=mock_issue_normalizer,
            pr_normalizer=mock_pr_normalizer,
            api_client_manager=mock_api_client_manager,
            state_manager=mock_state_manager,
            repository_registry=empty_registry,
            audit_logger=audit_logger,
        )

        event = WebhookEvent(
            event_id="test-push",
            event_type=WebhookEventType.PUSH,
            repository="unregistered/repo",
            timestamp=datetime.now(timezone.utc),
            payload=sample_push_payload,
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="not found in registry"):
            await handler.handle_event(event)
