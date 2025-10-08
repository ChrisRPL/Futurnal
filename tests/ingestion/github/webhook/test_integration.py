"""Integration tests for webhook system."""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest

from futurnal.ingestion.github.webhook.models import WebhookEvent, WebhookEventType
from futurnal.ingestion.github.webhook.server import WebhookServer
from futurnal.ingestion.github.webhook.handler import WebhookEventHandler


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_webhook_flow(
    sample_webhook_config,
    sample_push_payload,
    mock_sync_engine,
    mock_issue_normalizer,
    mock_pr_normalizer,
    mock_api_client_manager,
    mock_state_manager,
    mock_repository_registry,
    mock_audit_logger,
    mock_request,
):
    """Test complete webhook flow: receive → verify → queue → process."""
    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    # Disable async processing for predictable testing
    sample_webhook_config.process_async = False

    server = WebhookServer(
        config=sample_webhook_config,
        event_handler=handler,
        audit_logger=mock_audit_logger,
    )

    # Create webhook request with valid signature
    payload_body = json.dumps(sample_push_payload).encode("utf-8")
    signature = hmac.new(
        sample_webhook_config.secret.encode("utf-8"),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    request = mock_request(
        headers={
            "X-Hub-Signature-256": f"sha256={signature}",
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": "integration-test-id",
        },
        body=payload_body,
        json_data=sample_push_payload,
    )

    # Handle webhook
    response = await server.handle_webhook(request)

    # Verify response
    assert response.status == 200
    assert response.text == "Event queued"

    # Verify event was queued
    assert server.event_queue.qsize() == 1

    # Get event from queue
    event = await server.event_queue.get()

    assert event.event_id == "integration-test-id"
    assert event.event_type == WebhookEventType.PUSH
    assert event.repository == "octocat/Hello-World"

    # Process event
    await handler.handle_event(event)

    # Verify sync engine was called
    mock_sync_engine.sync_repository.assert_called_once()


@pytest.mark.asyncio
async def test_multiple_events_processing(
    sample_webhook_config,
    sample_push_payload,
    sample_pr_payload,
    sample_issue_payload,
    mock_sync_engine,
    mock_issue_normalizer,
    mock_pr_normalizer,
    mock_api_client_manager,
    mock_state_manager,
    mock_repository_registry,
    mock_audit_logger,
):
    """Test processing multiple different event types."""
    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    # Create events
    events = [
        WebhookEvent(
            event_id="push-event",
            event_type=WebhookEventType.PUSH,
            repository="octocat/Hello-World",
            timestamp=datetime.now(timezone.utc),
            payload=sample_push_payload,
        ),
        WebhookEvent(
            event_id="pr-event",
            event_type=WebhookEventType.PULL_REQUEST,
            repository="octocat/Hello-World",
            timestamp=datetime.now(timezone.utc),
            payload=sample_pr_payload,
        ),
        WebhookEvent(
            event_id="issue-event",
            event_type=WebhookEventType.ISSUES,
            repository="octocat/Hello-World",
            timestamp=datetime.now(timezone.utc),
            payload=sample_issue_payload,
        ),
    ]

    # Process all events
    for event in events:
        await handler.handle_event(event)

    # Verify all handlers were called
    mock_sync_engine.sync_repository.assert_called_once()
    mock_pr_normalizer.normalize_pull_request.assert_called_once()
    mock_issue_normalizer.normalize_issue.assert_called_once()
