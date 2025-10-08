"""Tests for webhook server."""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest

from futurnal.ingestion.github.webhook.server import WebhookServer
from futurnal.ingestion.github.webhook.models import WebhookEventType


# ---------------------------------------------------------------------------
# Server Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_health_check(
    sample_webhook_config,
    mock_sync_engine,
    mock_issue_normalizer,
    mock_pr_normalizer,
    mock_api_client_manager,
    mock_state_manager,
    mock_repository_registry,
    mock_audit_logger,
    mock_request,
):
    """Test health check endpoint."""
    from futurnal.ingestion.github.webhook.handler import WebhookEventHandler

    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    server = WebhookServer(
        config=sample_webhook_config,
        event_handler=handler,
        audit_logger=mock_audit_logger,
    )

    request = mock_request()
    response = await server.health_check(request)

    assert response.status == 200
    # Parse JSON from response body
    import json as json_lib
    data = json_lib.loads(response.body.decode())
    assert data["status"] == "healthy"
    assert "queue_size" in data
    assert "max_queue_size" in data


@pytest.mark.asyncio
async def test_server_webhook_valid_signature(
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
    """Test webhook handling with valid signature."""
    from futurnal.ingestion.github.webhook.handler import WebhookEventHandler

    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    server = WebhookServer(
        config=sample_webhook_config,
        event_handler=handler,
        audit_logger=mock_audit_logger,
    )

    # Create request with valid signature
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
            "X-GitHub-Delivery": "test-delivery-id",
        },
        body=payload_body,
        json_data=sample_push_payload,
    )

    response = await server.handle_webhook(request)

    assert response.status == 200
    assert response.text == "Event queued"


@pytest.mark.asyncio
async def test_server_webhook_invalid_signature(
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
    """Test webhook handling with invalid signature."""
    from futurnal.ingestion.github.webhook.handler import WebhookEventHandler

    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    server = WebhookServer(
        config=sample_webhook_config,
        event_handler=handler,
        audit_logger=mock_audit_logger,
    )

    payload_body = json.dumps(sample_push_payload).encode("utf-8")

    request = mock_request(
        headers={
            "X-Hub-Signature-256": "sha256=invalid_signature",
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": "test-delivery-id",
        },
        body=payload_body,
        json_data=sample_push_payload,
    )

    response = await server.handle_webhook(request)

    assert response.status == 401
    assert "signature" in response.text.lower()


@pytest.mark.asyncio
async def test_server_rate_limiting(
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
    """Test rate limiting blocks excessive requests."""
    from futurnal.ingestion.github.webhook.handler import WebhookEventHandler

    # Set low rate limit for testing
    sample_webhook_config.max_requests_per_minute = 2

    handler = WebhookEventHandler(
        sync_engine=mock_sync_engine,
        issue_normalizer=mock_issue_normalizer,
        pr_normalizer=mock_pr_normalizer,
        api_client_manager=mock_api_client_manager,
        state_manager=mock_state_manager,
        repository_registry=mock_repository_registry,
        audit_logger=mock_audit_logger,
    )

    server = WebhookServer(
        config=sample_webhook_config,
        event_handler=handler,
        audit_logger=mock_audit_logger,
    )

    payload_body = json.dumps(sample_push_payload).encode("utf-8")
    signature = hmac.new(
        sample_webhook_config.secret.encode("utf-8"),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    # First 2 requests should succeed
    for _ in range(2):
        request = mock_request(
            headers={
                "X-Hub-Signature-256": f"sha256={signature}",
                "X-GitHub-Event": "push",
                "X-GitHub-Delivery": "test-delivery-id",
            },
            body=payload_body,
            json_data=sample_push_payload,
        )
        response = await server.handle_webhook(request)
        assert response.status == 200

    # 3rd request should be rate limited
    request = mock_request(
        headers={
            "X-Hub-Signature-256": f"sha256={signature}",
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": "test-delivery-id",
        },
        body=payload_body,
        json_data=sample_push_payload,
    )
    response = await server.handle_webhook(request)
    assert response.status == 429
