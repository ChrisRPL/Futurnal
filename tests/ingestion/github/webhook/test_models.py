"""Tests for webhook data models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from futurnal.ingestion.github.webhook.models import (
    WebhookConfig,
    WebhookEvent,
    WebhookEventType,
)


# ---------------------------------------------------------------------------
# WebhookEventType Tests
# ---------------------------------------------------------------------------


def test_webhook_event_type_values():
    """Test webhook event type enum values."""
    assert WebhookEventType.PUSH == "push"
    assert WebhookEventType.PULL_REQUEST == "pull_request"
    assert WebhookEventType.ISSUES == "issues"
    assert WebhookEventType.ISSUE_COMMENT == "issue_comment"
    assert WebhookEventType.RELEASE == "release"
    assert WebhookEventType.CREATE == "create"
    assert WebhookEventType.DELETE == "delete"
    assert WebhookEventType.REPOSITORY == "repository"


# ---------------------------------------------------------------------------
# WebhookConfig Tests
# ---------------------------------------------------------------------------


def test_webhook_config_creation(sample_webhook_config):
    """Test webhook config creation with valid data."""
    assert sample_webhook_config.enabled is True
    assert sample_webhook_config.listen_host == "127.0.0.1"
    assert sample_webhook_config.listen_port == 8765
    assert sample_webhook_config.secret == "test-webhook-secret-1234567890"
    assert sample_webhook_config.verify_signature is True
    assert len(sample_webhook_config.enabled_events) == 3


def test_webhook_config_invalid_port():
    """Test webhook config with invalid port number."""
    with pytest.raises(ValidationError) as exc_info:
        WebhookConfig(
            secret="test-secret-1234567890",
            listen_port=99999,  # Invalid port
        )
    assert "listen_port" in str(exc_info.value)


def test_webhook_config_short_secret():
    """Test webhook config with too short secret."""
    with pytest.raises(ValidationError) as exc_info:
        WebhookConfig(
            secret="short",  # Too short
        )
    assert "secret" in str(exc_info.value).lower()


def test_webhook_config_invalid_public_url():
    """Test webhook config with invalid public URL."""
    with pytest.raises(ValidationError) as exc_info:
        WebhookConfig(
            secret="test-secret-1234567890",
            public_url="not-a-url",  # Invalid URL
        )
    assert "public_url" in str(exc_info.value).lower()


def test_webhook_config_no_events():
    """Test webhook config with no enabled events."""
    with pytest.raises(ValidationError) as exc_info:
        WebhookConfig(
            secret="test-secret-1234567890",
            enabled_events=[],  # Empty events
        )
    assert "event" in str(exc_info.value).lower()


def test_webhook_config_defaults():
    """Test webhook config default values."""
    config = WebhookConfig(secret="test-secret-1234567890")
    assert config.enabled is False
    assert config.listen_host == "127.0.0.1"
    assert config.listen_port == 8765
    assert config.verify_signature is True
    assert config.process_async is True
    assert config.queue_size == 1000


def test_webhook_config_custom_queue_size():
    """Test webhook config with custom queue size."""
    config = WebhookConfig(
        secret="test-secret-1234567890",
        queue_size=5000,
    )
    assert config.queue_size == 5000


def test_webhook_config_queue_size_bounds():
    """Test webhook config queue size bounds."""
    # Too small
    with pytest.raises(ValidationError):
        WebhookConfig(
            secret="test-secret-1234567890",
            queue_size=5,  # < 10
        )

    # Too large
    with pytest.raises(ValidationError):
        WebhookConfig(
            secret="test-secret-1234567890",
            queue_size=20000,  # > 10000
        )


# ---------------------------------------------------------------------------
# WebhookEvent Tests
# ---------------------------------------------------------------------------


def test_webhook_event_creation(sample_webhook_event):
    """Test webhook event creation."""
    assert sample_webhook_event.event_id == "12345-67890-abcdef"
    assert sample_webhook_event.event_type == WebhookEventType.PUSH
    assert sample_webhook_event.repository == "octocat/Hello-World"
    assert sample_webhook_event.processed is False
    assert sample_webhook_event.processing_error is None


def test_webhook_event_timezone_aware():
    """Test webhook event enforces timezone-aware datetime."""
    # Without timezone (should be converted to UTC)
    event = WebhookEvent(
        event_id="test-id",
        event_type=WebhookEventType.PUSH,
        repository="owner/repo",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),  # No timezone
        payload={"test": "data"},
    )
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.tzinfo == timezone.utc


def test_webhook_event_invalid_repository():
    """Test webhook event with invalid repository format."""
    with pytest.raises(ValidationError) as exc_info:
        WebhookEvent(
            event_id="test-id",
            event_type=WebhookEventType.PUSH,
            repository="invalid",  # Missing /
            timestamp=datetime.now(timezone.utc),
            payload={},
        )
    assert "repository" in str(exc_info.value).lower()


def test_webhook_event_mark_processed():
    """Test marking event as processed."""
    event = WebhookEvent(
        event_id="test-id",
        event_type=WebhookEventType.PUSH,
        repository="owner/repo",
        timestamp=datetime.now(timezone.utc),
        payload={},
    )

    assert event.processed is False

    event.mark_processed()

    assert event.processed is True
    assert event.processing_error is None


def test_webhook_event_mark_failed():
    """Test marking event as failed."""
    event = WebhookEvent(
        event_id="test-id",
        event_type=WebhookEventType.PUSH,
        repository="owner/repo",
        timestamp=datetime.now(timezone.utc),
        payload={},
    )

    error_msg = "Test error message"
    event.mark_failed(error_msg)

    assert event.processed is True
    assert event.processing_error == error_msg


def test_webhook_event_owner_property():
    """Test webhook event owner property."""
    event = WebhookEvent(
        event_id="test-id",
        event_type=WebhookEventType.PUSH,
        repository="octocat/Hello-World",
        timestamp=datetime.now(timezone.utc),
        payload={},
    )

    assert event.owner == "octocat"


def test_webhook_event_repo_name_property():
    """Test webhook event repo_name property."""
    event = WebhookEvent(
        event_id="test-id",
        event_type=WebhookEventType.PUSH,
        repository="octocat/Hello-World",
        timestamp=datetime.now(timezone.utc),
        payload={},
    )

    assert event.repo_name == "Hello-World"


def test_webhook_event_with_payload(sample_push_payload):
    """Test webhook event with full payload."""
    event = WebhookEvent(
        event_id="test-id",
        event_type=WebhookEventType.PUSH,
        repository="octocat/Hello-World",
        timestamp=datetime.now(timezone.utc),
        payload=sample_push_payload,
    )

    assert "repository" in event.payload
    assert event.payload["repository"]["full_name"] == "octocat/Hello-World"
    assert "commits" in event.payload


def test_webhook_event_multiple_events():
    """Test creating multiple webhook events with different types."""
    events = []

    for event_type in [
        WebhookEventType.PUSH,
        WebhookEventType.PULL_REQUEST,
        WebhookEventType.ISSUES,
    ]:
        event = WebhookEvent(
            event_id=f"test-{event_type.value}",
            event_type=event_type,
            repository="owner/repo",
            timestamp=datetime.now(timezone.utc),
            payload={},
        )
        events.append(event)

    assert len(events) == 3
    assert events[0].event_type == WebhookEventType.PUSH
    assert events[1].event_type == WebhookEventType.PULL_REQUEST
    assert events[2].event_type == WebhookEventType.ISSUES
