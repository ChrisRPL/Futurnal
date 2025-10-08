"""Data models for GitHub webhook integration.

This module defines the Pydantic models used for webhook configuration, event
parsing, and processing. All models follow Futurnal's privacy-first principles
with timezone-aware datetime handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WebhookEventType(str, Enum):
    """GitHub webhook event types supported by Futurnal."""

    PUSH = "push"  # New commits pushed
    PULL_REQUEST = "pull_request"  # PR opened/updated/closed
    ISSUES = "issues"  # Issue opened/updated/closed
    ISSUE_COMMENT = "issue_comment"  # Comment on issue/PR
    RELEASE = "release"  # Release published
    CREATE = "create"  # Branch/tag created
    DELETE = "delete"  # Branch/tag deleted
    REPOSITORY = "repository"  # Repository settings changed


# ---------------------------------------------------------------------------
# Webhook Configuration
# ---------------------------------------------------------------------------


class WebhookConfig(BaseModel):
    """Configuration for webhook integration.

    This model defines the webhook server configuration including network
    settings, security parameters, and event processing options.

    Attributes:
        enabled: Whether webhook server is enabled
        listen_host: Host to bind server to (default localhost for security)
        listen_port: Port to listen on (default 8765)
        public_url: Public URL for GitHub webhook configuration
        secret: HMAC secret for signature verification
        verify_signature: Whether to verify webhook signatures
        enabled_events: List of event types to process
        queue_size: Maximum event queue size
        process_async: Whether to process events asynchronously
        max_requests_per_minute: Rate limit per repository

    Example:
        >>> config = WebhookConfig(
        ...     enabled=True,
        ...     listen_port=8765,
        ...     secret="your-webhook-secret",
        ...     public_url="https://example.com/webhook/github"
        ... )
    """

    # Server settings
    enabled: bool = Field(default=False, description="Enable webhook server")
    listen_host: str = Field(
        default="127.0.0.1",
        description="Host to bind server to (localhost only by default)"
    )
    listen_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="Port to listen on"
    )
    public_url: Optional[str] = Field(
        default=None,
        description="Public URL for GitHub webhook configuration"
    )

    # Security
    secret: str = Field(
        ...,
        min_length=16,
        description="HMAC secret for signature verification"
    )
    verify_signature: bool = Field(
        default=True,
        description="Verify webhook signatures (should always be True)"
    )
    max_requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Maximum requests per minute per repository"
    )

    # Event filtering
    enabled_events: List[WebhookEventType] = Field(
        default_factory=lambda: [
            WebhookEventType.PUSH,
            WebhookEventType.PULL_REQUEST,
            WebhookEventType.ISSUES,
        ],
        description="Event types to process"
    )

    # Processing
    queue_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum event queue size"
    )
    process_async: bool = Field(
        default=True,
        description="Process events asynchronously"
    )

    @field_validator("secret")
    @classmethod
    def _validate_secret(cls, value: str) -> str:
        """Validate webhook secret is strong enough."""
        if len(value) < 16:
            raise ValueError("Webhook secret must be at least 16 characters")
        return value

    @field_validator("public_url")
    @classmethod
    def _validate_public_url(cls, value: Optional[str]) -> Optional[str]:
        """Validate public URL format."""
        if value and not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError("Public URL must start with http:// or https://")
        return value

    @field_validator("enabled_events")
    @classmethod
    def _validate_events(cls, value: List[WebhookEventType]) -> List[WebhookEventType]:
        """Ensure at least one event type is enabled."""
        if not value:
            raise ValueError("At least one event type must be enabled")
        return value


# ---------------------------------------------------------------------------
# Webhook Event
# ---------------------------------------------------------------------------


class WebhookEvent(BaseModel):
    """Parsed webhook event from GitHub.

    This model represents a parsed and validated webhook event ready for
    processing. It contains event metadata and the full payload for routing
    to appropriate handlers.

    Attributes:
        event_id: Unique event identifier (X-GitHub-Delivery header)
        event_type: Type of webhook event
        repository: Repository full name (owner/repo)
        timestamp: Event received timestamp
        payload: Raw webhook payload
        processed: Whether event has been processed
        processing_error: Error message if processing failed

    Example:
        >>> event = WebhookEvent(
        ...     event_id="12345-67890-abcdef",
        ...     event_type=WebhookEventType.PUSH,
        ...     repository="octocat/Hello-World",
        ...     timestamp=datetime.now(timezone.utc),
        ...     payload={"ref": "refs/heads/main", ...}
        ... )
    """

    event_id: str = Field(
        ...,
        description="Unique event identifier from X-GitHub-Delivery header"
    )
    event_type: WebhookEventType = Field(
        ...,
        description="Type of webhook event"
    )
    repository: str = Field(
        ...,
        description="Repository full name (owner/repo)"
    )
    timestamp: datetime = Field(
        ...,
        description="Event received timestamp"
    )

    # Event payload
    payload: Dict[str, Any] = Field(
        ...,
        description="Raw webhook payload from GitHub"
    )

    # Processing status
    processed: bool = Field(
        default=False,
        description="Whether event has been processed"
    )
    processing_error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )

    @field_validator("timestamp")
    @classmethod
    def _ensure_timezone(cls, value: datetime) -> datetime:
        """Ensure datetime has timezone info (UTC)."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @field_validator("repository")
    @classmethod
    def _validate_repository(cls, value: str) -> str:
        """Validate repository format (owner/repo)."""
        if "/" not in value:
            raise ValueError("Repository must be in format 'owner/repo'")
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError("Repository must be in format 'owner/repo'")
        if not parts[0] or not parts[1]:
            raise ValueError("Repository owner and name cannot be empty")
        return value

    def mark_processed(self) -> None:
        """Mark event as successfully processed."""
        self.processed = True
        self.processing_error = None

    def mark_failed(self, error: str) -> None:
        """Mark event as failed with error message.

        Args:
            error: Sanitized error message (no sensitive data)
        """
        self.processed = True
        self.processing_error = error

    @property
    def owner(self) -> str:
        """Extract repository owner from full name."""
        return self.repository.split("/")[0]

    @property
    def repo_name(self) -> str:
        """Extract repository name from full name."""
        return self.repository.split("/")[1]


__all__ = [
    "WebhookConfig",
    "WebhookEvent",
    "WebhookEventType",
]
