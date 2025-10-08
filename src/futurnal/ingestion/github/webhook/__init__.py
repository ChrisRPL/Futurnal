"""GitHub webhook integration for real-time repository updates.

This module implements optional GitHub webhook support for real-time repository
updates, eliminating polling delays and reducing API usage. Key features:

- **WebhookServer**: aiohttp-based server for receiving GitHub webhooks
- **WebhookEventHandler**: Routes webhook events to appropriate handlers
- **Security**: HMAC-SHA256 signature verification and rate limiting
- **Privacy**: Audit logging without payload content exposure

The webhook system is optional and works alongside polling as a fallback.

Example usage:
    >>> from futurnal.ingestion.github.webhook import WebhookServer, WebhookConfig
    >>> config = WebhookConfig(
    ...     enabled=True,
    ...     listen_port=8765,
    ...     secret="your-webhook-secret"
    ... )
    >>> server = WebhookServer(config=config, event_handler=handler)
    >>> await server.start()
"""

from .models import (
    WebhookConfig,
    WebhookEvent,
    WebhookEventType,
)
from .security import (
    verify_webhook_signature,
    WebhookRateLimiter,
)
from .server import WebhookServer
from .handler import WebhookEventHandler
from .configurator import GitHubWebhookConfigurator

__all__ = [
    # Models
    "WebhookConfig",
    "WebhookEvent",
    "WebhookEventType",
    # Security
    "verify_webhook_signature",
    "WebhookRateLimiter",
    # Server
    "WebhookServer",
    # Handler
    "WebhookEventHandler",
    # Configurator
    "GitHubWebhookConfigurator",
]
