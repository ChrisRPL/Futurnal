"""GitHub webhook server implementation using aiohttp.

This module implements the HTTP server that receives GitHub webhook events,
verifies signatures, and queues events for asynchronous processing.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from aiohttp import web

from futurnal.privacy.audit import AuditEvent, AuditLogger

from .models import WebhookConfig, WebhookEvent, WebhookEventType
from .security import verify_webhook_signature, WebhookRateLimiter

if TYPE_CHECKING:
    from .handler import WebhookEventHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Webhook Server
# ---------------------------------------------------------------------------


class WebhookServer:
    """Receives and processes GitHub webhooks.

    This class implements an aiohttp-based HTTP server that:
    - Receives webhook POST requests from GitHub
    - Verifies HMAC-SHA256 signatures
    - Applies rate limiting per repository
    - Queues events for asynchronous processing
    - Provides health check endpoint

    The server responds quickly to GitHub (< 1s) by queuing events and
    processing them asynchronously in the background.

    Attributes:
        config: Webhook server configuration
        event_handler: Handler for processing webhook events
        event_queue: Async queue for event processing
        rate_limiter: Per-repository rate limiter
        audit_logger: Privacy-aware audit logger
        app: aiohttp web application
        runner: aiohttp app runner
        site: aiohttp TCP site

    Example:
        >>> server = WebhookServer(
        ...     config=config,
        ...     event_handler=handler,
        ...     audit_logger=audit_logger
        ... )
        >>> await server.start()
        >>> # Server running...
        >>> await server.stop()
    """

    def __init__(
        self,
        config: WebhookConfig,
        event_handler: WebhookEventHandler,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """Initialize webhook server.

        Args:
            config: Webhook server configuration
            event_handler: Handler for processing webhook events
            audit_logger: Optional audit logger for privacy-aware logging
        """
        self.config = config
        self.event_handler = event_handler
        self.audit_logger = audit_logger

        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)

        # Rate limiter
        self.rate_limiter = WebhookRateLimiter(
            max_requests_per_minute=config.max_requests_per_minute
        )

        # aiohttp application
        self.app = web.Application()
        self._setup_routes()

        # Server components (initialized in start())
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._processor_task: Optional[asyncio.Task] = None

    def _setup_routes(self) -> None:
        """Setup HTTP routes."""
        self.app.router.add_post("/webhook/github", self.handle_webhook)
        self.app.router.add_get("/health", self.health_check)

    async def handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming GitHub webhook.

        This endpoint:
        1. Verifies HMAC-SHA256 signature
        2. Checks rate limits
        3. Parses event metadata
        4. Queues event for processing
        5. Returns 200 OK immediately

        Args:
            request: aiohttp web request

        Returns:
            HTTP response (200, 401, 429, or 503)
        """
        # Extract headers
        signature_header = request.headers.get("X-Hub-Signature-256", "")
        event_type_header = request.headers.get("X-GitHub-Event", "")
        delivery_id = request.headers.get("X-GitHub-Delivery", "unknown")

        # Read request body
        body = await request.read()

        # Verify signature
        if self.config.verify_signature:
            if not verify_webhook_signature(body, signature_header, self.config.secret):
                logger.warning(
                    f"Invalid webhook signature for event {delivery_id}",
                    extra={"event_id": delivery_id, "event_type": event_type_header},
                )
                if self.audit_logger:
                    self.audit_logger.record(
                        AuditEvent(
                            job_id=delivery_id,
                            source="github_webhook",
                            action="webhook_received",
                            status="signature_failed",
                            timestamp=datetime.now(timezone.utc),
                            metadata={"event_type": event_type_header},
                        )
                    )
                return web.Response(status=401, text="Invalid signature")

        # Parse JSON payload
        try:
            payload = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse webhook payload: {e}")
            return web.Response(status=400, text="Invalid JSON payload")

        # Extract repository
        repository = payload.get("repository", {}).get("full_name", "unknown/unknown")

        # Check rate limits
        if not self.rate_limiter.allow_request(repository):
            logger.warning(
                f"Rate limit exceeded for repository {repository}",
                extra={"repository": repository, "event_id": delivery_id},
            )
            if self.audit_logger:
                self.audit_logger.record(
                    AuditEvent(
                        job_id=delivery_id,
                        source="github_webhook",
                        action="webhook_received",
                        status="rate_limited",
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            "event_type": event_type_header,
                            "repository": repository,
                        },
                    )
                )
            return web.Response(status=429, text="Rate limit exceeded")

        # Filter events
        try:
            event_type = WebhookEventType(event_type_header)
        except ValueError:
            logger.debug(f"Unknown event type: {event_type_header}")
            return web.Response(status=200, text="Event type not supported")

        if event_type not in self.config.enabled_events:
            logger.debug(f"Event type {event_type} not enabled, ignoring")
            return web.Response(status=200, text="Event type not enabled")

        # Create webhook event
        event = WebhookEvent(
            event_id=delivery_id,
            event_type=event_type,
            repository=repository,
            timestamp=datetime.now(timezone.utc),
            payload=payload,
        )

        # Queue for processing
        try:
            await asyncio.wait_for(
                self.event_queue.put(event),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Event queue full, dropping event {delivery_id}",
                extra={"queue_size": self.event_queue.qsize()},
            )
            if self.audit_logger:
                self.audit_logger.record(
                    AuditEvent(
                        job_id=delivery_id,
                        source="github_webhook",
                        action="webhook_received",
                        status="queue_full",
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            "event_type": event_type.value,
                            "repository": repository,
                        },
                    )
                )
            return web.Response(status=503, text="Queue full")

        # Log successful queuing
        logger.info(
            f"Webhook event queued: {event_type.value} for {repository}",
            extra={
                "event_id": delivery_id,
                "event_type": event_type.value,
                "repository": repository,
            },
        )
        if self.audit_logger:
            self.audit_logger.record(
                AuditEvent(
                    job_id=delivery_id,
                    source="github_webhook",
                    action="webhook_queued",
                    status="success",
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "event_type": event_type.value,
                        "repository": repository,
                    },
                )
            )

        return web.Response(status=200, text="Event queued")

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint.

        Returns:
            JSON response with server health status
        """
        return web.json_response(
            {
                "status": "healthy",
                "queue_size": self.event_queue.qsize(),
                "max_queue_size": self.config.queue_size,
                "rate_limiter_tracked_repos": len(self.rate_limiter.request_times),
            }
        )

    async def start(self) -> None:
        """Start webhook server.

        This starts the aiohttp server and background event processor.
        """
        if not self.config.enabled:
            logger.info("Webhook server is disabled, not starting")
            return

        # Setup app runner
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        # Create TCP site
        self.site = web.TCPSite(
            self.runner,
            self.config.listen_host,
            self.config.listen_port,
        )
        await self.site.start()

        logger.info(
            f"Webhook server listening on {self.config.listen_host}:{self.config.listen_port}",
            extra={
                "host": self.config.listen_host,
                "port": self.config.listen_port,
                "public_url": self.config.public_url,
            },
        )

        # Start event processor
        if self.config.process_async:
            self._processor_task = asyncio.create_task(self._process_events())
            logger.info("Webhook event processor started")

    async def stop(self) -> None:
        """Stop webhook server gracefully.

        This stops the server, drains the event queue, and cleans up resources.
        """
        logger.info("Stopping webhook server...")

        # Stop accepting new requests
        if self.site:
            await self.site.stop()

        # Cancel processor task
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Wait for queue to drain (with timeout)
        if not self.event_queue.empty():
            logger.info(
                f"Draining event queue ({self.event_queue.qsize()} events remaining)..."
            )
            try:
                await asyncio.wait_for(self.event_queue.join(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Queue drain timeout, some events may be lost")

        # Cleanup runner
        if self.runner:
            await self.runner.cleanup()

        logger.info("Webhook server stopped")

    async def _process_events(self) -> None:
        """Process queued webhook events asynchronously.

        This background task continuously pulls events from the queue and
        processes them using the event handler.
        """
        logger.info("Webhook event processor loop started")

        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()

                logger.debug(
                    f"Processing webhook event: {event.event_type.value} for {event.repository}",
                    extra={
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "repository": event.repository,
                    },
                )

                # Process event
                try:
                    await self.event_handler.handle_event(event)
                    event.mark_processed()

                    logger.info(
                        f"Webhook event processed successfully: {event.event_id}",
                        extra={
                            "event_id": event.event_id,
                            "event_type": event.event_type.value,
                            "repository": event.repository,
                        },
                    )

                    if self.audit_logger:
                        self.audit_logger.record(
                            AuditEvent(
                                job_id=event.event_id,
                                source="github_webhook",
                                action="webhook_processed",
                                status="success",
                                timestamp=datetime.now(timezone.utc),
                                metadata={
                                    "event_type": event.event_type.value,
                                    "repository": event.repository,
                                },
                            )
                        )

                except Exception as e:
                    error_msg = str(e)
                    event.mark_failed(error_msg)

                    logger.error(
                        f"Error processing webhook event {event.event_id}: {e}",
                        extra={
                            "event_id": event.event_id,
                            "event_type": event.event_type.value,
                            "repository": event.repository,
                        },
                        exc_info=True,
                    )

                    if self.audit_logger:
                        self.audit_logger.record(
                            AuditEvent(
                                job_id=event.event_id,
                                source="github_webhook",
                                action="webhook_processed",
                                status="failed",
                                timestamp=datetime.now(timezone.utc),
                                metadata={
                                    "event_type": event.event_type.value,
                                    "repository": event.repository,
                                    "error": error_msg,
                                },
                            )
                        )

            except asyncio.CancelledError:
                logger.info("Event processor task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in event processor: {e}", exc_info=True)
                # Continue processing despite errors
                await asyncio.sleep(1)
            finally:
                self.event_queue.task_done()


__all__ = ["WebhookServer"]
