"""GitHub webhook configuration via API.

This module manages GitHub webhooks via the GitHub REST API, allowing
users to create, list, update, and delete webhooks programmatically.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..api_client_manager import GitHubAPIClientManager
from ..descriptor import GitHubRepositoryDescriptor, RepositoryRegistry

from .models import WebhookEventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Webhook Configurator
# ---------------------------------------------------------------------------


class GitHubWebhookConfigurator:
    """Manages GitHub webhooks via REST API.

    This class provides methods to create, list, update, test, and delete
    GitHub webhooks for repositories. It integrates with the existing
    GitHubAPIClientManager for authenticated requests.

    Attributes:
        api_client_manager: GitHub API client manager
        repository_registry: Repository descriptor registry

    Example:
        >>> configurator = GitHubWebhookConfigurator(
        ...     api_client_manager=api_manager,
        ...     repository_registry=registry
        ... )
        >>> webhook = await configurator.configure_webhook(
        ...     descriptor=repo_descriptor,
        ...     webhook_url="https://example.com/webhook/github",
        ...     secret="webhook-secret"
        ... )
    """

    def __init__(
        self,
        api_client_manager: GitHubAPIClientManager,
        repository_registry: RepositoryRegistry,
    ):
        """Initialize webhook configurator.

        Args:
            api_client_manager: GitHub API client manager
            repository_registry: Repository descriptor registry
        """
        self.api_client_manager = api_client_manager
        self.repository_registry = repository_registry

    async def configure_webhook(
        self,
        descriptor: GitHubRepositoryDescriptor,
        webhook_url: str,
        secret: str,
        events: Optional[List[WebhookEventType]] = None,
        active: bool = True,
    ) -> Dict[str, Any]:
        """Configure webhook via GitHub API.

        This creates a new webhook for the repository with the specified
        configuration. If a webhook with the same URL already exists, it
        will be updated instead.

        Args:
            descriptor: Repository descriptor
            webhook_url: Public URL for webhook delivery
            secret: HMAC secret for signature verification
            events: List of event types to subscribe to (default: push, pull_request, issues)
            active: Whether webhook should be active

        Returns:
            Webhook configuration dict from GitHub API

        Raises:
            Exception: If API request fails

        Example:
            >>> webhook = await configurator.configure_webhook(
            ...     descriptor=repo,
            ...     webhook_url="https://example.com/webhook/github",
            ...     secret="my-secret",
            ...     events=[WebhookEventType.PUSH, WebhookEventType.PULL_REQUEST]
            ... )
        """
        if events is None:
            events = [
                WebhookEventType.PUSH,
                WebhookEventType.PULL_REQUEST,
                WebhookEventType.ISSUES,
                WebhookEventType.ISSUE_COMMENT,
                WebhookEventType.RELEASE,
                WebhookEventType.CREATE,
                WebhookEventType.DELETE,
            ]

        # Convert events to string list
        event_names = [event.value for event in events]

        # Check if webhook already exists
        existing_webhooks = await self.list_webhooks(descriptor)
        existing_webhook = None
        for hook in existing_webhooks:
            if hook.get("config", {}).get("url") == webhook_url:
                existing_webhook = hook
                break

        if existing_webhook:
            logger.info(
                f"Webhook already exists for {descriptor.full_name}, updating...",
                extra={"repository": descriptor.full_name, "webhook_id": existing_webhook["id"]},
            )
            return await self.update_webhook(
                descriptor=descriptor,
                webhook_id=existing_webhook["id"],
                webhook_url=webhook_url,
                secret=secret,
                events=event_names,
                active=active,
            )

        # Create new webhook
        payload = {
            "config": {
                "url": webhook_url,
                "content_type": "json",
                "secret": secret,
                "insecure_ssl": "0",  # Require HTTPS
            },
            "events": event_names,
            "active": active,
        }

        logger.info(
            f"Creating webhook for {descriptor.full_name}",
            extra={
                "repository": descriptor.full_name,
                "webhook_url": webhook_url,
                "events": event_names,
            },
        )

        # Get API client
        client = await self.api_client_manager.get_client(descriptor.credential_id)

        # Create webhook via REST API
        response = await client.rest_request(
            method="POST",
            endpoint=f"/repos/{descriptor.owner}/{descriptor.repo}/hooks",
            data=payload,
        )

        logger.info(
            f"Webhook created successfully for {descriptor.full_name}",
            extra={
                "repository": descriptor.full_name,
                "webhook_id": response.get("id"),
            },
        )

        return response

    async def list_webhooks(
        self,
        descriptor: GitHubRepositoryDescriptor,
    ) -> List[Dict[str, Any]]:
        """List all webhooks for repository.

        Args:
            descriptor: Repository descriptor

        Returns:
            List of webhook configuration dicts

        Example:
            >>> webhooks = await configurator.list_webhooks(descriptor)
            >>> for webhook in webhooks:
            ...     print(f"Webhook {webhook['id']}: {webhook['config']['url']}")
        """
        logger.debug(
            f"Listing webhooks for {descriptor.full_name}",
            extra={"repository": descriptor.full_name},
        )

        # Get API client
        client = await self.api_client_manager.get_client(descriptor.credential_id)

        # List webhooks via REST API
        response = await client.rest_request(
            method="GET",
            endpoint=f"/repos/{descriptor.owner}/{descriptor.repo}/hooks",
        )

        logger.debug(
            f"Found {len(response)} webhooks for {descriptor.full_name}",
            extra={"repository": descriptor.full_name, "webhook_count": len(response)},
        )

        return response

    async def get_webhook(
        self,
        descriptor: GitHubRepositoryDescriptor,
        webhook_id: int,
    ) -> Dict[str, Any]:
        """Get specific webhook configuration.

        Args:
            descriptor: Repository descriptor
            webhook_id: Webhook ID from GitHub

        Returns:
            Webhook configuration dict

        Example:
            >>> webhook = await configurator.get_webhook(descriptor, webhook_id=12345)
        """
        logger.debug(
            f"Getting webhook {webhook_id} for {descriptor.full_name}",
            extra={"repository": descriptor.full_name, "webhook_id": webhook_id},
        )

        # Get API client
        client = await self.api_client_manager.get_client(descriptor.credential_id)

        # Get webhook via REST API
        response = await client.rest_request(
            method="GET",
            endpoint=f"/repos/{descriptor.owner}/{descriptor.repo}/hooks/{webhook_id}",
        )

        return response

    async def update_webhook(
        self,
        descriptor: GitHubRepositoryDescriptor,
        webhook_id: int,
        webhook_url: Optional[str] = None,
        secret: Optional[str] = None,
        events: Optional[List[str]] = None,
        active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update existing webhook configuration.

        Args:
            descriptor: Repository descriptor
            webhook_id: Webhook ID to update
            webhook_url: New webhook URL (optional)
            secret: New HMAC secret (optional)
            events: New list of event types (optional)
            active: New active status (optional)

        Returns:
            Updated webhook configuration dict

        Example:
            >>> webhook = await configurator.update_webhook(
            ...     descriptor=repo,
            ...     webhook_id=12345,
            ...     active=False
            ... )
        """
        logger.info(
            f"Updating webhook {webhook_id} for {descriptor.full_name}",
            extra={"repository": descriptor.full_name, "webhook_id": webhook_id},
        )

        # Get current webhook config
        current = await self.get_webhook(descriptor, webhook_id)

        # Build update payload
        payload: Dict[str, Any] = {}

        if webhook_url is not None or secret is not None:
            config = current.get("config", {}).copy()
            if webhook_url is not None:
                config["url"] = webhook_url
            if secret is not None:
                config["secret"] = secret
            payload["config"] = config

        if events is not None:
            payload["events"] = events

        if active is not None:
            payload["active"] = active

        # Get API client
        client = await self.api_client_manager.get_client(descriptor.credential_id)

        # Update webhook via REST API
        response = await client.rest_request(
            method="PATCH",
            endpoint=f"/repos/{descriptor.owner}/{descriptor.repo}/hooks/{webhook_id}",
            data=payload,
        )

        logger.info(
            f"Webhook {webhook_id} updated successfully",
            extra={"repository": descriptor.full_name, "webhook_id": webhook_id},
        )

        return response

    async def delete_webhook(
        self,
        descriptor: GitHubRepositoryDescriptor,
        webhook_id: int,
    ) -> None:
        """Delete webhook.

        Args:
            descriptor: Repository descriptor
            webhook_id: Webhook ID to delete

        Example:
            >>> await configurator.delete_webhook(descriptor, webhook_id=12345)
        """
        logger.info(
            f"Deleting webhook {webhook_id} for {descriptor.full_name}",
            extra={"repository": descriptor.full_name, "webhook_id": webhook_id},
        )

        # Get API client
        client = await self.api_client_manager.get_client(descriptor.credential_id)

        # Delete webhook via REST API
        await client.rest_request(
            method="DELETE",
            endpoint=f"/repos/{descriptor.owner}/{descriptor.repo}/hooks/{webhook_id}",
        )

        logger.info(
            f"Webhook {webhook_id} deleted successfully",
            extra={"repository": descriptor.full_name, "webhook_id": webhook_id},
        )

    async def test_webhook(
        self,
        descriptor: GitHubRepositoryDescriptor,
        webhook_id: int,
    ) -> None:
        """Send test webhook delivery.

        This triggers GitHub to send a test ping event to the webhook URL.

        Args:
            descriptor: Repository descriptor
            webhook_id: Webhook ID to test

        Example:
            >>> await configurator.test_webhook(descriptor, webhook_id=12345)
        """
        logger.info(
            f"Testing webhook {webhook_id} for {descriptor.full_name}",
            extra={"repository": descriptor.full_name, "webhook_id": webhook_id},
        )

        # Get API client
        client = await self.api_client_manager.get_client(descriptor.credential_id)

        # Test webhook via REST API
        await client.rest_request(
            method="POST",
            endpoint=f"/repos/{descriptor.owner}/{descriptor.repo}/hooks/{webhook_id}/tests",
        )

        logger.info(
            f"Test webhook sent for {descriptor.full_name}",
            extra={"repository": descriptor.full_name, "webhook_id": webhook_id},
        )


__all__ = ["GitHubWebhookConfigurator"]
