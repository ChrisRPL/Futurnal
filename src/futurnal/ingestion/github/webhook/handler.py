"""Webhook event handler for routing GitHub events to appropriate processors.

This module implements the WebhookEventHandler which routes webhook events
to the appropriate handlers (sync engine, normalizers) based on event type.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..descriptor import GitHubRepositoryDescriptor, RepositoryRegistry
from ..incremental_sync import IncrementalSyncEngine
from ..issue_normalizer import IssueNormalizer
from ..pr_normalizer import PullRequestNormalizer
from ..sync_state_manager import SyncStateManager

from .models import WebhookEvent, WebhookEventType

if TYPE_CHECKING:
    from ..api_client_manager import GitHubAPIClientManager
    from futurnal.privacy.audit import AuditLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Webhook Event Handler
# ---------------------------------------------------------------------------


class WebhookEventHandler:
    """Routes webhook events to appropriate handlers.

    This class processes GitHub webhook events and routes them to the
    appropriate handlers:
    - Push events → IncrementalSyncEngine (branch sync)
    - PR events → PullRequestNormalizer (metadata update)
    - Issue events → IssueNormalizer (metadata update)
    - Create/Delete events → Branch/tag lifecycle management

    Attributes:
        sync_engine: Incremental sync engine for repository syncs
        issue_normalizer: Issue metadata normalizer
        pr_normalizer: Pull request metadata normalizer
        api_client_manager: GitHub API client manager
        state_manager: Sync state manager
        repository_registry: Repository descriptor registry
        audit_logger: Optional audit logger

    Example:
        >>> handler = WebhookEventHandler(
        ...     sync_engine=sync_engine,
        ...     issue_normalizer=issue_normalizer,
        ...     pr_normalizer=pr_normalizer,
        ...     repository_registry=registry
        ... )
        >>> await handler.handle_event(webhook_event)
    """

    def __init__(
        self,
        sync_engine: IncrementalSyncEngine,
        issue_normalizer: IssueNormalizer,
        pr_normalizer: PullRequestNormalizer,
        api_client_manager: GitHubAPIClientManager,
        state_manager: SyncStateManager,
        repository_registry: RepositoryRegistry,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """Initialize webhook event handler.

        Args:
            sync_engine: Incremental sync engine
            issue_normalizer: Issue normalizer
            pr_normalizer: Pull request normalizer
            api_client_manager: API client manager
            state_manager: Sync state manager
            repository_registry: Repository registry
            audit_logger: Optional audit logger
        """
        self.sync_engine = sync_engine
        self.issue_normalizer = issue_normalizer
        self.pr_normalizer = pr_normalizer
        self.api_client_manager = api_client_manager
        self.state_manager = state_manager
        self.repository_registry = repository_registry
        self.audit_logger = audit_logger

    async def handle_event(self, event: WebhookEvent) -> None:
        """Route event to appropriate handler.

        Args:
            event: Webhook event to process

        Raises:
            ValueError: If repository not found in registry
            Exception: If event processing fails
        """
        logger.info(
            f"Handling webhook event: {event.event_type.value} for {event.repository}",
            extra={
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "repository": event.repository,
            },
        )

        # Route to specific handler
        if event.event_type == WebhookEventType.PUSH:
            await self._handle_push(event)
        elif event.event_type == WebhookEventType.PULL_REQUEST:
            await self._handle_pull_request(event)
        elif event.event_type == WebhookEventType.ISSUES:
            await self._handle_issue(event)
        elif event.event_type == WebhookEventType.ISSUE_COMMENT:
            await self._handle_issue_comment(event)
        elif event.event_type == WebhookEventType.RELEASE:
            await self._handle_release(event)
        elif event.event_type == WebhookEventType.CREATE:
            await self._handle_create(event)
        elif event.event_type == WebhookEventType.DELETE:
            await self._handle_delete(event)
        elif event.event_type == WebhookEventType.REPOSITORY:
            await self._handle_repository(event)
        else:
            logger.debug(f"Unhandled event type: {event.event_type.value}")

    def _get_repository_descriptor(
        self, repo_full_name: str
    ) -> Optional[GitHubRepositoryDescriptor]:
        """Get repository descriptor from registry.

        Args:
            repo_full_name: Repository full name (owner/repo)

        Returns:
            Repository descriptor or None if not found
        """
        # Search registry for matching repository
        for descriptor in self.repository_registry.list():
            if descriptor.full_name == repo_full_name:
                return descriptor

        logger.warning(
            f"Repository {repo_full_name} not found in registry",
            extra={"repository": repo_full_name},
        )
        return None

    async def _handle_push(self, event: WebhookEvent) -> None:
        """Handle push event (new commits).

        This triggers an incremental sync for the affected branch.

        Args:
            event: Push webhook event
        """
        payload = event.payload
        ref = payload.get("ref", "")  # e.g., "refs/heads/main"
        branch = ref.replace("refs/heads/", "")

        logger.info(
            f"Processing push event for {event.repository}, branch: {branch}",
            extra={
                "repository": event.repository,
                "branch": branch,
                "ref": ref,
            },
        )

        # Get repository descriptor
        descriptor = self._get_repository_descriptor(event.repository)
        if not descriptor:
            raise ValueError(
                f"Repository {event.repository} not found in registry. "
                "Please register the repository before enabling webhooks."
            )

        # Check if branch is configured for sync
        if descriptor.branches and branch not in descriptor.branches:
            logger.debug(
                f"Branch {branch} not configured for sync, skipping",
                extra={"repository": event.repository, "branch": branch},
            )
            return

        # Trigger incremental sync for this branch
        logger.info(
            f"Triggering sync for {event.repository}, branch: {branch}",
            extra={"repository": event.repository, "branch": branch},
        )

        try:
            # Perform sync using IncrementalSyncEngine
            await self.sync_engine.sync_repository(
                descriptor=descriptor,
                # Note: sync_repository will detect new commits and process them
            )
            logger.info(
                f"Sync completed for {event.repository}, branch: {branch}",
                extra={"repository": event.repository, "branch": branch},
            )
        except Exception as e:
            logger.error(
                f"Failed to sync repository {event.repository}: {e}",
                extra={"repository": event.repository, "branch": branch},
                exc_info=True,
            )
            raise

    async def _handle_pull_request(self, event: WebhookEvent) -> None:
        """Handle pull request event.

        This normalizes PR metadata and updates the PKG.

        Args:
            event: Pull request webhook event
        """
        payload = event.payload
        action = payload.get("action", "")  # opened, closed, synchronize, etc.
        pr_number = payload.get("pull_request", {}).get("number")
        repo = event.repository

        logger.info(
            f"Processing PR event: {action} for {repo} #{pr_number}",
            extra={
                "repository": repo,
                "pr_number": pr_number,
                "action": action,
            },
        )

        # Get repository descriptor
        descriptor = self._get_repository_descriptor(repo)
        if not descriptor:
            raise ValueError(f"Repository {repo} not found in registry")

        # Normalize and update PR
        owner, repo_name = repo.split("/")

        try:
            pr_metadata = await self.pr_normalizer.normalize_pull_request(
                repo_owner=owner,
                repo_name=repo_name,
                pr_number=pr_number,
                credential_id=descriptor.credential_id,
            )

            logger.info(
                f"PR metadata normalized for {repo} #{pr_number}",
                extra={
                    "repository": repo,
                    "pr_number": pr_number,
                    "pr_state": pr_metadata.state.value,
                },
            )

            # TODO: Send to PKG via element sink (future enhancement)
            # For now, normalizer handles PKG updates

        except Exception as e:
            logger.error(
                f"Failed to normalize PR {repo} #{pr_number}: {e}",
                extra={"repository": repo, "pr_number": pr_number},
                exc_info=True,
            )
            raise

    async def _handle_issue(self, event: WebhookEvent) -> None:
        """Handle issue event.

        This normalizes issue metadata and updates the PKG.

        Args:
            event: Issue webhook event
        """
        payload = event.payload
        action = payload.get("action", "")  # opened, closed, edited, etc.
        issue_number = payload.get("issue", {}).get("number")
        repo = event.repository

        logger.info(
            f"Processing issue event: {action} for {repo} #{issue_number}",
            extra={
                "repository": repo,
                "issue_number": issue_number,
                "action": action,
            },
        )

        # Get repository descriptor
        descriptor = self._get_repository_descriptor(repo)
        if not descriptor:
            raise ValueError(f"Repository {repo} not found in registry")

        # Normalize and update issue
        owner, repo_name = repo.split("/")

        try:
            issue_metadata = await self.issue_normalizer.normalize_issue(
                repo_owner=owner,
                repo_name=repo_name,
                issue_number=issue_number,
                credential_id=descriptor.credential_id,
            )

            logger.info(
                f"Issue metadata normalized for {repo} #{issue_number}",
                extra={
                    "repository": repo,
                    "issue_number": issue_number,
                    "issue_state": issue_metadata.state.value,
                },
            )

            # TODO: Send to PKG via element sink (future enhancement)

        except Exception as e:
            logger.error(
                f"Failed to normalize issue {repo} #{issue_number}: {e}",
                extra={"repository": repo, "issue_number": issue_number},
                exc_info=True,
            )
            raise

    async def _handle_issue_comment(self, event: WebhookEvent) -> None:
        """Handle issue comment event.

        This could update issue metadata to reflect new comments.

        Args:
            event: Issue comment webhook event
        """
        payload = event.payload
        issue_number = payload.get("issue", {}).get("number")
        repo = event.repository

        logger.info(
            f"Processing issue comment for {repo} #{issue_number}",
            extra={"repository": repo, "issue_number": issue_number},
        )

        # Re-normalize the issue to pick up new comment
        # This reuses the issue handler logic
        await self._handle_issue(event)

    async def _handle_release(self, event: WebhookEvent) -> None:
        """Handle release event.

        This could be used for release tracking in the PKG.

        Args:
            event: Release webhook event
        """
        payload = event.payload
        action = payload.get("action", "")
        release_tag = payload.get("release", {}).get("tag_name", "")
        repo = event.repository

        logger.info(
            f"Processing release event: {action} for {repo}, tag: {release_tag}",
            extra={"repository": repo, "release_tag": release_tag, "action": action},
        )

        # TODO: Implement release tracking in PKG

    async def _handle_create(self, event: WebhookEvent) -> None:
        """Handle branch/tag creation.

        This could trigger sync setup for new branches.

        Args:
            event: Create webhook event
        """
        payload = event.payload
        ref_type = payload.get("ref_type", "")  # "branch" or "tag"
        ref = payload.get("ref", "")
        repo = event.repository

        logger.info(
            f"Processing create event: {ref_type} '{ref}' for {repo}",
            extra={"repository": repo, "ref_type": ref_type, "ref": ref},
        )

        if ref_type == "branch":
            # Check if we should sync this new branch
            descriptor = self._get_repository_descriptor(repo)
            if descriptor and (not descriptor.branches or ref in descriptor.branches):
                logger.info(
                    f"New branch {ref} created, could trigger initial sync",
                    extra={"repository": repo, "branch": ref},
                )
                # TODO: Trigger initial branch sync if configured

    async def _handle_delete(self, event: WebhookEvent) -> None:
        """Handle branch/tag deletion.

        This could clean up sync state for deleted branches.

        Args:
            event: Delete webhook event
        """
        payload = event.payload
        ref_type = payload.get("ref_type", "")
        ref = payload.get("ref", "")
        repo = event.repository

        logger.info(
            f"Processing delete event: {ref_type} '{ref}' for {repo}",
            extra={"repository": repo, "ref_type": ref_type, "ref": ref},
        )

        if ref_type == "branch":
            # Clean up branch state
            descriptor = self._get_repository_descriptor(repo)
            if descriptor:
                logger.info(
                    f"Branch {ref} deleted, could clean up sync state",
                    extra={"repository": repo, "branch": ref},
                )
                # TODO: Clean up branch-specific sync state

    async def _handle_repository(self, event: WebhookEvent) -> None:
        """Handle repository settings change.

        This could update repository descriptor metadata.

        Args:
            event: Repository webhook event
        """
        payload = event.payload
        action = payload.get("action", "")
        repo = event.repository

        logger.info(
            f"Processing repository event: {action} for {repo}",
            extra={"repository": repo, "action": action},
        )

        # TODO: Update repository descriptor if settings changed


__all__ = ["WebhookEventHandler"]
