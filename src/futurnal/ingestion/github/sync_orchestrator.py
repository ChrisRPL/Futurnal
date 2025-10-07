"""High-level orchestration for GitHub repository synchronization.

This module provides the main orchestrator for repository sync operations,
coordinating between GraphQL and Git Clone modes, managing state, and
providing intelligent mode selection based on repository characteristics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .api_client_manager import GitHubAPIClientManager
from .credential_manager import GitHubCredentialManager
from .descriptor import GitHubRepositoryDescriptor, SyncMode
from .git_clone_sync import GitCloneRepositorySync
from .graphql_sync import GraphQLRepositorySync
from .sync_models import DiskSpaceEstimate, SyncResult, SyncState, SyncStrategy
from .sync_state_manager import SyncStateManager
from .sync_utils import (
    check_disk_space_sufficient,
    estimate_git_clone_size,
    format_sync_statistics,
    get_default_clone_base_dir,
)

logger = logging.getLogger(__name__)


@dataclass
class GitHubSyncOrchestrator:
    """High-level orchestrator for GitHub repository synchronization."""

    credential_manager: GitHubCredentialManager
    api_client_manager: GitHubAPIClientManager
    state_manager: SyncStateManager
    clone_base_dir: Optional[Path] = None
    workspace_dir: Optional[Path] = None

    def __init__(
        self,
        credential_manager: GitHubCredentialManager,
        api_client_manager: GitHubAPIClientManager,
        state_manager: Optional[SyncStateManager] = None,
        clone_base_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
    ):
        """Initialize sync orchestrator.

        Args:
            credential_manager: Credential manager
            api_client_manager: API client manager
            state_manager: State manager (default: new instance)
            clone_base_dir: Base directory for git clones
            workspace_dir: Workspace directory for GraphQL mode
        """
        self.credential_manager = credential_manager
        self.api_client_manager = api_client_manager

        if state_manager is None:
            state_manager = SyncStateManager()
        self.state_manager = state_manager

        if clone_base_dir is None:
            clone_base_dir = get_default_clone_base_dir()
        self.clone_base_dir = clone_base_dir

        if workspace_dir is None:
            workspace_dir = (
                Path.home() / ".futurnal" / "workspace" / "github"
            )
        self.workspace_dir = workspace_dir

    async def sync_repository(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: Optional[SyncStrategy] = None,
        force_mode: Optional[SyncMode] = None,
    ) -> SyncResult:
        """Sync repository using configured or recommended mode.

        Args:
            descriptor: Repository descriptor
            strategy: Optional sync strategy (uses defaults if not provided)
            force_mode: Force specific sync mode (overrides descriptor)

        Returns:
            SyncResult with sync statistics
        """
        # Get or create sync state
        state = self.state_manager.get_or_create(
            repo_id=descriptor.id,
            sync_mode=descriptor.sync_mode.value,
            local_clone_path=descriptor.local_clone_path,
        )

        # Build strategy with defaults if not provided
        if strategy is None:
            strategy = self._build_default_strategy(descriptor)

        # Determine sync mode
        if force_mode:
            sync_mode = force_mode
            logger.info(f"Using forced sync mode: {sync_mode.value}")
        else:
            sync_mode = descriptor.sync_mode

        logger.info(
            f"Starting sync for {descriptor.full_name} "
            f"(mode: {sync_mode.value})"
        )

        # Check disk space if using Git Clone mode
        if sync_mode == SyncMode.GIT_CLONE:
            disk_check = self._check_disk_space(descriptor, strategy)
            if not disk_check.is_sufficient:
                error_msg = (
                    f"Insufficient disk space: need {disk_check.estimated_size_gb:.2f} GB, "
                    f"have {disk_check.available_space_gb:.2f} GB available"
                )
                logger.error(error_msg)

                # Return failed result
                return SyncResult(
                    repo_id=descriptor.id,
                    sync_mode=sync_mode.value,
                    status="failed",
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    error_message=error_msg,
                )

        # Mark sync as started
        self.state_manager.mark_sync_started(descriptor.id)

        try:
            # Dispatch to appropriate sync implementation
            if sync_mode == SyncMode.GRAPHQL_API:
                result = await self._sync_graphql(descriptor, strategy, state)
            else:
                result = await self._sync_git_clone(descriptor, strategy, state)

            # Update state based on result
            if result.is_success():
                self.state_manager.mark_sync_completed(
                    repo_id=descriptor.id,
                    files_synced=result.files_synced,
                    bytes_synced=result.bytes_synced,
                    commits=result.commits_processed,
                )
            else:
                self.state_manager.mark_sync_failed(
                    repo_id=descriptor.id,
                    error_count=result.files_failed,
                )

            # Log statistics
            stats = format_sync_statistics(result)
            logger.info(f"Sync completed:\n{stats}")

            return result

        except Exception as e:
            logger.error(f"Sync failed with exception: {e}", exc_info=True)

            self.state_manager.mark_sync_failed(descriptor.id)

            return SyncResult(
                repo_id=descriptor.id,
                sync_mode=sync_mode.value,
                status="failed",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )

    async def _sync_graphql(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
        state: SyncState,
    ) -> SyncResult:
        """Sync using GraphQL API mode.

        Args:
            descriptor: Repository descriptor
            strategy: Sync strategy
            state: Current sync state

        Returns:
            SyncResult
        """
        sync = GraphQLRepositorySync(
            api_client_manager=self.api_client_manager,
            workspace_dir=self.workspace_dir,
        )

        return await sync.sync_repository(descriptor, strategy, state)

    async def _sync_git_clone(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
        state: SyncState,
    ) -> SyncResult:
        """Sync using Git Clone mode.

        Args:
            descriptor: Repository descriptor
            strategy: Sync strategy
            state: Current sync state

        Returns:
            SyncResult
        """
        sync = GitCloneRepositorySync(
            credential_manager=self.credential_manager,
            clone_base_dir=self.clone_base_dir,
        )

        return await sync.sync_repository(descriptor, strategy, state)

    def recommend_sync_mode(
        self,
        repo_size_kb: int,
        file_count: int,
        available_disk_gb: float,
    ) -> SyncMode:
        """Recommend sync mode based on repository characteristics.

        Args:
            repo_size_kb: Repository size in kilobytes
            file_count: Number of files in repository
            available_disk_gb: Available disk space in GB

        Returns:
            Recommended sync mode
        """
        repo_size_gb = repo_size_kb / (1024 * 1024)

        # Large repository with limited disk space
        if repo_size_gb > available_disk_gb * 0.5:
            logger.info(
                f"Recommending GraphQL mode: repo size ({repo_size_gb:.2f} GB) "
                f"exceeds 50% of available space"
            )
            return SyncMode.GRAPHQL_API

        # Small repository - clone is efficient
        if repo_size_gb < 0.1:  # < 100 MB
            logger.info(
                f"Recommending Git Clone mode: small repo ({repo_size_gb:.2f} GB)"
            )
            return SyncMode.GIT_CLONE

        # Many files - GraphQL API may hit rate limits
        if file_count > 10000:
            logger.info(
                f"Recommending Git Clone mode: large file count ({file_count})"
            )
            return SyncMode.GIT_CLONE

        # Default to GraphQL for most cases
        logger.info("Recommending GraphQL mode (default)")
        return SyncMode.GRAPHQL_API

    def _check_disk_space(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
    ) -> DiskSpaceEstimate:
        """Check if sufficient disk space is available.

        Args:
            descriptor: Repository descriptor
            strategy: Sync strategy

        Returns:
            DiskSpaceEstimate
        """
        # Estimate required size (use a conservative estimate if not available)
        # This would ideally use repository metadata from GitHub API
        estimated_size = 100 * 1024 * 1024  # 100 MB default

        # Adjust based on clone depth
        if strategy.clone_depth:
            estimated_size = int(estimated_size * 0.5)  # Shallow clone ~50% size

        # Check against clone base directory
        target_dir = descriptor.local_clone_path or self.clone_base_dir

        estimate = check_disk_space_sufficient(
            required_bytes=estimated_size,
            path=target_dir,
            buffer_percentage=0.2,
        )

        # Fill in repo information
        estimate.repo_id = descriptor.id
        estimate.sync_mode = "git_clone"

        return estimate

    def estimate_sync_size(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
        repo_size_kb: int,
    ) -> DiskSpaceEstimate:
        """Estimate disk space required for sync.

        Args:
            descriptor: Repository descriptor
            strategy: Sync strategy
            repo_size_kb: Repository size from GitHub API

        Returns:
            DiskSpaceEstimate
        """
        if descriptor.sync_mode == SyncMode.GRAPHQL_API:
            # GraphQL mode: minimal disk usage (just cached files)
            estimated_bytes = int(repo_size_kb * 1024 * 0.1)  # ~10% of repo
        else:
            # Git Clone mode: full repository
            estimated_bytes = estimate_git_clone_size(
                repo_size_kb, strategy.clone_depth
            )

        target_dir = (
            descriptor.local_clone_path
            if descriptor.sync_mode == SyncMode.GIT_CLONE
            else self.workspace_dir
        )

        estimate = check_disk_space_sufficient(
            required_bytes=estimated_bytes,
            path=target_dir,
            buffer_percentage=0.2,
        )

        estimate.repo_id = descriptor.id
        estimate.sync_mode = descriptor.sync_mode.value

        return estimate

    def _build_default_strategy(
        self,
        descriptor: GitHubRepositoryDescriptor,
    ) -> SyncStrategy:
        """Build default sync strategy from descriptor.

        Args:
            descriptor: Repository descriptor

        Returns:
            SyncStrategy with defaults
        """
        return SyncStrategy(
            branches=descriptor.branches or ["main"],
            include_all_branches=False,
            include_tags=descriptor.sync_mode == SyncMode.GIT_CLONE,
            file_patterns=descriptor.include_paths,
            exclude_patterns=descriptor.exclude_paths,
            max_file_size_mb=descriptor.privacy_settings.max_file_size_mb,
            fetch_file_content=True,
            batch_size=10,
            clone_depth=descriptor.clone_depth,
            use_sparse_checkout=descriptor.sparse_checkout,
            clone_submodules=False,
            single_branch=True,
        )

    def get_sync_status(self, repo_id: str) -> Optional[SyncState]:
        """Get current sync status for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            SyncState if exists, None otherwise
        """
        return self.state_manager.load(repo_id)

    def list_all_syncs(self) -> list[SyncState]:
        """List all repository sync states.

        Returns:
            List of sync states
        """
        return self.state_manager.list_all()

    def cleanup_old_syncs(self, days: int = 90) -> int:
        """Clean up old sync states.

        Args:
            days: Days to retain

        Returns:
            Number of states cleaned up
        """
        return self.state_manager.cleanup_old_states(days)


__all__ = [
    "GitHubSyncOrchestrator",
]
