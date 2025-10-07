"""Git clone-based repository synchronization implementation.

This module implements repository sync using git clone operations, providing
full-fidelity access to repository history, offline capabilities, and support
for advanced git features like shallow clones and sparse checkout.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .credential_manager import (
    GitHubCredentialManager,
    OAuthToken,
    OAuthTokens,
    PersonalAccessToken,
)
from .descriptor import GitHubRepositoryDescriptor
from .sync_models import SyncResult, SyncState, SyncStatus, SyncStrategy
from .sync_utils import get_git_binary_path, validate_git_installed

logger = logging.getLogger(__name__)


@dataclass
class GitCloneRepositorySync:
    """Sync repository using git clone."""

    credential_manager: GitHubCredentialManager
    clone_base_dir: Path
    git_binary: str = "git"

    def __init__(
        self,
        credential_manager: GitHubCredentialManager,
        clone_base_dir: Optional[Path] = None,
        git_binary: Optional[str] = None,
    ):
        """Initialize Git clone sync.

        Args:
            credential_manager: Credential manager for token retrieval
            clone_base_dir: Base directory for cloned repositories
            git_binary: Path to git binary (default: "git")
        """
        self.credential_manager = credential_manager

        if clone_base_dir is None:
            clone_base_dir = (
                Path.home() / ".futurnal" / "repositories" / "github"
            )

        self.clone_base_dir = clone_base_dir
        self.clone_base_dir.mkdir(parents=True, exist_ok=True)

        # Validate git installation
        if git_binary is None:
            # Try to find git
            git_path = get_git_binary_path()
            self.git_binary = git_path or "git"
        else:
            self.git_binary = git_binary

        is_installed, version = validate_git_installed()
        if not is_installed:
            raise RuntimeError(
                "Git is not installed or not accessible. "
                "Please install git to use clone mode."
            )

        logger.info(f"Using git binary: {self.git_binary} ({version})")

    async def sync_repository(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
        state: Optional[SyncState] = None,
    ) -> SyncResult:
        """Sync repository using git clone.

        Args:
            descriptor: Repository descriptor
            strategy: Sync strategy configuration
            state: Optional current sync state

        Returns:
            SyncResult with sync statistics
        """
        started_at = datetime.now(timezone.utc)
        result = SyncResult(
            repo_id=descriptor.id,
            sync_mode="git_clone",
            status=SyncStatus.IN_PROGRESS,
            started_at=started_at,
        )

        # Determine clone directory
        if descriptor.local_clone_path:
            repo_dir = descriptor.local_clone_path
        else:
            repo_dir = self.clone_base_dir / descriptor.id

        try:
            logger.info(f"Starting Git clone sync for {descriptor.full_name}")

            if repo_dir.exists() and (repo_dir / ".git").exists():
                # Update existing repository
                logger.info(f"Updating existing clone at {repo_dir}")
                await self._update_repository(repo_dir, descriptor, strategy)
            else:
                # Initial clone
                logger.info(f"Cloning repository to {repo_dir}")
                await self._clone_repository(repo_dir, descriptor, strategy)

            # Count files and calculate size
            file_count = self._count_files(repo_dir)
            repo_size = self._calculate_repo_size(repo_dir)

            result.files_synced = file_count
            result.bytes_synced = repo_size
            result.branches_synced = strategy.branches
            result.status = SyncStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc)

            if result.completed_at and result.started_at:
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()

            logger.info(
                f"Git clone sync completed: {file_count} files, "
                f"{result.bytes_synced_mb:.2f} MB in {result.duration_seconds:.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Git clone sync failed: {e}", exc_info=True)

            result.status = SyncStatus.FAILED
            result.error_message = str(e)
            result.error_details = {"exception_type": type(e).__name__}
            result.completed_at = datetime.now(timezone.utc)

            if result.completed_at and result.started_at:
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()

            return result

    async def _clone_repository(
        self,
        repo_dir: Path,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
    ) -> None:
        """Perform initial repository clone.

        Args:
            repo_dir: Target directory for clone
            descriptor: Repository descriptor
            strategy: Sync strategy
        """
        # Build clone URL with authentication
        clone_url = self._build_clone_url(descriptor)

        # Build git clone command
        cmd = [self.git_binary, "clone"]

        # Shallow clone if configured
        if strategy.clone_depth:
            cmd.extend(["--depth", str(strategy.clone_depth)])
            logger.debug(f"Using shallow clone with depth {strategy.clone_depth}")

        # Single branch if not cloning all
        if strategy.single_branch and strategy.branches:
            primary_branch = strategy.branches[0]
            cmd.extend(["--branch", primary_branch, "--single-branch"])
            logger.debug(f"Cloning single branch: {primary_branch}")

        # No tags if not needed
        if not strategy.include_tags:
            cmd.append("--no-tags")
            logger.debug("Skipping tags")

        # Sparse checkout preparation
        if strategy.use_sparse_checkout:
            cmd.append("--filter=blob:none")
            cmd.append("--sparse")
            logger.debug("Using sparse checkout")

        # Add URL and target directory
        cmd.extend([clone_url, str(repo_dir)])

        # Execute clone
        logger.info(f"Executing: git clone [URL_REDACTED] {repo_dir}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                # Redact credentials from error message
                error_msg = self._redact_credentials(error_msg, clone_url)
                raise RuntimeError(f"Git clone failed: {error_msg}")

            logger.info("Clone completed successfully")

            # Configure sparse checkout if needed
            if strategy.use_sparse_checkout:
                await self._configure_sparse_checkout(repo_dir, strategy)

        except Exception as e:
            # Clean up on failure
            if repo_dir.exists():
                try:
                    shutil.rmtree(repo_dir)
                except Exception:
                    pass
            raise

    async def _update_repository(
        self,
        repo_dir: Path,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
    ) -> None:
        """Update existing cloned repository.

        Args:
            repo_dir: Repository directory
            descriptor: Repository descriptor
            strategy: Sync strategy
        """
        logger.info("Fetching latest changes")

        # Fetch latest changes
        cmd = [self.git_binary, "-C", str(repo_dir), "fetch", "origin"]

        if strategy.include_tags:
            cmd.append("--tags")

        if strategy.clone_depth:
            cmd.extend(["--depth", str(strategy.clone_depth)])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"Git fetch failed: {error_msg}")

            logger.debug("Fetch completed")

            # Update working tree for each branch
            for branch in strategy.branches:
                await self._update_branch(repo_dir, branch)

        except Exception as e:
            logger.error(f"Update failed: {e}")
            raise

    async def _update_branch(
        self,
        repo_dir: Path,
        branch: str,
    ) -> None:
        """Update working tree for specific branch.

        Args:
            repo_dir: Repository directory
            branch: Branch name
        """
        logger.debug(f"Updating branch: {branch}")

        # Checkout branch
        cmd_checkout = [
            self.git_binary,
            "-C",
            str(repo_dir),
            "checkout",
            branch,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd_checkout,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(f"Failed to checkout branch {branch}")
            return

        # Pull latest changes
        cmd_pull = [
            self.git_binary,
            "-C",
            str(repo_dir),
            "pull",
            "origin",
            branch,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd_pull,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(f"Failed to pull branch {branch}")
        else:
            logger.debug(f"Branch {branch} updated successfully")

    async def _configure_sparse_checkout(
        self,
        repo_dir: Path,
        strategy: SyncStrategy,
    ) -> None:
        """Configure sparse checkout patterns.

        Args:
            repo_dir: Repository directory
            strategy: Sync strategy with patterns
        """
        logger.info("Configuring sparse checkout")

        patterns = strategy.file_patterns or ["/*"]

        # Initialize sparse checkout
        cmd_init = [
            self.git_binary,
            "-C",
            str(repo_dir),
            "sparse-checkout",
            "init",
            "--cone",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd_init,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await process.communicate()

        # Set sparse checkout patterns
        cmd_set = [
            self.git_binary,
            "-C",
            str(repo_dir),
            "sparse-checkout",
            "set",
        ]
        cmd_set.extend(patterns)

        process = await asyncio.create_subprocess_exec(
            *cmd_set,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace")
            logger.warning(f"Sparse checkout configuration failed: {error_msg}")
        else:
            logger.info(f"Sparse checkout configured with {len(patterns)} patterns")

    def _build_clone_url(
        self,
        descriptor: GitHubRepositoryDescriptor,
    ) -> str:
        """Build authenticated clone URL.

        Args:
            descriptor: Repository descriptor

        Returns:
            Authenticated HTTPS clone URL
        """
        # Retrieve credentials
        credentials = self.credential_manager.retrieve_credentials(
            descriptor.credential_id
        )

        # Extract token
        if isinstance(credentials, OAuthTokens):
            token = credentials.access_token
        elif isinstance(credentials, OAuthToken):
            token = credentials.token
        elif isinstance(credentials, PersonalAccessToken):
            token = credentials.token
        else:
            raise ValueError(f"Unsupported credential type: {type(credentials)}")

        # Build URL with token
        if descriptor.github_host == "github.com":
            return f"https://{token}@github.com/{descriptor.owner}/{descriptor.repo}.git"
        else:
            # GitHub Enterprise
            return f"https://{token}@{descriptor.github_host}/{descriptor.owner}/{descriptor.repo}.git"

    def _count_files(self, repo_dir: Path) -> int:
        """Count files in repository (excluding .git).

        Args:
            repo_dir: Repository directory

        Returns:
            Number of files
        """
        count = 0
        try:
            for item in repo_dir.rglob("*"):
                if item.is_file() and ".git" not in item.parts:
                    count += 1
        except Exception as e:
            logger.warning(f"Failed to count files: {e}")

        return count

    def _calculate_repo_size(self, repo_dir: Path) -> int:
        """Calculate repository size in bytes.

        Args:
            repo_dir: Repository directory

        Returns:
            Size in bytes
        """
        total_size = 0
        try:
            for item in repo_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate size: {e}")

        return total_size

    def _redact_credentials(self, message: str, clone_url: str) -> str:
        """Redact credentials from error messages.

        Args:
            message: Error message
            clone_url: Clone URL containing credentials

        Returns:
            Redacted message
        """
        # Extract token from URL
        if "@" in clone_url:
            token_part = clone_url.split("@")[0].split("//")[-1]
            message = message.replace(token_part, "[REDACTED]")
            message = message.replace(clone_url, "[URL_REDACTED]")

        return message


__all__ = [
    "GitCloneRepositorySync",
]
