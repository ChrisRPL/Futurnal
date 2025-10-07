"""GraphQL-based repository synchronization implementation.

This module implements repository sync using GitHub's GraphQL API, optimized for
selective file access and minimal disk usage. Uses batched queries and respects
rate limits through GitHubAPIClientManager integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .api_client_manager import GitHubAPIClientManager
from .descriptor import GitHubRepositoryDescriptor
from .sync_models import (
    FileContent,
    FileEntry,
    SyncResult,
    SyncState,
    SyncStatus,
    SyncStrategy,
)
from .sync_utils import PatternMatcher

logger = logging.getLogger(__name__)


@dataclass
class GraphQLRepositorySync:
    """Sync repository using GitHub GraphQL API."""

    api_client_manager: GitHubAPIClientManager
    workspace_dir: Path

    def __init__(
        self,
        api_client_manager: GitHubAPIClientManager,
        workspace_dir: Optional[Path] = None,
    ):
        """Initialize GraphQL sync.

        Args:
            api_client_manager: GitHub API client manager
            workspace_dir: Workspace directory for cached files
        """
        self.api_client_manager = api_client_manager

        if workspace_dir is None:
            workspace_dir = (
                Path.home() / ".futurnal" / "workspace" / "github" / "graphql"
            )

        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    async def sync_repository(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
        state: Optional[SyncState] = None,
    ) -> SyncResult:
        """Sync repository using GraphQL API.

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
            sync_mode="graphql_api",
            status=SyncStatus.IN_PROGRESS,
            started_at=started_at,
        )

        try:
            # Determine which branch to sync (primary branch)
            primary_branch = strategy.branches[0] if strategy.branches else "main"

            logger.info(
                f"Starting GraphQL sync for {descriptor.full_name} on branch {primary_branch}"
            )

            # Fetch repository tree
            tree_entries = await self._fetch_repository_tree(
                descriptor=descriptor,
                branch=primary_branch,
                path="",  # Root path
            )

            logger.debug(
                f"Fetched {len(tree_entries)} entries from repository tree"
            )

            # Filter files based on patterns
            filtered_files = self._filter_files(tree_entries, strategy)

            logger.info(
                f"Filtered to {len(filtered_files)} files (from {len(tree_entries)} total)"
            )

            result.files_skipped = len(tree_entries) - len(filtered_files)

            # Fetch file contents if requested
            if strategy.fetch_file_content and filtered_files:
                contents = await self._fetch_file_contents_batch(
                    descriptor=descriptor,
                    branch=primary_branch,
                    files=filtered_files,
                    batch_size=strategy.batch_size,
                )

                # Count successes and failures
                result.files_synced = len([c for c in contents if c is not None])
                result.files_failed = len([c for c in contents if c is None])
                result.bytes_synced = sum(
                    c.size for c in contents if c is not None
                )

                logger.info(
                    f"Synced {result.files_synced} files "
                    f"({result.bytes_synced_mb:.2f} MB)"
                )
            else:
                # Metadata only sync
                result.files_synced = len(filtered_files)
                logger.info(f"Synced metadata for {result.files_synced} files")

            # Get latest commit SHA for this branch
            commit_sha = await self._fetch_latest_commit_sha(
                descriptor, primary_branch
            )

            result.commits_processed = 1
            result.branches_synced = [primary_branch]
            result.status = SyncStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc)

            if result.completed_at and result.started_at:
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()

            logger.info(
                f"GraphQL sync completed successfully in {result.duration_seconds:.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"GraphQL sync failed: {e}", exc_info=True)

            result.status = SyncStatus.FAILED
            result.error_message = str(e)
            result.error_details = {"exception_type": type(e).__name__}
            result.completed_at = datetime.now(timezone.utc)

            if result.completed_at and result.started_at:
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()

            return result

    async def _fetch_repository_tree(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch: str,
        path: str = "",
    ) -> List[FileEntry]:
        """Fetch repository file tree via GraphQL.

        Args:
            descriptor: Repository descriptor
            branch: Branch name
            path: Path within repository (empty = root)

        Returns:
            List of file entries
        """
        query = """
        query($owner: String!, $repo: String!, $expression: String!) {
          repository(owner: $owner, name: $repo) {
            object(expression: $expression) {
              ... on Tree {
                entries {
                  name
                  type
                  mode
                  path
                  object {
                    ... on Blob {
                      byteSize
                      isBinary
                      oid
                    }
                  }
                }
              }
            }
          }
        }
        """

        # Build expression: branch:path
        expression = f"{branch}:{path}" if path else f"{branch}:"

        variables = {
            "owner": descriptor.owner,
            "repo": descriptor.repo,
            "expression": expression,
        }

        response = await self.api_client_manager.graphql_request(
            credential_id=descriptor.credential_id,
            query=query,
            variables=variables,
            github_host=descriptor.github_host,
        )

        # Parse response
        entries: List[FileEntry] = []

        try:
            repo_data = response.get("data", {}).get("repository", {})
            obj = repo_data.get("object", {})
            graphql_entries = obj.get("entries", [])

            for entry in graphql_entries:
                # Handle both files and directories
                entry_type = entry.get("type", "blob").lower()

                # Get blob object data if available
                blob_obj = entry.get("object", {})
                size = blob_obj.get("byteSize")
                is_binary = blob_obj.get("isBinary", False)
                oid = blob_obj.get("oid")

                file_entry = FileEntry(
                    path=entry.get("path", ""),
                    name=entry.get("name", ""),
                    type=entry_type,
                    mode=str(entry.get("mode", "")),
                    size=size,
                    is_binary=is_binary,
                    sha=oid,
                )

                entries.append(file_entry)

                # Recursively fetch subdirectories if needed
                if entry_type == "tree":
                    subdir_path = entry.get("path", "")
                    if subdir_path:
                        try:
                            sub_entries = await self._fetch_repository_tree(
                                descriptor, branch, subdir_path
                            )
                            entries.extend(sub_entries)
                        except Exception as e:
                            logger.warning(
                                f"Failed to fetch subdirectory {subdir_path}: {e}"
                            )

        except Exception as e:
            logger.error(f"Failed to parse tree response: {e}")
            raise

        return entries

    async def _fetch_latest_commit_sha(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch: str,
    ) -> str:
        """Fetch latest commit SHA for branch.

        Args:
            descriptor: Repository descriptor
            branch: Branch name

        Returns:
            Commit SHA
        """
        query = """
        query($owner: String!, $repo: String!, $branch: String!) {
          repository(owner: $owner, name: $repo) {
            ref(qualifiedName: $branch) {
              target {
                ... on Commit {
                  oid
                }
              }
            }
          }
        }
        """

        variables = {
            "owner": descriptor.owner,
            "repo": descriptor.repo,
            "branch": f"refs/heads/{branch}",
        }

        response = await self.api_client_manager.graphql_request(
            credential_id=descriptor.credential_id,
            query=query,
            variables=variables,
            github_host=descriptor.github_host,
        )

        try:
            repo_data = response.get("data", {}).get("repository", {})
            ref_data = repo_data.get("ref", {})
            target = ref_data.get("target", {})
            return target.get("oid", "")
        except Exception as e:
            logger.error(f"Failed to fetch commit SHA: {e}")
            return ""

    async def _fetch_file_contents_batch(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch: str,
        files: List[FileEntry],
        batch_size: int = 10,
    ) -> List[Optional[FileContent]]:
        """Fetch multiple file contents in batches.

        Args:
            descriptor: Repository descriptor
            branch: Branch name
            files: Files to fetch
            batch_size: Number of files per batch

        Returns:
            List of file contents (None for failed fetches)
        """
        all_contents: List[Optional[FileContent]] = []

        # Process in batches
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]

            logger.debug(
                f"Fetching batch {i // batch_size + 1} "
                f"({len(batch)} files)"
            )

            try:
                batch_contents = await self._fetch_single_batch(
                    descriptor, branch, batch
                )
                all_contents.extend(batch_contents)
            except Exception as e:
                logger.error(f"Batch fetch failed: {e}")
                # Add None for each file in failed batch
                all_contents.extend([None] * len(batch))

        return all_contents

    async def _fetch_single_batch(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch: str,
        files: List[FileEntry],
    ) -> List[Optional[FileContent]]:
        """Fetch a single batch of file contents.

        Args:
            descriptor: Repository descriptor
            branch: Branch name
            files: Files to fetch in this batch

        Returns:
            List of file contents
        """
        # Build query with aliases for each file
        query_parts = ["query($owner: String!, $repo: String!) {"]
        query_parts.append("  repository(owner: $owner, name: $repo) {")

        for idx, file_entry in enumerate(files):
            alias = f"file{idx}"
            expression = f"{branch}:{file_entry.path}"
            query_parts.append(f'    {alias}: object(expression: "{expression}") {{')
            query_parts.append("      ... on Blob {")
            query_parts.append("        text")
            query_parts.append("        byteSize")
            query_parts.append("        isBinary")
            query_parts.append("        oid")
            query_parts.append("      }")
            query_parts.append("    }")

        query_parts.append("  }")
        query_parts.append("}")

        query = "\n".join(query_parts)

        variables = {
            "owner": descriptor.owner,
            "repo": descriptor.repo,
        }

        response = await self.api_client_manager.graphql_request(
            credential_id=descriptor.credential_id,
            query=query,
            variables=variables,
            github_host=descriptor.github_host,
        )

        # Parse response
        contents: List[Optional[FileContent]] = []

        try:
            repo_data = response.get("data", {}).get("repository", {})

            for idx, file_entry in enumerate(files):
                alias = f"file{idx}"
                file_data = repo_data.get(alias, {})

                if not file_data:
                    logger.warning(f"No data for file: {file_entry.path}")
                    contents.append(None)
                    continue

                text = file_data.get("text", "")
                size = file_data.get("byteSize", 0)
                is_binary = file_data.get("isBinary", False)
                oid = file_data.get("oid", "")

                if text or size == 0:  # Empty files are OK
                    file_content = FileContent(
                        path=file_entry.path,
                        content=text,
                        size=size,
                        sha=oid,
                        is_binary=is_binary,
                    )
                    contents.append(file_content)
                else:
                    logger.warning(
                        f"Empty content for non-empty file: {file_entry.path}"
                    )
                    contents.append(None)

        except Exception as e:
            logger.error(f"Failed to parse batch response: {e}")
            # Return None for all files in this batch
            return [None] * len(files)

        return contents

    def _filter_files(
        self,
        entries: List[FileEntry],
        strategy: SyncStrategy,
    ) -> List[FileEntry]:
        """Filter files based on patterns and size limits.

        Args:
            entries: File entries to filter
            strategy: Sync strategy with filter configuration

        Returns:
            Filtered list of file entries
        """
        matcher = PatternMatcher(
            include_patterns=strategy.file_patterns,
            exclude_patterns=strategy.exclude_patterns,
        )

        filtered: List[FileEntry] = []

        for entry in entries:
            # Skip directories
            if entry.type != "blob":
                continue

            # Check patterns
            if not matcher.should_include(entry.path):
                logger.debug(f"Skipping {entry.path} (pattern mismatch)")
                continue

            # Check size limit
            if entry.should_skip(strategy.max_file_size_mb):
                logger.debug(
                    f"Skipping {entry.path} (size: {entry.size_mb:.1f} MB)"
                )
                continue

            filtered.append(entry)

        return filtered


__all__ = [
    "GraphQLRepositorySync",
]
