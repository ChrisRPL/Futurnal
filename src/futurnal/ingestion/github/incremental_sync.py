"""Incremental synchronization engine for GitHub repositories.

This module implements commit-aware delta synchronization that tracks repository
changes at the commit level, detects force pushes and rebases, and processes only
changed files since the last sync. Designed for efficient incremental learning while
maintaining full-fidelity change detection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .api_client_manager import GitHubAPIClientManager
from .descriptor import GitHubRepositoryDescriptor, SyncMode
from .file_classifier import FileClassifier
from .graphql_sync import GraphQLRepositorySync
from .git_clone_sync import GitCloneRepositorySync
from .sync_models import (
    BranchSyncState,
    FileContent,
    FileEntry,
    SyncResult,
    SyncState,
    SyncStatus,
    SyncStrategy,
)
from .sync_state_manager import SyncStateManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Element Sink Protocol
# ---------------------------------------------------------------------------


class ElementSink:
    """Protocol for processing synced files."""

    def handle(self, element: dict) -> None:
        """Handle a processed element."""
        raise NotImplementedError

    def handle_deletion(self, element: dict) -> None:
        """Handle element deletion."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Incremental Sync Engine
# ---------------------------------------------------------------------------


@dataclass
class IncrementalSyncEngine:
    """Manages incremental repository synchronization with commit tracking.

    This engine implements efficient delta synchronization by:
    - Tracking commit SHAs per branch
    - Detecting force pushes and rebases via commit ancestry
    - Processing only changed files since last sync
    - Maintaining resilient state across sync operations

    Attributes:
        api_client_manager: GitHub API client for GraphQL and REST requests
        state_manager: Persistent state storage manager
        file_classifier: Optional file classifier for content categorization
        element_sink: Optional sink for processed file elements
    """

    api_client_manager: GitHubAPIClientManager
    state_manager: SyncStateManager
    file_classifier: Optional[FileClassifier] = None
    element_sink: Optional[ElementSink] = None

    def __init__(
        self,
        api_client_manager: GitHubAPIClientManager,
        state_manager: SyncStateManager,
        file_classifier: Optional[FileClassifier] = None,
        element_sink: Optional[ElementSink] = None,
    ):
        """Initialize incremental sync engine.

        Args:
            api_client_manager: GitHub API client manager
            state_manager: Sync state manager
            file_classifier: Optional file classifier
            element_sink: Optional element sink for processed files
        """
        self.api_client_manager = api_client_manager
        self.state_manager = state_manager
        self.file_classifier = file_classifier
        self.element_sink = element_sink

    async def sync_repository(
        self,
        descriptor: GitHubRepositoryDescriptor,
    ) -> SyncResult:
        """Perform incremental sync of repository.

        Args:
            descriptor: Repository descriptor with sync configuration

        Returns:
            SyncResult with statistics and delta information
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc)

        logger.info(
            f"Starting incremental sync for {descriptor.full_name} "
            f"({len(descriptor.branches)} branches)"
        )

        # Load or create sync state
        state = self.state_manager.get_or_create(
            repo_id=descriptor.id,
            sync_mode=descriptor.sync_mode.value,
            local_clone_path=descriptor.local_clone_path,
        )

        state.mark_sync_started()
        self.state_manager.save(state)

        # Aggregate result
        aggregate_result = SyncResult(
            repo_id=descriptor.id,
            sync_mode=descriptor.sync_mode.value,
            status=SyncStatus.IN_PROGRESS,
            started_at=started_at,
        )

        try:
            # Sync each configured branch
            branch_results: List[SyncResult] = []
            for branch in descriptor.branches:
                logger.info(f"Syncing branch: {branch}")
                result = await self._sync_branch(descriptor, branch, state)
                branch_results.append(result)

            # Detect deleted branches
            deleted_branches = self._detect_deleted_branches(descriptor, state)
            if deleted_branches:
                logger.info(f"Cleaning up deleted branches: {deleted_branches}")
                for branch in deleted_branches:
                    state.branch_states.pop(branch, None)

            # Merge results
            aggregate_result = self._merge_results(branch_results, started_at)

            # Update global state
            state.mark_sync_completed(
                files_synced=aggregate_result.files_synced,
                bytes_synced=aggregate_result.bytes_synced,
                commits=aggregate_result.commits_processed,
            )

            # Save final state
            self.state_manager.save(state)

            aggregate_result.duration_seconds = time.time() - start_time
            aggregate_result.status = SyncStatus.COMPLETED
            aggregate_result.completed_at = datetime.now(timezone.utc)

            logger.info(
                f"Incremental sync completed: {aggregate_result.commits_processed} commits, "
                f"{aggregate_result.files_synced} files, "
                f"{aggregate_result.bytes_synced / (1024*1024):.2f} MB in "
                f"{aggregate_result.duration_seconds:.1f}s"
            )

            return aggregate_result

        except Exception as e:
            logger.error(f"Incremental sync failed: {e}", exc_info=True)

            state.mark_sync_failed(error_count=1)
            self.state_manager.save(state)

            aggregate_result.status = SyncStatus.FAILED
            aggregate_result.error_message = str(e)
            aggregate_result.error_details = {"exception_type": type(e).__name__}
            aggregate_result.completed_at = datetime.now(timezone.utc)
            aggregate_result.duration_seconds = time.time() - start_time

            return aggregate_result

    async def _sync_branch(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch_name: str,
        state: SyncState,
    ) -> SyncResult:
        """Sync a single branch incrementally.

        Args:
            descriptor: Repository descriptor
            branch_name: Branch name to sync
            state: Current repository sync state

        Returns:
            SyncResult for this branch
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc)

        # Get current branch HEAD commit SHA
        try:
            current_head = await self._get_branch_head(
                owner=descriptor.owner,
                repo=descriptor.repo,
                branch=branch_name,
                credential_id=descriptor.credential_id,
                github_host=descriptor.github_host,
            )
        except Exception as e:
            logger.error(f"Failed to get HEAD for branch {branch_name}: {e}")
            return SyncResult(
                repo_id=descriptor.id,
                sync_mode=descriptor.sync_mode.value,
                status=SyncStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                error_message=f"Failed to get branch HEAD: {e}",
            )

        if not current_head:
            logger.warning(f"Branch {branch_name} not found or empty")
            return SyncResult(
                repo_id=descriptor.id,
                sync_mode=descriptor.sync_mode.value,
                status=SyncStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                branches_synced=[branch_name],
            )

        # Check if branch exists in state
        branch_state = state.branch_states.get(branch_name)

        if not branch_state:
            # First sync of this branch
            logger.info(f"Performing initial sync of branch {branch_name}")
            result = await self._initial_branch_sync(
                descriptor, branch_name, current_head
            )
        elif branch_state.last_commit_sha == current_head:
            # No changes since last sync
            logger.info(f"No changes detected for branch {branch_name}")
            result = SyncResult(
                repo_id=descriptor.id,
                sync_mode=descriptor.sync_mode.value,
                status=SyncStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                branches_synced=[branch_name],
            )
        else:
            # Incremental sync - process delta
            logger.info(
                f"Performing incremental sync of branch {branch_name} "
                f"({branch_state.last_commit_sha[:8]}...{current_head[:8]})"
            )
            result = await self._incremental_branch_sync(
                descriptor, branch_name, branch_state, current_head
            )

        result.duration_seconds = time.time() - start_time

        # Update branch state
        previous_commits = branch_state.commits_processed if branch_state else 0
        new_branch_state = BranchSyncState(
            branch_name=branch_name,
            last_commit_sha=current_head,
            last_sync_time=datetime.now(timezone.utc),
            file_count=result.files_synced,
            total_bytes=result.bytes_synced,
            sync_errors=result.files_failed,
            parent_commit_sha=branch_state.last_commit_sha if branch_state else None,
            force_push_detected=result.force_push_handled,
            commits_processed=previous_commits + result.commits_processed,
        )

        state.branch_states[branch_name] = new_branch_state

        return result

    async def _get_branch_head(
        self,
        owner: str,
        repo: str,
        branch: str,
        credential_id: str,
        github_host: str = "github.com",
    ) -> str:
        """Get current HEAD commit SHA for branch via GraphQL.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            credential_id: Credential identifier
            github_host: GitHub host (default: github.com)

        Returns:
            Commit SHA or empty string if branch doesn't exist
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
            "owner": owner,
            "repo": repo,
            "branch": f"refs/heads/{branch}",
        }

        try:
            response = await self.api_client_manager.graphql_request(
                credential_id=credential_id,
                query=query,
                variables=variables,
                github_host=github_host,
            )

            repo_data = response.get("data", {}).get("repository", {})
            ref_data = repo_data.get("ref")

            if not ref_data:
                return ""

            target = ref_data.get("target", {})
            return target.get("oid", "")

        except Exception as e:
            logger.error(f"Failed to fetch branch HEAD via GraphQL: {e}")
            raise

    async def _incremental_branch_sync(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch_name: str,
        branch_state: BranchSyncState,
        current_head: str,
    ) -> SyncResult:
        """Sync only new commits since last sync.

        Args:
            descriptor: Repository descriptor
            branch_name: Branch name
            branch_state: Current branch state
            current_head: Current HEAD commit SHA

        Returns:
            SyncResult with delta changes
        """
        started_at = datetime.now(timezone.utc)

        # Get commit history from last_commit_sha to current_head
        try:
            comparison = await self._get_commit_range(
                owner=descriptor.owner,
                repo=descriptor.repo,
                from_sha=branch_state.last_commit_sha,
                to_sha=current_head,
                credential_id=descriptor.credential_id,
                github_host=descriptor.github_host,
            )
        except Exception as e:
            logger.error(f"Failed to get commit range: {e}")
            return SyncResult(
                repo_id=descriptor.id,
                sync_mode=descriptor.sync_mode.value,
                status=SyncStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                error_message=f"Failed to get commit range: {e}",
            )

        commits = comparison.get("commits", [])
        status = comparison.get("status", "")

        # Check for force push
        if self._is_force_push(commits, branch_state.last_commit_sha, status):
            logger.warning(
                f"Force push detected on {branch_name}, performing full resync"
            )
            result = await self._handle_force_push(descriptor, branch_name, current_head)
            result.force_push_handled = True
            return result

        if not commits:
            # No new commits
            return SyncResult(
                repo_id=descriptor.id,
                sync_mode=descriptor.sync_mode.value,
                status=SyncStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                branches_synced=[branch_name],
            )

        logger.info(f"Processing {len(commits)} new commits")

        # Process each new commit to collect file changes
        modified_files: Set[str] = set()
        deleted_files: Set[str] = set()
        added_files: Set[str] = set()
        commit_shas: List[str] = []

        for commit in commits:
            commit_sha = commit.get("sha") or commit.get("oid", "")
            if not commit_sha:
                continue

            commit_shas.append(commit_sha)

            try:
                changes = await self._get_commit_changes(
                    owner=descriptor.owner,
                    repo=descriptor.repo,
                    commit_sha=commit_sha,
                    credential_id=descriptor.credential_id,
                    github_host=descriptor.github_host,
                )

                modified_files.update(changes.get("modified", []))
                deleted_files.update(changes.get("deleted", []))
                added_files.update(changes.get("added", []))

            except Exception as e:
                logger.warning(f"Failed to get changes for commit {commit_sha[:8]}: {e}")

        # Sync changed files (modified + added)
        files_to_sync = list(modified_files | added_files)
        files_synced = 0
        bytes_synced = 0

        if files_to_sync:
            logger.info(f"Syncing {len(files_to_sync)} changed files")
            files_synced, bytes_synced = await self._sync_files(
                descriptor=descriptor,
                file_paths=files_to_sync,
                branch=branch_name,
            )

        # Handle deletions
        if deleted_files:
            logger.info(f"Handling {len(deleted_files)} deleted files")
            await self._handle_deleted_files(
                descriptor=descriptor,
                file_paths=list(deleted_files),
            )

        return SyncResult(
            repo_id=descriptor.id,
            sync_mode=descriptor.sync_mode.value,
            status=SyncStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            branches_synced=[branch_name],
            new_commits=commit_shas,
            modified_files=list(modified_files),
            deleted_files=list(deleted_files),
            added_files=list(added_files),
            commits_processed=len(commits),
            files_synced=files_synced,
            bytes_synced=bytes_synced,
        )

    async def _get_commit_range(
        self,
        owner: str,
        repo: str,
        from_sha: str,
        to_sha: str,
        credential_id: str,
        github_host: str = "github.com",
    ) -> Dict[str, Any]:
        """Get commits between two SHAs using REST API.

        Args:
            owner: Repository owner
            repo: Repository name
            from_sha: Starting commit SHA
            to_sha: Ending commit SHA
            credential_id: Credential identifier
            github_host: GitHub host

        Returns:
            Dictionary with commits array and status
        """
        endpoint = f"/repos/{owner}/{repo}/compare/{from_sha}...{to_sha}"

        try:
            response = await self.api_client_manager.rest_request(
                credential_id=credential_id,
                method="GET",
                endpoint=endpoint,
                github_host=github_host,
            )

            return response

        except Exception as e:
            logger.error(f"Failed to compare commits: {e}")
            raise

    async def _get_commit_changes(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        credential_id: str,
        github_host: str = "github.com",
    ) -> Dict[str, List[str]]:
        """Get file changes in a commit using REST API.

        Args:
            owner: Repository owner
            repo: Repository name
            commit_sha: Commit SHA
            credential_id: Credential identifier
            github_host: GitHub host

        Returns:
            Dictionary with added, modified, and deleted file lists
        """
        endpoint = f"/repos/{owner}/{repo}/commits/{commit_sha}"

        try:
            response = await self.api_client_manager.rest_request(
                credential_id=credential_id,
                method="GET",
                endpoint=endpoint,
                github_host=github_host,
            )

            changes: Dict[str, List[str]] = {
                "added": [],
                "modified": [],
                "deleted": [],
            }

            files = response.get("files", [])
            for file_data in files:
                filename = file_data.get("filename", "")
                status = file_data.get("status", "")

                if status == "added":
                    changes["added"].append(filename)
                elif status == "modified":
                    changes["modified"].append(filename)
                elif status == "removed":
                    changes["deleted"].append(filename)
                elif status == "renamed":
                    # Treat renames as delete + add
                    changes["deleted"].append(file_data.get("previous_filename", ""))
                    changes["added"].append(filename)

            return changes

        except Exception as e:
            logger.error(f"Failed to get commit changes: {e}")
            raise

    def _is_force_push(
        self,
        commits: List[Dict[str, Any]],
        expected_parent_sha: str,
        compare_status: str,
    ) -> bool:
        """Detect if force push occurred.

        Args:
            commits: List of commit objects
            expected_parent_sha: Expected parent commit SHA
            compare_status: Status from compare API (ahead/behind/identical/diverged)

        Returns:
            True if force push detected
        """
        # Primary detection: check compare API status
        # The compare API returns "diverged" when branches have diverged (force push indicator)
        if compare_status == "diverged":
            logger.debug("Force push detected via compare API status='diverged'")
            return True

        # Note: We do NOT check if parent SHA is in commits list, because the
        # compare API only returns NEW commits between base and head, not including
        # the base commit itself. For a normal incremental update from A to C
        # (A -> B -> C), compare A...C returns [B, C], not [A, B, C].

        # If compare status is not "diverged" and we have commits, it's a normal update
        return False

    async def _handle_force_push(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch_name: str,
        current_head: str,
    ) -> SyncResult:
        """Handle force push by performing full resync.

        Args:
            descriptor: Repository descriptor
            branch_name: Branch name
            current_head: Current HEAD commit SHA

        Returns:
            SyncResult from full resync
        """
        logger.warning(
            f"Force push detected on {branch_name}, "
            f"triggering full resync to {current_head[:8]}"
        )

        result = await self._initial_branch_sync(descriptor, branch_name, current_head)
        result.force_push_handled = True

        return result

    async def _initial_branch_sync(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch_name: str,
        head_sha: str,
    ) -> SyncResult:
        """Perform initial full sync of branch.

        Delegates to GraphQL or Git Clone sync based on descriptor configuration.

        Args:
            descriptor: Repository descriptor
            branch_name: Branch name
            head_sha: HEAD commit SHA

        Returns:
            SyncResult from full sync
        """
        # Create strategy for single branch
        strategy = SyncStrategy(
            branches=[branch_name],
            include_all_branches=False,
            include_tags=False,
            file_patterns=descriptor.include_paths,
            exclude_patterns=descriptor.exclude_paths,
            max_file_size_mb=descriptor.privacy_settings.max_file_size_mb,
            fetch_file_content=True,
            batch_size=10,
        )

        # Get or create state for delegation
        state = self.state_manager.get_or_create(
            repo_id=descriptor.id,
            sync_mode=descriptor.sync_mode.value,
            local_clone_path=descriptor.local_clone_path,
        )

        # Delegate based on sync mode
        if descriptor.sync_mode == SyncMode.GRAPHQL_API:
            sync = GraphQLRepositorySync(
                api_client_manager=self.api_client_manager,
            )
            return await sync.sync_repository(descriptor, strategy, state)
        else:
            # Git Clone mode - requires credential manager
            # For now, return error as we need credential manager instance
            logger.error("Git Clone mode not yet supported in incremental sync")
            return SyncResult(
                repo_id=descriptor.id,
                sync_mode=descriptor.sync_mode.value,
                status=SyncStatus.FAILED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                error_message="Git Clone mode not yet supported in incremental sync",
            )

    async def _sync_files(
        self,
        descriptor: GitHubRepositoryDescriptor,
        file_paths: List[str],
        branch: str,
    ) -> tuple[int, int]:
        """Sync changed files by fetching content and processing.

        Args:
            descriptor: Repository descriptor
            file_paths: List of file paths to sync
            branch: Branch name

        Returns:
            Tuple of (files_synced, bytes_synced)
        """
        if not file_paths:
            return 0, 0

        files_synced = 0
        bytes_synced = 0

        # Fetch file contents via GraphQL (batch)
        try:
            file_entries = [
                FileEntry(
                    path=path,
                    name=path.split("/")[-1],
                    type="blob",
                )
                for path in file_paths
            ]

            # Use GraphQL sync's batch fetch logic
            sync = GraphQLRepositorySync(
                api_client_manager=self.api_client_manager,
            )

            contents = await sync._fetch_file_contents_batch(
                descriptor=descriptor,
                branch=branch,
                files=file_entries,
                batch_size=10,
            )

            # Process each file through classifier and sink
            for content in contents:
                if content is None:
                    continue

                files_synced += 1
                bytes_synced += content.size

                # Process through element sink if available
                if self.element_sink:
                    element = {
                        "sha256": content.sha or "",
                        "path": content.path,
                        "source": f"{descriptor.full_name}:{branch}",
                        "content": content.content,
                        "size": content.size,
                    }
                    try:
                        self.element_sink.handle(element)
                    except Exception as e:
                        logger.warning(f"Failed to process {content.path}: {e}")

        except Exception as e:
            logger.error(f"Failed to sync files: {e}")

        return files_synced, bytes_synced

    async def _handle_deleted_files(
        self,
        descriptor: GitHubRepositoryDescriptor,
        file_paths: List[str],
    ) -> None:
        """Handle deleted files by notifying element sink.

        Args:
            descriptor: Repository descriptor
            file_paths: List of deleted file paths
        """
        if not self.element_sink or not file_paths:
            return

        for path in file_paths:
            element = {
                "sha256": "",  # Unknown for deletions
                "path": path,
                "source": descriptor.full_name,
            }

            try:
                if hasattr(self.element_sink, "handle_deletion"):
                    self.element_sink.handle_deletion(element)
            except Exception as e:
                logger.warning(f"Failed to handle deletion of {path}: {e}")

    def _detect_deleted_branches(
        self,
        descriptor: GitHubRepositoryDescriptor,
        state: SyncState,
    ) -> List[str]:
        """Detect branches that were deleted from descriptor.

        Args:
            descriptor: Repository descriptor
            state: Current sync state

        Returns:
            List of deleted branch names
        """
        configured_branches = set(descriptor.branches)
        state_branches = set(state.branch_states.keys())

        deleted_branches = state_branches - configured_branches

        return list(deleted_branches)

    def _merge_results(
        self,
        branch_results: List[SyncResult],
        started_at: datetime,
    ) -> SyncResult:
        """Merge multiple branch results into aggregate result.

        Args:
            branch_results: List of per-branch results
            started_at: Sync start time

        Returns:
            Aggregated SyncResult
        """
        if not branch_results:
            return SyncResult(
                repo_id="",
                sync_mode="incremental",
                status=SyncStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        # Use first result as template
        merged = SyncResult(
            repo_id=branch_results[0].repo_id,
            sync_mode=branch_results[0].sync_mode,
            status=SyncStatus.COMPLETED,
            started_at=started_at,
        )

        # Aggregate statistics
        all_new_commits: List[str] = []
        all_modified_files: List[str] = []
        all_deleted_files: List[str] = []
        all_added_files: List[str] = []
        all_branches: List[str] = []

        for result in branch_results:
            merged.files_synced += result.files_synced
            merged.files_skipped += result.files_skipped
            merged.files_failed += result.files_failed
            merged.bytes_synced += result.bytes_synced
            merged.commits_processed += result.commits_processed

            all_new_commits.extend(result.new_commits)
            all_modified_files.extend(result.modified_files)
            all_deleted_files.extend(result.deleted_files)
            all_added_files.extend(result.added_files)
            all_branches.extend(result.branches_synced)

            if result.force_push_handled:
                merged.force_push_handled = True

            if result.status == SyncStatus.FAILED:
                merged.status = SyncStatus.FAILED
                if result.error_message:
                    if not merged.error_message:
                        merged.error_message = result.error_message
                    else:
                        merged.error_message += f"; {result.error_message}"

        # Deduplicate lists
        merged.new_commits = list(set(all_new_commits))
        merged.modified_files = list(set(all_modified_files))
        merged.deleted_files = list(set(all_deleted_files))
        merged.added_files = list(set(all_added_files))
        merged.branches_synced = list(set(all_branches))

        return merged


__all__ = [
    "IncrementalSyncEngine",
]
