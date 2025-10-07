"""Data models for GitHub repository synchronization.

This module defines the data models used for repository sync operations,
including sync strategies, state tracking, and result reporting. All models
are designed with privacy-first principles and timezone-aware datetime handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SyncStatus(str, Enum):
    """Status of a sync operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Sync Strategy
# ---------------------------------------------------------------------------


class SyncStrategy(BaseModel):
    """Configuration for repository sync strategy.

    This model extends the sync configuration in GitHubRepositoryDescriptor
    with additional runtime parameters for controlling sync behavior.
    """

    # Branch selection
    branches: List[str] = Field(
        default_factory=lambda: ["main"],
        description="Branches to sync (empty = use descriptor defaults)",
    )
    include_all_branches: bool = Field(
        default=False, description="Sync all branches regardless of whitelist"
    )
    include_tags: bool = Field(default=False, description="Include tags in sync")

    # File selection
    file_patterns: List[str] = Field(
        default_factory=list,
        description="Glob patterns for files to sync (empty = all)",
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [
            ".git/",
            "node_modules/",
            "__pycache__/",
            "*.pyc",
            ".env*",
            "secrets.*",
            "credentials.*",
        ],
        description="Patterns to exclude from sync",
    )

    # GraphQL API Mode settings
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Skip files larger than this (API mode)",
    )
    fetch_file_content: bool = Field(
        default=True, description="Fetch actual file content (vs metadata only)"
    )
    batch_size: int = Field(
        default=10, ge=1, le=50, description="Files per GraphQL batch request"
    )

    # Git Clone Mode settings
    clone_depth: Optional[int] = Field(
        default=None,
        ge=1,
        description="Shallow clone depth (None = full history)",
    )
    use_sparse_checkout: bool = Field(
        default=False, description="Use sparse checkout for large repos"
    )
    clone_submodules: bool = Field(
        default=False, description="Clone submodules recursively"
    )
    single_branch: bool = Field(
        default=True, description="Clone only specified branch (vs all)"
    )

    @field_validator("branches")
    @classmethod
    def _validate_branches(cls, value: List[str]) -> List[str]:
        """Validate branch names."""
        if not value:
            return ["main"]  # Default fallback
        return [b.strip() for b in value if b.strip()]

    @field_validator("file_patterns", "exclude_patterns")
    @classmethod
    def _validate_patterns(cls, value: List[str]) -> List[str]:
        """Validate glob patterns."""
        return [p.strip() for p in value if p.strip()]


# ---------------------------------------------------------------------------
# File Entry Models
# ---------------------------------------------------------------------------


class FileEntry(BaseModel):
    """File metadata from repository tree traversal."""

    path: str = Field(..., description="File path relative to repository root")
    name: str = Field(..., description="File name")
    type: str = Field(..., description="Entry type (blob, tree)")
    mode: Optional[str] = Field(
        default=None, description="Git file mode (e.g., 100644)"
    )
    size: Optional[int] = Field(default=None, ge=0, description="File size in bytes")
    is_binary: bool = Field(default=False, description="Whether file is binary")
    sha: Optional[str] = Field(default=None, description="Git object SHA")

    @property
    def size_mb(self) -> float:
        """File size in megabytes."""
        return (self.size or 0) / (1024 * 1024)

    def should_skip(self, max_size_mb: int) -> bool:
        """Check if file should be skipped based on size."""
        if self.is_binary:
            return True
        if self.size and self.size_mb > max_size_mb:
            return True
        return False


class FileContent(BaseModel):
    """File content with metadata."""

    path: str = Field(..., description="File path")
    content: str = Field(..., description="File content (base64 for binary)")
    size: int = Field(..., ge=0, description="Content size in bytes")
    sha: Optional[str] = Field(default=None, description="Content SHA")
    encoding: str = Field(default="utf-8", description="Content encoding")
    is_binary: bool = Field(default=False, description="Whether content is binary")


# ---------------------------------------------------------------------------
# Branch Sync State
# ---------------------------------------------------------------------------


class BranchSyncState(BaseModel):
    """Sync state for a single branch."""

    branch_name: str = Field(..., description="Branch name")
    last_commit_sha: str = Field(..., description="Latest synced commit SHA")
    last_sync_time: datetime = Field(..., description="Last successful sync time")
    file_count: int = Field(default=0, ge=0, description="Number of files synced")
    total_bytes: int = Field(default=0, ge=0, description="Total bytes synced")
    sync_errors: int = Field(default=0, ge=0, description="Number of sync errors")

    # Incremental sync fields
    parent_commit_sha: Optional[str] = Field(
        default=None,
        description="Previous HEAD before update (for force-push detection)",
    )
    force_push_detected: bool = Field(
        default=False, description="Whether force push was detected"
    )
    commits_processed: int = Field(
        default=0, ge=0, description="Total commits processed for this branch"
    )

    @field_validator("last_sync_time")
    @classmethod
    def _ensure_timezone(cls, value: datetime) -> datetime:
        """Ensure datetime has timezone info."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


# ---------------------------------------------------------------------------
# Sync State
# ---------------------------------------------------------------------------


class SyncState(BaseModel):
    """Persistent state for repository sync operations."""

    repo_id: str = Field(..., description="Repository identifier")
    sync_mode: str = Field(..., description="Sync mode (graphql_api or git_clone)")
    status: SyncStatus = Field(
        default=SyncStatus.PENDING, description="Current sync status"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="State creation time",
    )
    last_sync_time: Optional[datetime] = Field(
        default=None, description="Last successful sync time"
    )
    last_attempt_time: Optional[datetime] = Field(
        default=None, description="Last sync attempt time"
    )

    # Commit tracking
    last_commit_sha: Optional[str] = Field(
        default=None, description="Latest commit SHA synced"
    )
    branch_states: Dict[str, BranchSyncState] = Field(
        default_factory=dict, description="Per-branch sync state"
    )

    # Statistics
    total_files_synced: int = Field(
        default=0, ge=0, description="Total files successfully synced"
    )
    total_bytes_synced: int = Field(
        default=0, ge=0, description="Total bytes synced"
    )
    total_commits_processed: int = Field(
        default=0, ge=0, description="Total commits processed"
    )
    sync_errors: int = Field(default=0, ge=0, description="Total sync errors")
    consecutive_failures: int = Field(
        default=0, ge=0, description="Consecutive failed sync attempts"
    )

    # Paths (for Git Clone mode)
    local_clone_path: Optional[Path] = Field(
        default=None, description="Local clone directory path"
    )

    @field_validator("created_at", "last_sync_time", "last_attempt_time")
    @classmethod
    def _ensure_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime has timezone info."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def mark_sync_started(self) -> None:
        """Mark sync as started."""
        self.status = SyncStatus.IN_PROGRESS
        self.last_attempt_time = datetime.now(timezone.utc)

    def mark_sync_completed(
        self, *, files_synced: int = 0, bytes_synced: int = 0, commits: int = 0
    ) -> None:
        """Mark sync as completed successfully."""
        self.status = SyncStatus.COMPLETED
        self.last_sync_time = datetime.now(timezone.utc)
        self.total_files_synced += files_synced
        self.total_bytes_synced += bytes_synced
        self.total_commits_processed += commits
        self.consecutive_failures = 0

    def mark_sync_failed(self, error_count: int = 1) -> None:
        """Mark sync as failed."""
        self.status = SyncStatus.FAILED
        self.sync_errors += error_count
        self.consecutive_failures += 1

    def update_branch_state(
        self,
        branch_name: str,
        commit_sha: str,
        file_count: int = 0,
        bytes_synced: int = 0,
        errors: int = 0,
    ) -> None:
        """Update state for a specific branch."""
        self.branch_states[branch_name] = BranchSyncState(
            branch_name=branch_name,
            last_commit_sha=commit_sha,
            last_sync_time=datetime.now(timezone.utc),
            file_count=file_count,
            total_bytes=bytes_synced,
            sync_errors=errors,
        )
        # Update global commit SHA if this is the primary branch
        if branch_name in ["main", "master"] or not self.last_commit_sha:
            self.last_commit_sha = commit_sha

    def get_branch_state(self, branch_name: str) -> Optional[BranchSyncState]:
        """Get state for a specific branch."""
        return self.branch_states.get(branch_name)

    def is_healthy(self) -> bool:
        """Check if sync is healthy (not too many consecutive failures)."""
        return self.consecutive_failures < 5

    @property
    def total_bytes_mb(self) -> float:
        """Total bytes synced in megabytes."""
        return self.total_bytes_synced / (1024 * 1024)


# ---------------------------------------------------------------------------
# Sync Result
# ---------------------------------------------------------------------------


class SyncResult(BaseModel):
    """Result of a sync operation."""

    repo_id: str = Field(..., description="Repository identifier")
    sync_mode: str = Field(..., description="Sync mode used")
    status: SyncStatus = Field(..., description="Sync status")

    # Statistics
    files_synced: int = Field(default=0, ge=0, description="Files successfully synced")
    files_skipped: int = Field(default=0, ge=0, description="Files skipped")
    files_failed: int = Field(default=0, ge=0, description="Files that failed")
    bytes_synced: int = Field(default=0, ge=0, description="Total bytes synced")
    commits_processed: int = Field(
        default=0, ge=0, description="Commits processed (if applicable)"
    )

    # Timing
    started_at: datetime = Field(..., description="Sync start time")
    completed_at: Optional[datetime] = Field(
        default=None, description="Sync completion time"
    )
    duration_seconds: Optional[float] = Field(
        default=None, ge=0, description="Sync duration in seconds"
    )

    # Branches
    branches_synced: List[str] = Field(
        default_factory=list, description="Branches that were synced"
    )

    # Error details
    error_message: Optional[str] = Field(
        default=None, description="Error message if sync failed"
    )
    error_details: Optional[Dict[str, str]] = Field(
        default=None, description="Detailed error information"
    )

    # Incremental sync fields
    new_commits: List[str] = Field(
        default_factory=list, description="Commit SHAs processed in this sync"
    )
    modified_files: List[str] = Field(
        default_factory=list, description="Files modified in this sync"
    )
    deleted_files: List[str] = Field(
        default_factory=list, description="Files deleted in this sync"
    )
    added_files: List[str] = Field(
        default_factory=list, description="Files added in this sync"
    )
    force_push_handled: bool = Field(
        default=False, description="Whether force push was handled"
    )

    @field_validator("started_at", "completed_at")
    @classmethod
    def _ensure_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime has timezone info."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @property
    def bytes_synced_mb(self) -> float:
        """Bytes synced in megabytes."""
        return self.bytes_synced / (1024 * 1024)

    @property
    def success_rate(self) -> float:
        """Success rate percentage."""
        total = self.files_synced + self.files_failed
        if total == 0:
            return 0.0
        return (self.files_synced / total) * 100

    def is_success(self) -> bool:
        """Check if sync was successful."""
        return self.status == SyncStatus.COMPLETED and self.files_failed == 0

    def has_partial_success(self) -> bool:
        """Check if sync had partial success (some files synced, some failed)."""
        return self.files_synced > 0 and self.files_failed > 0


# ---------------------------------------------------------------------------
# Disk Space Estimate
# ---------------------------------------------------------------------------


class DiskSpaceEstimate(BaseModel):
    """Estimate of disk space required for repository sync."""

    repo_id: str = Field(..., description="Repository identifier")
    sync_mode: str = Field(..., description="Sync mode")
    estimated_size_bytes: int = Field(
        ..., ge=0, description="Estimated size in bytes"
    )
    available_space_bytes: int = Field(
        ..., ge=0, description="Available disk space"
    )
    is_sufficient: bool = Field(
        ..., description="Whether available space is sufficient"
    )
    buffer_percentage: float = Field(
        default=0.2, ge=0, le=1, description="Safety buffer percentage"
    )

    @property
    def estimated_size_mb(self) -> float:
        """Estimated size in megabytes."""
        return self.estimated_size_bytes / (1024 * 1024)

    @property
    def available_space_mb(self) -> float:
        """Available space in megabytes."""
        return self.available_space_bytes / (1024 * 1024)

    @property
    def estimated_size_gb(self) -> float:
        """Estimated size in gigabytes."""
        return self.estimated_size_bytes / (1024 * 1024 * 1024)

    @property
    def available_space_gb(self) -> float:
        """Available space in gigabytes."""
        return self.available_space_bytes / (1024 * 1024 * 1024)

    def space_after_sync_bytes(self) -> int:
        """Estimated available space after sync."""
        return max(0, self.available_space_bytes - self.estimated_size_bytes)

    def space_after_sync_gb(self) -> float:
        """Estimated available space after sync in GB."""
        return self.space_after_sync_bytes() / (1024 * 1024 * 1024)


__all__ = [
    "BranchSyncState",
    "DiskSpaceEstimate",
    "FileContent",
    "FileEntry",
    "SyncResult",
    "SyncState",
    "SyncStatus",
    "SyncStrategy",
]
