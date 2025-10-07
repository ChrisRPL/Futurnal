"""Utility functions for GitHub repository synchronization.

This module provides helper functions for pattern matching, disk space checks,
git validation, and formatting utilities used across sync implementations.
"""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .sync_models import DiskSpaceEstimate, SyncResult


# ---------------------------------------------------------------------------
# Pattern Matching
# ---------------------------------------------------------------------------


class PatternMatcher:
    """Glob pattern matcher for file filtering."""

    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """Initialize pattern matcher.

        Args:
            include_patterns: Patterns to include (empty = include all)
            exclude_patterns: Patterns to exclude
        """
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []

    def should_include(self, path: str) -> bool:
        """Check if path should be included based on patterns.

        Args:
            path: File path to check

        Returns:
            True if file should be included, False otherwise
        """
        # Normalize path separators
        normalized_path = path.replace("\\", "/")

        # Check exclude patterns first (higher priority)
        if self._matches_any(normalized_path, self.exclude_patterns):
            return False

        # If no include patterns specified, include everything not excluded
        if not self.include_patterns:
            return True

        # Check include patterns
        return self._matches_any(normalized_path, self.include_patterns)

    def _matches_any(self, path: str, patterns: List[str]) -> bool:
        """Check if path matches any of the patterns.

        Args:
            path: File path to check
            patterns: List of glob patterns

        Returns:
            True if path matches any pattern, False otherwise
        """
        for pattern in patterns:
            # Handle directory patterns (ending with /)
            if pattern.endswith("/"):
                if path.startswith(pattern) or ("/" + pattern) in path:
                    return True
            # Handle wildcard patterns
            elif fnmatch.fnmatch(path, pattern):
                return True
            # Handle exact match
            elif path == pattern:
                return True
            # Handle path components
            elif "/" + pattern + "/" in "/" + path + "/":
                return True

        return False

    def filter_paths(self, paths: List[str]) -> List[str]:
        """Filter list of paths based on patterns.

        Args:
            paths: List of file paths

        Returns:
            Filtered list of paths
        """
        return [path for path in paths if self.should_include(path)]


# ---------------------------------------------------------------------------
# Disk Space Utilities
# ---------------------------------------------------------------------------


def get_available_disk_space(path: Path) -> int:
    """Get available disk space in bytes.

    Args:
        path: Path to check (will check the filesystem containing this path)

    Returns:
        Available space in bytes
    """
    try:
        stat = shutil.disk_usage(path)
        return stat.free
    except Exception:
        # If we can't determine disk space, return 0 to be safe
        return 0


def get_available_disk_space_gb(path: Path) -> float:
    """Get available disk space in gigabytes.

    Args:
        path: Path to check

    Returns:
        Available space in GB
    """
    bytes_available = get_available_disk_space(path)
    return bytes_available / (1024 * 1024 * 1024)


def check_disk_space_sufficient(
    required_bytes: int,
    path: Path,
    buffer_percentage: float = 0.2,
) -> DiskSpaceEstimate:
    """Check if sufficient disk space is available.

    Args:
        required_bytes: Required space in bytes
        path: Path where space is needed
        buffer_percentage: Safety buffer (0.2 = 20% buffer)

    Returns:
        DiskSpaceEstimate with availability information
    """
    available_bytes = get_available_disk_space(path)
    required_with_buffer = int(required_bytes * (1 + buffer_percentage))
    is_sufficient = available_bytes >= required_with_buffer

    return DiskSpaceEstimate(
        repo_id="",  # Will be filled by caller
        sync_mode="",  # Will be filled by caller
        estimated_size_bytes=required_bytes,
        available_space_bytes=available_bytes,
        is_sufficient=is_sufficient,
        buffer_percentage=buffer_percentage,
    )


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string.

    Args:
        bytes_value: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB")
    """
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.1f} KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.2f} GB"


# ---------------------------------------------------------------------------
# Git Utilities
# ---------------------------------------------------------------------------


def validate_git_installed() -> Tuple[bool, Optional[str]]:
    """Check if git is installed and accessible.

    Returns:
        Tuple of (is_installed, version_string)
    """
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False, None


def get_git_binary_path() -> Optional[str]:
    """Get path to git binary.

    Returns:
        Path to git binary or None if not found
    """
    try:
        result = subprocess.run(
            ["which", "git"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Try common locations
        common_paths = [
            "/usr/bin/git",
            "/usr/local/bin/git",
            "/opt/homebrew/bin/git",
        ]
        for path in common_paths:
            if Path(path).exists():
                return path
        return None


def parse_git_remote_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse git remote URL to extract host, owner, and repo.

    Args:
        url: Git remote URL (HTTPS or SSH)

    Returns:
        Tuple of (host, owner, repo) or (None, None, None) if parsing fails

    Examples:
        >>> parse_git_remote_url("https://github.com/owner/repo.git")
        ("github.com", "owner", "repo")
        >>> parse_git_remote_url("git@github.com:owner/repo.git")
        ("github.com", "owner", "repo")
    """
    import re

    # HTTPS URL: https://github.com/owner/repo.git
    https_pattern = r"https?://([^/]+)/([^/]+)/([^/]+?)(?:\.git)?$"
    match = re.match(https_pattern, url)
    if match:
        host, owner, repo = match.groups()
        return host, owner, repo

    # SSH URL: git@github.com:owner/repo.git
    ssh_pattern = r"git@([^:]+):([^/]+)/([^/]+?)(?:\.git)?$"
    match = re.match(ssh_pattern, url)
    if match:
        host, owner, repo = match.groups()
        return host, owner, repo

    return None, None, None


def estimate_git_clone_size(repo_size_kb: int, clone_depth: Optional[int] = None) -> int:
    """Estimate disk space required for git clone.

    Args:
        repo_size_kb: Repository size in kilobytes (from GitHub API)
        clone_depth: Shallow clone depth (None = full clone)

    Returns:
        Estimated size in bytes
    """
    # Convert KB to bytes
    repo_bytes = repo_size_kb * 1024

    # Full clone: repo size + ~30% overhead for .git directory
    if clone_depth is None:
        return int(repo_bytes * 1.3)

    # Shallow clone: estimate based on depth
    # Rough heuristic: full_size * (depth / 100) but minimum 20% of full size
    shallow_factor = max(0.2, min(1.0, clone_depth / 100))
    return int(repo_bytes * shallow_factor * 1.2)  # 20% overhead


# ---------------------------------------------------------------------------
# Formatting Utilities
# ---------------------------------------------------------------------------


def format_sync_statistics(result: SyncResult) -> str:
    """Format sync result as human-readable statistics.

    Args:
        result: Sync result to format

    Returns:
        Formatted statistics string
    """
    lines = [
        f"Sync completed: {result.status.value}",
        f"Mode: {result.sync_mode}",
        f"Files synced: {result.files_synced}",
        f"Files skipped: {result.files_skipped}",
        f"Files failed: {result.files_failed}",
        f"Data synced: {format_bytes(result.bytes_synced)}",
    ]

    if result.commits_processed > 0:
        lines.append(f"Commits processed: {result.commits_processed}")

    if result.branches_synced:
        lines.append(f"Branches: {', '.join(result.branches_synced)}")

    if result.duration_seconds:
        lines.append(f"Duration: {result.duration_seconds:.1f}s")

    if result.files_synced > 0 and result.duration_seconds:
        rate = result.files_synced / result.duration_seconds
        lines.append(f"Rate: {rate:.1f} files/sec")

    if result.error_message:
        lines.append(f"Error: {result.error_message}")

    return "\n".join(lines)


def format_progress(current: int, total: int, prefix: str = "Progress") -> str:
    """Format progress as percentage.

    Args:
        current: Current progress count
        total: Total count
        prefix: Prefix text

    Returns:
        Formatted progress string
    """
    if total == 0:
        percentage = 0.0
    else:
        percentage = (current / total) * 100

    return f"{prefix}: {current}/{total} ({percentage:.1f}%)"


def truncate_sha(sha: str, length: int = 7) -> str:
    """Truncate git SHA to short form.

    Args:
        sha: Full SHA string
        length: Desired length (default: 7)

    Returns:
        Truncated SHA
    """
    return sha[:length] if len(sha) >= length else sha


# ---------------------------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------------------------


def ensure_clone_directory(base_dir: Path, repo_id: str) -> Path:
    """Ensure clone directory exists and return path.

    Args:
        base_dir: Base directory for clones
        repo_id: Repository ID

    Returns:
        Path to clone directory
    """
    clone_dir = base_dir / repo_id
    clone_dir.mkdir(parents=True, exist_ok=True)
    return clone_dir


def get_default_clone_base_dir() -> Path:
    """Get default base directory for repository clones.

    Returns:
        Default clone base directory path
    """
    return Path.home() / ".futurnal" / "repositories" / "github"


def cleanup_clone_directory(clone_dir: Path, force: bool = False) -> bool:
    """Clean up clone directory.

    Args:
        clone_dir: Directory to clean up
        force: If True, remove even if not empty

    Returns:
        True if cleaned up successfully, False otherwise
    """
    if not clone_dir.exists():
        return True

    try:
        if force:
            shutil.rmtree(clone_dir)
            return True
        else:
            # Only remove if empty or only contains .git
            items = list(clone_dir.iterdir())
            if not items or (len(items) == 1 and items[0].name == ".git"):
                shutil.rmtree(clone_dir)
                return True
            return False
    except Exception:
        return False


__all__ = [
    "PatternMatcher",
    "check_disk_space_sufficient",
    "cleanup_clone_directory",
    "ensure_clone_directory",
    "estimate_git_clone_size",
    "format_bytes",
    "format_progress",
    "format_sync_statistics",
    "get_available_disk_space",
    "get_available_disk_space_gb",
    "get_default_clone_base_dir",
    "get_git_binary_path",
    "parse_git_remote_url",
    "truncate_sha",
    "validate_git_installed",
]
