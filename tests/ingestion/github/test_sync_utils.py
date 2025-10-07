"""Tests for GitHub sync utility functions."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from futurnal.ingestion.github.sync_utils import (
    PatternMatcher,
    check_disk_space_sufficient,
    cleanup_clone_directory,
    ensure_clone_directory,
    estimate_git_clone_size,
    format_bytes,
    format_progress,
    format_sync_statistics,
    get_available_disk_space,
    get_available_disk_space_gb,
    get_default_clone_base_dir,
    get_git_binary_path,
    parse_git_remote_url,
    truncate_sha,
    validate_git_installed,
)
from futurnal.ingestion.github.sync_models import SyncResult, SyncStatus


# ---------------------------------------------------------------------------
# PatternMatcher Tests
# ---------------------------------------------------------------------------


def test_pattern_matcher_no_patterns():
    """Test matcher with no patterns includes everything."""
    matcher = PatternMatcher()

    assert matcher.should_include("any/file.py") is True
    assert matcher.should_include("another/path.txt") is True


def test_pattern_matcher_include_patterns():
    """Test include pattern matching."""
    matcher = PatternMatcher(include_patterns=["*.py", "*.md"])

    assert matcher.should_include("file.py") is True
    assert matcher.should_include("README.md") is True
    assert matcher.should_include("file.txt") is False
    assert matcher.should_include("src/main.py") is True


def test_pattern_matcher_exclude_patterns():
    """Test exclude pattern matching."""
    matcher = PatternMatcher(exclude_patterns=["tests/*", "*.pyc", "node_modules/"])

    assert matcher.should_include("src/main.py") is True
    assert matcher.should_include("tests/test_main.py") is False
    assert matcher.should_include("file.pyc") is False
    assert matcher.should_include("node_modules/package.json") is False


def test_pattern_matcher_combined_patterns():
    """Test combined include and exclude patterns."""
    matcher = PatternMatcher(
        include_patterns=["*.py"],
        exclude_patterns=["tests/*"],
    )

    assert matcher.should_include("src/main.py") is True
    assert matcher.should_include("tests/test_main.py") is False
    assert matcher.should_include("README.md") is False


def test_pattern_matcher_directory_patterns():
    """Test directory pattern matching."""
    matcher = PatternMatcher(exclude_patterns=[".git/", "node_modules/"])

    assert matcher.should_include(".git/config") is False
    assert matcher.should_include("src/.git/config") is False
    assert matcher.should_include("node_modules/pkg/index.js") is False
    assert matcher.should_include("src/main.py") is True


def test_pattern_matcher_filter_paths():
    """Test batch path filtering."""
    matcher = PatternMatcher(
        include_patterns=["*.py"],
        exclude_patterns=["tests/*"],
    )

    paths = [
        "src/main.py",
        "src/utils.py",
        "tests/test_main.py",
        "README.md",
    ]

    filtered = matcher.filter_paths(paths)

    assert "src/main.py" in filtered
    assert "src/utils.py" in filtered
    assert "tests/test_main.py" not in filtered
    assert "README.md" not in filtered


# ---------------------------------------------------------------------------
# Disk Space Utilities Tests
# ---------------------------------------------------------------------------


def test_get_available_disk_space():
    """Test getting available disk space."""
    # Should return non-negative number
    space = get_available_disk_space(Path.home())
    assert space >= 0


def test_get_available_disk_space_gb():
    """Test getting disk space in GB."""
    space_gb = get_available_disk_space_gb(Path.home())
    assert space_gb >= 0


def test_check_disk_space_sufficient():
    """Test disk space sufficiency check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Very small requirement should be sufficient
        estimate = check_disk_space_sufficient(
            required_bytes=1024,  # 1 KB
            path=Path(tmpdir),
            buffer_percentage=0.2,
        )

        assert estimate.is_sufficient is True
        assert estimate.estimated_size_bytes == 1024
        assert estimate.available_space_bytes > 0


def test_format_bytes():
    """Test byte formatting."""
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(1048576) == "1.0 MB"
    assert format_bytes(1073741824) == "1.00 GB"


# ---------------------------------------------------------------------------
# Git Utilities Tests
# ---------------------------------------------------------------------------


def test_validate_git_installed():
    """Test git installation validation."""
    is_installed, version = validate_git_installed()

    # Git should be installed in test environment
    assert is_installed is True
    if version:
        assert "git version" in version.lower()


def test_get_git_binary_path():
    """Test getting git binary path."""
    git_path = get_git_binary_path()

    # Should find git (or return None gracefully)
    if git_path:
        assert isinstance(git_path, str)
        assert len(git_path) > 0


def test_parse_git_remote_url_https():
    """Test parsing HTTPS git URLs."""
    host, owner, repo = parse_git_remote_url(
        "https://github.com/octocat/Hello-World.git"
    )

    assert host == "github.com"
    assert owner == "octocat"
    assert repo == "Hello-World"


def test_parse_git_remote_url_https_no_extension():
    """Test parsing HTTPS URLs without .git extension."""
    host, owner, repo = parse_git_remote_url(
        "https://github.com/octocat/Hello-World"
    )

    assert host == "github.com"
    assert owner == "octocat"
    assert repo == "Hello-World"


def test_parse_git_remote_url_ssh():
    """Test parsing SSH git URLs."""
    host, owner, repo = parse_git_remote_url(
        "git@github.com:octocat/Hello-World.git"
    )

    assert host == "github.com"
    assert owner == "octocat"
    assert repo == "Hello-World"


def test_parse_git_remote_url_enterprise():
    """Test parsing GitHub Enterprise URLs."""
    host, owner, repo = parse_git_remote_url(
        "https://github.company.com/team/project.git"
    )

    assert host == "github.company.com"
    assert owner == "team"
    assert repo == "project"


def test_parse_git_remote_url_invalid():
    """Test parsing invalid URLs."""
    host, owner, repo = parse_git_remote_url("invalid-url")

    assert host is None
    assert owner is None
    assert repo is None


def test_estimate_git_clone_size_full():
    """Test clone size estimation for full clone."""
    # 1000 KB repo with full clone
    estimated = estimate_git_clone_size(repo_size_kb=1000, clone_depth=None)

    # Should be ~1.3x original size (30% overhead)
    assert 1200000 <= estimated <= 1400000


def test_estimate_git_clone_size_shallow():
    """Test clone size estimation for shallow clone."""
    # 10000 KB repo with depth 10
    estimated = estimate_git_clone_size(repo_size_kb=10000, clone_depth=10)

    # Should be much smaller than full clone
    full_clone = estimate_git_clone_size(repo_size_kb=10000, clone_depth=None)
    assert estimated < full_clone


# ---------------------------------------------------------------------------
# Formatting Utilities Tests
# ---------------------------------------------------------------------------


def test_format_sync_statistics():
    """Test sync statistics formatting."""
    result = SyncResult(
        repo_id="test-repo",
        sync_mode="graphql_api",
        status=SyncStatus.COMPLETED,
        files_synced=100,
        files_skipped=10,
        files_failed=5,
        bytes_synced=5242880,  # 5 MB
        commits_processed=10,
        branches_synced=["main", "develop"],
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_seconds=30.5,
    )

    stats = format_sync_statistics(result)

    assert "Sync completed: completed" in stats
    assert "Files synced: 100" in stats
    assert "Files skipped: 10" in stats
    assert "Files failed: 5" in stats
    assert "Data synced:" in stats
    assert "Commits processed: 10" in stats
    assert "Branches: main, develop" in stats
    assert "Duration: 30.5s" in stats


def test_format_progress():
    """Test progress formatting."""
    progress = format_progress(50, 100, prefix="Syncing")

    assert "Syncing: 50/100 (50.0%)" in progress


def test_format_progress_zero_total():
    """Test progress formatting with zero total."""
    progress = format_progress(0, 0)

    assert "0/0 (0.0%)" in progress


def test_truncate_sha():
    """Test SHA truncation."""
    long_sha = "1234567890abcdef1234567890abcdef12345678"

    assert truncate_sha(long_sha) == "1234567"
    assert truncate_sha(long_sha, length=10) == "1234567890"
    assert truncate_sha("abc", length=7) == "abc"  # Shorter than requested


# ---------------------------------------------------------------------------
# Path Utilities Tests
# ---------------------------------------------------------------------------


def test_ensure_clone_directory():
    """Test clone directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        repo_id = "test-repo-123"

        clone_dir = ensure_clone_directory(base_dir, repo_id)

        assert clone_dir.exists()
        assert clone_dir.is_dir()
        assert clone_dir.name == repo_id


def test_get_default_clone_base_dir():
    """Test default clone directory."""
    default_dir = get_default_clone_base_dir()

    assert default_dir == Path.home() / ".futurnal" / "repositories" / "github"


def test_cleanup_clone_directory_empty():
    """Test cleaning up empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "test-repo"
        clone_dir.mkdir()

        result = cleanup_clone_directory(clone_dir, force=False)

        assert result is True
        assert not clone_dir.exists()


def test_cleanup_clone_directory_with_files():
    """Test cleaning up directory with files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "test-repo"
        clone_dir.mkdir()
        (clone_dir / "file.txt").write_text("test")

        # Without force, should not clean up
        result = cleanup_clone_directory(clone_dir, force=False)
        assert result is False
        assert clone_dir.exists()

        # With force, should clean up
        result = cleanup_clone_directory(clone_dir, force=True)
        assert result is True
        assert not clone_dir.exists()


def test_cleanup_clone_directory_with_git():
    """Test cleaning up directory with only .git."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "test-repo"
        clone_dir.mkdir()
        git_dir = clone_dir / ".git"
        git_dir.mkdir()

        # Directory with only .git should be cleaned up even without force
        result = cleanup_clone_directory(clone_dir, force=False)
        assert result is True
        assert not clone_dir.exists()


def test_cleanup_nonexistent_directory():
    """Test cleaning up non-existent directory."""
    result = cleanup_clone_directory(Path("/nonexistent/path"))

    assert result is True  # Returns True if already doesn't exist
