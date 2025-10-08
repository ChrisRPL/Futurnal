"""Tests for GitHub quarantine handler.

This module tests the quarantine workflow for failed GitHub file processing,
including entry creation, retry logic, backoff calculations, and cleanup.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from futurnal.ingestion.github.quarantine import (
    GitHubQuarantineHandler,
    QuarantineEntry,
)


@pytest.fixture
def quarantine_dir(tmp_path):
    """Create temporary quarantine directory."""
    return tmp_path / "quarantine"


@pytest.fixture
def handler(quarantine_dir):
    """Create quarantine handler instance."""
    return GitHubQuarantineHandler(
        quarantine_dir=quarantine_dir,
        max_retries=3,
        base_backoff_seconds=60,
    )


def test_handler_initialization(handler, quarantine_dir):
    """Test quarantine handler initialization."""
    assert handler.quarantine_dir == quarantine_dir
    assert quarantine_dir.exists()
    assert handler.max_retries == 3
    assert handler.base_backoff_seconds == 60


def test_quarantine_file_basic(handler):
    """Test basic file quarantine."""
    repo_id = "test-repo-123"
    file_path = "src/test.py"
    content = b"def test(): pass"
    error = ValueError("Test error")

    # Quarantine file
    qid = handler.quarantine_file(
        repo_id=repo_id,
        file_path=file_path,
        content=content,
        error=error,
        retry_count=0,
    )

    # Verify quarantine ID format
    assert qid.startswith(repo_id[:8])
    assert len(qid) == 17  # repo_id[:8] + "_" + hash[:8]

    # Verify entry file exists
    entry_path = handler.quarantine_dir / f"{qid}.json"
    assert entry_path.exists()

    # Verify content file exists
    content_path = handler.quarantine_dir / f"{qid}.bin"
    assert content_path.exists()
    assert content_path.read_bytes() == content


def test_quarantine_entry_metadata(handler):
    """Test quarantine entry metadata."""
    repo_id = "test-repo-456"
    file_path = "lib/module.py"
    content = b"import os"
    error = RuntimeError("Processing failed")
    file_metadata = {"size": "8", "type": "python"}

    qid = handler.quarantine_file(
        repo_id=repo_id,
        file_path=file_path,
        content=content,
        error=error,
        retry_count=1,
        file_metadata=file_metadata,
    )

    # Load entry
    entry = handler.get_entry(qid)
    assert entry is not None
    assert entry.quarantine_id == qid
    assert entry.repo_id == repo_id
    assert entry.error_type == "RuntimeError"
    assert entry.error_message == "Processing failed"
    assert entry.retry_count == 1
    assert entry.max_retries == 3
    assert entry.content_size == len(content)
    assert entry.file_metadata == file_metadata
    assert entry.last_retry_at is None


def test_get_content(handler):
    """Test retrieving quarantined content."""
    content = b"test content with special chars: \x00\xff"
    qid = handler.quarantine_file(
        repo_id="repo-1",
        file_path="test.bin",
        content=content,
        error=Exception("test"),
    )

    # Retrieve content
    retrieved = handler.get_content(qid)
    assert retrieved == content


def test_get_nonexistent_entry(handler):
    """Test getting nonexistent entry returns None."""
    entry = handler.get_entry("nonexistent-id")
    assert entry is None

    content = handler.get_content("nonexistent-id")
    assert content is None


def test_list_entries_empty(handler):
    """Test listing entries when empty."""
    entries = handler.list_entries()
    assert entries == []


def test_list_entries_multiple(handler):
    """Test listing multiple entries."""
    # Create multiple quarantine entries
    qid1 = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file1.py",
        content=b"content1",
        error=Exception("error1"),
    )
    qid2 = handler.quarantine_file(
        repo_id="repo-2",
        file_path="file2.py",
        content=b"content2",
        error=Exception("error2"),
    )
    qid3 = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file3.py",
        content=b"content3",
        error=Exception("error3"),
        retry_count=4,  # Exhausted
    )

    # List all entries
    entries = handler.list_entries()
    assert len(entries) == 3
    entry_ids = [e.quarantine_id for e in entries]
    assert qid1 in entry_ids
    assert qid2 in entry_ids
    assert qid3 in entry_ids


def test_list_entries_filter_by_repo(handler):
    """Test filtering entries by repository ID."""
    qid1 = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file1.py",
        content=b"content1",
        error=Exception("error1"),
    )
    qid2 = handler.quarantine_file(
        repo_id="repo-2",
        file_path="file2.py",
        content=b"content2",
        error=Exception("error2"),
    )

    # Filter by repo-1
    entries = handler.list_entries(repo_id="repo-1")
    assert len(entries) == 1
    assert entries[0].quarantine_id == qid1

    # Filter by repo-2
    entries = handler.list_entries(repo_id="repo-2")
    assert len(entries) == 1
    assert entries[0].quarantine_id == qid2


def test_list_entries_retriable_only(handler):
    """Test filtering retriable entries."""
    qid1 = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file1.py",
        content=b"content1",
        error=Exception("error1"),
        retry_count=2,  # Can retry
    )
    qid2 = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file2.py",
        content=b"content2",
        error=Exception("error2"),
        retry_count=3,  # Exhausted
    )

    # Get only retriable
    entries = handler.list_entries(retriable_only=True)
    assert len(entries) == 1
    assert entries[0].quarantine_id == qid1


def test_can_retry_basic(handler):
    """Test can_retry logic."""
    # Create entry with retry count < max
    qid = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file.py",
        content=b"content",
        error=Exception("error"),
        retry_count=1,
    )

    # Should be able to retry
    assert handler.can_retry(qid) is True

    # Update to exhausted
    entry = handler.get_entry(qid)
    entry.retry_count = 3
    entry_path = handler.quarantine_dir / f"{qid}.json"
    entry_path.write_text(json.dumps(entry.to_dict(), indent=2))

    # Should not be able to retry
    assert handler.can_retry(qid) is False


def test_can_retry_nonexistent(handler):
    """Test can_retry with nonexistent entry."""
    assert handler.can_retry("nonexistent") is False


def test_mark_retry_attempted(handler):
    """Test marking retry attempt."""
    qid = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file.py",
        content=b"content",
        error=Exception("error"),
        retry_count=0,
    )

    # Get initial entry
    entry_before = handler.get_entry(qid)
    assert entry_before.retry_count == 0
    assert entry_before.last_retry_at is None

    # Mark retry
    handler.mark_retry_attempted(qid)

    # Verify retry was recorded
    entry_after = handler.get_entry(qid)
    assert entry_after.retry_count == 1
    assert entry_after.last_retry_at is not None
    assert isinstance(entry_after.last_retry_at, datetime)


def test_mark_retry_nonexistent(handler):
    """Test marking retry on nonexistent entry (should not crash)."""
    handler.mark_retry_attempted("nonexistent")
    # Should log warning but not crash


def test_remove_entry(handler):
    """Test removing quarantine entry."""
    qid = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file.py",
        content=b"content",
        error=Exception("error"),
    )

    # Verify files exist
    entry_path = handler.quarantine_dir / f"{qid}.json"
    content_path = handler.quarantine_dir / f"{qid}.bin"
    assert entry_path.exists()
    assert content_path.exists()

    # Remove entry
    removed = handler.remove_entry(qid)
    assert removed is True

    # Verify files removed
    assert not entry_path.exists()
    assert not content_path.exists()

    # Verify entry no longer accessible
    assert handler.get_entry(qid) is None
    assert handler.get_content(qid) is None


def test_remove_nonexistent_entry(handler):
    """Test removing nonexistent entry."""
    removed = handler.remove_entry("nonexistent")
    assert removed is False


def test_cleanup_exhausted(handler):
    """Test cleanup of exhausted entries."""
    # Create old exhausted entry
    qid1 = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file1.py",
        content=b"content1",
        error=Exception("error1"),
        retry_count=3,  # Exhausted
    )

    # Manually set old timestamp (31 days ago)
    entry = handler.get_entry(qid1)
    old_time = datetime.now(timezone.utc).timestamp() - (31 * 86400)
    entry.quarantined_at = datetime.fromtimestamp(old_time, tz=timezone.utc)
    entry_path = handler.quarantine_dir / f"{qid1}.json"
    entry_path.write_text(json.dumps(entry.to_dict(), indent=2))

    # Create recent exhausted entry
    qid2 = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file2.py",
        content=b"content2",
        error=Exception("error2"),
        retry_count=3,  # Exhausted but recent
    )

    # Create retriable entry
    qid3 = handler.quarantine_file(
        repo_id="repo-1",
        file_path="file3.py",
        content=b"content3",
        error=Exception("error3"),
        retry_count=1,  # Not exhausted
    )

    # Cleanup with 30 day threshold
    removed = handler.cleanup_exhausted(older_than_days=30)

    # Should remove only old exhausted entry
    assert removed == 1
    assert handler.get_entry(qid1) is None  # Removed
    assert handler.get_entry(qid2) is not None  # Kept (recent)
    assert handler.get_entry(qid3) is not None  # Kept (retriable)


def test_get_statistics(handler):
    """Test statistics generation."""
    # Create various entries
    handler.quarantine_file(
        repo_id="repo-1",
        file_path="file1.py",
        content=b"a" * 1000,
        error=ValueError("error1"),
        retry_count=1,
    )
    handler.quarantine_file(
        repo_id="repo-2",
        file_path="file2.py",
        content=b"b" * 2000,
        error=ValueError("error2"),
        retry_count=3,  # Exhausted
    )
    handler.quarantine_file(
        repo_id="repo-1",
        file_path="file3.py",
        content=b"c" * 500,
        error=RuntimeError("error3"),
        retry_count=0,
    )

    stats = handler.get_statistics()

    assert stats["total_entries"] == 3
    assert stats["retriable_entries"] == 2
    assert stats["exhausted_entries"] == 1
    assert stats["unique_repositories"] == 2
    assert stats["total_bytes_quarantined"] == 3500
    assert stats["error_types"]["ValueError"] == 2
    assert stats["error_types"]["RuntimeError"] == 1


def test_calculate_backoff(handler):
    """Test exponential backoff calculation."""
    # Verify exponential backoff
    assert handler._calculate_backoff(0) == 60  # 60 * 2^0
    assert handler._calculate_backoff(1) == 120  # 60 * 2^1
    assert handler._calculate_backoff(2) == 240  # 60 * 2^2
    assert handler._calculate_backoff(3) == 480  # 60 * 2^3


def test_quarantine_entry_serialization():
    """Test QuarantineEntry serialization and deserialization."""
    entry = QuarantineEntry(
        quarantine_id="test-id",
        repo_id="repo-123",
        file_path_hash="abc123def456",
        error_type="ValueError",
        error_message="Test error",
        retry_count=2,
        max_retries=3,
        quarantined_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        last_retry_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
        content_size=1024,
        file_metadata={"type": "python", "size": "1024"},
    )

    # Serialize to dict
    data = entry.to_dict()
    assert data["quarantine_id"] == "test-id"
    assert data["repo_id"] == "repo-123"
    assert data["error_type"] == "ValueError"
    assert data["retry_count"] == 2
    assert data["quarantined_at"] == "2024-01-01T12:00:00+00:00"
    assert data["last_retry_at"] == "2024-01-01T13:00:00+00:00"

    # Deserialize from dict
    restored = QuarantineEntry.from_dict(data)
    assert restored.quarantine_id == entry.quarantine_id
    assert restored.repo_id == entry.repo_id
    assert restored.error_type == entry.error_type
    assert restored.retry_count == entry.retry_count
    assert restored.quarantined_at == entry.quarantined_at
    assert restored.last_retry_at == entry.last_retry_at
    assert restored.file_metadata == entry.file_metadata


def test_quarantine_with_special_chars(handler):
    """Test quarantine with special characters in paths and content."""
    file_path = "src/λ-function/测试.py"
    content = "def test():\n    return 'こんにちは'".encode("utf-8")
    error = Exception("Failed: special © chars")

    qid = handler.quarantine_file(
        repo_id="repo-unicode",
        file_path=file_path,
        content=content,
        error=error,
    )

    # Verify retrieval
    entry = handler.get_entry(qid)
    assert entry is not None
    assert entry.error_message == "Failed: special © chars"

    retrieved_content = handler.get_content(qid)
    assert retrieved_content == content
