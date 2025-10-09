"""Tests for document versioning and provenance tracking.

Tests cover:
- DocumentVersionRecord validation and invariants
- DocumentVersionStore CRUD operations
- ProvenanceTracker change detection
- Version chain tracking (parent_hash linkage)
- Temporal metadata handling
- Thread safety and concurrent access
- Hash computation stability
- Edge cases and error handling
- Privacy guarantees (no content storage)
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from futurnal.pipeline.versioning import (
    DocumentVersionRecord,
    DocumentVersionStore,
    ProvenanceTracker,
)
from futurnal.pipeline.models import compute_content_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide temporary database path for testing."""
    return tmp_path / "test_versions.db"


@pytest.fixture
def version_store(temp_db_path: Path) -> DocumentVersionStore:
    """Provide DocumentVersionStore instance for testing."""
    store = DocumentVersionStore(temp_db_path)
    yield store
    store.close()


@pytest.fixture
def provenance_tracker(version_store: DocumentVersionStore) -> ProvenanceTracker:
    """Provide ProvenanceTracker instance for testing."""
    return ProvenanceTracker(version_store)


# ---------------------------------------------------------------------------
# DocumentVersionRecord Tests
# ---------------------------------------------------------------------------


def test_version_record_creation():
    """Test basic DocumentVersionRecord creation."""
    now = datetime.now(timezone.utc)

    record = DocumentVersionRecord(
        source_path="/vault/note.md",
        content_hash="a" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now,
        version_number=1
    )

    assert record.source_path == "/vault/note.md"
    assert record.content_hash == "a" * 64
    assert record.parent_hash is None
    assert record.version_number == 1
    assert record.ingested_at.tzinfo == timezone.utc


def test_version_record_with_parent():
    """Test version record with parent hash."""
    now = datetime.now(timezone.utc)

    record = DocumentVersionRecord(
        source_path="/vault/note.md",
        content_hash="b" * 64,
        parent_hash="a" * 64,
        created_at=None,
        modified_at=now,
        ingested_at=now,
        version_number=2
    )

    assert record.parent_hash == "a" * 64
    assert record.version_number == 2


def test_version_record_immutable():
    """Test that DocumentVersionRecord is immutable."""
    now = datetime.now(timezone.utc)

    record = DocumentVersionRecord(
        source_path="/test.md",
        content_hash="c" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now,
        version_number=1
    )

    with pytest.raises(AttributeError):
        record.content_hash = "d" * 64  # type: ignore


def test_version_record_validates_version_number():
    """Test version number validation."""
    now = datetime.now(timezone.utc)

    # Valid version number
    record = DocumentVersionRecord(
        source_path="/test.md",
        content_hash="e" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now,
        version_number=1
    )
    assert record.version_number == 1

    # Invalid version number (0)
    with pytest.raises(ValueError, match="version_number must be >= 1"):
        DocumentVersionRecord(
            source_path="/test.md",
            content_hash="f" * 64,
            parent_hash=None,
            created_at=now,
            modified_at=now,
            ingested_at=now,
            version_number=0
        )

    # Invalid version number (negative)
    with pytest.raises(ValueError, match="version_number must be >= 1"):
        DocumentVersionRecord(
            source_path="/test.md",
            content_hash="1" * 64,
            parent_hash=None,
            created_at=now,
            modified_at=now,
            ingested_at=now,
            version_number=-1
        )


def test_version_record_validates_content_hash():
    """Test content hash validation."""
    now = datetime.now(timezone.utc)

    # Valid hash
    record = DocumentVersionRecord(
        source_path="/test.md",
        content_hash="2" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now,
        version_number=1
    )
    assert len(record.content_hash) == 64

    # Invalid hash (wrong length)
    with pytest.raises(ValueError, match="content_hash must be valid SHA-256"):
        DocumentVersionRecord(
            source_path="/test.md",
            content_hash="abc123",
            parent_hash=None,
            created_at=now,
            modified_at=now,
            ingested_at=now,
            version_number=1
        )


def test_version_record_validates_parent_hash():
    """Test parent hash validation."""
    now = datetime.now(timezone.utc)

    # Valid parent hash
    record = DocumentVersionRecord(
        source_path="/test.md",
        content_hash="3" * 64,
        parent_hash="4" * 64,
        created_at=now,
        modified_at=now,
        ingested_at=now,
        version_number=2
    )
    assert len(record.parent_hash) == 64

    # Invalid parent hash (wrong length)
    with pytest.raises(ValueError, match="parent_hash must be valid SHA-256"):
        DocumentVersionRecord(
            source_path="/test.md",
            content_hash="5" * 64,
            parent_hash="invalid",
            created_at=now,
            modified_at=now,
            ingested_at=now,
            version_number=2
        )


def test_version_record_validates_source_path():
    """Test source path validation."""
    now = datetime.now(timezone.utc)

    # Empty source path
    with pytest.raises(ValueError, match="source_path cannot be empty"):
        DocumentVersionRecord(
            source_path="",
            content_hash="6" * 64,
            parent_hash=None,
            created_at=now,
            modified_at=now,
            ingested_at=now,
            version_number=1
        )


def test_version_record_timezone_aware():
    """Test that naive datetimes are converted to UTC."""
    naive_dt = datetime(2025, 1, 1, 12, 0, 0)  # No timezone

    record = DocumentVersionRecord(
        source_path="/test.md",
        content_hash="7" * 64,
        parent_hash=None,
        created_at=naive_dt,
        modified_at=naive_dt,
        ingested_at=naive_dt,
        version_number=1
    )

    assert record.created_at.tzinfo == timezone.utc
    assert record.modified_at.tzinfo == timezone.utc
    assert record.ingested_at.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# DocumentVersionStore Tests
# ---------------------------------------------------------------------------


def test_version_store_initialization(temp_db_path: Path):
    """Test DocumentVersionStore initialization."""
    store = DocumentVersionStore(temp_db_path)

    assert temp_db_path.exists()
    assert temp_db_path.is_file()

    store.close()


def test_version_store_creates_tables(version_store: DocumentVersionStore):
    """Test that required tables are created."""
    cursor = version_store._conn.cursor()

    # Check document_versions table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='document_versions'"
    )
    assert cursor.fetchone() is not None

    # Check version_history table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='version_history'"
    )
    assert cursor.fetchone() is not None


def test_version_store_record_first_version(version_store: DocumentVersionStore):
    """Test recording first version of a document."""
    now = datetime.now(timezone.utc)

    record = version_store.record_version(
        source_path="/vault/note.md",
        content_hash="a" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    assert record.version_number == 1
    assert record.content_hash == "a" * 64
    assert record.parent_hash is None


def test_version_store_record_subsequent_version(version_store: DocumentVersionStore):
    """Test recording subsequent versions increments version number."""
    now = datetime.now(timezone.utc)

    # Record first version
    v1 = version_store.record_version(
        source_path="/vault/note.md",
        content_hash="a" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )
    assert v1.version_number == 1

    # Record second version
    v2 = version_store.record_version(
        source_path="/vault/note.md",
        content_hash="b" * 64,
        parent_hash="a" * 64,
        created_at=now,
        modified_at=now + timedelta(hours=1),
        ingested_at=now + timedelta(hours=1)
    )
    assert v2.version_number == 2
    assert v2.parent_hash == "a" * 64

    # Record third version
    v3 = version_store.record_version(
        source_path="/vault/note.md",
        content_hash="c" * 64,
        parent_hash="b" * 64,
        created_at=now,
        modified_at=now + timedelta(hours=2),
        ingested_at=now + timedelta(hours=2)
    )
    assert v3.version_number == 3
    assert v3.parent_hash == "b" * 64


def test_version_store_get_current_version(version_store: DocumentVersionStore):
    """Test retrieving current version."""
    now = datetime.now(timezone.utc)

    # No version yet
    current = version_store.get_current_version("/vault/note.md")
    assert current is None

    # Record version
    version_store.record_version(
        source_path="/vault/note.md",
        content_hash="a" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    # Should retrieve current version
    current = version_store.get_current_version("/vault/note.md")
    assert current is not None
    assert current.content_hash == "a" * 64
    assert current.version_number == 1

    # Update to new version
    version_store.record_version(
        source_path="/vault/note.md",
        content_hash="b" * 64,
        parent_hash="a" * 64,
        created_at=now,
        modified_at=now + timedelta(hours=1),
        ingested_at=now + timedelta(hours=1)
    )

    # Should retrieve updated version
    current = version_store.get_current_version("/vault/note.md")
    assert current is not None
    assert current.content_hash == "b" * 64
    assert current.version_number == 2


def test_version_store_get_version_history(version_store: DocumentVersionStore):
    """Test retrieving version history."""
    now = datetime.now(timezone.utc)

    # No history yet
    history = version_store.get_version_history("/vault/note.md")
    assert len(history) == 0

    # Record three versions
    version_store.record_version(
        source_path="/vault/note.md",
        content_hash="a" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    version_store.record_version(
        source_path="/vault/note.md",
        content_hash="b" * 64,
        parent_hash="a" * 64,
        created_at=now,
        modified_at=now + timedelta(hours=1),
        ingested_at=now + timedelta(hours=1)
    )

    version_store.record_version(
        source_path="/vault/note.md",
        content_hash="c" * 64,
        parent_hash="b" * 64,
        created_at=now,
        modified_at=now + timedelta(hours=2),
        ingested_at=now + timedelta(hours=2)
    )

    # Retrieve full history
    history = version_store.get_version_history("/vault/note.md")
    assert len(history) == 3

    # Should be in reverse chronological order (newest first)
    assert history[0].content_hash == "c" * 64
    assert history[0].version_number == 3
    assert history[1].content_hash == "b" * 64
    assert history[1].version_number == 2
    assert history[2].content_hash == "a" * 64
    assert history[2].version_number == 1


def test_version_store_get_version_history_with_limit(version_store: DocumentVersionStore):
    """Test retrieving limited version history."""
    now = datetime.now(timezone.utc)

    # Record three versions
    for i in range(3):
        version_store.record_version(
            source_path="/vault/note.md",
            content_hash=str(i) * 64,
            parent_hash=str(i-1) * 64 if i > 0 else None,
            created_at=now,
            modified_at=now + timedelta(hours=i),
            ingested_at=now + timedelta(hours=i)
        )

    # Get last 2 versions
    history = version_store.get_version_history("/vault/note.md", limit=2)
    assert len(history) == 2
    assert history[0].content_hash == "2" * 64
    assert history[1].content_hash == "1" * 64


def test_version_store_get_version_count(version_store: DocumentVersionStore):
    """Test getting version count."""
    now = datetime.now(timezone.utc)

    # No versions yet
    count = version_store.get_version_count("/vault/note.md")
    assert count == 0

    # Record version
    version_store.record_version(
        source_path="/vault/note.md",
        content_hash="a" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    count = version_store.get_version_count("/vault/note.md")
    assert count == 1

    # Record another version
    version_store.record_version(
        source_path="/vault/note.md",
        content_hash="b" * 64,
        parent_hash="a" * 64,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    count = version_store.get_version_count("/vault/note.md")
    assert count == 2


def test_version_store_multiple_documents(version_store: DocumentVersionStore):
    """Test tracking versions for multiple documents."""
    now = datetime.now(timezone.utc)

    # Record versions for multiple documents
    version_store.record_version(
        source_path="/vault/note1.md",
        content_hash="a" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    version_store.record_version(
        source_path="/vault/note2.md",
        content_hash="b" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    version_store.record_version(
        source_path="/vault/note3.md",
        content_hash="c" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    # Each document should have independent version tracking
    assert version_store.get_version_count("/vault/note1.md") == 1
    assert version_store.get_version_count("/vault/note2.md") == 1
    assert version_store.get_version_count("/vault/note3.md") == 1

    # Total documents
    assert version_store.get_document_count() == 3


def test_version_store_iter_all_documents(version_store: DocumentVersionStore):
    """Test iterating over all tracked documents."""
    now = datetime.now(timezone.utc)

    # Record versions for multiple documents
    for i in range(5):
        version_store.record_version(
            source_path=f"/vault/note{i}.md",
            content_hash=str(i) * 64,
            parent_hash=None,
            created_at=now,
            modified_at=now,
            ingested_at=now
        )

    # Iterate and collect
    all_docs = list(version_store.iter_all_documents())
    assert len(all_docs) == 5

    # Verify source paths
    source_paths = {doc.source_path for doc in all_docs}
    assert source_paths == {f"/vault/note{i}.md" for i in range(5)}


def test_version_store_validates_hash_format(version_store: DocumentVersionStore):
    """Test hash format validation in record_version."""
    now = datetime.now(timezone.utc)

    # Invalid content hash (too short)
    with pytest.raises(ValueError, match="content_hash must be valid SHA-256"):
        version_store.record_version(
            source_path="/test.md",
            content_hash="abc",
            parent_hash=None,
            created_at=now,
            modified_at=now,
            ingested_at=now
        )

    # Invalid parent hash (too short)
    with pytest.raises(ValueError, match="parent_hash must be valid SHA-256"):
        version_store.record_version(
            source_path="/test.md",
            content_hash="d" * 64,
            parent_hash="xyz",
            created_at=now,
            modified_at=now,
            ingested_at=now
        )


def test_version_store_thread_safety(version_store: DocumentVersionStore):
    """Test concurrent access to version store."""
    now = datetime.now(timezone.utc)

    def record_versions(thread_id: int):
        for i in range(10):
            version_store.record_version(
                source_path=f"/vault/thread{thread_id}_note{i}.md",
                content_hash=f"{thread_id}{i}".ljust(64, '0'),
                parent_hash=None,
                created_at=now,
                modified_at=now,
                ingested_at=now
            )

    # Run multiple threads concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(record_versions, i) for i in range(5)]
        for future in futures:
            future.result()

    # Verify all documents were recorded
    assert version_store.get_document_count() == 50


def test_version_store_privacy_no_content_stored(version_store: DocumentVersionStore):
    """Test that no document content is stored, only hashes."""
    now = datetime.now(timezone.utc)
    sensitive_content = "This is sensitive personal data that should not be stored"

    # Compute hash but don't store content
    content_hash = compute_content_hash(sensitive_content)

    version_store.record_version(
        source_path="/vault/private.md",
        content_hash=content_hash,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )

    # Verify content is not in database
    cursor = version_store._conn.cursor()
    cursor.execute("SELECT * FROM document_versions WHERE source_path = ?", ("/vault/private.md",))
    row = cursor.fetchone()

    # Convert row to string and check content is not present
    row_str = str(row)
    assert sensitive_content not in row_str
    assert content_hash in row_str  # Hash should be present


# ---------------------------------------------------------------------------
# ProvenanceTracker Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provenance_tracker_compute_hash(provenance_tracker: ProvenanceTracker):
    """Test hash computation."""
    content = "Hello, world!"
    hash1 = provenance_tracker.compute_hash(content)

    # Should be valid SHA-256
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)

    # Should be deterministic
    hash2 = provenance_tracker.compute_hash(content)
    assert hash1 == hash2

    # Different content should produce different hash
    hash3 = provenance_tracker.compute_hash("Different content")
    assert hash1 != hash3


@pytest.mark.asyncio
async def test_provenance_tracker_detect_change_new_document(provenance_tracker: ProvenanceTracker):
    """Test change detection for new document."""
    content = "This is a new document"
    content_hash = provenance_tracker.compute_hash(content)

    has_changed, prev_hash = await provenance_tracker.detect_change(
        source_path="/vault/new.md",
        content_hash=content_hash
    )

    assert has_changed is True
    assert prev_hash is None


@pytest.mark.asyncio
async def test_provenance_tracker_detect_change_unchanged_document(provenance_tracker: ProvenanceTracker):
    """Test change detection for unchanged document."""
    now = datetime.now(timezone.utc)
    content = "This is unchanged content"
    content_hash = provenance_tracker.compute_hash(content)

    # Record initial version
    await provenance_tracker.record_version(
        source_path="/vault/unchanged.md",
        content_hash=content_hash,
        parent_hash=None,
        timestamp=now
    )

    # Check again with same hash
    has_changed, prev_hash = await provenance_tracker.detect_change(
        source_path="/vault/unchanged.md",
        content_hash=content_hash
    )

    assert has_changed is False
    assert prev_hash is None  # No previous hash when unchanged


@pytest.mark.asyncio
async def test_provenance_tracker_detect_change_modified_document(provenance_tracker: ProvenanceTracker):
    """Test change detection for modified document."""
    now = datetime.now(timezone.utc)

    # Original content
    original_content = "Original content"
    original_hash = provenance_tracker.compute_hash(original_content)

    # Record initial version
    await provenance_tracker.record_version(
        source_path="/vault/modified.md",
        content_hash=original_hash,
        parent_hash=None,
        timestamp=now
    )

    # Modified content
    modified_content = "Modified content"
    modified_hash = provenance_tracker.compute_hash(modified_content)

    # Check with new hash
    has_changed, prev_hash = await provenance_tracker.detect_change(
        source_path="/vault/modified.md",
        content_hash=modified_hash
    )

    assert has_changed is True
    assert prev_hash == original_hash


@pytest.mark.asyncio
async def test_provenance_tracker_record_version(provenance_tracker: ProvenanceTracker):
    """Test version recording."""
    now = datetime.now(timezone.utc)
    content = "Test content"
    content_hash = provenance_tracker.compute_hash(content)

    record = await provenance_tracker.record_version(
        source_path="/vault/test.md",
        content_hash=content_hash,
        parent_hash=None,
        timestamp=now,
        created_at=now,
        modified_at=now
    )

    assert record.version_number == 1
    assert record.content_hash == content_hash
    assert record.parent_hash is None


@pytest.mark.asyncio
async def test_provenance_tracker_version_chain(provenance_tracker: ProvenanceTracker):
    """Test building version chain with parent hashes."""
    now = datetime.now(timezone.utc)

    # Version 1
    content1 = "Version 1 content"
    hash1 = provenance_tracker.compute_hash(content1)

    v1 = await provenance_tracker.record_version(
        source_path="/vault/doc.md",
        content_hash=hash1,
        parent_hash=None,
        timestamp=now
    )
    assert v1.version_number == 1

    # Version 2
    content2 = "Version 2 content"
    hash2 = provenance_tracker.compute_hash(content2)

    v2 = await provenance_tracker.record_version(
        source_path="/vault/doc.md",
        content_hash=hash2,
        parent_hash=hash1,
        timestamp=now + timedelta(hours=1)
    )
    assert v2.version_number == 2
    assert v2.parent_hash == hash1

    # Version 3
    content3 = "Version 3 content"
    hash3 = provenance_tracker.compute_hash(content3)

    v3 = await provenance_tracker.record_version(
        source_path="/vault/doc.md",
        content_hash=hash3,
        parent_hash=hash2,
        timestamp=now + timedelta(hours=2)
    )
    assert v3.version_number == 3
    assert v3.parent_hash == hash2

    # Verify version chain
    history = provenance_tracker.get_version_history("/vault/doc.md")
    assert len(history) == 3
    assert history[0].content_hash == hash3  # Newest first
    assert history[1].content_hash == hash2
    assert history[2].content_hash == hash1


@pytest.mark.asyncio
async def test_provenance_tracker_get_version_count(provenance_tracker: ProvenanceTracker):
    """Test getting version count."""
    now = datetime.now(timezone.utc)

    # No versions
    count = provenance_tracker.get_version_count("/vault/doc.md")
    assert count == 0

    # Record versions
    for i in range(5):
        content = f"Version {i} content"
        content_hash = provenance_tracker.compute_hash(content)
        await provenance_tracker.record_version(
            source_path="/vault/doc.md",
            content_hash=content_hash,
            parent_hash=None,
            timestamp=now + timedelta(hours=i)
        )

    count = provenance_tracker.get_version_count("/vault/doc.md")
    assert count == 5


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_content_hash(provenance_tracker: ProvenanceTracker):
    """Test hashing empty content."""
    empty_content = ""
    hash_result = provenance_tracker.compute_hash(empty_content)

    assert len(hash_result) == 64
    assert all(c in "0123456789abcdef" for c in hash_result)

    # Empty content should have consistent hash
    hash2 = provenance_tracker.compute_hash("")
    assert hash_result == hash2


@pytest.mark.asyncio
async def test_unicode_content_hash(provenance_tracker: ProvenanceTracker):
    """Test hashing Unicode content."""
    unicode_content = "Hello, 世界! 🌍 Emoji and CJK: 你好"
    hash_result = provenance_tracker.compute_hash(unicode_content)

    assert len(hash_result) == 64
    assert all(c in "0123456789abcdef" for c in hash_result)

    # Should be deterministic
    hash2 = provenance_tracker.compute_hash(unicode_content)
    assert hash_result == hash2


@pytest.mark.asyncio
async def test_large_content_hash(provenance_tracker: ProvenanceTracker):
    """Test hashing large content."""
    large_content = "x" * 10_000_000  # 10MB
    hash_result = provenance_tracker.compute_hash(large_content)

    assert len(hash_result) == 64
    assert all(c in "0123456789abcdef" for c in hash_result)


def test_version_store_persistence(temp_db_path: Path):
    """Test that versions persist across store reopening."""
    now = datetime.now(timezone.utc)

    # Create store and record version
    store1 = DocumentVersionStore(temp_db_path)
    store1.record_version(
        source_path="/vault/persistent.md",
        content_hash="a" * 64,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now
    )
    store1.close()

    # Reopen store
    store2 = DocumentVersionStore(temp_db_path)
    current = store2.get_current_version("/vault/persistent.md")

    assert current is not None
    assert current.content_hash == "a" * 64
    assert current.version_number == 1

    store2.close()


@pytest.mark.asyncio
async def test_concurrent_change_detection(provenance_tracker: ProvenanceTracker):
    """Test concurrent change detection calls."""
    now = datetime.now(timezone.utc)

    # Record initial version
    content = "Initial content"
    content_hash = provenance_tracker.compute_hash(content)
    await provenance_tracker.record_version(
        source_path="/vault/concurrent.md",
        content_hash=content_hash,
        parent_hash=None,
        timestamp=now
    )

    # Multiple concurrent change detections
    async def check_change():
        return await provenance_tracker.detect_change(
            source_path="/vault/concurrent.md",
            content_hash=content_hash
        )

    results = await asyncio.gather(*[check_change() for _ in range(10)])

    # All should report no change
    for has_changed, prev_hash in results:
        assert has_changed is False
        assert prev_hash is None


# ---------------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.asyncio
async def test_hash_computation_performance(provenance_tracker: ProvenanceTracker):
    """Benchmark hash computation performance."""
    import time

    # Test with 1MB document
    content = "x" * 1_000_000
    start = time.perf_counter()

    for _ in range(100):
        provenance_tracker.compute_hash(content)

    elapsed = time.perf_counter() - start

    # Should process at least 100MB/s (100 iterations * 1MB)
    throughput_mb_s = 100 / elapsed
    assert throughput_mb_s > 100, f"Hash computation too slow: {throughput_mb_s:.2f} MB/s"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_version_tracking_performance(provenance_tracker: ProvenanceTracker):
    """Benchmark version tracking throughput."""
    import time

    now = datetime.now(timezone.utc)
    start = time.perf_counter()

    # Record 1000 versions across 100 documents
    for doc_id in range(100):
        for version in range(10):
            content = f"Document {doc_id} version {version}"
            content_hash = provenance_tracker.compute_hash(content)
            await provenance_tracker.record_version(
                source_path=f"/vault/doc{doc_id}.md",
                content_hash=content_hash,
                parent_hash=None,
                timestamp=now
            )

    elapsed = time.perf_counter() - start
    throughput = 1000 / elapsed

    # Should handle at least 100 versions per second
    assert throughput > 100, f"Version tracking too slow: {throughput:.2f} versions/s"


@pytest.mark.performance
def test_version_history_query_performance(version_store: DocumentVersionStore):
    """Benchmark version history query performance."""
    import time

    now = datetime.now(timezone.utc)

    # Create document with 1000 versions
    for i in range(1000):
        version_store.record_version(
            source_path="/vault/large_history.md",
            content_hash=str(i).ljust(64, '0'),
            parent_hash=str(i-1).ljust(64, '0') if i > 0 else None,
            created_at=now,
            modified_at=now,
            ingested_at=now
        )

    # Benchmark full history retrieval
    start = time.perf_counter()
    history = version_store.get_version_history("/vault/large_history.md")
    elapsed = time.perf_counter() - start

    assert len(history) == 1000
    # Should retrieve in under 100ms
    assert elapsed < 0.1, f"History query too slow: {elapsed*1000:.2f}ms"
