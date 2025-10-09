"""Integration tests for versioning with normalization pipeline.

Tests cover:
- Full pipeline: source document → normalization → version tracking
- Multi-version document evolution scenarios
- Integration with NormalizedMetadata
- Concurrent document processing
- Real-world usage patterns
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from futurnal.pipeline.versioning import (
    DocumentVersionStore,
    ProvenanceTracker,
)
from futurnal.pipeline.models import (
    NormalizedDocument,
    NormalizedMetadata,
    DocumentFormat,
    compute_content_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide temporary database path for testing."""
    return tmp_path / "test_integration_versions.db"


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
# Pipeline Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalized_document_with_versioning(provenance_tracker: ProvenanceTracker):
    """Test versioning integration with NormalizedDocument."""
    now = datetime.now(timezone.utc)

    # Create normalized document
    content = "This is the document content."
    content_hash = compute_content_hash(content)

    metadata = NormalizedMetadata(
        source_path="/vault/note.md",
        source_id="note123",
        source_type="obsidian_vault",
        format=DocumentFormat.MARKDOWN,
        content_type="text/markdown",
        character_count=len(content),
        word_count=5,
        line_count=1,
        content_hash=content_hash,
        parent_hash=None,  # First version
        created_at=now,
        modified_at=now,
        ingested_at=now,
    )

    doc = NormalizedDocument(
        document_id="doc123",
        sha256=content_hash,
        content=content,
        metadata=metadata,
    )

    # Detect change (should be new document)
    has_changed, prev_hash = await provenance_tracker.detect_change(
        source_path=doc.metadata.source_path,
        content_hash=doc.metadata.content_hash
    )

    assert has_changed is True
    assert prev_hash is None

    # Record version
    version_record = await provenance_tracker.record_version(
        source_path=doc.metadata.source_path,
        content_hash=doc.metadata.content_hash,
        parent_hash=doc.metadata.parent_hash,
        timestamp=doc.metadata.ingested_at,
        created_at=doc.metadata.created_at,
        modified_at=doc.metadata.modified_at
    )

    assert version_record.version_number == 1
    assert version_record.content_hash == content_hash


@pytest.mark.asyncio
async def test_document_modification_detection(provenance_tracker: ProvenanceTracker):
    """Test detecting document modifications through version tracking."""
    now = datetime.now(timezone.utc)
    source_path = "/vault/evolving_note.md"

    # Version 1: Original content
    original_content = "# Original Title\n\nOriginal content here."
    original_hash = compute_content_hash(original_content)

    metadata_v1 = NormalizedMetadata(
        source_path=source_path,
        source_id="note456",
        source_type="obsidian_vault",
        format=DocumentFormat.MARKDOWN,
        content_type="text/markdown",
        character_count=len(original_content),
        word_count=5,
        line_count=3,
        content_hash=original_hash,
        parent_hash=None,
        created_at=now,
        modified_at=now,
        ingested_at=now,
    )

    # Record first version
    has_changed_v1, prev_hash_v1 = await provenance_tracker.detect_change(
        source_path=source_path,
        content_hash=original_hash
    )
    assert has_changed_v1 is True
    assert prev_hash_v1 is None

    await provenance_tracker.record_version(
        source_path=source_path,
        content_hash=original_hash,
        parent_hash=None,
        timestamp=now,
        created_at=metadata_v1.created_at,
        modified_at=metadata_v1.modified_at
    )

    # Version 2: Modified content (1 hour later)
    modified_content = "# Updated Title\n\nUpdated content with more details."
    modified_hash = compute_content_hash(modified_content)
    modified_time = now + timedelta(hours=1)

    metadata_v2 = NormalizedMetadata(
        source_path=source_path,
        source_id="note456",
        source_type="obsidian_vault",
        format=DocumentFormat.MARKDOWN,
        content_type="text/markdown",
        character_count=len(modified_content),
        word_count=7,
        line_count=3,
        content_hash=modified_hash,
        parent_hash=original_hash,  # Link to previous version
        created_at=now,
        modified_at=modified_time,
        ingested_at=modified_time,
    )

    # Detect change
    has_changed_v2, prev_hash_v2 = await provenance_tracker.detect_change(
        source_path=source_path,
        content_hash=modified_hash
    )
    assert has_changed_v2 is True
    assert prev_hash_v2 == original_hash

    # Record second version
    await provenance_tracker.record_version(
        source_path=source_path,
        content_hash=modified_hash,
        parent_hash=original_hash,
        timestamp=modified_time,
        created_at=metadata_v2.created_at,
        modified_at=metadata_v2.modified_at
    )

    # Verify version history
    history = provenance_tracker.get_version_history(source_path)
    assert len(history) == 2
    assert history[0].content_hash == modified_hash  # Newest first
    assert history[0].parent_hash == original_hash
    assert history[1].content_hash == original_hash
    assert history[1].parent_hash is None


@pytest.mark.asyncio
async def test_version_chain_across_multiple_edits(provenance_tracker: ProvenanceTracker):
    """Test building complete version chain across multiple document edits."""
    base_time = datetime.now(timezone.utc)
    source_path = "/vault/active_project.md"

    # Simulate 10 edits over 10 days
    version_hashes = []

    for day in range(10):
        content = f"# Project Notes - Day {day + 1}\n\nProgress: {(day + 1) * 10}%"
        content_hash = compute_content_hash(content)
        version_hashes.append(content_hash)

        timestamp = base_time + timedelta(days=day)
        parent_hash = version_hashes[day - 1] if day > 0 else None

        # Detect change
        has_changed, prev_hash = await provenance_tracker.detect_change(
            source_path=source_path,
            content_hash=content_hash
        )

        if day == 0:
            assert has_changed is True
            assert prev_hash is None
        else:
            assert has_changed is True
            assert prev_hash == version_hashes[day - 1]

        # Record version
        await provenance_tracker.record_version(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=parent_hash,
            timestamp=timestamp
        )

    # Verify complete version chain
    history = provenance_tracker.get_version_history(source_path)
    assert len(history) == 10

    # Verify parent linkage forms complete chain
    for i in range(len(history) - 1):
        current = history[i]
        next_older = history[i + 1]

        # Current version's parent should be next older version's hash
        assert current.parent_hash == next_older.content_hash

    # Oldest version should have no parent
    assert history[-1].parent_hash is None


@pytest.mark.asyncio
async def test_idempotent_processing(provenance_tracker: ProvenanceTracker):
    """Test that reprocessing unchanged documents is idempotent."""
    now = datetime.now(timezone.utc)
    source_path = "/vault/stable_doc.md"

    content = "This document doesn't change."
    content_hash = compute_content_hash(content)

    # Process first time
    has_changed_1, prev_hash_1 = await provenance_tracker.detect_change(
        source_path=source_path,
        content_hash=content_hash
    )
    assert has_changed_1 is True

    await provenance_tracker.record_version(
        source_path=source_path,
        content_hash=content_hash,
        parent_hash=None,
        timestamp=now
    )

    # Process second time (unchanged)
    has_changed_2, prev_hash_2 = await provenance_tracker.detect_change(
        source_path=source_path,
        content_hash=content_hash
    )
    assert has_changed_2 is False
    assert prev_hash_2 is None

    # Process third time (unchanged)
    has_changed_3, prev_hash_3 = await provenance_tracker.detect_change(
        source_path=source_path,
        content_hash=content_hash
    )
    assert has_changed_3 is False

    # Version count should still be 1 (didn't re-record)
    count = provenance_tracker.get_version_count(source_path)
    assert count == 1


@pytest.mark.asyncio
async def test_concurrent_document_processing(provenance_tracker: ProvenanceTracker):
    """Test concurrent processing of multiple documents."""
    base_time = datetime.now(timezone.utc)

    async def process_document(doc_id: int):
        """Process a single document with version tracking."""
        source_path = f"/vault/concurrent_doc_{doc_id}.md"
        content = f"Document {doc_id} content"
        content_hash = compute_content_hash(content)

        # Detect change
        has_changed, prev_hash = await provenance_tracker.detect_change(
            source_path=source_path,
            content_hash=content_hash
        )

        # Record version
        await provenance_tracker.record_version(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=prev_hash,
            timestamp=base_time
        )

        return source_path

    # Process 100 documents concurrently
    tasks = [process_document(i) for i in range(100)]
    source_paths = await asyncio.gather(*tasks)

    # Verify all documents were tracked
    assert len(source_paths) == 100

    # Verify each has exactly 1 version
    for source_path in source_paths:
        count = provenance_tracker.get_version_count(source_path)
        assert count == 1


@pytest.mark.asyncio
async def test_metadata_temporal_tracking(provenance_tracker: ProvenanceTracker):
    """Test that temporal metadata is preserved through versioning."""
    created_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    modified_time = datetime(2025, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
    ingested_time = datetime(2025, 1, 1, 16, 0, 0, tzinfo=timezone.utc)

    content = "Document with temporal metadata"
    content_hash = compute_content_hash(content)

    # Record version with full temporal metadata
    version_record = await provenance_tracker.record_version(
        source_path="/vault/temporal_doc.md",
        content_hash=content_hash,
        parent_hash=None,
        timestamp=ingested_time,
        created_at=created_time,
        modified_at=modified_time
    )

    # Verify temporal metadata preserved
    assert version_record.created_at == created_time
    assert version_record.modified_at == modified_time
    assert version_record.ingested_at == ingested_time

    # Retrieve and verify persistence
    history = provenance_tracker.get_version_history("/vault/temporal_doc.md")
    assert len(history) == 1
    assert history[0].created_at == created_time
    assert history[0].modified_at == modified_time
    assert history[0].ingested_at == ingested_time


@pytest.mark.asyncio
async def test_multi_source_type_versioning(provenance_tracker: ProvenanceTracker):
    """Test versioning across different source types."""
    now = datetime.now(timezone.utc)

    # Obsidian note
    obsidian_content = "# Obsidian Note"
    obsidian_hash = compute_content_hash(obsidian_content)
    await provenance_tracker.record_version(
        source_path="/vault/obsidian_note.md",
        content_hash=obsidian_hash,
        parent_hash=None,
        timestamp=now
    )

    # Local file
    local_content = "Local file content"
    local_hash = compute_content_hash(local_content)
    await provenance_tracker.record_version(
        source_path="/documents/local_file.txt",
        content_hash=local_hash,
        parent_hash=None,
        timestamp=now
    )

    # Email
    email_content = "Email message body"
    email_hash = compute_content_hash(email_content)
    await provenance_tracker.record_version(
        source_path="imap://user@example.com/INBOX/123",
        content_hash=email_hash,
        parent_hash=None,
        timestamp=now
    )

    # GitHub file
    github_content = "# README\n\nProject documentation"
    github_hash = compute_content_hash(github_content)
    await provenance_tracker.record_version(
        source_path="github://owner/repo/main/README.md",
        content_hash=github_hash,
        parent_hash=None,
        timestamp=now
    )

    # Verify all source types tracked independently
    assert provenance_tracker.get_version_count("/vault/obsidian_note.md") == 1
    assert provenance_tracker.get_version_count("/documents/local_file.txt") == 1
    assert provenance_tracker.get_version_count("imap://user@example.com/INBOX/123") == 1
    assert provenance_tracker.get_version_count("github://owner/repo/main/README.md") == 1


# ---------------------------------------------------------------------------
# Real-World Scenario Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_journal_versioning(provenance_tracker: ProvenanceTracker):
    """Simulate daily journal entry versioning over a month."""
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # Create 30 daily journal entries
    for day in range(1, 31):
        source_path = f"/vault/daily/{2025:04d}-01-{day:02d}.md"
        content = f"# Daily Journal - January {day}, 2025\n\n## Morning\n\nToday's plans...\n"
        content_hash = compute_content_hash(content)
        timestamp = base_date + timedelta(days=day - 1)

        await provenance_tracker.record_version(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=None,
            timestamp=timestamp,
            created_at=timestamp,
            modified_at=timestamp
        )

        # Some entries are edited later in the day
        if day % 3 == 0:  # Edit every 3rd entry
            updated_content = content + "\n## Evening\n\nReflections..."
            updated_hash = compute_content_hash(updated_content)
            updated_timestamp = timestamp + timedelta(hours=12)

            await provenance_tracker.record_version(
                source_path=source_path,
                content_hash=updated_hash,
                parent_hash=content_hash,
                timestamp=updated_timestamp,
                created_at=timestamp,
                modified_at=updated_timestamp
            )

    # Verify: 30 journals created, 10 of them have 2 versions (edited)
    # Total: 30 with 1 version, 10 with 2 versions = 40 total versions

    edited_count = 0
    for day in range(1, 31):
        source_path = f"/vault/daily/{2025:04d}-01-{day:02d}.md"
        count = provenance_tracker.get_version_count(source_path)

        if day % 3 == 0:
            assert count == 2  # Was edited
            edited_count += 1
        else:
            assert count == 1  # Not edited

    assert edited_count == 10


@pytest.mark.asyncio
async def test_project_document_evolution(provenance_tracker: ProvenanceTracker):
    """Simulate realistic project document evolution."""
    project_doc = "/vault/projects/futurnal/architecture.md"
    base_time = datetime.now(timezone.utc)

    # Version 1: Initial draft
    v1_content = "# Futurnal Architecture\n\n## Overview\n\nInitial thoughts..."
    v1_hash = compute_content_hash(v1_content)
    await provenance_tracker.record_version(
        source_path=project_doc,
        content_hash=v1_hash,
        parent_hash=None,
        timestamp=base_time
    )

    # Version 2: Add PKG section (next day)
    v2_content = v1_content + "\n\n## PKG Layer\n\nPersonal Knowledge Graph design..."
    v2_hash = compute_content_hash(v2_content)
    await provenance_tracker.record_version(
        source_path=project_doc,
        content_hash=v2_hash,
        parent_hash=v1_hash,
        timestamp=base_time + timedelta(days=1)
    )

    # Version 3: Add versioning section (week later)
    v3_content = v2_content + "\n\n## Versioning\n\nDocument provenance tracking..."
    v3_hash = compute_content_hash(v3_content)
    await provenance_tracker.record_version(
        source_path=project_doc,
        content_hash=v3_hash,
        parent_hash=v2_hash,
        timestamp=base_time + timedelta(weeks=1)
    )

    # Version 4: Major refactor (month later)
    v4_content = "# Futurnal System Architecture\n\n[Completely restructured content]"
    v4_hash = compute_content_hash(v4_content)
    await provenance_tracker.record_version(
        source_path=project_doc,
        content_hash=v4_hash,
        parent_hash=v3_hash,
        timestamp=base_time + timedelta(weeks=4)
    )

    # Verify evolution chain
    history = provenance_tracker.get_version_history(project_doc)
    assert len(history) == 4

    # Verify parent chain
    assert history[0].content_hash == v4_hash
    assert history[0].parent_hash == v3_hash
    assert history[1].content_hash == v3_hash
    assert history[1].parent_hash == v2_hash
    assert history[2].content_hash == v2_hash
    assert history[2].parent_hash == v1_hash
    assert history[3].content_hash == v1_hash
    assert history[3].parent_hash is None


@pytest.mark.asyncio
async def test_rapid_edit_sequence(provenance_tracker: ProvenanceTracker):
    """Test rapid sequence of edits (e.g., active writing session)."""
    source_path = "/vault/drafts/blog_post.md"
    base_time = datetime.now(timezone.utc)

    # Simulate 50 rapid edits over 2 hours (editing every ~2.4 minutes)
    prev_hash = None

    for edit_num in range(50):
        content = f"# Blog Post Draft\n\nParagraphs: {edit_num + 1}\n\n{'Lorem ipsum. ' * (edit_num + 1)}"
        content_hash = compute_content_hash(content)
        timestamp = base_time + timedelta(minutes=edit_num * 2.4)

        await provenance_tracker.record_version(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=prev_hash,
            timestamp=timestamp
        )

        prev_hash = content_hash

    # Verify all 50 versions tracked
    count = provenance_tracker.get_version_count(source_path)
    assert count == 50

    # Verify complete chain
    history = provenance_tracker.get_version_history(source_path)
    assert len(history) == 50

    # Verify each version links to previous (except first)
    for i in range(len(history) - 1):
        assert history[i].parent_hash == history[i + 1].content_hash

    assert history[-1].parent_hash is None  # First version has no parent


# ---------------------------------------------------------------------------
# Performance Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.asyncio
async def test_large_scale_document_versioning(provenance_tracker: ProvenanceTracker):
    """Test versioning performance with large document collection."""
    import time

    base_time = datetime.now(timezone.utc)
    start = time.perf_counter()

    # Process 1000 documents
    for doc_id in range(1000):
        source_path = f"/vault/collection/doc_{doc_id:04d}.md"
        content = f"Document {doc_id} content with some text."
        content_hash = compute_content_hash(content)

        await provenance_tracker.record_version(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=None,
            timestamp=base_time
        )

    elapsed = time.perf_counter() - start
    throughput = 1000 / elapsed

    # Should handle at least 100 documents/second
    assert throughput > 100, f"Large scale versioning too slow: {throughput:.2f} docs/s"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_version_history_retrieval_performance(provenance_tracker: ProvenanceTracker):
    """Test performance of retrieving version history."""
    import time

    base_time = datetime.now(timezone.utc)
    source_path = "/vault/large_history_doc.md"

    # Create document with 500 versions
    prev_hash = None
    for version in range(500):
        content = f"Version {version} content"
        content_hash = compute_content_hash(content)

        await provenance_tracker.record_version(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=prev_hash,
            timestamp=base_time + timedelta(minutes=version)
        )

        prev_hash = content_hash

    # Benchmark history retrieval
    start = time.perf_counter()
    history = provenance_tracker.get_version_history(source_path)
    elapsed = time.perf_counter() - start

    assert len(history) == 500
    # Should retrieve in under 50ms
    assert elapsed < 0.05, f"History retrieval too slow: {elapsed*1000:.2f}ms"
