"""Tests for paper state store.

Tests cover:
- Paper record creation and retrieval
- Download state tracking
- Ingestion state tracking
- Getting unprocessed papers
- Database persistence
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from futurnal.agents.paper_search.state_store import (
    PaperRecord,
    PaperStateStore,
    get_default_state_store,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "papers_state.db"


@pytest.fixture
def state_store(temp_db_path):
    """Create a PaperStateStore with a temporary database."""
    store = PaperStateStore(temp_db_path)
    yield store
    store.close()


def test_create_paper_record():
    """Test creation of a PaperRecord."""
    record = PaperRecord(
        paper_id="test123",
        title="Test Paper",
    )

    assert record.paper_id == "test123"
    assert record.title == "Test Paper"
    assert record.local_path is None
    assert record.download_status == "pending"
    assert record.ingestion_status == "pending"
    assert record.downloaded_at is None
    assert record.ingested_at is None


def test_state_store_mark_downloaded(state_store):
    """Test marking a paper as downloaded."""
    state_store.mark_downloaded(
        paper_id="paper1",
        title="Test Paper 1",
        local_path="/path/to/paper.pdf",
        file_size_bytes=1024,
    )

    record = state_store.get("paper1")

    assert record is not None
    assert record.paper_id == "paper1"
    assert record.title == "Test Paper 1"
    assert record.local_path == "/path/to/paper.pdf"
    assert record.file_size_bytes == 1024
    assert record.download_status == "downloaded"
    assert record.ingestion_status == "pending"
    assert record.downloaded_at is not None


def test_state_store_mark_ingestion_queued(state_store):
    """Test marking a paper for ingestion."""
    # First mark as downloaded
    state_store.mark_downloaded(
        paper_id="paper1",
        title="Test Paper 1",
        local_path="/path/to/paper.pdf",
        file_size_bytes=1024,
    )

    # Then mark ingestion queued
    state_store.mark_ingestion_queued("paper1")

    record = state_store.get("paper1")

    assert record is not None
    assert record.download_status == "downloaded"
    assert record.ingestion_status == "queued"


def test_state_store_mark_ingestion_completed(state_store):
    """Test marking a paper ingestion as completed."""
    # First mark as downloaded
    state_store.mark_downloaded(
        paper_id="paper1",
        title="Test Paper 1",
        local_path="/path/to/paper.pdf",
        file_size_bytes=1024,
    )

    # Then mark ingestion queued and completed
    state_store.mark_ingestion_queued("paper1")
    state_store.mark_ingestion_completed("paper1")

    record = state_store.get("paper1")

    assert record is not None
    assert record.ingestion_status == "completed"
    assert record.ingested_at is not None


def test_state_store_mark_ingestion_failed(state_store):
    """Test marking a paper ingestion as failed."""
    # First mark as downloaded
    state_store.mark_downloaded(
        paper_id="paper1",
        title="Test Paper 1",
        local_path="/path/to/paper.pdf",
        file_size_bytes=1024,
    )

    # Then mark ingestion failed
    state_store.mark_ingestion_failed("paper1", error="Processing error")

    record = state_store.get("paper1")

    assert record is not None
    assert record.ingestion_status == "failed"


def test_state_store_get_unprocessed_papers(state_store):
    """Test getting list of unprocessed papers."""
    # Add some papers in various states
    state_store.mark_downloaded(
        paper_id="paper1",
        title="Paper 1",
        local_path="/path/to/paper1.pdf",
        file_size_bytes=1024,
    )

    state_store.mark_downloaded(
        paper_id="paper2",
        title="Paper 2",
        local_path="/path/to/paper2.pdf",
        file_size_bytes=2048,
    )

    state_store.mark_downloaded(
        paper_id="paper3",
        title="Paper 3",
        local_path="/path/to/paper3.pdf",
        file_size_bytes=3072,
    )

    # Mark paper2 as ingested
    state_store.mark_ingestion_queued("paper2")
    state_store.mark_ingestion_completed("paper2")

    # Get unprocessed papers
    unprocessed = state_store.get_unprocessed_papers()

    assert len(unprocessed) == 2
    paper_ids = {p.paper_id for p in unprocessed}
    assert "paper1" in paper_ids
    assert "paper3" in paper_ids
    assert "paper2" not in paper_ids


def test_state_store_get_nonexistent_paper(state_store):
    """Test getting status of non-existent paper returns None."""
    record = state_store.get("nonexistent")
    assert record is None


def test_state_store_persistence(temp_db_path):
    """Test that state persists across store instances."""
    # Create store and add a paper
    store1 = PaperStateStore(temp_db_path)
    store1.mark_downloaded(
        paper_id="persistent",
        title="Persistent Paper",
        local_path="/path/to/paper.pdf",
        file_size_bytes=1024,
    )
    store1.close()

    # Create new store instance and verify paper exists
    store2 = PaperStateStore(temp_db_path)
    record = store2.get("persistent")
    store2.close()

    assert record is not None
    assert record.paper_id == "persistent"
    assert record.title == "Persistent Paper"


def test_state_store_update_existing_paper(state_store):
    """Test updating an existing paper record (re-download preserves original title)."""
    # Add initial paper
    state_store.mark_downloaded(
        paper_id="paper1",
        title="Original Title",
        local_path="/original/path.pdf",
        file_size_bytes=1024,
    )

    # Update with new download (simulating re-download)
    # Note: mark_downloaded preserves the original title when updating
    state_store.mark_downloaded(
        paper_id="paper1",
        title="Updated Title",  # This should be ignored for existing records
        local_path="/new/path.pdf",
        file_size_bytes=2048,
    )

    record = state_store.get("paper1")

    # Title is preserved from original download
    assert record.title == "Original Title"
    # But path and size are updated
    assert record.local_path == "/new/path.pdf"
    assert record.file_size_bytes == 2048


def test_state_store_multiple_papers(state_store):
    """Test handling multiple papers."""
    # Add multiple papers
    for i in range(10):
        state_store.mark_downloaded(
            paper_id=f"paper{i}",
            title=f"Paper {i}",
            local_path=f"/path/to/paper{i}.pdf",
            file_size_bytes=1024 * (i + 1),
        )

    # Verify all papers exist
    for i in range(10):
        record = state_store.get(f"paper{i}")
        assert record is not None
        assert record.title == f"Paper {i}"


def test_state_store_workflow(state_store):
    """Test complete paper workflow: download -> queue -> complete."""
    paper_id = "workflow_test"

    # Initial state - paper doesn't exist
    assert state_store.get(paper_id) is None

    # Step 1: Download
    state_store.mark_downloaded(
        paper_id=paper_id,
        title="Workflow Test Paper",
        local_path="/path/to/paper.pdf",
        file_size_bytes=1024,
    )
    record = state_store.get(paper_id)
    assert record.download_status == "downloaded"
    assert record.ingestion_status == "pending"

    # Step 2: Queue for ingestion
    state_store.mark_ingestion_queued(paper_id)
    record = state_store.get(paper_id)
    assert record.ingestion_status == "queued"

    # Step 3: Complete ingestion
    state_store.mark_ingestion_completed(paper_id)
    record = state_store.get(paper_id)
    assert record.ingestion_status == "completed"
    assert record.ingested_at is not None


def test_get_default_state_store():
    """Test getting default state store creates appropriate instance."""
    store = get_default_state_store()
    assert store is not None
    assert isinstance(store, PaperStateStore)
    store.close()
