"""Integration Tests for PKG ↔ Vector Store Sync (Module 05).

Tests synchronization between PKG (Neo4j) and Vector Store (ChromaDB).

From production plan:
- test_entity_creation_triggers_sync: Create entity → verify sync event
- test_sync_uses_same_identifier: SHA-256 consistency check
- test_deletion_syncs_both_stores: Delete → removed from both

Success Metrics:
- All sync events emitted correctly
- SHA-256 identifiers consistent across stores
- Deletion propagates to both stores

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from tests.pkg.conftest import (
    requires_neo4j,
    requires_testcontainers,
    requires_docker,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def generate_sha256(content: str) -> str:
    """Generate SHA256 hash for content."""
    return hashlib.sha256(content.encode()).hexdigest()


def create_test_document(content: str, path: str = "/test/doc.md") -> dict:
    """Create a test document payload."""
    sha256 = generate_sha256(content)
    return {
        "sha256": sha256,
        "path": path,
        "source": "obsidian_vault",
        "metadata": {
            "filetype": "markdown",
            "created_at": datetime.utcnow().isoformat(),
        },
        "text": content,
    }


# ---------------------------------------------------------------------------
# Integration Tests: Vector Store Sync
# ---------------------------------------------------------------------------


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestVectorStoreSync:
    """Tests for PKG ↔ Vector store synchronization.

    From production plan 05-integration-testing.md:
    Validates that PKG updates trigger proper vector store sync.
    """

    def test_entity_creation_triggers_sync(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Create entity → verify vector upsert called.

        From production plan:
        - Create entity via NormalizationSink
        - Capture sync events
        - Verify SyncEvent emitted with correct entity_id
        """
        document = create_test_document(
            content="Test document for sync verification",
            path="/vault/notes/sync-test.md",
        )

        # Process document through sink
        normalization_sink.handle(document)

        # Verify sync events emitted
        assert sync_event_capture.count >= 2, "Should emit at least 2 sync events"

        # Get events for this document
        events = sync_event_capture.get_events_for_entity(document["sha256"])
        assert len(events) >= 2, "Should have both PKG and Vector events"

        # Verify PKG write event
        pkg_events = [e for e in events if "pkg" in str(e.source_operation).lower()]
        assert len(pkg_events) >= 1, "Should have PKG write event"
        assert pkg_events[0].entity_id == document["sha256"]
        assert pkg_events[0].entity_type == "Document"

        # Verify Vector write event
        vector_events = [e for e in events if "vector" in str(e.source_operation).lower()]
        assert len(vector_events) >= 1, "Should have Vector write event"
        assert vector_events[0].entity_id == document["sha256"]

    def test_sync_uses_same_identifier(
        self,
        normalization_sink,
        mock_pkg_writer,
        mock_vector_writer,
    ):
        """SHA-256 consistency between PKG and Vector.

        From production plan:
        - Store document via sink
        - Query PKG by sha256
        - Query Vector by same sha256
        - Verify both found with matching data
        """
        document = create_test_document(
            content="Consistent identifier test document",
            path="/vault/notes/identifier-test.md",
        )
        sha256 = document["sha256"]

        # Process through sink
        normalization_sink.handle(document)

        # Verify PKG has the document (via mock writer's internal storage)
        assert sha256 in mock_pkg_writer.documents
        pkg_doc = mock_pkg_writer.documents[sha256]
        assert pkg_doc["path"] == document["path"]

        # Verify Vector has the document
        assert sha256 in mock_vector_writer.embeddings
        vector_doc = mock_vector_writer.embeddings[sha256]
        assert vector_doc["sha256"] == sha256
        assert vector_doc["path"] == document["path"]

        # Verify identifiers match
        assert pkg_doc["sha256"] == vector_doc["sha256"]

    def test_deletion_syncs_both_stores(
        self,
        normalization_sink,
        sync_event_capture,
        mock_pkg_writer,
        mock_vector_writer,
    ):
        """Delete → removed from both PKG and Vector.

        From production plan:
        - Create document in both stores
        - Call handle_deletion()
        - Verify removed from PKG
        - Verify removed from Vector
        """
        document = create_test_document(
            content="Document to be deleted",
            path="/vault/notes/delete-test.md",
        )
        sha256 = document["sha256"]

        # First, create the document
        normalization_sink.handle(document)

        # Verify it exists in both stores
        assert sha256 in mock_pkg_writer.documents
        assert sha256 in mock_vector_writer.embeddings

        # Clear capture for deletion events
        sync_event_capture.clear()

        # Delete the document
        normalization_sink.handle_deletion({"sha256": sha256})

        # Verify removed from PKG
        assert sha256 not in mock_pkg_writer.documents

        # Verify removed from Vector
        assert sha256 not in mock_vector_writer.embeddings

        # Verify deletion sync events emitted
        events = sync_event_capture.get_events_for_entity(sha256)
        assert len(events) >= 2, "Should emit deletion events for both stores"

        # Verify event types
        delete_events = [
            e for e in events
            if "deleted" in str(e.event_type).lower()
        ]
        assert len(delete_events) >= 2, "Should have deletion events"


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestSyncEventCapture:
    """Tests for SyncEventCapture utility functionality."""

    def test_sync_event_capture_records_all_events(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Verify SyncEventCapture records all events."""
        # Process multiple documents
        for i in range(3):
            doc = create_test_document(
                content=f"Test document {i}",
                path=f"/vault/notes/test-{i}.md",
            )
            normalization_sink.handle(doc)

        # Should have 6 events (2 per document: pkg_write + vector_write)
        assert sync_event_capture.count >= 6

        # Verify unique entities tracked
        unique_entities = {e.entity_id for e in sync_event_capture.events}
        assert len(unique_entities) == 3

    def test_sync_event_capture_get_by_type(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Verify filtering by event type works."""
        doc = create_test_document(
            content="Filter test document",
            path="/vault/notes/filter-test.md",
        )

        normalization_sink.handle(doc)

        # Filter by type
        created_events = sync_event_capture.get_events_by_type("entity_created")
        assert len(created_events) >= 2

    def test_sync_event_capture_statistics(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Verify statistics calculation works."""
        doc = create_test_document(
            content="Stats test document",
            path="/vault/notes/stats-test.md",
        )

        normalization_sink.handle(doc)

        stats = sync_event_capture.get_statistics()
        assert stats["total_events"] >= 2
        assert stats["unique_entities"] >= 1


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestSyncEventIntegration:
    """Tests for sync event integration with real database."""

    def test_pkg_write_creates_node_in_neo4j(
        self,
        neo4j_driver,
        mock_pkg_writer,
    ):
        """Verify PKG write actually creates node in Neo4j."""
        document = create_test_document(
            content="Real Neo4j test",
            path="/vault/notes/neo4j-test.md",
        )

        # Write through mock (which uses real Neo4j)
        mock_pkg_writer.write_document(document)

        # Verify node exists in Neo4j
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (d:Document {sha256: $sha256}) RETURN d",
                {"sha256": document["sha256"]},
            )
            record = result.single()
            assert record is not None
            assert record["d"]["path"] == document["path"]

    def test_sync_completion_status_tracking(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Verify sync status progresses from pending to completed."""
        document = create_test_document(
            content="Status tracking test",
            path="/vault/notes/status-test.md",
        )

        normalization_sink.handle(document)

        events = sync_event_capture.get_events_for_entity(document["sha256"])

        # First event should be PKG write (pending vector sync)
        pkg_event = next(
            (e for e in events if "pkg" in str(e.source_operation).lower()),
            None,
        )
        assert pkg_event is not None
        assert str(pkg_event.vector_sync_status) == "pending"

        # Second event should be Vector write (completed)
        vector_event = next(
            (e for e in events if "vector" in str(e.source_operation).lower()),
            None,
        )
        assert vector_event is not None
        assert str(vector_event.vector_sync_status) == "completed"

    def test_batch_document_processing_sync(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Verify batch document processing emits correct sync events."""
        documents = [
            create_test_document(
                content=f"Batch document {i}",
                path=f"/vault/notes/batch-{i}.md",
            )
            for i in range(5)
        ]

        # Process all documents
        for doc in documents:
            normalization_sink.handle(doc)

        # Verify events for all documents
        assert sync_event_capture.count >= 10  # 2 events per document

        # Verify all documents have completed sync
        for doc in documents:
            events = sync_event_capture.get_events_for_entity(doc["sha256"])
            assert len(events) >= 2

            # Last event should be completed
            latest = events[-1]
            assert str(latest.vector_sync_status) == "completed"


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestSyncEdgeCases:
    """Tests for edge cases in sync functionality."""

    def test_empty_document_sync(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Verify empty documents sync correctly."""
        document = create_test_document(
            content="",
            path="/vault/notes/empty.md",
        )

        normalization_sink.handle(document)

        events = sync_event_capture.get_events_for_entity(document["sha256"])
        assert len(events) >= 2

    def test_large_document_sync(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Verify large documents sync correctly."""
        # Create 100KB document
        large_content = "A" * 100000
        document = create_test_document(
            content=large_content,
            path="/vault/notes/large.md",
        )

        normalization_sink.handle(document)

        events = sync_event_capture.get_events_for_entity(document["sha256"])
        assert len(events) >= 2
        assert str(events[-1].vector_sync_status) == "completed"

    def test_special_characters_in_path_sync(
        self,
        normalization_sink,
        sync_event_capture,
    ):
        """Verify documents with special characters in path sync correctly."""
        document = create_test_document(
            content="Special path test",
            path="/vault/notes/special chars & symbols (test).md",
        )

        normalization_sink.handle(document)

        events = sync_event_capture.get_events_for_entity(document["sha256"])
        assert len(events) >= 2
