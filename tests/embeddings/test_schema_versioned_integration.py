"""Integration tests for Schema-Versioned Storage module.

Tests end-to-end scenarios with real ChromaDB (ephemeral)
and mocked PKG components.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/03-schema-versioned-storage.md
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Check if sentence_transformers is available (required for real ChromaDB)
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Mark all tests in this module as integration tests
# Skip if sentence_transformers not installed
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not HAS_SENTENCE_TRANSFORMERS,
        reason="sentence_transformers not installed - required for integration tests"
    ),
]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for ChromaDB persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_schema_manager():
    """Create a mock SchemaVersionManager."""
    manager = MagicMock()

    # Schema version node
    version_node = MagicMock()
    version_node.version = 1
    version_node.entity_types = ["Person", "Organization", "Event", "Concept"]
    version_node.relationship_types = ["RELATED_TO", "WORKS_AT", "CAUSES"]

    manager.get_current_version.return_value = version_node
    manager.get_version.return_value = version_node

    return manager


@pytest.fixture
def schema_versioned_store(temp_persist_dir, mock_schema_manager):
    """Create SchemaVersionedEmbeddingStore with real ChromaDB."""
    # Patch SchemaVersionManager at the source module
    with patch(
        "futurnal.pkg.schema.migration.SchemaVersionManager"
    ) as mock_manager_class:
        mock_manager_class.return_value = mock_schema_manager

        from futurnal.embeddings.config import EmbeddingServiceConfig
        from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore

        config = EmbeddingServiceConfig()

        # Create mock driver to enable schema version tracking
        mock_driver = MagicMock()

        store = SchemaVersionedEmbeddingStore(
            config=config,
            neo4j_driver=mock_driver,
            persist_directory=temp_persist_dir,
        )

        yield store

        store.close()


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestSchemaVersionedStoreIntegration:
    """Integration tests for SchemaVersionedEmbeddingStore with real ChromaDB."""

    def test_store_and_retrieve_embedding(self, schema_versioned_store):
        """Test full store and query cycle."""
        store = schema_versioned_store

        # Generate test embedding
        embedding = np.random.rand(768).tolist()

        # Store embedding
        embedding_id = store.store_embedding(
            entity_id="person_test_001",
            entity_type="Person",
            embedding=embedding,
            model_id="instructor-large",
            extraction_confidence=0.95,
            source_document_id="test_doc_001",
        )

        assert embedding_id is not None
        assert len(embedding_id) == 36  # UUID format

        # Query embeddings
        results = store.query_embeddings(
            query_vector=embedding,
            top_k=10,
        )

        assert len(results) >= 1
        assert results[0].entity_id == "person_test_001"
        assert results[0].similarity_score > 0.99  # Same vector should be ~1.0

    def test_schema_version_in_stored_metadata(self, schema_versioned_store):
        """Verify schema version is correctly stored in metadata."""
        store = schema_versioned_store

        embedding = np.random.rand(768).tolist()

        store.store_embedding(
            entity_id="person_test_002",
            entity_type="Person",
            embedding=embedding,
            model_id="instructor-large",
            extraction_confidence=0.9,
            source_document_id="test_doc_002",
        )

        results = store.query_embeddings(
            query_vector=embedding,
            top_k=1,
        )

        assert len(results) == 1
        assert results[0].metadata.get("schema_version") == 1
        assert "schema_hash" in results[0].metadata

    def test_store_event_with_temporal_context(self, schema_versioned_store):
        """Test storing events with temporal context."""
        store = schema_versioned_store

        from futurnal.embeddings.models import TemporalEmbeddingContext

        embedding = np.random.rand(768).tolist()
        temporal_context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
            duration=timedelta(hours=2),
            temporal_type="DURING",
        )

        embedding_id = store.store_embedding(
            entity_id="event_test_001",
            entity_type="Event",
            embedding=embedding,
            model_id="instructor-temporal",
            extraction_confidence=0.88,
            source_document_id="test_doc_003",
            temporal_context=temporal_context,
        )

        assert embedding_id is not None

        # Query from events collection
        results = store.query_embeddings(
            query_vector=embedding,
            top_k=1,
            entity_type="Event",
        )

        assert len(results) == 1
        assert results[0].metadata.get("timestamp") == "2024-01-15T14:30:00"
        assert results[0].metadata.get("duration_seconds") == 7200.0

    def test_mark_and_retrieve_for_reembedding(self, schema_versioned_store):
        """Test marking embeddings for re-embedding and retrieving them."""
        store = schema_versioned_store

        # Store multiple embeddings
        for i in range(3):
            embedding = np.random.rand(768).tolist()
            store.store_embedding(
                entity_id=f"person_mark_test_{i}",
                entity_type="Person",
                embedding=embedding,
                model_id="instructor-large",
                extraction_confidence=0.9,
                source_document_id=f"doc_mark_{i}",
            )

        # Mark first two for re-embedding
        marked = store.mark_for_reembedding(
            entity_ids=["person_mark_test_0", "person_mark_test_1"],
            reason="quality",
        )

        assert marked >= 2

        # Retrieve embeddings needing re-embedding
        pending = store.get_embeddings_needing_reembedding(limit=10)

        assert len(pending) >= 2

        entity_ids = [p["entity_id"] for p in pending]
        assert "person_mark_test_0" in entity_ids
        assert "person_mark_test_1" in entity_ids

    def test_query_with_schema_version_filter(self, schema_versioned_store):
        """Test querying with schema version filter."""
        store = schema_versioned_store

        embedding = np.random.rand(768).tolist()

        store.store_embedding(
            entity_id="person_version_test",
            entity_type="Person",
            embedding=embedding,
            model_id="instructor-large",
            extraction_confidence=0.9,
            source_document_id="doc_version",
        )

        # Query with minimum schema version
        results = store.query_embeddings(
            query_vector=embedding,
            top_k=10,
            min_schema_version=1,  # Current version
        )

        assert len(results) >= 1

        # Query with higher minimum version should return empty
        results_empty = store.query_embeddings(
            query_vector=embedding,
            top_k=10,
            min_schema_version=999,  # Much higher than current
        )

        assert len(results_empty) == 0

    def test_get_embedding_count(self, schema_versioned_store):
        """Test embedding count tracking."""
        store = schema_versioned_store

        initial_count = store.get_embedding_count()

        # Store some embeddings
        for i in range(5):
            embedding = np.random.rand(768).tolist()
            store.store_embedding(
                entity_id=f"count_test_{i}",
                entity_type="Person",
                embedding=embedding,
                model_id="instructor-large",
                extraction_confidence=0.9,
                source_document_id=f"doc_count_{i}",
            )

        new_count = store.get_embedding_count()

        assert new_count["entities"] >= initial_count["entities"] + 5
        assert new_count["total"] >= initial_count["total"] + 5

    def test_delete_embedding(self, schema_versioned_store):
        """Test embedding deletion."""
        store = schema_versioned_store

        embedding = np.random.rand(768).tolist()

        embedding_id = store.store_embedding(
            entity_id="delete_test",
            entity_type="Person",
            embedding=embedding,
            model_id="instructor-large",
            extraction_confidence=0.9,
            source_document_id="doc_delete",
        )

        # Verify it exists
        results = store.query_embeddings(query_vector=embedding, top_k=1)
        assert len(results) >= 1

        # Delete
        deleted = store.delete_embedding(embedding_id)
        assert deleted is True

    def test_clear_reembedding_flag(self, schema_versioned_store):
        """Test clearing re-embedding flag after processing."""
        store = schema_versioned_store

        embedding = np.random.rand(768).tolist()

        embedding_id = store.store_embedding(
            entity_id="clear_flag_test",
            entity_type="Person",
            embedding=embedding,
            model_id="instructor-large",
            extraction_confidence=0.9,
            source_document_id="doc_clear",
        )

        # Mark for re-embedding
        store.mark_for_reembedding(entity_ids=["clear_flag_test"], reason="test")

        # Verify it's marked
        pending = store.get_embeddings_needing_reembedding()
        assert any(p["entity_id"] == "clear_flag_test" for p in pending)

        # Clear flag
        cleared = store.clear_reembedding_flag([embedding_id])
        assert cleared >= 1

        # Verify flag is cleared
        pending_after = store.get_embeddings_needing_reembedding()
        assert not any(p["entity_id"] == "clear_flag_test" for p in pending_after)


class TestSchemaEvolutionScenario:
    """Integration tests for schema evolution scenarios."""

    def test_schema_version_refresh(self, schema_versioned_store, mock_schema_manager):
        """Test schema cache refresh after version change."""
        store = schema_versioned_store

        # Initial version
        assert store.current_schema_version == 1

        # Simulate schema evolution
        new_version_node = MagicMock()
        new_version_node.version = 2
        new_version_node.entity_types = ["Person", "Organization", "Event", "Concept", "Project"]
        new_version_node.relationship_types = ["RELATED_TO", "WORKS_AT", "CAUSES", "ASSIGNED_TO"]

        mock_schema_manager.get_current_version.return_value = new_version_node

        # Refresh cache
        store.refresh_schema_cache()

        # Should now be version 2
        assert store.current_schema_version == 2

    def test_schema_hash_changes_with_version(
        self,
        schema_versioned_store,
        mock_schema_manager,
    ):
        """Test that schema hash changes when schema evolves."""
        store = schema_versioned_store

        # Get initial hash
        initial_hash = store.current_schema_hash

        # Simulate schema evolution
        new_version_node = MagicMock()
        new_version_node.version = 2
        new_version_node.entity_types = ["Person", "Organization", "Event", "Concept", "Project"]
        new_version_node.relationship_types = ["RELATED_TO", "WORKS_AT", "CAUSES"]

        mock_schema_manager.get_current_version.return_value = new_version_node

        # Refresh and get new hash
        store.refresh_schema_cache()
        new_hash = store.current_schema_hash

        # Hashes should be different
        assert initial_hash != new_hash

    def test_embeddings_from_different_versions(
        self,
        schema_versioned_store,
        mock_schema_manager,
    ):
        """Test storing embeddings from different schema versions."""
        store = schema_versioned_store

        # Store embedding at version 1
        embedding_v1 = np.random.rand(768).tolist()
        store.store_embedding(
            entity_id="version_test_v1",
            entity_type="Person",
            embedding=embedding_v1,
            model_id="instructor-large",
            extraction_confidence=0.9,
            source_document_id="doc_v1",
        )

        # Simulate schema evolution to version 2
        new_version_node = MagicMock()
        new_version_node.version = 2
        new_version_node.entity_types = ["Person", "Organization", "Event", "Concept", "Project"]
        new_version_node.relationship_types = ["RELATED_TO", "WORKS_AT", "CAUSES"]

        mock_schema_manager.get_current_version.return_value = new_version_node
        store.refresh_schema_cache()

        # Store embedding at version 2
        embedding_v2 = np.random.rand(768).tolist()
        store.store_embedding(
            entity_id="version_test_v2",
            entity_type="Person",
            embedding=embedding_v2,
            model_id="instructor-large",
            extraction_confidence=0.9,
            source_document_id="doc_v2",
        )

        # Query all - should get both
        results_all = store.query_embeddings(
            query_vector=embedding_v1,
            top_k=10,
        )

        versions_found = {r.metadata.get("schema_version") for r in results_all}
        assert 1 in versions_found
        assert 2 in versions_found

        # Query with min_schema_version=2 - should only get v2
        results_v2_only = store.query_embeddings(
            query_vector=embedding_v1,
            top_k=10,
            min_schema_version=2,
        )

        for result in results_v2_only:
            assert result.metadata.get("schema_version") >= 2


class TestReembeddingServiceIntegration:
    """Integration tests for ReembeddingService."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=None)
        return driver, session

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock MultiModelEmbeddingService."""
        service = MagicMock()

        result = MagicMock()
        result.embedding = np.random.rand(768).tolist()
        result.model_version = "mock:instructor-large"

        service.embed.return_value = result
        return service

    def test_full_reembedding_workflow(
        self,
        schema_versioned_store,
        mock_neo4j_driver,
        mock_embedding_service,
        mock_schema_manager,
    ):
        """Test complete re-embedding workflow."""
        store = schema_versioned_store
        driver, session = mock_neo4j_driver

        # Store initial embeddings
        for i in range(3):
            embedding = np.random.rand(768).tolist()
            store.store_embedding(
                entity_id=f"workflow_test_{i}",
                entity_type="Person",
                embedding=embedding,
                model_id="instructor-large",
                extraction_confidence=0.9,
                source_document_id=f"doc_workflow_{i}",
            )

        # Setup mock schema versions for change detection
        v1 = MagicMock()
        v1.version = 1
        v1.entity_types = ["Person", "Organization", "Event"]
        v1.relationship_types = ["RELATED_TO"]

        v2 = MagicMock()
        v2.version = 2
        v2.entity_types = ["Person", "Organization", "Event", "Project"]
        v2.relationship_types = ["RELATED_TO", "ASSIGNED_TO"]

        def get_version(version):
            return v1 if version == 1 else v2 if version == 2 else None

        mock_schema_manager.get_version = get_version

        # Create ReembeddingService
        # Note: ReembeddingService already receives mock_schema_manager via the store fixture
        from futurnal.embeddings.reembedding import ReembeddingService

        service = ReembeddingService(
            store=store,
            embedding_service=mock_embedding_service,
            neo4j_driver=driver,
        )

        # Detect schema changes
        changes = service.detect_schema_changes(old_version=1, new_version=2)

        assert changes.requires_reembedding is True
        assert "Project" in changes.new_entity_types
        assert "ASSIGNED_TO" in changes.new_relationship_types

        # Get initial stats
        initial_stats = service.get_reembedding_stats()

        # Mark embeddings from v1 for re-embedding
        store.mark_for_reembedding(schema_version=1, reason="schema_evolution")

        # Verify they're marked
        new_stats = service.get_reembedding_stats()
        assert new_stats["total_pending"] > initial_stats["total_pending"]
