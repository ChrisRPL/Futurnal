"""Tests for SchemaVersionedEmbeddingStore and EmbeddingMetadata.

Tests schema version tracking, ChromaDB metadata conversion,
and re-embedding management functionality.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/03-schema-versioned-storage.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.models import EmbeddingMetadata, TemporalEmbeddingContext


# -----------------------------------------------------------------------------
# EmbeddingMetadata Tests
# -----------------------------------------------------------------------------


class TestEmbeddingMetadata:
    """Tests for the EmbeddingMetadata model."""

    def test_required_fields_validation(self):
        """Validate required fields are enforced."""
        with pytest.raises(ValueError):
            EmbeddingMetadata(
                # Missing required fields
                entity_id="test",
            )

    def test_create_valid_metadata(self):
        """Validate successful metadata creation with all fields."""
        metadata = EmbeddingMetadata(
            embedding_id="emb_123",
            entity_id="person_456",
            entity_type="Person",
            model_id="instructor-large",
            model_version="st:instructor-large",
            schema_version=2,
            schema_hash="abc123def456",
            extraction_confidence=0.95,
            source_document_id="doc_789",
        )

        assert metadata.embedding_id == "emb_123"
        assert metadata.entity_id == "person_456"
        assert metadata.entity_type == "Person"
        assert metadata.schema_version == 2
        assert metadata.schema_hash == "abc123def456"
        assert metadata.extraction_confidence == 0.95
        assert metadata.needs_reembedding is False
        assert metadata.reembedding_reason is None

    def test_schema_version_minimum_value(self):
        """Validate schema_version must be >= 1."""
        with pytest.raises(ValueError):
            EmbeddingMetadata(
                embedding_id="emb_123",
                entity_id="person_456",
                entity_type="Person",
                model_id="instructor-large",
                model_version="st:instructor-large",
                schema_version=0,  # Invalid - must be >= 1
                schema_hash="abc123",
                source_document_id="doc_789",
            )

    def test_extraction_confidence_range(self):
        """Validate extraction_confidence is between 0.0 and 1.0."""
        # Valid
        metadata = EmbeddingMetadata(
            embedding_id="emb_123",
            entity_id="person_456",
            entity_type="Person",
            model_id="instructor-large",
            model_version="st:instructor-large",
            schema_version=1,
            schema_hash="abc123",
            extraction_confidence=0.5,
            source_document_id="doc_789",
        )
        assert metadata.extraction_confidence == 0.5

        # Invalid - above 1.0
        with pytest.raises(ValueError):
            EmbeddingMetadata(
                embedding_id="emb_123",
                entity_id="person_456",
                entity_type="Person",
                model_id="instructor-large",
                model_version="st:instructor-large",
                schema_version=1,
                schema_hash="abc123",
                extraction_confidence=1.5,  # Invalid
                source_document_id="doc_789",
            )

    def test_to_chromadb_metadata_primitives(self):
        """Validate ChromaDB conversion produces only primitive types."""
        metadata = EmbeddingMetadata(
            embedding_id="emb_123",
            entity_id="person_456",
            entity_type="Person",
            model_id="instructor-large",
            model_version="st:instructor-large",
            schema_version=2,
            schema_hash="abc123def456",
            extraction_confidence=0.95,
            source_document_id="doc_789",
            extraction_template_version="template_v1",
            embedding_quality_score=0.88,
        )

        chroma_meta = metadata.to_chromadb_metadata()

        # All values should be primitives
        for key, value in chroma_meta.items():
            assert isinstance(value, (str, int, float, bool)), \
                f"Field {key} has non-primitive type {type(value)}"

        assert chroma_meta["schema_version"] == 2
        assert chroma_meta["entity_type"] == "Person"
        assert chroma_meta["extraction_confidence"] == 0.95
        assert chroma_meta["embedding_quality_score"] == 0.88
        assert chroma_meta["needs_reembedding"] is False

    def test_to_chromadb_metadata_handles_none_values(self):
        """Validate None values are converted to sentinel values."""
        metadata = EmbeddingMetadata(
            embedding_id="emb_123",
            entity_id="person_456",
            entity_type="Person",
            model_id="instructor-large",
            model_version="st:instructor-large",
            schema_version=1,
            schema_hash="abc123",
            source_document_id="doc_789",
            # Optional fields left as None
            extraction_template_version=None,
            embedding_quality_score=None,
            reembedding_reason=None,
        )

        chroma_meta = metadata.to_chromadb_metadata()

        # None values should be converted to sentinels
        assert chroma_meta["extraction_template_version"] == ""
        assert chroma_meta["embedding_quality_score"] == -1.0
        assert chroma_meta["reembedding_reason"] == ""

    def test_chromadb_metadata_roundtrip(self):
        """Validate metadata can be reconstructed from ChromaDB format."""
        original = EmbeddingMetadata(
            embedding_id="emb_123",
            entity_id="person_456",
            entity_type="Person",
            model_id="instructor-large",
            model_version="st:instructor-large",
            schema_version=2,
            schema_hash="abc123def456789",
            extraction_confidence=0.95,
            source_document_id="doc_789",
            extraction_template_version="template_v1",
            embedding_quality_score=0.88,
            needs_reembedding=True,
            reembedding_reason="schema_evolution",
        )

        # Convert to ChromaDB format
        chroma_meta = original.to_chromadb_metadata()

        # Reconstruct
        reconstructed = EmbeddingMetadata.from_chromadb_metadata(chroma_meta)

        # Verify all fields match
        assert reconstructed.embedding_id == original.embedding_id
        assert reconstructed.entity_id == original.entity_id
        assert reconstructed.entity_type == original.entity_type
        assert reconstructed.schema_version == original.schema_version
        assert reconstructed.schema_hash == original.schema_hash
        assert reconstructed.extraction_confidence == original.extraction_confidence
        assert reconstructed.extraction_template_version == original.extraction_template_version
        assert reconstructed.embedding_quality_score == original.embedding_quality_score
        assert reconstructed.needs_reembedding == original.needs_reembedding
        assert reconstructed.reembedding_reason == original.reembedding_reason

    def test_chromadb_roundtrip_handles_none_values(self):
        """Validate None values survive roundtrip conversion."""
        original = EmbeddingMetadata(
            embedding_id="emb_123",
            entity_id="person_456",
            entity_type="Person",
            model_id="instructor-large",
            model_version="st:instructor-large",
            schema_version=1,
            schema_hash="abc123",
            source_document_id="doc_789",
            # Leave optional fields as None
        )

        chroma_meta = original.to_chromadb_metadata()
        reconstructed = EmbeddingMetadata.from_chromadb_metadata(chroma_meta)

        assert reconstructed.extraction_template_version is None
        assert reconstructed.embedding_quality_score is None
        assert reconstructed.reembedding_reason is None

    def test_datetime_fields_serialization(self):
        """Validate datetime fields are serialized to ISO format."""
        metadata = EmbeddingMetadata(
            embedding_id="emb_123",
            entity_id="person_456",
            entity_type="Person",
            model_id="instructor-large",
            model_version="st:instructor-large",
            schema_version=1,
            schema_hash="abc123",
            source_document_id="doc_789",
            created_at=datetime(2024, 1, 15, 14, 30, 0),
            last_validated=datetime(2024, 1, 16, 10, 0, 0),
        )

        chroma_meta = metadata.to_chromadb_metadata()

        assert chroma_meta["created_at"] == "2024-01-15T14:30:00"
        assert chroma_meta["last_validated"] == "2024-01-16T10:00:00"

        # Verify roundtrip
        reconstructed = EmbeddingMetadata.from_chromadb_metadata(chroma_meta)
        assert reconstructed.created_at == datetime(2024, 1, 15, 14, 30, 0)
        assert reconstructed.last_validated == datetime(2024, 1, 16, 10, 0, 0)


# -----------------------------------------------------------------------------
# SchemaVersionedEmbeddingStore Tests
# -----------------------------------------------------------------------------


class TestSchemaVersionedEmbeddingStore:
    """Tests for SchemaVersionedEmbeddingStore."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=None)
        return driver

    @pytest.fixture
    def mock_schema_version_node(self):
        """Create a mock SchemaVersionNode."""
        node = MagicMock()
        node.version = 3
        node.entity_types = ["Person", "Organization", "Event", "Concept"]
        node.relationship_types = ["RELATED_TO", "WORKS_AT", "CAUSES"]
        return node

    @pytest.fixture
    def mock_store(self, mock_neo4j_driver, mock_schema_version_node):
        """Create a SchemaVersionedEmbeddingStore with mocked dependencies."""
        with patch("futurnal.embeddings.schema_versioned_store.TemporalAwareVectorWriter") as mock_writer:
            # Setup mock collections
            mock_events_collection = MagicMock()
            mock_entities_collection = MagicMock()
            mock_sequences_collection = MagicMock()

            writer_instance = MagicMock()
            writer_instance._events_collection = mock_events_collection
            writer_instance._entities_collection = mock_entities_collection
            writer_instance._sequences_collection = mock_sequences_collection

            mock_writer.return_value = writer_instance

            # Patch SchemaVersionManager at the source module
            with patch("futurnal.pkg.schema.migration.SchemaVersionManager") as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager.get_current_version.return_value = mock_schema_version_node
                mock_manager_class.return_value = mock_manager

                from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore

                store = SchemaVersionedEmbeddingStore(
                    neo4j_driver=mock_neo4j_driver,
                )

                # Attach mocks for testing
                store._mock_events_collection = mock_events_collection
                store._mock_entities_collection = mock_entities_collection
                store._mock_manager = mock_manager

                return store

    def test_schema_version_from_pkg(self, mock_store, mock_schema_version_node):
        """Validate schema version is fetched from PKG."""
        version = mock_store._get_current_schema_version()
        assert version == 3

    def test_schema_version_caching(self, mock_store, mock_schema_version_node):
        """Validate schema version is cached after first fetch."""
        # First call
        version1 = mock_store._get_current_schema_version()

        # Second call should use cache
        version2 = mock_store._get_current_schema_version()

        assert version1 == version2 == 3

        # SchemaVersionManager should only be called once (cached)
        assert mock_store._mock_manager.get_current_version.call_count == 1

    def test_schema_hash_computation(self, mock_store):
        """Validate schema hash is deterministic."""
        hash1 = mock_store._compute_schema_hash()
        mock_store._current_schema_hash = None  # Clear cache
        hash2 = mock_store._compute_schema_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_refresh_schema_cache(self, mock_store, mock_schema_version_node):
        """Validate refresh_schema_cache clears and re-fetches."""
        # Initial fetch
        mock_store._get_current_schema_version()
        mock_store._compute_schema_hash()

        # Clear cache
        mock_store.refresh_schema_cache()

        # Should have re-fetched
        assert mock_store._mock_manager.get_current_version.call_count == 2

    def test_store_embedding_creates_metadata(self, mock_store):
        """Validate store_embedding creates proper metadata."""
        embedding = np.random.rand(768).tolist()

        embedding_id = mock_store.store_embedding(
            entity_id="person_123",
            entity_type="Person",
            embedding=embedding,
            model_id="instructor-large",
            extraction_confidence=0.95,
            source_document_id="doc_456",
        )

        # Should have called upsert on entities collection
        mock_store._mock_entities_collection.upsert.assert_called_once()

        # Verify metadata includes schema version
        call_args = mock_store._mock_entities_collection.upsert.call_args
        metadata = call_args.kwargs["metadatas"][0]

        assert metadata["schema_version"] == 3
        assert "schema_hash" in metadata
        assert metadata["entity_id"] == "person_123"
        assert metadata["entity_type"] == "Person"
        assert metadata["extraction_confidence"] == 0.95

    def test_store_event_with_temporal_context(self, mock_store):
        """Validate events are stored in events collection with temporal context."""
        embedding = np.random.rand(768).tolist()
        temporal_context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
            duration=timedelta(hours=2),
        )

        embedding_id = mock_store.store_embedding(
            entity_id="event_123",
            entity_type="Event",
            embedding=embedding,
            model_id="instructor-temporal",
            extraction_confidence=0.9,
            source_document_id="doc_456",
            temporal_context=temporal_context,
        )

        # Should have called upsert on events collection (not entities)
        mock_store._mock_events_collection.upsert.assert_called_once()
        mock_store._mock_entities_collection.upsert.assert_not_called()

        # Verify temporal metadata
        call_args = mock_store._mock_events_collection.upsert.call_args
        metadata = call_args.kwargs["metadatas"][0]

        assert metadata["timestamp"] == "2024-01-15T14:30:00"
        assert metadata["duration_seconds"] == 7200.0

    def test_mark_for_reembedding_by_entity_ids(self, mock_store):
        """Validate mark_for_reembedding marks specific entities."""
        # Setup mock return value
        mock_store._mock_entities_collection.get.return_value = {
            "ids": ["emb_1"],
            "metadatas": [{"entity_id": "person_123", "schema_version": 2}],
        }

        marked = mock_store.mark_for_reembedding(
            entity_ids=["person_123"],
            reason="quality",
        )

        # Verify update was called with re-embedding flags
        mock_store._mock_entities_collection.update.assert_called()
        call_args = mock_store._mock_entities_collection.update.call_args
        updated_metadata = call_args.kwargs["metadatas"][0]

        assert updated_metadata["needs_reembedding"] is True
        assert updated_metadata["reembedding_reason"] == "quality"

    def test_mark_for_reembedding_by_schema_version(self, mock_store):
        """Validate mark_for_reembedding marks all from schema version."""
        # Setup mock return values
        mock_store._mock_events_collection.get.return_value = {
            "ids": ["emb_1", "emb_2"],
            "metadatas": [
                {"entity_id": "event_1", "schema_version": 1},
                {"entity_id": "event_2", "schema_version": 1},
            ],
        }
        mock_store._mock_entities_collection.get.return_value = {
            "ids": ["emb_3"],
            "metadatas": [
                {"entity_id": "person_1", "schema_version": 1},
            ],
        }

        marked = mock_store.mark_for_reembedding(
            schema_version=1,
            reason="schema_evolution",
        )

        # Both collections should be queried and updated
        mock_store._mock_events_collection.get.assert_called()
        mock_store._mock_entities_collection.get.assert_called()

    def test_get_embeddings_needing_reembedding(self, mock_store):
        """Validate get_embeddings_needing_reembedding retrieves flagged items."""
        mock_store._mock_events_collection.get.return_value = {
            "ids": ["emb_1"],
            "metadatas": [
                {
                    "entity_id": "event_1",
                    "needs_reembedding": True,
                    "reembedding_reason": "schema_evolution",
                },
            ],
        }
        mock_store._mock_entities_collection.get.return_value = {
            "ids": [],
            "metadatas": [],
        }

        pending = mock_store.get_embeddings_needing_reembedding(limit=100)

        assert len(pending) == 1
        assert pending[0]["entity_id"] == "event_1"
        assert pending[0]["needs_reembedding"] is True

    def test_query_embeddings_with_schema_version_filter(self, mock_store):
        """Validate query_embeddings supports schema version filtering."""
        mock_store._mock_entities_collection.query.return_value = {
            "ids": [["emb_1", "emb_2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[
                {"entity_id": "person_1", "entity_type": "static_entity", "schema_version": 3},
                {"entity_id": "person_2", "entity_type": "static_entity", "schema_version": 3},
            ]],
        }

        query_vector = np.random.rand(768).tolist()
        results = mock_store.query_embeddings(
            query_vector=query_vector,
            top_k=10,
            min_schema_version=2,
        )

        # Verify filter was passed
        call_args = mock_store._mock_entities_collection.query.call_args
        where_filter = call_args.kwargs.get("where")

        assert where_filter is not None
        assert where_filter["schema_version"] == {"$gte": 2}

        # Verify results
        assert len(results) == 2

    def test_query_embeddings_exclude_needs_reembedding(self, mock_store):
        """Validate query can exclude embeddings needing re-embedding."""
        mock_store._mock_entities_collection.query.return_value = {
            "ids": [["emb_1"]],
            "distances": [[0.1]],
            "metadatas": [[{"entity_id": "person_1", "entity_type": "static_entity"}]],
        }

        query_vector = np.random.rand(768).tolist()
        results = mock_store.query_embeddings(
            query_vector=query_vector,
            exclude_needs_reembedding=True,
        )

        # Verify filter was passed
        call_args = mock_store._mock_entities_collection.query.call_args
        where_filter = call_args.kwargs.get("where")

        assert where_filter is not None
        needs_reembedding_filter = where_filter["needs_reembedding"]
        if isinstance(needs_reembedding_filter, dict):
            assert needs_reembedding_filter.get("$eq") is False
        else:
            assert needs_reembedding_filter is False

    def test_current_schema_version_property(self, mock_store):
        """Validate current_schema_version property."""
        assert mock_store.current_schema_version == 3

    def test_current_schema_hash_property(self, mock_store):
        """Validate current_schema_hash property."""
        schema_hash = mock_store.current_schema_hash
        assert isinstance(schema_hash, str)
        assert len(schema_hash) == 64

    def test_get_embedding_count(self, mock_store):
        """Validate get_embedding_count returns collection counts."""
        mock_store._mock_events_collection.count.return_value = 100
        mock_store._mock_entities_collection.count.return_value = 200

        counts = mock_store.get_embedding_count()

        assert counts["events"] == 100
        assert counts["entities"] == 200
        assert counts["total"] == 300


class TestSchemaVersionedStoreWithoutPKG:
    """Tests for SchemaVersionedEmbeddingStore without PKG connection."""

    @pytest.fixture
    def store_without_pkg(self):
        """Create store without Neo4j connection."""
        with patch("futurnal.embeddings.schema_versioned_store.TemporalAwareVectorWriter"):
            from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore

            store = SchemaVersionedEmbeddingStore(
                # No neo4j_driver provided
            )
            return store

    def test_default_schema_version_without_pkg(self, store_without_pkg):
        """Validate default version 1 when no PKG connection."""
        version = store_without_pkg._get_current_schema_version()
        assert version == 1

    def test_empty_schema_hash_without_pkg(self, store_without_pkg):
        """Validate empty hash computation without PKG."""
        schema_hash = store_without_pkg._compute_schema_hash()
        assert isinstance(schema_hash, str)
        assert len(schema_hash) == 64  # SHA-256 of empty string
