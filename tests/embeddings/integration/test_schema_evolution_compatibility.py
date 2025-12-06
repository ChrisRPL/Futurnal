"""Schema evolution compatibility tests.

Validates:
- Schema version tracking in embeddings
- Re-embedding triggered on schema evolution
- Backward compatibility with older schema versions

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/06-integration-testing.md
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np
import pytest

from tests.embeddings.integration.conftest import (
    EmbeddingPipeline,
    create_test_entity,
    create_embedding_pipeline,
)


class TestSchemaEvolutionCompatibility:
    """Validate schema evolution doesn't break embeddings."""

    def test_schema_version_tracking(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate embeddings track schema versions."""
        pipeline = embedding_pipeline

        # Create embeddings (schema v1 by default)
        entity1 = create_test_entity("Person", "John", "Engineer")
        result1 = pipeline.embedding_service.embed(
            entity_type=entity1.type,
            content=entity1.content,
        )

        id1 = pipeline.store.store_embedding(
            entity_id=entity1.id,
            entity_type=entity1.type,
            embedding=result1.embedding,
            model_id="test-model",
            extraction_confidence=0.9,
            source_document_id="doc1",
        )

        # Query and verify schema version
        results = pipeline.store.query_embeddings(
            query_vector=result1.embedding,
            top_k=10,
        )

        assert len(results) > 0
        # All results should have schema version
        for result in results:
            assert "schema_version" in result.metadata
            assert result.metadata["schema_version"] >= 1

    def test_schema_version_filtering(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate schema version filtering in queries."""
        pipeline = embedding_pipeline

        # Create multiple embeddings at schema v1
        for i in range(5):
            entity = create_test_entity("Person", f"Person {i}")
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content,
            )
            pipeline.store.store_embedding(
                entity_id=entity.id,
                entity_type=entity.type,
                embedding=result.embedding,
                model_id="test-model",
                extraction_confidence=0.9,
                source_document_id=f"doc_{i}",
            )

        # Query with schema version filter
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=10,
            min_schema_version=1,
        )

        # Should return results with schema version >= 1
        assert len(results) > 0
        for result in results:
            assert result.metadata.get("schema_version", 0) >= 1

    def test_mark_for_reembedding(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate embeddings can be marked for re-embedding."""
        pipeline = embedding_pipeline

        # Create and store embedding
        entity = create_test_entity("Person", "Mark Test")
        result = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
        )

        embedding_id = pipeline.store.store_embedding(
            entity_id=entity.id,
            entity_type=entity.type,
            embedding=result.embedding,
            model_id="test-model",
            extraction_confidence=0.9,
            source_document_id="test_doc",
        )

        # Mark for re-embedding
        marked_count = pipeline.store.mark_for_reembedding(
            entity_ids=[entity.id],
            reason="schema_evolution",
        )

        assert marked_count >= 1

        # Verify embeddings needing re-embedding can be retrieved
        pending = pipeline.store.get_embeddings_needing_reembedding(limit=10)
        entity_ids = [p.get("entity_id") for p in pending]
        assert entity.id in entity_ids

    def test_clear_reembedding_flag(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate re-embedding flag can be cleared."""
        pipeline = embedding_pipeline

        # Create, store, and mark embedding
        entity = create_test_entity("Person", "Clear Test")
        result = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
        )

        embedding_id = pipeline.store.store_embedding(
            entity_id=entity.id,
            entity_type=entity.type,
            embedding=result.embedding,
            model_id="test-model",
            extraction_confidence=0.9,
            source_document_id="test_doc",
        )

        pipeline.store.mark_for_reembedding(
            entity_ids=[entity.id],
            reason="test",
        )

        # Clear flag
        cleared = pipeline.store.clear_reembedding_flag([embedding_id])
        assert cleared >= 1

        # Verify no longer in pending list
        pending = pipeline.store.get_embeddings_needing_reembedding(limit=100)
        embedding_ids = [p.get("embedding_id") for p in pending]
        assert embedding_id not in embedding_ids

    def test_schema_version_persists_after_query(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate schema version persists across queries."""
        pipeline = embedding_pipeline

        # Store with specific schema version
        entity = create_test_entity("Person", "Schema Persist Test")
        result = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
        )

        pipeline.store.store_embedding(
            entity_id=entity.id,
            entity_type=entity.type,
            embedding=result.embedding,
            model_id="test-model",
            extraction_confidence=0.9,
            source_document_id="test_doc",
        )

        # Query multiple times
        for _ in range(3):
            results = pipeline.store.query_embeddings(
                query_vector=result.embedding,
                top_k=1,
            )
            assert len(results) > 0
            assert results[0].metadata.get("schema_version") == 1

    def test_multiple_schema_versions_coexist(self) -> None:
        """Validate embeddings with different schema versions can coexist."""
        # Create pipeline with schema v1
        pipeline1 = create_embedding_pipeline(current_schema_version=1)

        # Store at v1
        entity1 = create_test_entity("Person", "V1 Person")
        result1 = pipeline1.embedding_service.embed(
            entity_type=entity1.type,
            content=entity1.content,
        )
        pipeline1.store.store_embedding(
            entity_id=entity1.id,
            entity_type=entity1.type,
            embedding=result1.embedding,
            model_id="test-model",
            extraction_confidence=0.9,
            source_document_id="doc1",
        )

        # Update schema version to v2
        pipeline1.store.current_schema_version = 2

        # Store at v2
        entity2 = create_test_entity("Person", "V2 Person")
        result2 = pipeline1.embedding_service.embed(
            entity_type=entity2.type,
            content=entity2.content,
        )
        pipeline1.store.store_embedding(
            entity_id=entity2.id,
            entity_type=entity2.type,
            embedding=result2.embedding,
            model_id="test-model",
            extraction_confidence=0.9,
            source_document_id="doc2",
        )

        # Query all - should get both versions
        query_vector = np.random.rand(768).tolist()
        all_results = pipeline1.store.query_embeddings(
            query_vector=query_vector,
            top_k=10,
        )
        assert len(all_results) == 2

        # Query v2 only
        v2_results = pipeline1.store.query_embeddings(
            query_vector=query_vector,
            top_k=10,
            min_schema_version=2,
        )
        assert len(v2_results) == 1
        assert v2_results[0].entity_id == entity2.id

        pipeline1.close()

    def test_reembedding_reason_tracked(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate re-embedding reason is tracked."""
        pipeline = embedding_pipeline

        entity = create_test_entity("Person", "Reason Test")
        result = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
        )

        pipeline.store.store_embedding(
            entity_id=entity.id,
            entity_type=entity.type,
            embedding=result.embedding,
            model_id="test-model",
            extraction_confidence=0.9,
            source_document_id="test_doc",
        )

        # Mark with specific reason
        pipeline.store.mark_for_reembedding(
            entity_ids=[entity.id],
            reason="model_upgrade",
        )

        pending = pipeline.store.get_embeddings_needing_reembedding(limit=10)
        matching = [p for p in pending if p.get("entity_id") == entity.id]
        assert len(matching) == 1
        assert matching[0].get("reason") == "model_upgrade"
