"""End-to-end embedding pipeline tests.

Validates:
- Full flow: PKG entity -> embedding -> storage -> query
- Temporal event embedding with context preservation
- Multi-entity type batch processing
- Model routing correctness

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/06-integration-testing.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pytest

from tests.embeddings.integration.conftest import (
    EmbeddingPipeline,
    SampleEntity,
    create_test_entity,
    create_embedding_pipeline,
    cosine_similarity,
)
from futurnal.embeddings.models import TemporalEmbeddingContext
from futurnal.embeddings.request import EmbeddingRequest


class TestFullEmbeddingPipeline:
    """End-to-end embedding pipeline tests."""

    def test_entity_to_embedding_full_flow(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate full pipeline: PKG entity -> embedding -> storage -> query."""
        # Setup
        entity = create_test_entity("Person", "John Doe", "Software Engineer")
        pipeline = embedding_pipeline

        # 1. Generate embedding
        result = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
            entity_id=entity.id,
            entity_name=entity.name,
        )

        assert result.embedding is not None
        assert len(result.embedding) == 768

        # 2. Store with schema version
        embedding_id = pipeline.store.store_embedding(
            entity_id=entity.id,
            entity_type=entity.type,
            embedding=result.embedding,
            model_id=result.model_version,
            extraction_confidence=0.9,
            source_document_id="test_doc",
        )
        assert embedding_id is not None

        # 3. Query back
        results = pipeline.store.query_embeddings(
            query_vector=result.embedding,
            top_k=1,
        )

        assert len(results) > 0
        assert results[0].entity_id == entity.id

    def test_temporal_event_embedding_flow(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate temporal event embedding with context preservation."""
        # Create event with temporal context
        event = create_test_entity(
            "Event",
            "Meeting with stakeholders",
            "Quarterly planning session",
            timestamp=datetime(2024, 1, 15, 10, 0),
            duration=timedelta(hours=2),
        )

        pipeline = embedding_pipeline

        # Generate temporal embedding
        result = pipeline.embedding_service.embed(
            entity_type="Event",
            content=event.content,
            temporal_context=event.temporal_context,
            entity_id=event.id,
            entity_name=event.name,
        )

        assert result.embedding is not None
        assert len(result.embedding) == 768

        # Store and retrieve
        embedding_id = pipeline.store.store_embedding(
            entity_id=event.id,
            entity_type="Event",
            embedding=result.embedding,
            model_id=result.model_version,
            extraction_confidence=0.85,
            source_document_id="test_doc",
            temporal_context=event.temporal_context,
        )

        # Verify can query by entity type
        results = pipeline.store.query_embeddings(
            query_vector=result.embedding,
            top_k=1,
            entity_type="Event",
        )

        assert len(results) > 0
        assert results[0].metadata.get("entity_id") == event.id

    def test_multi_entity_type_batch_processing(
        self,
        embedding_pipeline: EmbeddingPipeline,
        sample_test_entities: Dict[str, SampleEntity],
    ) -> None:
        """Validate batch processing across different entity types."""
        pipeline = embedding_pipeline

        # Create mixed batch requests
        requests = []
        for entity in sample_test_entities.values():
            requests.append(
                EmbeddingRequest(
                    entity_type=entity.type,
                    content=entity.content,
                    entity_id=entity.id,
                    temporal_context=entity.temporal_context,
                )
            )

        # Batch embed
        results = pipeline.embedding_service.embed_batch(requests)

        # Validate all succeeded
        assert len(results) == len(sample_test_entities)
        assert all(r.embedding is not None for r in results)

        # Validate different entity types were processed
        entity_types = {r.entity_type for r in results}
        assert len(entity_types) >= 2  # At least 2 different entity types

    def test_embedding_determinism(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate that same input produces same embedding."""
        entity = create_test_entity("Person", "Jane Smith", "Data Scientist")
        pipeline = embedding_pipeline

        result1 = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
        )

        result2 = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
        )

        # Same content should produce same embedding (with mock models)
        np.testing.assert_array_almost_equal(
            result1.embedding,
            result2.embedding,
            decimal=5,
        )

    def test_schema_version_in_stored_embeddings(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate schema version is stored with embeddings."""
        entity = create_test_entity("Person", "Test Person")
        pipeline = embedding_pipeline

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

        # Query and verify schema version present
        results = pipeline.store.query_embeddings(
            query_vector=result.embedding,
            top_k=1,
        )

        assert len(results) > 0
        assert "schema_version" in results[0].metadata
        assert results[0].metadata["schema_version"] >= 1

    def test_different_entities_different_embeddings(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate different entities get different embeddings."""
        pipeline = embedding_pipeline

        person = create_test_entity("Person", "Alice", "Engineer")
        org = create_test_entity("Organization", "TechCorp", "Software company")

        result1 = pipeline.embedding_service.embed(
            entity_type=person.type,
            content=person.content,
        )

        result2 = pipeline.embedding_service.embed(
            entity_type=org.type,
            content=org.content,
        )

        # Different content should produce different embeddings
        similarity = cosine_similarity(result1.embedding, result2.embedding)
        assert similarity < 0.99, "Different entities should have different embeddings"

    def test_store_and_retrieve_multiple_entities(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate storing and retrieving multiple entities."""
        pipeline = embedding_pipeline

        # Store multiple entities
        entities = [
            create_test_entity("Person", f"Person {i}", f"Description {i}")
            for i in range(10)
        ]

        for entity in entities:
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

        # Query with random vector
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=10,
        )

        assert len(results) == 10

    def test_embedding_result_metadata(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate embedding result contains expected metadata."""
        pipeline = embedding_pipeline
        entity = create_test_entity("Person", "Test Person")

        result = pipeline.embedding_service.embed(
            entity_type=entity.type,
            content=entity.content,
        )

        assert result.embedding_dimension == 768
        assert result.generation_time_ms >= 0
        assert result.model_version is not None
        assert result.metadata is not None

    def test_batch_processing_with_mixed_types(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate batch processing handles mixed entity types correctly."""
        pipeline = embedding_pipeline

        requests = [
            EmbeddingRequest(
                entity_type="Person",
                content="Person content",
            ),
            EmbeddingRequest(
                entity_type="Organization",
                content="Org content",
            ),
            EmbeddingRequest(
                entity_type="Event",
                content="Event content",
                temporal_context=TemporalEmbeddingContext(
                    timestamp=datetime.utcnow(),
                ),
            ),
            EmbeddingRequest(
                entity_type="Concept",
                content="Concept content",
            ),
        ]

        results = pipeline.embedding_service.embed_batch(requests)

        assert len(results) == 4
        assert all(len(r.embedding) == 768 for r in results)

    def test_query_with_entity_type_filter(
        self,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        """Validate query filtering by entity type."""
        pipeline = embedding_pipeline

        # Store mixed entities
        person = create_test_entity("Person", "Test Person")
        event = create_test_entity(
            "Event",
            "Test Event",
            timestamp=datetime.utcnow(),
        )

        for entity in [person, event]:
            result = pipeline.embedding_service.embed(
                entity_type=entity.type,
                content=entity.content,
                temporal_context=entity.temporal_context,
            )
            pipeline.store.store_embedding(
                entity_id=entity.id,
                entity_type=entity.type,
                embedding=result.embedding,
                model_id="test-model",
                extraction_confidence=0.9,
                source_document_id="test_doc",
            )

        # Query only persons
        query_vector = np.random.rand(768).tolist()
        results = pipeline.store.query_embeddings(
            query_vector=query_vector,
            top_k=10,
            entity_type="Person",
        )

        assert len(results) == 1
        assert results[0].entity_id == person.id
