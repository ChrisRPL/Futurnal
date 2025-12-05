"""Tests for MultiModelEmbeddingService.

Tests cover:
- Single entity embedding
- Batch embedding
- Metrics tracking
- Option B temporal enforcement
- Error handling
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.exceptions import BatchProcessingError
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingResult,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.registry import ModelRegistry
from futurnal.embeddings.request import EmbeddingRequest
from futurnal.embeddings.service import MultiModelEmbeddingService


# Mock fixtures
@pytest.fixture
def mock_embedding_result():
    """Create a mock embedding result."""
    return EmbeddingResult(
        embedding=[0.1] * 768,
        entity_type=EmbeddingEntityType.STATIC_ENTITY,
        model_version="mock:v1",
        embedding_dimension=768,
        generation_time_ms=100.0,
        metadata={},
        temporal_context_encoded=False,
        causal_context_encoded=False,
    )


@pytest.fixture
def mock_temporal_embedding_result():
    """Create a mock temporal embedding result."""
    return EmbeddingResult(
        embedding=[0.1] * 768,
        entity_type=EmbeddingEntityType.TEMPORAL_EVENT,
        model_version="mock:v1",
        embedding_dimension=768,
        generation_time_ms=150.0,
        metadata={"timestamp": "2024-01-15T14:30:00"},
        temporal_context_encoded=True,
        causal_context_encoded=False,
    )


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    config = EmbeddingServiceConfig()

    with patch.object(ModelManager, "_load_model"):
        manager = ModelManager(config)

        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=[0.1] * 768)

        manager._models["content-instructor"] = mock_model
        manager._models["temporal-minilm"] = mock_model
        manager._model_versions["content-instructor"] = "mock:instructor"
        manager._model_versions["temporal-minilm"] = "mock:minilm"

        return manager


@pytest.fixture
def embedding_service(mock_model_manager, mock_embedding_result):
    """Create a MultiModelEmbeddingService with mocks."""
    registry = ModelRegistry()
    config = EmbeddingServiceConfig()

    service = MultiModelEmbeddingService(config=config, registry=registry)
    service._model_manager = mock_model_manager

    # Patch the embedders to return mock results
    with patch.object(
        service._router, "_init_embedders"
    ):
        # Manually set up mock embedders
        mock_static_embedder = MagicMock()
        mock_static_embedder.embed = MagicMock(return_value=mock_embedding_result)
        mock_static_embedder.entity_type = EmbeddingEntityType.STATIC_ENTITY

        mock_temporal_embedder = MagicMock()
        mock_temporal_embedder.entity_type = EmbeddingEntityType.TEMPORAL_EVENT

        service._router._embedders = {
            "static": mock_static_embedder,
            "temporal": mock_temporal_embedder,
            "code": mock_static_embedder,
            "document": mock_static_embedder,
            "sequence": mock_temporal_embedder,
        }
        service._router._embedders_initialized = True

        return service


class TestMultiModelEmbeddingService:
    """Tests for MultiModelEmbeddingService."""

    def test_embed_person(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should embed Person entity."""
        result = embedding_service.embed(
            entity_type="Person",
            content="John Smith, Software Engineer",
        )

        assert result is not None
        assert result.embedding_dimension == 768

    def test_embed_organization(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should embed Organization entity."""
        result = embedding_service.embed(
            entity_type="Organization",
            content="Futurnal Inc - AI Research Company",
        )

        assert result is not None

    def test_embed_concept(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should embed Concept entity."""
        result = embedding_service.embed(
            entity_type="Concept",
            content="Machine Learning",
        )

        assert result is not None

    def test_embed_event_requires_temporal(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should raise error for Event without temporal_context."""
        with pytest.raises(ValueError) as exc_info:
            embedding_service.embed(
                entity_type="Event",
                content="Team Meeting",
            )

        assert "temporal_context is REQUIRED" in str(exc_info.value)

    def test_embed_event_with_context(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_temporal_embedding_result,
    ):
        """Should embed Event with temporal context."""
        # Update mock to return temporal result
        embedding_service._router._embedders["temporal"].embed = MagicMock(
            return_value=mock_temporal_embedding_result
        )

        context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
            duration=timedelta(hours=2),
        )

        result = embedding_service.embed(
            entity_type="Event",
            content="Team Meeting: Quarterly planning",
            temporal_context=context,
        )

        assert result is not None
        assert result.temporal_context_encoded is True

    def test_embed_with_all_options(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should embed with all optional parameters."""
        result = embedding_service.embed(
            entity_type="Person",
            content="John Smith",
            entity_id="pkg-123",
            entity_name="John Smith",
            metadata={"role": "Engineer"},
        )

        assert result is not None

    def test_embed_batch(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should process batch of mixed entity types."""
        requests = [
            EmbeddingRequest(entity_type="Person", content="John"),
            EmbeddingRequest(entity_type="Organization", content="Acme"),
            EmbeddingRequest(entity_type="Concept", content="AI"),
        ]

        results = embedding_service.embed_batch(requests)

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_embed_batch_empty(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should handle empty batch."""
        results = embedding_service.embed_batch([])

        assert results == []

    def test_metrics_tracking(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should track embedding metrics."""
        # Generate some embeddings
        embedding_service.embed(entity_type="Person", content="John")
        embedding_service.embed(entity_type="Organization", content="Acme")

        metrics = embedding_service.get_metrics()

        assert metrics["total_embeddings"] >= 2
        assert "entity_type_distribution" in metrics
        assert "model_metrics" in metrics

    def test_latency_summary(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should provide latency summary."""
        embedding_service.embed(entity_type="Person", content="John")

        latency = embedding_service.get_latency_summary()

        assert isinstance(latency, dict)

    def test_unload_models(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should unload all models."""
        embedding_service.unload_all_models()

        # After unloading, models should not be loaded
        assert len(embedding_service._model_manager.loaded_models) == 0

    def test_supported_entity_types(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should return supported entity types."""
        entity_types = embedding_service.get_supported_entity_types()

        assert "Person" in entity_types
        assert "Organization" in entity_types
        assert "Event" in entity_types

    def test_get_model_for_entity_type(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should get model ID for entity type."""
        model_id = embedding_service.get_model_for_entity_type("Person")

        assert model_id == "instructor-large-entity"

    def test_schema_version_property(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should expose schema version."""
        assert embedding_service.schema_version is not None

    def test_context_manager(self):
        """Should work as context manager."""
        config = EmbeddingServiceConfig()
        registry = ModelRegistry()

        with patch.object(ModelManager, "_load_model"):
            with MultiModelEmbeddingService(
                config=config, registry=registry
            ) as service:
                assert service is not None

    def test_service_repr(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should have string representation."""
        repr_str = repr(embedding_service)

        assert "MultiModelEmbeddingService" in repr_str
        assert "models=" in repr_str


class TestBatchProcessing:
    """Tests for batch embedding processing."""

    def test_batch_groups_by_entity_type(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Batch should group requests by entity type."""
        requests = [
            EmbeddingRequest(entity_type="Person", content="John"),
            EmbeddingRequest(entity_type="Person", content="Jane"),
            EmbeddingRequest(entity_type="Organization", content="Acme"),
        ]

        results = embedding_service.embed_batch(requests)

        assert len(results) == 3

    def test_batch_fail_fast_true(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should stop on first error when fail_fast=True."""
        # Make the embedder raise an error
        embedding_service._router._embedders["static"].embed = MagicMock(
            side_effect=Exception("Test error")
        )

        requests = [
            EmbeddingRequest(entity_type="Person", content="John"),
            EmbeddingRequest(entity_type="Person", content="Jane"),
        ]

        with pytest.raises(BatchProcessingError) as exc_info:
            embedding_service.embed_batch(requests, fail_fast=True)

        assert len(exc_info.value.failed_indices) >= 1

    def test_batch_fail_fast_false_continues(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should continue processing when fail_fast=False."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First fails")
            return mock_embedding_result

        embedding_service._router._embedders["static"].embed = MagicMock(
            side_effect=side_effect
        )

        requests = [
            EmbeddingRequest(entity_type="Person", content="John"),
            EmbeddingRequest(entity_type="Person", content="Jane"),
        ]

        with pytest.raises(BatchProcessingError) as exc_info:
            embedding_service.embed_batch(requests, fail_fast=False)

        # Should have one successful result
        assert exc_info.value.partial_success is True
        assert len(exc_info.value.successful_results) == 1


class TestOptionBCompliance:
    """Tests for Option B compliance in service."""

    def test_event_requires_temporal_context(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Option B: Events must have temporal context."""
        with pytest.raises(ValueError) as exc_info:
            embedding_service.embed(
                entity_type="Event",
                content="Important Meeting",
            )

        assert "temporal_context is REQUIRED" in str(exc_info.value)

    def test_static_entities_no_temporal_required(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Option B: Static entities don't require temporal context."""
        for entity_type in ["Person", "Organization", "Concept"]:
            result = embedding_service.embed(
                entity_type=entity_type,
                content=f"Test {entity_type}",
            )
            assert result is not None


class TestMetrics:
    """Tests for metrics collection."""

    def test_metrics_track_entity_types(
        self,
        embedding_service: MultiModelEmbeddingService,
        mock_embedding_result,
    ):
        """Should track entity type distribution."""
        embedding_service.embed(entity_type="Person", content="John")
        embedding_service.embed(entity_type="Person", content="Jane")
        embedding_service.embed(entity_type="Organization", content="Acme")

        metrics = embedding_service.get_metrics()

        assert "Person" in metrics["entity_type_distribution"]
        assert metrics["entity_type_distribution"]["Person"] >= 2

    def test_metrics_track_uptime(
        self,
        embedding_service: MultiModelEmbeddingService,
    ):
        """Should track service uptime."""
        metrics = embedding_service.get_metrics()

        assert "uptime_seconds" in metrics
        assert metrics["uptime_seconds"] >= 0
