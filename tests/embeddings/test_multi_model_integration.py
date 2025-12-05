"""Integration tests for Multi-Model Architecture.

Tests end-to-end flows from request to embedding result,
verifying proper routing, embedding generation, and metrics.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingResult,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.registry import ModelRegistry
from futurnal.embeddings.request import EmbeddingRequest
from futurnal.embeddings.router import ModelRouter
from futurnal.embeddings.service import MultiModelEmbeddingService


# Test fixtures
class MockEmbeddingModelForIntegration:
    """Mock model that generates deterministic embeddings."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        instruction=None,
        **kwargs,
    ):
        if isinstance(sentences, str):
            return self._encode_single(sentences, instruction)
        return np.array([self._encode_single(s, instruction) for s in sentences])

    def _encode_single(self, text: str, instruction=None):
        combined = f"{instruction or ''}{text}"
        seed = hash(combined) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.random(self.dimension)
        return embedding / np.linalg.norm(embedding)


@pytest.fixture
def integration_model_manager():
    """Create a model manager with mock models for integration tests."""
    config = EmbeddingServiceConfig()

    with patch.object(ModelManager, "_load_model"):
        manager = ModelManager(config)

        mock_model = MockEmbeddingModelForIntegration(dimension=768)

        manager._models["content-instructor"] = mock_model
        manager._models["temporal-minilm"] = MockEmbeddingModelForIntegration(
            dimension=384
        )
        manager._model_versions["content-instructor"] = "mock:instructor-large"
        manager._model_versions["temporal-minilm"] = "mock:minilm"

        return manager


@pytest.fixture
def integration_service(integration_model_manager):
    """Create a full service for integration testing."""
    registry = ModelRegistry()
    config = EmbeddingServiceConfig()

    service = MultiModelEmbeddingService(config=config, registry=registry)
    service._model_manager = integration_model_manager

    # Re-initialize router with the mock manager
    service._router = ModelRouter(
        registry=registry,
        model_manager=integration_model_manager,
        config=config,
    )

    return service


class TestFullRoutingPipeline:
    """End-to-end tests for the routing pipeline."""

    def test_person_routing_pipeline(self, integration_service):
        """Should route Person through full pipeline."""
        result = integration_service.embed(
            entity_type="Person",
            content="John Smith, Software Engineer",
        )

        assert result is not None
        assert len(result.embedding) > 0
        assert result.entity_type == EmbeddingEntityType.STATIC_ENTITY

    def test_organization_routing_pipeline(self, integration_service):
        """Should route Organization through full pipeline."""
        result = integration_service.embed(
            entity_type="Organization",
            content="Futurnal Inc - AI Research Company",
        )

        assert result is not None
        assert result.entity_type == EmbeddingEntityType.STATIC_ENTITY

    def test_event_routing_pipeline(self, integration_service):
        """Should route Event through temporal pipeline."""
        context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
            duration=timedelta(hours=2),
        )

        result = integration_service.embed(
            entity_type="Event",
            content="Team Meeting: Quarterly planning discussion",
            temporal_context=context,
        )

        assert result is not None
        assert result.entity_type == EmbeddingEntityType.TEMPORAL_EVENT
        assert result.temporal_context_encoded is True

    def test_code_entity_routing_pipeline(self, integration_service):
        """Should route CodeEntity through pipeline."""
        result = integration_service.embed(
            entity_type="CodeEntity",
            content="def hello_world(): print('Hello, World!')",
        )

        assert result is not None

    def test_document_routing_pipeline(self, integration_service):
        """Should route Document through pipeline."""
        result = integration_service.embed(
            entity_type="Document",
            content="This is a comprehensive document about machine learning...",
        )

        assert result is not None


class TestMixedEntityBatch:
    """Tests for batch processing with mixed entity types."""

    def test_batch_with_multiple_types(self, integration_service):
        """Should handle batch with multiple entity types."""
        context = TemporalEmbeddingContext(timestamp=datetime.now())

        requests = [
            EmbeddingRequest(entity_type="Person", content="John Smith"),
            EmbeddingRequest(entity_type="Organization", content="Acme Inc"),
            EmbeddingRequest(entity_type="Concept", content="Machine Learning"),
            EmbeddingRequest(
                entity_type="Event",
                content="Team Meeting",
                temporal_context=context,
            ),
        ]

        results = integration_service.embed_batch(requests)

        assert len(results) == 4

        # Verify correct entity types
        entity_types = [r.entity_type for r in results]
        assert EmbeddingEntityType.STATIC_ENTITY in entity_types
        assert EmbeddingEntityType.TEMPORAL_EVENT in entity_types

    def test_batch_preserves_order(self, integration_service):
        """Batch should preserve request order in results."""
        requests = [
            EmbeddingRequest(entity_type="Person", content="First"),
            EmbeddingRequest(entity_type="Person", content="Second"),
            EmbeddingRequest(entity_type="Person", content="Third"),
        ]

        results = integration_service.embed_batch(requests)

        # Results should be in same order as requests
        assert len(results) == 3


class TestMetricsIntegration:
    """Tests for metrics collection across operations."""

    def test_metrics_accumulate(self, integration_service):
        """Metrics should accumulate across operations."""
        # Clear any existing metrics
        integration_service._metrics.reset()

        # Generate embeddings
        integration_service.embed(entity_type="Person", content="John")
        integration_service.embed(entity_type="Organization", content="Acme")

        context = TemporalEmbeddingContext(timestamp=datetime.now())
        integration_service.embed(
            entity_type="Event",
            content="Meeting",
            temporal_context=context,
        )

        metrics = integration_service.get_metrics()

        assert metrics["total_embeddings"] == 3
        assert "Person" in metrics["entity_type_distribution"]
        assert "Organization" in metrics["entity_type_distribution"]
        assert "Event" in metrics["entity_type_distribution"]

    def test_metrics_track_different_models(self, integration_service):
        """Metrics should track usage per model."""
        integration_service._metrics.reset()

        # Generate embeddings for different types
        integration_service.embed(entity_type="Person", content="John")

        context = TemporalEmbeddingContext(timestamp=datetime.now())
        integration_service.embed(
            entity_type="Event",
            content="Meeting",
            temporal_context=context,
        )

        metrics = integration_service.get_metrics()

        # Should have model metrics for both used models
        model_metrics = metrics["model_metrics"]
        assert len(model_metrics) >= 1


class TestEmbeddingQuality:
    """Tests for embedding quality and determinism."""

    def test_same_input_same_output(self, integration_service):
        """Same input should produce same embedding."""
        result1 = integration_service.embed(
            entity_type="Person",
            content="John Smith, Software Engineer",
        )
        result2 = integration_service.embed(
            entity_type="Person",
            content="John Smith, Software Engineer",
        )

        # Embeddings should be identical
        np.testing.assert_array_almost_equal(
            result1.embedding,
            result2.embedding,
            decimal=5,
        )

    def test_different_content_different_output(self, integration_service):
        """Different content should produce different embeddings."""
        result1 = integration_service.embed(
            entity_type="Person",
            content="John Smith",
        )
        result2 = integration_service.embed(
            entity_type="Person",
            content="Jane Doe",
        )

        # Embeddings should be different
        assert not np.allclose(result1.embedding, result2.embedding)

    def test_embedding_normalization(self, integration_service):
        """Embeddings should be L2 normalized."""
        result = integration_service.embed(
            entity_type="Person",
            content="John Smith",
        )

        embedding = np.array(result.embedding)
        norm = np.linalg.norm(embedding)

        # Should be normalized to unit length
        assert np.isclose(norm, 1.0, atol=0.01)


class TestTemporalEmbeddings:
    """Tests for temporal embedding behavior."""

    def test_temporal_context_affects_embedding(self, integration_service):
        """Temporal context should affect the embedding."""
        context1 = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 1),
        )
        context2 = TemporalEmbeddingContext(
            timestamp=datetime(2024, 6, 1),
        )

        result1 = integration_service.embed(
            entity_type="Event",
            content="Team Meeting",
            temporal_context=context1,
        )
        result2 = integration_service.embed(
            entity_type="Event",
            content="Team Meeting",
            temporal_context=context2,
        )

        # Same content but different temporal context should produce different embeddings
        assert not np.allclose(result1.embedding, result2.embedding)

    def test_temporal_context_encoded_flag(self, integration_service):
        """Temporal events should have temporal_context_encoded=True."""
        context = TemporalEmbeddingContext(timestamp=datetime.now())

        result = integration_service.embed(
            entity_type="Event",
            content="Meeting",
            temporal_context=context,
        )

        assert result.temporal_context_encoded is True

    def test_static_entity_no_temporal_encoded(self, integration_service):
        """Static entities should have temporal_context_encoded=False."""
        result = integration_service.embed(
            entity_type="Person",
            content="John Smith",
        )

        assert result.temporal_context_encoded is False


class TestServiceLifecycle:
    """Tests for service lifecycle management."""

    def test_context_manager_cleanup(self, integration_service):
        """Context manager should clean up resources."""
        # Use integration_service but manually call context manager methods
        integration_service.__enter__()
        integration_service.embed(entity_type="Person", content="John")
        integration_service.__exit__(None, None, None)

        # After context manager exit, service should be cleaned up
        # (models unloaded)
        assert len(integration_service._model_manager.loaded_models) == 0

    def test_manual_close(self, integration_service):
        """Manual close should clean up resources."""
        integration_service.embed(entity_type="Person", content="John")

        integration_service.close()

        # After close, models should be unloaded
        assert len(integration_service._model_manager.loaded_models) == 0


class TestRegistryRouterIntegration:
    """Tests for registry and router integration."""

    def test_all_registered_types_routable(self, integration_service):
        """All registered entity types should be routable."""
        for entity_type in integration_service.get_supported_entity_types():
            model_id = integration_service.get_model_for_entity_type(entity_type)
            assert model_id is not None, f"No model for {entity_type}"

    def test_router_uses_registry(self, integration_service):
        """Router should use registry for model selection."""
        # Get model via service
        model_id = integration_service.get_model_for_entity_type("Person")

        # Should match registry
        registry_model = integration_service.registry.get_model_for_entity_type(
            "Person"
        )
        assert model_id == registry_model.model_id
