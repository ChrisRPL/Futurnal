"""Tests for ModelRouter.

Tests cover:
- Entity type routing
- Embedder selection
- Model lookup
- Unknown type handling
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.exceptions import ModelNotFoundError
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import EmbeddingEntityType, TemporalEmbeddingContext
from futurnal.embeddings.registry import ModelRegistry
from futurnal.embeddings.request import EmbeddingRequest
from futurnal.embeddings.router import ModelRouter


@pytest.fixture
def model_registry() -> ModelRegistry:
    """Create a test model registry."""
    return ModelRegistry()


@pytest.fixture
def mock_model_manager_for_router() -> ModelManager:
    """Create a mock model manager for router tests."""
    config = EmbeddingServiceConfig()

    with patch.object(ModelManager, "_load_model"):
        manager = ModelManager(config)

        # Mock the model loading
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=[0.1] * 768)

        manager._models["content-instructor"] = mock_model
        manager._models["temporal-minilm"] = mock_model
        manager._model_versions["content-instructor"] = "mock:instructor"
        manager._model_versions["temporal-minilm"] = "mock:minilm"

        return manager


@pytest.fixture
def model_router(
    model_registry: ModelRegistry,
    mock_model_manager_for_router: ModelManager,
) -> ModelRouter:
    """Create a model router with mocks."""
    return ModelRouter(
        registry=model_registry,
        model_manager=mock_model_manager_for_router,
    )


class TestModelRouter:
    """Tests for ModelRouter."""

    def test_route_person_request(self, model_router: ModelRouter):
        """Should route Person to static embedder."""
        request = EmbeddingRequest(
            entity_type="Person",
            content="John Smith",
        )

        model_id, embedder = model_router.route_request(request)

        assert model_id == "instructor-large-entity"
        assert embedder is not None
        # StaticEntityEmbedder
        assert embedder.entity_type == EmbeddingEntityType.STATIC_ENTITY

    def test_route_organization_request(self, model_router: ModelRouter):
        """Should route Organization to static embedder."""
        request = EmbeddingRequest(
            entity_type="Organization",
            content="Acme Inc",
        )

        model_id, embedder = model_router.route_request(request)

        assert model_id == "instructor-large-entity"
        assert embedder.entity_type == EmbeddingEntityType.STATIC_ENTITY

    def test_route_concept_request(self, model_router: ModelRouter):
        """Should route Concept to static embedder."""
        request = EmbeddingRequest(
            entity_type="Concept",
            content="Machine Learning",
        )

        model_id, embedder = model_router.route_request(request)

        assert model_id == "instructor-large-entity"
        assert embedder.entity_type == EmbeddingEntityType.STATIC_ENTITY

    def test_route_event_request(self, model_router: ModelRouter):
        """Should route Event to temporal embedder."""
        context = TemporalEmbeddingContext(timestamp=datetime.now())
        request = EmbeddingRequest(
            entity_type="Event",
            content="Team Meeting",
            temporal_context=context,
        )

        model_id, embedder = model_router.route_request(request)

        assert model_id == "instructor-temporal-event"
        assert embedder.entity_type == EmbeddingEntityType.TEMPORAL_EVENT

    def test_route_code_entity_request(self, model_router: ModelRouter):
        """Should route CodeEntity appropriately."""
        request = EmbeddingRequest(
            entity_type="CodeEntity",
            content="def hello(): pass",
        )

        model_id, embedder = model_router.route_request(request)

        assert model_id == "codebert-code"
        # Currently uses static embedder, but model is code-specific
        assert embedder is not None

    def test_route_document_request(self, model_router: ModelRouter):
        """Should route Document appropriately."""
        request = EmbeddingRequest(
            entity_type="Document",
            content="This is a long document...",
        )

        model_id, embedder = model_router.route_request(request)

        assert model_id == "instructor-document"
        assert embedder is not None

    def test_route_unknown_type_raises(self, model_router: ModelRouter):
        """Should raise ModelNotFoundError for unknown entity types."""
        # Create a request with an unknown type (bypass validation for test)
        request = MagicMock()
        request.entity_type = "UnknownType"
        request.content = "test"

        with pytest.raises(ModelNotFoundError) as exc_info:
            model_router.route_request(request)

        assert "UnknownType" in str(exc_info.value)
        assert "Supported types" in str(exc_info.value)

    def test_route_entity_type_simple(self, model_router: ModelRouter):
        """Should route entity type without full request."""
        model_id, embedder = model_router.route_entity_type("Person")

        assert model_id == "instructor-large-entity"
        assert embedder is not None

    def test_get_registered_model(self, model_router: ModelRouter):
        """Should get registered model for entity type."""
        model = model_router.get_registered_model("Person")

        assert model is not None
        assert model.model_id == "instructor-large-entity"

    def test_get_registered_model_none_for_unknown(self, model_router: ModelRouter):
        """Should return None for unknown entity type."""
        model = model_router.get_registered_model("UnknownType")

        assert model is None

    def test_is_temporal_entity_event(self, model_router: ModelRouter):
        """Should identify Event as temporal entity."""
        assert model_router.is_temporal_entity("Event") is True

    def test_is_temporal_entity_person(self, model_router: ModelRouter):
        """Should identify Person as non-temporal entity."""
        assert model_router.is_temporal_entity("Person") is False

    def test_get_supported_entity_types(self, model_router: ModelRouter):
        """Should return all supported entity types."""
        entity_types = model_router.get_supported_entity_types()

        assert "Person" in entity_types
        assert "Organization" in entity_types
        assert "Concept" in entity_types
        assert "Event" in entity_types
        assert "CodeEntity" in entity_types
        assert "Document" in entity_types

    def test_embedder_initialization_lazy(
        self,
        model_registry: ModelRegistry,
        mock_model_manager_for_router: ModelManager,
    ):
        """Should initialize embedders lazily."""
        router = ModelRouter(
            registry=model_registry,
            model_manager=mock_model_manager_for_router,
        )

        # Before any routing, embedders not initialized
        assert router._embedders_initialized is False

        # After routing, embedders are initialized
        router.route_entity_type("Person")
        assert router._embedders_initialized is True

    def test_embedders_property(self, model_router: ModelRouter):
        """Should return all embedders via property."""
        embedders = model_router.embedders

        assert "static" in embedders
        assert "temporal" in embedders
        assert "sequence" in embedders
        assert "document" in embedders
        assert "code" in embedders

    def test_get_embedder_category_static(self, model_router: ModelRouter):
        """Should return 'static' category for Person."""
        category = model_router.get_embedder_category("Person")
        assert category == "static"

    def test_get_embedder_category_temporal(self, model_router: ModelRouter):
        """Should return 'temporal' category for Event."""
        category = model_router.get_embedder_category("Event")
        assert category == "temporal"

    def test_get_embedder_category_code(self, model_router: ModelRouter):
        """Should return 'code' category for CodeEntity."""
        category = model_router.get_embedder_category("CodeEntity")
        assert category == "code"

    def test_get_embedder_category_document(self, model_router: ModelRouter):
        """Should return 'document' category for Document."""
        category = model_router.get_embedder_category("Document")
        assert category == "document"

    def test_get_embedder_category_unknown(self, model_router: ModelRouter):
        """Should return 'static' as default for unknown types."""
        category = model_router.get_embedder_category("UnknownType")
        assert category == "static"

    def test_registry_property(self, model_router: ModelRouter, model_registry):
        """Should expose registry via property."""
        assert model_router.registry is model_registry

    def test_model_manager_property(
        self, model_router: ModelRouter, mock_model_manager_for_router
    ):
        """Should expose model manager via property."""
        assert model_router.model_manager is mock_model_manager_for_router

    def test_router_repr(self, model_router: ModelRouter):
        """Should have string representation."""
        repr_str = repr(model_router)

        assert "ModelRouter" in repr_str
        assert "models=" in repr_str
        assert "entity_types=" in repr_str


class TestEntityTypeMapping:
    """Tests for entity type to embedding type mapping."""

    def test_entity_type_map_completeness(self):
        """ENTITY_TYPE_MAP should cover all common types."""
        expected_types = {
            "Person",
            "Organization",
            "Concept",
            "Event",
            "CodeEntity",
            "Document",
        }

        assert expected_types == set(ModelRouter.ENTITY_TYPE_MAP.keys())

    def test_static_entities_mapped_correctly(self):
        """Person, Org, Concept should map to STATIC_ENTITY."""
        for entity_type in ["Person", "Organization", "Concept"]:
            assert (
                ModelRouter.ENTITY_TYPE_MAP[entity_type]
                == EmbeddingEntityType.STATIC_ENTITY
            )

    def test_event_mapped_to_temporal(self):
        """Event should map to TEMPORAL_EVENT."""
        assert (
            ModelRouter.ENTITY_TYPE_MAP["Event"]
            == EmbeddingEntityType.TEMPORAL_EVENT
        )

    def test_code_entity_mapped_correctly(self):
        """CodeEntity should map to CODE_ENTITY."""
        assert (
            ModelRouter.ENTITY_TYPE_MAP["CodeEntity"]
            == EmbeddingEntityType.CODE_ENTITY
        )

    def test_document_mapped_correctly(self):
        """Document should map to DOCUMENT."""
        assert (
            ModelRouter.ENTITY_TYPE_MAP["Document"]
            == EmbeddingEntityType.DOCUMENT
        )
