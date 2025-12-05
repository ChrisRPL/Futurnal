"""Tests for ModelRegistry and RegisteredModel.

Tests cover:
- Default model loading
- Entity type to model routing
- Custom model registration
- Memory calculations
"""

from __future__ import annotations

import pytest

from futurnal.embeddings.config import ModelType
from futurnal.embeddings.registry import ModelRegistry, RegisteredModel


class TestRegisteredModel:
    """Tests for RegisteredModel Pydantic model."""

    def test_create_registered_model(self):
        """Should create a RegisteredModel with required fields."""
        model = RegisteredModel(
            model_id="test-model",
            model_type=ModelType.INSTRUCTOR_LARGE,
            model_path="hkunlp/instructor-large",
            entity_types=["Person", "Organization"],
        )

        assert model.model_id == "test-model"
        assert model.model_type == ModelType.INSTRUCTOR_LARGE
        assert model.entity_types == ["Person", "Organization"]
        assert model.vector_dimension == 768  # default
        assert model.max_sequence_length == 512  # default

    def test_registered_model_with_all_fields(self):
        """Should create a RegisteredModel with all optional fields."""
        model = RegisteredModel(
            model_id="custom-model",
            model_type=ModelType.CODEBERT_BASE,
            model_path="microsoft/codebert-base",
            entity_types=["CodeEntity"],
            vector_dimension=1024,
            max_sequence_length=1024,
            quantized=True,
            memory_mb=1500,
            avg_latency_ms=200.0,
            instruction="Represent the code:",
        )

        assert model.vector_dimension == 1024
        assert model.max_sequence_length == 1024
        assert model.quantized is True
        assert model.memory_mb == 1500
        assert model.avg_latency_ms == 200.0
        assert model.instruction == "Represent the code:"

    def test_registered_model_immutable(self):
        """RegisteredModel should be immutable (frozen)."""
        model = RegisteredModel(
            model_id="test-model",
            model_type=ModelType.INSTRUCTOR_LARGE,
            model_path="hkunlp/instructor-large",
            entity_types=["Person"],
        )

        with pytest.raises(Exception):  # ValidationError or AttributeError
            model.model_id = "changed"


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_load_default_models(self):
        """Should load 4 default model configurations."""
        registry = ModelRegistry()

        assert len(registry.registered_models) == 4
        assert "instructor-large-entity" in registry.registered_models
        assert "instructor-temporal-event" in registry.registered_models
        assert "codebert-code" in registry.registered_models
        assert "instructor-document" in registry.registered_models

    def test_get_model_for_person(self):
        """Should return instructor-large-entity for Person."""
        registry = ModelRegistry()

        model = registry.get_model_for_entity_type("Person")

        assert model is not None
        assert model.model_id == "instructor-large-entity"

    def test_get_model_for_organization(self):
        """Should return instructor-large-entity for Organization."""
        registry = ModelRegistry()

        model = registry.get_model_for_entity_type("Organization")

        assert model is not None
        assert model.model_id == "instructor-large-entity"

    def test_get_model_for_concept(self):
        """Should return instructor-large-entity for Concept."""
        registry = ModelRegistry()

        model = registry.get_model_for_entity_type("Concept")

        assert model is not None
        assert model.model_id == "instructor-large-entity"

    def test_get_model_for_event(self):
        """Should return instructor-temporal-event for Event."""
        registry = ModelRegistry()

        model = registry.get_model_for_entity_type("Event")

        assert model is not None
        assert model.model_id == "instructor-temporal-event"

    def test_get_model_for_code_entity(self):
        """Should return codebert-code for CodeEntity."""
        registry = ModelRegistry()

        model = registry.get_model_for_entity_type("CodeEntity")

        assert model is not None
        assert model.model_id == "codebert-code"

    def test_get_model_for_document(self):
        """Should return instructor-document for Document."""
        registry = ModelRegistry()

        model = registry.get_model_for_entity_type("Document")

        assert model is not None
        assert model.model_id == "instructor-document"
        assert model.max_sequence_length == 2048  # longer context

    def test_unsupported_entity_type(self):
        """Should return None for unsupported entity types."""
        registry = ModelRegistry()

        model = registry.get_model_for_entity_type("UnknownType")

        assert model is None

    def test_custom_model_registration(self):
        """Should allow custom model registration."""
        registry = ModelRegistry()

        custom_model = RegisteredModel(
            model_id="custom-embedding",
            model_type=ModelType.MINILM_L6_V2,
            model_path="custom/path",
            entity_types=["CustomEntity"],
        )
        registry.register_model(custom_model)

        assert "custom-embedding" in registry.registered_models
        assert "CustomEntity" in registry.supported_entity_types

        retrieved = registry.get_model_for_entity_type("CustomEntity")
        assert retrieved is not None
        assert retrieved.model_id == "custom-embedding"

    def test_model_unregistration(self):
        """Should allow model unregistration."""
        registry = ModelRegistry()

        # Register a custom model
        custom_model = RegisteredModel(
            model_id="to-remove",
            model_type=ModelType.MINILM_L6_V2,
            model_path="path",
            entity_types=["TempEntity"],
        )
        registry.register_model(custom_model)
        assert "to-remove" in registry.registered_models

        # Unregister
        result = registry.unregister_model("to-remove")
        assert result is True
        assert "to-remove" not in registry.registered_models
        assert "TempEntity" not in registry.supported_entity_types

    def test_unregister_nonexistent_model(self):
        """Should return False when unregistering nonexistent model."""
        registry = ModelRegistry()

        result = registry.unregister_model("nonexistent")

        assert result is False

    def test_get_model_by_id(self):
        """Should get model by ID."""
        registry = ModelRegistry()

        model = registry.get_model_by_id("instructor-large-entity")

        assert model is not None
        assert model.model_id == "instructor-large-entity"

    def test_get_nonexistent_model_by_id(self):
        """Should return None for nonexistent model ID."""
        registry = ModelRegistry()

        model = registry.get_model_by_id("nonexistent")

        assert model is None

    def test_supported_entity_types(self):
        """Should return all supported entity types."""
        registry = ModelRegistry()

        entity_types = registry.supported_entity_types

        assert "Person" in entity_types
        assert "Organization" in entity_types
        assert "Concept" in entity_types
        assert "Event" in entity_types
        assert "CodeEntity" in entity_types
        assert "Document" in entity_types

    def test_total_memory_calculation(self):
        """Should calculate total memory across all models."""
        registry = ModelRegistry()

        total_memory = registry.get_total_memory_mb()

        # Default models: 800 + 800 + 600 + 1200 = 3400 MB
        assert total_memory == 3400

    def test_get_models_by_type(self):
        """Should get all models of a specific type."""
        registry = ModelRegistry()

        instructor_models = registry.get_models_by_type(ModelType.INSTRUCTOR_LARGE)

        # instructor-large-entity, instructor-temporal-event, instructor-document
        assert len(instructor_models) == 3

    def test_get_models_for_multiple_entity_types(self):
        """Should get models for multiple entity types."""
        registry = ModelRegistry()

        models = registry.get_models_for_entity_types(["Person", "Event", "Unknown"])

        assert len(models) == 2
        assert "Person" in models
        assert "Event" in models
        assert "Unknown" not in models

    def test_registry_len(self):
        """Should return number of registered models."""
        registry = ModelRegistry()

        assert len(registry) == 4

    def test_registry_contains(self):
        """Should support 'in' operator."""
        registry = ModelRegistry()

        assert "instructor-large-entity" in registry
        assert "nonexistent" not in registry

    def test_registry_repr(self):
        """Should have string representation."""
        registry = ModelRegistry()

        repr_str = repr(registry)

        assert "ModelRegistry" in repr_str
        assert "models=4" in repr_str
