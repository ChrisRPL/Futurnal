"""Model Registry for Multi-Model Architecture.

Provides centralized management of embedding model configurations
and entity type to model mappings.

Option B Compliance:
- Models are FROZEN (configurations only, no training)
- Tracks model versions via model_id
- Supports multiple specialized models per entity type

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/02-multi-model-architecture.md
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from futurnal.embeddings.config import ModelType

logger = logging.getLogger(__name__)


class RegisteredModel(BaseModel):
    """Configuration for a registered embedding model.

    Defines all parameters needed for model loading and routing.

    Attributes:
        model_id: Unique identifier for this model registration
        model_type: Type of model (from ModelType enum)
        model_path: HuggingFace path or local path
        entity_types: Entity types this model handles
        vector_dimension: Output embedding dimension
        max_sequence_length: Max input tokens
        quantized: Whether to apply quantization for on-device efficiency
        memory_mb: Expected memory footprint in MB
        avg_latency_ms: Benchmarked average latency in milliseconds
        instruction: Default instruction for Instructor models

    Example:
        model = RegisteredModel(
            model_id="instructor-large-entity",
            model_type=ModelType.INSTRUCTOR_LARGE,
            model_path="hkunlp/instructor-large",
            entity_types=["Person", "Organization", "Concept"],
            vector_dimension=768,
            max_sequence_length=512,
            quantized=True,
            memory_mb=800,
            avg_latency_ms=150,
            instruction="Represent the entity for semantic retrieval:",
        )
    """

    model_id: str = Field(..., description="Unique model identifier")
    model_type: ModelType = Field(..., description="Model type enum")
    model_path: str = Field(..., description="HuggingFace or local path")
    entity_types: List[str] = Field(..., description="Supported entity types")
    vector_dimension: int = Field(default=768, description="Output embedding dimension")
    max_sequence_length: int = Field(default=512, description="Max input tokens")
    quantized: bool = Field(default=False, description="Apply quantization")
    memory_mb: int = Field(default=800, description="Expected memory footprint in MB")
    avg_latency_ms: float = Field(default=150.0, description="Benchmarked avg latency")
    instruction: Optional[str] = Field(
        default=None, description="Default instruction for Instructor models"
    )

    model_config = {"frozen": True}


class ModelRegistry:
    """Registry of available embedding models.

    Manages model configurations and entity type mappings for the
    multi-model architecture. Supports:
    - Default model configurations for all entity types
    - Custom model registration
    - Entity type to model routing
    - Memory estimation

    Option B Compliance:
        - Models are FROZEN (configurations only, no training)
        - Tracks model versions via model_id for schema versioning

    Thread Safety:
        This class is not thread-safe. If concurrent access is needed,
        wrap operations with appropriate locking.

    Example:
        registry = ModelRegistry()

        # Get model for entity type
        model = registry.get_model_for_entity_type("Person")
        assert model.model_id == "instructor-large-entity"

        # Register custom model
        registry.register_model(RegisteredModel(
            model_id="custom-model",
            model_type=ModelType.MINILM_L6_V2,
            model_path="custom/path",
            entity_types=["CustomEntity"],
        ))
    """

    def __init__(self) -> None:
        """Initialize the model registry with default configurations."""
        self._models: Dict[str, RegisteredModel] = {}
        self._entity_type_map: Dict[str, str] = {}  # entity_type -> model_id
        self._load_default_models()

    def _load_default_models(self) -> None:
        """Load default model configurations from production plan.

        Default models:
        - instructor-large-entity: Person, Organization, Concept (static entities)
        - instructor-temporal-event: Event (temporal events)
        - codebert-code: CodeEntity (code-specific)
        - instructor-document: Document (longer context)
        """
        # Static entity model (Person, Organization, Concept)
        self.register_model(
            RegisteredModel(
                model_id="instructor-large-entity",
                model_type=ModelType.INSTRUCTOR_LARGE,
                model_path="hkunlp/instructor-large",
                entity_types=["Person", "Organization", "Concept"],
                vector_dimension=768,
                max_sequence_length=512,
                quantized=True,
                memory_mb=800,
                avg_latency_ms=150,
                instruction="Represent the entity for semantic retrieval:",
            )
        )

        # Temporal event model
        self.register_model(
            RegisteredModel(
                model_id="instructor-temporal-event",
                model_type=ModelType.INSTRUCTOR_LARGE,
                model_path="hkunlp/instructor-large",
                entity_types=["Event"],
                vector_dimension=768,
                max_sequence_length=512,
                quantized=True,
                memory_mb=800,
                avg_latency_ms=150,
                instruction="Represent the event with temporal context for retrieval:",
            )
        )

        # Code entity model
        self.register_model(
            RegisteredModel(
                model_id="codebert-code",
                model_type=ModelType.CODEBERT_BASE,
                model_path="microsoft/codebert-base",
                entity_types=["CodeEntity"],
                vector_dimension=768,
                max_sequence_length=512,
                quantized=True,
                memory_mb=600,
                avg_latency_ms=120,
            )
        )

        # Document model (longer context)
        self.register_model(
            RegisteredModel(
                model_id="instructor-document",
                model_type=ModelType.INSTRUCTOR_LARGE,
                model_path="hkunlp/instructor-large",
                entity_types=["Document"],
                vector_dimension=768,
                max_sequence_length=2048,
                quantized=True,
                memory_mb=1200,
                avg_latency_ms=300,
                instruction="Represent the document for retrieval:",
            )
        )

        logger.info(
            f"Loaded {len(self._models)} default model configurations: "
            f"{list(self._models.keys())}"
        )

    def register_model(self, model: RegisteredModel) -> None:
        """Register a model configuration.

        Args:
            model: Model configuration to register

        Note:
            If a model with the same ID exists, it will be replaced.
            Entity type mappings are updated accordingly.
        """
        self._models[model.model_id] = model

        # Update entity type mappings
        for entity_type in model.entity_types:
            self._entity_type_map[entity_type] = model.model_id
            logger.debug(f"Mapped entity type '{entity_type}' to model '{model.model_id}'")

        logger.debug(f"Registered model: {model.model_id}")

    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model configuration.

        Args:
            model_id: ID of the model to unregister

        Returns:
            True if model was unregistered, False if not found
        """
        if model_id not in self._models:
            return False

        model = self._models[model_id]

        # Remove entity type mappings
        for entity_type in model.entity_types:
            if self._entity_type_map.get(entity_type) == model_id:
                del self._entity_type_map[entity_type]

        del self._models[model_id]
        logger.debug(f"Unregistered model: {model_id}")
        return True

    def get_model_for_entity_type(self, entity_type: str) -> Optional[RegisteredModel]:
        """Get best model for entity type.

        Args:
            entity_type: Entity type to find model for (e.g., "Person", "Event")

        Returns:
            RegisteredModel if found, None otherwise
        """
        model_id = self._entity_type_map.get(entity_type)
        if model_id:
            return self._models.get(model_id)
        return None

    def get_model_by_id(self, model_id: str) -> Optional[RegisteredModel]:
        """Get model by ID.

        Args:
            model_id: Unique identifier of the model

        Returns:
            RegisteredModel if found, None otherwise
        """
        return self._models.get(model_id)

    def get_models_for_entity_types(
        self, entity_types: List[str]
    ) -> Dict[str, RegisteredModel]:
        """Get models for multiple entity types.

        Args:
            entity_types: List of entity types

        Returns:
            Dict mapping entity_type to RegisteredModel (only for found types)
        """
        result = {}
        for entity_type in entity_types:
            model = self.get_model_for_entity_type(entity_type)
            if model:
                result[entity_type] = model
        return result

    @property
    def registered_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self._models.keys())

    @property
    def supported_entity_types(self) -> Set[str]:
        """Set of all supported entity types."""
        return set(self._entity_type_map.keys())

    def get_total_memory_mb(self) -> int:
        """Calculate total memory if all models loaded.

        Returns:
            Total memory in MB across all registered models
        """
        return sum(m.memory_mb for m in self._models.values())

    def get_models_by_type(self, model_type: ModelType) -> List[RegisteredModel]:
        """Get all models of a specific type.

        Args:
            model_type: ModelType enum value

        Returns:
            List of RegisteredModel with matching type
        """
        return [m for m in self._models.values() if m.model_type == model_type]

    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        """Check if model ID is registered."""
        return model_id in self._models

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModelRegistry(models={len(self._models)}, "
            f"entity_types={len(self._entity_type_map)})"
        )
