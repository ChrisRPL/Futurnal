"""Model Router for Multi-Model Architecture.

Routes embedding requests to appropriate models and embedders based on
entity type. Core component of the multi-model architecture.

Option B Compliance:
- Models are FROZEN (no fine-tuning)
- Routes temporal entities (Events) to temporal-aware embedders
- Enforces temporal context requirements via request validation

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/02-multi-model-architecture.md
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from futurnal.embeddings.base import BaseEmbedder
from futurnal.embeddings.config import EmbeddingServiceConfig
from futurnal.embeddings.exceptions import ModelNotFoundError, RoutingError
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import EmbeddingEntityType
from futurnal.embeddings.registry import ModelRegistry, RegisteredModel

if TYPE_CHECKING:
    from futurnal.embeddings.request import EmbeddingRequest

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes embedding requests to appropriate models and embedders.

    Core component of the multi-model architecture that:
    - Maps entity types to specialized models
    - Manages embedder instances for each entity category
    - Coordinates with ModelManager for model loading
    - Validates routing decisions

    Entity Type Categories:
        - STATIC_ENTITY: Person, Organization, Concept (no temporal context)
        - TEMPORAL_EVENT: Event (temporal context required)
        - CODE_ENTITY: CodeEntity (code-specific embeddings)
        - DOCUMENT: Document (longer context support)

    Option B Compliance:
        - Models are FROZEN (no fine-tuning, loaded as-is)
        - Temporal context enforced for Event type
        - Routes events to TemporalEventEmbedder for temporal-aware embedding

    Example:
        registry = ModelRegistry()
        manager = ModelManager(config)
        router = ModelRouter(registry, manager)

        # Route a request
        request = EmbeddingRequest(entity_type="Person", content="John Smith")
        model_id, embedder = router.route_request(request)

        # Generate embedding
        result = embedder.embed(
            entity_type="Person",
            entity_name="John Smith",
            entity_description="Software Engineer",
        )
    """

    # Map entity types to EmbeddingEntityType enum categories
    ENTITY_TYPE_MAP: Dict[str, EmbeddingEntityType] = {
        "Person": EmbeddingEntityType.STATIC_ENTITY,
        "Organization": EmbeddingEntityType.STATIC_ENTITY,
        "Concept": EmbeddingEntityType.STATIC_ENTITY,
        "Event": EmbeddingEntityType.TEMPORAL_EVENT,
        "CodeEntity": EmbeddingEntityType.CODE_ENTITY,
        "Document": EmbeddingEntityType.DOCUMENT,
    }

    def __init__(
        self,
        registry: ModelRegistry,
        model_manager: ModelManager,
        config: Optional[EmbeddingServiceConfig] = None,
    ) -> None:
        """Initialize the router.

        Args:
            registry: Model registry with model configurations
            model_manager: Manager for loading models
            config: Service configuration (uses defaults if None)
        """
        self._registry = registry
        self._model_manager = model_manager
        self._config = config or EmbeddingServiceConfig()

        # Embedder cache (keyed by embedder category)
        self._embedders: Dict[str, BaseEmbedder] = {}

        # Initialize embedders lazily to avoid import cycles
        self._embedders_initialized = False

        logger.info(
            f"Initialized ModelRouter with {len(registry)} models, "
            f"supporting {len(registry.supported_entity_types)} entity types"
        )

    def _init_embedders(self) -> None:
        """Initialize embedders for each entity type category.

        Lazy initialization to avoid import cycles and allow
        deferred model loading.
        """
        if self._embedders_initialized:
            return

        # Import here to avoid circular imports
        from futurnal.embeddings.static_entity import StaticEntityEmbedder
        from futurnal.embeddings.temporal_event import TemporalEventEmbedder
        from futurnal.embeddings.event_sequence import EventSequenceEmbedder

        # Static entity embedder (Person, Organization, Concept)
        self._embedders["static"] = StaticEntityEmbedder(
            self._model_manager,
            content_model_id="content-instructor",
        )

        # Temporal event embedder
        self._embedders["temporal"] = TemporalEventEmbedder(
            self._model_manager,
            content_model_id="content-instructor",
            temporal_model_id="temporal-minilm",
        )

        # Event sequence embedder (uses temporal embedder)
        self._embedders["sequence"] = EventSequenceEmbedder(
            self._model_manager,
            event_embedder=self._embedders["temporal"],
        )

        # Document embedder (use static with longer context)
        # Note: StaticEntityEmbedder handles documents appropriately
        self._embedders["document"] = StaticEntityEmbedder(
            self._model_manager,
            content_model_id="content-instructor",
        )

        # Code embedder (use static for now, future: CodeEntityEmbedder)
        self._embedders["code"] = StaticEntityEmbedder(
            self._model_manager,
            content_model_id="content-instructor",
        )

        self._embedders_initialized = True
        logger.debug(f"Initialized {len(self._embedders)} embedders")

    def route_request(
        self, request: "EmbeddingRequest"
    ) -> Tuple[str, BaseEmbedder]:
        """Route an embedding request to appropriate model and embedder.

        Args:
            request: The embedding request (already validated)

        Returns:
            Tuple of (model_id, embedder) for processing the request

        Raises:
            ModelNotFoundError: If no model registered for entity type
            RoutingError: If routing fails for other reasons
        """
        entity_type = request.entity_type

        # Get registered model for this entity type
        registered_model = self._registry.get_model_for_entity_type(entity_type)
        if registered_model is None:
            raise ModelNotFoundError(
                f"No model registered for entity type: {entity_type}. "
                f"Supported types: {sorted(self._registry.supported_entity_types)}"
            )

        # Get appropriate embedder
        embedder = self._get_embedder_for_entity_type(entity_type)

        logger.debug(
            f"Routed {entity_type} request to model '{registered_model.model_id}' "
            f"using {type(embedder).__name__}"
        )

        return registered_model.model_id, embedder

    def route_entity_type(self, entity_type: str) -> Tuple[str, BaseEmbedder]:
        """Route an entity type to appropriate model and embedder.

        Simpler interface when you don't have a full request object.

        Args:
            entity_type: Type of entity (Person, Event, etc.)

        Returns:
            Tuple of (model_id, embedder)

        Raises:
            ModelNotFoundError: If no model registered for entity type
        """
        registered_model = self._registry.get_model_for_entity_type(entity_type)
        if registered_model is None:
            raise ModelNotFoundError(
                f"No model registered for entity type: {entity_type}. "
                f"Supported types: {sorted(self._registry.supported_entity_types)}"
            )

        embedder = self._get_embedder_for_entity_type(entity_type)
        return registered_model.model_id, embedder

    def _get_embedder_for_entity_type(self, entity_type: str) -> BaseEmbedder:
        """Get the embedder for an entity type.

        Args:
            entity_type: Type of entity

        Returns:
            Appropriate embedder for the entity type

        Raises:
            RoutingError: If no embedder available for entity type
        """
        # Ensure embedders are initialized
        self._init_embedders()

        # Get entity category
        entity_category = self.ENTITY_TYPE_MAP.get(entity_type)

        if entity_category == EmbeddingEntityType.STATIC_ENTITY:
            return self._embedders["static"]
        elif entity_category == EmbeddingEntityType.TEMPORAL_EVENT:
            return self._embedders["temporal"]
        elif entity_category == EmbeddingEntityType.CODE_ENTITY:
            return self._embedders["code"]
        elif entity_category == EmbeddingEntityType.DOCUMENT:
            return self._embedders["document"]
        else:
            # Unknown entity type - try static embedder as default
            logger.warning(
                f"Unknown entity type '{entity_type}', using static embedder"
            )
            return self._embedders["static"]

    def get_embedder_category(self, entity_type: str) -> str:
        """Get the embedder category for an entity type.

        Args:
            entity_type: Type of entity

        Returns:
            Category string: "static", "temporal", "code", or "document"
        """
        entity_category = self.ENTITY_TYPE_MAP.get(entity_type)

        if entity_category == EmbeddingEntityType.STATIC_ENTITY:
            return "static"
        elif entity_category == EmbeddingEntityType.TEMPORAL_EVENT:
            return "temporal"
        elif entity_category == EmbeddingEntityType.CODE_ENTITY:
            return "code"
        elif entity_category == EmbeddingEntityType.DOCUMENT:
            return "document"
        else:
            return "static"

    def get_registered_model(self, entity_type: str) -> Optional[RegisteredModel]:
        """Get registered model configuration for entity type.

        Args:
            entity_type: Type of entity

        Returns:
            RegisteredModel if found, None otherwise
        """
        return self._registry.get_model_for_entity_type(entity_type)

    def is_temporal_entity(self, entity_type: str) -> bool:
        """Check if entity type requires temporal context.

        Args:
            entity_type: Type of entity

        Returns:
            True if entity type requires temporal context
        """
        return (
            self.ENTITY_TYPE_MAP.get(entity_type) == EmbeddingEntityType.TEMPORAL_EVENT
        )

    def get_supported_entity_types(self) -> list[str]:
        """Get list of all supported entity types.

        Returns:
            Sorted list of supported entity type strings
        """
        return sorted(self._registry.supported_entity_types)

    @property
    def registry(self) -> ModelRegistry:
        """Access the model registry."""
        return self._registry

    @property
    def model_manager(self) -> ModelManager:
        """Access the model manager."""
        return self._model_manager

    @property
    def embedders(self) -> Dict[str, BaseEmbedder]:
        """Access initialized embedders (initializes if needed)."""
        self._init_embedders()
        return dict(self._embedders)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModelRouter(models={len(self._registry)}, "
            f"entity_types={len(self._registry.supported_entity_types)}, "
            f"embedders_initialized={self._embedders_initialized})"
        )
