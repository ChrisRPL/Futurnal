"""Entity Type Retrieval Strategies.

Different retrieval strategies per entity type, with optimized
vector/graph weight configurations.

Strategy Rationale:
- Event: Higher graph weight (temporal relationships matter)
- CodeEntity: Higher vector weight (semantic similarity matters)
- Document: Higher vector weight (content focus)
- Static entities: Balanced (Person, Organization, Concept)

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md

Option B Compliance:
- Entity type strategies support temporal-first design
- Events get special temporal-aware handling
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.types import HybridSearchResult, TemporalQueryContext

if TYPE_CHECKING:
    from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval

logger = logging.getLogger(__name__)


class EntityTypeRetrievalStrategy:
    """Different retrieval strategies per entity type.

    Provides entity-type-aware search with optimized weights:

    Entity Weights (from production plan):
    - Event: 40% vector, 60% graph (temporal relationships matter)
    - CodeEntity: 70% vector, 30% graph (semantic similarity)
    - Document: 60% vector, 40% graph (content focus)
    - Person/Organization/Concept: 50/50 (balanced)

    The strategy layer wraps SchemaAwareRetrieval and applies
    entity-type-specific configurations.

    Example:
        >>> from futurnal.search.hybrid import (
        ...     SchemaAwareRetrieval,
        ...     EntityTypeRetrievalStrategy,
        ... )

        >>> retrieval = SchemaAwareRetrieval(...)
        >>> strategy = EntityTypeRetrievalStrategy(retrieval)

        >>> # Search for events (uses temporal weights)
        >>> results = strategy.search_by_entity_type(
        ...     query="project kickoff",
        ...     target_entity_type="Event",
        ...     top_k=10,
        ... )

        >>> # Search for code (uses code weights)
        >>> results = strategy.search_by_entity_type(
        ...     query="authentication handler",
        ...     target_entity_type="CodeEntity",
        ...     top_k=10,
        ... )
    """

    # Default entity type weights (from production plan)
    ENTITY_WEIGHTS: Dict[str, Dict[str, float]] = {
        "Event": {"vector": 0.4, "graph": 0.6},
        "CodeEntity": {"vector": 0.7, "graph": 0.3},
        "Document": {"vector": 0.6, "graph": 0.4},
        "Person": {"vector": 0.5, "graph": 0.5},
        "Organization": {"vector": 0.5, "graph": 0.5},
        "Concept": {"vector": 0.5, "graph": 0.5},
        "Chunk": {"vector": 0.7, "graph": 0.3},  # Chunks are content-focused
    }

    # Intent mapping by entity type
    ENTITY_INTENT_MAP: Dict[str, str] = {
        "Event": "temporal",
        "CodeEntity": "lookup",
        "Document": "exploratory",
        "Person": "lookup",
        "Organization": "lookup",
        "Concept": "exploratory",
    }

    def __init__(
        self,
        retrieval: "SchemaAwareRetrieval",
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """Initialize entity type retrieval strategy.

        Args:
            retrieval: SchemaAwareRetrieval engine to wrap
            config: Configuration (uses retrieval's config if None)
        """
        self._retrieval = retrieval
        self._config = config or retrieval.config

        # Merge config weights with default weights
        self._entity_weights = self._build_weights_map()

        logger.info(
            f"Initialized EntityTypeRetrievalStrategy for "
            f"{len(self._entity_weights)} entity types"
        )

    @property
    def config(self) -> HybridSearchConfig:
        """Get the configuration."""
        return self._config

    def _build_weights_map(self) -> Dict[str, Dict[str, float]]:
        """Build entity weights map from config and defaults."""
        weights = dict(self.ENTITY_WEIGHTS)

        # Override with config values
        weights["Event"] = {
            "vector": self._config.event_vector_weight,
            "graph": self._config.event_graph_weight,
        }
        weights["CodeEntity"] = {
            "vector": self._config.code_vector_weight,
            "graph": self._config.code_graph_weight,
        }
        weights["Document"] = {
            "vector": self._config.document_vector_weight,
            "graph": self._config.document_graph_weight,
        }

        return weights

    def get_weights_for_type(
        self,
        entity_type: str,
    ) -> Dict[str, float]:
        """Get vector/graph weights for an entity type.

        Args:
            entity_type: Entity type name

        Returns:
            Dict with "vector" and "graph" weights
        """
        return self._entity_weights.get(
            entity_type,
            {
                "vector": self._config.default_vector_weight,
                "graph": self._config.default_graph_weight,
            },
        )

    def get_intent_for_type(self, entity_type: str) -> str:
        """Get recommended search intent for an entity type.

        Args:
            entity_type: Entity type name

        Returns:
            Recommended intent string
        """
        return self.ENTITY_INTENT_MAP.get(entity_type, "exploratory")

    def search_by_entity_type(
        self,
        query: str,
        target_entity_type: str,
        top_k: int = 10,
        temporal_context: Optional[TemporalQueryContext] = None,
    ) -> List[HybridSearchResult]:
        """Search with entity type focus.

        Adjusts retrieval strategy based on target entity type:
        - Selects appropriate vector/graph weights
        - Chooses optimal search intent
        - Filters results to target type

        Args:
            query: Natural language search query
            target_entity_type: Target entity type to search for
            top_k: Maximum results to return
            temporal_context: Optional temporal context

        Returns:
            List of HybridSearchResult filtered to target type
        """
        # Get weights for this entity type
        weights = self.get_weights_for_type(target_entity_type)

        # Get recommended intent
        intent = self.get_intent_for_type(target_entity_type)

        logger.debug(
            f"Searching for {target_entity_type}: "
            f"intent={intent}, weights={weights}"
        )

        # Execute search with entity-specific configuration
        results = self._retrieval.hybrid_search(
            query=query,
            intent=intent,
            top_k=top_k * 2,  # Get more, then filter
            vector_weight=weights["vector"],
            graph_weight=weights["graph"],
            temporal_context=temporal_context,
        )

        # Filter to target type
        filtered = [
            r for r in results if r.entity_type == target_entity_type
        ]

        logger.debug(
            f"Entity type filter: {len(results)} -> {len(filtered)} "
            f"(target: {target_entity_type})"
        )

        return filtered[:top_k]

    def search_events(
        self,
        query: str,
        top_k: int = 10,
        temporal_context: Optional[TemporalQueryContext] = None,
    ) -> List[HybridSearchResult]:
        """Search for events with temporal awareness.

        Convenience method for Event search with temporal-optimized settings.

        Args:
            query: Natural language search query
            top_k: Maximum results to return
            temporal_context: Optional temporal context

        Returns:
            List of HybridSearchResult for Event entities
        """
        return self.search_by_entity_type(
            query=query,
            target_entity_type="Event",
            top_k=top_k,
            temporal_context=temporal_context,
        )

    def search_code(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[HybridSearchResult]:
        """Search for code entities using CodeBERT embeddings.

        Convenience method for CodeEntity search with code-optimized settings.

        Args:
            query: Code-related search query
            top_k: Maximum results to return

        Returns:
            List of HybridSearchResult for CodeEntity
        """
        return self.search_by_entity_type(
            query=query,
            target_entity_type="CodeEntity",
            top_k=top_k,
        )

    def search_documents(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[HybridSearchResult]:
        """Search for documents with document embeddings.

        Convenience method for Document search with content-optimized settings.

        Args:
            query: Document search query
            top_k: Maximum results to return

        Returns:
            List of HybridSearchResult for Document entities
        """
        return self.search_by_entity_type(
            query=query,
            target_entity_type="Document",
            top_k=top_k,
        )

    def search_entities(
        self,
        query: str,
        entity_type: str,
        top_k: int = 10,
    ) -> List[HybridSearchResult]:
        """Search for static entities (Person, Organization, Concept).

        Convenience method for static entity search with balanced settings.

        Args:
            query: Entity search query
            entity_type: One of Person, Organization, Concept
            top_k: Maximum results to return

        Returns:
            List of HybridSearchResult for specified entity type
        """
        return self.search_by_entity_type(
            query=query,
            target_entity_type=entity_type,
            top_k=top_k,
        )
