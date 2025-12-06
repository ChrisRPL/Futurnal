"""Query Embedding Router.

Routes query embedding requests to appropriate models based on intent.
This is DIFFERENT from ModelRouter in embeddings/ which routes ENTITY embeddings.

Key difference:
- ModelRouter: Routes by entity TYPE (Person -> instructor-large, Event -> temporal-aware)
- QueryEmbeddingRouter: Routes by query INTENT (temporal -> temporal-aware, code -> CodeBERT)

The same query might use different embedding strategies depending on
what the user is looking for.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md

Option B Compliance:
- Uses frozen pre-trained models (no fine-tuning)
- Temporal queries use temporal-aware embeddings
- Integrates with existing MultiModelEmbeddingService
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from futurnal.search.hybrid.config import HybridSearchConfig
from futurnal.search.hybrid.exceptions import QueryRoutingError
from futurnal.search.hybrid.types import QueryEmbeddingType, TemporalQueryContext

if TYPE_CHECKING:
    from futurnal.embeddings.models import TemporalEmbeddingContext
    from futurnal.embeddings.service import MultiModelEmbeddingService

logger = logging.getLogger(__name__)


class QueryEmbeddingRouter:
    """Routes query embedding requests to appropriate models.

    This router determines which embedding model and strategy to use
    based on the query intent and content. Unlike entity embedding
    which is determined by entity type, query embedding is determined
    by what the user is searching FOR.

    Routing Rules:
    1. Intent-based (takes precedence):
       - temporal intent -> TEMPORAL embedding
       - causal intent -> TEMPORAL embedding (causal needs temporal context)

    2. Content-based detection:
       - Code keywords (function, class, bug) -> CODE embedding
       - Document keywords (file, note, paper) -> DOCUMENT embedding
       - Default -> GENERAL embedding

    Integration:
        Uses MultiModelEmbeddingService for actual embedding generation.
        The router determines which entity_type to request embeddings for
        based on query analysis.

    Example:
        >>> from futurnal.embeddings.service import MultiModelEmbeddingService
        >>> from futurnal.search.hybrid import QueryEmbeddingRouter, HybridSearchConfig

        >>> service = MultiModelEmbeddingService()
        >>> router = QueryEmbeddingRouter(service)

        >>> # Temporal query
        >>> query_type = router.determine_query_type(
        ...     "What happened in January?",
        ...     intent="temporal"
        ... )
        >>> assert query_type == QueryEmbeddingType.TEMPORAL

        >>> # Code query (detected from content)
        >>> query_type = router.determine_query_type(
        ...     "Find the authentication function",
        ...     intent="exploratory"
        ... )
        >>> assert query_type == QueryEmbeddingType.CODE
    """

    # Entity type mapping for embedding generation
    # QueryEmbeddingType -> entity_type to use with MultiModelEmbeddingService
    QUERY_TO_ENTITY_TYPE = {
        QueryEmbeddingType.GENERAL: "Concept",  # Use general semantic embeddings
        QueryEmbeddingType.TEMPORAL: "Event",  # Use temporal-aware embeddings
        QueryEmbeddingType.CAUSAL: "Event",  # Causal also needs temporal context
        QueryEmbeddingType.CODE: "CodeEntity",  # Use CodeBERT embeddings
        QueryEmbeddingType.DOCUMENT: "Document",  # Use long-context embeddings
    }

    # Query instructions for different embedding types
    QUERY_INSTRUCTIONS = {
        QueryEmbeddingType.GENERAL: "Represent the search query for retrieving relevant entities:",
        QueryEmbeddingType.TEMPORAL: "Represent the temporal query for retrieving time-relevant events:",
        QueryEmbeddingType.CAUSAL: "Represent the causal query for retrieving events in causal chains:",
        QueryEmbeddingType.CODE: "Represent the code search query for retrieving relevant code:",
        QueryEmbeddingType.DOCUMENT: "Represent the query for retrieving relevant documents:",
    }

    def __init__(
        self,
        embedding_service: "MultiModelEmbeddingService",
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """Initialize the query embedding router.

        Args:
            embedding_service: MultiModelEmbeddingService instance for embedding generation
            config: Hybrid search configuration (uses defaults if None)
        """
        self._embedding_service = embedding_service
        self._config = config or HybridSearchConfig()

        logger.info(
            "Initialized QueryEmbeddingRouter with "
            f"{len(self._config.code_keywords)} code keywords, "
            f"{len(self._config.document_keywords)} document keywords"
        )

    @property
    def config(self) -> HybridSearchConfig:
        """Get the configuration."""
        return self._config

    def determine_query_type(
        self,
        query: str,
        intent: str,
    ) -> QueryEmbeddingType:
        """Determine query embedding type based on query and intent.

        The embedding type determines which model and strategy to use
        for generating the query embedding.

        Priority:
        1. Explicit intent (temporal, causal) overrides content detection
        2. Content-based detection for code/document keywords
        3. Default to GENERAL

        Args:
            query: Natural language search query
            intent: Search intent: temporal, causal, lookup, exploratory

        Returns:
            QueryEmbeddingType indicating which embedding strategy to use
        """
        intent_lower = intent.lower()

        # Intent-based routing (takes precedence)
        if intent_lower == "temporal":
            logger.debug(f"Intent-based routing: temporal -> TEMPORAL")
            return QueryEmbeddingType.TEMPORAL

        if intent_lower == "causal":
            # Causal queries need temporal context to understand causal chains
            logger.debug(f"Intent-based routing: causal -> TEMPORAL (causal needs temporal)")
            return QueryEmbeddingType.TEMPORAL

        # Content-based detection
        query_lower = query.lower()

        # Code detection
        if self._has_code_indicators(query_lower):
            logger.debug(f"Content-based routing: code keywords detected -> CODE")
            return QueryEmbeddingType.CODE

        # Document detection
        if self._has_document_indicators(query_lower):
            logger.debug(f"Content-based routing: document keywords detected -> DOCUMENT")
            return QueryEmbeddingType.DOCUMENT

        # Temporal detection (secondary, after explicit intent)
        if self._has_temporal_indicators(query_lower):
            logger.debug(f"Content-based routing: temporal keywords detected -> TEMPORAL")
            return QueryEmbeddingType.TEMPORAL

        # Default to general
        logger.debug(f"Default routing -> GENERAL")
        return QueryEmbeddingType.GENERAL

    def embed_query(
        self,
        query: str,
        query_type: QueryEmbeddingType,
        temporal_context: Optional[TemporalQueryContext] = None,
    ) -> List[float]:
        """Embed query using appropriate model and strategy.

        Generates embedding for the query using the model and instruction
        appropriate for the query type.

        Args:
            query: Natural language search query
            query_type: Embedding type determined by determine_query_type()
            temporal_context: Optional temporal context for temporal/causal queries

        Returns:
            List of floats representing the query embedding vector

        Raises:
            QueryRoutingError: If embedding generation fails
        """
        try:
            # Augment query with temporal context if provided
            augmented_query = self._augment_query(query, query_type, temporal_context)

            # Get entity type for embedding service
            entity_type = self.QUERY_TO_ENTITY_TYPE[query_type]

            # Build temporal context for embedding service if needed
            embedding_temporal_context = None
            if query_type in (QueryEmbeddingType.TEMPORAL, QueryEmbeddingType.CAUSAL):
                embedding_temporal_context = self._build_embedding_temporal_context(
                    temporal_context
                )

            # Generate embedding using the embedding service
            result = self._embedding_service.embed(
                entity_type=entity_type,
                content=augmented_query,
                temporal_context=embedding_temporal_context,
                metadata={"query_type": query_type.value},
            )

            logger.debug(
                f"Generated {query_type.value} embedding: "
                f"dim={result.embedding_dimension}"
            )

            return list(result.embedding)

        except Exception as e:
            raise QueryRoutingError(
                f"Failed to generate {query_type.value} embedding for query: {e}"
            ) from e

    def embed_query_with_type_detection(
        self,
        query: str,
        intent: str,
        temporal_context: Optional[TemporalQueryContext] = None,
    ) -> tuple[List[float], QueryEmbeddingType]:
        """Determine query type and generate embedding in one call.

        Convenience method that combines determine_query_type() and embed_query().

        Args:
            query: Natural language search query
            intent: Search intent
            temporal_context: Optional temporal context

        Returns:
            Tuple of (embedding vector, query embedding type)
        """
        query_type = self.determine_query_type(query, intent)
        embedding = self.embed_query(query, query_type, temporal_context)
        return embedding, query_type

    def _has_code_indicators(self, query_lower: str) -> bool:
        """Check if query contains code-related keywords."""
        return any(kw in query_lower for kw in self._config.code_keywords)

    def _has_document_indicators(self, query_lower: str) -> bool:
        """Check if query contains document-related keywords."""
        return any(kw in query_lower for kw in self._config.document_keywords)

    def _has_temporal_indicators(self, query_lower: str) -> bool:
        """Check if query contains temporal-related keywords."""
        return any(kw in query_lower for kw in self._config.temporal_keywords)

    def _augment_query(
        self,
        query: str,
        query_type: QueryEmbeddingType,
        temporal_context: Optional[TemporalQueryContext],
    ) -> str:
        """Augment query with instruction and context.

        Prepares the query text for embedding by adding:
        - Type-specific instruction prefix
        - Temporal context (if provided and relevant)

        Args:
            query: Original query text
            query_type: Embedding type
            temporal_context: Optional temporal context

        Returns:
            Augmented query string
        """
        # Get instruction for this query type
        instruction = self.QUERY_INSTRUCTIONS.get(
            query_type,
            self.QUERY_INSTRUCTIONS[QueryEmbeddingType.GENERAL],
        )

        # Build augmented query
        parts = [instruction, query]

        # Add temporal context for temporal/causal queries
        if temporal_context and query_type in (
            QueryEmbeddingType.TEMPORAL,
            QueryEmbeddingType.CAUSAL,
        ):
            temporal_str = temporal_context.format_for_embedding()
            if temporal_str:
                parts.append(f"[{temporal_str}]")

        return " ".join(parts)

    def _build_embedding_temporal_context(
        self,
        temporal_context: Optional[TemporalQueryContext],
    ) -> Optional["TemporalEmbeddingContext"]:
        """Build TemporalEmbeddingContext for embedding service.

        Converts TemporalQueryContext to the format expected by
        MultiModelEmbeddingService.

        Args:
            temporal_context: Query temporal context

        Returns:
            TemporalEmbeddingContext or None
        """
        from futurnal.embeddings.models import TemporalEmbeddingContext

        if temporal_context is None:
            # Create minimal temporal context for Event embedding
            # (required by temporal-first design)
            return TemporalEmbeddingContext(
                timestamp=datetime.utcnow(),
            )

        # Use reference timestamp or current time
        timestamp = temporal_context.reference_timestamp or datetime.utcnow()

        return TemporalEmbeddingContext(
            timestamp=timestamp,
            duration=temporal_context.time_window,
            temporal_type=temporal_context.temporal_relation,
        )
