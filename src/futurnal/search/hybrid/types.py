"""Hybrid Search Types and Models.

Core type definitions for schema-aware hybrid retrieval:
- QueryEmbeddingType: Query embedding strategies based on intent
- VectorSearchResult: Result from vector similarity search
- GraphSearchResult: Result from graph traversal
- HybridSearchResult: Combined result from hybrid retrieval
- HybridSearchQuery: Input model for hybrid search requests
- TemporalQueryContext: Temporal context for query augmentation

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md

Option B Compliance:
- Temporal context support for temporal-first design
- Schema version tracking in results
- Support for causal and temporal intent
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryEmbeddingType(str, Enum):
    """Query embedding strategies based on intent.

    Determines which embedding model and strategy to use for query embedding.
    This is different from entity embedding types - queries have different
    semantic needs than stored entities.

    Values:
        GENERAL: Standard entity lookup using general-purpose embeddings
        TEMPORAL: Time-aware queries using temporal-enhanced embeddings
        CAUSAL: Causal chain queries (uses temporal embeddings + causal context)
        CODE: Code-related queries using CodeBERT embeddings
        DOCUMENT: Full document search using long-context embeddings
    """

    GENERAL = "general"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CODE = "code"
    DOCUMENT = "document"


class TemporalQueryContext(BaseModel):
    """Temporal context for query augmentation.

    Used to enhance query embeddings with temporal information
    for more accurate temporal-aware retrieval.

    Option B Compliance:
    - Supports temporal-first design
    - Enables time-range filtering
    - Captures temporal relationships
    """

    time_range_start: Optional[datetime] = Field(
        default=None,
        description="Start of time range filter",
    )
    time_range_end: Optional[datetime] = Field(
        default=None,
        description="End of time range filter",
    )
    reference_timestamp: Optional[datetime] = Field(
        default=None,
        description="Reference timestamp for relative queries",
    )
    temporal_relation: Optional[str] = Field(
        default=None,
        description="Temporal relation type: BEFORE, AFTER, DURING, SIMULTANEOUS",
    )
    time_window: Optional[timedelta] = Field(
        default=None,
        description="Time window around reference for neighborhood queries",
    )

    def format_for_embedding(self) -> str:
        """Format temporal context for embedding augmentation.

        Returns string representation to append to query for
        temporal-aware embedding generation.
        """
        parts = []

        if self.time_range_start and self.time_range_end:
            parts.append(
                f"Time: {self.time_range_start.isoformat()} to "
                f"{self.time_range_end.isoformat()}"
            )
        elif self.reference_timestamp:
            parts.append(f"Reference: {self.reference_timestamp.isoformat()}")

        if self.temporal_relation:
            parts.append(f"Relation: {self.temporal_relation}")

        if self.time_window:
            parts.append(f"Window: {self.time_window.total_seconds()} seconds")

        return " | ".join(parts) if parts else ""


class VectorSearchResult(BaseModel):
    """Result from vector similarity search.

    Contains entity information and similarity score from
    embedding-based retrieval.

    Attributes:
        entity_id: PKG entity identifier
        entity_type: Entity type (Person, Event, Document, etc.)
        content: Text content or summary
        similarity_score: Cosine similarity score (0-1)
        schema_version: PKG schema version when embedding was created
        metadata: Additional metadata from embedding store
    """

    entity_id: str = Field(description="PKG entity identifier")
    entity_type: str = Field(description="Entity type")
    content: str = Field(default="", description="Text content or summary")
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score",
    )
    schema_version: int = Field(
        default=1,
        description="Schema version when embedding was created",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @field_validator("similarity_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure similarity score is in valid range."""
        return max(0.0, min(1.0, v))


class GraphSearchResult(BaseModel):
    """Result from graph traversal.

    Contains entity information and path details from
    graph-based expansion.

    Attributes:
        entity_id: PKG entity identifier
        entity_type: Entity type (Person, Event, Document, etc.)
        path_from_seed: List of entity IDs in traversal path
        path_score: Score based on path length and confidence
        relationship_types: Types of relationships in path
        metadata: Additional metadata from graph query
    """

    entity_id: str = Field(description="PKG entity identifier")
    entity_type: str = Field(description="Entity type")
    path_from_seed: List[str] = Field(
        default_factory=list,
        description="Entity IDs in traversal path",
    )
    path_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Path-based relevance score",
    )
    relationship_types: List[str] = Field(
        default_factory=list,
        description="Relationship types in path",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class GraphContext(BaseModel):
    """Graph traversal context for search results.

    Contains information about how a result was discovered through
    graph traversal, including related entities and relationship paths.

    Per GFM-RAG paper (2502.01113v1):
    - Shows "why" a result is relevant via graph connections
    - Enables path visualization for user understanding
    - Supports multi-hop reasoning explanation
    """

    related_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Entities connected via graph traversal",
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relationships traversed to reach this result",
    )
    path_to_query: List[str] = Field(
        default_factory=list,
        description="Path from query entity to this result",
    )
    hop_count: int = Field(
        default=0,
        ge=0,
        description="Number of hops from seed entities",
    )
    path_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the traversal path",
    )


class HybridSearchResult(BaseModel):
    """Combined result from hybrid retrieval.

    Fuses vector similarity and graph traversal results
    into a single scored result.

    Attributes:
        entity_id: PKG entity identifier
        entity_type: Entity type
        vector_score: Score from vector similarity (0 if not found)
        graph_score: Score from graph traversal (0 if not found)
        combined_score: Weighted combination of vector and graph scores
        source: Where result originated: "vector", "graph", or "hybrid"
        content: Text content or summary (if available)
        schema_version: Schema version (from vector result if available)
        metadata: Combined metadata from both sources
        graph_context: Graph traversal context (per GFM-RAG paper)
    """

    entity_id: str = Field(description="PKG entity identifier")
    entity_type: str = Field(description="Entity type")
    vector_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Vector similarity score",
    )
    graph_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Graph traversal score",
    )
    combined_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted combined score",
    )
    source: str = Field(
        default="hybrid",
        description="Result source: vector, graph, or hybrid",
    )
    content: str = Field(
        default="",
        description="Text content or summary",
    )
    schema_version: Optional[int] = Field(
        default=None,
        description="Schema version from embedding",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Combined metadata",
    )
    graph_context: Optional[GraphContext] = Field(
        default=None,
        description="Graph traversal context showing path and related entities",
    )


class HybridSearchQuery(BaseModel):
    """Input model for hybrid search requests.

    Encapsulates all parameters for a hybrid search operation.

    Attributes:
        query_text: Natural language search query
        intent: Search intent affecting retrieval strategy
        top_k: Maximum number of results to return
        vector_weight: Weight for vector similarity (0-1)
        graph_weight: Weight for graph traversal (0-1)
        temporal_context: Optional temporal context for time-aware queries
        entity_type_filter: Optional filter to specific entity type
        min_schema_version: Minimum schema version for embeddings
    """

    query_text: str = Field(
        min_length=1,
        description="Natural language search query",
    )
    intent: str = Field(
        default="exploratory",
        description="Search intent: temporal, causal, lookup, exploratory",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum results to return",
    )
    vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity",
    )
    graph_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for graph traversal",
    )
    temporal_context: Optional[TemporalQueryContext] = Field(
        default=None,
        description="Temporal context for time-aware queries",
    )
    entity_type_filter: Optional[str] = Field(
        default=None,
        description="Filter to specific entity type",
    )
    min_schema_version: Optional[int] = Field(
        default=None,
        description="Minimum schema version for embeddings",
    )

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        """Validate intent is a known value."""
        valid_intents = {"temporal", "causal", "lookup", "exploratory"}
        if v.lower() not in valid_intents:
            raise ValueError(
                f"Invalid intent '{v}'. Must be one of: {valid_intents}"
            )
        return v.lower()

    def model_post_init(self, __context: Any) -> None:
        """Normalize weights after initialization."""
        # Normalize weights to sum to 1
        total = self.vector_weight + self.graph_weight
        if total > 0 and abs(total - 1.0) > 0.001:
            self.vector_weight = self.vector_weight / total
            self.graph_weight = self.graph_weight / total


class SchemaCompatibilityResult(BaseModel):
    """Result of schema version compatibility check.

    Attributes:
        compatible: Whether embedding is compatible
        score_factor: Multiplier for similarity score (0-1)
        transform_required: Whether transformation is needed
        reembedding_required: Whether re-embedding is needed
        drift_level: Drift severity: none, minor, moderate, severe
        version_diff: Difference between versions
    """

    compatible: bool = Field(description="Whether embedding is compatible")
    score_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Score multiplier based on drift",
    )
    transform_required: bool = Field(
        default=False,
        description="Whether transformation is needed",
    )
    reembedding_required: bool = Field(
        default=False,
        description="Whether re-embedding is needed",
    )
    drift_level: str = Field(
        default="none",
        description="Drift severity: none, minor, moderate, severe",
    )
    version_diff: int = Field(
        default=0,
        description="Version difference",
    )


# =============================================================================
# Query Routing Types (Module 04)
# =============================================================================
# Production Plan Reference:
# docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md


class QueryIntent(str, Enum):
    """Query intent classification for routing.

    Determines which retrieval strategy to use based on query semantics.
    LLM-based intent classification routes queries to optimal strategies.

    Values:
        LOOKUP: Specific entity/fact lookup (e.g., "What is X?", "Who is Y?")
        EXPLORATORY: Broad exploration (e.g., "Tell me about...", "What do I know...")
        TEMPORAL: Time-based queries (e.g., "when", "before", "after", dates)
        CAUSAL: Causation queries (e.g., "why", "what caused", "what led to")

    Option B Compliance:
    - Temporal-first design: TEMPORAL queries prioritized
    - Supports causal inference preparation for Phase 3
    """

    LOOKUP = "lookup"
    EXPLORATORY = "exploratory"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


class QueryPlan(BaseModel):
    """Execution plan for a query.

    Generated by QueryRouter after intent classification.
    Defines strategies and weights for multi-strategy retrieval.

    Attributes:
        query_id: Unique identifier for this query
        original_query: The original query text
        intent: Classified intent type
        intent_confidence: Confidence score for intent classification (0-1)
        primary_strategy: Main retrieval strategy to use
        secondary_strategy: Optional fallback/supplementary strategy
        weights: Strategy weights for result fusion
        estimated_latency_ms: Estimated execution time
        created_at: When the plan was created
        detected_language: Detected query language (e.g., "pl" for Polish)
        selected_model: LLM model selected for this query

    Production Plan Reference:
    docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md
    """

    query_id: str = Field(description="Unique query identifier")
    original_query: str = Field(description="Original query text")
    intent: QueryIntent = Field(description="Classified intent type")
    intent_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Intent classification confidence",
    )
    primary_strategy: str = Field(
        description="Primary retrieval strategy: temporal_query, causal_chain, hybrid_retrieval",
    )
    secondary_strategy: Optional[str] = Field(
        default=None,
        description="Secondary retrieval strategy",
    )
    weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Strategy weights for fusion (e.g., {'temporal': 0.7, 'hybrid': 0.3})",
    )
    estimated_latency_ms: int = Field(
        default=500,
        ge=0,
        description="Estimated execution time in milliseconds",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Plan creation timestamp",
    )
    detected_language: Optional[str] = Field(
        default=None,
        description="Detected language code (e.g., 'pl', 'en')",
    )
    selected_model: Optional[str] = Field(
        default=None,
        description="LLM model used for classification",
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights are valid (0-1 range)."""
        for key, weight in v.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Weight for '{key}' must be between 0 and 1")
        return v


class QueryResult(BaseModel):
    """Unified query result with provenance.

    Returned by QueryRouter after executing a QueryPlan.
    Contains entities, relationships, and context from all strategies.

    Attributes:
        result_id: Unique result identifier
        query_id: Reference to the QueryPlan
        entities: Retrieved entities with scores
        relationships: Retrieved relationships
        temporal_context: Temporal context if relevant
        causal_chain: Causal chain if relevant
        relevance_scores: Strategy-specific relevance scores
        provenance: Source document IDs for audit trail
        latency_ms: Actual execution time

    Option B Compliance:
    - Includes provenance for audit trail
    - Temporal and causal context for Phase 2/3 preparation
    """

    result_id: str = Field(description="Unique result identifier")
    query_id: str = Field(description="Reference to QueryPlan.query_id")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved entities with scores and metadata",
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved relationships",
    )
    temporal_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Temporal context from temporal queries",
    )
    causal_chain: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Causal chain from causal queries",
    )
    relevance_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Strategy-specific relevance scores",
    )
    provenance: List[str] = Field(
        default_factory=list,
        description="Source document IDs for audit trail",
    )
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Actual execution time in milliseconds",
    )
