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
