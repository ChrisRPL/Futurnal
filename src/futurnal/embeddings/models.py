"""Embedding data models for Temporal-Aware Embeddings Module.

This module defines all Pydantic models and enums for the embedding service:
- Entity type classification for embedding strategy selection
- Temporal context for event embeddings
- Fusion configuration for weighted embedding combination
- Embedding results with metadata

Option B Compliance:
- Ghost model FROZEN (no fine-tuning, models used as-is)
- Temporal-first design (timestamp REQUIRED for temporal entities)
- Schema versioned (embeddings tagged with model version)

Implementation follows production plan:
docs/phase-1/vector-embedding-service-production-plan/01-temporal-aware-embeddings.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator


class EmbeddingEntityType(str, Enum):
    """Types of entities requiring different embedding strategies.

    Different entity types have fundamentally different embedding needs:
    - Static entities focus on semantic content
    - Temporal events include temporal context
    - Code entities may use specialized models
    """

    STATIC_ENTITY = "static_entity"  # Person, Organization, Concept
    TEMPORAL_EVENT = "temporal_event"  # Event with timestamp
    TEMPORAL_RELATIONSHIP = "temporal_relationship"  # BEFORE/AFTER/CAUSES
    CODE_ENTITY = "code_entity"  # Code snippets (future enhancement)
    DOCUMENT = "document"  # Full documents


class TemporalEmbeddingContext(BaseModel):
    """Temporal context for event embeddings.

    Preserves temporal semantics critical for:
    - Phase 2 correlation detection
    - Phase 3 causal inference
    - Temporal similarity search

    Option B Compliance:
    - timestamp is REQUIRED (temporal-first design)
    - Supports causal chain encoding for Phase 3 preparation
    """

    timestamp: datetime = Field(
        ...,
        description="When the event occurred (REQUIRED for Option B compliance)",
    )
    duration: Optional[timedelta] = Field(
        default=None,
        description="How long the event lasted",
    )
    temporal_type: Optional[str] = Field(
        default=None,
        description="Temporal relationship type (BEFORE/AFTER/DURING/CAUSES)",
    )
    event_sequence: List[str] = Field(
        default_factory=list,
        description="IDs of related events in temporal sequence",
    )
    causal_chain: List[str] = Field(
        default_factory=list,
        description="IDs of causal predecessors/successors (Phase 3 prep)",
    )
    temporal_neighbors: List[str] = Field(
        default_factory=list,
        description="IDs of events within temporal window",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timestamp(cls, v: Any) -> datetime:
        """Validate timestamp is present (Option B requirement)."""
        if v is None:
            raise ValueError(
                "timestamp is REQUIRED for TemporalEmbeddingContext "
                "(Option B: temporal-first design)"
            )
        return v

    def format_for_embedding(self) -> str:
        """Format temporal context as natural language for embedding.

        Returns text representation suitable for embedding models.
        Example: "occurred on 2024-01-15T14:30:00, lasted 2 hours,
                 temporal relationship: BEFORE"
        """
        parts = [f"occurred on {self.timestamp.isoformat()}"]

        if self.duration:
            hours = self.duration.total_seconds() / 3600
            if hours >= 1:
                parts.append(f"lasted {hours:.1f} hours")
            else:
                minutes = self.duration.total_seconds() / 60
                parts.append(f"lasted {minutes:.0f} minutes")

        if self.temporal_type:
            parts.append(f"temporal relationship: {self.temporal_type}")

        return ", ".join(parts)

    def format_causal_chain(self) -> str:
        """Format causal chain as natural language for embedding.

        Returns text representation of causal context.
        Example: "causal context: Meeting -> Decision -> Publication"
        """
        if not self.causal_chain:
            return ""
        return f"causal context: {' -> '.join(self.causal_chain)}"


class FusionWeights(BaseModel):
    """Configurable weights for embedding fusion.

    Default strategy (from production plan):
    - Content: 60% (what happened)
    - Temporal: 30% (when it happened)
    - Causal: 10% (what led to it)

    Weights must sum to 1.0.
    """

    content_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for content embedding (what happened)",
    )
    temporal_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for temporal embedding (when it happened)",
    )
    causal_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for causal embedding (what led to it)",
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "FusionWeights":
        """Ensure weights sum to 1.0."""
        total = self.content_weight + self.temporal_weight + self.causal_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Fusion weights must sum to 1.0, got {total:.3f}")
        return self


class EmbeddingResult(BaseModel):
    """Result of embedding generation.

    Contains the embedding vector along with metadata for:
    - Model version tracking (schema versioning)
    - Performance metrics (generation time)
    - Context flags (temporal/causal encoded)
    """

    embedding: Sequence[float] = Field(
        ...,
        description="Generated embedding vector",
    )
    entity_type: EmbeddingEntityType = Field(
        ...,
        description="Type of entity that was embedded",
    )
    model_version: str = Field(
        ...,
        description="Version of model used (for schema tracking)",
    )
    embedding_dimension: int = Field(
        ...,
        description="Dimension of embedding vector",
    )
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate embedding in milliseconds",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the embedding",
    )

    # Temporal context flags
    temporal_context_encoded: bool = Field(
        default=False,
        description="Whether temporal context was included in embedding",
    )
    causal_context_encoded: bool = Field(
        default=False,
        description="Whether causal context was included in embedding",
    )

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def set_dimension_from_embedding(cls, v: Any, info: Any) -> int:
        """Auto-set dimension from embedding if not provided."""
        if v is None and "embedding" in info.data:
            return len(info.data["embedding"])
        return v


class EmbeddingQuery(BaseModel):
    """Query for similarity search in embedding space.

    Supports both vector-based and text-based queries.
    """

    query_embedding: Optional[Sequence[float]] = Field(
        default=None,
        description="Embedding vector for similarity search",
    )
    query_text: Optional[str] = Field(
        default=None,
        description="Text to embed for similarity search",
    )
    entity_type_filter: Optional[EmbeddingEntityType] = Field(
        default=None,
        description="Filter results by entity type",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    min_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold",
    )

    @model_validator(mode="after")
    def validate_query_input(self) -> "EmbeddingQuery":
        """Ensure either embedding or text is provided."""
        if self.query_embedding is None and self.query_text is None:
            raise ValueError("Either query_embedding or query_text must be provided")
        return self


class SimilarityResult(BaseModel):
    """Result from similarity search."""

    entity_id: str = Field(
        ...,
        description="ID of the matching entity",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score",
    )
    entity_type: EmbeddingEntityType = Field(
        ...,
        description="Type of the matching entity",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entity metadata",
    )
