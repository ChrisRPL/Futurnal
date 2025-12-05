"""Embedding Request Model for Multi-Model Architecture.

Provides validated request models for embedding generation with
Option B temporal-first enforcement.

Option B Compliance:
- Temporal context REQUIRED for Event type
- Validation enforces temporal-first design
- Fails fast if temporal context missing for temporal entities

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/02-multi-model-architecture.md
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from futurnal.embeddings.models import TemporalEmbeddingContext


# Entity types that require temporal context (Option B compliance)
TEMPORAL_ENTITY_TYPES = {"Event"}


class EmbeddingRequest(BaseModel):
    """Request for embedding generation.

    Validates that temporal entities have required context.
    This is a critical Option B enforcement point.

    Attributes:
        entity_type: Entity type (Person, Organization, Concept, Event, CodeEntity, Document)
        content: Content to embed (must be non-empty)
        entity_id: Optional PKG entity ID for tracking and provenance
        entity_name: Optional entity name for metadata
        temporal_context: Temporal context (REQUIRED for Event type)
        metadata: Additional metadata to include in result

    Option B Compliance:
        - Temporal context REQUIRED for Event type
        - Validation enforces temporal-first design
        - Explicit error messages guide correct usage

    Example:
        # Static entity request (no temporal context needed)
        person_request = EmbeddingRequest(
            entity_type="Person",
            content="John Smith, Software Engineer at Futurnal",
            entity_name="John Smith",
        )

        # Event request (temporal context REQUIRED)
        event_request = EmbeddingRequest(
            entity_type="Event",
            content="Team Meeting: Quarterly planning discussion",
            entity_name="Team Meeting",
            temporal_context=TemporalEmbeddingContext(
                timestamp=datetime(2024, 1, 15, 14, 30),
                duration=timedelta(hours=2),
            ),
        )

    Raises:
        ValueError: If Event type without temporal_context
        ValueError: If content is empty
    """

    entity_type: str = Field(
        ...,
        description="Entity type: Person, Organization, Concept, Event, CodeEntity, Document",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Content to embed (must be non-empty)",
    )
    entity_id: Optional[str] = Field(
        default=None,
        description="PKG entity ID for tracking and provenance",
    )
    entity_name: Optional[str] = Field(
        default=None,
        description="Entity name for metadata",
    )
    temporal_context: Optional[TemporalEmbeddingContext] = Field(
        default=None,
        description="Temporal context (REQUIRED for Event type)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to include in result",
    )

    @model_validator(mode="after")
    def validate_temporal_requirement(self) -> "EmbeddingRequest":
        """Enforce temporal context for Event type (Option B compliance).

        This is a critical validation point for Option B temporal-first design.
        Events MUST have temporal context to enable:
        - Phase 2 correlation detection
        - Phase 3 causal inference
        - Temporal similarity computations

        Raises:
            ValueError: If Event type without temporal_context
        """
        if self.entity_type in TEMPORAL_ENTITY_TYPES and self.temporal_context is None:
            raise ValueError(
                f"temporal_context is REQUIRED for {self.entity_type} entity type. "
                "Option B temporal-first design requires all temporal entities "
                "(Events) to have temporal context for correlation detection "
                "and causal inference. Please provide a TemporalEmbeddingContext "
                "with at minimum a timestamp."
            )
        return self

    @property
    def requires_temporal_context(self) -> bool:
        """Check if this entity type requires temporal context."""
        return self.entity_type in TEMPORAL_ENTITY_TYPES

    @property
    def has_temporal_context(self) -> bool:
        """Check if temporal context is provided."""
        return self.temporal_context is not None

    @property
    def has_causal_context(self) -> bool:
        """Check if causal chain is provided in temporal context."""
        if self.temporal_context is None:
            return False
        return len(self.temporal_context.causal_chain) > 0

    def get_effective_name(self) -> str:
        """Get entity name, falling back to content prefix if not set.

        Returns:
            Entity name or first 50 chars of content
        """
        if self.entity_name:
            return self.entity_name
        # Use first 50 chars of content as fallback
        return self.content[:50].strip()

    model_config = {"frozen": True}


class BatchEmbeddingRequest(BaseModel):
    """Request for batch embedding generation.

    Wraps multiple EmbeddingRequest instances for efficient processing.

    Attributes:
        requests: List of individual embedding requests
        fail_fast: If True, stop on first error. If False, continue and collect errors.

    Example:
        batch = BatchEmbeddingRequest(
            requests=[
                EmbeddingRequest(entity_type="Person", content="John Smith"),
                EmbeddingRequest(
                    entity_type="Event",
                    content="Team Meeting",
                    temporal_context=TemporalEmbeddingContext(timestamp=datetime.now()),
                ),
            ],
            fail_fast=False,
        )
    """

    requests: List[EmbeddingRequest] = Field(
        ...,
        min_length=1,
        description="List of embedding requests",
    )
    fail_fast: bool = Field(
        default=True,
        description="Stop on first error if True, continue and collect errors if False",
    )

    @property
    def entity_types(self) -> List[str]:
        """Get unique entity types in this batch."""
        return list(set(r.entity_type for r in self.requests))

    @property
    def size(self) -> int:
        """Get number of requests in batch."""
        return len(self.requests)

    def group_by_entity_type(self) -> Dict[str, List[EmbeddingRequest]]:
        """Group requests by entity type for efficient batch processing.

        Returns:
            Dict mapping entity_type to list of requests
        """
        grouped: Dict[str, List[EmbeddingRequest]] = {}
        for request in self.requests:
            if request.entity_type not in grouped:
                grouped[request.entity_type] = []
            grouped[request.entity_type].append(request)
        return grouped

    model_config = {"frozen": True}
