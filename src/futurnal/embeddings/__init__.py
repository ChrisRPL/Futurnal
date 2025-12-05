"""Temporal-Aware Embeddings Module for Futurnal.

This module provides temporal-aware embedding generation and management for the
Vector Embedding Service. It implements distinct embedding strategies for:

- **Temporal Events**: Embeddings that preserve temporal context (timestamp,
  duration, causal relationships) for Phase 2 correlation detection and
  Phase 3 causal inference.

- **Static Entities**: Standard semantic embeddings for Person, Organization,
  and Concept entities without temporal context.

- **Event Sequences**: Sequence embeddings for causal pattern matching.

Option B Compliance:
- Ghost model FROZEN (pre-trained models, no fine-tuning)
- Temporal-first design (timestamp REQUIRED for events)
- Schema versioned (embeddings tagged with model version)
- Causal structure prepared (embeddings optimized for Phase 3)

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/01-temporal-aware-embeddings.md

Example Usage:
    from futurnal.embeddings import (
        TemporalEventEmbedder,
        StaticEntityEmbedder,
        TemporalEmbeddingContext,
        EmbeddingServiceConfig,
        ModelManager,
    )

    # Initialize
    config = EmbeddingServiceConfig()
    manager = ModelManager(config)

    # Embed a temporal event
    event_embedder = TemporalEventEmbedder(manager)
    context = TemporalEmbeddingContext(
        timestamp=datetime(2024, 1, 15, 14, 30),
        duration=timedelta(hours=2),
    )
    result = event_embedder.embed(
        event_name="Team Meeting",
        event_description="Quarterly planning discussion",
        temporal_context=context,
    )

    # Embed a static entity
    entity_embedder = StaticEntityEmbedder(manager)
    result = entity_embedder.embed(
        entity_type="Person",
        entity_name="John Smith",
        entity_description="Software Engineer",
    )
"""

from futurnal.embeddings.config import (
    EmbeddingServiceConfig,
    ModelConfig,
    ModelType,
    get_default_config,
    get_high_quality_config,
    get_lightweight_config,
)
from futurnal.embeddings.event_sequence import EventSequenceEmbedder
from futurnal.embeddings.exceptions import (
    EmbeddingError,
    EmbeddingGenerationError,
    FusionError,
    ModelLoadError,
    ModelNotFoundError,
    StorageError,
    TemporalContextError,
)
from futurnal.embeddings.fusion import (
    ConcatenationFusion,
    EmbeddingFusion,
    ProjectionFusion,
)
from futurnal.embeddings.integration import TemporalAwareVectorWriter
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingQuery,
    EmbeddingResult,
    FusionWeights,
    SimilarityResult,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.static_entity import StaticEntityEmbedder
from futurnal.embeddings.temporal_event import TemporalEventEmbedder

__all__ = [
    # Configuration
    "EmbeddingServiceConfig",
    "ModelConfig",
    "ModelType",
    "get_default_config",
    "get_lightweight_config",
    "get_high_quality_config",
    # Manager
    "ModelManager",
    # Embedders
    "TemporalEventEmbedder",
    "StaticEntityEmbedder",
    "EventSequenceEmbedder",
    # Fusion
    "EmbeddingFusion",
    "ConcatenationFusion",
    "ProjectionFusion",
    # Integration
    "TemporalAwareVectorWriter",
    # Models
    "EmbeddingEntityType",
    "TemporalEmbeddingContext",
    "FusionWeights",
    "EmbeddingResult",
    "EmbeddingQuery",
    "SimilarityResult",
    # Exceptions
    "EmbeddingError",
    "ModelLoadError",
    "ModelNotFoundError",
    "EmbeddingGenerationError",
    "FusionError",
    "TemporalContextError",
    "StorageError",
]
