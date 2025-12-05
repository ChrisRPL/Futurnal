"""Temporal-Aware Embeddings Module for Futurnal.

This module provides temporal-aware embedding generation and management for the
Vector Embedding Service. It implements distinct embedding strategies for:

- **Temporal Events**: Embeddings that preserve temporal context (timestamp,
  duration, causal relationships) for Phase 2 correlation detection and
  Phase 3 causal inference.

- **Static Entities**: Standard semantic embeddings for Person, Organization,
  and Concept entities without temporal context.

- **Event Sequences**: Sequence embeddings for causal pattern matching.

- **Schema-Versioned Storage**: Embeddings tracked by PKG schema version with
  automatic re-embedding triggers when schema evolves.

Option B Compliance:
- Ghost model FROZEN (pre-trained models, no fine-tuning)
- Temporal-first design (timestamp REQUIRED for events)
- Schema versioned (embeddings tagged with PKG schema version)
- Schema evolution support (re-embedding on schema changes)
- Causal structure prepared (embeddings optimized for Phase 3)

Production Plan References:
- docs/phase-1/vector-embedding-service-production-plan/01-temporal-aware-embeddings.md
- docs/phase-1/vector-embedding-service-production-plan/02-multi-model-architecture.md
- docs/phase-1/vector-embedding-service-production-plan/03-schema-versioned-storage.md

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
    BatchProcessingError,
    EmbeddingError,
    EmbeddingGenerationError,
    FusionError,
    ModelLoadError,
    ModelNotFoundError,
    ReembeddingError,
    RoutingError,
    SchemaVersionError,
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
from futurnal.embeddings.metrics import EmbeddingMetrics, ModelMetrics
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingMetadata,
    EmbeddingQuery,
    EmbeddingResult,
    FusionWeights,
    SimilarityResult,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.reembedding import (
    ReembeddingProgress,
    ReembeddingService,
    SchemaChangeDetection,
)
from futurnal.embeddings.registry import ModelRegistry, RegisteredModel
from futurnal.embeddings.request import BatchEmbeddingRequest, EmbeddingRequest
from futurnal.embeddings.router import ModelRouter
from futurnal.embeddings.schema_versioned_store import SchemaVersionedEmbeddingStore
from futurnal.embeddings.service import MultiModelEmbeddingService
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
    # Multi-Model Architecture (Module 02)
    "ModelRegistry",
    "RegisteredModel",
    "ModelRouter",
    "MultiModelEmbeddingService",
    "EmbeddingRequest",
    "BatchEmbeddingRequest",
    "EmbeddingMetrics",
    "ModelMetrics",
    # Schema-Versioned Storage (Module 03)
    "SchemaVersionedEmbeddingStore",
    "ReembeddingService",
    "SchemaChangeDetection",
    "ReembeddingProgress",
    "EmbeddingMetadata",
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
    "RoutingError",
    "BatchProcessingError",
    "SchemaVersionError",
    "ReembeddingError",
]
