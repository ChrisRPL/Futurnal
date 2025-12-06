"""Quality Evolution Module for Vector Embedding Service.

This module provides embedding quality tracking and evolution mechanisms
integrated with the experiential learning pipeline for continuous quality
improvement.

Components:
- **EmbeddingQualityMetrics**: Pydantic model for tracking quality metrics
- **QualityMetricsStore**: SQLite-backed persistence for trend analysis
- **QualityMetricsTracker**: Quality score computation and tracking
- **GoldenEmbeddingsManager**: Reference embeddings for quality comparison
- **EmbeddingQualityEvolution**: Integration with experiential learning
- **EmbeddingPerformanceProfiler**: Performance tracking and bottleneck detection

Option B Compliance:
- Embedding quality improves over time via experiential learning
- Re-embed low-quality extractions when extraction pipeline improves
- Integration with TrainingFreeGRPO/WorldStateModel
- No model parameter updates (frozen Ghost models)
- Quality tracked as metrics, not weights

Performance Targets:
- <2s single embedding latency
- >100/min batch throughput
- Low-quality identification >90% precision

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md

Example Usage:
    from futurnal.embeddings.quality import (
        EmbeddingQualityMetrics,
        QualityMetricsTracker,
        EmbeddingQualityEvolution,
        EmbeddingPerformanceProfiler,
    )

    # Initialize tracker with SQLite store
    store = QualityMetricsStore(db_path=Path("~/.futurnal/quality_metrics.db"))
    tracker = QualityMetricsTracker(store=store)

    # Record quality metrics for a new embedding
    metrics = tracker.record_embedding_quality(
        embedding_id="emb_123",
        entity_id="ent_456",
        entity_type="Person",
        embedding=embedding_vector,
        extraction_confidence=0.85,
        embedding_latency_ms=150.0,
        model_id="instructor-large-entity",
        vector_dimension=768,
    )

    # Identify low-quality embeddings for re-embedding
    low_quality = tracker.identify_low_quality_embeddings(
        min_quality_score=0.6,
        limit=100,
    )

    # Monitor for quality evolution
    evolution = EmbeddingQualityEvolution(
        quality_tracker=tracker,
        reembedding_service=reembedding_service,
    )
    evolution.trigger_quality_based_reembedding(quality_threshold=0.7)
"""

from futurnal.embeddings.quality.metrics import EmbeddingQualityMetrics
from futurnal.embeddings.quality.store import QualityMetricsStore
from futurnal.embeddings.quality.golden import GoldenEmbeddingsManager
from futurnal.embeddings.quality.tracker import QualityMetricsTracker
from futurnal.embeddings.quality.evolution import (
    EmbeddingQualityEvolution,
    ExperientialLearningMonitor,
)
from futurnal.embeddings.quality.profiler import (
    EmbeddingPerformanceProfiler,
    PerformanceMetric,
)

__all__ = [
    # Data Models
    "EmbeddingQualityMetrics",
    # Persistence
    "QualityMetricsStore",
    # Golden Embeddings
    "GoldenEmbeddingsManager",
    # Quality Tracking
    "QualityMetricsTracker",
    # Evolution Integration
    "EmbeddingQualityEvolution",
    "ExperientialLearningMonitor",
    # Performance Profiling
    "EmbeddingPerformanceProfiler",
    "PerformanceMetric",
]
