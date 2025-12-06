"""Embedding Quality Metrics Model.

Pydantic model for tracking embedding quality assessment metrics.
Supports quality evolution through experiential learning integration.

Option B Compliance:
- Tracks extraction quality from GRPO evolution
- Supports quality-based re-embedding triggers
- Preserves temporal quality metrics for events

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class EmbeddingQualityMetrics(BaseModel):
    """Metrics for embedding quality assessment.

    Tracks multiple quality dimensions to enable:
    - Quality score computation
    - Trend analysis over time
    - Re-embedding triggers for quality improvement

    Quality Components:
    1. Extraction quality (from entity extraction pipeline)
    2. Embedding coherence (internal vector quality)
    3. Golden similarity (comparison to reference embeddings)
    4. Temporal quality (for Event entities)

    Option B Compliance:
    - Tracks extraction confidence from GRPO evolution
    - Supports quality-based re-embedding triggers
    - Preserves temporal accuracy for events
    """

    # Identifiers
    embedding_id: str = Field(
        ...,
        description="Unique embedding identifier (UUID)",
    )
    entity_id: str = Field(
        ...,
        description="PKG entity ID this embedding represents",
    )

    # Extraction quality (from extraction pipeline)
    extraction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from entity extraction",
    )
    extraction_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quality score from extraction validation",
    )

    # Embedding quality metrics
    embedding_similarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity to golden reference embeddings",
    )
    embedding_coherence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Internal consistency (norm, validity, distribution)",
    )
    embedding_distinctiveness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Distance from unrelated embeddings",
    )

    # Temporal quality (for Event entities - Option B critical)
    temporal_accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Temporal context preservation quality",
    )
    causal_pattern_quality: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Causal pattern matching quality",
    )

    # Performance metrics
    embedding_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Time to generate embedding in milliseconds",
    )
    model_id: str = Field(
        ...,
        description="Model used for embedding generation",
    )
    vector_dimension: int = Field(
        ...,
        ge=1,
        description="Dimension of embedding vector",
    )

    # Quality evolution tracking
    quality_trend: Optional[Literal["improving", "stable", "degrading"]] = Field(
        default=None,
        description="Quality trend over time",
    )
    reembedding_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this entity has been re-embedded",
    )
    last_reembedded: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last re-embedding",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Metrics creation timestamp",
    )
    last_validated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last validation timestamp",
    )

    # Composite quality score (computed by QualityMetricsTracker)
    overall_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weighted composite quality score",
    )

    @field_validator("quality_trend", mode="before")
    @classmethod
    def validate_quality_trend(cls, v: Any) -> Optional[str]:
        """Validate quality_trend is a valid value."""
        if v is None:
            return None
        valid_values = {"improving", "stable", "degrading"}
        if v not in valid_values:
            raise ValueError(
                f"quality_trend must be one of {valid_values}, got '{v}'"
            )
        return v

    def to_sqlite_dict(self) -> Dict[str, Any]:
        """Convert to SQLite-compatible dictionary.

        SQLite stores None as NULL, datetime as ISO string.

        Returns:
            Dictionary with primitive values for SQLite storage.
        """
        return {
            "embedding_id": self.embedding_id,
            "entity_id": self.entity_id,
            "extraction_confidence": self.extraction_confidence,
            "extraction_quality_score": self.extraction_quality_score,
            "embedding_similarity_score": self.embedding_similarity_score,
            "embedding_coherence": self.embedding_coherence,
            "embedding_distinctiveness": self.embedding_distinctiveness,
            "temporal_accuracy": self.temporal_accuracy,
            "causal_pattern_quality": self.causal_pattern_quality,
            "embedding_latency_ms": self.embedding_latency_ms,
            "model_id": self.model_id,
            "vector_dimension": self.vector_dimension,
            "quality_trend": self.quality_trend,
            "reembedding_count": self.reembedding_count,
            "last_reembedded": (
                self.last_reembedded.isoformat() if self.last_reembedded else None
            ),
            "created_at": self.created_at.isoformat(),
            "last_validated": self.last_validated.isoformat(),
            "overall_quality_score": self.overall_quality_score,
        }

    @classmethod
    def from_sqlite_row(cls, row: Dict[str, Any]) -> "EmbeddingQualityMetrics":
        """Reconstruct EmbeddingQualityMetrics from SQLite row.

        Reverses to_sqlite_dict(), converting ISO strings back to datetime.

        Args:
            row: Dictionary from SQLite query result.

        Returns:
            Reconstructed EmbeddingQualityMetrics instance.
        """
        # Parse datetime fields
        created_at = row.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        last_validated = row.get("last_validated")
        if isinstance(last_validated, str):
            last_validated = datetime.fromisoformat(last_validated)
        elif last_validated is None:
            last_validated = datetime.utcnow()

        last_reembedded = row.get("last_reembedded")
        if isinstance(last_reembedded, str):
            last_reembedded = datetime.fromisoformat(last_reembedded)

        return cls(
            embedding_id=row["embedding_id"],
            entity_id=row["entity_id"],
            extraction_confidence=row["extraction_confidence"],
            extraction_quality_score=row.get("extraction_quality_score"),
            embedding_similarity_score=row.get("embedding_similarity_score"),
            embedding_coherence=row.get("embedding_coherence"),
            embedding_distinctiveness=row.get("embedding_distinctiveness"),
            temporal_accuracy=row.get("temporal_accuracy"),
            causal_pattern_quality=row.get("causal_pattern_quality"),
            embedding_latency_ms=row["embedding_latency_ms"],
            model_id=row["model_id"],
            vector_dimension=row["vector_dimension"],
            quality_trend=row.get("quality_trend"),
            reembedding_count=row.get("reembedding_count", 0),
            last_reembedded=last_reembedded,
            created_at=created_at,
            last_validated=last_validated,
            overall_quality_score=row.get("overall_quality_score"),
        )

    def is_low_quality(self, threshold: float = 0.6) -> bool:
        """Check if this embedding is below quality threshold.

        Args:
            threshold: Minimum acceptable quality score (default: 0.6)

        Returns:
            True if overall_quality_score is below threshold or None
        """
        if self.overall_quality_score is None:
            return True  # Unknown quality treated as low
        return self.overall_quality_score < threshold

    def needs_reembedding_due_to_quality(self, threshold: float = 0.6) -> bool:
        """Check if this embedding should be re-embedded due to quality.

        Considers both overall quality score and extraction confidence.

        Args:
            threshold: Minimum acceptable quality score (default: 0.6)

        Returns:
            True if embedding should be re-embedded
        """
        if self.is_low_quality(threshold):
            return True
        if self.extraction_confidence < threshold:
            return True
        if self.embedding_coherence is not None and self.embedding_coherence < 0.5:
            return True
        return False
