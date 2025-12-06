"""Tests for EmbeddingQualityMetrics Pydantic model.

Tests data model creation, validation, serialization for SQLite.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

import pytest
from datetime import datetime, timedelta

from futurnal.embeddings.quality.metrics import EmbeddingQualityMetrics


class TestEmbeddingQualityMetrics:
    """Tests for EmbeddingQualityMetrics Pydantic model."""

    def test_basic_creation(self):
        """Validate basic creation with required fields."""
        metrics = EmbeddingQualityMetrics(
            embedding_id="emb_123",
            entity_id="ent_456",
            extraction_confidence=0.85,
            embedding_latency_ms=150.0,
            model_id="instructor-large-entity",
            vector_dimension=768,
        )

        assert metrics.embedding_id == "emb_123"
        assert metrics.entity_id == "ent_456"
        assert metrics.extraction_confidence == 0.85
        assert metrics.embedding_latency_ms == 150.0
        assert metrics.model_id == "instructor-large-entity"
        assert metrics.vector_dimension == 768

    def test_optional_fields_default(self):
        """Validate optional fields default to None."""
        metrics = EmbeddingQualityMetrics(
            embedding_id="emb_123",
            entity_id="ent_456",
            extraction_confidence=0.85,
            embedding_latency_ms=150.0,
            model_id="test-model",
            vector_dimension=768,
        )

        assert metrics.extraction_quality_score is None
        assert metrics.embedding_similarity_score is None
        assert metrics.embedding_coherence is None
        assert metrics.embedding_distinctiveness is None
        assert metrics.temporal_accuracy is None
        assert metrics.causal_pattern_quality is None
        assert metrics.quality_trend is None
        assert metrics.last_reembedded is None
        assert metrics.overall_quality_score is None
        assert metrics.reembedding_count == 0

    def test_quality_trend_validation(self):
        """Validate quality_trend enum constraint."""
        # Valid values
        for trend in ["improving", "stable", "degrading", None]:
            metrics = EmbeddingQualityMetrics(
                embedding_id="emb_123",
                entity_id="ent_456",
                extraction_confidence=0.85,
                embedding_latency_ms=150.0,
                model_id="test-model",
                vector_dimension=768,
                quality_trend=trend,
            )
            assert metrics.quality_trend == trend

        # Invalid value
        with pytest.raises(ValueError):
            EmbeddingQualityMetrics(
                embedding_id="emb_123",
                entity_id="ent_456",
                extraction_confidence=0.85,
                embedding_latency_ms=150.0,
                model_id="test-model",
                vector_dimension=768,
                quality_trend="invalid",  # Should fail
            )

    def test_confidence_bounds(self):
        """Validate extraction_confidence bounds [0, 1]."""
        # Valid boundaries
        metrics_low = EmbeddingQualityMetrics(
            embedding_id="emb_1",
            entity_id="ent_1",
            extraction_confidence=0.0,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
        )
        assert metrics_low.extraction_confidence == 0.0

        metrics_high = EmbeddingQualityMetrics(
            embedding_id="emb_2",
            entity_id="ent_2",
            extraction_confidence=1.0,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
        )
        assert metrics_high.extraction_confidence == 1.0

        # Out of bounds
        with pytest.raises(ValueError):
            EmbeddingQualityMetrics(
                embedding_id="emb_3",
                entity_id="ent_3",
                extraction_confidence=1.5,  # Out of bounds
                embedding_latency_ms=100.0,
                model_id="test",
                vector_dimension=768,
            )

    def test_to_sqlite_dict(self):
        """Validate SQLite serialization."""
        now = datetime.utcnow()
        metrics = EmbeddingQualityMetrics(
            embedding_id="emb_123",
            entity_id="ent_456",
            extraction_confidence=0.85,
            embedding_coherence=0.9,
            embedding_similarity_score=0.75,
            temporal_accuracy=0.8,
            embedding_latency_ms=150.0,
            model_id="test-model",
            vector_dimension=768,
            quality_trend="improving",
            reembedding_count=2,
            last_reembedded=now,
            created_at=now,
            last_validated=now,
            overall_quality_score=0.82,
        )

        data = metrics.to_sqlite_dict()

        assert data["embedding_id"] == "emb_123"
        assert data["entity_id"] == "ent_456"
        assert data["extraction_confidence"] == 0.85
        assert data["embedding_coherence"] == 0.9
        assert data["embedding_similarity_score"] == 0.75
        assert data["temporal_accuracy"] == 0.8
        assert data["embedding_latency_ms"] == 150.0
        assert data["model_id"] == "test-model"
        assert data["vector_dimension"] == 768
        assert data["quality_trend"] == "improving"
        assert data["reembedding_count"] == 2
        assert data["last_reembedded"] == now.isoformat()
        assert data["created_at"] == now.isoformat()
        assert data["overall_quality_score"] == 0.82

    def test_from_sqlite_row(self):
        """Validate SQLite deserialization."""
        now = datetime.utcnow()
        row = {
            "embedding_id": "emb_123",
            "entity_id": "ent_456",
            "extraction_confidence": 0.85,
            "extraction_quality_score": None,
            "embedding_similarity_score": 0.75,
            "embedding_coherence": 0.9,
            "embedding_distinctiveness": None,
            "temporal_accuracy": 0.8,
            "causal_pattern_quality": None,
            "embedding_latency_ms": 150.0,
            "model_id": "test-model",
            "vector_dimension": 768,
            "quality_trend": "stable",
            "reembedding_count": 1,
            "last_reembedded": now.isoformat(),
            "created_at": now.isoformat(),
            "last_validated": now.isoformat(),
            "overall_quality_score": 0.82,
        }

        metrics = EmbeddingQualityMetrics.from_sqlite_row(row)

        assert metrics.embedding_id == "emb_123"
        assert metrics.entity_id == "ent_456"
        assert metrics.extraction_confidence == 0.85
        assert metrics.embedding_coherence == 0.9
        assert metrics.quality_trend == "stable"
        assert metrics.reembedding_count == 1

    def test_roundtrip_serialization(self):
        """Validate roundtrip to SQLite and back."""
        original = EmbeddingQualityMetrics(
            embedding_id="emb_roundtrip",
            entity_id="ent_roundtrip",
            extraction_confidence=0.92,
            embedding_coherence=0.88,
            embedding_similarity_score=0.85,
            temporal_accuracy=0.9,
            embedding_latency_ms=200.0,
            model_id="test-model",
            vector_dimension=1024,
            quality_trend="improving",
            reembedding_count=3,
            overall_quality_score=0.89,
        )

        # Convert to SQLite dict and back
        sqlite_dict = original.to_sqlite_dict()
        restored = EmbeddingQualityMetrics.from_sqlite_row(sqlite_dict)

        assert restored.embedding_id == original.embedding_id
        assert restored.entity_id == original.entity_id
        assert restored.extraction_confidence == original.extraction_confidence
        assert restored.embedding_coherence == original.embedding_coherence
        assert restored.quality_trend == original.quality_trend
        assert restored.overall_quality_score == original.overall_quality_score

    def test_is_low_quality(self):
        """Validate is_low_quality method."""
        # Low quality
        low = EmbeddingQualityMetrics(
            embedding_id="emb_low",
            entity_id="ent_low",
            extraction_confidence=0.5,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
            overall_quality_score=0.4,
        )
        assert low.is_low_quality(threshold=0.6) is True

        # High quality
        high = EmbeddingQualityMetrics(
            embedding_id="emb_high",
            entity_id="ent_high",
            extraction_confidence=0.9,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
            overall_quality_score=0.85,
        )
        assert high.is_low_quality(threshold=0.6) is False

        # None quality (treated as low)
        unknown = EmbeddingQualityMetrics(
            embedding_id="emb_unknown",
            entity_id="ent_unknown",
            extraction_confidence=0.9,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
            overall_quality_score=None,
        )
        assert unknown.is_low_quality(threshold=0.6) is True

    def test_needs_reembedding_due_to_quality(self):
        """Validate needs_reembedding_due_to_quality method."""
        # Low overall quality
        low_overall = EmbeddingQualityMetrics(
            embedding_id="emb_1",
            entity_id="ent_1",
            extraction_confidence=0.9,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
            overall_quality_score=0.4,
        )
        assert low_overall.needs_reembedding_due_to_quality(threshold=0.6) is True

        # Low extraction confidence
        low_confidence = EmbeddingQualityMetrics(
            embedding_id="emb_2",
            entity_id="ent_2",
            extraction_confidence=0.4,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
            overall_quality_score=0.7,
        )
        assert low_confidence.needs_reembedding_due_to_quality(threshold=0.6) is True

        # Low coherence
        low_coherence = EmbeddingQualityMetrics(
            embedding_id="emb_3",
            entity_id="ent_3",
            extraction_confidence=0.9,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
            embedding_coherence=0.3,
            overall_quality_score=0.7,
        )
        assert low_coherence.needs_reembedding_due_to_quality(threshold=0.6) is True

        # Good quality
        good = EmbeddingQualityMetrics(
            embedding_id="emb_4",
            entity_id="ent_4",
            extraction_confidence=0.9,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
            embedding_coherence=0.8,
            overall_quality_score=0.85,
        )
        assert good.needs_reembedding_due_to_quality(threshold=0.6) is False

    def test_timestamps_default_to_now(self):
        """Validate timestamps default to current time."""
        before = datetime.utcnow()
        metrics = EmbeddingQualityMetrics(
            embedding_id="emb_time",
            entity_id="ent_time",
            extraction_confidence=0.85,
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
        )
        after = datetime.utcnow()

        assert before <= metrics.created_at <= after
        assert before <= metrics.last_validated <= after
