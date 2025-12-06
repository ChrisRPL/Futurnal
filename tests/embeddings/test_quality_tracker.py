"""Tests for QualityMetricsTracker.

Tests quality score computation, low-quality identification,
and thread-safety.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import threading
import tempfile
from pathlib import Path

from futurnal.embeddings.quality.tracker import QualityMetricsTracker
from futurnal.embeddings.quality.store import QualityMetricsStore
from futurnal.embeddings.quality.golden import GoldenEmbeddingsManager
from futurnal.embeddings.models import TemporalEmbeddingContext


@pytest.fixture
def in_memory_store():
    """Create an in-memory store for testing."""
    return QualityMetricsStore(in_memory=True)


@pytest.fixture
def temp_golden_manager(tmp_path):
    """Create a golden manager with temp storage."""
    return GoldenEmbeddingsManager(storage_path=tmp_path / "golden")


@pytest.fixture
def tracker(in_memory_store, temp_golden_manager):
    """Create a tracker with in-memory store and temp golden manager."""
    return QualityMetricsTracker(
        store=in_memory_store,
        golden_manager=temp_golden_manager,
    )


def create_normalized_embedding(dim: int = 768) -> np.ndarray:
    """Create a normalized random embedding."""
    embedding = np.random.rand(dim).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


class TestQualityMetricsTracker:
    """Tests for QualityMetricsTracker."""

    def test_record_embedding_quality(self, tracker):
        """Test recording quality metrics for a new embedding."""
        embedding = create_normalized_embedding()

        metrics = tracker.record_embedding_quality(
            embedding_id="emb_123",
            entity_id="ent_456",
            entity_type="Person",
            embedding=embedding,
            extraction_confidence=0.85,
            embedding_latency_ms=150.0,
            model_id="instructor-large-entity",
            vector_dimension=768,
        )

        assert metrics.embedding_id == "emb_123"
        assert metrics.entity_id == "ent_456"
        assert metrics.extraction_confidence == 0.85
        assert metrics.embedding_coherence is not None
        assert metrics.overall_quality_score is not None
        assert 0.0 <= metrics.overall_quality_score <= 1.0

    def test_compute_quality_score_basic(self, tracker):
        """Validate quality score computation."""
        embedding = create_normalized_embedding()

        score = tracker.compute_quality_score(
            embedding=embedding,
            entity_type="Person",
            extraction_confidence=0.9,
            coherence=0.85,
            golden_similarity=None,
            temporal_accuracy=None,
        )

        # With extraction=0.9, coherence=0.85, and normalized weights
        # Score should be reasonable
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably good

    def test_compute_quality_score_with_golden(self, tracker, temp_golden_manager):
        """Test quality score includes golden similarity when available."""
        embedding = create_normalized_embedding()

        # Add golden embedding for Person
        golden = create_normalized_embedding()
        temp_golden_manager.add_golden_embedding("Person", golden)

        # The golden manager now has Person type
        assert "Person" in temp_golden_manager.supported_types

        # Record with golden embedding available
        metrics = tracker.record_embedding_quality(
            embedding_id="emb_golden",
            entity_id="ent_golden",
            entity_type="Person",
            embedding=embedding,
            extraction_confidence=0.85,
            embedding_latency_ms=150.0,
            model_id="test-model",
            vector_dimension=768,
        )

        # Golden similarity should be computed
        assert metrics.embedding_similarity_score is not None

    def test_compute_quality_score_temporal_event(self, tracker):
        """Test quality score for Event with temporal context."""
        embedding = create_normalized_embedding()
        temporal_context = TemporalEmbeddingContext(
            timestamp=datetime.utcnow(),
            duration=timedelta(hours=2),
            temporal_type="DURING",
            causal_chain=["event_1", "event_2"],
        )

        metrics = tracker.record_embedding_quality(
            embedding_id="emb_event",
            entity_id="ent_event",
            entity_type="Event",
            embedding=embedding,
            extraction_confidence=0.85,
            embedding_latency_ms=150.0,
            model_id="instructor-temporal-event",
            vector_dimension=768,
            temporal_context=temporal_context,
        )

        # Temporal accuracy should be computed for Events
        assert metrics.temporal_accuracy is not None
        # Causal pattern quality should be set due to causal_chain
        assert metrics.causal_pattern_quality is not None

    def test_compute_coherence_normalized(self, tracker):
        """Validate coherence for normalized embeddings."""
        # Perfect normalized embedding
        embedding = create_normalized_embedding()

        coherence = tracker._compute_coherence(embedding)

        # Normalized embedding should have high coherence
        assert coherence > 0.8

    def test_compute_coherence_unnormalized(self, tracker):
        """Validate coherence for unnormalized embeddings."""
        # Unnormalized embedding (norm != 1)
        embedding = np.random.rand(768).astype(np.float32) * 10  # Large values

        coherence = tracker._compute_coherence(embedding)

        # Should still be valid but lower score for norm check
        assert 0.0 <= coherence <= 1.0

    def test_compute_coherence_invalid_nan(self, tracker):
        """Validate coherence handles NaN values."""
        embedding = np.array([1.0, np.nan, 0.5, np.inf, 0.2], dtype=np.float32)

        coherence = tracker._compute_coherence(embedding)

        # Should have low coherence due to invalid values
        assert coherence < 0.5

    def test_compute_coherence_zero_vector(self, tracker):
        """Validate coherence for zero vector."""
        embedding = np.zeros(768, dtype=np.float32)

        coherence = tracker._compute_coherence(embedding)

        # Zero vector should have low coherence
        assert coherence < 0.5

    def test_compute_coherence_constant_vector(self, tracker):
        """Validate coherence for constant vector."""
        embedding = np.ones(768, dtype=np.float32)

        coherence = tracker._compute_coherence(embedding)

        # Constant vector has zero std dev
        assert coherence < 0.7  # Distribution score will be low

    def test_compute_golden_similarity(self, tracker, temp_golden_manager):
        """Test golden similarity computation."""
        # Add golden embedding
        golden = create_normalized_embedding()
        temp_golden_manager.add_golden_embedding("Person", golden)

        # Test with similar embedding
        embedding = golden.copy()  # Identical
        similarity = tracker._compute_golden_similarity(embedding, "Person")

        # Should be very high (close to 1.0 after conversion)
        assert similarity > 0.9

    def test_compute_golden_similarity_no_goldens(self, tracker):
        """Test golden similarity returns neutral when no goldens exist."""
        embedding = create_normalized_embedding()

        similarity = tracker._compute_golden_similarity(embedding, "NonExistentType")

        # Should return neutral score
        assert similarity == 0.7

    def test_compute_temporal_quality(self, tracker):
        """Test temporal quality computation."""
        embedding = create_normalized_embedding()

        # Full temporal context
        full_context = TemporalEmbeddingContext(
            timestamp=datetime.utcnow(),
            duration=timedelta(hours=1),
            temporal_type="CAUSES",
            causal_chain=["a", "b", "c"],
            event_sequence=["1", "2"],
            temporal_neighbors=["n1", "n2"],
        )

        quality = tracker._compute_temporal_quality(embedding, full_context)

        # Full context should have high quality
        assert quality >= 0.9

    def test_compute_temporal_quality_minimal(self, tracker):
        """Test temporal quality with minimal context."""
        embedding = create_normalized_embedding()

        # Minimal context (just timestamp)
        minimal_context = TemporalEmbeddingContext(
            timestamp=datetime.utcnow(),
        )

        quality = tracker._compute_temporal_quality(embedding, minimal_context)

        # Minimal context should have base quality
        assert 0.5 <= quality <= 0.7

    def test_identify_low_quality_embeddings(self, tracker, in_memory_store):
        """Validate identification of low-quality embeddings."""
        # Insert mix of qualities
        for i in range(20):
            quality = 0.3 + i * 0.035  # 0.3 to 0.965 (within valid range)
            embedding = create_normalized_embedding()

            tracker.record_embedding_quality(
                embedding_id=f"emb_id_{i}",
                entity_id=f"ent_id_{i}",
                entity_type="Person",
                embedding=embedding,
                extraction_confidence=quality,
                embedding_latency_ms=100.0,
                model_id="test-model",
                vector_dimension=768,
            )

        # Find low quality
        low_quality = tracker.identify_low_quality_embeddings(
            min_quality_score=0.6,
            limit=100,
        )

        assert isinstance(low_quality, list)
        # Should have found some low-quality entities
        assert len(low_quality) > 0
        # Should be less than total
        assert len(low_quality) < 20

    def test_get_quality_summary(self, tracker):
        """Test getting quality summary for an entity."""
        embedding = create_normalized_embedding()

        tracker.record_embedding_quality(
            embedding_id="emb_summary",
            entity_id="ent_summary",
            entity_type="Person",
            embedding=embedding,
            extraction_confidence=0.85,
            embedding_latency_ms=150.0,
            model_id="test-model",
            vector_dimension=768,
        )

        summary = tracker.get_quality_summary("ent_summary")

        assert summary["has_metrics"] is True
        assert summary["entity_id"] == "ent_summary"
        assert summary["latest_quality_score"] is not None
        assert summary["extraction_confidence"] == 0.85
        assert summary["metrics_count"] == 1

    def test_get_quality_summary_not_found(self, tracker):
        """Test quality summary for non-existent entity."""
        summary = tracker.get_quality_summary("nonexistent")

        assert summary["has_metrics"] is False
        assert summary["entity_id"] == "nonexistent"

    def test_update_quality_trend(self, tracker, in_memory_store):
        """Test updating quality trend for an entity."""
        entity_id = "ent_trend"

        # Insert historical metrics with improving quality
        for i in range(5):
            embedding = create_normalized_embedding()

            metrics = tracker.record_embedding_quality(
                embedding_id=f"emb_trend_{i}",
                entity_id=entity_id,
                entity_type="Person",
                embedding=embedding,
                extraction_confidence=0.5 + i * 0.1,  # 0.5 -> 0.9
                embedding_latency_ms=100.0,
                model_id="test-model",
                vector_dimension=768,
            )

        # Update trend
        trend = tracker.update_quality_trend(entity_id, lookback_days=30)

        assert trend == "improving"

    def test_get_statistics(self, tracker):
        """Test getting tracker statistics."""
        # Record some metrics
        for i in range(5):
            embedding = create_normalized_embedding()
            tracker.record_embedding_quality(
                embedding_id=f"emb_stats_{i}",
                entity_id=f"ent_stats_{i}",
                entity_type="Person",
                embedding=embedding,
                extraction_confidence=0.8,
                embedding_latency_ms=100.0,
                model_id="test-model",
                vector_dimension=768,
            )

        stats = tracker.get_statistics()

        assert "total_count" in stats
        assert stats["total_count"] == 5
        assert "golden_embeddings" in stats
        assert "quality_weights" in stats

    def test_thread_safety(self, tracker):
        """Validate thread-safe operations."""
        errors = []

        def record_metrics(thread_id):
            try:
                for i in range(10):
                    embedding = create_normalized_embedding()
                    tracker.record_embedding_quality(
                        embedding_id=f"emb_thread_{thread_id}_{i}",
                        entity_id=f"ent_thread_{thread_id}",
                        entity_type="Person",
                        embedding=embedding,
                        extraction_confidence=0.8,
                        embedding_latency_ms=100.0,
                        model_id="test-model",
                        vector_dimension=768,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_metrics, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All metrics should be recorded
        stats = tracker.get_statistics()
        assert stats["total_count"] == 50  # 5 threads * 10 records

    def test_custom_weights(self, in_memory_store, temp_golden_manager):
        """Test tracker with custom weights."""
        custom_weights = {
            "extraction": 0.5,
            "coherence": 0.2,
            "golden": 0.2,
            "temporal": 0.1,
        }

        tracker = QualityMetricsTracker(
            store=in_memory_store,
            golden_manager=temp_golden_manager,
            weights=custom_weights,
        )

        embedding = create_normalized_embedding()

        # With high extraction confidence and custom weights
        score = tracker.compute_quality_score(
            embedding=embedding,
            entity_type="Person",
            extraction_confidence=0.95,
            coherence=0.5,  # Lower coherence
        )

        # Score should be influenced more by extraction (50%) than coherence (20%)
        assert score > 0.7  # Should be pulled up by high extraction

    def test_quality_score_weight_normalization(self, tracker):
        """Test that missing components result in weight renormalization."""
        embedding = create_normalized_embedding()

        # Only extraction and coherence (no golden, no temporal)
        score = tracker.compute_quality_score(
            embedding=embedding,
            entity_type="Person",  # Not Event, so no temporal
            extraction_confidence=0.8,
            coherence=0.9,
            golden_similarity=None,  # No golden
            temporal_accuracy=None,
        )

        # Score should be valid
        assert 0.0 <= score <= 1.0

        # With only extraction=0.8 (40%) and coherence=0.9 (30%)
        # After renormalization, these should be weighted to sum to 1.0
        # Expected: (0.8*0.4 + 0.9*0.3) / 0.7 ≈ 0.84
        assert score > 0.7
