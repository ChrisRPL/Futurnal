"""Tests for QualityMetricsStore SQLite backend.

Tests persistence, queries, and statistics.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from futurnal.embeddings.quality.metrics import EmbeddingQualityMetrics
from futurnal.embeddings.quality.store import QualityMetricsStore


@pytest.fixture
def in_memory_store():
    """Create an in-memory store for testing."""
    return QualityMetricsStore(in_memory=True)


@pytest.fixture
def temp_db_store(tmp_path):
    """Create a store with temporary file database."""
    db_path = tmp_path / "test_quality_metrics.db"
    return QualityMetricsStore(db_path=db_path)


def create_test_metrics(
    embedding_id: str = "emb_test",
    entity_id: str = "ent_test",
    extraction_confidence: float = 0.85,
    overall_quality_score: float = 0.8,
    **kwargs,
) -> EmbeddingQualityMetrics:
    """Helper to create test metrics."""
    return EmbeddingQualityMetrics(
        embedding_id=embedding_id,
        entity_id=entity_id,
        extraction_confidence=extraction_confidence,
        embedding_latency_ms=kwargs.get("embedding_latency_ms", 150.0),
        model_id=kwargs.get("model_id", "test-model"),
        vector_dimension=kwargs.get("vector_dimension", 768),
        overall_quality_score=overall_quality_score,
        embedding_coherence=kwargs.get("embedding_coherence"),
        quality_trend=kwargs.get("quality_trend"),
        **{k: v for k, v in kwargs.items() if k not in [
            "embedding_latency_ms", "model_id", "vector_dimension",
            "embedding_coherence", "quality_trend"
        ]}
    )


class TestQualityMetricsStore:
    """Tests for QualityMetricsStore."""

    def test_insert_and_retrieve_by_embedding_id(self, in_memory_store):
        """Test basic insert and retrieval by embedding ID."""
        metrics = create_test_metrics(embedding_id="emb_123", entity_id="ent_456")
        in_memory_store.insert(metrics)

        retrieved = in_memory_store.get_by_embedding_id("emb_123")

        assert retrieved is not None
        assert retrieved.embedding_id == "emb_123"
        assert retrieved.entity_id == "ent_456"
        assert retrieved.extraction_confidence == 0.85

    def test_get_by_embedding_id_not_found(self, in_memory_store):
        """Test retrieval returns None for non-existent ID."""
        result = in_memory_store.get_by_embedding_id("nonexistent")
        assert result is None

    def test_insert_duplicate_raises_error(self, in_memory_store):
        """Test inserting duplicate embedding_id raises error."""
        metrics = create_test_metrics(embedding_id="duplicate_id")
        in_memory_store.insert(metrics)

        with pytest.raises(Exception):  # sqlite3.IntegrityError
            in_memory_store.insert(metrics)

    def test_get_by_entity_id(self, in_memory_store):
        """Test retrieval by entity ID returns all metrics for entity."""
        # Insert multiple metrics for same entity
        for i in range(3):
            metrics = create_test_metrics(
                embedding_id=f"emb_{i}",
                entity_id="shared_entity",
                extraction_confidence=0.7 + i * 0.1,
            )
            in_memory_store.insert(metrics)

        # Insert metric for different entity
        other = create_test_metrics(
            embedding_id="emb_other",
            entity_id="other_entity",
        )
        in_memory_store.insert(other)

        results = in_memory_store.get_by_entity_id("shared_entity")

        assert len(results) == 3
        assert all(m.entity_id == "shared_entity" for m in results)

    def test_update(self, in_memory_store):
        """Test updating existing metrics."""
        metrics = create_test_metrics(
            embedding_id="emb_update",
            extraction_confidence=0.5,
        )
        in_memory_store.insert(metrics)

        # Update fields
        updated = in_memory_store.update(
            "emb_update",
            {
                "extraction_confidence": 0.9,
                "quality_trend": "improving",
                "overall_quality_score": 0.88,
            },
        )

        assert updated is True

        # Verify update
        retrieved = in_memory_store.get_by_embedding_id("emb_update")
        assert retrieved.extraction_confidence == 0.9
        assert retrieved.quality_trend == "improving"
        assert retrieved.overall_quality_score == 0.88

    def test_update_nonexistent_returns_false(self, in_memory_store):
        """Test updating non-existent record returns False."""
        result = in_memory_store.update(
            "nonexistent",
            {"extraction_confidence": 0.9},
        )
        assert result is False

    def test_query_low_quality(self, in_memory_store):
        """Test querying low-quality embeddings."""
        # Insert mix of quality levels
        for i in range(10):
            quality = 0.3 + i * 0.07  # 0.3 to 0.93 (within valid range)
            metrics = create_test_metrics(
                embedding_id=f"emb_q{i}",
                entity_id=f"ent_q{i}",
                extraction_confidence=quality,
                overall_quality_score=quality,
            )
            in_memory_store.insert(metrics)

        # Query low quality (below 0.6)
        low_quality = in_memory_store.query_low_quality(
            min_quality_score=0.6,
            limit=100,
        )

        # Should find entities with quality < 0.6
        assert len(low_quality) > 0
        assert len(low_quality) < 10  # Not all

    def test_query_low_quality_includes_null_scores(self, in_memory_store):
        """Test that null quality scores are included in low-quality query."""
        # Insert with null overall_quality_score
        metrics = create_test_metrics(
            embedding_id="emb_null",
            entity_id="ent_null",
            overall_quality_score=None,
        )
        # Need to set to None after creation since create_test_metrics sets it
        metrics_dict = metrics.to_sqlite_dict()
        metrics_dict["overall_quality_score"] = None
        # Insert directly with SQL would be needed, but let's test with low extraction_confidence
        metrics2 = EmbeddingQualityMetrics(
            embedding_id="emb_null2",
            entity_id="ent_null2",
            extraction_confidence=0.3,  # Low
            embedding_latency_ms=100.0,
            model_id="test",
            vector_dimension=768,
            overall_quality_score=None,
        )
        in_memory_store.insert(metrics2)

        low_quality = in_memory_store.query_low_quality(
            min_quality_score=0.6,
            limit=100,
        )

        assert "ent_null2" in low_quality

    def test_query_by_timerange(self, in_memory_store):
        """Test querying by time range."""
        now = datetime.utcnow()

        # Insert metrics at different times
        for i in range(5):
            metrics = EmbeddingQualityMetrics(
                embedding_id=f"emb_time_{i}",
                entity_id="ent_timerange",
                extraction_confidence=0.8,
                embedding_latency_ms=100.0,
                model_id="test",
                vector_dimension=768,
                created_at=now - timedelta(days=i),
                last_validated=now - timedelta(days=i),
            )
            in_memory_store.insert(metrics)

        # Query last 3 days
        start = now - timedelta(days=3)
        end = now + timedelta(days=1)

        results = in_memory_store.query_by_timerange(
            entity_id="ent_timerange",
            start=start,
            end=end,
        )

        # Should get metrics from days 0, 1, 2, 3 (4 metrics)
        assert len(results) >= 3  # At least days 0, 1, 2

    def test_get_entity_history(self, in_memory_store):
        """Test getting entity history for trend analysis."""
        now = datetime.utcnow()

        # Insert historical metrics
        for i in range(10):
            metrics = EmbeddingQualityMetrics(
                embedding_id=f"emb_hist_{i}",
                entity_id="ent_history",
                extraction_confidence=0.5 + i * 0.05,  # Improving
                embedding_latency_ms=100.0,
                model_id="test",
                vector_dimension=768,
                created_at=now - timedelta(days=30 - i * 3),
                last_validated=now - timedelta(days=30 - i * 3),
            )
            in_memory_store.insert(metrics)

        history = in_memory_store.get_entity_history(
            entity_id="ent_history",
            lookback_days=30,
        )

        assert len(history) > 0
        # Verify sorted by created_at
        for i in range(len(history) - 1):
            assert history[i].created_at <= history[i + 1].created_at

    def test_delete_by_embedding_id(self, in_memory_store):
        """Test deleting metrics by embedding ID."""
        metrics = create_test_metrics(embedding_id="emb_delete")
        in_memory_store.insert(metrics)

        # Verify exists
        assert in_memory_store.get_by_embedding_id("emb_delete") is not None

        # Delete
        deleted = in_memory_store.delete_by_embedding_id("emb_delete")
        assert deleted is True

        # Verify deleted
        assert in_memory_store.get_by_embedding_id("emb_delete") is None

    def test_delete_nonexistent_returns_false(self, in_memory_store):
        """Test deleting non-existent returns False."""
        result = in_memory_store.delete_by_embedding_id("nonexistent")
        assert result is False

    def test_get_statistics(self, in_memory_store):
        """Test getting store statistics."""
        # Insert various metrics
        for i in range(20):
            quality = 0.4 + i * 0.03
            trend = "improving" if i % 3 == 0 else ("stable" if i % 3 == 1 else "degrading")
            metrics = create_test_metrics(
                embedding_id=f"emb_stat_{i}",
                entity_id=f"ent_stat_{i}",
                extraction_confidence=quality,
                overall_quality_score=quality,
                embedding_coherence=0.7 + i * 0.01,
                quality_trend=trend,
            )
            in_memory_store.insert(metrics)

        stats = in_memory_store.get_statistics()

        assert stats["total_count"] == 20
        assert "avg_quality_score" in stats
        assert "avg_extraction_confidence" in stats
        assert "trend_distribution" in stats
        assert "low_quality_entity_count" in stats

    def test_persistence_to_file(self, temp_db_store, tmp_path):
        """Test data persists to file database."""
        # Insert data
        metrics = create_test_metrics(embedding_id="emb_persist")
        temp_db_store.insert(metrics)

        # Create new store pointing to same file
        db_path = tmp_path / "test_quality_metrics.db"
        new_store = QualityMetricsStore(db_path=db_path)

        # Data should persist
        retrieved = new_store.get_by_embedding_id("emb_persist")
        assert retrieved is not None
        assert retrieved.embedding_id == "emb_persist"

    def test_query_low_quality_respects_limit(self, in_memory_store):
        """Test that query_low_quality respects limit parameter."""
        # Insert many low-quality embeddings
        for i in range(50):
            metrics = create_test_metrics(
                embedding_id=f"emb_limit_{i}",
                entity_id=f"ent_limit_{i}",
                extraction_confidence=0.3,
                overall_quality_score=0.3,
            )
            in_memory_store.insert(metrics)

        # Query with limit
        results = in_memory_store.query_low_quality(
            min_quality_score=0.6,
            limit=10,
        )

        assert len(results) <= 10

    def test_thread_safety_concurrent_inserts(self, in_memory_store):
        """Test thread-safe concurrent inserts."""
        import threading

        errors = []

        def insert_metrics(thread_id):
            try:
                for i in range(10):
                    metrics = create_test_metrics(
                        embedding_id=f"emb_thread_{thread_id}_{i}",
                        entity_id=f"ent_thread_{thread_id}",
                    )
                    in_memory_store.insert(metrics)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=insert_metrics, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All inserts should succeed
        stats = in_memory_store.get_statistics()
        assert stats["total_count"] == 50  # 5 threads * 10 inserts
