Summary: Implement embedding quality tracking and evolution mechanisms integrated with experiential learning pipeline.

# 05 · Quality Evolution & Performance

## Purpose
Implement embedding quality metrics, experiential learning integration, and performance optimization to ensure embedding quality evolves alongside extraction quality through Ghost→Animal evolution.

**Criticality**: MEDIUM - Enables continuous quality improvement

## Scope
- Embedding quality metrics and tracking
- Integration with experiential learning pipeline
- Re-embedding low-quality extractions
- Performance optimization and profiling
- Quality gates for production readiness

## Requirements Alignment
- **Option B Requirement**: "Embedding quality must improve over time"
- **Experiential Learning Integration**: Track and improve embeddings as system learns
- **Performance Target**: <2s single embedding, >100/min batch
- **Enables**: Continuous quality improvement without manual intervention

## Component Design

### Quality Metrics

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import numpy as np


class EmbeddingQualityMetrics(BaseModel):
    """Metrics for embedding quality assessment."""
    embedding_id: str
    entity_id: str

    # Extraction quality (inherited from extraction pipeline)
    extraction_confidence: float
    extraction_quality_score: Optional[float] = None

    # Embedding quality
    embedding_similarity_score: Optional[float] = None  # Similarity to golden embeddings
    embedding_coherence: Optional[float] = None         # Internal consistency
    embedding_distinctiveness: Optional[float] = None   # Distance from unrelated embeddings

    # Temporal quality (for events)
    temporal_accuracy: Optional[float] = None           # Temporal context preservation
    causal_pattern_quality: Optional[float] = None      # Causal pattern matching quality

    # Performance metrics
    embedding_latency_ms: float
    model_id: str
    vector_dimension: int

    # Quality evolution
    quality_trend: Optional[str] = None  # "improving", "stable", "degrading"
    reembedding_count: int = 0
    last_reembedded: Optional[datetime] = None

    created_at: datetime
    last_validated: datetime


class QualityMetricsTracker:
    """Tracks embedding quality over time."""

    def __init__(self, metrics_store):
        self.store = metrics_store

    def record_embedding_quality(
        self,
        embedding_id: str,
        entity_id: str,
        extraction_confidence: float,
        embedding_latency_ms: float,
        model_id: str,
        vector_dimension: int
    ) -> EmbeddingQualityMetrics:
        """Record quality metrics for new embedding."""
        metrics = EmbeddingQualityMetrics(
            embedding_id=embedding_id,
            entity_id=entity_id,
            extraction_confidence=extraction_confidence,
            embedding_latency_ms=embedding_latency_ms,
            model_id=model_id,
            vector_dimension=vector_dimension,
            created_at=datetime.now(),
            last_validated=datetime.now()
        )

        # Store metrics
        self.store.insert(metrics.dict())

        return metrics

    def compute_quality_score(
        self,
        embedding: np.ndarray,
        entity_type: str,
        extraction_confidence: float,
        golden_embeddings: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Compute overall quality score for embedding.

        Components:
        1. Extraction confidence (from extraction pipeline)
        2. Embedding coherence (internal quality)
        3. Similarity to golden embeddings (if available)
        4. Temporal accuracy (for events)
        """
        scores = []

        # Component 1: Extraction confidence
        scores.append(extraction_confidence)

        # Component 2: Embedding coherence (vector norm, density)
        coherence = self._compute_coherence(embedding)
        scores.append(coherence)

        # Component 3: Golden embedding similarity
        if golden_embeddings:
            similarity = self._compute_golden_similarity(embedding, golden_embeddings)
            scores.append(similarity)

        # Component 4: Temporal quality (for events)
        if entity_type == "Event":
            temporal_quality = self._compute_temporal_quality(embedding)
            scores.append(temporal_quality)

        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1] if len(scores) == 4 else [0.5, 0.3, 0.2]
        quality_score = sum(s * w for s, w in zip(scores, weights[:len(scores)]))

        return quality_score

    def _compute_coherence(self, embedding: np.ndarray) -> float:
        """
        Compute embedding coherence.

        Measures internal quality of embedding vector.
        """
        # Check vector norm (should be normalized)
        norm = np.linalg.norm(embedding)
        norm_score = 1.0 if abs(norm - 1.0) < 0.01 else 0.5

        # Check for NaN/Inf
        validity_score = 1.0 if np.isfinite(embedding).all() else 0.0

        # Check distribution (should not be all zeros or constants)
        std_dev = np.std(embedding)
        distribution_score = min(1.0, std_dev * 10)  # Scale appropriately

        return (norm_score + validity_score + distribution_score) / 3

    def _compute_golden_similarity(
        self,
        embedding: np.ndarray,
        golden_embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute similarity to golden/reference embeddings.

        Golden embeddings are high-quality, manually validated embeddings.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = [
            cosine_similarity([embedding], [golden])[0][0]
            for golden in golden_embeddings
        ]

        # Return max similarity (best match)
        return max(similarities) if similarities else 0.0

    def _compute_temporal_quality(self, embedding: np.ndarray) -> float:
        """
        Compute temporal quality for event embeddings.

        Placeholder - would measure temporal context preservation.
        """
        # This would involve:
        # - Checking temporal similarity to nearby events
        # - Validating causal pattern embeddings
        # For now, return neutral score
        return 0.7

    def identify_low_quality_embeddings(
        self,
        min_quality_score: float = 0.6,
        limit: int = 100
    ) -> List[str]:
        """
        Identify embeddings below quality threshold.

        Returns: List of entity_ids needing re-embedding
        """
        query = """
            SELECT entity_id
            FROM embedding_quality_metrics
            WHERE extraction_confidence < ?
               OR embedding_coherence < ?
            ORDER BY extraction_confidence ASC
            LIMIT ?
        """

        results = self.store.query(query, (min_quality_score, 0.5, limit))

        return [r["entity_id"] for r in results]
```

### Experiential Learning Integration

```python
class EmbeddingQualityEvolution:
    """
    Integrates embedding quality with experiential learning pipeline.

    As extraction quality improves, re-embed low-quality entities.
    """

    def __init__(
        self,
        quality_tracker: QualityMetricsTracker,
        reembedding_service,
        experiential_learning_monitor
    ):
        self.quality_tracker = quality_tracker
        self.reembedding_service = reembedding_service
        self.learning_monitor = experiential_learning_monitor

    def monitor_extraction_quality_improvement(self):
        """
        Monitor extraction pipeline for quality improvements.

        When extraction quality improves significantly, trigger
        re-embedding of entities extracted with lower quality.
        """
        # Get current extraction quality
        current_quality = self.learning_monitor.get_current_quality_metrics()

        # Get historical quality baseline
        baseline_quality = self.learning_monitor.get_baseline_quality()

        # Check for significant improvement
        improvement = current_quality["precision"] - baseline_quality["precision"]

        if improvement > 0.05:  # 5% improvement threshold
            # Identify entities extracted before improvement
            low_quality_entities = self.quality_tracker.identify_low_quality_embeddings(
                min_quality_score=baseline_quality["precision"],
                limit=1000
            )

            # Trigger re-embedding
            self.reembedding_service.trigger_reembedding(
                entity_ids=low_quality_entities,
                reason="extraction_quality_improvement"
            )

    def trigger_quality_based_reembedding(
        self,
        quality_threshold: float = 0.7
    ):
        """
        Trigger re-embedding for low-quality embeddings.

        This is a continuous quality improvement mechanism.
        """
        # Identify low-quality embeddings
        low_quality = self.quality_tracker.identify_low_quality_embeddings(
            min_quality_score=quality_threshold,
            limit=500
        )

        if low_quality:
            # Re-embed in batches
            self.reembedding_service.trigger_reembedding(
                entity_ids=low_quality,
                reason="quality_improvement",
                batch_size=50
            )

    def measure_quality_trend(
        self,
        entity_id: str,
        lookback_days: int = 30
    ) -> str:
        """
        Measure quality trend for entity embeddings.

        Returns: "improving", "stable", or "degrading"
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=lookback_days)

        # Get historical quality metrics
        metrics = self.quality_tracker.store.query("""
            SELECT extraction_confidence, created_at
            FROM embedding_quality_metrics
            WHERE entity_id = ?
              AND created_at > ?
            ORDER BY created_at ASC
        """, (entity_id, cutoff))

        if len(metrics) < 2:
            return "stable"  # Not enough data

        # Compute trend
        qualities = [m["extraction_confidence"] for m in metrics]
        trend = qualities[-1] - qualities[0]

        if trend > 0.05:
            return "improving"
        elif trend < -0.05:
            return "degrading"
        else:
            return "stable"
```

### Performance Profiler

```python
class EmbeddingPerformanceProfiler:
    """
    Profiles embedding performance and identifies bottlenecks.

    Tracks latency, throughput, memory usage.
    """

    def __init__(self):
        self.metrics = []

    def profile_embedding_request(
        self,
        model_id: str,
        entity_type: str,
        content_length: int,
        latency_ms: float,
        memory_mb: float
    ):
        """Record performance metrics for embedding request."""
        self.metrics.append({
            "model_id": model_id,
            "entity_type": entity_type,
            "content_length": content_length,
            "latency_ms": latency_ms,
            "memory_mb": memory_mb,
            "timestamp": datetime.now()
        })

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report.

        Returns: {
            "avg_latency_ms": float,
            "p95_latency_ms": float,
            "p99_latency_ms": float,
            "avg_memory_mb": float,
            "throughput_per_minute": float
        }
        """
        if not self.metrics:
            return {}

        latencies = [m["latency_ms"] for m in self.metrics]

        return {
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "avg_memory_mb": np.mean([m["memory_mb"] for m in self.metrics]),
            "throughput_per_minute": len(self.metrics) / (
                (self.metrics[-1]["timestamp"] - self.metrics[0]["timestamp"]).seconds / 60
            ) if len(self.metrics) > 1 else 0
        }

    def identify_performance_bottlenecks(self) -> List[str]:
        """
        Identify performance bottlenecks.

        Returns: List of recommendations
        """
        report = self.generate_performance_report()
        recommendations = []

        # Check latency
        if report.get("p95_latency_ms", 0) > 2000:
            recommendations.append("Consider model quantization to reduce latency")

        if report.get("avg_memory_mb", 0) > 2000:
            recommendations.append("Memory usage high - enable model unloading")

        if report.get("throughput_per_minute", 0) < 30:
            recommendations.append("Low throughput - increase batch size")

        return recommendations
```

## Implementation Details

### Week 5: Quality Evolution & Performance

**Deliverable**: Quality tracking and performance optimization

1. **Implement quality metrics**:
   - `EmbeddingQualityMetrics` model
   - Quality score computation
   - Golden embedding comparison

2. **Implement experiential learning integration**:
   - Monitor extraction quality improvements
   - Trigger quality-based re-embedding
   - Track quality trends

3. **Implement performance profiling**:
   - Latency tracking
   - Memory profiling
   - Bottleneck identification

4. **Performance optimization**:
   - Model quantization tuning
   - Batch size optimization
   - Caching strategies

## Testing Strategy

```python
class TestQualityEvolution:
    def test_quality_score_computation(self):
        """Validate quality score computation."""
        tracker = QualityMetricsTracker(metrics_store)

        embedding = np.random.rand(768)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        score = tracker.compute_quality_score(
            embedding=embedding,
            entity_type="Person",
            extraction_confidence=0.9,
            golden_embeddings=None
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonable quality

    def test_low_quality_identification(self):
        """Validate identification of low-quality embeddings."""
        tracker = QualityMetricsTracker(metrics_store)

        # Create some low-quality metrics
        low_quality_ids = tracker.identify_low_quality_embeddings(
            min_quality_score=0.6,
            limit=100
        )

        assert isinstance(low_quality_ids, list)

    def test_performance_profiling(self):
        """Validate performance profiling."""
        profiler = EmbeddingPerformanceProfiler()

        # Record some metrics
        profiler.profile_embedding_request(
            model_id="instructor-large",
            entity_type="Person",
            content_length=100,
            latency_ms=150,
            memory_mb=800
        )

        report = profiler.generate_performance_report()

        assert "avg_latency_ms" in report
        assert report["avg_latency_ms"] > 0
```

## Success Metrics

- ✅ Quality score computation functional for all entity types
- ✅ Low-quality embedding identification accurate (>90% precision)
- ✅ Re-embedding triggered automatically on quality improvement
- ✅ Performance profiling captures key metrics
- ✅ Latency targets met (<2s single, >100/min batch)

## Dependencies

- Experiential learning pipeline (from entity-relationship extraction)
- Quality metrics infrastructure
- Re-embedding service (from module 03)
- Performance monitoring tools

## Next Steps

After quality evolution complete:
1. Production testing and validation (06-integration-testing.md)
2. Performance benchmarking
3. Quality gates validation

**This module enables continuous quality improvement through experiential learning integration.**
