"""Tests for embedding data models.

Validates:
- Pydantic model validation
- Temporal context formatting
- Fusion weight constraints
- Result construction
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingQuery,
    EmbeddingResult,
    FusionWeights,
    SimilarityResult,
    TemporalEmbeddingContext,
)


class TestEmbeddingEntityType:
    """Tests for EmbeddingEntityType enum."""

    def test_entity_types_defined(self):
        """All expected entity types should be defined."""
        assert EmbeddingEntityType.STATIC_ENTITY == "static_entity"
        assert EmbeddingEntityType.TEMPORAL_EVENT == "temporal_event"
        assert EmbeddingEntityType.TEMPORAL_RELATIONSHIP == "temporal_relationship"
        assert EmbeddingEntityType.CODE_ENTITY == "code_entity"
        assert EmbeddingEntityType.DOCUMENT == "document"


class TestTemporalEmbeddingContext:
    """Tests for TemporalEmbeddingContext model."""

    def test_valid_context_with_timestamp(self):
        """Should accept valid context with required timestamp."""
        ctx = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
        )
        assert ctx.timestamp == datetime(2024, 1, 15, 14, 30)
        assert ctx.duration is None
        assert ctx.temporal_type is None
        assert ctx.causal_chain == []

    def test_full_context(self):
        """Should accept full context with all fields."""
        ctx = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
            duration=timedelta(hours=2),
            temporal_type="DURING",
            event_sequence=["evt1", "evt2"],
            causal_chain=["Meeting", "Decision"],
        )
        assert ctx.duration == timedelta(hours=2)
        assert ctx.temporal_type == "DURING"
        assert len(ctx.event_sequence) == 2
        assert len(ctx.causal_chain) == 2

    def test_missing_timestamp_raises(self):
        """Should raise error if timestamp is missing (Option B requirement)."""
        with pytest.raises(ValidationError) as exc_info:
            TemporalEmbeddingContext(timestamp=None)
        assert "timestamp" in str(exc_info.value)

    def test_format_for_embedding(self):
        """Should format context correctly for embedding."""
        ctx = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
            duration=timedelta(hours=2),
            temporal_type="BEFORE",
        )
        formatted = ctx.format_for_embedding()
        assert "2024-01-15" in formatted
        assert "14:30" in formatted
        assert "2.0 hours" in formatted
        assert "BEFORE" in formatted

    def test_format_for_embedding_minutes(self):
        """Should format short duration in minutes."""
        ctx = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15),
            duration=timedelta(minutes=30),
        )
        formatted = ctx.format_for_embedding()
        assert "30 minutes" in formatted

    def test_format_causal_chain(self):
        """Should format causal chain correctly."""
        ctx = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15),
            causal_chain=["Meeting", "Discussion", "Decision"],
        )
        formatted = ctx.format_causal_chain()
        assert "causal context:" in formatted
        assert "Meeting -> Discussion -> Decision" in formatted

    def test_format_empty_causal_chain(self):
        """Should return empty string for empty causal chain."""
        ctx = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15),
        )
        assert ctx.format_causal_chain() == ""


class TestFusionWeights:
    """Tests for FusionWeights model."""

    def test_default_weights(self):
        """Default weights should be 60/30/10."""
        weights = FusionWeights()
        assert weights.content_weight == 0.6
        assert weights.temporal_weight == 0.3
        assert weights.causal_weight == 0.1

    def test_custom_weights(self):
        """Should accept custom weights that sum to 1."""
        weights = FusionWeights(
            content_weight=0.5,
            temporal_weight=0.4,
            causal_weight=0.1,
        )
        assert weights.content_weight == 0.5
        assert weights.temporal_weight == 0.4

    def test_weights_must_sum_to_one(self):
        """Should raise error if weights don't sum to 1."""
        with pytest.raises(ValidationError) as exc_info:
            FusionWeights(
                content_weight=0.5,
                temporal_weight=0.3,
                causal_weight=0.1,
            )
        assert "sum to 1.0" in str(exc_info.value)

    def test_weights_must_be_positive(self):
        """Should raise error if weights are negative."""
        with pytest.raises(ValidationError):
            FusionWeights(
                content_weight=-0.1,
                temporal_weight=0.5,
                causal_weight=0.6,
            )

    def test_weights_must_be_at_most_one(self):
        """Should raise error if any weight exceeds 1."""
        with pytest.raises(ValidationError):
            FusionWeights(
                content_weight=1.1,
                temporal_weight=0.0,
                causal_weight=0.0,
            )


class TestEmbeddingResult:
    """Tests for EmbeddingResult model."""

    def test_valid_result(self):
        """Should accept valid embedding result."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3, 0.4],
            entity_type=EmbeddingEntityType.TEMPORAL_EVENT,
            model_version="test:v1",
            embedding_dimension=4,
            generation_time_ms=10.5,
        )
        assert len(result.embedding) == 4
        assert result.entity_type == EmbeddingEntityType.TEMPORAL_EVENT
        assert result.embedding_dimension == 4
        assert result.temporal_context_encoded is False
        assert result.causal_context_encoded is False

    def test_result_with_context_flags(self):
        """Should track temporal and causal context encoding."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            entity_type=EmbeddingEntityType.TEMPORAL_EVENT,
            model_version="test:v1",
            embedding_dimension=3,
            generation_time_ms=10.0,
            temporal_context_encoded=True,
            causal_context_encoded=True,
        )
        assert result.temporal_context_encoded is True
        assert result.causal_context_encoded is True

    def test_result_with_metadata(self):
        """Should store additional metadata."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2],
            entity_type=EmbeddingEntityType.STATIC_ENTITY,
            model_version="test:v1",
            embedding_dimension=2,
            generation_time_ms=5.0,
            metadata={"event_name": "Meeting", "source": "test"},
        )
        assert result.metadata["event_name"] == "Meeting"


class TestEmbeddingQuery:
    """Tests for EmbeddingQuery model."""

    def test_query_with_embedding(self):
        """Should accept query with embedding."""
        query = EmbeddingQuery(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )
        assert len(query.query_embedding) == 3
        assert query.top_k == 5

    def test_query_with_text(self):
        """Should accept query with text."""
        query = EmbeddingQuery(
            query_text="Find similar events",
            top_k=10,
        )
        assert query.query_text == "Find similar events"

    def test_query_requires_embedding_or_text(self):
        """Should raise error if neither embedding nor text provided."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingQuery(top_k=10)
        assert "query_embedding or query_text" in str(exc_info.value)

    def test_query_with_filters(self):
        """Should accept filters."""
        query = EmbeddingQuery(
            query_embedding=[0.1, 0.2],
            entity_type_filter=EmbeddingEntityType.TEMPORAL_EVENT,
            min_similarity=0.5,
        )
        assert query.entity_type_filter == EmbeddingEntityType.TEMPORAL_EVENT
        assert query.min_similarity == 0.5


class TestSimilarityResult:
    """Tests for SimilarityResult model."""

    def test_valid_similarity_result(self):
        """Should accept valid similarity result."""
        result = SimilarityResult(
            entity_id="evt_123",
            similarity_score=0.85,
            entity_type=EmbeddingEntityType.TEMPORAL_EVENT,
            metadata={"event_name": "Meeting"},
        )
        assert result.entity_id == "evt_123"
        assert result.similarity_score == 0.85

    def test_similarity_score_bounds(self):
        """Similarity score should be between 0 and 1."""
        with pytest.raises(ValidationError):
            SimilarityResult(
                entity_id="test",
                similarity_score=1.5,
                entity_type=EmbeddingEntityType.STATIC_ENTITY,
            )
