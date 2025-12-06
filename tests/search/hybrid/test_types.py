"""Tests for hybrid search types and models.

Tests Pydantic model validation and enum values.
"""

from datetime import datetime, timedelta

import pytest

from futurnal.search.hybrid.types import (
    GraphSearchResult,
    HybridSearchQuery,
    HybridSearchResult,
    QueryEmbeddingType,
    SchemaCompatibilityResult,
    TemporalQueryContext,
    VectorSearchResult,
)


class TestQueryEmbeddingType:
    """Test QueryEmbeddingType enum."""

    def test_all_values_present(self):
        """Ensure all expected values are present."""
        assert QueryEmbeddingType.GENERAL == "general"
        assert QueryEmbeddingType.TEMPORAL == "temporal"
        assert QueryEmbeddingType.CAUSAL == "causal"
        assert QueryEmbeddingType.CODE == "code"
        assert QueryEmbeddingType.DOCUMENT == "document"

    def test_enum_count(self):
        """Ensure correct number of embedding types."""
        assert len(QueryEmbeddingType) == 5


class TestTemporalQueryContext:
    """Test TemporalQueryContext model."""

    def test_empty_context(self):
        """Empty context is valid."""
        ctx = TemporalQueryContext()
        assert ctx.time_range_start is None
        assert ctx.time_range_end is None

    def test_time_range_context(self):
        """Time range context."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)
        ctx = TemporalQueryContext(
            time_range_start=start,
            time_range_end=end,
        )
        assert ctx.time_range_start == start
        assert ctx.time_range_end == end

    def test_format_for_embedding_with_time_range(self):
        """format_for_embedding includes time range."""
        ctx = TemporalQueryContext(
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2024, 3, 31),
        )
        formatted = ctx.format_for_embedding()
        assert "Time:" in formatted
        assert "2024-01-01" in formatted

    def test_format_for_embedding_with_relation(self):
        """format_for_embedding includes relation."""
        ctx = TemporalQueryContext(
            temporal_relation="BEFORE",
        )
        formatted = ctx.format_for_embedding()
        assert "Relation: BEFORE" in formatted

    def test_format_for_embedding_empty(self):
        """format_for_embedding returns empty for empty context."""
        ctx = TemporalQueryContext()
        assert ctx.format_for_embedding() == ""


class TestVectorSearchResult:
    """Test VectorSearchResult model."""

    def test_valid_result(self):
        """Valid result creation."""
        result = VectorSearchResult(
            entity_id="test_1",
            entity_type="Event",
            content="Test content",
            similarity_score=0.85,
            schema_version=5,
            metadata={"key": "value"},
        )
        assert result.entity_id == "test_1"
        assert result.similarity_score == 0.85

    def test_score_validation_high(self):
        """Scores above 1.0 raise validation error."""
        with pytest.raises(ValueError):
            VectorSearchResult(
                entity_id="test_1",
                entity_type="Event",
                similarity_score=1.5,
            )

    def test_score_validation_low(self):
        """Scores below 0.0 raise validation error."""
        with pytest.raises(ValueError):
            VectorSearchResult(
                entity_id="test_1",
                entity_type="Event",
                similarity_score=-0.5,
            )


class TestGraphSearchResult:
    """Test GraphSearchResult model."""

    def test_valid_result(self):
        """Valid result creation."""
        result = GraphSearchResult(
            entity_id="test_1",
            entity_type="Event",
            path_from_seed=["seed_1", "test_1"],
            path_score=0.75,
            relationship_types=["CAUSES"],
        )
        assert result.entity_id == "test_1"
        assert result.path_score == 0.75
        assert len(result.path_from_seed) == 2

    def test_default_values(self):
        """Default values are applied."""
        result = GraphSearchResult(
            entity_id="test_1",
            entity_type="Event",
            path_score=0.5,
        )
        assert result.path_from_seed == []
        assert result.relationship_types == []
        assert result.metadata == {}


class TestHybridSearchResult:
    """Test HybridSearchResult model."""

    def test_valid_result(self):
        """Valid result creation."""
        result = HybridSearchResult(
            entity_id="test_1",
            entity_type="Event",
            vector_score=0.8,
            graph_score=0.6,
            combined_score=0.7,
            source="hybrid",
        )
        assert result.entity_id == "test_1"
        assert result.combined_score == 0.7
        assert result.source == "hybrid"

    def test_default_values(self):
        """Default values are applied."""
        result = HybridSearchResult(
            entity_id="test_1",
            entity_type="Event",
            combined_score=0.5,
        )
        assert result.vector_score == 0.0
        assert result.graph_score == 0.0
        assert result.source == "hybrid"
        assert result.schema_version is None


class TestHybridSearchQuery:
    """Test HybridSearchQuery model."""

    def test_valid_query(self):
        """Valid query creation."""
        query = HybridSearchQuery(
            query_text="What happened?",
            intent="temporal",
            top_k=10,
        )
        assert query.query_text == "What happened?"
        assert query.intent == "temporal"
        assert query.top_k == 10

    def test_intent_validation_valid(self):
        """Valid intents are accepted."""
        for intent in ["temporal", "causal", "lookup", "exploratory"]:
            query = HybridSearchQuery(
                query_text="test",
                intent=intent,
            )
            assert query.intent == intent

    def test_intent_validation_invalid(self):
        """Invalid intent raises error."""
        with pytest.raises(ValueError, match="Invalid intent"):
            HybridSearchQuery(
                query_text="test",
                intent="invalid_intent",
            )

    def test_weight_normalization(self):
        """Weights are normalized to sum to 1."""
        query = HybridSearchQuery(
            query_text="test",
            vector_weight=0.3,
            graph_weight=0.3,
        )
        # After normalization: 0.3 / 0.6 = 0.5 each
        assert abs(query.vector_weight - 0.5) < 0.01
        assert abs(query.graph_weight - 0.5) < 0.01

    def test_defaults(self):
        """Default values are applied."""
        query = HybridSearchQuery(query_text="test")
        assert query.intent == "exploratory"
        assert query.top_k == 20
        assert query.vector_weight == 0.5
        assert query.graph_weight == 0.5


class TestSchemaCompatibilityResult:
    """Test SchemaCompatibilityResult model."""

    def test_full_compatibility(self):
        """Full compatibility result."""
        result = SchemaCompatibilityResult(
            compatible=True,
            score_factor=1.0,
            drift_level="none",
            version_diff=0,
        )
        assert result.compatible
        assert result.score_factor == 1.0
        assert not result.reembedding_required

    def test_incompatible_result(self):
        """Incompatible result requiring re-embedding."""
        result = SchemaCompatibilityResult(
            compatible=False,
            score_factor=0.0,
            reembedding_required=True,
            drift_level="severe",
            version_diff=5,
        )
        assert not result.compatible
        assert result.reembedding_required
        assert result.version_diff == 5
