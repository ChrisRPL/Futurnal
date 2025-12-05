"""Tests for TemporalEventEmbedder.

Validates:
- Event embedding with temporal context
- Content formatting
- Temporal context formatting
- Causal chain encoding
- Result metadata
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from futurnal.embeddings.exceptions import TemporalContextError
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    FusionWeights,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.temporal_event import TemporalEventEmbedder


class TestTemporalEventEmbedder:
    """Tests for TemporalEventEmbedder class."""

    def test_entity_type(self, mock_model_manager):
        """Should return TEMPORAL_EVENT entity type."""
        embedder = TemporalEventEmbedder(mock_model_manager)
        assert embedder.entity_type == EmbeddingEntityType.TEMPORAL_EVENT

    def test_embed_basic(self, mock_model_manager, sample_temporal_context):
        """Should embed event with temporal context."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        result = embedder.embed(
            event_name="Team Meeting",
            event_description="Quarterly planning",
            temporal_context=sample_temporal_context,
        )

        assert result.entity_type == EmbeddingEntityType.TEMPORAL_EVENT
        assert result.temporal_context_encoded is True
        assert len(result.embedding) > 0
        assert result.generation_time_ms > 0

    def test_embed_with_causal_context(
        self, mock_model_manager, sample_temporal_context_with_causal
    ):
        """Should encode causal chain when present."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        result = embedder.embed(
            event_name="Decision",
            event_description="Project direction decided",
            temporal_context=sample_temporal_context_with_causal,
        )

        assert result.causal_context_encoded is True
        assert result.metadata["has_causal_chain"] is True
        assert result.metadata["causal_chain_length"] == 3

    def test_embed_without_causal_context(
        self, mock_model_manager, sample_temporal_context
    ):
        """Should work without causal chain."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        result = embedder.embed(
            event_name="Meeting",
            event_description="Team sync",
            temporal_context=sample_temporal_context,
        )

        assert result.causal_context_encoded is False
        assert result.metadata["has_causal_chain"] is False

    def test_embed_requires_temporal_context(self, mock_model_manager):
        """Should raise error if temporal_context is None (Option B)."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        with pytest.raises(TemporalContextError) as exc_info:
            embedder.embed(
                event_name="Meeting",
                event_description="Test",
                temporal_context=None,
            )

        assert "REQUIRED" in str(exc_info.value)
        assert "Option B" in str(exc_info.value)

    def test_embed_result_normalized(self, mock_model_manager, sample_temporal_context):
        """Embedding should be L2 normalized."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        result = embedder.embed(
            event_name="Test Event",
            event_description="Test description",
            temporal_context=sample_temporal_context,
        )

        # Check L2 norm is approximately 1
        embedding = np.array(result.embedding)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_embed_metadata(self, mock_model_manager, sample_temporal_context):
        """Should include expected metadata."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        result = embedder.embed(
            event_name="Important Meeting",
            event_description="Critical discussion",
            temporal_context=sample_temporal_context,
        )

        assert result.metadata["event_name"] == "Important Meeting"
        assert "timestamp" in result.metadata
        assert "fusion_weights" in result.metadata
        assert result.metadata["fusion_weights"]["content"] == 0.6

    def test_embed_model_version_tracked(
        self, mock_model_manager, sample_temporal_context
    ):
        """Should track model version for schema versioning."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        result = embedder.embed(
            event_name="Test",
            event_description="Test",
            temporal_context=sample_temporal_context,
        )

        assert result.model_version is not None
        assert "mock" in result.model_version.lower()

    def test_embed_custom_weights(self, mock_model_manager, sample_temporal_context):
        """Should use custom fusion weights."""
        custom_weights = FusionWeights(
            content_weight=0.4,
            temporal_weight=0.5,
            causal_weight=0.1,
        )
        embedder = TemporalEventEmbedder(
            mock_model_manager,
            fusion_weights=custom_weights,
        )

        result = embedder.embed(
            event_name="Test",
            event_description="Test",
            temporal_context=sample_temporal_context,
        )

        assert result.metadata["fusion_weights"]["temporal"] == 0.5

    def test_embed_content_only(self, mock_model_manager):
        """Should generate content-only embedding for comparison."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        embedding = embedder.embed_content_only(
            event_name="Test Event",
            event_description="Test description",
        )

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
        # Should be normalized
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01

    def test_embed_temporal_only(
        self, mock_model_manager, sample_temporal_context
    ):
        """Should generate temporal-only embedding for comparison."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        embedding = embedder.embed_temporal_only(sample_temporal_context)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01

    def test_with_weights_creates_new_embedder(self, mock_model_manager):
        """Should create new embedder with different weights."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        new_weights = FusionWeights(
            content_weight=0.5,
            temporal_weight=0.4,
            causal_weight=0.1,
        )
        new_embedder = embedder.with_weights(new_weights)

        assert new_embedder is not embedder
        assert new_embedder.fusion_weights.content_weight == 0.5
        assert embedder.fusion_weights.content_weight == 0.6

    def test_content_formatting(self, mock_model_manager, sample_temporal_context):
        """Test internal content formatting."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        formatted = embedder._format_content("Team Meeting", "Quarterly planning")
        assert "Event: Team Meeting" in formatted
        assert "Quarterly planning" in formatted

    def test_content_formatting_empty_description(
        self, mock_model_manager, sample_temporal_context
    ):
        """Should handle empty description."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        formatted = embedder._format_content("Meeting", "")
        assert "Event: Meeting" in formatted
        assert ". " not in formatted  # No trailing period for empty description


class TestTemporalEventEmbedderDeterminism:
    """Tests for embedding determinism."""

    def test_same_input_same_output(self, mock_model_manager):
        """Same inputs should produce same embeddings."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
        )

        result1 = embedder.embed(
            event_name="Meeting",
            event_description="Team sync",
            temporal_context=context,
        )
        result2 = embedder.embed(
            event_name="Meeting",
            event_description="Team sync",
            temporal_context=context,
        )

        np.testing.assert_array_almost_equal(
            result1.embedding, result2.embedding, decimal=5
        )

    def test_different_content_different_output(self, mock_model_manager):
        """Different content should produce different embeddings."""
        embedder = TemporalEventEmbedder(mock_model_manager)

        context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
        )

        result1 = embedder.embed(
            event_name="Meeting",
            event_description="Team sync",
            temporal_context=context,
        )
        result2 = embedder.embed(
            event_name="Conference",
            event_description="Industry event",
            temporal_context=context,
        )

        # Embeddings should be different
        assert not np.allclose(result1.embedding, result2.embedding)
