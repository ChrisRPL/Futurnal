"""Tests for embedding fusion strategies.

Validates:
- Weighted fusion with default and custom weights
- Dimension mismatch handling
- Causal weight redistribution
"""

import numpy as np
import pytest

from futurnal.embeddings.exceptions import FusionError
from futurnal.embeddings.fusion import (
    ConcatenationFusion,
    EmbeddingFusion,
    ProjectionFusion,
)
from futurnal.embeddings.models import FusionWeights


class TestEmbeddingFusion:
    """Tests for EmbeddingFusion class."""

    def test_default_weights(self):
        """Should use default weights (60/30/10)."""
        fusion = EmbeddingFusion()
        assert fusion.weights.content_weight == 0.6
        assert fusion.weights.temporal_weight == 0.3
        assert fusion.weights.causal_weight == 0.1

    def test_custom_weights(self):
        """Should accept custom weights."""
        weights = FusionWeights(
            content_weight=0.5,
            temporal_weight=0.4,
            causal_weight=0.1,
        )
        fusion = EmbeddingFusion(weights)
        assert fusion.weights.content_weight == 0.5

    def test_fuse_same_dimension(self):
        """Should fuse embeddings of same dimension."""
        fusion = EmbeddingFusion()

        content = np.array([1.0, 0.0, 0.0])
        temporal = np.array([0.0, 1.0, 0.0])
        causal = np.array([0.0, 0.0, 1.0])

        result = fusion.fuse(content, temporal, causal)

        # Check weighted combination
        expected = 0.6 * content + 0.3 * temporal + 0.1 * causal
        np.testing.assert_array_almost_equal(result, expected)

    def test_fuse_without_causal(self):
        """Should redistribute causal weight to content when no causal."""
        fusion = EmbeddingFusion()

        content = np.array([1.0, 0.0, 0.0])
        temporal = np.array([0.0, 1.0, 0.0])

        result = fusion.fuse(content, temporal)

        # Causal weight (0.1) should be added to content (0.6 -> 0.7)
        expected = 0.7 * content + 0.3 * temporal
        np.testing.assert_array_almost_equal(result, expected)

    def test_fuse_dimension_mismatch(self):
        """Should handle dimension mismatch via padding."""
        fusion = EmbeddingFusion()

        content = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # 5D
        temporal = np.array([0.0, 1.0, 0.0])  # 3D
        causal = np.array([0.0, 0.0, 1.0, 0.0])  # 4D

        result = fusion.fuse(content, temporal, causal)

        # Result should be 5D (max dimension)
        assert result.shape[0] == 5

    def test_fuse_result_magnitude(self):
        """Fused embedding should have reasonable magnitude."""
        fusion = EmbeddingFusion()

        content = np.random.randn(384)
        content = content / np.linalg.norm(content)

        temporal = np.random.randn(384)
        temporal = temporal / np.linalg.norm(temporal)

        result = fusion.fuse(content, temporal)

        # Result magnitude should be between 0 and sqrt(sum of squared weights)
        assert np.linalg.norm(result) <= 1.5

    def test_fuse_weighted_general(self):
        """Should fuse arbitrary number of embeddings."""
        fusion = EmbeddingFusion()

        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
        ]
        weights = [0.4, 0.3, 0.2, 0.1]

        result = fusion.fuse_weighted(embeddings, weights)

        expected = (
            0.4 * embeddings[0]
            + 0.3 * embeddings[1]
            + 0.2 * embeddings[2]
            + 0.1 * embeddings[3]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_fuse_weighted_mismatched_lengths(self):
        """Should raise error if embeddings and weights have different lengths."""
        fusion = EmbeddingFusion()

        embeddings = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        weights = [0.5, 0.3, 0.2]  # Wrong length

        with pytest.raises(FusionError) as exc_info:
            fusion.fuse_weighted(embeddings, weights)
        assert "must match" in str(exc_info.value)

    def test_fuse_weighted_weights_not_sum_to_one(self):
        """Should raise error if weights don't sum to 1."""
        fusion = EmbeddingFusion()

        embeddings = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        weights = [0.5, 0.3]  # Sum = 0.8

        with pytest.raises(FusionError) as exc_info:
            fusion.fuse_weighted(embeddings, weights)
        assert "sum to 1.0" in str(exc_info.value)

    def test_fuse_weighted_empty_list(self):
        """Should raise error for empty embedding list."""
        fusion = EmbeddingFusion()

        with pytest.raises(FusionError) as exc_info:
            fusion.fuse_weighted([], [])
        assert "empty" in str(exc_info.value).lower()


class TestConcatenationFusion:
    """Tests for ConcatenationFusion class."""

    def test_concatenate_embeddings(self):
        """Should concatenate embeddings into single vector."""
        fusion = ConcatenationFusion()

        content = np.array([1.0, 2.0])
        temporal = np.array([3.0, 4.0])
        causal = np.array([5.0, 6.0])

        result = fusion.fuse(content, temporal, causal)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_concatenate_without_causal(self):
        """Should concatenate without causal embedding."""
        fusion = ConcatenationFusion()

        content = np.array([1.0, 2.0])
        temporal = np.array([3.0, 4.0])

        result = fusion.fuse(content, temporal)

        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result, expected)

    def test_concatenate_dimension(self):
        """Result dimension should be sum of input dimensions."""
        fusion = ConcatenationFusion()

        content = np.random.randn(384)
        temporal = np.random.randn(256)
        causal = np.random.randn(384)

        result = fusion.fuse(content, temporal, causal)

        assert result.shape[0] == 384 + 256 + 384


class TestProjectionFusion:
    """Tests for ProjectionFusion class."""

    def test_projection_uses_fallback(self):
        """Should fall back to weighted fusion (Option B: no training)."""
        fusion = ProjectionFusion(
            input_dims=[384, 256, 384],
            output_dim=512,
        )

        content = np.random.randn(384)
        temporal = np.random.randn(256)

        result = fusion.fuse(content, temporal)

        # Should produce output of target dimension
        assert result.shape[0] == 512

    def test_projection_truncates_if_larger(self):
        """Should truncate if fused result exceeds output_dim."""
        fusion = ProjectionFusion(
            input_dims=[768, 384],
            output_dim=256,
        )

        content = np.random.randn(768)
        temporal = np.random.randn(384)

        result = fusion.fuse(content, temporal)

        assert result.shape[0] == 256

    def test_projection_pads_if_smaller(self):
        """Should pad if fused result is smaller than output_dim."""
        fusion = ProjectionFusion(
            input_dims=[128, 128],
            output_dim=512,
        )

        content = np.random.randn(128)
        temporal = np.random.randn(128)

        result = fusion.fuse(content, temporal)

        assert result.shape[0] == 512
