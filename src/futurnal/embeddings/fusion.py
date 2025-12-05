"""Embedding fusion strategies for combining multiple embeddings.

Implements weighted combination of:
- Content embedding (what happened) - 60% default
- Temporal embedding (when it happened) - 30% default
- Causal embedding (what led to it) - 10% default

The fusion strategy preserves temporal semantics by giving significant
weight to temporal context while maintaining content as the primary signal.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/01-temporal-aware-embeddings.md
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from futurnal.embeddings.exceptions import FusionError
from futurnal.embeddings.models import FusionWeights


class EmbeddingFusion:
    """Fuses multiple embeddings with configurable weights.

    Combines content, temporal, and causal embeddings into a single
    vector that preserves all relevant semantics.

    Default strategy (from production plan):
    - Content: 60% (what happened - primary semantic signal)
    - Temporal: 30% (when it happened - temporal context)
    - Causal: 10% (what led to it - causal relationships)

    Handles dimension mismatches by padding smaller embeddings to match
    the largest dimension.

    Example:
        fusion = EmbeddingFusion()  # Uses default weights

        # Fuse with all three embeddings
        result = fusion.fuse(
            content_embedding,
            temporal_embedding,
            causal_embedding,
        )

        # Fuse without causal (redistributes weight)
        result = fusion.fuse(
            content_embedding,
            temporal_embedding,
        )
    """

    def __init__(self, weights: Optional[FusionWeights] = None) -> None:
        """Initialize the fusion strategy.

        Args:
            weights: Custom fusion weights (defaults to 60/30/10 split)
        """
        self._weights = weights or FusionWeights()

    @property
    def weights(self) -> FusionWeights:
        """Current fusion weights."""
        return self._weights

    def fuse(
        self,
        content_embedding: np.ndarray,
        temporal_embedding: np.ndarray,
        causal_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fuse embeddings with configured weights.

        When causal_embedding is None, its weight is redistributed to
        content_embedding proportionally.

        Args:
            content_embedding: Content/semantic embedding
            temporal_embedding: Temporal context embedding
            causal_embedding: Optional causal chain embedding

        Returns:
            Fused embedding vector

        Raises:
            FusionError: If fusion fails
        """
        try:
            # Determine target dimension (use largest)
            dims = [content_embedding.shape[0], temporal_embedding.shape[0]]
            if causal_embedding is not None:
                dims.append(causal_embedding.shape[0])
            target_dim = max(dims)

            # Pad embeddings to target dimension
            content_padded = self._pad_to_dim(content_embedding, target_dim)
            temporal_padded = self._pad_to_dim(temporal_embedding, target_dim)

            if causal_embedding is not None:
                # Full fusion with all three embeddings
                causal_padded = self._pad_to_dim(causal_embedding, target_dim)
                combined = (
                    self._weights.content_weight * content_padded
                    + self._weights.temporal_weight * temporal_padded
                    + self._weights.causal_weight * causal_padded
                )
            else:
                # Redistribute causal weight to content
                # This maintains the relative importance of temporal context
                adjusted_content_weight = (
                    self._weights.content_weight + self._weights.causal_weight
                )
                combined = (
                    adjusted_content_weight * content_padded
                    + self._weights.temporal_weight * temporal_padded
                )

            return combined

        except Exception as e:
            raise FusionError(f"Failed to fuse embeddings: {e}") from e

    def _pad_to_dim(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad embedding to target dimension with zeros.

        Zero-padding preserves the original embedding's semantics while
        allowing combination with larger-dimensional embeddings.

        Args:
            embedding: Original embedding
            target_dim: Target dimension

        Returns:
            Padded embedding of target dimension
        """
        current_dim = embedding.shape[0]

        if current_dim == target_dim:
            return embedding
        elif current_dim < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:current_dim] = embedding
            return padded
        else:
            # Truncate if larger (shouldn't happen with max strategy)
            return embedding[:target_dim]

    def fuse_weighted(
        self,
        embeddings: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        """Fuse arbitrary number of embeddings with custom weights.

        General-purpose fusion for advanced use cases.

        Args:
            embeddings: List of embedding vectors
            weights: Corresponding weights (must sum to 1.0)

        Returns:
            Fused embedding vector

        Raises:
            FusionError: If inputs are invalid or fusion fails
        """
        if len(embeddings) != len(weights):
            raise FusionError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of weights ({len(weights)})"
            )

        if not embeddings:
            raise FusionError("Cannot fuse empty list of embeddings")

        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 0.001:
            raise FusionError(f"Weights must sum to 1.0, got {weight_sum:.3f}")

        try:
            # Find target dimension
            target_dim = max(e.shape[0] for e in embeddings)

            # Pad all embeddings
            padded = [self._pad_to_dim(e, target_dim) for e in embeddings]

            # Weighted sum
            combined = np.zeros(target_dim)
            for embedding, weight in zip(padded, weights):
                combined += weight * embedding

            return combined

        except Exception as e:
            raise FusionError(f"Failed to fuse embeddings: {e}") from e


class ConcatenationFusion:
    """Alternative fusion strategy using concatenation.

    Instead of weighted averaging, concatenates embeddings into a
    larger vector. This preserves all information but increases
    dimensionality.

    Useful when:
    - You want to preserve all embedding dimensions
    - Downstream models can handle larger vectors
    - Weighted averaging loses important distinctions
    """

    def fuse(
        self,
        content_embedding: np.ndarray,
        temporal_embedding: np.ndarray,
        causal_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Concatenate embeddings into a single vector.

        Args:
            content_embedding: Content/semantic embedding
            temporal_embedding: Temporal context embedding
            causal_embedding: Optional causal chain embedding

        Returns:
            Concatenated embedding vector
        """
        parts = [content_embedding, temporal_embedding]
        if causal_embedding is not None:
            parts.append(causal_embedding)

        return np.concatenate(parts)


class ProjectionFusion:
    """Fusion strategy with learned projection (for future enhancement).

    Projects embeddings through a learned transformation before combining.
    This can improve fusion quality but requires training data.

    Note: This is a placeholder for future enhancement. Current implementation
    falls back to weighted averaging since Option B requires frozen models
    (no training).
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: int,
        weights: Optional[FusionWeights] = None,
    ) -> None:
        """Initialize projection fusion.

        Args:
            input_dims: Dimensions of each input embedding
            output_dim: Target output dimension
            weights: Fusion weights (default: equal weighting)
        """
        self._input_dims = input_dims
        self._output_dim = output_dim
        self._weights = weights or FusionWeights()

        # Placeholder: In future, this could load learned projection matrices
        # For now, fall back to weighted averaging with padding
        self._fallback = EmbeddingFusion(self._weights)

    def fuse(
        self,
        content_embedding: np.ndarray,
        temporal_embedding: np.ndarray,
        causal_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fuse embeddings with projection (currently falls back to weighted).

        Args:
            content_embedding: Content/semantic embedding
            temporal_embedding: Temporal context embedding
            causal_embedding: Optional causal chain embedding

        Returns:
            Fused embedding vector of output_dim dimension
        """
        # Fall back to weighted fusion
        result = self._fallback.fuse(
            content_embedding,
            temporal_embedding,
            causal_embedding,
        )

        # Truncate or pad to output_dim if needed
        if len(result) > self._output_dim:
            return result[: self._output_dim]
        elif len(result) < self._output_dim:
            padded = np.zeros(self._output_dim)
            padded[: len(result)] = result
            return padded

        return result
