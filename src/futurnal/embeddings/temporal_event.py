"""Temporal Event Embedder - core component for temporal-aware embeddings.

Generates embeddings for temporal events that preserve:
- Content semantics (what happened)
- Temporal context (when it happened)
- Causal context (what led to it / what it caused)

Key differences from static entity embeddings:
- Incorporates temporal context into embedding
- Optimized for temporal similarity and causal pattern matching
- Uses weighted fusion of content, temporal, and causal embeddings

Option B Compliance:
- Uses frozen pre-trained models (no fine-tuning)
- Temporal context is REQUIRED (temporal-first design)
- Embeddings optimized for Phase 2 correlation detection and Phase 3 causal inference

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/01-temporal-aware-embeddings.md
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from futurnal.embeddings.base import BaseEmbedder, TimingContext
from futurnal.embeddings.exceptions import TemporalContextError
from futurnal.embeddings.fusion import EmbeddingFusion
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingResult,
    FusionWeights,
    TemporalEmbeddingContext,
)


class TemporalEventEmbedder(BaseEmbedder):
    """Embedder for temporal events with full context preservation.

    Embedding Strategy:
    1. Content embedding (60% weight) - event name + description
    2. Temporal embedding (30% weight) - timestamp + duration formatted as text
    3. Causal embedding (10% weight) - causal chain formatted as text
    4. Weighted fusion with L2 normalization

    This strategy ensures:
    - Events closer in time have higher similarity
    - Similar causal patterns are recognized across time
    - Content remains the primary semantic signal

    Example:
        manager = ModelManager(EmbeddingServiceConfig())
        embedder = TemporalEventEmbedder(manager)

        context = TemporalEmbeddingContext(
            timestamp=datetime(2024, 1, 15, 14, 30),
            duration=timedelta(hours=2),
            causal_chain=["Meeting", "Discussion"],
        )

        result = embedder.embed(
            event_name="Team Planning",
            event_description="Quarterly planning session",
            temporal_context=context,
        )
    """

    # Default model IDs (matching EmbeddingServiceConfig)
    DEFAULT_CONTENT_MODEL = "content-instructor"
    DEFAULT_TEMPORAL_MODEL = "temporal-minilm"

    def __init__(
        self,
        model_manager: ModelManager,
        fusion_weights: Optional[FusionWeights] = None,
        content_model_id: Optional[str] = None,
        temporal_model_id: Optional[str] = None,
    ) -> None:
        """Initialize the temporal event embedder.

        Args:
            model_manager: Manager for loading embedding models
            fusion_weights: Custom fusion weights (default: 60/30/10)
            content_model_id: ID of content embedding model
            temporal_model_id: ID of temporal context embedding model
        """
        super().__init__(model_manager)
        self._fusion_weights = fusion_weights or FusionWeights()
        self._fusion = EmbeddingFusion(self._fusion_weights)
        self._content_model_id = content_model_id or self.DEFAULT_CONTENT_MODEL
        self._temporal_model_id = temporal_model_id or self.DEFAULT_TEMPORAL_MODEL

    @property
    def entity_type(self) -> EmbeddingEntityType:
        """Return entity type for temporal events."""
        return EmbeddingEntityType.TEMPORAL_EVENT

    @property
    def fusion_weights(self) -> FusionWeights:
        """Current fusion weights."""
        return self._fusion_weights

    def embed(
        self,
        event_name: str,
        event_description: str,
        temporal_context: TemporalEmbeddingContext,
    ) -> EmbeddingResult:
        """Generate embedding for a temporal event.

        Combines content, temporal, and causal context into a single
        embedding that preserves all relevant semantics.

        Args:
            event_name: Name/title of the event
            event_description: Description of what happened
            temporal_context: REQUIRED temporal context (timestamp, duration, etc.)

        Returns:
            EmbeddingResult with fused embedding

        Raises:
            TemporalContextError: If temporal_context is missing or invalid
        """
        # Validate temporal context (Option B: temporal-first)
        if temporal_context is None:
            raise TemporalContextError(
                "temporal_context is REQUIRED for event embeddings "
                "(Option B: temporal-first design)"
            )

        with TimingContext() as timer:
            # 1. Generate content embedding
            content_text = self._format_content(event_name, event_description)
            content_embedding = self._encode_text(
                content_text,
                self._content_model_id,
                instruction="Represent the event for temporal retrieval:",
            )

            # 2. Generate temporal context embedding
            temporal_text = temporal_context.format_for_embedding()
            temporal_embedding = self._encode_text(
                temporal_text,
                self._temporal_model_id,
            )

            # 3. Generate causal context embedding (if available)
            causal_embedding = None
            has_causal_context = bool(temporal_context.causal_chain)
            if has_causal_context:
                causal_text = temporal_context.format_causal_chain()
                causal_embedding = self._encode_text(
                    causal_text,
                    self._content_model_id,
                    instruction="Represent the causal context:",
                )

            # 4. Fuse embeddings with weights
            fused = self._fusion.fuse(
                content_embedding,
                temporal_embedding,
                causal_embedding,
            )

            # 5. L2 normalize
            normalized = self._normalize_l2(fused)

        # Build result with metadata
        return self._build_result(
            embedding=normalized,
            model_id=self._content_model_id,
            generation_time_ms=timer.elapsed_ms,
            metadata={
                "event_name": event_name,
                "timestamp": temporal_context.timestamp.isoformat(),
                "has_duration": temporal_context.duration is not None,
                "has_causal_chain": has_causal_context,
                "causal_chain_length": len(temporal_context.causal_chain),
                "fusion_weights": {
                    "content": self._fusion_weights.content_weight,
                    "temporal": self._fusion_weights.temporal_weight,
                    "causal": self._fusion_weights.causal_weight,
                },
            },
            temporal_context_encoded=True,
            causal_context_encoded=has_causal_context,
        )

    def embed_content_only(
        self,
        event_name: str,
        event_description: str,
    ) -> np.ndarray:
        """Generate content-only embedding (for testing/comparison).

        Useful for comparing with full temporal-aware embeddings.

        Args:
            event_name: Name/title of the event
            event_description: Description of what happened

        Returns:
            Content embedding as numpy array
        """
        content_text = self._format_content(event_name, event_description)
        embedding = self._encode_text(
            content_text,
            self._content_model_id,
            instruction="Represent the event:",
        )
        return self._normalize_l2(embedding)

    def embed_temporal_only(
        self,
        temporal_context: TemporalEmbeddingContext,
    ) -> np.ndarray:
        """Generate temporal-only embedding (for testing/comparison).

        Useful for analyzing temporal similarity independent of content.

        Args:
            temporal_context: Temporal context to embed

        Returns:
            Temporal embedding as numpy array
        """
        temporal_text = temporal_context.format_for_embedding()
        embedding = self._encode_text(
            temporal_text,
            self._temporal_model_id,
        )
        return self._normalize_l2(embedding)

    def _format_content(self, event_name: str, description: str) -> str:
        """Format event content for embedding.

        Creates a coherent text representation of the event.

        Args:
            event_name: Event name/title
            description: Event description

        Returns:
            Formatted content string
        """
        parts = [f"Event: {event_name}"]
        if description and description.strip():
            parts.append(description.strip())
        return ". ".join(parts)

    def with_weights(self, weights: FusionWeights) -> "TemporalEventEmbedder":
        """Create a new embedder with different fusion weights.

        Useful for experimenting with different weight configurations.

        Args:
            weights: New fusion weights

        Returns:
            New TemporalEventEmbedder with specified weights
        """
        return TemporalEventEmbedder(
            model_manager=self._model_manager,
            fusion_weights=weights,
            content_model_id=self._content_model_id,
            temporal_model_id=self._temporal_model_id,
        )
