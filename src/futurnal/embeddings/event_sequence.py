"""Event Sequence Embedder for causal pattern matching.

Embeds sequences of events to enable:
- Phase 2 correlation detection
- Phase 3 causal inference
- Finding similar event patterns across time

Key features:
- Preserves temporal ordering through positional weighting
- Captures causal relationships between events
- Supports abstract pattern queries

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/01-temporal-aware-embeddings.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from futurnal.embeddings.base import BaseEmbedder, TimingContext
from futurnal.embeddings.exceptions import EmbeddingGenerationError
from futurnal.embeddings.manager import ModelManager
from futurnal.embeddings.models import (
    EmbeddingEntityType,
    EmbeddingResult,
    TemporalEmbeddingContext,
)
from futurnal.embeddings.temporal_event import TemporalEventEmbedder


@runtime_checkable
class EventLike(Protocol):
    """Protocol for event-like objects.

    Matches the Event class from futurnal.extraction.temporal.models
    while allowing flexibility for different event representations.
    """

    @property
    def name(self) -> str:
        """Event name."""
        ...

    @property
    def event_type(self) -> str:
        """Event type."""
        ...


class EventSequenceEmbedder(BaseEmbedder):
    """Embed sequences of events for causal pattern matching.

    Generates embeddings for event sequences that preserve:
    - Temporal ordering (earlier events get higher weight)
    - Causal relationships between events
    - Event semantics

    Use cases:
    - Finding similar event patterns: "Meeting -> Decision -> Publication"
    - Correlation detection across time periods
    - Causal hypothesis exploration

    Example:
        manager = ModelManager(EmbeddingServiceConfig())
        embedder = EventSequenceEmbedder(manager)

        events = [
            Event(name="Meeting", event_type="meeting", ...),
            Event(name="Decision", event_type="decision", ...),
            Event(name="Publication", event_type="publication", ...),
        ]
        contexts = [
            TemporalEmbeddingContext(timestamp=datetime(2024, 1, 1)),
            TemporalEmbeddingContext(timestamp=datetime(2024, 1, 2)),
            TemporalEmbeddingContext(timestamp=datetime(2024, 1, 3)),
        ]

        result = embedder.embed(events, contexts)
    """

    DEFAULT_CONTENT_MODEL = "content-instructor"

    def __init__(
        self,
        model_manager: ModelManager,
        event_embedder: Optional[TemporalEventEmbedder] = None,
        content_model_id: Optional[str] = None,
        decay_factor: float = 0.9,
    ) -> None:
        """Initialize the event sequence embedder.

        Args:
            model_manager: Manager for loading embedding models
            event_embedder: Optional pre-configured event embedder
            content_model_id: ID of content model for pattern embedding
            decay_factor: Weight decay for temporal ordering (default: 0.9)
        """
        super().__init__(model_manager)
        self._event_embedder = event_embedder or TemporalEventEmbedder(model_manager)
        self._content_model_id = content_model_id or self.DEFAULT_CONTENT_MODEL
        self._decay_factor = decay_factor

    @property
    def entity_type(self) -> EmbeddingEntityType:
        """Return entity type for event sequences (treated as temporal)."""
        return EmbeddingEntityType.TEMPORAL_EVENT

    @property
    def decay_factor(self) -> float:
        """Weight decay factor for positional weighting."""
        return self._decay_factor

    def embed(
        self,
        events: List[Any],
        temporal_contexts: List[TemporalEmbeddingContext],
    ) -> EmbeddingResult:
        """Generate embedding for a sequence of events.

        Strategy:
        1. Embed each event with its temporal context
        2. Aggregate with positional weighting (earlier = higher weight)
        3. L2 normalize

        Args:
            events: List of events in temporal order (must have name attribute)
            temporal_contexts: Corresponding temporal contexts (same length)

        Returns:
            EmbeddingResult representing the sequence

        Raises:
            ValueError: If events and contexts have different lengths
            EmbeddingGenerationError: If embedding fails
        """
        if len(events) != len(temporal_contexts):
            raise ValueError(
                f"Events ({len(events)}) and temporal_contexts "
                f"({len(temporal_contexts)}) must have same length"
            )

        if not events:
            raise ValueError("Cannot embed empty event sequence")

        with TimingContext() as timer:
            # Embed each event
            event_embeddings = []
            for event, context in zip(events, temporal_contexts):
                # Get event name and description
                event_name = getattr(event, "name", str(event))
                event_description = getattr(event, "description", "")

                result = self._event_embedder.embed(
                    event_name=event_name,
                    event_description=event_description,
                    temporal_context=context,
                )
                event_embeddings.append(np.array(result.embedding))

            # Aggregate with positional weighting
            sequence_embedding = self._aggregate_sequence(event_embeddings)

            # L2 normalize
            normalized = self._normalize_l2(sequence_embedding)

        # Extract event names for metadata
        event_names = [
            getattr(e, "name", str(e)) for e in events
        ]

        # Check if any event has causal context
        has_causal = any(bool(ctx.causal_chain) for ctx in temporal_contexts)

        return self._build_result(
            embedding=normalized,
            model_id="sequence-aggregate",
            generation_time_ms=timer.elapsed_ms,
            metadata={
                "sequence_length": len(events),
                "event_names": event_names,
                "event_pattern": " -> ".join(event_names),
                "decay_factor": self._decay_factor,
            },
            temporal_context_encoded=True,
            causal_context_encoded=has_causal,
        )

    def embed_pattern(
        self,
        event_types: List[str],
        descriptions: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """Embed an abstract event pattern for similarity search.

        Used for queries like:
        "Find patterns similar to: Meeting -> Decision -> Publication"

        This embeds the abstract pattern without specific timestamps,
        useful for finding similar patterns across different time periods.

        Args:
            event_types: List of event types (e.g., ["meeting", "decision"])
            descriptions: Optional descriptions for each event type

        Returns:
            EmbeddingResult for the abstract pattern
        """
        with TimingContext() as timer:
            # Create pattern text
            if descriptions:
                pattern_parts = []
                for et, desc in zip(event_types, descriptions):
                    if desc:
                        pattern_parts.append(f"{et}: {desc}")
                    else:
                        pattern_parts.append(et)
                pattern_text = " -> ".join(pattern_parts)
            else:
                pattern_text = " -> ".join(event_types)

            # Embed pattern
            embedding = self._encode_text(
                pattern_text,
                self._content_model_id,
                instruction="Represent the event pattern for similarity search:",
            )

            normalized = self._normalize_l2(embedding)

        return self._build_result(
            embedding=normalized,
            model_id=self._content_model_id,
            generation_time_ms=timer.elapsed_ms,
            metadata={
                "pattern": event_types,
                "pattern_text": pattern_text,
                "is_abstract_pattern": True,
            },
            temporal_context_encoded=False,
            causal_context_encoded=False,
        )

    def embed_from_event_names(
        self,
        event_names: List[str],
        timestamps: List[datetime],
        descriptions: Optional[List[str]] = None,
    ) -> EmbeddingResult:
        """Convenience method to embed from simple event data.

        Args:
            event_names: List of event names in order
            timestamps: Corresponding timestamps
            descriptions: Optional descriptions

        Returns:
            EmbeddingResult for the sequence
        """
        # Create simple event objects
        events = [
            _SimpleEvent(name=name, description=desc or "")
            for name, desc in zip(
                event_names,
                descriptions or [""] * len(event_names),
            )
        ]

        # Create temporal contexts
        contexts = [
            TemporalEmbeddingContext(timestamp=ts)
            for ts in timestamps
        ]

        return self.embed(events, contexts)

    def _aggregate_sequence(
        self,
        embeddings: List[np.ndarray],
    ) -> np.ndarray:
        """Aggregate event embeddings with temporal decay weighting.

        Earlier events get higher weight because they set the stage
        for later events in a causal sequence.

        Weight calculation:
        - First event: weight = 1.0
        - Second event: weight = decay_factor
        - Third event: weight = decay_factor^2
        - etc.

        Args:
            embeddings: List of event embeddings in temporal order

        Returns:
            Aggregated embedding
        """
        if not embeddings:
            raise EmbeddingGenerationError("Cannot aggregate empty embedding list")

        n = len(embeddings)

        # Calculate weights with temporal decay
        weights = np.array([self._decay_factor**i for i in range(n)])
        weights = weights / weights.sum()  # Normalize to sum to 1

        # Find max dimension
        max_dim = max(e.shape[0] for e in embeddings)

        # Pad embeddings to same dimension and compute weighted sum
        aggregated = np.zeros(max_dim)
        for embedding, weight in zip(embeddings, weights):
            if embedding.shape[0] < max_dim:
                padded = np.zeros(max_dim)
                padded[: embedding.shape[0]] = embedding
                aggregated += weight * padded
            else:
                aggregated += weight * embedding

        return aggregated

    def with_decay_factor(self, decay_factor: float) -> "EventSequenceEmbedder":
        """Create a new embedder with different decay factor.

        Args:
            decay_factor: New decay factor (0.0 to 1.0)

        Returns:
            New EventSequenceEmbedder with specified decay
        """
        return EventSequenceEmbedder(
            model_manager=self._model_manager,
            event_embedder=self._event_embedder,
            content_model_id=self._content_model_id,
            decay_factor=decay_factor,
        )


class _SimpleEvent:
    """Simple event class for convenience methods."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
