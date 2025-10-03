"""Experiential event models for temporal AI learning.

Experiential events represent timestamped occurrences in the user's data stream,
emphasizing temporal context and sequential relationships. Unlike static documents,
experiential events enable the Ghost to learn from patterns over time.

Phase 2 (Analyst) will use these models for:
- Temporal correlation detection (e.g., "Monday proposals have higher acceptance")
- Pattern recognition in behavioral sequences
- Causal hypothesis generation based on temporal proximity

Phase 3 (Guide) will extend with:
- Causal relationship confidence scoring
- Confounding factor analysis
- Counterfactual reasoning support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class ExperientialEvent:
    """Represents a timestamped event in the user's experience stream.

    Unlike static documents, experiential events emphasize:
    - **Temporal context**: When did this occur in the user's life?
    - **Event type**: What kind of experience does this represent?
    - **Sequential relationships**: What came before/after this event?

    This enables Phase 2/3 capabilities:
    - Temporal correlation detection
    - Causal pattern recognition
    - Behavioral sequence analysis

    Attributes:
        event_id: Unique identifier for this experiential event
        timestamp: When this event occurred in the user's timeline
        event_type: Category of experience (e.g., 'note_created', 'file_modified',
                   'email_sent', 'code_committed')
        source_uri: Reference to the source data (file path, note URI, etc.)
        context: Event-specific metadata and contextual information
        related_events: Links to temporally or causally related events (Phase 2/3)
        potential_causes: Events that may have caused this one (Phase 3)
        potential_effects: Events this one may have influenced (Phase 3)
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    source_uri: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Phase 2 prep: temporal correlation detection
    related_events: List[str] = field(default_factory=list)

    # Phase 3 prep: causal inference
    potential_causes: List[str] = field(default_factory=list)
    potential_effects: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate required fields."""
        if not self.event_type:
            raise ValueError("event_type is required for ExperientialEvent")
        if not self.source_uri:
            raise ValueError("source_uri is required for ExperientialEvent")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "source_uri": self.source_uri,
            "context": self.context,
            "related_events": self.related_events,
            "potential_causes": self.potential_causes,
            "potential_effects": self.potential_effects,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperientialEvent:
        """Create from dictionary."""
        data_copy = data.copy()
        if "timestamp" in data_copy and isinstance(data_copy["timestamp"], str):
            data_copy["timestamp"] = datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)


@dataclass
class ExperientialStream:
    """Collection of experiential events with temporal ordering.

    Provides utilities for querying and analyzing the user's experience stream,
    which is the primary data structure for Phase 2/3 temporal analysis.

    Phase 2 usage:
    - Query events by time range for correlation detection
    - Find event sequences for pattern recognition
    - Group events by type for statistical analysis

    Phase 3 usage:
    - Build temporal causal chains
    - Identify confounding factors across time
    - Support counterfactual "what if" queries

    Attributes:
        events: List of experiential events in temporal order
        stream_id: Identifier for this event stream (e.g., user_id, vault_id)
        metadata: Stream-level metadata
    """

    events: List[ExperientialEvent] = field(default_factory=list)
    stream_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: ExperientialEvent) -> None:
        """Add an event and maintain temporal ordering."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.timestamp)

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None
    ) -> List[ExperientialEvent]:
        """Query events within a time range.

        Phase 2 (Analyst): Used for temporal correlation detection.
        """
        filtered = [
            e for e in self.events
            if start <= e.timestamp <= end
        ]
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        return filtered

    def get_event_sequences(
        self,
        event_types: List[str],
        max_gap_seconds: int = 3600
    ) -> List[List[ExperientialEvent]]:
        """Find sequences of events matching the given types.

        Phase 2 (Analyst): Used for behavioral pattern recognition.

        Args:
            event_types: Sequence of event types to match
            max_gap_seconds: Maximum time gap between events in sequence

        Returns:
            List of event sequences matching the pattern
        """
        # TODO: Phase 2 implementation - pattern matching across temporal events
        sequences: List[List[ExperientialEvent]] = []
        return sequences

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "stream_id": self.stream_id,
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperientialStream:
        """Create from dictionary."""
        return cls(
            stream_id=data.get("stream_id", str(uuid4())),
            metadata=data.get("metadata", {}),
            events=[
                ExperientialEvent.from_dict(e)
                for e in data.get("events", [])
            ],
        )
