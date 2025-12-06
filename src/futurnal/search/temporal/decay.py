"""Temporal Decay Scoring Module.

Implements exponential decay scoring for temporal search results,
weighting recent events higher than older ones.

Formula: score = base_score * exp(-ln(2) / half_life * days_since_event)

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md

Option B Compliance:
- Configurable half-life for different use cases
- Preserves original scores for audit/analysis
- Supports experiential learning feedback integration
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

from futurnal.pkg.schema.models import EventNode
from futurnal.search.config import TemporalSearchConfig
from futurnal.search.temporal.exceptions import DecayScoringError
from futurnal.search.temporal.results import ScoredEvent

logger = logging.getLogger(__name__)


class TemporalDecayScorer:
    """Exponential decay scoring for temporal search results.

    Applies time-based decay to event scores, weighting recent
    events higher than older ones using an exponential decay function.

    The decay function uses a configurable half-life:
    - At half_life days old, score is 50% of original
    - At 2*half_life days old, score is 25% of original
    - etc.

    Example:
        >>> scorer = TemporalDecayScorer(half_life_days=30.0)
        >>> # Score an event from 30 days ago
        >>> score = scorer.score(event, base_score=1.0)
        >>> print(f"Score: {score:.3f}")  # ~0.5

        >>> # Apply decay to list of events
        >>> scored = scorer.apply_decay(events)
        >>> for event, score in scored[:5]:
        ...     print(f"{event.name}: {score:.3f}")

    Attributes:
        half_life_days: Number of days for score to decay by 50%
        decay_constant: Lambda value in exp(-lambda * days)
    """

    def __init__(
        self,
        half_life_days: float = 30.0,
        config: Optional[TemporalSearchConfig] = None,
    ):
        """Initialize the decay scorer.

        Args:
            half_life_days: Days for score to decay to 50%. Default: 30
            config: Optional config to read half_life from. Overrides half_life_days.
        """
        if config is not None:
            half_life_days = config.decay_half_life_days

        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")

        self._half_life = half_life_days
        # Lambda = ln(2) / half_life
        # This ensures: exp(-lambda * half_life) = 0.5
        self._lambda = math.log(2) / half_life_days

        logger.debug(
            f"Initialized TemporalDecayScorer with half_life={half_life_days} days, "
            f"lambda={self._lambda:.6f}"
        )

    @property
    def half_life_days(self) -> float:
        """Half-life in days."""
        return self._half_life

    @property
    def decay_constant(self) -> float:
        """Decay constant (lambda) for the exponential function."""
        return self._lambda

    def score(
        self,
        event: EventNode,
        reference_time: Optional[datetime] = None,
        base_score: float = 1.0,
    ) -> float:
        """Calculate decay-weighted score for an event.

        Args:
            event: Event to score
            reference_time: Reference time for decay calculation.
                          Defaults to current UTC time.
            base_score: Base relevance score before decay (default 1.0)

        Returns:
            Decay-weighted score in range [0, base_score]

        Raises:
            DecayScoringError: If event has invalid timestamp
        """
        if reference_time is None:
            reference_time = datetime.utcnow()

        # Ensure timestamps are comparable (strip timezone)
        event_time = self._normalize_datetime(event.timestamp)
        ref_time = self._normalize_datetime(reference_time)

        # Calculate days since event
        time_delta = ref_time - event_time
        days_since = time_delta.total_seconds() / 86400.0  # seconds per day

        # Future events get full score (no penalty)
        if days_since < 0:
            return base_score

        # Apply exponential decay
        decay_factor = math.exp(-self._lambda * days_since)
        return base_score * decay_factor

    def compute_decay_factor(
        self,
        event_time: datetime,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """Compute just the decay factor (0-1) for a timestamp.

        Args:
            event_time: Time of the event
            reference_time: Reference time. Defaults to current UTC time.

        Returns:
            Decay factor in range (0, 1]
        """
        if reference_time is None:
            reference_time = datetime.utcnow()

        event_time = self._normalize_datetime(event_time)
        ref_time = self._normalize_datetime(reference_time)

        time_delta = ref_time - event_time
        days_since = time_delta.total_seconds() / 86400.0

        if days_since < 0:
            return 1.0

        return math.exp(-self._lambda * days_since)

    def apply_decay(
        self,
        events: List[EventNode],
        reference_time: Optional[datetime] = None,
        base_scores: Optional[List[float]] = None,
    ) -> List[ScoredEvent]:
        """Apply decay scoring to a list of events.

        Returns events sorted by decay-weighted score (descending).

        Args:
            events: List of events to score
            reference_time: Reference time for decay. Defaults to now.
            base_scores: Optional base scores for each event.
                        If None, all events start with score 1.0.

        Returns:
            List of ScoredEvent sorted by final_score descending

        Raises:
            DecayScoringError: If events and base_scores lengths don't match
        """
        if not events:
            return []

        if base_scores is not None and len(base_scores) != len(events):
            raise DecayScoringError(
                f"base_scores length ({len(base_scores)}) must match "
                f"events length ({len(events)})"
            )

        if reference_time is None:
            reference_time = datetime.utcnow()

        scored: List[ScoredEvent] = []
        for i, event in enumerate(events):
            base = base_scores[i] if base_scores else 1.0

            try:
                decay_factor = self.compute_decay_factor(
                    event.timestamp, reference_time
                )
            except Exception as e:
                logger.warning(
                    f"Failed to compute decay for event {event.id}: {e}"
                )
                decay_factor = 1.0  # Default to no decay on error

            scored.append(ScoredEvent(
                event=event,
                base_score=base,
                decay_score=decay_factor,
            ))

        # Sort by final score descending
        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored

    def apply_decay_tuples(
        self,
        events: List[EventNode],
        reference_time: Optional[datetime] = None,
    ) -> List[Tuple[EventNode, float]]:
        """Apply decay scoring and return (event, score) tuples.

        Convenience method returning simple tuples instead of ScoredEvent.

        Args:
            events: List of events to score
            reference_time: Reference time for decay. Defaults to now.

        Returns:
            List of (event, score) tuples sorted by score descending
        """
        scored = self.apply_decay(events, reference_time)
        return [(s.event, s.final_score) for s in scored]

    def days_for_score(self, target_score: float, base_score: float = 1.0) -> float:
        """Calculate days needed for score to decay to target.

        Useful for understanding decay behavior.

        Args:
            target_score: Target decay-weighted score
            base_score: Starting base score

        Returns:
            Days for score to decay from base_score to target_score

        Raises:
            ValueError: If target_score > base_score or target_score <= 0
        """
        if target_score > base_score:
            raise ValueError("target_score cannot exceed base_score")
        if target_score <= 0:
            raise ValueError("target_score must be positive")

        # score = base * exp(-lambda * days)
        # target/base = exp(-lambda * days)
        # ln(target/base) = -lambda * days
        # days = -ln(target/base) / lambda
        return -math.log(target_score / base_score) / self._lambda

    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Normalize datetime for comparison (strip timezone).

        Args:
            dt: Datetime to normalize

        Returns:
            Datetime with timezone removed
        """
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TemporalDecayScorer(half_life_days={self._half_life}, "
            f"decay_constant={self._lambda:.6f})"
        )


def create_decay_scorer(
    half_life_days: Optional[float] = None,
    config: Optional[TemporalSearchConfig] = None,
) -> TemporalDecayScorer:
    """Factory function to create a decay scorer.

    Args:
        half_life_days: Explicit half-life (takes precedence)
        config: Config to read half-life from

    Returns:
        Configured TemporalDecayScorer instance
    """
    if half_life_days is not None:
        return TemporalDecayScorer(half_life_days=half_life_days)
    return TemporalDecayScorer(config=config)
