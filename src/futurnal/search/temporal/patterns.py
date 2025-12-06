"""Temporal Pattern Matching Module.

Implements pattern matching for temporal event sequences:
- Find sequences matching specified patterns (e.g., Meeting -> Decision)
- Discover recurring patterns automatically
- Support for Phase 2 correlation detection

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md

Option B Compliance:
- Critical for Phase 2 correlation detection
- Patterns flagged as causal candidates for Phase 3
- Temporal-first: all patterns based on event timestamps
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from futurnal.pkg.schema.models import EventNode
from futurnal.search.config import TemporalSearchConfig
from futurnal.search.temporal.exceptions import PatternNotFoundError
from futurnal.search.temporal.results import RecurringPattern, SequenceMatch

if TYPE_CHECKING:
    from futurnal.pkg.queries.temporal import TemporalGraphQueries

logger = logging.getLogger(__name__)


class TemporalPatternMatcher:
    """Match temporal patterns in event sequences.

    Finds event sequences matching specified patterns and discovers
    recurring patterns automatically. Critical for Phase 2 correlation
    detection.

    Example:
        >>> matcher = TemporalPatternMatcher(pkg_queries)
        >>> # Find Meeting -> Decision sequences
        >>> matches = matcher.find_sequences(
        ...     pattern=["Meeting", "Decision"],
        ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
        ...     max_gap=timedelta(days=7),
        ... )
        >>> for match in matches:
        ...     print(f"{match.pattern_description}: {match.total_span.days} days")

        >>> # Discover recurring patterns
        >>> patterns = matcher.find_recurring_patterns(
        ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
        ...     min_occurrences=3,
        ... )
        >>> for p in patterns:
        ...     print(f"{p.pattern_description}: {p.occurrences} occurrences")
    """

    def __init__(
        self,
        pkg_queries: "TemporalGraphQueries",
        config: Optional[TemporalSearchConfig] = None,
    ):
        """Initialize the pattern matcher.

        Args:
            pkg_queries: PKG temporal queries service
            config: Optional configuration
        """
        self._pkg = pkg_queries
        self._config = config or TemporalSearchConfig()

    def find_sequences(
        self,
        pattern: List[str],
        time_range: Tuple[datetime, datetime],
        max_gap: Optional[timedelta] = None,
        min_confidence: float = 0.7,
    ) -> List[SequenceMatch]:
        """Find all event sequences matching the pattern.

        Searches for sequences where events of the specified types
        occur in the given order within the max_gap constraint.

        Args:
            pattern: Event types in order (e.g., ["Meeting", "Decision"])
            time_range: (start, end) time range to search
            max_gap: Maximum gap between consecutive events.
                    Defaults to config.default_max_gap_days.
            min_confidence: Minimum confidence threshold

        Returns:
            List of SequenceMatch objects

        Raises:
            ValueError: If pattern has fewer than 2 event types
            PatternNotFoundError: If no matching sequences found

        Example:
            >>> matches = matcher.find_sequences(
            ...     pattern=["Meeting", "Decision", "Publication"],
            ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            ... )
        """
        if len(pattern) < 2:
            raise ValueError("Pattern must have at least 2 event types")

        if max_gap is None:
            max_gap = timedelta(days=self._config.default_max_gap_days)

        start_time, end_time = time_range
        logger.debug(
            f"Finding sequences for pattern {pattern} in "
            f"{start_time} to {end_time}, max_gap={max_gap.days} days"
        )

        # Get all events of required types in time range
        events_by_type: Dict[str, List[EventNode]] = {}
        for event_type in set(pattern):
            events = self._pkg.query_events_in_timerange(
                start=start_time,
                end=end_time,
                event_type=event_type,
            )
            # Filter by confidence
            events = [e for e in events if (e.confidence or 1.0) >= min_confidence]
            events_by_type[event_type] = sorted(events, key=lambda e: e.timestamp)

        # Check if we have events for all pattern types
        for event_type in pattern:
            if not events_by_type.get(event_type):
                logger.debug(f"No events of type '{event_type}' found")
                return []

        # Find matching sequences using sliding window approach
        matches = self._find_pattern_matches(
            pattern=pattern,
            events_by_type=events_by_type,
            max_gap=max_gap,
        )

        logger.info(f"Found {len(matches)} sequences matching pattern {pattern}")
        return matches

    def _find_pattern_matches(
        self,
        pattern: List[str],
        events_by_type: Dict[str, List[EventNode]],
        max_gap: timedelta,
    ) -> List[SequenceMatch]:
        """Find all valid pattern matches using dynamic programming.

        Uses a sliding window approach to efficiently find all
        sequences matching the pattern within gap constraints.
        """
        matches: List[SequenceMatch] = []

        # Start with first event type
        first_type = pattern[0]
        first_events = events_by_type.get(first_type, [])

        for start_event in first_events:
            # Try to build a complete sequence starting from this event
            sequence = self._build_sequence(
                pattern=pattern,
                events_by_type=events_by_type,
                start_event=start_event,
                max_gap=max_gap,
            )
            if sequence:
                matches.append(sequence)

        return matches

    def _build_sequence(
        self,
        pattern: List[str],
        events_by_type: Dict[str, List[EventNode]],
        start_event: EventNode,
        max_gap: timedelta,
    ) -> Optional[SequenceMatch]:
        """Build a complete sequence starting from start_event.

        Uses greedy matching: finds the first valid event of each
        subsequent type that satisfies the gap constraint.
        """
        events: List[EventNode] = [start_event]
        gaps: List[timedelta] = []
        current_time = start_event.timestamp

        for i in range(1, len(pattern)):
            event_type = pattern[i]
            candidates = events_by_type.get(event_type, [])

            # Find first event after current_time within max_gap
            next_event = None
            for candidate in candidates:
                gap = candidate.timestamp - current_time
                # Must be after current event and within max_gap
                if timedelta(0) < gap <= max_gap:
                    next_event = candidate
                    gaps.append(gap)
                    break

            if next_event is None:
                # Could not find valid next event
                return None

            events.append(next_event)
            current_time = next_event.timestamp

        # Build the match
        return SequenceMatch(
            events=events,
            pattern=pattern,
            gaps=gaps,
            confidence=self._calculate_sequence_confidence(events),
            match_quality=1.0,  # Exact match
        )

    def _calculate_sequence_confidence(self, events: List[EventNode]) -> float:
        """Calculate confidence score for a sequence.

        Based on the average confidence of events and temporal consistency.
        """
        if not events:
            return 0.0

        # Average event confidence
        confidences = [e.confidence or 1.0 for e in events]
        return sum(confidences) / len(confidences)

    def find_recurring_patterns(
        self,
        time_range: Tuple[datetime, datetime],
        min_pattern_length: int = 2,
        max_pattern_length: Optional[int] = None,
        min_occurrences: int = 3,
    ) -> List[RecurringPattern]:
        """Discover recurring temporal patterns automatically.

        Analyzes events in the time range to find patterns that
        occur multiple times. Critical for Phase 2 correlation detection.

        Args:
            time_range: (start, end) time range to analyze
            min_pattern_length: Minimum pattern length. Default: 2
            max_pattern_length: Maximum pattern length. Default: from config
            min_occurrences: Minimum occurrences for significance. Default: 3

        Returns:
            List of RecurringPattern objects, sorted by occurrences descending

        Example:
            >>> patterns = matcher.find_recurring_patterns(
            ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            ...     min_occurrences=5,
            ... )
            >>> for p in patterns[:10]:
            ...     print(f"{p.pattern_description}: {p.occurrences}x")
        """
        if max_pattern_length is None:
            max_pattern_length = self._config.max_pattern_length

        start_time, end_time = time_range
        logger.debug(
            f"Finding recurring patterns in {start_time} to {end_time}, "
            f"min_length={min_pattern_length}, min_occurrences={min_occurrences}"
        )

        # Get all events in time range
        all_events = self._pkg.query_events_in_timerange(
            start=start_time,
            end=end_time,
        )

        if not all_events:
            return []

        # Sort by timestamp
        all_events = sorted(all_events, key=lambda e: e.timestamp)

        # Extract event type sequences
        event_types = [e.event_type for e in all_events if e.event_type]

        # Count pattern occurrences for each length
        pattern_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        pattern_examples: Dict[Tuple[str, ...], List[List[EventNode]]] = defaultdict(list)

        for length in range(min_pattern_length, max_pattern_length + 1):
            for i in range(len(all_events) - length + 1):
                window = all_events[i:i + length]
                pattern = tuple(e.event_type for e in window if e.event_type)

                if len(pattern) == length:
                    pattern_counts[pattern] += 1
                    if len(pattern_examples[pattern]) < 5:  # Keep up to 5 examples
                        pattern_examples[pattern].append(window)

        # Filter by min_occurrences and build results
        recurring: List[RecurringPattern] = []

        for pattern, count in pattern_counts.items():
            if count >= min_occurrences:
                examples = pattern_examples[pattern]
                gaps = self._calculate_pattern_gaps(examples)

                recurring.append(RecurringPattern(
                    pattern=list(pattern),
                    occurrences=count,
                    average_gap=gaps["average"],
                    min_gap=gaps["min"],
                    max_gap=gaps["max"],
                    examples=[
                        SequenceMatch(
                            events=ex,
                            pattern=list(pattern),
                            gaps=self._calculate_event_gaps(ex),
                        )
                        for ex in examples[:5]
                    ],
                    statistical_significance=self._calculate_significance(
                        count, len(all_events), len(pattern)
                    ),
                    is_causal_candidate=count >= min_occurrences * 2,  # Strong patterns
                ))

        # Sort by occurrences descending
        recurring.sort(key=lambda p: p.occurrences, reverse=True)

        logger.info(f"Found {len(recurring)} recurring patterns")
        return recurring

    def _calculate_pattern_gaps(
        self,
        examples: List[List[EventNode]],
    ) -> Dict[str, timedelta]:
        """Calculate gap statistics across pattern examples."""
        all_gaps: List[timedelta] = []

        for example in examples:
            gaps = self._calculate_event_gaps(example)
            all_gaps.extend(gaps)

        if not all_gaps:
            return {
                "average": timedelta(0),
                "min": timedelta(0),
                "max": timedelta(0),
            }

        avg_seconds = sum(g.total_seconds() for g in all_gaps) / len(all_gaps)
        return {
            "average": timedelta(seconds=avg_seconds),
            "min": min(all_gaps),
            "max": max(all_gaps),
        }

    def _calculate_event_gaps(self, events: List[EventNode]) -> List[timedelta]:
        """Calculate gaps between consecutive events."""
        gaps: List[timedelta] = []
        for i in range(len(events) - 1):
            gap = events[i + 1].timestamp - events[i].timestamp
            gaps.append(gap)
        return gaps

    def _calculate_significance(
        self,
        occurrences: int,
        total_events: int,
        pattern_length: int,
    ) -> float:
        """Calculate statistical significance of a pattern.

        Simple heuristic based on occurrence frequency relative
        to expected random occurrence.
        """
        if total_events <= pattern_length:
            return 0.0

        # Expected occurrences if random (very rough approximation)
        max_possible = total_events - pattern_length + 1
        frequency = occurrences / max_possible

        # Sigmoid-like scaling
        return min(1.0, frequency * 10)

    def find_pattern_variants(
        self,
        base_pattern: List[str],
        time_range: Tuple[datetime, datetime],
        min_occurrences: int = 2,
    ) -> List[RecurringPattern]:
        """Find variants of a base pattern.

        Searches for patterns that contain the base pattern elements
        with possible additional events in between.

        Args:
            base_pattern: Core event types to look for
            time_range: Time range to search
            min_occurrences: Minimum occurrences for each variant

        Returns:
            List of pattern variants found
        """
        # Find all recurring patterns
        all_patterns = self.find_recurring_patterns(
            time_range=time_range,
            min_pattern_length=len(base_pattern),
            max_pattern_length=len(base_pattern) + 2,
            min_occurrences=min_occurrences,
        )

        # Filter to those containing base pattern elements
        variants: List[RecurringPattern] = []
        base_set = set(base_pattern)

        for pattern in all_patterns:
            pattern_set = set(pattern.pattern)
            if base_set.issubset(pattern_set):
                # Check ordering is preserved
                if self._preserves_order(base_pattern, pattern.pattern):
                    variants.append(pattern)

        return variants

    def _preserves_order(self, base: List[str], full: List[str]) -> bool:
        """Check if base pattern order is preserved in full pattern."""
        base_idx = 0
        for event_type in full:
            if base_idx < len(base) and event_type == base[base_idx]:
                base_idx += 1
        return base_idx == len(base)
