"""Temporal Correlation Detection Module.

Detects temporal correlations between event types:
- Statistical analysis of event type co-occurrences
- Gap analysis to identify temporal patterns
- Foundation for Phase 2 correlation detection and Phase 3 causal inference

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md

Option B Compliance:
- Critical for Phase 2 correlation detection
- Results flagged as causal candidates for Phase 3
- Timestamp-based detection (works without explicit BEFORE relationships)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from futurnal.pkg.schema.models import EventNode
from futurnal.search.config import TemporalSearchConfig
from futurnal.search.temporal.exceptions import (
    CorrelationAnalysisError,
    InsufficientDataError,
)
from futurnal.search.temporal.results import TemporalCorrelationResult

if TYPE_CHECKING:
    from futurnal.pkg.queries.temporal import TemporalGraphQueries

logger = logging.getLogger(__name__)


class TemporalCorrelationDetector:
    """Detect temporal correlations between event types.

    Analyzes event data to find statistical correlations where
    one event type typically precedes another. Forms the foundation
    for Phase 2 correlation detection and Phase 3 causal inference.

    Uses timestamp-based detection (not explicit BEFORE relationships)
    to work with any event data that has temporal metadata.

    Example:
        >>> detector = TemporalCorrelationDetector(pkg_queries)

        >>> # Check if Meeting typically precedes Decision
        >>> result = detector.detect_correlation(
        ...     event_type_a="Meeting",
        ...     event_type_b="Decision",
        ...     max_gap=timedelta(days=7),
        ... )
        >>> if result.correlation_found:
        ...     print(result.temporal_pattern)
        ...     # "Meeting typically precedes Decision by 3.5 days"

        >>> # Scan for all correlations
        >>> all_correlations = detector.scan_all_correlations(
        ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
        ... )
    """

    def __init__(
        self,
        pkg_queries: "TemporalGraphQueries",
        config: Optional[TemporalSearchConfig] = None,
    ):
        """Initialize the correlation detector.

        Args:
            pkg_queries: PKG temporal queries service
            config: Optional configuration
        """
        self._pkg = pkg_queries
        self._config = config or TemporalSearchConfig()

    def detect_correlation(
        self,
        event_type_a: str,
        event_type_b: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        max_gap: Optional[timedelta] = None,
        min_occurrences: Optional[int] = None,
    ) -> TemporalCorrelationResult:
        """Detect if event_type_a typically precedes event_type_b.

        Analyzes co-occurrences where A happens before B within max_gap
        and calculates statistical properties of the correlation.

        Args:
            event_type_a: First event type (potential precedent)
            event_type_b: Second event type (potential consequent)
            time_range: Optional (start, end) to limit analysis.
                       Defaults to all available data.
            max_gap: Maximum gap for A to precede B. Default: 30 days.
            min_occurrences: Minimum co-occurrences for significance.
                           Default: from config.

        Returns:
            TemporalCorrelationResult with correlation statistics

        Example:
            >>> result = detector.detect_correlation(
            ...     event_type_a="Meeting",
            ...     event_type_b="Decision",
            ... )
            >>> if result.correlation_found:
            ...     print(f"Found {result.co_occurrences} co-occurrences")
            ...     print(f"Average gap: {result.avg_gap_days:.1f} days")
        """
        if max_gap is None:
            max_gap = timedelta(days=self._config.default_max_gap_days)
        if min_occurrences is None:
            min_occurrences = self._config.correlation_min_occurrences

        logger.debug(
            f"Detecting correlation: {event_type_a} -> {event_type_b}, "
            f"max_gap={max_gap.days} days, min_occurrences={min_occurrences}"
        )

        try:
            # Get events of both types
            if time_range:
                start_time, end_time = time_range
            else:
                # Use wide range as default
                start_time = datetime(1970, 1, 1)
                end_time = datetime(2100, 1, 1)

            events_a = self._pkg.query_events_in_timerange(
                start=start_time,
                end=end_time,
                event_type=event_type_a,
            )
            events_b = self._pkg.query_events_in_timerange(
                start=start_time,
                end=end_time,
                event_type=event_type_b,
            )

            # Sort by timestamp
            events_a = sorted(events_a, key=lambda e: e.timestamp)
            events_b = sorted(events_b, key=lambda e: e.timestamp)

            logger.debug(
                f"Found {len(events_a)} events of type {event_type_a}, "
                f"{len(events_b)} events of type {event_type_b}"
            )

            # Find co-occurrences where A precedes B within max_gap
            co_occurrences = self._find_co_occurrences(
                events_a=events_a,
                events_b=events_b,
                max_gap=max_gap,
            )

            # Check if we have enough co-occurrences
            if len(co_occurrences) < min_occurrences:
                return TemporalCorrelationResult(
                    correlation_found=False,
                    event_type_a=event_type_a,
                    event_type_b=event_type_b,
                    co_occurrences=len(co_occurrences),
                )

            # Calculate statistics
            stats = self._calculate_correlation_stats(co_occurrences)

            # Determine if this is a causal candidate
            is_causal_candidate = self._is_causal_candidate(
                co_occurrences=len(co_occurrences),
                gap_consistency=stats["consistency"],
            )

            return TemporalCorrelationResult(
                correlation_found=True,
                event_type_a=event_type_a,
                event_type_b=event_type_b,
                co_occurrences=len(co_occurrences),
                avg_gap_days=stats["avg_gap_days"],
                min_gap_days=stats["min_gap_days"],
                max_gap_days=stats["max_gap_days"],
                std_gap_days=stats["std_gap_days"],
                correlation_strength=stats["strength"],
                is_causal_candidate=is_causal_candidate,
                examples=co_occurrences[:5],  # First 5 examples
            )

        except Exception as e:
            logger.error(f"Correlation detection failed: {e}")
            raise CorrelationAnalysisError(
                f"Failed to detect correlation: {e}",
                event_type_a=event_type_a,
                event_type_b=event_type_b,
                cause=e,
            )

    def _find_co_occurrences(
        self,
        events_a: List[EventNode],
        events_b: List[EventNode],
        max_gap: timedelta,
    ) -> List[Tuple[str, str, float]]:
        """Find co-occurrences where A precedes B within max_gap.

        Returns list of (event_a_id, event_b_id, gap_days) tuples.
        Uses efficient algorithm to avoid O(n*m) complexity.
        """
        co_occurrences: List[Tuple[str, str, float]] = []
        used_b: Set[str] = set()  # Track which B events are already matched

        b_idx = 0
        for event_a in events_a:
            a_time = self._normalize_datetime(event_a.timestamp)

            # Move b_idx forward to events after a_time
            while b_idx < len(events_b):
                b_time = self._normalize_datetime(events_b[b_idx].timestamp)
                if b_time > a_time:
                    break
                b_idx += 1

            # Check B events starting from b_idx
            for i in range(b_idx, len(events_b)):
                event_b = events_b[i]

                # Skip if already matched
                if event_b.id in used_b:
                    continue

                b_time = self._normalize_datetime(event_b.timestamp)
                gap = b_time - a_time

                # Check if within max_gap
                if gap > max_gap:
                    break  # No more valid B events for this A

                if gap.total_seconds() > 0:  # Must be strictly after
                    gap_days = gap.total_seconds() / 86400.0
                    co_occurrences.append((event_a.id, event_b.id, gap_days))
                    used_b.add(event_b.id)  # Each B can only be matched once
                    break  # Found match for this A, move to next A

        return co_occurrences

    def _calculate_correlation_stats(
        self,
        co_occurrences: List[Tuple[str, str, float]],
    ) -> Dict[str, float]:
        """Calculate statistical properties of co-occurrences."""
        gaps = [gap for _, _, gap in co_occurrences]

        if not gaps:
            return {
                "avg_gap_days": 0.0,
                "min_gap_days": 0.0,
                "max_gap_days": 0.0,
                "std_gap_days": 0.0,
                "strength": 0.0,
                "consistency": 0.0,
            }

        avg_gap = sum(gaps) / len(gaps)
        min_gap = min(gaps)
        max_gap = max(gaps)

        # Standard deviation
        if len(gaps) > 1:
            variance = sum((g - avg_gap) ** 2 for g in gaps) / (len(gaps) - 1)
            std_gap = math.sqrt(variance)
        else:
            std_gap = 0.0

        # Consistency measure (1 - normalized coefficient of variation)
        if avg_gap > 0:
            cv = std_gap / avg_gap
            consistency = max(0.0, 1.0 - min(cv, 1.0))
        else:
            consistency = 1.0

        # Correlation strength based on consistency and sample size
        # More co-occurrences and lower variance = stronger correlation
        sample_factor = min(1.0, len(gaps) / 10.0)  # Caps at 10
        strength = consistency * sample_factor

        return {
            "avg_gap_days": avg_gap,
            "min_gap_days": min_gap,
            "max_gap_days": max_gap,
            "std_gap_days": std_gap,
            "strength": strength,
            "consistency": consistency,
        }

    def _is_causal_candidate(
        self,
        co_occurrences: int,
        gap_consistency: float,
    ) -> bool:
        """Determine if correlation is a causal candidate.

        A correlation is flagged as a causal candidate if:
        - Has sufficient co-occurrences (>= 2x minimum)
        - Has consistent timing (consistency >= threshold)
        """
        min_for_causal = self._config.correlation_min_occurrences * 2
        return (
            co_occurrences >= min_for_causal
            and gap_consistency >= self._config.correlation_significance_threshold
        )

    def scan_all_correlations(
        self,
        time_range: Tuple[datetime, datetime],
        min_occurrences: Optional[int] = None,
        max_gap: Optional[timedelta] = None,
    ) -> List[TemporalCorrelationResult]:
        """Scan for all significant correlations in time range.

        Analyzes all pairs of event types to find temporal correlations.
        Useful for Phase 2 automated correlation discovery.

        Args:
            time_range: (start, end) time range to analyze
            min_occurrences: Minimum co-occurrences for significance
            max_gap: Maximum gap for correlation

        Returns:
            List of TemporalCorrelationResult for all found correlations,
            sorted by correlation_strength descending

        Example:
            >>> correlations = detector.scan_all_correlations(
            ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            ... )
            >>> for c in correlations[:10]:
            ...     print(f"{c.temporal_pattern}: strength={c.correlation_strength:.2f}")
        """
        if min_occurrences is None:
            min_occurrences = self._config.correlation_min_occurrences
        if max_gap is None:
            max_gap = timedelta(days=self._config.default_max_gap_days)

        start_time, end_time = time_range
        logger.info(
            f"Scanning for correlations in {start_time} to {end_time}"
        )

        # Get all events in range
        all_events = self._pkg.query_events_in_timerange(
            start=start_time,
            end=end_time,
        )

        # Group by event type
        events_by_type: Dict[str, List[EventNode]] = defaultdict(list)
        for event in all_events:
            if event.event_type:
                events_by_type[event.event_type].append(event)

        # Sort each group by timestamp
        for event_type in events_by_type:
            events_by_type[event_type].sort(key=lambda e: e.timestamp)

        event_types = list(events_by_type.keys())
        logger.debug(f"Found {len(event_types)} distinct event types")

        # Check all pairs
        correlations: List[TemporalCorrelationResult] = []

        for i, type_a in enumerate(event_types):
            for type_b in event_types[i + 1:]:
                # Check A -> B
                result_ab = self._check_pair_correlation(
                    events_a=events_by_type[type_a],
                    events_b=events_by_type[type_b],
                    type_a=type_a,
                    type_b=type_b,
                    max_gap=max_gap,
                    min_occurrences=min_occurrences,
                )
                if result_ab.correlation_found:
                    correlations.append(result_ab)

                # Check B -> A
                result_ba = self._check_pair_correlation(
                    events_a=events_by_type[type_b],
                    events_b=events_by_type[type_a],
                    type_a=type_b,
                    type_b=type_a,
                    max_gap=max_gap,
                    min_occurrences=min_occurrences,
                )
                if result_ba.correlation_found:
                    correlations.append(result_ba)

        # Sort by correlation strength descending
        correlations.sort(
            key=lambda c: c.correlation_strength or 0.0,
            reverse=True,
        )

        logger.info(f"Found {len(correlations)} significant correlations")
        return correlations

    def _check_pair_correlation(
        self,
        events_a: List[EventNode],
        events_b: List[EventNode],
        type_a: str,
        type_b: str,
        max_gap: timedelta,
        min_occurrences: int,
    ) -> TemporalCorrelationResult:
        """Check correlation between a specific pair of event types."""
        co_occurrences = self._find_co_occurrences(
            events_a=events_a,
            events_b=events_b,
            max_gap=max_gap,
        )

        if len(co_occurrences) < min_occurrences:
            return TemporalCorrelationResult(
                correlation_found=False,
                event_type_a=type_a,
                event_type_b=type_b,
                co_occurrences=len(co_occurrences),
            )

        stats = self._calculate_correlation_stats(co_occurrences)

        return TemporalCorrelationResult(
            correlation_found=True,
            event_type_a=type_a,
            event_type_b=type_b,
            co_occurrences=len(co_occurrences),
            avg_gap_days=stats["avg_gap_days"],
            min_gap_days=stats["min_gap_days"],
            max_gap_days=stats["max_gap_days"],
            std_gap_days=stats["std_gap_days"],
            correlation_strength=stats["strength"],
            is_causal_candidate=self._is_causal_candidate(
                len(co_occurrences), stats["consistency"]
            ),
            examples=co_occurrences[:5],
        )

    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Normalize datetime for comparison (strip timezone)."""
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    def get_correlation_summary(
        self,
        correlations: List[TemporalCorrelationResult],
    ) -> Dict[str, any]:
        """Generate summary statistics for a list of correlations.

        Args:
            correlations: List of correlation results

        Returns:
            Dictionary with summary statistics
        """
        if not correlations:
            return {
                "total_correlations": 0,
                "causal_candidates": 0,
                "avg_strength": 0.0,
                "strongest": None,
            }

        found = [c for c in correlations if c.correlation_found]
        causal = [c for c in found if c.is_causal_candidate]

        strengths = [c.correlation_strength or 0.0 for c in found]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0

        strongest = max(found, key=lambda c: c.correlation_strength or 0.0) if found else None

        return {
            "total_correlations": len(found),
            "causal_candidates": len(causal),
            "avg_strength": avg_strength,
            "strongest": strongest.temporal_pattern if strongest else None,
        }
