"""Temporal Correlation Detection Module.

Detects temporal correlations between event types:
- Statistical analysis of event type co-occurrences
- Gap analysis to identify temporal patterns
- Foundation for Phase 2 correlation detection and Phase 3 causal inference

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md

Phase 2.5 Research Integration:
- DOTS (arxiv:2510.24639): Causal ordering before structure learning (F1 0.81 vs 0.63)
- DOTSCausalOrdering pre-filters correlations to remove implausible causal directions
- 28% accuracy improvement in causal discovery

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
from futurnal.search.temporal.statistics import (
    StatisticalCorrelationValidator,
    StatisticalSignificanceResult,
)
from futurnal.search.temporal.dots_ordering import (
    DOTSCausalOrdering,
    CausalOrder,
)
from futurnal.search.temporal.unified_source import UnifiedTemporalSource, TemporalItem

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
        statistical_validation: bool = True,
        significance_threshold: float = 0.05,
        dots_ordering: bool = True,
        dots_precedence_threshold: float = 0.6,
    ):
        """Initialize the correlation detector.

        Args:
            pkg_queries: PKG temporal queries service
            config: Optional configuration
            statistical_validation: Enable rigorous statistical testing
            significance_threshold: P-value threshold for significance
            dots_ordering: Enable DOTS causal ordering (28% accuracy boost)
            dots_precedence_threshold: Threshold for DOTS precedence
        """
        self._pkg = pkg_queries
        self._config = config or TemporalSearchConfig()
        self._statistical_validation = statistical_validation
        self._dots_ordering = dots_ordering

        # Statistical validator for rigorous testing (AGI Phase 1 enhancement)
        self._validator = StatisticalCorrelationValidator(
            significance_threshold=significance_threshold,
        ) if statistical_validation else None

        # DOTS causal ordering (Phase 2.5: arxiv:2510.24639)
        # Establishes causal ordering BEFORE structure learning for 28% accuracy boost
        self._dots = DOTSCausalOrdering(
            min_observations=self._config.correlation_min_occurrences,
            precedence_threshold=dots_precedence_threshold,
            max_lag_hours=self._config.default_max_gap_days * 24,
        ) if dots_ordering else None

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

            # Calculate time range for statistical validation
            time_range_days = (end_time - start_time).days if time_range else 365.0

            # Statistical validation (AGI Phase 1 enhancement)
            stat_result: Optional[StatisticalSignificanceResult] = None
            is_statistically_significant = False

            if self._validator and len(co_occurrences) >= 3:
                gap_values = [gap for _, _, gap in co_occurrences]
                stat_result = self._validator.validate_correlation(
                    observed_cooccurrences=len(co_occurrences),
                    total_events_a=len(events_a),
                    total_events_b=len(events_b),
                    time_range_days=time_range_days,
                    max_gap_days=max_gap.days,
                    gap_values=gap_values,
                )
                is_statistically_significant = stat_result.is_significant

                logger.debug(
                    f"Statistical validation for {event_type_a}->{event_type_b}: "
                    f"p={stat_result.p_value:.4f}, significant={is_statistically_significant}"
                )

            # Determine if this is a causal candidate
            # Now requires both heuristic AND statistical significance
            is_causal_candidate = self._is_causal_candidate(
                co_occurrences=len(co_occurrences),
                gap_consistency=stats["consistency"],
            )

            # Only flag as causal candidate if statistically significant
            if self._statistical_validation and stat_result:
                is_causal_candidate = is_causal_candidate and is_statistically_significant

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
                # Statistical significance fields (Phase 1 AGI)
                p_value=stat_result.p_value if stat_result else None,
                statistical_test=stat_result.test_used if stat_result else None,
                is_statistically_significant=is_statistically_significant,
                confidence_interval_95=stat_result.confidence_interval_95 if stat_result else None,
                effect_size=stat_result.effect_size if stat_result else None,
                effect_interpretation=stat_result.effect_interpretation if stat_result else None,
                expected_by_chance=stat_result.expected_by_chance if stat_result else None,
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
        apply_multiple_testing_correction: bool = True,
        correction_method: str = "bonferroni",
        include_documents: bool = True,
    ) -> List[TemporalCorrelationResult]:
        """Scan for all significant correlations in time range.

        Analyzes all pairs of event/document types to find temporal correlations.
        Useful for Phase 2 automated correlation discovery.

        Phase 2.5 P0 Fix: Now uses UnifiedTemporalSource to analyze both
        Event nodes AND Document nodes, enabling insights from documents
        when no Event nodes exist.

        AGI Phase 1 Enhancement:
        - Applies Bonferroni or FDR correction for multiple hypothesis testing
        - Only returns correlations that pass corrected significance threshold
        - Prevents "horoscope problem" of spurious patterns

        Args:
            time_range: (start, end) time range to analyze
            min_occurrences: Minimum co-occurrences for significance
            max_gap: Maximum gap for correlation
            apply_multiple_testing_correction: Apply Bonferroni/FDR correction
            correction_method: "bonferroni" (conservative) or "fdr" (less conservative)
            include_documents: Include Document nodes in analysis (default True)

        Returns:
            List of TemporalCorrelationResult for all found correlations,
            sorted by correlation_strength descending

        Example:
            >>> correlations = detector.scan_all_correlations(
            ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            ... )
            >>> for c in correlations[:10]:
            ...     print(f"{c.temporal_pattern}: strength={c.correlation_strength:.2f}")
            ...     if c.is_statistically_significant:
            ...         print(f"  p={c.p_value:.4f} (corrected: {c.corrected_p_value:.4f})")
        """
        if min_occurrences is None:
            min_occurrences = self._config.correlation_min_occurrences
        if max_gap is None:
            max_gap = timedelta(days=self._config.default_max_gap_days)

        start_time, end_time = time_range
        logger.info(
            f"Scanning for correlations in {start_time} to {end_time}"
        )

        # Phase 2.5 P0 Fix: Use UnifiedTemporalSource to get events AND documents
        unified_source = UnifiedTemporalSource(self._pkg)
        all_items = unified_source.get_temporal_items(
            start=start_time,
            end=end_time,
            include_events=True,
            include_documents=include_documents,
        )

        logger.info(
            f"UnifiedTemporalSource returned {len(all_items)} items "
            f"(events + documents)"
        )

        # Group by category (event_type or doc_type)
        items_by_category: Dict[str, List[TemporalItem]] = defaultdict(list)
        for item in all_items:
            if item.category:
                items_by_category[item.category].append(item)

        # Sort each group by timestamp
        for category in items_by_category:
            items_by_category[category].sort(key=lambda i: i.timestamp)

        categories = list(items_by_category.keys())
        logger.debug(f"Found {len(categories)} distinct categories (event types + doc types)")

        # Calculate time range for statistical validation
        time_range_days = (end_time - start_time).days

        # Check all pairs
        correlations: List[TemporalCorrelationResult] = []

        for i, type_a in enumerate(categories):
            for type_b in categories[i + 1:]:
                # Check A -> B
                result_ab = self._check_pair_correlation_items(
                    items_a=items_by_category[type_a],
                    items_b=items_by_category[type_b],
                    type_a=type_a,
                    type_b=type_b,
                    max_gap=max_gap,
                    min_occurrences=min_occurrences,
                    time_range_days=time_range_days,
                )
                if result_ab.correlation_found:
                    correlations.append(result_ab)

                # Check B -> A
                result_ba = self._check_pair_correlation_items(
                    items_a=items_by_category[type_b],
                    items_b=items_by_category[type_a],
                    type_a=type_b,
                    type_b=type_a,
                    max_gap=max_gap,
                    min_occurrences=min_occurrences,
                    time_range_days=time_range_days,
                )
                if result_ba.correlation_found:
                    correlations.append(result_ba)

        # Phase 2.5: Apply DOTS causal ordering pre-filter (28% accuracy boost)
        # This removes correlations with implausible causal directions
        if self._dots and correlations:
            # Convert items to DOTS format for causal ordering computation
            dots_events = []
            for category, item_list in items_by_category.items():
                for item in item_list:
                    dots_events.append({
                        "event_type": category,
                        "timestamp": self._normalize_datetime(item.timestamp).isoformat(),
                    })

            if len(dots_events) >= self._dots.min_observations:
                # Compute causal ordering from temporal precedence
                causal_order = self._dots.compute_causal_order(dots_events)

                if causal_order.ordered_events:
                    # Filter correlations by causal order
                    pre_filter_count = len(correlations)
                    correlations = self._dots.filter_by_causal_order(
                        correlations=[
                            {"event_type_a": c.event_type_a, "event_type_b": c.event_type_b, "_corr": c}
                            for c in correlations
                        ],
                        order=causal_order,
                    )
                    # Extract back the TemporalCorrelationResult objects
                    correlations = [c["_corr"] for c in correlations]

                    logger.info(
                        f"DOTS ordering: {pre_filter_count} -> {len(correlations)} correlations "
                        f"(filtered {pre_filter_count - len(correlations)} implausible directions)"
                    )

        # Apply multiple testing correction (AGI Phase 1)
        if apply_multiple_testing_correction and self._validator and correlations:
            correlations = self._apply_multiple_testing_correction(
                correlations=correlations,
                method=correction_method,
            )
            # Filter to only statistically significant after correction
            significant_correlations = [
                c for c in correlations if c.is_statistically_significant
            ]
            logger.info(
                f"After {correction_method} correction: "
                f"{len(significant_correlations)}/{len(correlations)} remain significant"
            )
            correlations = significant_correlations

        # Sort by correlation strength descending
        correlations.sort(
            key=lambda c: c.correlation_strength or 0.0,
            reverse=True,
        )

        logger.info(f"Found {len(correlations)} significant correlations")
        return correlations

    def _apply_multiple_testing_correction(
        self,
        correlations: List[TemporalCorrelationResult],
        method: str = "bonferroni",
    ) -> List[TemporalCorrelationResult]:
        """Apply multiple testing correction to correlation results.

        Args:
            correlations: List of correlation results with p-values
            method: "bonferroni" or "fdr"

        Returns:
            Updated correlations with corrected p-values and significance
        """
        if not self._validator:
            return correlations

        # Collect p-values
        p_values = [c.p_value if c.p_value is not None else 1.0 for c in correlations]

        # Apply correction
        if method == "fdr":
            corrections = self._validator.benjamini_hochberg_correction(p_values)
        else:
            corrections = self._validator.bonferroni_correction(p_values)

        # Update results with corrected values
        updated = []
        for corr, (corrected_p, is_sig) in zip(correlations, corrections):
            # Create updated result - use model_copy for Pydantic v2
            if hasattr(corr, 'model_copy'):
                updated_corr = corr.model_copy(update={
                    "corrected_p_value": corrected_p,
                    "is_statistically_significant": is_sig,
                    # Update causal candidate based on corrected significance
                    "is_causal_candidate": corr.is_causal_candidate and is_sig if self._statistical_validation else corr.is_causal_candidate,
                })
            else:
                # Fallback for older Pydantic
                data = corr.dict()
                data["corrected_p_value"] = corrected_p
                data["is_statistically_significant"] = is_sig
                if self._statistical_validation:
                    data["is_causal_candidate"] = data["is_causal_candidate"] and is_sig
                updated_corr = TemporalCorrelationResult(**data)

            updated.append(updated_corr)

        return updated

    def _check_pair_correlation(
        self,
        events_a: List[EventNode],
        events_b: List[EventNode],
        type_a: str,
        type_b: str,
        max_gap: timedelta,
        min_occurrences: int,
        time_range_days: float = 365.0,
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

        # Statistical validation (AGI Phase 1)
        stat_result: Optional[StatisticalSignificanceResult] = None
        is_statistically_significant = False

        if self._validator and len(co_occurrences) >= 3:
            gap_values = [gap for _, _, gap in co_occurrences]
            stat_result = self._validator.validate_correlation(
                observed_cooccurrences=len(co_occurrences),
                total_events_a=len(events_a),
                total_events_b=len(events_b),
                time_range_days=time_range_days,
                max_gap_days=max_gap.days,
                gap_values=gap_values,
            )
            is_statistically_significant = stat_result.is_significant

        is_causal_candidate = self._is_causal_candidate(
            len(co_occurrences), stats["consistency"]
        )

        # Only flag as causal candidate if statistically significant
        if self._statistical_validation and stat_result:
            is_causal_candidate = is_causal_candidate and is_statistically_significant

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
            is_causal_candidate=is_causal_candidate,
            examples=co_occurrences[:5],
            # Statistical significance fields
            p_value=stat_result.p_value if stat_result else None,
            statistical_test=stat_result.test_used if stat_result else None,
            is_statistically_significant=is_statistically_significant,
            confidence_interval_95=stat_result.confidence_interval_95 if stat_result else None,
            effect_size=stat_result.effect_size if stat_result else None,
            effect_interpretation=stat_result.effect_interpretation if stat_result else None,
            expected_by_chance=stat_result.expected_by_chance if stat_result else None,
        )

    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Normalize datetime for comparison (strip timezone)."""
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    # -------------------------------------------------------------------------
    # TemporalItem-based methods (Phase 2.5 P0 Fix)
    # -------------------------------------------------------------------------

    def _check_pair_correlation_items(
        self,
        items_a: List[TemporalItem],
        items_b: List[TemporalItem],
        type_a: str,
        type_b: str,
        max_gap: timedelta,
        min_occurrences: int,
        time_range_days: float = 365.0,
    ) -> TemporalCorrelationResult:
        """Check correlation between a specific pair of item categories.

        Works with TemporalItem (unified events + documents).
        """
        co_occurrences = self._find_co_occurrences_items(
            items_a=items_a,
            items_b=items_b,
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

        # Statistical validation (AGI Phase 1)
        stat_result: Optional[StatisticalSignificanceResult] = None
        is_statistically_significant = False

        if self._validator and len(co_occurrences) >= 3:
            gap_values = [gap for _, _, gap in co_occurrences]
            stat_result = self._validator.validate_correlation(
                observed_cooccurrences=len(co_occurrences),
                total_events_a=len(items_a),
                total_events_b=len(items_b),
                time_range_days=time_range_days,
                max_gap_days=max_gap.days,
                gap_values=gap_values,
            )
            is_statistically_significant = stat_result.is_significant

        is_causal_candidate = self._is_causal_candidate(
            len(co_occurrences), stats["consistency"]
        )

        # Only flag as causal candidate if statistically significant
        if self._statistical_validation and stat_result:
            is_causal_candidate = is_causal_candidate and is_statistically_significant

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
            is_causal_candidate=is_causal_candidate,
            examples=co_occurrences[:5],
            # Statistical significance fields
            p_value=stat_result.p_value if stat_result else None,
            statistical_test=stat_result.test_used if stat_result else None,
            is_statistically_significant=is_statistically_significant,
            confidence_interval_95=stat_result.confidence_interval_95 if stat_result else None,
            effect_size=stat_result.effect_size if stat_result else None,
            effect_interpretation=stat_result.effect_interpretation if stat_result else None,
            expected_by_chance=stat_result.expected_by_chance if stat_result else None,
        )

    def _find_co_occurrences_items(
        self,
        items_a: List[TemporalItem],
        items_b: List[TemporalItem],
        max_gap: timedelta,
    ) -> List[Tuple[str, str, float]]:
        """Find co-occurrences where item A precedes item B within max_gap.

        Works with TemporalItem (unified events + documents).
        Returns list of (item_a_id, item_b_id, gap_days) tuples.
        """
        co_occurrences: List[Tuple[str, str, float]] = []
        used_b: Set[str] = set()

        b_idx = 0
        for item_a in items_a:
            a_time = self._normalize_datetime(item_a.timestamp)

            # Move b_idx forward to items after a_time
            while b_idx < len(items_b):
                b_time = self._normalize_datetime(items_b[b_idx].timestamp)
                if b_time > a_time:
                    break
                b_idx += 1

            # Check B items starting from b_idx
            for i in range(b_idx, len(items_b)):
                item_b = items_b[i]

                # Skip if already matched
                if item_b.item_id in used_b:
                    continue

                b_time = self._normalize_datetime(item_b.timestamp)
                gap = b_time - a_time

                # Check if within max_gap
                if gap > max_gap:
                    break  # No more valid B items for this A

                if gap.total_seconds() > 0:  # Must be strictly after
                    gap_days = gap.total_seconds() / 86400.0
                    co_occurrences.append((item_a.item_id, item_b.item_id, gap_days))
                    used_b.add(item_b.item_id)
                    break  # Found match for this A, move to next A

        return co_occurrences

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

    def detect_day_of_week_patterns(
        self,
        time_range: Tuple[datetime, datetime],
        event_type: Optional[str] = None,
        min_events_per_day: int = 3,
    ) -> List[Dict[str, any]]:
        """Detect day-of-week patterns in event frequency.

        Identifies patterns like "productivity peaks on Tuesdays" or
        "commits are 2x more frequent on Mondays".

        Phase 2 Feature: Weekly rhythm pattern detection.

        Args:
            time_range: (start, end) time range to analyze
            event_type: Optional filter for specific event type
            min_events_per_day: Minimum events per day for significance

        Returns:
            List of day-of-week patterns with statistics

        Example:
            >>> patterns = detector.detect_day_of_week_patterns(
            ...     time_range=(start, end),
            ...     event_type="note_created",
            ... )
            >>> for p in patterns:
            ...     print(f"{p['day_name']}: {p['event_count']} events ({p['deviation_pct']:+.0f}%)")
        """
        start_time, end_time = time_range

        # Get all events in range
        all_events = self._pkg.query_events_in_timerange(
            start=start_time,
            end=end_time,
            event_type=event_type,
        )

        if len(all_events) < min_events_per_day * 7:
            logger.debug(f"Insufficient events ({len(all_events)}) for day-of-week analysis")
            return []

        # Group events by day of week (0=Monday, 6=Sunday)
        day_counts: Dict[int, int] = defaultdict(int)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        for event in all_events:
            dt = self._normalize_datetime(event.timestamp)
            day_of_week = dt.weekday()
            day_counts[day_of_week] += 1

        # Calculate statistics
        total_events = sum(day_counts.values())
        avg_per_day = total_events / 7.0 if total_events > 0 else 0

        patterns = []
        for day_idx in range(7):
            count = day_counts[day_idx]
            deviation = ((count - avg_per_day) / avg_per_day * 100) if avg_per_day > 0 else 0

            patterns.append({
                "day_index": day_idx,
                "day_name": day_names[day_idx],
                "event_count": count,
                "average_count": avg_per_day,
                "deviation_pct": deviation,
                "is_peak": deviation > 25,  # 25% above average
                "is_trough": deviation < -25,  # 25% below average
                "event_type": event_type or "all",
            })

        # Sort by deviation (peaks first)
        patterns.sort(key=lambda p: p["deviation_pct"], reverse=True)

        # Log significant patterns
        peaks = [p for p in patterns if p["is_peak"]]
        troughs = [p for p in patterns if p["is_trough"]]
        if peaks:
            logger.info(f"Detected peak days: {[p['day_name'] for p in peaks]}")
        if troughs:
            logger.info(f"Detected trough days: {[p['day_name'] for p in troughs]}")

        return patterns

    def detect_time_lagged_correlations(
        self,
        event_type_a: str,
        event_type_b: str,
        time_range: Tuple[datetime, datetime],
        lag_hours_range: Tuple[int, int] = (1, 72),
        lag_bucket_hours: int = 6,
    ) -> List[Dict[str, any]]:
        """Detect time-lagged correlations between event types.

        Finds patterns where event A precedes event B by a specific time lag.
        Example: "Meeting notes precede decision documents by 24-48 hours"

        Phase 2 Feature: Time-lagged correlation detection.

        Args:
            event_type_a: First event type (potential precursor)
            event_type_b: Second event type (potential consequent)
            time_range: (start, end) time range to analyze
            lag_hours_range: (min, max) hours to analyze for lag
            lag_bucket_hours: Size of time buckets for analysis

        Returns:
            List of lag patterns with frequency and strength

        Example:
            >>> lags = detector.detect_time_lagged_correlations(
            ...     event_type_a="meeting",
            ...     event_type_b="decision",
            ...     time_range=(start, end),
            ... )
            >>> for lag in lags:
            ...     print(f"{lag['lag_hours']}h: {lag['occurrence_count']} occurrences")
        """
        start_time, end_time = time_range
        min_lag_hours, max_lag_hours = lag_hours_range

        # Get events of both types
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

        if len(events_a) < 3 or len(events_b) < 3:
            logger.debug(f"Insufficient events for time-lagged analysis: A={len(events_a)}, B={len(events_b)}")
            return []

        # Sort by timestamp
        events_a = sorted(events_a, key=lambda e: e.timestamp)
        events_b = sorted(events_b, key=lambda e: e.timestamp)

        # Create lag buckets
        num_buckets = (max_lag_hours - min_lag_hours) // lag_bucket_hours + 1
        lag_buckets: Dict[int, List[Tuple[str, str, float]]] = defaultdict(list)

        # Find A→B pairs within lag range
        used_b = set()
        for event_a in events_a:
            a_time = self._normalize_datetime(event_a.timestamp)

            for event_b in events_b:
                if event_b.id in used_b:
                    continue

                b_time = self._normalize_datetime(event_b.timestamp)
                lag_hours = (b_time - a_time).total_seconds() / 3600.0

                # Check if within lag range
                if min_lag_hours <= lag_hours <= max_lag_hours:
                    # Determine bucket
                    bucket_idx = int((lag_hours - min_lag_hours) // lag_bucket_hours)
                    bucket_hours = min_lag_hours + bucket_idx * lag_bucket_hours
                    lag_buckets[bucket_hours].append((event_a.id, event_b.id, lag_hours))
                    used_b.add(event_b.id)
                    break  # Move to next A event

        # Calculate statistics for each bucket
        patterns = []
        total_pairs = sum(len(pairs) for pairs in lag_buckets.values())

        for bucket_hours in sorted(lag_buckets.keys()):
            pairs = lag_buckets[bucket_hours]
            if not pairs:
                continue

            avg_lag = sum(lag for _, _, lag in pairs) / len(pairs)
            proportion = len(pairs) / total_pairs if total_pairs > 0 else 0

            patterns.append({
                "lag_hours": bucket_hours,
                "lag_range": f"{bucket_hours}-{bucket_hours + lag_bucket_hours}h",
                "occurrence_count": len(pairs),
                "avg_actual_lag_hours": avg_lag,
                "proportion": proportion,
                "event_type_a": event_type_a,
                "event_type_b": event_type_b,
                "is_significant": len(pairs) >= 3 and proportion > 0.15,
            })

        # Sort by occurrence count
        patterns.sort(key=lambda p: p["occurrence_count"], reverse=True)

        # Log significant patterns
        significant = [p for p in patterns if p["is_significant"]]
        if significant:
            logger.info(
                f"Detected significant lag patterns for {event_type_a}→{event_type_b}: "
                f"{[p['lag_range'] for p in significant]}"
            )

        return patterns
