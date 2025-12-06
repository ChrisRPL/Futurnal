"""Temporal Query Engine.

High-level API integrating PKG temporal queries with vector embeddings
for hybrid temporal search. This is the main entry point for Module 01
of the Hybrid Search API.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/01-temporal-query-engine.md

Option B Compliance:
- Temporal-first design: All queries centered on temporal metadata
- Integrates with PKG temporal queries (Module 04)
- Integrates with embedding service for hybrid search
- Foundation for Phase 2 correlation detection and Phase 3 causal inference
- Local-first: All processing on-device
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from futurnal.pkg.queries.models import (
    CausalChainResult,
    TemporalNeighborhood,
    TemporalQueryResult,
)
from futurnal.pkg.schema.models import EventNode
from futurnal.search.config import SearchConfig, TemporalSearchConfig
from futurnal.search.temporal.correlation import TemporalCorrelationDetector
from futurnal.search.temporal.decay import TemporalDecayScorer
from futurnal.search.temporal.exceptions import (
    InvalidTemporalQueryError,
    TemporalSearchError,
)
from futurnal.search.temporal.patterns import TemporalPatternMatcher
from futurnal.search.temporal.results import (
    HybridNeighborhoodResult,
    RecurringPattern,
    ScoredEvent,
    SequenceMatch,
    TemporalCorrelationResult,
    TemporalSearchResult,
)
from futurnal.search.temporal.types import TemporalQuery, TemporalQueryType

if TYPE_CHECKING:
    from futurnal.embeddings.integration import TemporalAwareVectorWriter
    from futurnal.embeddings.models import TemporalEmbeddingContext
    from futurnal.pkg.queries.temporal import TemporalGraphQueries
    from futurnal.privacy.audit import AuditLogger

logger = logging.getLogger(__name__)


class TemporalQueryEngine:
    """High-level temporal query API integrating PKG and embeddings.

    Provides a unified interface for temporal search that:
    - Wraps PKG TemporalGraphQueries for graph-based queries
    - Integrates with TemporalAwareVectorWriter for vector similarity
    - Applies temporal decay scoring for recency weighting
    - Supports pattern matching and correlation detection
    - Enables hybrid graph+vector queries

    This is the main entry point for the Hybrid Search API's temporal
    query capabilities (Module 01).

    Example:
        >>> from futurnal.pkg.queries.temporal import TemporalGraphQueries
        >>> from futurnal.embeddings.integration import TemporalAwareVectorWriter
        >>> from futurnal.search.temporal import TemporalQueryEngine, TemporalQuery

        >>> # Initialize engine
        >>> engine = TemporalQueryEngine(pkg_queries, vector_store)

        >>> # Time range query with decay scoring
        >>> result = engine.query_time_range(
        ...     start=datetime(2024, 1, 1),
        ...     end=datetime(2024, 3, 31),
        ...     enable_decay=True,
        ... )
        >>> for item in result.items:
        ...     print(f"{item.event.name}: {item.final_score:.2f}")

        >>> # Detect temporal correlations
        >>> correlation = engine.query_temporal_correlation(
        ...     event_type_a="Meeting",
        ...     event_type_b="Decision",
        ... )
        >>> if correlation.correlation_found:
        ...     print(correlation.temporal_pattern)

    Attributes:
        config: Temporal search configuration
    """

    def __init__(
        self,
        pkg_queries: "TemporalGraphQueries",
        vector_store: Optional["TemporalAwareVectorWriter"] = None,
        config: Optional[TemporalSearchConfig] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """Initialize the temporal query engine.

        Args:
            pkg_queries: PKG temporal queries service (required)
            vector_store: Vector embedding store (optional, for hybrid queries)
            config: Temporal search configuration
            audit_logger: Optional audit logger for query tracking
        """
        self._pkg = pkg_queries
        self._vectors = vector_store
        self._config = config or TemporalSearchConfig()
        self._audit = audit_logger

        # Initialize components
        self._decay_scorer = TemporalDecayScorer(config=self._config)
        self._pattern_matcher = TemporalPatternMatcher(pkg_queries, self._config)
        self._correlation_detector = TemporalCorrelationDetector(pkg_queries, self._config)

        logger.info(
            f"Initialized TemporalQueryEngine with decay_half_life="
            f"{self._config.decay_half_life_days} days"
        )

    @property
    def config(self) -> TemporalSearchConfig:
        """Get the configuration."""
        return self._config

    @property
    def decay_scorer(self) -> TemporalDecayScorer:
        """Get the decay scorer component."""
        return self._decay_scorer

    @property
    def pattern_matcher(self) -> TemporalPatternMatcher:
        """Get the pattern matcher component."""
        return self._pattern_matcher

    @property
    def correlation_detector(self) -> TemporalCorrelationDetector:
        """Get the correlation detector component."""
        return self._correlation_detector

    # -------------------------------------------------------------------------
    # Unified Query Interface
    # -------------------------------------------------------------------------

    def query(self, query: TemporalQuery) -> Union[
        TemporalSearchResult,
        TemporalNeighborhood,
        HybridNeighborhoodResult,
        List[SequenceMatch],
        TemporalCorrelationResult,
        CausalChainResult,
    ]:
        """Execute a temporal query based on query type.

        Unified dispatcher that routes to the appropriate method
        based on the query type.

        Args:
            query: TemporalQuery specification

        Returns:
            Result type depends on query_type:
            - TIME_RANGE, BEFORE, AFTER: TemporalSearchResult
            - TEMPORAL_NEIGHBORHOOD: TemporalNeighborhood or HybridNeighborhoodResult
            - TEMPORAL_SEQUENCE: List[SequenceMatch]
            - TEMPORAL_CORRELATION: TemporalCorrelationResult
            - CAUSAL_CHAIN: CausalChainResult

        Raises:
            InvalidTemporalQueryError: If query parameters are invalid
            TemporalSearchError: If query execution fails
        """
        start_time = time.perf_counter()
        logger.debug(f"Executing query: {query.query_type.value}")

        try:
            result = self._dispatch_query(query)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self._audit_query(query.query_type.value, elapsed_ms)
            logger.debug(f"Query completed in {elapsed_ms:.1f}ms")

            return result

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def _dispatch_query(self, query: TemporalQuery):
        """Dispatch query to appropriate handler."""
        qt = query.query_type

        if qt == TemporalQueryType.TIME_RANGE:
            return self.query_time_range(
                start=query.start_time,
                end=query.end_time,
                event_types=query.event_types,
                enable_decay=query.enable_decay_scoring,
                limit=query.limit,
                offset=query.offset,
            )

        elif qt == TemporalQueryType.BEFORE:
            events = self.query_before(
                reference_event_id=query.reference_event_id,
                reference_timestamp=query.reference_timestamp,
                time_window=query.time_window,
                limit=query.limit,
            )
            return self._events_to_search_result(
                events, query.enable_decay_scoring
            )

        elif qt == TemporalQueryType.AFTER:
            events = self.query_after(
                reference_event_id=query.reference_event_id,
                reference_timestamp=query.reference_timestamp,
                time_window=query.time_window,
                limit=query.limit,
            )
            return self._events_to_search_result(
                events, query.enable_decay_scoring
            )

        elif qt == TemporalQueryType.TEMPORAL_NEIGHBORHOOD:
            if query.query_text and self._vectors:
                return self.query_temporal_neighborhood_hybrid(
                    reference_event_id=query.reference_event_id,
                    time_window=query.time_window or timedelta(days=7),
                    query_text=query.query_text,
                    vector_weight=query.vector_weight,
                )
            return self.query_temporal_neighborhood(
                entity_id=query.reference_event_id,
                time_window=query.time_window or timedelta(days=7),
            )

        elif qt == TemporalQueryType.TEMPORAL_SEQUENCE:
            return self.query_temporal_sequence(
                pattern=query.pattern,
                time_range=(query.start_time, query.end_time),
                max_gap=query.max_gap,
            )

        elif qt == TemporalQueryType.TEMPORAL_CORRELATION:
            return self.query_temporal_correlation(
                event_type_a=query.event_type_a,
                event_type_b=query.event_type_b,
                time_range=(query.start_time, query.end_time) if query.start_time else None,
                max_gap=query.max_gap,
                min_occurrences=query.min_occurrences,
            )

        elif qt == TemporalQueryType.CAUSAL_CHAIN:
            return self.query_causal_chain(
                start_event_id=query.reference_event_id,
                max_hops=query.max_hops,
            )

        else:
            raise InvalidTemporalQueryError(
                f"Unsupported query type: {qt}",
                query_type=qt.value,
            )

    # -------------------------------------------------------------------------
    # Time Range Queries
    # -------------------------------------------------------------------------

    def query_time_range(
        self,
        start: datetime,
        end: datetime,
        event_types: Optional[List[str]] = None,
        enable_decay: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> TemporalSearchResult:
        """Query events in a time range with optional decay scoring.

        Delegates to PKG TemporalGraphQueries and applies decay scoring.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            event_types: Optional filter by event types
            enable_decay: Apply decay scoring. Default: True
            limit: Maximum results. Default: 100
            offset: Pagination offset. Default: 0

        Returns:
            TemporalSearchResult with scored events

        Example:
            >>> result = engine.query_time_range(
            ...     start=datetime(2024, 1, 1),
            ...     end=datetime(2024, 3, 31),
            ...     event_types=["meeting", "decision"],
            ... )
        """
        start_time = time.perf_counter()

        # Delegate to PKG
        if event_types and len(event_types) == 1:
            events = self._pkg.query_events_in_timerange(
                start=start,
                end=end,
                event_type=event_types[0],
                limit=limit,
                offset=offset,
            )
        else:
            events = self._pkg.query_events_in_timerange(
                start=start,
                end=end,
                limit=limit,
                offset=offset,
            )
            # Filter by event types if multiple specified
            if event_types:
                events = [e for e in events if e.event_type in event_types]

        # Apply decay scoring
        if enable_decay and self._config.enable_decay_by_default:
            scored_events = self._decay_scorer.apply_decay(events)
        else:
            scored_events = [ScoredEvent(event=e) for e in events]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TemporalSearchResult(
            items=scored_events,
            total_count=len(scored_events),
            offset=offset,
            limit=limit,
            query_time_ms=elapsed_ms,
            decay_applied=enable_decay,
        )

    def query_before(
        self,
        reference_event_id: Optional[str] = None,
        reference_timestamp: Optional[datetime] = None,
        time_window: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[EventNode]:
        """Query events before a reference point.

        Args:
            reference_event_id: ID of reference event
            reference_timestamp: Reference timestamp (alternative to event_id)
            time_window: Maximum time before reference
            limit: Maximum results

        Returns:
            List of events before reference, ordered by timestamp descending
        """
        if reference_event_id:
            return self._pkg.query_events_before(
                reference_event_id=reference_event_id,
                time_window=time_window,
                limit=limit,
            )
        elif reference_timestamp:
            start = reference_timestamp - time_window if time_window else datetime.min
            events = self._pkg.query_events_in_timerange(
                start=start,
                end=reference_timestamp,
                limit=limit,
            )
            return sorted(events, key=lambda e: e.timestamp, reverse=True)
        else:
            raise InvalidTemporalQueryError(
                "BEFORE query requires reference_event_id or reference_timestamp",
                query_type="before",
            )

    def query_after(
        self,
        reference_event_id: Optional[str] = None,
        reference_timestamp: Optional[datetime] = None,
        time_window: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[EventNode]:
        """Query events after a reference point.

        Args:
            reference_event_id: ID of reference event
            reference_timestamp: Reference timestamp (alternative to event_id)
            time_window: Maximum time after reference
            limit: Maximum results

        Returns:
            List of events after reference, ordered by timestamp ascending
        """
        if reference_event_id:
            return self._pkg.query_events_after(
                reference_event_id=reference_event_id,
                time_window=time_window,
                limit=limit,
            )
        elif reference_timestamp:
            end = reference_timestamp + time_window if time_window else datetime.max
            events = self._pkg.query_events_in_timerange(
                start=reference_timestamp,
                end=end,
                limit=limit,
            )
            return sorted(events, key=lambda e: e.timestamp)
        else:
            raise InvalidTemporalQueryError(
                "AFTER query requires reference_event_id or reference_timestamp",
                query_type="after",
            )

    # -------------------------------------------------------------------------
    # Temporal Neighborhood Queries
    # -------------------------------------------------------------------------

    def query_temporal_neighborhood(
        self,
        entity_id: str,
        time_window: timedelta,
        include_events: bool = True,
        include_entities: bool = True,
    ) -> TemporalNeighborhood:
        """Query temporal neighborhood from PKG.

        Delegates directly to PKG TemporalGraphQueries.

        Args:
            entity_id: Center entity ID
            time_window: Time window for neighborhood
            include_events: Include Event neighbors
            include_entities: Include non-Event neighbors

        Returns:
            TemporalNeighborhood from PKG
        """
        return self._pkg.query_temporal_neighborhood(
            entity_id=entity_id,
            time_window=time_window,
            include_events=include_events,
            include_entities=include_entities,
        )

    def query_temporal_neighborhood_hybrid(
        self,
        reference_event_id: str,
        time_window: timedelta,
        query_text: str,
        vector_weight: float = 0.3,
        top_k: int = 20,
    ) -> HybridNeighborhoodResult:
        """Query temporal neighborhood with hybrid graph+vector search.

        Combines PKG graph traversal with vector similarity search
        for more relevant results.

        Args:
            reference_event_id: Center event ID
            time_window: Time window for neighborhood
            query_text: Text query for vector similarity
            vector_weight: Weight for vector results (0-1). Default: 0.3
            top_k: Max vector results. Default: 20

        Returns:
            HybridNeighborhoodResult combining graph and vector results

        Raises:
            TemporalSearchError: If vector store not configured
        """
        if not self._vectors:
            raise TemporalSearchError(
                "Vector store not configured for hybrid queries",
                query_type="hybrid_neighborhood",
            )

        start_time = time.perf_counter()

        # Get graph-based neighborhood
        graph_start = time.perf_counter()
        neighborhood = self._pkg.query_temporal_neighborhood(
            entity_id=reference_event_id,
            time_window=time_window,
        )
        graph_time = (time.perf_counter() - graph_start) * 1000

        # Convert to ScoredEvents
        graph_neighbors = [
            ScoredEvent(
                event=n,
                base_score=1.0,
                decay_score=self._decay_scorer.compute_decay_factor(n.timestamp),
            )
            for n in neighborhood.event_neighbors
        ]

        # Get vector similarity results
        vector_start = time.perf_counter()
        query_embedding = self._get_query_embedding(query_text)

        # Build timestamp filter for ChromaDB
        time_bounds = neighborhood.time_bounds
        timestamp_filter = {
            "$and": [
                {"timestamp": {"$gte": time_bounds[0].isoformat()}},
                {"timestamp": {"$lte": time_bounds[1].isoformat()}},
            ]
        } if time_bounds else None

        similar = self._vectors.search_similar_events(
            query_embedding=query_embedding,
            top_k=top_k,
            timestamp_filter=timestamp_filter,
        )
        vector_time = (time.perf_counter() - vector_start) * 1000

        # Convert to ScoredEvents (need to fetch full event data)
        vector_neighbors = []
        for sim_result in similar:
            # Try to find event in graph results first
            event = None
            for gn in graph_neighbors:
                if gn.event_id == sim_result.entity_id:
                    event = gn.event
                    break

            # If not found, we'd need to fetch from PKG
            # For now, skip events not in graph neighborhood
            if event:
                vector_neighbors.append(ScoredEvent(
                    event=event,
                    base_score=1.0,
                    decay_score=1.0,
                    vector_similarity=sim_result.similarity_score,
                ))

        total_time = (time.perf_counter() - start_time) * 1000

        return HybridNeighborhoodResult(
            center_event_id=reference_event_id,
            center_event=neighborhood.center_entity if isinstance(
                neighborhood.center_entity, EventNode
            ) else None,
            graph_neighbors=graph_neighbors,
            vector_neighbors=vector_neighbors,
            vector_weight=vector_weight,
            time_window=time_window,
            time_bounds=time_bounds,
            query_time_ms=total_time,
            graph_query_time_ms=graph_time,
            vector_query_time_ms=vector_time,
        )

    # -------------------------------------------------------------------------
    # Pattern Matching
    # -------------------------------------------------------------------------

    def query_temporal_sequence(
        self,
        pattern: List[str],
        time_range: Tuple[datetime, datetime],
        max_gap: Optional[timedelta] = None,
    ) -> List[SequenceMatch]:
        """Find event sequences matching a pattern.

        Delegates to TemporalPatternMatcher.

        Args:
            pattern: Event types in order (e.g., ["Meeting", "Decision"])
            time_range: (start, end) time range to search
            max_gap: Maximum gap between events

        Returns:
            List of matching SequenceMatch objects

        Example:
            >>> matches = engine.query_temporal_sequence(
            ...     pattern=["Meeting", "Decision", "Publication"],
            ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            ... )
        """
        return self._pattern_matcher.find_sequences(
            pattern=pattern,
            time_range=time_range,
            max_gap=max_gap,
        )

    def find_recurring_patterns(
        self,
        time_range: Tuple[datetime, datetime],
        min_occurrences: int = 3,
        min_pattern_length: int = 2,
    ) -> List[RecurringPattern]:
        """Discover recurring temporal patterns.

        Delegates to TemporalPatternMatcher.

        Args:
            time_range: (start, end) time range to analyze
            min_occurrences: Minimum occurrences for significance
            min_pattern_length: Minimum pattern length

        Returns:
            List of RecurringPattern objects

        Example:
            >>> patterns = engine.find_recurring_patterns(
            ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            ...     min_occurrences=5,
            ... )
        """
        return self._pattern_matcher.find_recurring_patterns(
            time_range=time_range,
            min_pattern_length=min_pattern_length,
            min_occurrences=min_occurrences,
        )

    # -------------------------------------------------------------------------
    # Correlation Detection
    # -------------------------------------------------------------------------

    def query_temporal_correlation(
        self,
        event_type_a: str,
        event_type_b: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        max_gap: Optional[timedelta] = None,
        min_occurrences: Optional[int] = None,
    ) -> TemporalCorrelationResult:
        """Detect temporal correlation between event types.

        Delegates to TemporalCorrelationDetector.

        Args:
            event_type_a: First event type (potential precedent)
            event_type_b: Second event type (potential consequent)
            time_range: Optional time range to analyze
            max_gap: Maximum gap for correlation
            min_occurrences: Minimum co-occurrences for significance

        Returns:
            TemporalCorrelationResult with correlation statistics

        Example:
            >>> result = engine.query_temporal_correlation(
            ...     event_type_a="Meeting",
            ...     event_type_b="Decision",
            ... )
            >>> if result.correlation_found:
            ...     print(result.temporal_pattern)
        """
        return self._correlation_detector.detect_correlation(
            event_type_a=event_type_a,
            event_type_b=event_type_b,
            time_range=time_range,
            max_gap=max_gap,
            min_occurrences=min_occurrences,
        )

    def scan_all_correlations(
        self,
        time_range: Tuple[datetime, datetime],
        min_occurrences: Optional[int] = None,
    ) -> List[TemporalCorrelationResult]:
        """Scan for all significant correlations.

        Delegates to TemporalCorrelationDetector.

        Args:
            time_range: (start, end) time range to analyze
            min_occurrences: Minimum co-occurrences for significance

        Returns:
            List of TemporalCorrelationResult sorted by strength

        Example:
            >>> correlations = engine.scan_all_correlations(
            ...     time_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            ... )
        """
        return self._correlation_detector.scan_all_correlations(
            time_range=time_range,
            min_occurrences=min_occurrences,
        )

    # -------------------------------------------------------------------------
    # Causal Chain Queries
    # -------------------------------------------------------------------------

    def query_causal_chain(
        self,
        start_event_id: str,
        max_hops: int = 5,
    ) -> CausalChainResult:
        """Query causal chains from PKG.

        Delegates directly to PKG TemporalGraphQueries.

        Args:
            start_event_id: Starting event ID
            max_hops: Maximum chain depth (1-10)

        Returns:
            CausalChainResult from PKG
        """
        return self._pkg.query_causal_chain(
            start_event_id=start_event_id,
            max_hops=max_hops,
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for query text.

        Uses the vector store's temporal embedder to generate
        an embedding for the query.

        Args:
            query_text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        from futurnal.embeddings.models import TemporalEmbeddingContext

        context = TemporalEmbeddingContext(timestamp=datetime.utcnow())
        result = self._vectors.temporal_embedder.embed(
            event_name=query_text,
            event_description="",
            temporal_context=context,
        )
        return list(result.embedding)

    def _events_to_search_result(
        self,
        events: List[EventNode],
        apply_decay: bool,
    ) -> TemporalSearchResult:
        """Convert event list to TemporalSearchResult."""
        if apply_decay:
            scored = self._decay_scorer.apply_decay(events)
        else:
            scored = [ScoredEvent(event=e) for e in events]

        return TemporalSearchResult(
            items=scored,
            total_count=len(scored),
            decay_applied=apply_decay,
        )

    def _audit_query(self, query_type: str, elapsed_ms: float) -> None:
        """Record query to audit log."""
        if self._audit is None:
            return

        try:
            self._audit.record(
                job_id=f"temporal_search_{query_type}_{datetime.utcnow().isoformat()}",
                source="temporal_query_engine",
                action=f"search_{query_type}",
                status="completed",
                timestamp=datetime.utcnow(),
                metadata={
                    "query_type": query_type,
                    "elapsed_ms": elapsed_ms,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to record audit event: {e}")
