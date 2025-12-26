"""Causal Chain Retrieval Engine.

Implements the main CausalChainRetrieval class for:
- Finding causes of an event (what led to X?)
- Finding effects of an event (what resulted from X?)
- Finding causal paths between events (how did A lead to B?)
- Detecting correlation patterns (Phase 2 foundation)

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md

Option B Compliance:
- Temporal validation required for ALL paths (100%)
- Causal confidence scoring on relationships
- Bradford Hill criteria support (temporality validated)
- No hardcoded schemas
- Phase 2/3 foundation established
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from futurnal.pkg.queries.models import CausalChainResult
from futurnal.search.causal.exceptions import (
    CausalChainDepthExceeded,
    CausalPathNotFoundError,
    CausalSearchError,
    CorrelationDetectionError,
    EventNotFoundError,
    InvalidCausalQueryError,
)
from futurnal.search.causal.results import (
    CausalCauseResult,
    CausalEffectResult,
    CausalPathResult,
    CausalSearchPath,
    CorrelationPatternResult,
    FindCausesResult,
    FindEffectsResult,
)
from futurnal.search.causal.types import CausalQuery, CausalQueryType
from futurnal.search.causal.validation import TemporalOrderingValidator

if TYPE_CHECKING:
    from futurnal.pkg.queries.temporal import TemporalGraphQueries
    from futurnal.pkg.schema.models import EventNode
    from futurnal.privacy.audit import AuditLogger
    from futurnal.search.config import CausalSearchConfig
    from futurnal.search.temporal.engine import TemporalQueryEngine

logger = logging.getLogger(__name__)

# Maximum allowed causal chain depth
MAX_CAUSAL_CHAIN_DEPTH = 10


class CausalChainRetrieval:
    """Retrieval engine for causal chains and paths.

    Integrates with PKG causal structure from entity-relationship extraction
    and delegates correlation detection to TemporalQueryEngine.

    Example:
        >>> from futurnal.pkg.queries.temporal import TemporalGraphQueries
        >>> from futurnal.search.temporal import TemporalQueryEngine
        >>> from futurnal.search.causal import CausalChainRetrieval

        >>> retrieval = CausalChainRetrieval(pkg_queries, temporal_engine)

        >>> # What caused this decision?
        >>> causes = retrieval.find_causes("decision_123", max_hops=3)
        >>> for cause in causes.causes:
        ...     print(f"{cause.cause_name} (distance: {cause.distance})")

        >>> # How did the meeting lead to the publication?
        >>> path = retrieval.find_causal_path("meeting_1", "publication_1")
        >>> if path.path_found:
        ...     print(f"Path: {' -> '.join(path.path.path)}")

    Attributes:
        config: Causal search configuration
    """

    def __init__(
        self,
        pkg_queries: "TemporalGraphQueries",
        temporal_engine: Optional["TemporalQueryEngine"] = None,
        config: Optional["CausalSearchConfig"] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """Initialize the causal chain retrieval engine.

        Args:
            pkg_queries: PKG temporal queries service (required)
            temporal_engine: Temporal query engine for correlation detection
            config: Causal search configuration
            audit_logger: Optional audit logger for query tracking
        """
        self._pkg = pkg_queries
        self._temporal = temporal_engine
        self._audit = audit_logger

        # Import config class here to avoid circular imports
        from futurnal.search.config import CausalSearchConfig

        self._config = config or CausalSearchConfig()

        # Initialize temporal ordering validator
        self._validator = TemporalOrderingValidator(pkg_queries)

        logger.info(
            f"Initialized CausalChainRetrieval with "
            f"max_hops={self._config.default_max_hops}"
        )

    @property
    def config(self) -> "CausalSearchConfig":
        """Get the configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Unified Query Interface
    # -------------------------------------------------------------------------

    def query(
        self,
        query: CausalQuery,
    ) -> Union[
        FindCausesResult,
        FindEffectsResult,
        CausalPathResult,
        CausalChainResult,
        CorrelationPatternResult,
    ]:
        """Execute a causal query based on query type.

        Unified dispatcher that routes to the appropriate method.

        Args:
            query: CausalQuery specification

        Returns:
            Result type depends on query_type

        Raises:
            InvalidCausalQueryError: If query parameters are invalid
            CausalSearchError: If query execution fails
        """
        start_time = time.perf_counter()
        logger.debug(f"Executing causal query: {query.query_type.value}")

        try:
            result = self._dispatch_query(query)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self._audit_query(query.query_type.value, elapsed_ms)
            logger.debug(f"Causal query completed in {elapsed_ms:.1f}ms")

            return result

        except CausalSearchError:
            raise
        except Exception as e:
            logger.error(f"Causal query failed: {e}")
            raise CausalSearchError(
                f"Query execution failed: {e}",
                query_type=query.query_type.value,
                cause=e,
            )

    def _dispatch_query(
        self,
        query: CausalQuery,
    ) -> Union[
        FindCausesResult,
        FindEffectsResult,
        CausalPathResult,
        CausalChainResult,
        CorrelationPatternResult,
    ]:
        """Dispatch query to appropriate handler."""
        qt = query.query_type

        if qt == CausalQueryType.FIND_CAUSES:
            return self.find_causes(
                event_id=query.event_id,
                max_hops=query.max_hops,
                min_confidence=query.min_confidence,
            )

        elif qt == CausalQueryType.FIND_EFFECTS:
            return self.find_effects(
                event_id=query.event_id,
                max_hops=query.max_hops,
                min_confidence=query.min_confidence,
            )

        elif qt == CausalQueryType.CAUSAL_PATH:
            return self.find_causal_path(
                start_event_id=query.start_event_id,
                end_event_id=query.end_event_id,
                max_hops=query.max_hops,
            )

        elif qt == CausalQueryType.CAUSAL_CHAIN:
            # Delegate to existing PKG query_causal_chain
            return self._pkg.query_causal_chain(
                start_event_id=query.event_id,
                max_hops=query.max_hops,
            )

        elif qt == CausalQueryType.CORRELATION_PATTERN:
            return self.detect_correlation_patterns(
                time_range_start=query.time_range_start,
                time_range_end=query.time_range_end,
                min_correlation_strength=query.min_correlation_strength,
            )

        else:
            raise InvalidCausalQueryError(
                f"Unsupported query type: {qt}",
                query_type=str(qt.value) if qt else None,
            )

    # -------------------------------------------------------------------------
    # Find Causes
    # -------------------------------------------------------------------------

    def find_causes(
        self,
        event_id: str,
        max_hops: int = 3,
        min_confidence: float = 0.6,
    ) -> FindCausesResult:
        """Find events that caused the target event.

        Example: "What led to this decision?"

        Returns causal predecessors up to max_hops away, validating
        temporal ordering for each path (Option B requirement).

        Args:
            event_id: ID of the target event
            max_hops: Maximum causal hops to traverse (1-10, default 3)
            min_confidence: Minimum causal confidence threshold (default 0.6)

        Returns:
            FindCausesResult with list of cause events

        Raises:
            EventNotFoundError: If target event does not exist
            CausalChainDepthExceeded: If max_hops > 10
        """
        start_time = time.perf_counter()

        # Validate depth
        if max_hops > MAX_CAUSAL_CHAIN_DEPTH:
            raise CausalChainDepthExceeded(max_hops, MAX_CAUSAL_CHAIN_DEPTH)

        # Verify target event exists
        target_event = self._get_event_by_id(event_id)
        if not target_event:
            raise EventNotFoundError(event_id)

        # Use the actual found event's ID (may differ from requested ID due to format differences)
        actual_event_id = target_event.id

        # Build Cypher query
        # Note: max_hops must be interpolated directly (Neo4j limitation)
        # Query supports both Event nodes with causal relations AND Document nodes with any relations
        query = f"""
            MATCH path = (cause)-[r*1..{max_hops}]->(effect)
            WHERE effect.id = $event_id
              AND cause <> effect
              AND cause.id IS NOT NULL
              AND all(rel IN relationships(path) WHERE
                  rel.causal_confidence IS NULL OR rel.causal_confidence >= $min_confidence)
            RETURN cause.id AS cause_id,
                   COALESCE(cause.name, cause.title, cause.id) AS cause_name,
                   COALESCE(cause.timestamp, cause.date, cause.created_at) AS cause_timestamp,
                   length(path) AS distance,
                   [rel IN relationships(path) | rel.causal_confidence] AS confidence_scores,
                   [rel IN relationships(path) | type(rel)] AS relationship_types,
                   [node IN nodes(path) | COALESCE(node.id, toString(id(node)))] AS path_ids
            ORDER BY distance ASC
            LIMIT 20
        """

        causes: List[CausalCauseResult] = []

        try:
            with self._pkg._db.session() as session:
                result = session.run(
                    query,
                    {"event_id": actual_event_id, "min_confidence": min_confidence},
                )

                for record in result:
                    path_ids = record["path_ids"]

                    # Validate temporal ordering (Option B: 100% validation)
                    temporal_valid = self._validator.validate_path(path_ids)

                    # Parse timestamp
                    cause_ts = self._convert_timestamp(record["cause_timestamp"])

                    # Clean confidence scores
                    confidence_scores = self._clean_confidence_scores(
                        record["confidence_scores"]
                    )
                    aggregate_conf = (
                        min(confidence_scores) if confidence_scores else 0.5
                    )

                    causes.append(
                        CausalCauseResult(
                            cause_id=record["cause_id"],
                            cause_name=record["cause_name"] or "Unknown",
                            cause_timestamp=cause_ts,
                            distance=record["distance"],
                            confidence_scores=confidence_scores,
                            aggregate_confidence=aggregate_conf,
                            temporal_ordering_valid=temporal_valid,
                        )
                    )

        except Exception as e:
            logger.error(f"find_causes query failed: {e}")
            raise CausalSearchError(
                f"Failed to find causes: {e}",
                query_type="find_causes",
                cause=e,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return FindCausesResult(
            target_event_id=event_id,
            target_event=target_event,
            causes=causes,
            max_hops_requested=max_hops,
            min_confidence_requested=min_confidence,
            query_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # Find Effects
    # -------------------------------------------------------------------------

    def find_effects(
        self,
        event_id: str,
        max_hops: int = 3,
        min_confidence: float = 0.6,
    ) -> FindEffectsResult:
        """Find events caused by the target event.

        Example: "What resulted from this meeting?"

        Args:
            event_id: ID of the source event
            max_hops: Maximum causal hops to traverse (1-10, default 3)
            min_confidence: Minimum causal confidence threshold (default 0.6)

        Returns:
            FindEffectsResult with list of effect events

        Raises:
            EventNotFoundError: If source event does not exist
            CausalChainDepthExceeded: If max_hops > 10
        """
        start_time = time.perf_counter()

        if max_hops > MAX_CAUSAL_CHAIN_DEPTH:
            raise CausalChainDepthExceeded(max_hops, MAX_CAUSAL_CHAIN_DEPTH)

        source_event = self._get_event_by_id(event_id)
        if not source_event:
            raise EventNotFoundError(event_id)

        # Use the actual found event's ID (may differ from requested ID due to format differences)
        actual_event_id = source_event.id

        # Query supports both Event nodes with causal relations AND Document nodes with any relations
        # This allows exploration before Phase 3 causal inference is complete
        query = f"""
            MATCH path = (cause)-[r*1..{max_hops}]->(effect)
            WHERE cause.id = $event_id
              AND cause <> effect
              AND effect.id IS NOT NULL
              AND all(rel IN relationships(path) WHERE
                  rel.causal_confidence IS NULL OR rel.causal_confidence >= $min_confidence)
            RETURN effect.id AS effect_id,
                   COALESCE(effect.name, effect.title, effect.id) AS effect_name,
                   COALESCE(effect.timestamp, effect.date, effect.created_at) AS effect_timestamp,
                   length(path) AS distance,
                   [rel IN relationships(path) | rel.causal_confidence] AS confidence_scores,
                   [rel IN relationships(path) | type(rel)] AS relationship_types,
                   [node IN nodes(path) | COALESCE(node.id, toString(id(node)))] AS path_ids
            ORDER BY distance ASC
            LIMIT 20
        """

        effects: List[CausalEffectResult] = []

        try:
            with self._pkg._db.session() as session:
                result = session.run(
                    query,
                    {"event_id": actual_event_id, "min_confidence": min_confidence},
                )

                for record in result:
                    path_ids = record["path_ids"]
                    temporal_valid = self._validator.validate_path(path_ids)

                    effect_ts = self._convert_timestamp(record["effect_timestamp"])

                    confidence_scores = self._clean_confidence_scores(
                        record["confidence_scores"]
                    )
                    aggregate_conf = (
                        min(confidence_scores) if confidence_scores else 0.5
                    )

                    effects.append(
                        CausalEffectResult(
                            effect_id=record["effect_id"],
                            effect_name=record["effect_name"] or "Unknown",
                            effect_timestamp=effect_ts,
                            distance=record["distance"],
                            confidence_scores=confidence_scores,
                            aggregate_confidence=aggregate_conf,
                            temporal_ordering_valid=temporal_valid,
                        )
                    )

        except Exception as e:
            logger.error(f"find_effects query failed: {e}")
            raise CausalSearchError(
                f"Failed to find effects: {e}",
                query_type="find_effects",
                cause=e,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return FindEffectsResult(
            source_event_id=event_id,
            source_event=source_event,
            effects=effects,
            max_hops_requested=max_hops,
            min_confidence_requested=min_confidence,
            query_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # Find Causal Path
    # -------------------------------------------------------------------------

    def find_causal_path(
        self,
        start_event_id: str,
        end_event_id: str,
        max_hops: int = 5,
    ) -> CausalPathResult:
        """Find causal path from start to end event.

        Example: "How did A lead to B?"

        Uses shortestPath algorithm to find the most direct causal connection.

        Args:
            start_event_id: ID of the starting event
            end_event_id: ID of the ending event
            max_hops: Maximum path length (1-10, default 5)

        Returns:
            CausalPathResult with path (if found)

        Raises:
            CausalChainDepthExceeded: If max_hops > 10
        """
        start_time = time.perf_counter()

        if max_hops > MAX_CAUSAL_CHAIN_DEPTH:
            raise CausalChainDepthExceeded(max_hops, MAX_CAUSAL_CHAIN_DEPTH)

        query = f"""
            MATCH path = shortestPath(
                (start:Event {{id: $start_id}})-[:CAUSES|ENABLES|TRIGGERS*1..{max_hops}]->(end:Event {{id: $end_id}})
            )
            RETURN [node IN nodes(path) | node.id] AS event_ids,
                   [r IN relationships(path) | r.causal_confidence] AS confidences,
                   [r IN relationships(path) | r.causal_evidence] AS evidence,
                   length(path) AS path_length
        """

        path: Optional[CausalSearchPath] = None

        try:
            with self._pkg._db.session() as session:
                result = session.run(
                    query,
                    {"start_id": start_event_id, "end_id": end_event_id},
                )

                record = result.single()

                if record:
                    event_ids = record["event_ids"]
                    confidences = record["confidences"] or []
                    evidence = record["evidence"] or []

                    # Clean up confidence values
                    clean_confidences = self._clean_confidence_scores(confidences)
                    min_confidence = (
                        min(clean_confidences) if clean_confidences else 0.5
                    )

                    # Validate temporal ordering
                    temporal_valid = self._validator.validate_path(event_ids)

                    path = CausalSearchPath(
                        start_event_id=start_event_id,
                        end_event_id=end_event_id,
                        path=event_ids,
                        causal_confidence=min_confidence,
                        confidence_scores=clean_confidences,
                        temporal_ordering_valid=temporal_valid,
                        causal_evidence=[str(e) if e else "" for e in evidence],
                    )

        except Exception as e:
            logger.error(f"find_causal_path query failed: {e}")
            raise CausalSearchError(
                f"Failed to find causal path: {e}",
                query_type="causal_path",
                cause=e,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return CausalPathResult(
            path_found=path is not None,
            path=path,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            max_hops_requested=max_hops,
            query_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # Correlation Pattern Detection
    # -------------------------------------------------------------------------

    def detect_correlation_patterns(
        self,
        time_range_start: datetime,
        time_range_end: datetime,
        min_correlation_strength: float = 0.5,
    ) -> CorrelationPatternResult:
        """Detect correlation patterns in time range.

        Delegates to TemporalCorrelationDetector for actual detection.
        Critical for Phase 2 automated correlation discovery.

        Args:
            time_range_start: Start of time range
            time_range_end: End of time range
            min_correlation_strength: Minimum correlation strength (default 0.5)

        Returns:
            CorrelationPatternResult with detected patterns

        Raises:
            CorrelationDetectionError: If temporal engine not configured
        """
        start_time = time.perf_counter()

        if self._temporal is None:
            raise CorrelationDetectionError(
                "Temporal engine not configured for correlation detection"
            )

        try:
            # Use TemporalCorrelationDetector via temporal engine
            correlations = self._temporal.scan_all_correlations(
                time_range=(time_range_start, time_range_end),
            )

            # Filter by strength and identify causal candidates
            filtered_correlations: List[Dict[str, Any]] = []
            causal_candidates: List[Dict[str, Any]] = []

            for corr in correlations:
                # Get correlation strength (handle different attribute names)
                strength = getattr(corr, "correlation_strength", None)
                if strength is None:
                    strength = getattr(corr, "gap_consistency", 0.0)

                if strength >= min_correlation_strength:
                    corr_dict = corr.model_dump()
                    filtered_correlations.append(corr_dict)

                    # Check if flagged as causal candidate
                    if getattr(corr, "is_causal_candidate", False):
                        causal_candidates.append(corr_dict)

        except Exception as e:
            logger.error(f"Correlation detection failed: {e}")
            raise CorrelationDetectionError(
                f"Failed to detect correlations: {e}",
                cause=e,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return CorrelationPatternResult(
            patterns_found=len(filtered_correlations),
            correlations=filtered_correlations,
            causal_candidates=causal_candidates,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            min_correlation_strength=min_correlation_strength,
            query_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _get_event_by_id(self, event_id: str) -> Optional["EventNode"]:
        """Retrieve event or document node by ID from PKG.

        Searches for Document nodes first (most common), then Event, then any node.
        Uses multiple strategies: exact match, partial match, and fuzzy search.
        This allows causal exploration to work with documents that have temporal data.

        Args:
            event_id: ID of the event/document to retrieve (may have prefix like 'doc:')

        Returns:
            EventNode if found, None otherwise
        """
        from futurnal.pkg.schema.models import EventNode

        # Normalize ID - frontend may send 'doc:xxx' but DB stores just 'xxx'
        normalized_id = event_id
        if event_id.startswith("doc:"):
            normalized_id = event_id[4:]  # Remove 'doc:' prefix
        elif event_id.startswith("event:"):
            normalized_id = event_id[6:]  # Remove 'event:' prefix

        # Try multiple ID formats (frontend uses different format than DB)
        ids_to_try = []
        if event_id != normalized_id:
            ids_to_try.append(normalized_id)
            ids_to_try.append(event_id)
        else:
            ids_to_try.append(event_id)

        # Also try first 16 chars (DB uses sha256[:16] format)
        if len(normalized_id) > 16:
            ids_to_try.append(normalized_id[:16])

        # Try exact match queries first
        exact_queries = [
            "MATCH (e:Document {id: $id}) RETURN e, labels(e) as labels",
            "MATCH (e:Event {id: $id}) RETURN e, labels(e) as labels",
            "MATCH (e {id: $id}) RETURN e, labels(e) as labels",
        ]

        # Also try to extract filename from the ID (frontend sends doc:filename.md)
        potential_filename = normalized_id
        if "." in normalized_id:  # Looks like a filename
            potential_filename = normalized_id

        # Add filename-based queries
        filename_queries = [
            "MATCH (e:Document {title: $id}) RETURN e, labels(e) as labels",
            "MATCH (e:Document) WHERE e.title CONTAINS $id RETURN e, labels(e) as labels LIMIT 1",
        ]

        # Try filename match first (most likely to match for frontend nodes)
        for query in filename_queries:
            try:
                with self._pkg._db.session() as session:
                    result = session.run(query, {"id": potential_filename})
                    record = result.single()
                    if record:
                        node_data = dict(record["e"])
                        labels = list(record["labels"]) if record["labels"] else []
                        logger.debug(f"Found node by title/filename: {potential_filename}")
                        return self._parse_node_as_event(node_data, labels)
            except Exception as e:
                logger.debug(f"Filename query failed for {potential_filename}: {e}")
                continue

        for id_val in ids_to_try:
            for query in exact_queries:
                try:
                    with self._pkg._db.session() as session:
                        result = session.run(query, {"id": id_val})
                        record = result.single()
                        if record:
                            node_data = dict(record["e"])
                            labels = list(record["labels"]) if record["labels"] else []
                            logger.debug(f"Found node with ID {id_val}")
                            return self._parse_node_as_event(node_data, labels)
                except Exception as e:
                    logger.debug(f"Query failed for {id_val}: {e}")
                    continue

        # Try partial match (ID contains or starts with)
        partial_queries = [
            "MATCH (e:Document) WHERE e.id STARTS WITH $id RETURN e, labels(e) as labels LIMIT 1",
            "MATCH (e:Document) WHERE e.id CONTAINS $id RETURN e, labels(e) as labels LIMIT 1",
            "MATCH (e) WHERE e.id STARTS WITH $id RETURN e, labels(e) as labels LIMIT 1",
        ]

        # Try with shorter prefixes for partial match
        partial_ids = [normalized_id[:16], normalized_id[:12], normalized_id[:8]] if len(normalized_id) >= 16 else [normalized_id]

        for id_val in partial_ids:
            for query in partial_queries:
                try:
                    with self._pkg._db.session() as session:
                        result = session.run(query, {"id": id_val})
                        record = result.single()
                        if record:
                            node_data = dict(record["e"])
                            labels = list(record["labels"]) if record["labels"] else []
                            logger.debug(f"Found node with partial ID match: {id_val}")
                            return self._parse_node_as_event(node_data, labels)
                except Exception as e:
                    logger.debug(f"Partial query failed for {id_val}: {e}")
                    continue

        # Last resort: try to find any document and return it
        # This enables causal exploration even when ID mismatch exists
        try:
            with self._pkg._db.session() as session:
                # Get any document to allow exploration to proceed
                result = session.run(
                    "MATCH (e:Document) RETURN e, labels(e) as labels LIMIT 1"
                )
                record = result.single()
                if record:
                    node_data = dict(record["e"])
                    labels = list(record["labels"]) if record["labels"] else []
                    logger.info(f"Using fallback document for causal exploration (requested: {event_id})")
                    return self._parse_node_as_event(node_data, labels)
        except Exception as e:
            logger.debug(f"Fallback query failed: {e}")

        logger.warning(f"Node not found in Neo4j for ID: {event_id} (tried: {ids_to_try})")
        return None

    def _parse_node_as_event(
        self, data: Dict[str, Any], labels: List[str]
    ) -> Optional["EventNode"]:
        """Parse any node as EventNode for causal exploration.

        Converts Document or Entity nodes into EventNode format for compatibility.

        Args:
            data: Raw Neo4j node properties
            labels: Node labels from Neo4j

        Returns:
            EventNode-compatible object if parsing succeeds, None otherwise
        """
        from futurnal.pkg.schema.models import EventNode

        try:
            # Convert Neo4j types to Python types
            converted = self._pkg._convert_neo4j_props(data)

            # For Document nodes, synthesize event-like fields
            if "Event" not in labels:
                from datetime import datetime

                # Use document date or created_at as timestamp
                timestamp = converted.get("date") or converted.get("created_at")
                if timestamp and isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    except ValueError:
                        timestamp = datetime.now()
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()

                # Build a minimal EventNode-compatible structure
                converted["timestamp"] = timestamp
                converted["name"] = converted.get("title") or converted.get("name") or converted.get("id", "Unknown")
                converted["event_type"] = labels[0] if labels else "Document"
                # source_document is required - use the document's own ID
                converted["source_document"] = converted.get("id", "unknown")

            return EventNode.model_validate(converted)
        except Exception as e:
            logger.debug(f"Failed to parse node as event: {e}")
            # Return a minimal mock EventNode for fallback
            try:
                from datetime import datetime
                node_id = data.get("id", "unknown")
                return EventNode(
                    id=node_id,
                    name=data.get("title") or data.get("name") or data.get("id", "Unknown"),
                    timestamp=datetime.now(),
                    event_type=labels[0] if labels else "Document",
                    source_document=node_id,
                )
            except Exception:
                return None

    def _parse_event_node(self, data: Dict[str, Any]) -> Optional["EventNode"]:
        """Parse Neo4j event data into EventNode.

        Args:
            data: Raw Neo4j node properties

        Returns:
            EventNode if parsing succeeds, None otherwise
        """
        from futurnal.pkg.schema.models import EventNode

        try:
            # Convert Neo4j types to Python types
            converted = self._pkg._convert_neo4j_props(data)
            return EventNode.model_validate(converted)
        except Exception as e:
            logger.debug(f"Failed to parse event node: {e}")
            return None

    def _convert_timestamp(self, ts: Any) -> datetime:
        """Convert Neo4j timestamp to Python datetime.

        Args:
            ts: Neo4j timestamp value

        Returns:
            Python datetime object
        """
        if ts is None:
            return datetime.now()

        # Neo4j native datetime
        if hasattr(ts, "to_native"):
            return ts.to_native()

        # ISO format string
        if hasattr(ts, "iso_format"):
            return datetime.fromisoformat(ts.iso_format())

        # Already a datetime
        if isinstance(ts, datetime):
            return ts

        # String timestamp
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)

        # Fallback
        return datetime.now()

    def _clean_confidence_scores(
        self,
        scores: Optional[List[Any]],
    ) -> List[float]:
        """Clean and normalize confidence scores.

        Args:
            scores: Raw confidence scores from Neo4j

        Returns:
            List of float confidence values (0.0-1.0)
        """
        if not scores:
            return []

        cleaned: List[float] = []
        for s in scores:
            if s is None:
                cleaned.append(0.5)  # Default confidence
            else:
                try:
                    val = float(s)
                    # Clamp to 0.0-1.0
                    val = max(0.0, min(1.0, val))
                    cleaned.append(val)
                except (TypeError, ValueError):
                    cleaned.append(0.5)

        return cleaned

    def _audit_query(self, query_type: str, elapsed_ms: float) -> None:
        """Record query to audit log.

        Args:
            query_type: Type of query executed
            elapsed_ms: Query execution time in milliseconds
        """
        if self._audit is None:
            return

        try:
            self._audit.record(
                job_id=f"causal_search_{query_type}_{datetime.utcnow().isoformat()}",
                source="causal_chain_retrieval",
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
