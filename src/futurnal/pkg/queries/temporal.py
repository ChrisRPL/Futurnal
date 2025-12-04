"""Temporal Graph Queries Service.

Implements temporal query capabilities critical for Option B:
- Time range queries (events within period)
- Causal chain queries (A->B->C causation paths)
- Temporal neighborhood (entities/events within time window)

API matches production plan specification:
docs/phase-1/pkg-graph-storage-production-plan/04-temporal-query-support.md

Option B Compliance:
- Enables Phase 2 correlation detection
- Enables Phase 3 causal inference
- Temporal-first design: all queries centered on temporal metadata
- Uses event_timestamp_index for performance (<100ms target)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from futurnal.pkg.database.manager import PKGDatabaseManager
from futurnal.pkg.schema.models import (
    BaseNode,
    EventNode,
    PersonNode,
    OrganizationNode,
    ConceptNode,
    DocumentNode,
)
from futurnal.pkg.queries.models import (
    CausalPath,
    CausalChainResult,
    NeighborRelationship,
    TemporalNeighborhood,
    TemporalQueryResult,
)
from futurnal.pkg.queries.exceptions import (
    CausalChainDepthError,
    EntityNotFoundError,
    EventNotFoundError,
    InvalidTimeRangeError,
    QueryTimeoutError,
    TemporalQueryError,
)

if TYPE_CHECKING:
    from futurnal.privacy.audit import AuditLogger

logger = logging.getLogger(__name__)

# Maximum allowed causal chain depth to prevent runaway queries
MAX_CAUSAL_CHAIN_DEPTH = 10

# Default query timeout in milliseconds
DEFAULT_QUERY_TIMEOUT_MS = 5000.0

# Node label to model class mapping
NODE_LABEL_MAP: Dict[str, type] = {
    "Person": PersonNode,
    "Organization": OrganizationNode,
    "Concept": ConceptNode,
    "Document": DocumentNode,
    "Event": EventNode,
}


class TemporalGraphQueries:
    """Temporal and causal query support.

    Provides high-level query methods for temporal analysis of the PKG,
    enabling Phase 2 correlation detection and Phase 3 causal inference.

    API matches production plan specification exactly.

    Example:
        >>> from futurnal.pkg import PKGDatabaseManager, TemporalGraphQueries
        >>> with PKGDatabaseManager(storage_settings) as manager:
        ...     queries = TemporalGraphQueries(manager)
        ...     events = queries.query_events_in_timerange(
        ...         start=datetime(2024, 1, 1),
        ...         end=datetime(2024, 1, 31)
        ...     )
        ...     for event in events:
        ...         print(f"{event.timestamp}: {event.name}")
    """

    def __init__(
        self,
        db_manager: PKGDatabaseManager,
        audit_logger: Optional["AuditLogger"] = None,
        query_timeout_ms: float = DEFAULT_QUERY_TIMEOUT_MS,
    ):
        """Initialize the temporal query service.

        Args:
            db_manager: PKGDatabaseManager instance for database access.
                        Must be connected before calling query methods.
            audit_logger: Optional audit logger for recording query operations.
            query_timeout_ms: Query timeout in milliseconds (default 5000ms).
        """
        self._db = db_manager
        self._audit = audit_logger
        self._timeout_ms = query_timeout_ms

    # -------------------------------------------------------------------------
    # Time Range Queries
    # -------------------------------------------------------------------------

    def query_events_in_timerange(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[EventNode]:
        """Find all events within time range.

        Uses event_timestamp_index for performance. Target: <100ms for typical ranges.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            event_type: Optional event type filter (meeting, decision, etc.)
            limit: Maximum results to return (default 1000)
            offset: Offset for pagination (default 0)

        Returns:
            List of EventNode instances ordered by timestamp ascending.

        Raises:
            InvalidTimeRangeError: If start > end
            TemporalQueryError: If query fails

        Example:
            >>> events = queries.query_events_in_timerange(
            ...     start=datetime(2024, 1, 1),
            ...     end=datetime(2024, 1, 31),
            ...     event_type="meeting"
            ... )
        """
        # Validate time range
        if start > end:
            raise InvalidTimeRangeError(start, end)

        start_time = time.perf_counter()

        # Build Cypher query - uses event_timestamp_index
        query = """
        MATCH (e:Event)
        WHERE e.timestamp >= datetime($start) AND e.timestamp <= datetime($end)
        """

        params: Dict[str, Any] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

        if event_type:
            query += " AND e.event_type = $event_type"
            params["event_type"] = event_type

        query += """
        RETURN e
        ORDER BY e.timestamp ASC
        SKIP $offset
        LIMIT $limit
        """
        params["offset"] = offset
        params["limit"] = limit

        try:
            with self._db.session() as session:
                result = session.run(query, params)
                events = []
                for record in result:
                    event_data = dict(record["e"])
                    event_node = self._parse_event_node(event_data)
                    if event_node:
                        events.append(event_node)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Time range query returned {len(events)} events in {elapsed_ms:.1f}ms"
            )
            self._audit_query("time_range", elapsed_ms, len(events))

            return events

        except Exception as e:
            logger.error(f"Time range query failed: {e}")
            raise TemporalQueryError(
                f"Time range query failed: {e}",
                query_type="time_range"
            ) from e

    def query_events_in_timerange_paginated(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> TemporalQueryResult[EventNode]:
        """Find events in time range with pagination metadata.

        Same as query_events_in_timerange but returns TemporalQueryResult
        with pagination information.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            event_type: Optional event type filter
            limit: Maximum results per page (default 100)
            offset: Offset for pagination (default 0)

        Returns:
            TemporalQueryResult with events and pagination info.
        """
        # Validate time range
        if start > end:
            raise InvalidTimeRangeError(start, end)

        start_time = time.perf_counter()

        # Build count query
        count_query = """
        MATCH (e:Event)
        WHERE e.timestamp >= datetime($start) AND e.timestamp <= datetime($end)
        """
        params: Dict[str, Any] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        if event_type:
            count_query += " AND e.event_type = $event_type"
            params["event_type"] = event_type
        count_query += " RETURN count(e) as total"

        # Get events
        events = self.query_events_in_timerange(
            start=start,
            end=end,
            event_type=event_type,
            limit=limit,
            offset=offset,
        )

        # Get total count
        with self._db.session() as session:
            count_result = session.run(count_query, params)
            total = count_result.single()["total"]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TemporalQueryResult[EventNode](
            items=events,
            total_count=total,
            offset=offset,
            limit=limit,
            query_time_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # Causal Chain Queries
    # -------------------------------------------------------------------------

    def query_causal_chain(
        self,
        start_event_id: str,
        max_hops: int = 5,
    ) -> CausalChainResult:
        """Find causal chains starting from event.

        Returns paths: A -> B -> C (CAUSES relationships).
        Includes causal confidence for each link.

        Args:
            start_event_id: ID of the starting event
            max_hops: Maximum chain depth (1-10, default 5)

        Returns:
            CausalChainResult containing all paths found.

        Raises:
            EventNotFoundError: If start event does not exist
            CausalChainDepthError: If max_hops exceeds limit
            TemporalQueryError: If query fails

        Example:
            >>> result = queries.query_causal_chain("event_123", max_hops=3)
            >>> for path in result.paths:
            ...     print(f"Chain: {' -> '.join(e.name for e in path.events)}")
            ...     print(f"Confidence: {path.aggregate_confidence:.2f}")
        """
        # Validate depth
        if max_hops < 1:
            max_hops = 1
        if max_hops > MAX_CAUSAL_CHAIN_DEPTH:
            raise CausalChainDepthError(max_hops, MAX_CAUSAL_CHAIN_DEPTH)

        start_time = time.perf_counter()

        # First verify start event exists
        start_event = self._get_event_by_id(start_event_id)
        if not start_event:
            raise EventNotFoundError(start_event_id)

        # Build path query for causal chains
        # Uses CAUSES|ENABLES|TRIGGERS relationship types
        # Note: max_hops must be interpolated directly into query (Neo4j limitation)
        query = f"""
        MATCH path = (start:Event {{id: $start_id}})-[:CAUSES|ENABLES|TRIGGERS*1..{max_hops}]->(end:Event)
        RETURN path,
               [rel in relationships(path) | rel.causal_confidence] as confidences,
               [rel in relationships(path) | rel.causal_evidence] as evidence,
               length(path) as depth
        ORDER BY depth DESC, reduce(acc = 1.0, c IN [rel in relationships(path) | rel.causal_confidence] | acc * c) DESC
        """

        params = {
            "start_id": start_event_id,
        }

        try:
            with self._db.session() as session:
                result = session.run(query, params)
                paths: List[CausalPath] = []

                for record in result:
                    path = record["path"]
                    confidences = record["confidences"]
                    evidence = record["evidence"]

                    # Parse nodes from path
                    events = []
                    for node in path.nodes:
                        event_data = dict(node)
                        event_node = self._parse_event_node(event_data)
                        if event_node:
                            events.append(event_node)

                    if len(events) >= 2:
                        # Clean up confidence values (handle None)
                        clean_confidences = [
                            float(c) if c is not None else 0.5
                            for c in confidences
                        ]
                        # Clean up evidence (handle None)
                        clean_evidence = [
                            str(e) if e is not None else ""
                            for e in evidence
                        ]

                        paths.append(CausalPath(
                            events=events,
                            confidences=clean_confidences,
                            causal_evidence=clean_evidence,
                        ))

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Causal chain query found {len(paths)} paths in {elapsed_ms:.1f}ms"
            )
            self._audit_query("causal_chain", elapsed_ms, len(paths))

            return CausalChainResult(
                paths=paths,
                start_event_id=start_event_id,
                start_event=start_event,
                max_hops_requested=max_hops,
                query_time_ms=elapsed_ms,
            )

        except EventNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Causal chain query failed: {e}")
            raise TemporalQueryError(
                f"Causal chain query failed: {e}",
                query_type="causal_chain"
            ) from e

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
        """Find all entities/events related to entity within temporal window.

        Useful for Phase 2 correlation detection - finds the temporal context
        around a specific entity.

        Args:
            entity_id: ID of the center entity
            time_window: Time window for temporal filtering
            include_events: Include Event neighbors (default True)
            include_entities: Include non-Event neighbors (default True)

        Returns:
            TemporalNeighborhood containing center and related entities/events.

        Raises:
            EntityNotFoundError: If entity does not exist
            TemporalQueryError: If query fails

        Example:
            >>> neighborhood = queries.query_temporal_neighborhood(
            ...     entity_id="person_123",
            ...     time_window=timedelta(days=30)
            ... )
            >>> print(f"Found {neighborhood.total_neighbors} related nodes")
            >>> for event in neighborhood.event_neighbors:
            ...     print(f"  Event: {event.name}")
        """
        start_time = time.perf_counter()

        # First get the center entity to determine reference time
        center_entity = self._get_entity_by_id(entity_id)
        if not center_entity:
            raise EntityNotFoundError(entity_id)

        # Determine time bounds based on entity type
        if isinstance(center_entity, EventNode):
            # For events, use the event's timestamp as center
            reference_time = self._strip_timezone(center_entity.timestamp)
            time_start = reference_time - time_window
            time_end = reference_time + time_window
        else:
            # For non-event entities, find the most recent relationship activity
            reference_time = self._get_entity_latest_activity(entity_id)
            if reference_time is None:
                # Fallback to current time if no activity found
                reference_time = datetime.utcnow()
            else:
                reference_time = self._strip_timezone(reference_time)
            time_start = reference_time - time_window
            time_end = reference_time + time_window

        # Build query for neighbors with temporal relationships
        query = """
        MATCH (center {id: $entity_id})-[r]-(neighbor)
        WHERE (
            // Relationships with valid_from in window
            (r.valid_from IS NOT NULL AND
             r.valid_from >= datetime($time_start) AND
             r.valid_from <= datetime($time_end))
            OR
            // Relationships without valid_from (always included)
            r.valid_from IS NULL
            OR
            // Event neighbors with timestamp in window
            (neighbor:Event AND
             neighbor.timestamp >= datetime($time_start) AND
             neighbor.timestamp <= datetime($time_end))
        )
        RETURN neighbor, labels(neighbor) as labels, r, type(r) as rel_type,
               startNode(r) = center as is_outgoing
        """

        params = {
            "entity_id": entity_id,
            "time_start": time_start.isoformat(),
            "time_end": time_end.isoformat(),
        }

        try:
            with self._db.session() as session:
                result = session.run(query, params)

                neighbors: List[BaseNode] = []
                relationships: List[NeighborRelationship] = []
                seen_ids: set = set()

                for record in result:
                    neighbor_data = dict(record["neighbor"])
                    labels = record["labels"]
                    rel_data = dict(record["r"])
                    rel_type = record["rel_type"]
                    is_outgoing = record["is_outgoing"]

                    # Parse neighbor node
                    neighbor_node = self._parse_node_by_labels(neighbor_data, labels)
                    if neighbor_node is None:
                        continue

                    # Filter by include flags
                    is_event = isinstance(neighbor_node, EventNode)
                    if is_event and not include_events:
                        continue
                    if not is_event and not include_entities:
                        continue

                    # Avoid duplicates (same neighbor via multiple relationships)
                    if neighbor_node.id not in seen_ids:
                        neighbors.append(neighbor_node)
                        seen_ids.add(neighbor_node.id)

                    # Parse relationship
                    relationships.append(NeighborRelationship(
                        relationship_type=rel_type,
                        direction="outgoing" if is_outgoing else "incoming",
                        neighbor_id=neighbor_node.id,
                        properties=self._convert_neo4j_props(rel_data),
                    ))

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Temporal neighborhood query found {len(neighbors)} neighbors "
                f"in {elapsed_ms:.1f}ms"
            )
            self._audit_query("temporal_neighborhood", elapsed_ms, len(neighbors))

            return TemporalNeighborhood(
                center_id=entity_id,
                center_entity=center_entity,
                neighbors=neighbors,
                relationships=relationships,
                time_window=time_window,
                time_bounds=(time_start, time_end),
                query_time_ms=elapsed_ms,
            )

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Temporal neighborhood query failed: {e}")
            raise TemporalQueryError(
                f"Temporal neighborhood query failed: {e}",
                query_type="temporal_neighborhood"
            ) from e

    # -------------------------------------------------------------------------
    # Additional Temporal Queries
    # -------------------------------------------------------------------------

    def query_events_before(
        self,
        reference_event_id: str,
        time_window: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[EventNode]:
        """Find events that occurred before reference event.

        Args:
            reference_event_id: ID of the reference event
            time_window: Optional max time before reference
            limit: Maximum results (default 100)

        Returns:
            List of events before reference, ordered by timestamp descending.
        """
        ref_event = self._get_event_by_id(reference_event_id)
        if not ref_event:
            raise EventNotFoundError(reference_event_id)

        end_time = ref_event.timestamp
        start_time = datetime.min
        if time_window:
            start_time = end_time - time_window

        events = self.query_events_in_timerange(
            start=start_time,
            end=end_time,
            limit=limit + 1,  # +1 to exclude reference
        )

        # Exclude the reference event itself and reverse order
        return [e for e in events if e.id != reference_event_id][::-1][:limit]

    def query_events_after(
        self,
        reference_event_id: str,
        time_window: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[EventNode]:
        """Find events that occurred after reference event.

        Args:
            reference_event_id: ID of the reference event
            time_window: Optional max time after reference
            limit: Maximum results (default 100)

        Returns:
            List of events after reference, ordered by timestamp ascending.
        """
        ref_event = self._get_event_by_id(reference_event_id)
        if not ref_event:
            raise EventNotFoundError(reference_event_id)

        start_time = ref_event.timestamp
        end_time = datetime.max
        if time_window:
            end_time = start_time + time_window

        events = self.query_events_in_timerange(
            start=start_time,
            end=end_time,
            limit=limit + 1,  # +1 to exclude reference
        )

        # Exclude the reference event itself
        return [e for e in events if e.id != reference_event_id][:limit]

    def query_simultaneous_events(
        self,
        reference_event_id: str,
        tolerance: timedelta = timedelta(hours=1),
        limit: int = 100,
    ) -> List[EventNode]:
        """Find events that occurred at approximately the same time.

        Args:
            reference_event_id: ID of the reference event
            tolerance: How close in time to be considered simultaneous
            limit: Maximum results (default 100)

        Returns:
            List of events within tolerance window.
        """
        ref_event = self._get_event_by_id(reference_event_id)
        if not ref_event:
            raise EventNotFoundError(reference_event_id)

        events = self.query_events_in_timerange(
            start=ref_event.timestamp - tolerance,
            end=ref_event.timestamp + tolerance,
            limit=limit + 1,
        )

        # Exclude the reference event itself
        return [e for e in events if e.id != reference_event_id][:limit]

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _get_event_by_id(self, event_id: str) -> Optional[EventNode]:
        """Retrieve a single event by ID."""
        query = "MATCH (e:Event {id: $id}) RETURN e"
        with self._db.session() as session:
            result = session.run(query, {"id": event_id})
            record = result.single()
            if record:
                return self._parse_event_node(dict(record["e"]))
        return None

    def _get_entity_by_id(self, entity_id: str) -> Optional[BaseNode]:
        """Retrieve any entity by ID."""
        query = "MATCH (n {id: $id}) RETURN n, labels(n) as labels"
        with self._db.session() as session:
            result = session.run(query, {"id": entity_id})
            record = result.single()
            if record:
                return self._parse_node_by_labels(
                    dict(record["n"]),
                    record["labels"]
                )
        return None

    def _strip_timezone(self, dt: datetime) -> datetime:
        """Remove timezone info from datetime for consistent comparisons."""
        if dt is None:
            return None
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    def _get_entity_latest_activity(self, entity_id: str) -> Optional[datetime]:
        """Find the most recent relationship timestamp for an entity.

        Looks at:
        - valid_from on relationships
        - timestamp on connected Event nodes

        Returns the most recent timestamp found, or None if no activity.
        """
        query = """
        MATCH (n {id: $id})-[r]-(neighbor)
        WITH
            CASE WHEN r.valid_from IS NOT NULL THEN r.valid_from ELSE null END as rel_time,
            CASE WHEN neighbor:Event AND neighbor.timestamp IS NOT NULL THEN neighbor.timestamp ELSE null END as event_time
        WITH coalesce(rel_time, event_time) as activity_time
        WHERE activity_time IS NOT NULL
        RETURN max(activity_time) as latest
        """
        with self._db.session() as session:
            result = session.run(query, {"id": entity_id})
            record = result.single()
            if record and record["latest"]:
                latest = record["latest"]
                if hasattr(latest, "to_native"):
                    return latest.to_native()
                elif hasattr(latest, "iso_format"):
                    return datetime.fromisoformat(latest.iso_format())
                return latest
        return None

    def _parse_event_node(self, data: Dict[str, Any]) -> Optional[EventNode]:
        """Parse Neo4j node data into EventNode model."""
        try:
            converted = self._convert_neo4j_props(data)
            return EventNode.model_validate(converted)
        except Exception as e:
            logger.debug(f"Failed to parse event node: {e}")
            return None

    def _parse_node_by_labels(
        self,
        data: Dict[str, Any],
        labels: List[str],
    ) -> Optional[BaseNode]:
        """Parse Neo4j node data into appropriate model based on labels."""
        converted = self._convert_neo4j_props(data)

        # Try each known label in order of specificity
        for label in labels:
            if label in NODE_LABEL_MAP:
                try:
                    model_class = NODE_LABEL_MAP[label]
                    return model_class.model_validate(converted)
                except Exception as e:
                    logger.debug(f"Failed to parse {label} node: {e}")

        # Fallback to BaseNode
        try:
            return BaseNode.model_validate(converted)
        except Exception as e:
            logger.debug(f"Failed to parse base node: {e}")
            return None

    def _convert_neo4j_props(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Neo4j native types to Python types."""
        converted = {}
        for key, value in data.items():
            if value is None:
                converted[key] = value
            elif hasattr(value, "months") and hasattr(value, "seconds"):
                # Neo4j Duration type - convert to timedelta
                # Note: Neo4j Duration can have months which timedelta can't represent exactly
                # We approximate months as 30 days
                days = getattr(value, "days", 0)
                months = getattr(value, "months", 0)
                seconds = getattr(value, "seconds", 0)
                nanoseconds = getattr(value, "nanoseconds", 0)
                total_days = days + (months * 30)
                total_seconds = seconds + (nanoseconds / 1_000_000_000)
                converted[key] = timedelta(days=total_days, seconds=total_seconds)
            elif hasattr(value, "to_native"):
                # Neo4j datetime etc.
                native = value.to_native()
                # Check if to_native returned a dict (some Duration implementations)
                if isinstance(native, dict) and "seconds" in native:
                    days = native.get("days", 0)
                    months = native.get("months", 0)
                    seconds = native.get("seconds", 0)
                    nanoseconds = native.get("nanoseconds", 0)
                    total_days = days + (months * 30)
                    total_seconds = seconds + (nanoseconds / 1_000_000_000)
                    converted[key] = timedelta(days=total_days, seconds=total_seconds)
                else:
                    converted[key] = native
            elif hasattr(value, "iso_format"):
                # Neo4j datetime without to_native
                converted[key] = datetime.fromisoformat(value.iso_format())
            else:
                converted[key] = value
        return converted

    def _audit_query(
        self,
        query_type: str,
        elapsed_ms: float,
        result_count: int,
    ) -> None:
        """Record query metrics to audit log."""
        if self._audit is None:
            return

        try:
            self._audit.record(
                job_id=f"temporal_query_{query_type}_{datetime.utcnow().isoformat()}",
                source="temporal_graph_queries",
                action=f"query_{query_type}",
                status="completed",
                timestamp=datetime.utcnow(),
                metadata={
                    "query_type": query_type,
                    "elapsed_ms": elapsed_ms,
                    "result_count": result_count,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to record audit event: {e}")
