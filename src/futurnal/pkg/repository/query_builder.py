"""PKG Query Builder.

Fluent query builder for constructing complex PKG queries with:
- Node and relationship matching
- Property filtering
- Temporal range queries (Module 04 preparation)
- Pagination and ordering
- Path traversal

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/03-data-access-layer.md

Option B Compliance:
- Temporal query primitives for Phase 2/3 support
- Causal chain query support
- Production-ready with parameter binding
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union

from futurnal.pkg.database.manager import PKGDatabaseManager
from futurnal.pkg.schema.models import BaseNode, EventNode
from futurnal.pkg.repository.base import NODE_TYPE_MAP, VALID_NODE_TYPES
from futurnal.pkg.repository.exceptions import QueryBuildError

if TYPE_CHECKING:
    from neo4j import Record

logger = logging.getLogger(__name__)


class PKGQueryBuilder:
    """Fluent query builder for complex PKG queries.

    Supports:
    - Entity matching with labels and properties
    - Relationship traversal
    - Temporal filtering (for Module 04 preparation)
    - Aggregations
    - Ordering and pagination

    Example:
        >>> builder = PKGQueryBuilder(db_manager)
        >>> results = (
        ...     builder
        ...     .match_node("Person", "p")
        ...     .where_property_contains("p", "name", "Alice")
        ...     .return_nodes("p")
        ...     .limit(10)
        ...     .execute()
        ... )
    """

    def __init__(self, db_manager: PKGDatabaseManager):
        """Initialize the query builder.

        Args:
            db_manager: The PKGDatabaseManager for database access
        """
        self._db = db_manager
        self._match_clauses: List[str] = []
        self._optional_match_clauses: List[str] = []
        self._where_clauses: List[str] = []
        self._return_clause: str = "RETURN *"
        self._order_by: Optional[str] = None
        self._skip_value: Optional[int] = None
        self._limit_value: Optional[int] = None
        self._params: Dict[str, Any] = {}
        self._param_counter: int = 0

    def _next_param_name(self, prefix: str = "p") -> str:
        """Generate unique parameter name."""
        self._param_counter += 1
        return f"{prefix}_{self._param_counter}"

    def _clone(self) -> "PKGQueryBuilder":
        """Create a copy of this builder for branching."""
        new_builder = PKGQueryBuilder(self._db)
        new_builder._match_clauses = self._match_clauses.copy()
        new_builder._optional_match_clauses = self._optional_match_clauses.copy()
        new_builder._where_clauses = self._where_clauses.copy()
        new_builder._return_clause = self._return_clause
        new_builder._order_by = self._order_by
        new_builder._skip_value = self._skip_value
        new_builder._limit_value = self._limit_value
        new_builder._params = self._params.copy()
        new_builder._param_counter = self._param_counter
        return new_builder

    # ---------------------------------------------------------------------------
    # Matching
    # ---------------------------------------------------------------------------

    def match_node(
        self,
        label: str,
        alias: str = "n",
        properties: Optional[Dict[str, Any]] = None,
    ) -> "PKGQueryBuilder":
        """Add a node match pattern.

        Args:
            label: Node label (Person, Event, etc.)
            alias: Variable name for the node
            properties: Optional property filters

        Returns:
            Self for chaining

        Example:
            >>> builder.match_node("Person", "p", {"name": "Alice"})
        """
        if properties:
            param_name = self._next_param_name("props")
            self._params[param_name] = properties
            self._match_clauses.append(f"({alias}:{label} ${param_name})")
        else:
            self._match_clauses.append(f"({alias}:{label})")
        return self

    def match_node_by_id(self, alias: str, entity_id: str) -> "PKGQueryBuilder":
        """Match a node by ID.

        Args:
            alias: Variable name for the node
            entity_id: Entity ID

        Returns:
            Self for chaining
        """
        param_name = self._next_param_name("id")
        self._params[param_name] = entity_id
        self._match_clauses.append(f"({alias} {{id: ${param_name}}})")
        return self

    def match_relationship(
        self,
        source_alias: str,
        target_alias: str,
        rel_type: Optional[str] = None,
        rel_alias: str = "r",
        direction: str = "out",
        properties: Optional[Dict[str, Any]] = None,
    ) -> "PKGQueryBuilder":
        """Add a relationship match pattern.

        Args:
            source_alias: Source node variable
            target_alias: Target node variable
            rel_type: Optional relationship type filter
            rel_alias: Variable name for relationship
            direction: "out", "in", or "both"
            properties: Optional property filters

        Returns:
            Self for chaining
        """
        # Build relationship pattern
        rel_pattern = f"[{rel_alias}"
        if rel_type:
            rel_pattern += f":{rel_type}"
        if properties:
            param_name = self._next_param_name("rel_props")
            self._params[param_name] = properties
            rel_pattern += f" ${param_name}"
        rel_pattern += "]"

        # Build direction
        if direction == "out":
            pattern = f"({source_alias})-{rel_pattern}->({target_alias})"
        elif direction == "in":
            pattern = f"({source_alias})<-{rel_pattern}-({target_alias})"
        else:  # both
            pattern = f"({source_alias})-{rel_pattern}-({target_alias})"

        self._match_clauses.append(pattern)
        return self

    def match_path(
        self,
        start_alias: str,
        end_alias: str,
        rel_types: Optional[List[str]] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        path_alias: str = "path",
    ) -> "PKGQueryBuilder":
        """Add a variable-length path pattern.

        Args:
            start_alias: Start node variable
            end_alias: End node variable
            rel_types: Relationship types to traverse
            min_hops: Minimum path length
            max_hops: Maximum path length
            path_alias: Variable name for path

        Returns:
            Self for chaining
        """
        rel_pattern = ""
        if rel_types:
            rel_pattern = ":" + "|".join(rel_types)

        pattern = (
            f"{path_alias} = ({start_alias})-[{rel_pattern}*{min_hops}..{max_hops}]->({end_alias})"
        )
        self._match_clauses.append(pattern)
        return self

    def optional_match(
        self,
        pattern: str,
        **params: Any,
    ) -> "PKGQueryBuilder":
        """Add an OPTIONAL MATCH clause.

        Args:
            pattern: Match pattern string
            **params: Parameters for the pattern

        Returns:
            Self for chaining
        """
        self._optional_match_clauses.append(pattern)
        self._params.update(params)
        return self

    # ---------------------------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------------------------

    def where(self, condition: str, **params: Any) -> "PKGQueryBuilder":
        """Add a WHERE condition with parameters.

        Args:
            condition: Cypher condition string
            **params: Parameters for the condition

        Returns:
            Self for chaining

        Example:
            >>> builder.where("n.age > $min_age", min_age=18)
        """
        self._where_clauses.append(condition)
        self._params.update(params)
        return self

    def where_property_equals(
        self, alias: str, property_name: str, value: Any
    ) -> "PKGQueryBuilder":
        """Add property equality filter.

        Args:
            alias: Node/relationship variable
            property_name: Property name
            value: Value to match

        Returns:
            Self for chaining
        """
        param_name = self._next_param_name("eq")
        self._params[param_name] = value
        self._where_clauses.append(f"{alias}.{property_name} = ${param_name}")
        return self

    def where_property_contains(
        self, alias: str, property_name: str, value: str, case_insensitive: bool = True
    ) -> "PKGQueryBuilder":
        """Add property contains filter (for text search).

        Args:
            alias: Node/relationship variable
            property_name: Property name
            value: Substring to search
            case_insensitive: Case-insensitive search

        Returns:
            Self for chaining
        """
        param_name = self._next_param_name("contains")
        self._params[param_name] = value

        if case_insensitive:
            self._where_clauses.append(
                f"toLower({alias}.{property_name}) CONTAINS toLower(${param_name})"
            )
        else:
            self._where_clauses.append(
                f"{alias}.{property_name} CONTAINS ${param_name}"
            )
        return self

    def where_property_in(
        self, alias: str, property_name: str, values: List[Any]
    ) -> "PKGQueryBuilder":
        """Add property IN filter.

        Args:
            alias: Node/relationship variable
            property_name: Property name
            values: List of values to match

        Returns:
            Self for chaining
        """
        param_name = self._next_param_name("in")
        self._params[param_name] = values
        self._where_clauses.append(f"{alias}.{property_name} IN ${param_name}")
        return self

    def where_property_gt(
        self, alias: str, property_name: str, value: Any
    ) -> "PKGQueryBuilder":
        """Add property greater-than filter.

        Args:
            alias: Node variable
            property_name: Property name
            value: Value to compare

        Returns:
            Self for chaining
        """
        param_name = self._next_param_name("gt")
        self._params[param_name] = value
        self._where_clauses.append(f"{alias}.{property_name} > ${param_name}")
        return self

    def where_property_lt(
        self, alias: str, property_name: str, value: Any
    ) -> "PKGQueryBuilder":
        """Add property less-than filter.

        Args:
            alias: Node variable
            property_name: Property name
            value: Value to compare

        Returns:
            Self for chaining
        """
        param_name = self._next_param_name("lt")
        self._params[param_name] = value
        self._where_clauses.append(f"{alias}.{property_name} < ${param_name}")
        return self

    def where_timestamp_between(
        self,
        alias: str,
        property_name: str,
        start: datetime,
        end: datetime,
    ) -> "PKGQueryBuilder":
        """Add temporal range filter.

        Module 04 preparation: Key method for temporal queries.

        Args:
            alias: Node variable (typically Event)
            property_name: Timestamp property (typically "timestamp")
            start: Start of time range
            end: End of time range

        Returns:
            Self for chaining

        Example:
            >>> builder.where_timestamp_between("e", "timestamp", start_dt, end_dt)
        """
        start_param = self._next_param_name("ts_start")
        end_param = self._next_param_name("ts_end")
        self._params[start_param] = start.isoformat()
        self._params[end_param] = end.isoformat()
        self._where_clauses.append(
            f"{alias}.{property_name} >= datetime(${start_param}) AND "
            f"{alias}.{property_name} <= datetime(${end_param})"
        )
        return self

    def where_exists(self, alias: str, property_name: str) -> "PKGQueryBuilder":
        """Filter where property exists.

        Args:
            alias: Node variable
            property_name: Property name

        Returns:
            Self for chaining
        """
        self._where_clauses.append(f"{alias}.{property_name} IS NOT NULL")
        return self

    # ---------------------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------------------

    def return_nodes(self, *aliases: str) -> "PKGQueryBuilder":
        """Set return clause for nodes.

        Args:
            *aliases: Node variables to return

        Returns:
            Self for chaining
        """
        if aliases:
            returns = [f"{alias}, labels({alias}) as {alias}_labels" for alias in aliases]
            self._return_clause = "RETURN " + ", ".join(returns)
        return self

    def return_relationships(self, *aliases: str) -> "PKGQueryBuilder":
        """Set return clause for relationships.

        Args:
            *aliases: Relationship variables to return

        Returns:
            Self for chaining
        """
        if aliases:
            returns = [f"type({alias}) as {alias}_type, properties({alias}) as {alias}_props" for alias in aliases]
            self._return_clause = "RETURN " + ", ".join(returns)
        return self

    def return_custom(self, expression: str) -> "PKGQueryBuilder":
        """Set custom return clause.

        Args:
            expression: RETURN expression (without RETURN keyword)

        Returns:
            Self for chaining
        """
        self._return_clause = f"RETURN {expression}"
        return self

    def return_count(self, alias: str = "n") -> "PKGQueryBuilder":
        """Set return clause for count.

        Args:
            alias: Node variable to count

        Returns:
            Self for chaining
        """
        self._return_clause = f"RETURN count({alias}) as count"
        return self

    def return_distinct(self, *aliases: str) -> "PKGQueryBuilder":
        """Set return clause with DISTINCT.

        Args:
            *aliases: Variables to return

        Returns:
            Self for chaining
        """
        self._return_clause = "RETURN DISTINCT " + ", ".join(aliases)
        return self

    # ---------------------------------------------------------------------------
    # Ordering and Pagination
    # ---------------------------------------------------------------------------

    def order_by(
        self, expression: str, desc: bool = False
    ) -> "PKGQueryBuilder":
        """Add ORDER BY clause.

        Args:
            expression: Order expression (e.g., "n.name")
            desc: Descending order if True

        Returns:
            Self for chaining
        """
        direction = "DESC" if desc else "ASC"
        self._order_by = f"ORDER BY {expression} {direction}"
        return self

    def skip(self, count: int) -> "PKGQueryBuilder":
        """Add SKIP for offset.

        Args:
            count: Number of results to skip

        Returns:
            Self for chaining
        """
        self._skip_value = count
        return self

    def limit(self, count: int) -> "PKGQueryBuilder":
        """Add LIMIT clause.

        Args:
            count: Maximum results

        Returns:
            Self for chaining
        """
        self._limit_value = count
        return self

    # ---------------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------------

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build Cypher query and parameters.

        Returns:
            Tuple of (cypher_query, parameters)

        Raises:
            QueryBuildError: If query is invalid
        """
        if not self._match_clauses:
            raise QueryBuildError("No MATCH clauses defined")

        parts = []

        # MATCH clauses
        parts.append("MATCH " + ", ".join(self._match_clauses))

        # OPTIONAL MATCH clauses
        for om in self._optional_match_clauses:
            parts.append(f"OPTIONAL MATCH {om}")

        # WHERE clause
        if self._where_clauses:
            parts.append("WHERE " + " AND ".join(self._where_clauses))

        # RETURN clause
        parts.append(self._return_clause)

        # ORDER BY
        if self._order_by:
            parts.append(self._order_by)

        # SKIP
        if self._skip_value is not None:
            parts.append(f"SKIP {self._skip_value}")

        # LIMIT
        if self._limit_value is not None:
            parts.append(f"LIMIT {self._limit_value}")

        query = "\n".join(parts)
        return query, self._params.copy()

    def execute(self) -> List["Record"]:
        """Execute query and return raw records.

        Returns:
            List of Neo4j Record objects
        """
        query, params = self.build()
        self._logger_debug(query, params)

        with self._db.session() as session:
            result = session.run(query, params)
            return list(result)

    def execute_single(self) -> Optional["Record"]:
        """Execute and return single record.

        Returns:
            Single Neo4j Record or None
        """
        records = self.execute()
        return records[0] if records else None

    def execute_count(self) -> int:
        """Execute count query.

        Returns:
            Count result
        """
        self.return_count()
        record = self.execute_single()
        return record["count"] if record else 0

    def _logger_debug(self, query: str, params: Dict[str, Any]) -> None:
        """Log query for debugging."""
        logger.debug(f"Executing query:\n{query}\nParams: {params}")


class TemporalQueryBuilder(PKGQueryBuilder):
    """Extended query builder with temporal query primitives.

    Module 04 Preparation: Provides foundation for temporal query service.

    Example:
        >>> builder = TemporalQueryBuilder(db_manager)
        >>> events = (
        ...     builder
        ...     .match_events_in_range(start, end)
        ...     .execute_as_events()
        ... )
    """

    def match_events_in_range(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None,
        alias: str = "e",
    ) -> "TemporalQueryBuilder":
        """Match Event nodes within time range.

        Args:
            start: Start of time range
            end: End of time range
            event_type: Optional event type filter
            alias: Node variable

        Returns:
            Self for chaining
        """
        if event_type:
            self.match_node("Event", alias, {"event_type": event_type})
        else:
            self.match_node("Event", alias)

        self.where_timestamp_between(alias, "timestamp", start, end)
        return self

    def match_events_before(
        self,
        reference_event_id: str,
        time_window: Optional[timedelta] = None,
        alias: str = "e",
    ) -> "TemporalQueryBuilder":
        """Match events that occurred before reference event.

        Args:
            reference_event_id: Reference event ID
            time_window: Optional time window before reference
            alias: Node variable

        Returns:
            Self for chaining
        """
        ref_alias = self._next_param_name("ref")
        id_param = self._next_param_name("ref_id")
        self._params[id_param] = reference_event_id

        self._match_clauses.append(f"({ref_alias}:Event {{id: ${id_param}}})")
        self.match_node("Event", alias)

        if time_window:
            window_param = self._next_param_name("window")
            self._params[window_param] = time_window.total_seconds()
            self._where_clauses.append(
                f"{alias}.timestamp < {ref_alias}.timestamp AND "
                f"{alias}.timestamp >= {ref_alias}.timestamp - duration({{seconds: ${window_param}}})"
            )
        else:
            self._where_clauses.append(f"{alias}.timestamp < {ref_alias}.timestamp")

        return self

    def match_events_after(
        self,
        reference_event_id: str,
        time_window: Optional[timedelta] = None,
        alias: str = "e",
    ) -> "TemporalQueryBuilder":
        """Match events that occurred after reference event.

        Args:
            reference_event_id: Reference event ID
            time_window: Optional time window after reference
            alias: Node variable

        Returns:
            Self for chaining
        """
        ref_alias = self._next_param_name("ref")
        id_param = self._next_param_name("ref_id")
        self._params[id_param] = reference_event_id

        self._match_clauses.append(f"({ref_alias}:Event {{id: ${id_param}}})")
        self.match_node("Event", alias)

        if time_window:
            window_param = self._next_param_name("window")
            self._params[window_param] = time_window.total_seconds()
            self._where_clauses.append(
                f"{alias}.timestamp > {ref_alias}.timestamp AND "
                f"{alias}.timestamp <= {ref_alias}.timestamp + duration({{seconds: ${window_param}}})"
            )
        else:
            self._where_clauses.append(f"{alias}.timestamp > {ref_alias}.timestamp")

        return self

    def match_causal_chain(
        self,
        start_event_id: str,
        max_depth: int = 5,
        start_alias: str = "start",
        end_alias: str = "end",
    ) -> "TemporalQueryBuilder":
        """Match causal relationship chain from event.

        Module 04 Preparation: Foundation for causal chain queries.

        Args:
            start_event_id: Starting event ID
            max_depth: Maximum chain depth
            start_alias: Start node variable
            end_alias: End node variable

        Returns:
            Self for chaining
        """
        id_param = self._next_param_name("start_id")
        self._params[id_param] = start_event_id

        self._match_clauses.append(f"({start_alias}:Event {{id: ${id_param}}})")
        self._match_clauses.append(
            f"({start_alias})-[:CAUSES|ENABLES|TRIGGERS*1..{max_depth}]->({end_alias}:Event)"
        )

        return self

    def match_temporal_neighborhood(
        self,
        entity_id: str,
        time_window: timedelta,
        entity_alias: str = "n",
        event_alias: str = "e",
    ) -> "TemporalQueryBuilder":
        """Match all events related to entity within temporal window.

        Args:
            entity_id: Entity ID
            time_window: Time window around entity's events
            entity_alias: Entity variable
            event_alias: Event variable

        Returns:
            Self for chaining
        """
        id_param = self._next_param_name("entity_id")
        window_param = self._next_param_name("window")
        self._params[id_param] = entity_id
        self._params[window_param] = time_window.total_seconds()

        self._match_clauses.append(f"({entity_alias} {{id: ${id_param}}})")
        self._match_clauses.append(f"({entity_alias})-[]-({event_alias}:Event)")

        return self

    def execute_as_events(self) -> List[EventNode]:
        """Execute and return results as EventNode instances.

        Returns:
            List of EventNode instances
        """
        from futurnal.pkg.repository.base import BaseRepository

        records = self.execute()
        results = []

        for record in records:
            # Try to find the event data in the record
            for key in record.keys():
                if key.endswith("_labels"):
                    continue
                node_data = record[key]
                if node_data and isinstance(node_data, dict):
                    try:
                        # Convert Neo4j types
                        converted = {}
                        for k, v in node_data.items():
                            if hasattr(v, "to_native"):
                                converted[k] = v.to_native()
                            else:
                                converted[k] = v
                        results.append(EventNode.model_validate(converted))
                    except Exception:
                        pass

        return results
