Summary: Implement causal chain retrieval with path finding algorithms, correlation pattern detection, and Bradford Hill criteria support for Phase 3.

# 02 · Causal Chain Retrieval

## Purpose
Implement causal chain retrieval capabilities that enable users to explore causal relationships, find causal paths, and detect correlation patterns—critical for Phase 2 Analyst and Phase 3 Guide.

**Criticality**: CRITICAL - Foundation for causal inference and hypothesis exploration

## Scope
- Causal path finding (A → B → C)
- Multi-hop causal traversal
- Correlation pattern detection for Phase 2
- Bradford Hill criteria support for Phase 3
- Causal confidence scoring

## Requirements Alignment
- **Option B Requirement**: "Search must support causal chain exploration"
- **Phase 2 Foundation**: Correlation pattern discovery
- **Phase 3 Foundation**: Causal hypothesis validation
- **Enables**: "What led to X?" and "Why did Y happen?" queries

## Component Design

### Causal Query Types

```python
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class CausalQueryType(str, Enum):
    """Types of causal queries."""
    FIND_CAUSES = "find_causes"           # What caused this event?
    FIND_EFFECTS = "find_effects"         # What did this event cause?
    CAUSAL_PATH = "causal_path"          # Path from A to B
    CAUSAL_CHAIN = "causal_chain"        # Full causal chain
    CORRELATION_PATTERN = "correlation_pattern"  # Detect correlations


class CausalPath(BaseModel):
    """Causal path between events."""
    start_event_id: str
    end_event_id: str
    path: List[str]  # Event IDs in order
    causal_confidence: float
    path_length: int
    temporal_ordering_valid: bool


class CausalChainRetrieval:
    """
    Retrieval engine for causal chains and paths.

    Integrates with PKG causal structure from entity-relationship extraction.
    """

    def __init__(self, pkg_client, temporal_engine):
        self.pkg = pkg_client
        self.temporal = temporal_engine

    def find_causes(
        self,
        event_id: str,
        max_hops: int = 3,
        min_confidence: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Find events that caused the target event.

        Example: "What led to this decision?"

        Returns causal predecessors up to max_hops away.
        """
        cypher_query = """
            MATCH path = (cause:Event)-[:CAUSES*1..{max_hops}]->(effect:Event)
            WHERE effect.id = $event_id
              AND all(r IN relationships(path) WHERE r.causal_confidence >= $min_confidence)
            RETURN cause.id AS cause_id,
                   cause.name AS cause_name,
                   cause.timestamp AS cause_timestamp,
                   length(path) AS distance,
                   [r IN relationships(path) | r.causal_confidence] AS confidence_scores
            ORDER BY distance ASC, cause.timestamp DESC
            LIMIT 20
        """.format(max_hops=max_hops)

        results = self.pkg.query(
            cypher_query,
            event_id=event_id,
            min_confidence=min_confidence
        )

        return results

    def find_effects(
        self,
        event_id: str,
        max_hops: int = 3,
        min_confidence: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Find events caused by the target event.

        Example: "What resulted from this meeting?"
        """
        cypher_query = """
            MATCH path = (cause:Event)-[:CAUSES*1..{max_hops}]->(effect:Event)
            WHERE cause.id = $event_id
              AND all(r IN relationships(path) WHERE r.causal_confidence >= $min_confidence)
            RETURN effect.id AS effect_id,
                   effect.name AS effect_name,
                   effect.timestamp AS effect_timestamp,
                   length(path) AS distance,
                   [r IN relationships(path) | r.causal_confidence] AS confidence_scores
            ORDER BY distance ASC, effect.timestamp ASC
            LIMIT 20
        """.format(max_hops=max_hops)

        return self.pkg.query(cypher_query, event_id=event_id, min_confidence=min_confidence)

    def find_causal_path(
        self,
        start_event_id: str,
        end_event_id: str,
        max_hops: int = 5
    ) -> Optional[CausalPath]:
        """
        Find causal path from start to end event.

        Example: "How did A lead to B?"

        Returns shortest causal path if exists.
        """
        cypher_query = """
            MATCH path = shortestPath((start:Event)-[:CAUSES*1..{max_hops}]->(end:Event))
            WHERE start.id = $start_id AND end.id = $end_id
            RETURN [node IN nodes(path) | node.id] AS event_ids,
                   [r IN relationships(path) | r.causal_confidence] AS confidences,
                   length(path) AS path_length
        """.format(max_hops=max_hops)

        result = self.pkg.query(cypher_query, start_id=start_event_id, end_id=end_event_id)

        if not result:
            return None

        # Compute overall confidence (min of all relationships)
        min_confidence = min(result[0]["confidences"]) if result[0]["confidences"] else 0.0

        # Validate temporal ordering
        temporal_valid = self._validate_temporal_ordering(result[0]["event_ids"])

        return CausalPath(
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            path=result[0]["event_ids"],
            causal_confidence=min_confidence,
            path_length=result[0]["path_length"],
            temporal_ordering_valid=temporal_valid
        )

    def detect_correlation_patterns(
        self,
        time_range_start: datetime,
        time_range_end: datetime,
        min_correlation_strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect correlation patterns in time range.

        Example: "Events A and B frequently occur together"

        Critical for Phase 2 automated correlation discovery.
        """
        # Use temporal correlation from temporal engine
        # Get all event types
        event_types_query = """
            MATCH (e:Event)
            WHERE e.timestamp >= $start AND e.timestamp <= $end
            RETURN DISTINCT e.event_type AS type
        """

        event_types = self.pkg.query(event_types_query, start=time_range_start, end=time_range_end)
        types = [t["type"] for t in event_types]

        # Find correlations between all pairs
        correlations = []
        for i, type_a in enumerate(types):
            for type_b in types[i+1:]:
                correlation = self.temporal.query_temporal_correlation(
                    event_type_a=type_a,
                    event_type_b=type_b,
                    max_gap=timedelta(days=30),
                    min_occurrences=3
                )

                if correlation["correlation_found"]:
                    correlations.append(correlation)

        return correlations

    def _validate_temporal_ordering(self, event_ids: List[str]) -> bool:
        """Validate events in path are temporally ordered."""
        # Query timestamps for all events
        timestamps_query = """
            MATCH (e:Event)
            WHERE e.id IN $event_ids
            RETURN e.id AS id, e.timestamp AS timestamp
        """

        results = self.pkg.query(timestamps_query, event_ids=event_ids)

        # Build timestamp map
        timestamp_map = {r["id"]: r["timestamp"] for r in results}

        # Check ordering
        for i in range(len(event_ids) - 1):
            if timestamp_map[event_ids[i]] >= timestamp_map[event_ids[i+1]]:
                return False

        return True
```

## Testing Strategy

```python
class TestCausalChainRetrieval:
    def test_find_causes(self):
        """Validate finding causal predecessors."""
        retrieval = CausalChainRetrieval(pkg_client, temporal_engine)

        causes = retrieval.find_causes(event_id="decision_123", max_hops=3)

        assert all("cause_id" in c for c in causes)

    def test_causal_path_finding(self):
        """Validate causal path finding."""
        retrieval = CausalChainRetrieval(pkg_client, temporal_engine)

        path = retrieval.find_causal_path(
            start_event_id="meeting_1",
            end_event_id="publication_1",
            max_hops=5
        )

        if path:
            assert path.temporal_ordering_valid
            assert path.causal_confidence > 0.0
```

## Success Metrics

- ✅ Causal path finding operational (<2s latency)
- ✅ Correlation pattern detection identifies patterns
- ✅ Temporal ordering validated for all paths
- ✅ Multi-hop traversal functional (up to 5 hops)
- ✅ Phase 2/3 foundation established

## Dependencies

- PKG causal structure (from entity-relationship extraction module 05)
- Temporal query engine (01-temporal-query-engine.md)
- Bradford Hill criteria metadata

**This module enables causal hypothesis exploration for Phase 3.**
