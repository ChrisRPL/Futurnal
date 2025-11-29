Summary: Implement temporal query engine with time range queries, temporal relationship traversal, and temporal pattern matching.

# 01 · Temporal Query Engine

## Purpose
Implement temporal query capabilities that enable users to search across time ranges, traverse temporal relationships, and find temporal patterns—critical for Phase 2 correlation detection and Phase 3 causal inference.

**Criticality**: CRITICAL - Foundation for temporal search and correlation detection

## Scope
- Time range query support
- Temporal relationship traversal (BEFORE/AFTER/DURING/CAUSES)
- Temporal pattern matching
- Integration with PKG temporal query support
- Temporal neighborhood queries

## Requirements Alignment
- **Option B Requirement**: "Search must support temporal queries for correlation detection"
- **Phase 2 Foundation**: Temporal correlation pattern discovery
- **Phase 3 Foundation**: Temporal ordering validation for causal inference
- **Enables**: "What happened between X and Y?" queries

## Component Design

### Temporal Query Types

```python
from enum import Enum
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


class TemporalQueryType(str, Enum):
    """Types of temporal queries."""
    TIME_RANGE = "time_range"              # Events in time range
    BEFORE = "before"                       # Events before timestamp
    AFTER = "after"                         # Events after timestamp
    DURING = "during"                       # Events during another event
    TEMPORAL_NEIGHBORHOOD = "temporal_neighborhood"  # Events near timestamp
    TEMPORAL_SEQUENCE = "temporal_sequence"  # Event sequences
    TEMPORAL_CORRELATION = "temporal_correlation"    # Correlated temporal patterns


class TemporalQuery(BaseModel):
    """Temporal query specification."""
    query_type: TemporalQueryType
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    reference_timestamp: Optional[datetime] = None
    time_window: Optional[timedelta] = None
    event_types: Optional[List[str]] = None
    min_confidence: float = 0.7
```

### Temporal Query Engine

```python
class TemporalQueryEngine:
    """
    Engine for temporal queries over PKG and embeddings.

    Integrates with PKG temporal query support and vector embeddings.
    """

    def __init__(self, pkg_client, embedding_store):
        self.pkg = pkg_client
        self.embeddings = embedding_store

    def query_time_range(
        self,
        start: datetime,
        end: datetime,
        event_types: Optional[List[str]] = None,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find all events within time range.

        Example: "What happened between January and March 2024?"
        """
        # Query PKG for events in time range
        cypher_query = """
            MATCH (e:Event)
            WHERE e.timestamp >= $start
              AND e.timestamp <= $end
              AND e.extraction_confidence >= $min_confidence
        """

        if event_types:
            cypher_query += " AND e.event_type IN $event_types"

        cypher_query += """
            RETURN e.id AS id,
                   e.name AS name,
                   e.timestamp AS timestamp,
                   e.event_type AS type,
                   e.description AS description
            ORDER BY e.timestamp ASC
        """

        results = self.pkg.query(
            cypher_query,
            start=start,
            end=end,
            min_confidence=min_confidence,
            event_types=event_types
        )

        return results

    def query_before(
        self,
        reference_timestamp: datetime,
        max_results: int = 20,
        time_window: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """
        Find events before reference timestamp.

        Example: "What happened before this meeting?"
        """
        start_time = reference_timestamp - time_window if time_window else datetime.min

        return self.query_time_range(
            start=start_time,
            end=reference_timestamp,
            event_types=None
        )[:max_results]

    def query_after(
        self,
        reference_timestamp: datetime,
        max_results: int = 20,
        time_window: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """
        Find events after reference timestamp.

        Example: "What happened after this decision?"
        """
        end_time = reference_timestamp + time_window if time_window else datetime.max

        return self.query_time_range(
            start=reference_timestamp,
            end=end_time,
            event_types=None
        )[:max_results]

    def query_temporal_neighborhood(
        self,
        reference_timestamp: datetime,
        time_window: timedelta = timedelta(days=7),
        max_results: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find events in temporal neighborhood around reference.

        Example: "What was happening around this time?"

        Returns: {
            "before": [...],
            "concurrent": [...],
            "after": [...]
        }
        """
        # Events before
        before_events = self.query_time_range(
            start=reference_timestamp - time_window,
            end=reference_timestamp
        )

        # Events after
        after_events = self.query_time_range(
            start=reference_timestamp,
            end=reference_timestamp + time_window
        )

        # Concurrent events (within ±1 hour)
        concurrent_events = self.query_time_range(
            start=reference_timestamp - timedelta(hours=1),
            end=reference_timestamp + timedelta(hours=1)
        )

        return {
            "before": before_events[:max_results // 3],
            "concurrent": concurrent_events[:max_results // 3],
            "after": after_events[:max_results // 3]
        }

    def query_temporal_sequence(
        self,
        pattern: List[str],
        max_gap: timedelta = timedelta(days=30)
    ) -> List[Dict[str, Any]]:
        """
        Find sequences of events matching temporal pattern.

        Example: pattern=["Meeting", "Decision", "Publication"]
                 Finds: Meeting → Decision → Publication sequences

        Critical for Phase 2 correlation detection.
        """
        if not pattern or len(pattern) < 2:
            return []

        # Build Cypher query for sequence
        cypher_parts = []
        for i, event_type in enumerate(pattern):
            cypher_parts.append(f"(e{i}:Event {{event_type: '{event_type}'}})")

        # Add temporal ordering constraints
        relationships = []
        for i in range(len(pattern) - 1):
            relationships.append(f"e{i}.timestamp < e{i+1}.timestamp")

        # Add max gap constraint
        gap_constraints = []
        for i in range(len(pattern) - 1):
            gap_constraints.append(
                f"duration.between(e{i}.timestamp, e{i+1}.timestamp).days <= {max_gap.days}"
            )

        cypher_query = f"""
            MATCH {'-'.join(cypher_parts)}
            WHERE {' AND '.join(relationships)}
              AND {' AND '.join(gap_constraints)}
            RETURN {', '.join([f'e{i}' for i in range(len(pattern))])}
            LIMIT 50
        """

        results = self.pkg.query(cypher_query)

        return results

    def query_temporal_correlation(
        self,
        event_type_a: str,
        event_type_b: str,
        max_gap: timedelta = timedelta(days=30),
        min_occurrences: int = 3
    ) -> Dict[str, Any]:
        """
        Find temporal correlations between event types.

        Example: "Do meetings always precede decisions?"

        Returns correlation statistics for Phase 2 analysis.
        """
        cypher_query = """
            MATCH (a:Event {event_type: $type_a})-[:BEFORE]->(b:Event {event_type: $type_b})
            WHERE duration.between(a.timestamp, b.timestamp).days <= $max_gap_days
            RETURN count(*) AS co_occurrences,
                   avg(duration.between(a.timestamp, b.timestamp).days) AS avg_gap_days,
                   min(duration.between(a.timestamp, b.timestamp).days) AS min_gap_days,
                   max(duration.between(a.timestamp, b.timestamp).days) AS max_gap_days
        """

        result = self.pkg.query(
            cypher_query,
            type_a=event_type_a,
            type_b=event_type_b,
            max_gap_days=max_gap.days
        )

        if not result or result[0]["co_occurrences"] < min_occurrences:
            return {
                "correlation_found": False,
                "event_type_a": event_type_a,
                "event_type_b": event_type_b
            }

        return {
            "correlation_found": True,
            "event_type_a": event_type_a,
            "event_type_b": event_type_b,
            "co_occurrences": result[0]["co_occurrences"],
            "avg_gap_days": result[0]["avg_gap_days"],
            "min_gap_days": result[0]["min_gap_days"],
            "max_gap_days": result[0]["max_gap_days"],
            "temporal_pattern": f"{event_type_a} typically precedes {event_type_b} by {result[0]['avg_gap_days']} days"
        }
```

### Temporal Pattern Matcher

```python
class TemporalPatternMatcher:
    """
    Matches complex temporal patterns in event sequences.

    Used for correlation detection and causal hypothesis exploration.
    """

    def __init__(self, temporal_query_engine: TemporalQueryEngine):
        self.engine = temporal_query_engine

    def find_recurring_patterns(
        self,
        time_range_start: datetime,
        time_range_end: datetime,
        min_pattern_length: int = 2,
        min_occurrences: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find recurring temporal patterns in time range.

        Example: "Meeting → Decision" occurs 5 times

        Critical for Phase 2 automated correlation discovery.
        """
        # Get all events in range
        events = self.engine.query_time_range(
            start=time_range_start,
            end=time_range_end
        )

        # Extract event sequences
        sequences = self._extract_sequences(events, min_pattern_length)

        # Count pattern occurrences
        pattern_counts = {}
        for seq in sequences:
            pattern = tuple(seq)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Filter by min occurrences
        recurring = [
            {
                "pattern": list(pattern),
                "occurrences": count,
                "pattern_description": " → ".join(pattern)
            }
            for pattern, count in pattern_counts.items()
            if count >= min_occurrences
        ]

        return sorted(recurring, key=lambda x: x["occurrences"], reverse=True)

    def _extract_sequences(
        self,
        events: List[Dict[str, Any]],
        length: int
    ) -> List[List[str]]:
        """Extract event type sequences of specified length."""
        sequences = []
        for i in range(len(events) - length + 1):
            seq = [events[i + j]["type"] for j in range(length)]
            sequences.append(seq)
        return sequences
```

## Implementation Details

### Week 1: Time Range Queries

**Deliverable**: Basic temporal query support

1. Implement time range queries
2. Implement before/after queries
3. Integration with PKG temporal metadata
4. Unit tests for temporal queries

### Week 2: Advanced Temporal Patterns

**Deliverable**: Temporal pattern matching and correlation detection

1. Implement temporal neighborhood queries
2. Implement temporal sequence matching
3. Implement correlation detection
4. Recurring pattern discovery

## Testing Strategy

```python
class TestTemporalQueryEngine:
    def test_time_range_query(self):
        """Validate time range queries."""
        engine = TemporalQueryEngine(pkg_client, embedding_store)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)

        results = engine.query_time_range(start, end)

        assert all(start <= datetime.fromisoformat(r["timestamp"]) <= end for r in results)

    def test_temporal_correlation(self):
        """Validate temporal correlation detection."""
        engine = TemporalQueryEngine(pkg_client, embedding_store)

        correlation = engine.query_temporal_correlation(
            event_type_a="Meeting",
            event_type_b="Decision",
            max_gap=timedelta(days=7),
            min_occurrences=3
        )

        if correlation["correlation_found"]:
            assert "avg_gap_days" in correlation
            assert correlation["co_occurrences"] >= 3
```

## Success Metrics

- ✅ Time range queries functional with <1s latency
- ✅ Temporal pattern matching operational
- ✅ Correlation detection identifies patterns (>3 occurrences)
- ✅ Temporal neighborhood queries complete
- ✅ Integration with PKG temporal queries successful

## Dependencies

- PKG temporal query support (module 04 from PKG storage)
- Temporal extraction pipeline
- Event entities in PKG

## Next Steps

After temporal query engine complete:
1. Integrate with causal chain retrieval (02-causal-chain-retrieval.md)
2. Connect to schema-aware retrieval (03-schema-aware-retrieval.md)

**This module enables temporal search and Phase 2 correlation detection.**
