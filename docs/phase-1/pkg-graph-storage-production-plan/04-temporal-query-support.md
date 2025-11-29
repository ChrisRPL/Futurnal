Summary: Implement temporal query support for time range, temporal relationships, and causal chain queries.

# 04 · Temporal Query Support

## Purpose
Implement temporal query capabilities critical for Option B including time range queries, temporal relationship traversal, causal chain queries, and temporal neighborhood queries.

**Criticality**: CRITICAL - Enables Phase 2 correlation detection and Phase 3 causal inference

## Scope
- Time range queries (events within period)
- Temporal relationship traversal (BEFORE/AFTER chains)
- Causal chain queries (A→B→C causation paths)
- Temporal neighborhood (entities/events within time window)

## Temporal Query API

```python
class TemporalGraphQueries:
    """Temporal and causal query support."""

    def query_events_in_timerange(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None
    ) -> List[Event]:
        """
        Find all events within time range.

        Uses event_timestamp_index for performance.
        """
        query = """
        MATCH (e:Event)
        WHERE e.timestamp >= $start AND e.timestamp <= $end
        """
        if event_type:
            query += " AND e.event_type = $event_type"

        query += " RETURN e ORDER BY e.timestamp"

        return self.execute(query, {"start": start, "end": end, "event_type": event_type})

    def query_causal_chain(
        self,
        start_event_id: str,
        max_hops: int = 5
    ) -> List[CausalPath]:
        """
        Find causal chains starting from event.

        Returns paths: A → B → C (CAUSES relationships)
        """
        query = """
        MATCH path = (start:Event {id: $start_id})-[:CAUSES*1..$max_hops]->(end:Event)
        RETURN path, 
               [rel in relationships(path) | rel.causal_confidence] as confidences
        ORDER BY length(path) DESC
        """
        return self.execute(query, {"start_id": start_event_id, "max_hops": max_hops})

    def query_temporal_neighborhood(
        self,
        entity_id: str,
        time_window: timedelta
    ) -> Graph:
        """
        Find all entities/events related to entity within temporal window.

        Useful for Phase 2 correlation detection.
        """
        query = """
        MATCH (center {id: $entity_id})
        MATCH (center)-[r]-(neighbor)
        WHERE r.valid_from >= $start AND r.valid_from <= $end
        RETURN center, neighbor, r
        """
        # Implementation details...
```

## Testing

```python
class TestTemporalQueries:
    def test_time_range_query(self):
        """Validate time range queries work."""
        # Create events at different times
        e1 = create_event(timestamp=datetime(2024, 1, 1))
        e2 = create_event(timestamp=datetime(2024, 1, 15))
        e3 = create_event(timestamp=datetime(2024, 2, 1))

        # Query Jan 2024
        results = query_events_in_timerange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31)
        )

        assert e1 in results
        assert e2 in results
        assert e3 not in results

    def test_causal_chain_query(self):
        """Validate causal chain traversal."""
        # Create causal chain: A → B → C
        a = create_event("A")
        b = create_event("B")
        c = create_event("C")

        create_causes_relationship(a, b)
        create_causes_relationship(b, c)

        # Query chain from A
        chains = query_causal_chain(a.id)

        assert len(chains) >= 1
        assert c in flatten(chains)  # C reachable from A
```

## Success Metrics
- ✅ Time range queries <100ms for typical range
- ✅ Causal chain queries functional up to 5 hops
- ✅ Temporal neighborhood queries efficient

**Critical for Phase 2/3 - enables correlation and causal inference.**
