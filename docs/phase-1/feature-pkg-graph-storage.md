Summary: Details the PKG graph storage layer feature plan with scope, testing, and review steps.

# Feature · PKG Graph Storage Layer

## Goal
Implement the embedded graph database powering the Personal Knowledge Graph (PKG), supporting versioned triple storage with **comprehensive temporal and causal metadata**, efficient traversal for temporal queries and causal chain exploration, and synchronization with vector indices. This storage layer must support **Option B requirements** for Ghost→Animal evolution across all three phases.

## Success Criteria
- Embedded Neo4j (or equivalent) configured for on-device operation with encrypted storage.
- Schema accommodates:
  - **Entities**: Person, Organization, Concept, Event (with temporal grounding)
  - **Relationships**: Standard relationships + temporal types (BEFORE/AFTER/DURING/CAUSES/ENABLES/etc.)
  - **Temporal Metadata**: timestamps, durations, temporal_type, valid_from, valid_to
  - **Causal Structure**: Event entities, event-event relationships, causal candidates for Phase 3
  - **Provenance**: source chunks, extraction confidence, temporal source tracking
  - **Schema Versioning**: Track schema evolution for autonomous schema updates
- APIs expose create/update/delete operations with transactional guarantees.
- **Temporal Query Support**: Query by time ranges, temporal relationships, causal chains
- Traversal queries support hybrid search assembly (multi-hop pathfinding, neighborhood queries, temporal traversals).
- Graph updates trigger vector store sync events.

## Functional Scope
- Graph schema definition aligned with extraction outputs.
- Data access layer with batching, pagination, and streaming read support.
- Versioning mechanism storing history for diffing and rollback.
- Backup & restore workflows integrated with reliability requirements.
- Health checks and compaction routines.

## Non-Functional Guarantees
- Encrypted at rest using OS keychain for key management.
- Sub-second query latency for typical graph traversals on reference hardware.
- Resilient to abrupt shutdowns; ACID semantics preserved.
- Logging omits sensitive data, focuses on operation metrics.

## Dependencies
- Entity & relationship extraction pipeline ([feature-entity-relationship-extraction](feature-entity-relationship-extraction.md)).
- Privacy & audit logging infrastructure.
- Vector embedding service for synchronization.

## Implementation Guide

### 1. Schema Design (Option B Extensions)

**Core Node Types**:
```cypher
// Static Entities
CREATE (p:Person {
  id: string,
  name: string,
  aliases: [string],
  created_at: datetime,
  updated_at: datetime
})

CREATE (o:Organization {
  id: string,
  name: string,
  type: string,
  created_at: datetime
})

CREATE (c:Concept {
  id: string,
  name: string,
  description: string,
  category: string
})

// Event Entities (NEW - Option B)
CREATE (e:Event {
  id: string,
  name: string,
  event_type: string,
  timestamp: datetime,      // When did this occur?
  duration: duration,       // How long?
  description: string,
  created_at: datetime
})

// Document source tracking
CREATE (d:Document {
  id: string,
  source_id: string,
  source_type: string,
  content_hash: string,
  created_at: datetime
})
```

**Temporal Relationship Types** (NEW - Option B):
```cypher
// Standard relationships
CREATE (p)-[:WORKS_AT {
  valid_from: datetime,
  valid_to: datetime,
  confidence: float,
  provenance: string
}]->(o)

// Temporal relationships
CREATE (e1:Event)-[:BEFORE {
  temporal_confidence: float,
  temporal_source: string
}]->(e2:Event)

CREATE (e1:Event)-[:CAUSES {
  causal_confidence: float,
  causal_evidence: string,
  is_causal_candidate: boolean  // For Phase 3 validation
}]->(e2:Event)

CREATE (e:Event)-[:DURING {
  overlap_start: datetime,
  overlap_end: datetime
}]->(e2:Event)

// Other temporal types: AFTER, ENABLES, PREVENTS, TRIGGERS, SIMULTANEOUS
```

**Provenance Tracking** (Enhanced):
```cypher
CREATE (t:Triple {
  id: string,
  subject_id: string,
  predicate: string,
  object_id: string,
  extraction_method: string,  // "explicit", "inferred", "llm"
  model_version: string,
  template_version: string,   // NEW - thought template tracking
  confidence: float,
  temporal_confidence: float,  // NEW - temporal-specific confidence
  created_at: datetime
})-[:EXTRACTED_FROM]->(c:Chunk {
  id: string,
  document_id: string,
  content_hash: string,
  position: int
})
```

**Schema Versioning** (NEW - Option B):
```cypher
CREATE (sv:SchemaVersion {
  id: string,
  version: int,
  created_at: datetime,
  changes: string,  // JSON of schema changes
  reflection_quality: float,  // Quality metrics that triggered evolution
  entity_types: [string],
  relationship_types: [string]
})
```

### 2. Database Selection
Evaluate Neo4j embedded vs. alternatives (Memgraph, Cayley) referencing @Web research for state-of-the-art embedded graph stores with strong temporal query support.

### 3. Access Layer Extensions

**Temporal Query API** (NEW):
```python
class TemporalGraphQueries:
    """Temporal and causal query support."""

    def query_events_in_timerange(
        self,
        start: datetime,
        end: datetime,
        event_type: Optional[str] = None
    ) -> List[Event]:
        """Find all events within time range."""

    def query_causal_chain(
        self,
        start_event: str,
        max_hops: int = 5
    ) -> List[CausalPath]:
        """
        Find causal chains starting from event.
        Returns paths: A → B → C (CAUSES relationships)
        """

    def query_temporal_neighborhood(
        self,
        entity: str,
        time_window: timedelta
    ) -> Graph:
        """
        Find all entities/events related to entity
        within temporal window.
        """
```

### 4. Versioning Strategy
Use `valid_from/valid_to` properties for temporal validity; use SchemaVersion nodes for schema evolution tracking; ensure efficient diff queries for schema migration.

### 5. Sync Hooks
Emit events on mutations for vector embedding updates, audit logs, and schema evolution triggers.

### 6. Maintenance Tools
CLI commands for backup, restore, integrity checks, schema migration, and temporal consistency validation.

## Testing Strategy
- **Unit Tests:** Schema validation, transaction boundaries, versioning logic.
- **Integration Tests:** Write/read cycles with extraction outputs; compaction routines.
- **Performance Tests:** Benchmark traversal latency, bulk insert throughput.
- **Resilience Tests:** Simulate crashes to verify ACID recovery and backup integrity.

## Code Review Checklist
- Schema documented and aligned with upstream/downstream needs.
- Transactions atomic; rollback works under failure scenarios.
- Sync hooks reliable and idempotent.
- Security features (encryption, key management) implemented correctly.
- Tests cover backup/restore and high-load scenarios.

## Documentation & Follow-up
- Publish schema reference diagrams for engineering teams.
- Update operational runbooks for maintenance tasks.
- Coordinate with vector service to ensure index coherence.


