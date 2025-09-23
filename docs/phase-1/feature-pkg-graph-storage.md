Summary: Details the PKG graph storage layer feature plan with scope, testing, and review steps.

# Feature · PKG Graph Storage Layer

## Goal
Implement the embedded graph database powering the Personal Knowledge Graph (PKG), supporting versioned triple storage, efficient traversal, and synchronization with vector indices.

## Success Criteria
- Embedded Neo4j (or equivalent) configured for on-device operation with encrypted storage.
- Schema accommodates entities, relationships, temporal metadata, provenance, and confidence scores.
- APIs expose create/update/delete operations with transactional guarantees.
- Traversal queries support hybrid search assembly (multi-hop pathfinding, neighborhood queries).
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
1. **Schema Design:** Define node/edge labels, properties, indices; align with [system-architecture.md](../architecture/system-architecture.md) guidance.
2. **Database Selection:** Evaluate Neo4j embedded vs. alternatives (Memgraph, Cayley) referencing @Web research for state-of-the-art embedded graph stores.
3. **Access Layer:** Implement repository pattern with query builders; provide abstractions for connectors and search API.
4. **Versioning Strategy:** Use temporal tables or `valid_from/valid_to` properties; ensure efficient diff queries.
5. **Sync Hooks:** Emit events on mutations for vector embedding updates and audit logs.
6. **Maintenance Tools:** CLI commands for backup, restore, and integrity checks.

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


