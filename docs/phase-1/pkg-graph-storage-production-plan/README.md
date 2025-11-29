# PKG Graph Storage Production Plan

**Status**: Ready for Implementation
**Timeline**: 4-5 weeks
**Dependencies**: Entity-relationship extraction pipeline

## Overview

This production plan implements the Personal Knowledge Graph (PKG) storage layer with comprehensive temporal and causal metadata support required for Option B. The PKG serves as the foundation for all three phases of Futurnal's evolution.

## Critical for Option B

The PKG must support:
- ✅ **Temporal metadata**: timestamps, durations, temporal relationships
- ✅ **Causal structure**: event entities, event-event relationships, causal candidates
- ✅ **Schema evolution**: versioned schema for autonomous evolution
- ✅ **Phase 2/3 queries**: temporal ranges, causal chains, correlation patterns

## Implementation Modules

### [01 · Graph Schema Design](01-graph-schema-design.md)
**Timeline**: Week 1
**Deliverables**:
- Node types: Person, Organization, Concept, Event, Document
- Relationship types: Standard + temporal (BEFORE/AFTER/DURING/CAUSES/etc.)
- Provenance tracking with template versioning
- Schema versioning for evolution

### [02 · Database Setup & Configuration](02-database-setup.md)
**Timeline**: Week 1-2
**Deliverables**:
- Neo4j embedded vs alternatives evaluation
- Encrypted storage configuration
- ACID semantics validation
- Performance tuning for on-device operation

### [03 · Data Access Layer](03-data-access-layer.md)
**Timeline**: Week 2-3
**Deliverables**:
- Repository pattern implementation
- Query builders for common patterns
- Batching, pagination, streaming support
- Transaction management

### [04 · Temporal Query Support](04-temporal-query-support.md)
**Timeline**: Week 3
**Deliverables**:
- Time range queries
- Temporal relationship traversal
- Causal chain queries
- Temporal neighborhood queries

### [05 · Integration & Testing](05-integration-testing.md)
**Timeline**: Week 4
**Deliverables**:
- Extraction pipeline integration
- Vector store sync hooks
- Comprehensive test suite
- Performance benchmarks

## Success Metrics

- ✅ Sub-second query latency for typical traversals
- ✅ Temporal queries operational
- ✅ Schema versioning functional
- ✅ ACID semantics preserved
- ✅ Encrypted at rest
- ✅ Integration with extraction pipeline complete

## Dependencies

- Entity-relationship extraction pipeline (provides triples)
- Privacy & audit logging (for compliance)
- Vector embedding service (for sync)

## Next Steps

1. Begin Week 1: Schema design and database evaluation
2. Prioritize temporal query support (critical for Option B)
3. Ensure schema versioning ready for autonomous evolution

---

**This is the foundational storage layer for Ghost→Animal evolution across all three phases.**
