"""PKG Integration Tests - Module 05.

End-to-end integration tests validating the full PKG storage pipeline:
- Extraction → PKG storage pipeline
- Temporal extraction → temporal queries
- Vector store sync
- Performance benchmarks
- Resilience testing (crash recovery, ACID)

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md

Option B Compliance:
- All tests use real Neo4j via testcontainers (no mocks)
- Temporal metadata validated from day 1
- EventNode.timestamp required in all event tests
- Causal relationships tested with Bradford Hill structure
"""
