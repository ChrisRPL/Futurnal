# Hybrid Search API Production Plan

**Status**: Ready for Implementation
**Timeline**: 6 weeks
**Dependencies**: PKG Graph Storage, Vector Embedding Service

## Overview

This production plan implements the hybrid search API with **comprehensive temporal and causal query support** required for Option B. The API must blend vector similarity and graph traversal with temporal queries, causal chain exploration, and schema-aware retrieval for Ghost→Animal AI evolution.

## Critical for Option B

The hybrid search API must support:
- ✅ **Temporal query engine**: Time range queries, temporal relationship traversal
- ✅ **Causal chain retrieval**: Find causal paths, correlation patterns
- ✅ **Schema-aware retrieval**: Adaptive strategies for evolved schema
- ✅ **Intent classification**: Route queries to appropriate strategies (temporal, causal, exploratory)
- ✅ **Phase 2/3 readiness**: Correlation detection, causal hypothesis exploration
- ✅ **Performance**: Sub-1s latency for typical queries

## Implementation Modules

### [01 · Temporal Query Engine](01-temporal-query-engine.md)
**Timeline**: Week 1-2
**Deliverables**:
- Time range query support
- Temporal relationship traversal
- Before/after/during pattern matching
- Integration with PKG temporal queries

### [02 · Causal Chain Retrieval](02-causal-chain-retrieval.md)
**Timeline**: Week 2-3
**Deliverables**:
- Causal path finding algorithms
- Correlation pattern detection for Phase 2
- Bradford Hill criteria support for Phase 3
- Multi-hop causal traversal with confidence scoring

### [03 · Schema-Aware Hybrid Retrieval](03-schema-aware-retrieval.md)
**Timeline**: Week 3-4
**Deliverables**:
- Vector + graph fusion strategies
- Schema version compatibility handling
- Adaptive retrieval for evolved schema
- Entity vs Event retrieval strategies

### [04 · Query Routing & Orchestration](04-query-routing-orchestration.md)
**Timeline**: Week 4-5
**Deliverables**:
- Intent classification (lookup, exploratory, temporal, causal)
- Multi-strategy composition and fusion
- Result ranking and assembly
- Context generation for answers

### [05 · Performance & Caching](05-performance-caching.md)
**Timeline**: Week 5
**Deliverables**:
- Caching strategies for temporal queries
- Invalidation on PKG/schema updates
- Performance optimization for <1s latency
- Query plan optimization

### [06 · Integration Testing](06-integration-testing.md)
**Timeline**: Week 6
**Deliverables**:
- End-to-end query tests
- Relevance metrics validation (MRR, precision@5)
- Performance benchmarks
- Production readiness verification

## Success Metrics

- ✅ Temporal queries functional (<1s latency)
- ✅ Causal chain retrieval operational
- ✅ Schema-aware retrieval adapts to evolution
- ✅ Intent classification >85% accuracy
- ✅ Relevance metrics meet targets (MRR >0.7, precision@5 >0.8)
- ✅ Sub-1s latency for 95% of queries
- ✅ Integration with PKG and vector store complete

## Dependencies

- PKG Graph Storage (with temporal and causal support)
- Vector Embedding Service (with temporal-aware embeddings)
- Temporal extraction pipeline
- Schema evolution system

## Next Steps

1. Begin Week 1: Temporal query engine design and implementation
2. Prioritize causal chain retrieval (critical for Phase 2/3)
3. Ensure schema-aware retrieval ready for autonomous evolution

---

**This search API is the interface layer enabling Phase 2 correlation detection and Phase 3 causal hypothesis exploration.**
