Summary: Defines the hybrid search API feature with scope, testing, and review checkpoints.

# Feature · Hybrid Search API

## Goal
Deliver a unified search API that blends vector similarity and graph traversal results to provide high-quality answers for Archivist users.

## Success Criteria
- API accepts natural language queries and structured filters, returning ranked results with supporting context from both PKG and embeddings.
- Latency under 1 second on reference hardware for typical queries.
- Relevance meets or exceeds baseline metrics (MRR, precision@5) against curated query set.
- Supports both synchronous responses and streaming context packages for UI consumption.
- Provides audit trail of queries and data sources consulted.

## Functional Scope
- Query parsing and intent classification to route between vector/graph weighting strategies.
- Hybrid retrieval engine combining vector search results with graph traversals (e.g., N-hop expansion, path scoring).
- Scoring pipeline applying learned weights or heuristics tuned via @Web state-of-the-art research.
- API contract for returning enriched snippets, provenance, and recommended follow-up actions.
- Caching layer for frequent queries with invalidation tied to PKG updates.

## Non-Functional Guarantees
- Offline-first operation; all inference on-device unless user opts into cloud escalation.
- Deterministic responses for identical inputs within tolerance; random seeds set for sampling steps.
- Observability metrics (latency, cache hit rate, retrieval breakdown) captured for telemetry.
- Secure handling of user queries; logs sanitized per privacy policy.

## Dependencies
- Vector embedding service ([feature-vector-embedding-service](feature-vector-embedding-service.md)).
- PKG graph storage ([feature-pkg-graph-storage](feature-pkg-graph-storage.md)).
- Privacy & audit logging foundation.

## Implementation Guide
1. **Query Schema:** Define request/response contract including filters, pagination, trace tokens.
2. **Routing Logic:** Implement automata-style state machine to choose retrieval strategy based on query intent (lookup vs. exploratory).
3. **Hybrid Retrieval Engine:** Combine vector candidates with graph expansion; consider state-of-the-art GraphRAG techniques from @Web research.
4. **Scoring & Ranking:** Apply ensemble of similarity scores, path relevance, and user context; allow configuration for personalization.
5. **Result Assembly:** Generate answer summaries using local LLM with retrieved context; attach provenance references.
6. **Caching & Invalidation:** Implement cache keyed by query signature; invalidate on relevant PKG/embedding updates.

## Testing Strategy
- **Unit Tests:** Query parsing, routing decisions, scoring functions.
- **Integration Tests:** End-to-end retrieval using fixture PKG and embedding stores; validate ranking quality.
- **Performance Tests:** Latency benchmarking under load, including cache warm/cold scenarios.
- **Regression Tests:** Ensure updates to PKG/embeddings invalidate affected cache entries.

## Code Review Checklist
- API contract documented and versioned.
- Retrieval combination logic handles edge cases (no vector hits, dense graph clusters).
- Performance targets met; caching safe and privacy-compliant.
- Tests measure relevance metrics and include golden query set.
- Logging integrates with audit trails without exposing sensitive query content.

## Documentation & Follow-up
- Publish API reference for frontend and automation clients.
- Maintain benchmark suite for relevance metrics; share results in telemetry dashboards.
- Coordinate with desktop shell to ensure UX requirements met.


