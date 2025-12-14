# Architecture Gap Analysis: Current vs Target Search Implementation

## Status: VERIFIED

This document details the gap between the current search implementation (keyword matching)
and the target research-based implementation (semantic + GraphRAG).

---

## Executive Summary

**Core Problem**: The main Search API (`src/futurnal/search/api.py`) uses simple keyword matching
while sophisticated semantic infrastructure exists but remains disconnected.

**Impact**: Users experience basic text search instead of intelligent semantic + graph retrieval.

**Resolution**: Step 01 will wire the existing infrastructure to the main API.

---

## Current Implementation

### Main API: `src/futurnal/search/api.py`

The `HybridSearchAPI` class implements search but bypasses all semantic infrastructure:

#### Keyword Matching (Lines 593-594)
```python
# Current implementation - KEYWORD MATCHING
text_lower = text.lower()
matches = sum(1 for term in query_terms if term in text_lower)
```

#### Problems with Current Approach:
1. **No semantic understanding** - "meeting" won't match "conference" or "discussion"
2. **No embeddings used** - Vector similarity completely ignored
3. **No graph traversal** - PKG relationships not leveraged
4. **No temporal awareness** - Beyond basic date filtering
5. **No causal reasoning** - Causal search is stubbed

---

## Target Implementation (Research-Based)

### Infrastructure That EXISTS But Is NOT Wired

| Component | Location | Status | Purpose |
|-----------|----------|--------|---------|
| `SchemaAwareRetrieval` | `search/hybrid/retrieval.py` | PRODUCTION READY | Vector + Graph hybrid search |
| `QueryEmbeddingRouter` | `search/hybrid/query_router.py` | PRODUCTION READY | Intent-based embedding routing |
| `TemporalQueryEngine` | `search/temporal/engine.py` | PRODUCTION READY | Temporal-aware queries |
| `CausalChainRetrieval` | `search/causal/retrieval.py` | PRODUCTION READY | Causal path finding |
| `ResultFusion` | `search/hybrid/fusion.py` | PRODUCTION READY | Vector + Graph result merging |
| `TemporalDecayScorer` | `search/temporal/decay.py` | PRODUCTION READY | Recency weighting |

### How Search SHOULD Work (per GFM-RAG research)

```python
# Target implementation - Step 01 will wire this
def search(query: str) -> List[SearchResult]:
    # 1. Detect query intent (temporal, causal, lookup, exploratory)
    intent = query_router.detect_intent(query)

    # 2. Generate query embedding using appropriate model
    embedding = query_router.embed_query(query, intent)

    # 3. Vector search with schema filtering
    vector_results = vector_store.search(embedding, top_k=20)

    # 4. Graph expansion based on intent
    if intent == "temporal":
        graph_results = temporal_engine.expand_neighborhood(vector_results)
    elif intent == "causal":
        graph_results = causal_retrieval.find_related(vector_results)
    else:
        graph_results = pkg_queries.get_neighbors(vector_results)

    # 5. Fuse and rank results
    return result_fusion.fuse(vector_results, graph_results, intent)
```

---

## Gap Analysis by Search Type

### 1. General Search

| Aspect | Current | Target | Gap |
|--------|---------|--------|-----|
| Query Understanding | Term splitting | Intent classification | CRITICAL |
| Matching | Keyword frequency | Semantic similarity | CRITICAL |
| Ranking | Match count | Confidence + relevance | HIGH |
| Context | None | N-hop graph expansion | HIGH |

### 2. Temporal Search

| Aspect | Current | Target | Gap |
|--------|---------|--------|-----|
| Time Parsing | Basic regex | `TemporalQuery` types | MEDIUM |
| Range Queries | Date filtering | Decay scoring + patterns | HIGH |
| Relationships | None | 7 temporal types | CRITICAL |
| Patterns | None | `TemporalPatternMatcher` | HIGH |

### 3. Causal Search

| Aspect | Current | Target | Gap |
|--------|---------|--------|-----|
| Status | **STUBBED** | Full implementation | CRITICAL |
| Path Finding | None | `find_causes`/`find_effects` | CRITICAL |
| Validation | None | 100% temporal ordering | CRITICAL |
| Confidence | None | Bradford Hill prep | HIGH |

---

## Infrastructure Verification Results

From `tests/step00/test_infrastructure_verification.py`:

```
46 passed - ALL infrastructure components verified working
```

### Verified Components:

**Embeddings Service**
- MultiModelEmbeddingService: READY
- ModelRegistry: READY (4 models, 6 entity types)
- Temporal context enforcement: WORKING

**PKG Database**
- PKGDatabaseManager: READY
- Schema constraints: READY
- Temporal queries: READY

**Search Infrastructure**
- SchemaAwareRetrieval: READY
- QueryEmbeddingRouter: READY
- TemporalQueryEngine: READY
- CausalChainRetrieval: READY
- ResultFusion: READY

**ChromaDB**
- TemporalAwareVectorWriter: READY
- SchemaVersionedEmbeddingStore: READY

---

## Integration Path (Step 01 Preview)

### Phase 1: Wire Vector Search
Replace `_general_search()` keyword matching with:
```python
from futurnal.search.hybrid.retrieval import SchemaAwareRetrieval
results = retrieval.hybrid_search(query, intent="exploratory")
```

### Phase 2: Enable Temporal Engine
Replace `_temporal_search()` basic filtering with:
```python
from futurnal.search.temporal.engine import TemporalQueryEngine
results = temporal_engine.query_time_range(start, end, query)
```

### Phase 3: Activate Causal Search
Replace `_causal_search()` stub with:
```python
from futurnal.search.causal.retrieval import CausalChainRetrieval
causes = causal_retrieval.find_causes(event_id, max_hops=3)
```

### Phase 4: Add Intent Classification
Use QueryEmbeddingRouter for automatic intent detection:
```python
from futurnal.search.hybrid.query_router import QueryEmbeddingRouter
intent, embedding = router.route_query(query)
```

---

## Research Foundation

The target implementation is grounded in:

| Paper | Contribution | Component |
|-------|-------------|-----------|
| GFM-RAG (2502.01113v1) | Graph foundation models | SchemaAwareRetrieval |
| Time-R1 (2505.13508v2) | Temporal reasoning | TemporalQueryEngine |
| CausalRAG (ACL 2025) | Causal graphs in RAG | CausalChainRetrieval |
| AutoSchemaKG (2505.23628v1) | Schema evolution | SchemaVersionedEmbeddingStore |

---

## Conclusion

**The infrastructure is ready.** Step 01 will integrate it with the main API.

Current State:
- Keyword matching in API
- Sophisticated infrastructure unused
- Causal search stubbed

After Step 01:
- Semantic + Graph hybrid search
- Intent-based query routing
- Temporal and causal capabilities active

---

## Related Documents

- [00-foundation-research-alignment.md](00-foundation-research-alignment.md) - Parent step
- [01-intelligent-search-graphrag.md](01-intelligent-search-graphrag.md) - Next step (integration)
- [Option B Principles](../../../.cursor/rules/option-b-principles.mdc) - Compliance requirements
