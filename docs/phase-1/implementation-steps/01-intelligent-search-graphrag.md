# Step 01: Intelligent Search (ChromaDB + GraphRAG)

## Status: TODO

## Objective

Transform search from basic keyword matching to intelligent semantic search using ChromaDB embeddings and Neo4j graph traversal (GraphRAG). This is the **most user-visible change** and establishes the foundation for all intelligent features.

## Research Foundation

### Primary Papers:

#### GFM-RAG (2502.01113v1) - Graph Foundation Model for RAG
**Key Innovation**: 8M parameter graph foundation model trained on 60 KGs with 14M+ triples
**Application**: Use graph structure for multi-hop reasoning in retrieval

```
Standard RAG:          Query → Embed → Retrieve → Generate
GraphRAG (this step):  Query → Embed → Retrieve → Graph Traverse → Contextualize → Generate
```

#### Personalized Graph-Based Retrieval (2501.02157v2)
**Key Innovation**: User-centric knowledge graphs handling cold-start scenarios
**Application**: Personal knowledge graph retrieval with sparse data handling

#### LLM-Enhanced Symbolic Reasoning (2501.01246v1)
**Key Innovation**: Hybrid LLM + rule-based approach for knowledge base completion
**Application**: Combine semantic search with graph structure for better results

### Research Insight:
> "GraphRAG: When a user poses a query, the agent employs a GraphRAG methodology. Instead of only performing a vector search for relevant text chunks, it simultaneously traverses the PKG to retrieve connected entities, relationships, and contextual metadata."
> - FUTURNAL_CONCEPT.md

## Current State Analysis

### What Exists (Not Connected):
1. **ChromaDB** - Configured but not queried in search
2. **Neo4j PKG** - Running but search reads JSON files directly
3. **Embedding Service** - `src/futurnal/embeddings/service.py` exists
4. **Search API** - `src/futurnal/search/api.py` uses keyword matching

### Current Code (WRONG):
```python
# src/futurnal/search/api.py lines 593-594
text_lower = text.lower()
matches = sum(1 for term in query_terms if term in text_lower)
```

This is 1990s-style keyword search, not GraphRAG.

## Implementation Tasks

### 1. Wire ChromaDB for Semantic Search

**File**: `src/futurnal/search/api.py`

```python
# Add import
from futurnal.embeddings.service import EmbeddingService

# In HybridSearchAPI.__init__()
self.embedding_service = EmbeddingService()

# Replace keyword search with semantic search
async def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
    """Per GFM-RAG paper: Semantic retrieval using embeddings."""
    # 1. Embed query
    query_embedding = await self.embedding_service.embed_query(query)

    # 2. Query ChromaDB for similar content
    results = self.chroma_client.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,  # Over-fetch for re-ranking
    )

    return results
```

### 2. Add Neo4j Graph Traversal

**File**: `src/futurnal/search/api.py`

```python
# Per GFM-RAG: Graph context enriches retrieval
async def _graph_context(self, entity_ids: List[str], hops: int = 2) -> Dict:
    """Traverse PKG for contextual relationships."""
    query = """
    MATCH (n)-[r*1..{hops}]-(connected)
    WHERE n.id IN $entity_ids
    RETURN n, r, connected
    """.format(hops=hops)

    with self.pkg.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        return self._format_graph_context(result)
```

### 3. Implement Hybrid Retrieval Pipeline

**File**: `src/futurnal/search/api.py`

```python
async def _hybrid_search_graphrag(self, query: str, top_k: int) -> List[Dict]:
    """
    GraphRAG hybrid search per GFM-RAG paper.

    Pipeline:
    1. Semantic retrieval (ChromaDB)
    2. Graph traversal (Neo4j)
    3. Context fusion
    4. Re-ranking
    """
    # Step 1: Semantic retrieval
    semantic_results = await self._semantic_search(query, top_k)

    # Step 2: Extract entity IDs from results
    entity_ids = [r['id'] for r in semantic_results if r.get('entity_type')]

    # Step 3: Graph context enrichment
    graph_context = await self._graph_context(entity_ids)

    # Step 4: Fuse and re-rank
    fused_results = self._fuse_results(semantic_results, graph_context)

    # Step 5: Re-rank by relevance + graph centrality
    ranked_results = self._rerank_graphrag(fused_results, query)

    return ranked_results[:top_k]
```

### 4. Update Frontend Search Store

**File**: `desktop/src/stores/searchStore.ts`

The search response format changes to include graph context:

```typescript
interface EnrichedSearchResult extends SearchResult {
  graphContext: {
    relatedEntities: Entity[];
    relationships: Relationship[];
    pathToQuery: string[];
  };
}
```

### 5. Update Search Results Display

**File**: `desktop/src/components/search/SearchResults.tsx`

Show graph context alongside results:

```tsx
<ResultItem>
  <ResultContent>{result.content}</ResultContent>
  {result.graphContext && (
    <GraphContext>
      <RelatedEntities entities={result.graphContext.relatedEntities} />
      <RelationshipPath path={result.graphContext.pathToQuery} />
    </GraphContext>
  )}
</ResultItem>
```

## Success Criteria

### Functional:
- [ ] Search uses ChromaDB embeddings (not keyword matching)
- [ ] Search queries Neo4j for graph context
- [ ] Results include related entities and relationships
- [ ] Multi-hop graph traversal working (N=2 default)

### Performance:
- [ ] Search latency < 1 second
- [ ] Embedding generation < 100ms
- [ ] Graph traversal < 200ms

### Quality:
- [ ] Semantic similarity improves relevance (vs keyword)
- [ ] Graph context provides meaningful connections
- [ ] Results show "why" they're relevant (path visualization)

## Files to Modify

### Backend:
- `src/futurnal/search/api.py` - Main search implementation
- `src/futurnal/search/hybrid/retrieval.py` - Add GraphRAG retrieval
- `src/futurnal/embeddings/service.py` - Ensure query embedding works

### Frontend:
- `desktop/src/stores/searchStore.ts` - Update result types
- `desktop/src/components/search/SearchResults.tsx` - Show graph context

### Tests:
- `tests/search/test_graphrag.py` - New tests for GraphRAG

## Dependencies

- **Step 00**: Foundation audit complete
- **Infrastructure**: ChromaDB running, Neo4j running, Embeddings service working

## Next Step

After implementing intelligent search, proceed to **Step 02: LLM Answer Generation**.

## Research References

1. **GFM-RAG**: `docs/phase-1/papers/converted/2502.01113v1.md`
2. **Personalized Graph RAG**: `docs/phase-1/papers/converted/2501.02157v2.md`
3. **LLM-Enhanced Symbolic**: `docs/phase-1/papers/converted/2501.01246v1.md`
4. **SOTA Summary**: `docs/phase-1/SOTA_RESEARCH_SUMMARY.md` (Theme 3: Causal Understanding)
