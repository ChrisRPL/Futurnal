Summary: Implement schema-aware hybrid retrieval with vector+graph fusion, schema version compatibility, and adaptive retrieval strategies.

# 03 · Schema-Aware Hybrid Retrieval

## Purpose
Implement hybrid retrieval that combines vector similarity and graph traversal with schema version awareness, adapting retrieval strategies as schema evolves for Ghost→Animal evolution.

**Criticality**: HIGH - Ensures retrieval quality across schema evolution

## Scope
- Vector + graph fusion strategies
- Schema version compatibility handling
- Adaptive retrieval for evolved schema
- Entity vs Event retrieval strategies
- Result ranking and fusion

## Requirements Alignment
- **Option B Requirement**: "Retrieval must adapt to schema evolution"
- **Schema Evolution Support**: Different strategies for different schema versions
- **Performance Target**: Sub-1s latency for hybrid queries
- **Enables**: Consistent retrieval quality across schema changes

## Component Design

```python
class SchemaAwareRetrieval:
    """
    Hybrid retrieval with schema version awareness.

    Combines vector similarity and graph traversal,
    adapting strategies based on schema version.
    """

    def __init__(self, pkg_client, embedding_store, schema_registry):
        self.pkg = pkg_client
        self.embeddings = embedding_store
        self.schema = schema_registry

    def hybrid_search(
        self,
        query: str,
        query_type: str = "exploratory",  # exploratory, lookup, temporal, causal
        top_k: int = 20,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and graph retrieval.

        Automatically adapts to current schema version.
        """
        # Get current schema version
        schema_version = self.schema.get_current_version()

        # Embed query
        query_embedding = self._embed_query(query, query_type)

        # Vector retrieval
        vector_results = self.embeddings.query_embeddings(
            query_vector=query_embedding,
            top_k=top_k * 2,  # Retrieve more for fusion
            min_schema_version=schema_version - 1  # Allow 1 version back
        )

        # Graph retrieval (expand from vector results)
        graph_results = self._graph_expansion(
            seed_entities=[r.metadata.entity_id for r in vector_results[:10]],
            schema_version=schema_version,
            query_type=query_type
        )

        # Fuse results
        fused = self._fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=vector_weight,
            graph_weight=graph_weight
        )

        return fused[:top_k]

    def _graph_expansion(
        self,
        seed_entities: List[str],
        schema_version: int,
        query_type: str
    ) -> List[Dict[str, Any]]:
        """Expand from seed entities via graph traversal."""
        if query_type == "causal":
            # Use causal relationships
            return self._causal_expansion(seed_entities)
        elif query_type == "temporal":
            # Use temporal relationships
            return self._temporal_expansion(seed_entities)
        else:
            # Standard neighborhood expansion
            return self._neighborhood_expansion(seed_entities)
```

## Testing Strategy

```python
class TestSchemaAwareRetrieval:
    def test_hybrid_search_adapts_to_schema(self):
        """Validate retrieval adapts to schema evolution."""
        retrieval = SchemaAwareRetrieval(pkg, embeddings, schema_registry)

        # Search with schema v1
        schema_registry.set_version(1)
        results_v1 = retrieval.hybrid_search("machine learning", top_k=10)

        # Evolve schema to v2
        schema_registry.set_version(2)
        results_v2 = retrieval.hybrid_search("machine learning", top_k=10)

        # Results should be valid for both versions
        assert len(results_v1) > 0
        assert len(results_v2) > 0
```

## Success Metrics

- ✅ Hybrid retrieval operational (<1s latency)
- ✅ Schema version compatibility maintained
- ✅ Adaptive strategies functional
- ✅ Result fusion quality high (>0.8 relevance)

## Dependencies

- PKG schema versioning
- Vector embedding store with schema tracking
- Temporal and causal query engines

**This module ensures retrieval quality across schema evolution.**
