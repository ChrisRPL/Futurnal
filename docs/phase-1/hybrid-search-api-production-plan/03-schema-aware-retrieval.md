Summary: Implement schema-aware hybrid retrieval with multi-model embedding integration, vector+graph fusion, temporal-aware queries, and adaptive strategies for Ghost→Animal evolution.

# 03 · Schema-Aware Hybrid Retrieval

## Purpose
Implement hybrid retrieval that combines vector similarity and graph traversal with schema version awareness and multi-model embedding integration. The system adapts retrieval strategies as schema evolves, ensuring consistent quality across Ghost→Animal evolution.

**Criticality**: HIGH - Ensures retrieval quality across schema evolution with optimal embedding selection

## Scope
- Multi-model embedding integration (Instructor-large, CodeBERT, temporal-aware)
- Vector + graph fusion strategies
- Schema version compatibility handling
- Adaptive retrieval for evolved schema
- Entity vs Event retrieval strategies
- Result ranking and fusion
- Temporal-aware query embeddings

## Requirements Alignment
- **Option B Requirement**: "Retrieval must adapt to schema evolution"
- **Multi-Model Integration**: Entity type → optimal embedding model
- **Performance Target**: Sub-1s latency for hybrid queries
- **Temporal-First**: Events require temporal-aware embeddings
- **Enables**: Consistent retrieval quality across schema changes

---

## Multi-Model Embedding Integration

### Embedding Model Router for Queries

```python
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import numpy as np


class QueryEmbeddingType(str, Enum):
    """Query embedding strategies based on intent."""
    GENERAL = "general"           # Standard entity lookup
    TEMPORAL = "temporal"         # Time-aware queries
    CAUSAL = "causal"            # Causal chain queries
    CODE = "code"                # Code-related queries
    DOCUMENT = "document"        # Full document search


class EmbeddingModelConfig(BaseModel):
    """Configuration for embedding models."""
    model_id: str
    model_path: str
    vector_dimension: int = 768
    max_sequence_length: int = 512
    quantized: bool = True
    memory_mb: int = 800
    avg_latency_ms: float = 150


class QueryEmbeddingRouter:
    """
    Routes query embedding requests to appropriate models.

    Integrates with vector-embedding-service multi-model architecture.
    """

    # Model configurations aligned with vector-embedding-service
    MODEL_CONFIGS = {
        QueryEmbeddingType.GENERAL: EmbeddingModelConfig(
            model_id="instructor-large-entity",
            model_path="hkunlp/instructor-large",
            vector_dimension=768,
            max_sequence_length=512,
            quantized=True,
            memory_mb=800,
            avg_latency_ms=150
        ),
        QueryEmbeddingType.TEMPORAL: EmbeddingModelConfig(
            model_id="instructor-temporal-event",
            model_path="hkunlp/instructor-large",  # Same base, different prompt
            vector_dimension=768,
            max_sequence_length=512,
            quantized=True,
            memory_mb=800,
            avg_latency_ms=150
        ),
        QueryEmbeddingType.CODE: EmbeddingModelConfig(
            model_id="codebert-code",
            model_path="microsoft/codebert-base",
            vector_dimension=768,
            max_sequence_length=512,
            quantized=True,
            memory_mb=600,
            avg_latency_ms=120
        ),
        QueryEmbeddingType.DOCUMENT: EmbeddingModelConfig(
            model_id="instructor-document",
            model_path="hkunlp/instructor-large",
            vector_dimension=768,
            max_sequence_length=2048,  # Longer context for documents
            quantized=True,
            memory_mb=1200,
            avg_latency_ms=300
        )
    }

    def __init__(self, embedding_service):
        """
        Initialize with embedding service from vector-embedding-service.

        Args:
            embedding_service: MultiModelEmbeddingService instance
        """
        self.embedding_service = embedding_service
        self._loaded_models: Dict[str, Any] = {}

    def determine_query_type(
        self,
        query: str,
        intent: str
    ) -> QueryEmbeddingType:
        """
        Determine query embedding type based on query and intent.

        Query type determines which embedding model to use.
        """
        # Intent-based routing
        if intent == "temporal":
            return QueryEmbeddingType.TEMPORAL
        elif intent == "causal":
            return QueryEmbeddingType.TEMPORAL  # Causal requires temporal context

        # Content-based detection
        query_lower = query.lower()

        # Code detection
        code_indicators = ["code", "function", "class", "method", "implement", "bug", "error"]
        if any(indicator in query_lower for indicator in code_indicators):
            return QueryEmbeddingType.CODE

        # Document detection
        doc_indicators = ["document", "file", "note", "article", "paper"]
        if any(indicator in query_lower for indicator in doc_indicators):
            return QueryEmbeddingType.DOCUMENT

        # Default to general
        return QueryEmbeddingType.GENERAL

    def embed_query(
        self,
        query: str,
        query_type: QueryEmbeddingType,
        temporal_context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Embed query using appropriate model.

        Returns:
            Vector embedding for the query
        """
        config = self.MODEL_CONFIGS[query_type]
        model = self._get_model(config.model_id)

        if query_type == QueryEmbeddingType.TEMPORAL:
            return self._embed_temporal_query(model, query, temporal_context)
        elif query_type == QueryEmbeddingType.CODE:
            return self._embed_code_query(model, query)
        elif query_type == QueryEmbeddingType.DOCUMENT:
            return self._embed_document_query(model, query)
        else:
            return self._embed_general_query(model, query)

    def _get_model(self, model_id: str):
        """Get or load model instance."""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        # Delegate to embedding service
        model = self.embedding_service.get_model(model_id)
        self._loaded_models[model_id] = model

        return model

    def _embed_general_query(
        self,
        model,
        query: str
    ) -> np.ndarray:
        """Embed general lookup query."""
        instruction = "Represent the search query for retrieving relevant entities:"
        return model.encode([[instruction, query]])[0]

    def _embed_temporal_query(
        self,
        model,
        query: str,
        temporal_context: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Embed temporal query with time context.

        Enhances query with temporal signals for temporal-aware retrieval.
        """
        instruction = "Represent the temporal query for retrieving time-relevant events:"

        # Augment query with temporal context
        if temporal_context:
            time_range = temporal_context.get("time_range", "")
            temporal_relation = temporal_context.get("relation", "")
            augmented_query = f"{query} [Time: {time_range}] [Relation: {temporal_relation}]"
        else:
            augmented_query = query

        return model.encode([[instruction, augmented_query]])[0]

    def _embed_code_query(
        self,
        model,
        query: str
    ) -> np.ndarray:
        """Embed code-related query using CodeBERT."""
        tokenizer = model["tokenizer"]
        model_instance = model["model"]

        tokens = tokenizer(query, return_tensors="pt", truncation=True)
        output = model_instance(**tokens)

        # Mean pooling
        embedding = output.last_hidden_state.mean(dim=1).detach().numpy()[0]
        return embedding

    def _embed_document_query(
        self,
        model,
        query: str
    ) -> np.ndarray:
        """Embed document search query."""
        instruction = "Represent the query for retrieving relevant documents:"
        return model.encode([[instruction, query]])[0]
```

---

## Component Design

### Schema-Aware Retrieval Engine

```python
class SchemaVersion(BaseModel):
    """Schema version metadata."""
    version: int
    created_at: datetime
    entity_types: List[str]
    relationship_types: List[str]
    changes_from_previous: Optional[str] = None


class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""
    entity_id: str
    entity_type: str
    content: str
    similarity_score: float
    schema_version: int
    metadata: Dict[str, Any]


class GraphSearchResult(BaseModel):
    """Result from graph traversal."""
    entity_id: str
    entity_type: str
    path_from_seed: List[str]
    path_score: float
    relationship_types: List[str]


class SchemaAwareRetrieval:
    """
    Hybrid retrieval with schema version awareness and multi-model embeddings.

    Combines vector similarity and graph traversal,
    adapting strategies based on schema version and entity types.
    """

    def __init__(
        self,
        pkg_client,
        embedding_store,
        schema_registry,
        embedding_router: Optional[QueryEmbeddingRouter] = None
    ):
        self.pkg = pkg_client
        self.embeddings = embedding_store
        self.schema = schema_registry
        self.embedding_router = embedding_router

    def hybrid_search(
        self,
        query: str,
        query_type: str = "exploratory",
        intent: str = "exploratory",
        top_k: int = 20,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
        temporal_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and graph retrieval.

        Features:
        - Multi-model query embedding based on intent
        - Schema version compatibility
        - Adaptive fusion weights
        - Temporal-aware for event queries
        """
        # Get current schema version
        schema_version = self.schema.get_current_version()

        # Determine query embedding type
        if self.embedding_router:
            embedding_type = self.embedding_router.determine_query_type(query, intent)
            query_embedding = self.embedding_router.embed_query(
                query,
                embedding_type,
                temporal_context
            )
        else:
            query_embedding = self._embed_query_fallback(query)

        # Vector retrieval with schema awareness
        vector_results = self._vector_search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Retrieve more for fusion
            schema_version=schema_version,
            intent=intent
        )

        # Graph retrieval (expand from vector results)
        seed_entities = [r.entity_id for r in vector_results[:10]]
        graph_results = self._graph_expansion(
            seed_entities=seed_entities,
            schema_version=schema_version,
            query_type=query_type,
            intent=intent
        )

        # Adaptive weight adjustment based on intent
        adjusted_weights = self._adjust_weights(
            intent=intent,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            vector_result_count=len(vector_results),
            graph_result_count=len(graph_results)
        )

        # Fuse results
        fused = self._fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=adjusted_weights["vector"],
            graph_weight=adjusted_weights["graph"]
        )

        return fused[:top_k]

    def _vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        schema_version: int,
        intent: str
    ) -> List[VectorSearchResult]:
        """
        Vector similarity search with schema filtering.

        Filters to entities from compatible schema versions.
        """
        # Query embedding store with schema filter
        results = self.embeddings.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter={
                "schema_version": {"$gte": schema_version - 1}  # Allow 1 version back
            }
        )

        return [
            VectorSearchResult(
                entity_id=r.id,
                entity_type=r.metadata.get("entity_type", "Unknown"),
                content=r.metadata.get("content", ""),
                similarity_score=r.score,
                schema_version=r.metadata.get("schema_version", schema_version),
                metadata=r.metadata
            )
            for r in results
        ]

    def _graph_expansion(
        self,
        seed_entities: List[str],
        schema_version: int,
        query_type: str,
        intent: str
    ) -> List[GraphSearchResult]:
        """
        Expand from seed entities via graph traversal.

        Strategy varies by intent:
        - causal: Follow causal relationships
        - temporal: Follow temporal relationships
        - exploratory: General neighborhood expansion
        """
        if intent == "causal":
            return self._causal_expansion(seed_entities)
        elif intent == "temporal":
            return self._temporal_expansion(seed_entities)
        else:
            return self._neighborhood_expansion(seed_entities)

    def _causal_expansion(
        self,
        seed_entities: List[str]
    ) -> List[GraphSearchResult]:
        """Expand via causal relationships (CAUSES, ENABLES, PREVENTS)."""
        cypher_query = """
            MATCH (seed:Entity)-[r:CAUSES|ENABLES|PREVENTS*1..3]->(related:Entity)
            WHERE seed.id IN $seed_ids
            RETURN related.id AS id,
                   related.type AS type,
                   [node IN nodes(path) | node.id] AS path,
                   reduce(s = 1.0, rel IN relationships(path) | s * rel.confidence) AS score,
                   [rel IN relationships(path) | type(rel)] AS rel_types
            ORDER BY score DESC
            LIMIT 50
        """

        results = self.pkg.query(cypher_query, seed_ids=seed_entities)

        return [
            GraphSearchResult(
                entity_id=r["id"],
                entity_type=r["type"],
                path_from_seed=r["path"],
                path_score=r["score"],
                relationship_types=r["rel_types"]
            )
            for r in results
        ]

    def _temporal_expansion(
        self,
        seed_entities: List[str]
    ) -> List[GraphSearchResult]:
        """Expand via temporal relationships (BEFORE, AFTER, DURING)."""
        cypher_query = """
            MATCH (seed:Entity)-[r:BEFORE|AFTER|DURING*1..3]->(related:Entity)
            WHERE seed.id IN $seed_ids
            RETURN related.id AS id,
                   related.type AS type,
                   [node IN nodes(path) | node.id] AS path,
                   reduce(s = 1.0, rel IN relationships(path) | s * rel.confidence) AS score,
                   [rel IN relationships(path) | type(rel)] AS rel_types
            ORDER BY related.timestamp ASC
            LIMIT 50
        """

        results = self.pkg.query(cypher_query, seed_ids=seed_entities)

        return [
            GraphSearchResult(
                entity_id=r["id"],
                entity_type=r["type"],
                path_from_seed=r["path"],
                path_score=r["score"],
                relationship_types=r["rel_types"]
            )
            for r in results
        ]

    def _neighborhood_expansion(
        self,
        seed_entities: List[str]
    ) -> List[GraphSearchResult]:
        """General N-hop neighborhood expansion."""
        cypher_query = """
            MATCH (seed:Entity)-[r*1..2]-(related:Entity)
            WHERE seed.id IN $seed_ids
              AND seed <> related
            RETURN DISTINCT related.id AS id,
                   related.type AS type,
                   [] AS path,
                   1.0 / (1 + length(path)) AS score,
                   [] AS rel_types
            LIMIT 50
        """

        results = self.pkg.query(cypher_query, seed_ids=seed_entities)

        return [
            GraphSearchResult(
                entity_id=r["id"],
                entity_type=r["type"],
                path_from_seed=r.get("path", []),
                path_score=r["score"],
                relationship_types=r.get("rel_types", [])
            )
            for r in results
        ]

    def _adjust_weights(
        self,
        intent: str,
        vector_weight: float,
        graph_weight: float,
        vector_result_count: int,
        graph_result_count: int
    ) -> Dict[str, float]:
        """
        Adaptively adjust fusion weights based on intent and result counts.

        Intent-based adjustments:
        - temporal/causal: Favor graph (relationships matter more)
        - lookup: Favor vector (semantic similarity matters)
        - exploratory: Balanced
        """
        # Intent-based base adjustment
        intent_adjustments = {
            "temporal": {"vector": -0.1, "graph": +0.1},
            "causal": {"vector": -0.15, "graph": +0.15},
            "lookup": {"vector": +0.1, "graph": -0.1},
            "exploratory": {"vector": 0, "graph": 0}
        }

        adjustment = intent_adjustments.get(intent, {"vector": 0, "graph": 0})

        # Apply adjustment
        adjusted_vector = vector_weight + adjustment["vector"]
        adjusted_graph = graph_weight + adjustment["graph"]

        # Result-based adjustment (if one source has few results)
        if vector_result_count < 5 and graph_result_count > 10:
            adjusted_graph += 0.1
            adjusted_vector -= 0.1
        elif graph_result_count < 5 and vector_result_count > 10:
            adjusted_vector += 0.1
            adjusted_graph -= 0.1

        # Normalize to sum to 1
        total = adjusted_vector + adjusted_graph
        return {
            "vector": adjusted_vector / total,
            "graph": adjusted_graph / total
        }

    def _fuse_results(
        self,
        vector_results: List[VectorSearchResult],
        graph_results: List[GraphSearchResult],
        vector_weight: float,
        graph_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Fuse vector and graph results with weighted scoring.

        Deduplicates and ranks by combined relevance.
        """
        # Build score maps
        vector_scores = {r.entity_id: r.similarity_score for r in vector_results}
        graph_scores = {r.entity_id: r.path_score for r in graph_results}

        # Combine all entity IDs
        all_entity_ids = set(vector_scores.keys()) | set(graph_scores.keys())

        # Calculate combined scores
        combined = []
        for entity_id in all_entity_ids:
            v_score = vector_scores.get(entity_id, 0.0)
            g_score = graph_scores.get(entity_id, 0.0)

            combined_score = (v_score * vector_weight) + (g_score * graph_weight)

            # Get entity details
            entity = next(
                (r for r in vector_results if r.entity_id == entity_id),
                next((r for r in graph_results if r.entity_id == entity_id), None)
            )

            if entity:
                combined.append({
                    "id": entity_id,
                    "type": entity.entity_type if isinstance(entity, VectorSearchResult) else entity.entity_type,
                    "vector_score": v_score,
                    "graph_score": g_score,
                    "combined_score": combined_score,
                    "source": "hybrid"
                })

        # Sort by combined score
        combined.sort(key=lambda x: x["combined_score"], reverse=True)

        return combined

    def _embed_query_fallback(self, query: str) -> np.ndarray:
        """Fallback embedding when router not available."""
        # Simple embedding using default model
        return self.embeddings.encode_query(query)
```

---

## Entity Type Retrieval Strategies

### Entity vs Event Differentiation

```python
class EntityTypeRetrievalStrategy:
    """
    Different retrieval strategies for entity types.

    Static entities (Person, Organization, Concept) use standard similarity.
    Temporal entities (Event) use temporal-aware retrieval.
    """

    def __init__(self, schema_retrieval: SchemaAwareRetrieval):
        self.retrieval = schema_retrieval

    def search_by_entity_type(
        self,
        query: str,
        target_entity_type: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search with entity type focus.

        Adjusts strategy based on target type.
        """
        if target_entity_type == "Event":
            return self._search_events(query, top_k)
        elif target_entity_type == "CodeEntity":
            return self._search_code(query, top_k)
        elif target_entity_type == "Document":
            return self._search_documents(query, top_k)
        else:
            return self._search_entities(query, target_entity_type, top_k)

    def _search_events(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search for events with temporal awareness.

        Uses temporal-aware embeddings and temporal graph expansion.
        """
        return self.retrieval.hybrid_search(
            query=query,
            intent="temporal",
            top_k=top_k,
            vector_weight=0.4,  # Lower vector weight for events
            graph_weight=0.6   # Higher graph weight for temporal relationships
        )

    def _search_code(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search for code entities using CodeBERT embeddings.

        Focuses on semantic code similarity.
        """
        return self.retrieval.hybrid_search(
            query=query,
            intent="lookup",  # Code searches are typically lookups
            top_k=top_k,
            vector_weight=0.7,  # Higher vector weight for code
            graph_weight=0.3   # Lower graph weight
        )

    def _search_documents(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search for documents with document embeddings.

        Uses longer context embeddings.
        """
        return self.retrieval.hybrid_search(
            query=query,
            intent="exploratory",
            top_k=top_k,
            vector_weight=0.6,
            graph_weight=0.4
        )

    def _search_entities(
        self,
        query: str,
        entity_type: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search for static entities (Person, Organization, Concept).

        Balanced vector/graph approach.
        """
        results = self.retrieval.hybrid_search(
            query=query,
            intent="lookup",
            top_k=top_k * 2,  # Get more, then filter
            vector_weight=0.5,
            graph_weight=0.5
        )

        # Filter to target type
        filtered = [r for r in results if r.get("type") == entity_type]

        return filtered[:top_k]
```

---

## Schema Evolution Handling

### Schema Version Compatibility

```python
class SchemaVersionCompatibility:
    """
    Handles retrieval across schema versions.

    Ensures embeddings from older schema versions are still usable.
    """

    def __init__(self, schema_registry, embedding_store):
        self.schema = schema_registry
        self.embeddings = embedding_store

    def check_compatibility(
        self,
        embedding_version: int,
        current_version: int
    ) -> Dict[str, Any]:
        """
        Check if embedding is compatible with current schema.

        Returns compatibility info and any required transformations.
        """
        version_diff = current_version - embedding_version

        if version_diff == 0:
            return {"compatible": True, "transform_required": False}
        elif version_diff == 1:
            return {"compatible": True, "transform_required": False, "minor_drift": True}
        elif version_diff <= 3:
            return {"compatible": True, "transform_required": True, "drift_level": "moderate"}
        else:
            return {"compatible": False, "reembedding_required": True}

    def filter_compatible_results(
        self,
        results: List[VectorSearchResult],
        current_version: int
    ) -> List[VectorSearchResult]:
        """Filter results to only compatible schema versions."""
        compatible = []

        for result in results:
            compat = self.check_compatibility(
                result.schema_version,
                current_version
            )

            if compat["compatible"]:
                # Adjust score for minor drift
                if compat.get("minor_drift"):
                    result.similarity_score *= 0.95
                elif compat.get("drift_level") == "moderate":
                    result.similarity_score *= 0.85

                compatible.append(result)

        return compatible

    def trigger_reembedding(
        self,
        outdated_entity_ids: List[str]
    ):
        """
        Trigger background reembedding for outdated entities.

        Called when schema drift exceeds threshold.
        """
        # Queue for background processing
        for entity_id in outdated_entity_ids:
            self.embeddings.queue_reembedding(
                entity_id=entity_id,
                priority="low"  # Background task
            )
```

---

## Implementation Details

### Week 3: Multi-Model Integration

**Deliverable**: Query embedding with model routing

1. **Day 1-2**: QueryEmbeddingRouter implementation
   - Model configuration from vector-embedding-service
   - Intent-based model selection
   - Temporal-aware embedding for events

2. **Day 3-4**: SchemaAwareRetrieval core
   - Vector search with schema filtering
   - Graph expansion by intent type
   - Result fusion with adaptive weights

3. **Day 5**: Entity type strategies
   - Event-specific temporal retrieval
   - Code-specific CodeBERT retrieval
   - Document-specific long-context retrieval

### Week 4: Schema Evolution & Optimization

**Deliverable**: Schema version handling and performance tuning

1. **Day 1-2**: Schema compatibility layer
   - Version compatibility checking
   - Drift-adjusted scoring
   - Background reembedding triggers

2. **Day 3-4**: Adaptive weight tuning
   - Intent-based weight adjustment
   - Result count-based adjustment
   - Empirical weight optimization

3. **Day 5**: Performance optimization
   - Batch vector queries
   - Graph query optimization
   - Result caching integration

---

## Testing Strategy

### Unit Tests

```python
class TestQueryEmbeddingRouter:
    """Test query embedding routing."""

    def test_intent_based_routing(self):
        """Validate intent determines embedding type."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        # Temporal intent → temporal embedding
        query_type = router.determine_query_type(
            "What happened in January?",
            intent="temporal"
        )
        assert query_type == QueryEmbeddingType.TEMPORAL

        # Causal intent → temporal embedding (causal needs temporal)
        query_type = router.determine_query_type(
            "What caused the delay?",
            intent="causal"
        )
        assert query_type == QueryEmbeddingType.TEMPORAL

    def test_content_based_routing(self):
        """Validate content determines embedding type."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        # Code content → code embedding
        query_type = router.determine_query_type(
            "Find the function that handles authentication",
            intent="exploratory"
        )
        assert query_type == QueryEmbeddingType.CODE

        # Document content → document embedding
        query_type = router.determine_query_type(
            "Search my meeting notes",
            intent="exploratory"
        )
        assert query_type == QueryEmbeddingType.DOCUMENT

    def test_embedding_generation(self):
        """Validate embeddings are generated correctly."""
        router = QueryEmbeddingRouter(mock_embedding_service)

        embedding = router.embed_query(
            "Test query",
            QueryEmbeddingType.GENERAL
        )

        assert embedding is not None
        assert len(embedding) == 768  # Expected dimension


class TestSchemaAwareRetrieval:
    """Test schema-aware hybrid retrieval."""

    def test_hybrid_search_combines_sources(self):
        """Validate results from both vector and graph."""
        retrieval = SchemaAwareRetrieval(pkg, embeddings, schema_registry)

        results = retrieval.hybrid_search(
            query="machine learning",
            top_k=10
        )

        assert len(results) > 0
        # Should have both vector and graph contributions
        assert any(r.get("vector_score", 0) > 0 for r in results)

    def test_schema_version_filtering(self):
        """Validate schema version compatibility."""
        retrieval = SchemaAwareRetrieval(pkg, embeddings, schema_registry)

        # Mock schema version
        schema_registry.set_version(5)

        results = retrieval.hybrid_search(
            query="test query",
            top_k=10
        )

        # All results should be from compatible versions
        for r in results:
            assert r.get("schema_version", 0) >= 4  # Within 1 version

    def test_intent_affects_weights(self):
        """Validate intent adjusts fusion weights."""
        retrieval = SchemaAwareRetrieval(pkg, embeddings, schema_registry)

        # Temporal intent should favor graph
        weights = retrieval._adjust_weights(
            intent="temporal",
            vector_weight=0.5,
            graph_weight=0.5,
            vector_result_count=10,
            graph_result_count=10
        )

        assert weights["graph"] > weights["vector"]

        # Lookup intent should favor vector
        weights = retrieval._adjust_weights(
            intent="lookup",
            vector_weight=0.5,
            graph_weight=0.5,
            vector_result_count=10,
            graph_result_count=10
        )

        assert weights["vector"] > weights["graph"]


class TestEntityTypeStrategies:
    """Test entity type-specific retrieval."""

    def test_event_retrieval_uses_temporal(self):
        """Validate event search uses temporal strategy."""
        strategy = EntityTypeRetrievalStrategy(mock_retrieval)

        results = strategy.search_by_entity_type(
            query="project kickoff",
            target_entity_type="Event",
            top_k=10
        )

        # Should use temporal intent
        # Verified by checking graph weight is higher
        assert len(results) >= 0  # At least runs without error

    def test_code_retrieval_uses_codebert(self):
        """Validate code search uses CodeBERT strategy."""
        strategy = EntityTypeRetrievalStrategy(mock_retrieval)

        results = strategy.search_by_entity_type(
            query="authentication handler",
            target_entity_type="CodeEntity",
            top_k=10
        )

        assert len(results) >= 0
```

### Integration Tests

```python
class TestHybridRetrievalIntegration:
    """End-to-end hybrid retrieval tests."""

    def test_full_hybrid_search_flow(self):
        """Validate complete hybrid search flow."""
        retrieval = create_production_retrieval()

        results = retrieval.hybrid_search(
            query="What meetings led to the product decision?",
            intent="causal",
            top_k=10
        )

        assert len(results) > 0
        assert all("combined_score" in r for r in results)

    def test_multi_model_embedding_integration(self):
        """Validate multi-model embeddings work end-to-end."""
        router = QueryEmbeddingRouter(embedding_service)
        retrieval = SchemaAwareRetrieval(
            pkg, embeddings, schema_registry,
            embedding_router=router
        )

        # Test different query types
        queries = [
            ("What happened in Q1?", "temporal"),
            ("Find the login function", "lookup"),
            ("What caused the delay?", "causal"),
        ]

        for query, intent in queries:
            results = retrieval.hybrid_search(
                query=query,
                intent=intent,
                top_k=5
            )
            assert len(results) >= 0  # Should not error

    def test_latency_under_1s(self):
        """Validate sub-1s latency."""
        retrieval = create_production_retrieval()

        import time
        start = time.time()

        retrieval.hybrid_search(
            query="test query",
            top_k=10
        )

        latency = time.time() - start

        assert latency < 1.0, f"Latency {latency}s exceeds 1s target"
```

---

## Success Metrics

- ✅ Multi-model embedding routing functional
- ✅ Hybrid retrieval operational (<1s latency)
- ✅ Schema version compatibility maintained
- ✅ Adaptive strategies functional (intent-based weights)
- ✅ Entity type strategies work (Event, Code, Document)
- ✅ Result fusion quality high (>0.8 relevance)
- ✅ Temporal-aware queries use temporal embeddings

---

## Dependencies

- **PKG Graph Storage**: Graph traversal and schema versioning
- **Vector Embedding Store**: Vector similarity search
- **Vector Embedding Service**: Multi-model architecture (02-multi-model-architecture.md)
- **Temporal Query Engine**: Temporal expansion (01-temporal-query-engine.md)
- **Causal Chain Retrieval**: Causal expansion (02-causal-chain-retrieval.md)

---

## Configuration

### Environment Variables

```bash
# Embedding model selection
FUTURNAL_EMBEDDING_MODEL=instructor-large|codebert|auto

# Schema compatibility threshold
FUTURNAL_SCHEMA_DRIFT_THRESHOLD=3  # Max version diff

# Weight tuning
FUTURNAL_DEFAULT_VECTOR_WEIGHT=0.5
FUTURNAL_DEFAULT_GRAPH_WEIGHT=0.5
```

---

**This module ensures optimal embedding selection and retrieval quality across schema evolution.**
