# Search API Reference

`HybridSearchAPI` - Unified search across your personal knowledge graph.

## Overview

The HybridSearchAPI combines multiple search capabilities:
- Semantic vector search (ChromaDB)
- Graph-based retrieval (Neo4j)
- Temporal query engine
- Causal chain discovery
- Multi-modal support (voice, OCR)

## Quick Example

```python
from futurnal.search.api import create_hybrid_search_api

async def main():
    # Create API instance
    api = await create_hybrid_search_api()

    # Simple search
    results = await api.search("machine learning", top_k=10)

    # Temporal search
    results = await api.search(
        "what was I working on",
        temporal_range=("2024-12-01", "2024-12-15")
    )

    # Close when done
    await api.close()
```

## Factory Function

### `create_hybrid_search_api`

```python
async def create_hybrid_search_api(
    config: Optional[SearchConfig] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> HybridSearchAPI
```

Creates and initializes a HybridSearchAPI instance.

**Parameters**:
- `config`: Optional search configuration
- `audit_logger`: Optional audit logger for privacy compliance

**Returns**: Initialized `HybridSearchAPI` instance

**Example**:
```python
from futurnal.search.config import SearchConfig

config = SearchConfig(
    vector_weight=0.6,
    graph_weight=0.4,
    cache_enabled=True,
)

api = await create_hybrid_search_api(config=config)
```

## HybridSearchAPI Class

### Constructor

```python
class HybridSearchAPI:
    def __init__(
        self,
        config: SearchConfig,
        pkg_client: PKGClient,
        embedding_service: MultiModelEmbeddingService,
        answer_generator: AnswerGenerator,
        audit_logger: Optional[AuditLogger] = None,
    )
```

**Note**: Use `create_hybrid_search_api()` factory function instead of direct instantiation.

### Methods

#### `search`

```python
async def search(
    self,
    query: str,
    *,
    top_k: int = 10,
    temporal_range: Optional[Tuple[str, str]] = None,
    source_filter: Optional[List[str]] = None,
    entity_types: Optional[List[str]] = None,
    include_causal: bool = False,
) -> List[SearchResult]
```

Execute a hybrid search across the knowledge graph.

**Parameters**:
- `query`: Search query (natural language)
- `top_k`: Maximum results to return (default: 10)
- `temporal_range`: Optional date range tuple ("YYYY-MM-DD", "YYYY-MM-DD")
- `source_filter`: Optional list of sources to search
- `entity_types`: Optional list of entity types to filter
- `include_causal`: Include causal chain information (default: False)

**Returns**: List of `SearchResult` objects

**Example**:
```python
# Basic search
results = await api.search("python programming", top_k=5)

# Temporal search
results = await api.search(
    "meeting notes",
    temporal_range=("2024-12-01", "2024-12-31"),
    top_k=20
)

# Filtered search
results = await api.search(
    "project updates",
    source_filter=["vault:work-notes"],
    entity_types=["Document", "Note"]
)

# With causal chains
results = await api.search(
    "why did the bug occur",
    include_causal=True
)
```

#### `search_temporal`

```python
async def search_temporal(
    self,
    query: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    relative_range: Optional[str] = None,
    top_k: int = 10,
) -> List[SearchResult]
```

Execute a time-focused search.

**Parameters**:
- `query`: Search query
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `relative_range`: Relative time ("last week", "yesterday", "3 days ago")
- `top_k`: Maximum results

**Returns**: List of `SearchResult` sorted by temporal relevance

**Example**:
```python
# Absolute range
results = await api.search_temporal(
    "research",
    start_date="2024-12-01",
    end_date="2024-12-15"
)

# Relative range
results = await api.search_temporal(
    "what was I doing",
    relative_range="last week"
)
```

#### `search_causal`

```python
async def search_causal(
    self,
    query: str,
    *,
    max_hops: int = 3,
    top_k: int = 10,
) -> List[SearchResult]
```

Search for causal relationships.

**Parameters**:
- `query`: Query about causation
- `max_hops`: Maximum relationship hops (default: 3)
- `top_k`: Maximum results

**Returns**: List of `SearchResult` with causal chain information

**Example**:
```python
results = await api.search_causal(
    "what caused the performance issue",
    max_hops=5
)

for result in results:
    if result.causal_chain:
        print(f"Chain: {result.causal_chain}")
```

#### `search_multimodal`

```python
async def search_multimodal(
    self,
    *,
    text: Optional[str] = None,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
    top_k: int = 10,
) -> List[SearchResult]
```

Search with multiple input modalities.

**Parameters**:
- `text`: Text query (optional)
- `audio_path`: Path to audio file for transcription (optional)
- `image_path`: Path to image for OCR (optional)
- `top_k`: Maximum results

**Returns**: List of `SearchResult`

**Example**:
```python
# Voice search (audio transcribed first)
results = await api.search_multimodal(
    audio_path="/path/to/voice-memo.m4a",
    top_k=10
)

# Image search (OCR extracted first)
results = await api.search_multimodal(
    image_path="/path/to/whiteboard-photo.jpg"
)

# Combined
results = await api.search_multimodal(
    text="related to this",
    image_path="/path/to/diagram.png"
)
```

#### `generate_answer`

```python
async def generate_answer(
    self,
    query: str,
    *,
    top_k: int = 5,
    stream: bool = False,
) -> Union[str, AsyncIterator[str]]
```

Generate an LLM answer grounded in search results.

**Parameters**:
- `query`: Question to answer
- `top_k`: Number of context results to use
- `stream`: Stream response tokens (default: False)

**Returns**: Answer string or async iterator of tokens

**Example**:
```python
# Full answer
answer = await api.generate_answer(
    "What are my main research topics?"
)
print(answer)

# Streaming
async for token in await api.generate_answer(
    "Summarize my Python notes",
    stream=True
):
    print(token, end="", flush=True)
```

#### `submit_feedback`

```python
async def submit_feedback(
    self,
    query: str,
    result_id: str,
    feedback_type: str,
    clicked_position: Optional[int] = None,
) -> None
```

Submit quality feedback for experiential learning.

**Parameters**:
- `query`: Original query
- `result_id`: ID of the result receiving feedback
- `feedback_type`: Type of feedback ("click", "helpful", "not_helpful")
- `clicked_position`: Position in results list

**Example**:
```python
await api.submit_feedback(
    query="python tips",
    result_id="doc-123",
    feedback_type="helpful",
    clicked_position=2
)
```

#### `close`

```python
async def close(self) -> None
```

Close connections and cleanup resources.

**Example**:
```python
try:
    results = await api.search("query")
finally:
    await api.close()
```

## SearchResult Class

```python
@dataclass
class SearchResult:
    id: str                          # Unique identifier
    content: str                     # Text content
    score: float                     # Relevance score (0.0 - 1.0)
    confidence: float                # Confidence score (0.0 - 1.0)
    timestamp: Optional[str]         # ISO timestamp
    entity_type: Optional[str]       # Entity type
    source_type: Optional[str]       # Source type
    causal_chain: Optional[Dict]     # Causal chain info
    metadata: Dict[str, Any]         # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
```

**Example**:
```python
for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score:.2f}")
    print(f"Content: {result.content[:200]}...")
    print(f"Source: {result.source_type}")
    print(f"Timestamp: {result.timestamp}")
    print("---")
```

## SearchConfig Class

```python
@dataclass
class SearchConfig:
    # Weights
    vector_weight: float = 0.6       # Weight for vector results
    graph_weight: float = 0.4        # Weight for graph results

    # Retrieval
    top_k_vector: int = 20           # Vector candidates
    top_k_graph: int = 10            # Graph candidates

    # Performance
    cache_enabled: bool = True       # Enable result caching
    cache_ttl_seconds: int = 300     # Cache TTL

    # Timeouts
    search_timeout_ms: int = 5000    # Search timeout
    llm_timeout_ms: int = 30000      # LLM timeout
```

**Example**:
```python
config = SearchConfig(
    vector_weight=0.7,
    graph_weight=0.3,
    cache_enabled=True,
    search_timeout_ms=3000,
)
```

## Error Handling

```python
from futurnal.search.hybrid.exceptions import (
    SearchError,
    VectorSearchError,
    GraphSearchError,
    TemporalSearchError,
)

try:
    results = await api.search("query")
except TemporalSearchError as e:
    print(f"Temporal search failed: {e}")
except VectorSearchError as e:
    print(f"Vector search failed: {e}")
except GraphSearchError as e:
    print(f"Graph search failed: {e}")
except SearchError as e:
    print(f"General search error: {e}")
```

## Performance Tips

1. **Use caching**: Enable cache for repeated queries
2. **Limit `top_k`**: Smaller values = faster results
3. **Filter sources**: Narrow scope with `source_filter`
4. **Batch searches**: Group related searches
5. **Warm up**: Call `api.warmup()` on startup

```python
# Warmup common queries
await api.warmup(["recent", "important", "project"])
```

## Research Foundation

HybridSearchAPI implements research from:
- GFM-RAG (2502.01113v1): Graph foundation model
- PGraphRAG (2501.02157v2): Personalized retrieval
- Time-R1 (2505.13508v2): Temporal reasoning
