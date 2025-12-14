# Step 02: LLM Answer Generation

## Status: TODO

## Objective

Add LLM-powered answer synthesis to search results. Instead of returning raw document snippets, the system generates coherent, contextual answers using Ollama with retrieved context (RAG pattern).

## Research Foundation

### Primary Papers:

#### CausalRAG (ACL 2025)
**Key Innovation**: Generate with causal awareness from graph-retrieved context
**Application**: Use causal graph structure to generate more accurate answers

#### LLM-Enhanced Symbolic Reasoning (2501.01246v1)
**Key Innovation**: Combine LLM generation with rule-based reasoning
**Application**: Hybrid approach ensures factual grounding

### Research Insight from FUTURNAL_CONCEPT.md:
> "Result Assembly: Generate answer summaries using local LLM with retrieved context; attach provenance references."
> - feature-hybrid-search-api.md (Line 38)

**This is specified but NOT implemented!**

## Current State Analysis

### What Exists:
1. **Ollama Client** - `src/futurnal/extraction/ollama_client.py` configured
2. **Search API** - Returns raw snippets without synthesis
3. **Provenance Tracking** - Source metadata available

### What's Missing:
- No answer generation step in search pipeline
- No context assembly for LLM prompting
- No streaming support for longer responses

## Implementation Tasks

### 1. Create Answer Generation Module

**New File**: `src/futurnal/search/answer_generator.py`

```python
"""
Answer Generation Module - LLM-Powered Response Synthesis

Research Foundation:
- CausalRAG (ACL 2025): Causal-aware generation
- LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach

Per FUTURNAL_CONCEPT.md:
"Generate answer summaries using local LLM with retrieved context"
"""

from typing import List, Dict, Optional, AsyncIterator
from futurnal.extraction.ollama_client import OllamaClient

class AnswerGenerator:
    """Generate synthesized answers from retrieved context."""

    def __init__(self, model_name: str = "llama3.1:8b-instruct-q4_0"):
        self.client = OllamaClient(model=model_name)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """System prompt for answer generation."""
        return """You are Futurnal's knowledge assistant. Your role is to synthesize
answers from the user's personal knowledge graph.

RULES:
1. ONLY use information from the provided context
2. Always cite sources with [Source: filename] notation
3. If the context doesn't contain the answer, say so honestly
4. Be concise but comprehensive
5. Preserve the user's terminology and concepts

FORMAT:
- Start with a direct answer to the question
- Support with evidence from context
- End with source citations"""

    async def generate_answer(
        self,
        query: str,
        context: List[Dict],
        graph_context: Optional[Dict] = None,
    ) -> str:
        """
        Generate synthesized answer from retrieved context.

        Args:
            query: User's question
            context: Retrieved document snippets
            graph_context: Related entities and relationships from PKG
        """
        # Assemble context for LLM
        context_text = self._assemble_context(context, graph_context)

        # Build prompt
        prompt = f"""Based on the following context from my personal knowledge:

{context_text}

Question: {query}

Provide a comprehensive answer with source citations:"""

        # Generate answer
        response = await self.client.generate(
            prompt=prompt,
            system=self.system_prompt,
            temperature=0.3,  # Lower temperature for factual responses
        )

        return response

    async def stream_answer(
        self,
        query: str,
        context: List[Dict],
        graph_context: Optional[Dict] = None,
    ) -> AsyncIterator[str]:
        """Stream answer generation for real-time UI updates."""
        context_text = self._assemble_context(context, graph_context)

        prompt = f"""Based on the following context from my personal knowledge:

{context_text}

Question: {query}

Provide a comprehensive answer with source citations:"""

        async for chunk in self.client.stream_generate(
            prompt=prompt,
            system=self.system_prompt,
            temperature=0.3,
        ):
            yield chunk

    def _assemble_context(
        self,
        context: List[Dict],
        graph_context: Optional[Dict] = None,
    ) -> str:
        """Assemble context from retrieval results."""
        parts = []

        # Add document snippets
        for i, doc in enumerate(context[:10], 1):  # Limit to top 10
            source = doc.get('metadata', {}).get('source', 'Unknown')
            content = doc.get('content', '')[:500]  # Truncate long content
            parts.append(f"[Document {i} - {source}]\n{content}\n")

        # Add graph context if available
        if graph_context:
            relationships = graph_context.get('relationships', [])
            if relationships:
                parts.append("\n[Related Concepts and Connections]")
                for rel in relationships[:5]:
                    parts.append(f"- {rel['subject']} {rel['predicate']} {rel['object']}")

        return "\n".join(parts)
```

### 2. Integrate with Search API

**File**: `src/futurnal/search/api.py`

```python
from futurnal.search.answer_generator import AnswerGenerator

class HybridSearchAPI:
    def __init__(self, ...):
        # ... existing init
        self.answer_generator = AnswerGenerator()

    async def search_with_answer(
        self,
        query: str,
        top_k: int = 10,
        generate_answer: bool = True,
    ) -> Dict:
        """
        Search with optional LLM answer generation.

        Returns:
            {
                'answer': str,  # Synthesized answer (if generate_answer=True)
                'results': List[Dict],  # Raw search results
                'sources': List[str],  # Source documents used
            }
        """
        # Step 1: GraphRAG retrieval (from Step 01)
        results = await self._hybrid_search_graphrag(query, top_k)

        response = {
            'results': results,
            'sources': [r.get('metadata', {}).get('source') for r in results],
        }

        # Step 2: Generate answer if requested
        if generate_answer and results:
            graph_context = await self._get_graph_context_for_results(results)
            answer = await self.answer_generator.generate_answer(
                query=query,
                context=results,
                graph_context=graph_context,
            )
            response['answer'] = answer

        return response
```

### 3. Add Streaming Support to Tauri IPC

**File**: `src-tauri/src/commands/search.rs` (or equivalent)

```rust
#[tauri::command]
async fn search_with_streaming_answer(
    query: String,
    state: State<'_, AppState>,
) -> Result<impl Stream<Item = SearchStreamChunk>, Error> {
    // Stream answer chunks to frontend for real-time display
}
```

### 4. Update Frontend Search Results

**File**: `desktop/src/components/search/SearchResults.tsx`

```tsx
interface SearchWithAnswer {
  answer?: string;
  results: SearchResult[];
  sources: string[];
  isStreaming?: boolean;
}

export function SearchResults({ data, isStreaming }: { data: SearchWithAnswer; isStreaming?: boolean }) {
  return (
    <div className="search-results">
      {/* Synthesized Answer - Prominently displayed */}
      {data.answer && (
        <div className="answer-section bg-surface border border-white/10 rounded-lg p-4 mb-4">
          <h3 className="text-sm font-medium text-white/60 mb-2">Answer</h3>
          <div className="text-white prose prose-invert">
            {isStreaming ? (
              <StreamingText text={data.answer} />
            ) : (
              <Markdown>{data.answer}</Markdown>
            )}
          </div>
          {data.sources.length > 0 && (
            <div className="sources mt-3 text-xs text-white/40">
              Sources: {data.sources.filter(Boolean).join(', ')}
            </div>
          )}
        </div>
      )}

      {/* Raw Results - Below answer */}
      <div className="results-list">
        <h3 className="text-sm font-medium text-white/60 mb-2">Related Documents</h3>
        {data.results.map((result) => (
          <ResultItem key={result.id} result={result} />
        ))}
      </div>
    </div>
  );
}
```

### 5. Add Answer Toggle to Search UI

**File**: `desktop/src/components/search/CommandPalette.tsx`

```tsx
// Add toggle for answer generation
const [generateAnswer, setGenerateAnswer] = useState(true);

// In the search execution
const handleSearch = async () => {
  const response = await invoke('search_with_answer', {
    query,
    generateAnswer,
  });
  setResults(response);
};
```

## Success Criteria

### Functional:
- [ ] LLM generates synthesized answers from context
- [ ] Answers include source citations
- [ ] Streaming support for real-time display
- [ ] Toggle to enable/disable answer generation

### Quality:
- [ ] Answers are factually grounded in retrieved context
- [ ] No hallucinations (only uses provided context)
- [ ] Clear source attribution
- [ ] Concise but comprehensive responses

### Performance:
- [ ] First token < 500ms (streaming)
- [ ] Full answer < 3 seconds
- [ ] Does not block UI

## Files to Create/Modify

### Backend:
- **NEW**: `src/futurnal/search/answer_generator.py` - Answer generation module
- `src/futurnal/search/api.py` - Integrate answer generation
- `src/futurnal/extraction/ollama_client.py` - Ensure streaming works

### Frontend:
- `desktop/src/components/search/SearchResults.tsx` - Show synthesized answer
- `desktop/src/components/search/CommandPalette.tsx` - Add answer toggle
- `desktop/src/stores/searchStore.ts` - Update result types

### Tests:
- `tests/search/test_answer_generator.py` - Unit tests
- `tests/search/test_integration_answer.py` - Integration tests

## Dependencies

- **Step 01**: Intelligent search with GraphRAG (provides context)
- **Infrastructure**: Ollama running with llama3.1:8b-instruct model

## Next Step

After implementing answer generation, proceed to **Step 03: Chat Interface**.

## Research References

1. **CausalRAG**: `docs/phase-1/SOTA_RESEARCH_SUMMARY.md` (Paper #4)
2. **LLM-Enhanced Symbolic**: `docs/phase-1/papers/converted/2501.01246v1.md`
3. **Feature Spec**: `docs/phase-1/feature-hybrid-search-api.md` (Line 38)
