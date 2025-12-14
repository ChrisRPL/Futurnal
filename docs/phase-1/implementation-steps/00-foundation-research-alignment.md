# Step 00: Foundation & Research Alignment

## Status: TODO

## Objective

Ensure the codebase is aligned with SOTA research principles before implementing new features. This step audits existing code and establishes patterns that will be used throughout Phase 1 completion.

## Research Foundation

### Key Papers:
- **EDC Framework (2404.03868)**: Extract→Define→Canonicalize pipeline structure
- **AutoSchemaKG (2505.23628v1)**: Autonomous schema principles
- **SEAgent (2508.04700v2)**: Experiential learning patterns

### Research Insight:
> "The current feature spec represents 2020-era thinking. The 2024-2025 research provides the tools to build it right for Ghost→Animal evolution."
> - SOTA Research Summary

## Implementation Tasks

### 1. Audit Existing Search Implementation
**Current State**: Search in `src/futurnal/search/api.py` uses keyword matching
**Research Gap**: Should use embedding similarity + graph traversal (GFM-RAG)

```python
# Current (Line 593-594 in api.py) - WRONG
text_lower = text.lower()
matches = sum(1 for term in query_terms if term in text_lower)

# Should be (per GFM-RAG paper):
# 1. Embed query using embedding service
# 2. Retrieve similar nodes from ChromaDB
# 3. Traverse graph for context (GraphRAG)
# 4. Generate answer using LLM
```

### 2. Verify Infrastructure Exists
Confirm these components are ready (they exist but aren't wired):
- [ ] `src/futurnal/embeddings/service.py` - Embedding generation
- [ ] `src/futurnal/pkg/database/manager.py` - Neo4j connection
- [ ] `src/futurnal/extraction/ollama_client.py` - LLM inference
- [ ] ChromaDB configuration in settings

### 3. Establish EDC Pipeline Pattern
Per EDC Framework paper, structure all extraction as:
```
Phase 1 (Archivist): EXTRACT - Open information extraction
Phase 2 (Analyst): DEFINE - Schema induction from patterns
Phase 3 (Guide): CANONICALIZE - Transform to causal structures
```

### 4. Create Option B Compliance Checklist
Per `.cursor/rules/option-b-principles.mdc`:
- [ ] Ghost model FROZEN (no parameter updates)
- [ ] Learning via token priors (not fine-tuning)
- [ ] Training-free GRPO framework
- [ ] Temporal-first design

### 5. Document Current vs Target Architecture

| Component | Current | Target (Research-Based) |
|-----------|---------|------------------------|
| Search | Keyword matching | Semantic + GraphRAG |
| Answers | Raw snippets | LLM-synthesized |
| Schema | Hardcoded types | Autonomous evolution |
| Learning | Passive logging | Active experiential |
| Temporal | Not extracted | Full temporal graph |

## Success Criteria

- [ ] All infrastructure components verified working
- [ ] EDC pipeline pattern documented
- [ ] Option B compliance checklist complete
- [ ] Architecture comparison documented
- [ ] Ready for Step 01 implementation

## Files to Review

- `src/futurnal/search/api.py` - Current search implementation
- `src/futurnal/embeddings/service.py` - Embedding infrastructure
- `src/futurnal/pkg/database/manager.py` - Neo4j connection
- `src/futurnal/extraction/ollama_client.py` - LLM client
- `.cursor/rules/option-b-principles.mdc` - Compliance rules

## Dependencies

None - this is the foundation step.

## Next Step

After completing this audit, proceed to **Step 01: Intelligent Search**.
