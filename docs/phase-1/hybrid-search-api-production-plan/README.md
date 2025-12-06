# Hybrid Search API Production Plan

**Status**: Ready for Implementation
**Dependencies**: PKG Graph Storage, Vector Embedding Service, Ollama LLM Backend

## Overview

This production plan implements the hybrid search API with **comprehensive temporal and causal query support** required for Option B. The API must blend vector similarity and graph traversal with temporal queries, causal chain exploration, schema-aware retrieval, and multimodal content handling for Ghost→Animal AI evolution.

## Research Foundation

### Primary References
- **CausalRAG (ACL 2025)**: Integrates causal graphs into RAG for reasoning-based retrieval
- **Time-R1 (2025)**: Temporal reasoning framework for knowledge-intensive queries
- **DeepSearch**: Multi-hop retrieval optimization with adaptive strategies
- **AutoSchemaKG**: Schema-aware retrieval adaptation for evolving knowledge graphs

### Hybrid Retrieval Techniques
- Vector + graph fusion strategies (RRF, weighted combination)
- Re-ranking with LLM feedback (using Ollama backend)
- Temporal-aware scoring and decay functions
- Multimodal content handling (OCR, transcriptions)

### Training-Free Optimization
- **GRPO Integration**: Search quality feedback → experiential learning
- **TOTAL Framework**: Query understanding templates with textual gradients
- **Template Evolution**: Query patterns evolve via KEEP/FIX/DISCARD signals

---

## Critical for Option B

The hybrid search API must support:
- **Temporal query engine**: Time range queries, temporal relationship traversal
- **Causal chain retrieval**: Find causal paths, correlation patterns
- **Schema-aware retrieval**: Adaptive strategies for evolved schema
- **Intent classification**: Route queries via Ollama LLM backend
- **Multimodal search**: OCR-extracted and audio transcription content
- **Phase 2/3 readiness**: Correlation detection, causal hypothesis exploration
- **Performance**: Sub-1s latency for 95% of queries

---

## Implementation Modules

### [01 · Temporal Query Engine](01-temporal-query-engine.md)
**Criticality**: CRITICAL
**Deliverables**:
- Time range query support with temporal parsing
- Temporal relationship traversal (before/after/during)
- Decay-weighted relevance scoring
- Integration with PKG temporal queries
- Temporal-aware embeddings via multi-model architecture

### [02 · Causal Chain Retrieval](02-causal-chain-retrieval.md)
**Criticality**: HIGH (Phase 2/3 foundation)
**Deliverables**:
- Causal path finding algorithms
- Correlation pattern detection preparation
- Bradford Hill criteria support for Phase 3
- Multi-hop causal traversal with confidence scoring
- Graph-based causal evidence aggregation

### [03 · Schema-Aware Hybrid Retrieval](03-schema-aware-retrieval.md)
**Criticality**: HIGH
**Deliverables**:
- Multi-model embedding integration (Instructor-large, CodeBERT)
- Entity type retrieval strategies (Event, Code, Document)
- Schema version compatibility handling
- Adaptive retrieval for evolved schema
- Query embedding routing by content type

### [04 · Query Routing & Orchestration](04-query-routing-orchestration.md)
**Criticality**: CRITICAL
**Deliverables**:
- **Ollama LLM backend** for intent classification (800x speedup)
- Intent classification: temporal, causal, exploratory, factual, code
- **GRPO experiential learning hooks** for quality feedback
- **Thought template integration** for query understanding
- Multi-strategy composition and fusion
- Result ranking with confidence scoring

### [05 · Performance & Caching](05-performance-caching.md)
**Criticality**: HIGH
**Deliverables**:
- **Multi-layer cache** (query results, embeddings, LLM intent, graph traversal)
- **Ollama connection pooling** for LLM optimization
- **Model warm-up system** for cold start mitigation
- **Query plan optimizer** with cost-based strategy selection
- **Performance profiler** with bottleneck detection
- Cache invalidation on PKG/schema updates

### [06 · Integration Testing](06-integration-testing.md)
**Criticality**: CRITICAL (Production gate)
**Deliverables**:
- End-to-end query tests (temporal, causal, exploratory)
- Relevance metrics validation (MRR >0.7, precision@5 >0.8)
- LLM backend integration tests (Ollama)
- Multimodal content search tests
- Performance benchmarks (P95 <1s)
- Production readiness verification

### [07 · Multimodal Query Handling](07-multimodal-query-handling.md) *(NEW)*
**Criticality**: HIGH
**Deliverables**:
- **DeepSeek-OCR integration** for scanned document search
- **Whisper V3 integration** for audio transcription search
- Modality hint detection in queries
- Source-aware ranking with confidence weighting
- **Cross-modal fusion** for unified results
- OCR fuzzy matching for error tolerance
- Semantic-first search for transcriptions

---

## Architecture Integration

### LLM Backend (Ollama)
```
Environment Variables:
- FUTURNAL_LLM_BACKEND=ollama|hf|auto
- FUTURNAL_PRODUCTION_LLM=llama3.1|phi3|qwen|bielik|kimi|gpt-oss|auto

Available Models:
| Alias    | Model               | VRAM  | Use Case              |
|----------|---------------------|-------|-----------------------|
| phi3     | phi3:mini           | 4GB   | Fast testing, CI/CD   |
| llama3.1 | llama3.1:8b         | 8GB   | Production default    |
| qwen     | qwen2.5-coder:32b   | 16GB  | Code queries          |
| bielik   | bielik:4.5b         | 5GB   | Polish language       |
| kimi/k2  | kimi-k2:thinking    | 16GB  | Advanced reasoning    |
| gpt-oss  | gpt-oss:20b         | 12GB  | Unrestricted content  |

Dynamic Model Selection:
- Polish queries → Bielik 4.5B (auto-detected)
- Advanced reasoning → Kimi-K2-Thinking
- Code queries → Qwen 2.5 Coder
- Default → Llama 3.1 8B

Runtime Switching:
- Set FUTURNAL_PRODUCTION_LLM=auto for query-based selection
- Or specify model alias directly (e.g., bielik, kimi)

Performance: 800x speedup vs HuggingFace (documented in entity-extraction plan)
```

### Multi-Model Embeddings
```
Model Router:
- General entities → Instructor-large
- Code entities → CodeBERT
- Events → Instructor-large with temporal context
- Documents → Instructor-large (longer context)
```

### Multimodal Pipeline
```
Content Sources:
- TEXT_NATIVE: Direct text files (Markdown, etc.)
- OCR_DOCUMENT: Scanned PDFs via DeepSeek-OCR
- OCR_IMAGE: Images with text via DeepSeek-OCR
- AUDIO_TRANSCRIPTION: Whisper V3 transcriptions
- VIDEO_TRANSCRIPTION: Video audio tracks
```

---

## Success Metrics

| Metric | Target | Module |
|--------|--------|--------|
| Temporal query functional | <1s latency | 01 |
| Causal chain retrieval | Operational | 02 |
| Schema-aware adaptation | Functional | 03 |
| Intent classification | >85% accuracy | 04 |
| LLM inference latency | <100ms (Ollama) | 04, 05 |
| Cache hit rate | >60% | 05 |
| P95 query latency | <1s | 05, 06 |
| MRR | >0.7 | 06 |
| Precision@5 | >0.8 | 06 |
| OCR content relevance | >80% | 07 |
| Audio content relevance | >75% | 07 |

---

## Option B Compliance Checklist

- [x] **Ghost Model Frozen**: Ollama/LLM used for inference only, no fine-tuning
- [x] **Experiential Learning Hooks**: Search quality feedback → template evolution
- [x] **Temporal-First Design**: Events require temporal metadata, first-class queries
- [x] **Schema Evolution Support**: Adapts to evolved PKG schema
- [x] **Causal Structure Preparation**: Ready for Phase 2/3 causal queries
- [x] **Quality Gates Defined**: MRR >0.7, precision@5 >0.8, latency <1s
- [x] **Local-First Processing**: All search runs on-device

---

## Dependencies

### Internal Dependencies
- PKG Graph Storage (with temporal and causal support)
- Vector Embedding Service (multi-model architecture)
- Temporal extraction pipeline
- Schema evolution system
- Ollama LLM backend (from entity-extraction)

### External Dependencies
- **Ollama**: Local LLM inference server
- **Instructor-large**: Primary embedding model
- **CodeBERT**: Code entity embeddings
- **DeepSeek-OCR**: Document OCR extraction
- **Whisper V3**: Audio transcription

### Infrastructure Requirements
- Ollama server running (localhost:11434)
- ≥12GB VRAM for multi-model support
- Cache backend (in-memory or Redis)

---

## Quality Gates (Production Deployment)

All gates must pass before production deployment:

| Gate | Requirement | Module |
|------|-------------|--------|
| Temporal Queries | Functional | 01 |
| Causal Retrieval | Functional | 02 |
| Schema Adaptation | Functional | 03 |
| Intent Classification | >85% accuracy | 04 |
| Relevance MRR | >0.7 | 06 |
| Relevance Precision | >0.8 | 06 |
| Latency P95 | <1s | 05, 06 |
| Integration | End-to-end passing | 06 |

---

## Next Steps

1. **Implement Module 01**: Temporal query engine with decay scoring
2. **Implement Module 02**: Causal chain retrieval (Phase 2/3 foundation)
3. **Implement Module 04**: Query routing with Ollama backend integration
4. **Implement Module 03**: Multi-model embedding integration
5. **Implement Module 05**: Performance optimization and caching
6. **Implement Module 07**: Multimodal content search
7. **Execute Module 06**: Integration testing and quality validation

---

**This search API is the interface layer enabling Phase 2 correlation detection and Phase 3 causal hypothesis exploration, with comprehensive multimodal support and production-grade performance.**
