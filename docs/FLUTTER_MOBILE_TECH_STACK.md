# Futurnal Mobile: Full Native On-Device Architecture

> **Architectural Decision:** Full native Dart rewrite - NO remote server  
> **Principle:** Privacy-first, everything runs on-device like desktop  
> **Date:** December 2025

---

## Confirmed Decisions

| Decision | Choice |
|----------|--------|
| **LLM Strategy** | ✅ Full On-Device |
| **Feature Scope** | ✅ All Phases (1-3) |
| **Architecture** | ✅ Full Native Dart Rewrite |
| **Graph Sync** | ✅ Full Local Sync |

---

## Tech Stack

### Core Storage
| Need | Solution |
|------|----------|
| Graph Storage | Isar with IsarLinks |
| Vector Index | HNSW in Dart |
| Settings | Hive |
| Secure Storage | flutter_secure_storage |

### On-Device AI
| Need | Solution |
|------|----------|
| LLM Inference | Cactus (GGUF: Qwen/Gemma) |
| Embeddings | TFLite (MiniLM/BGE) |
| NLP Tasks | Google ML Kit |

---

## Component Mapping (Python → Dart)

| Desktop (Python) | Mobile (Dart) |
|------------------|---------------|
| Neo4j embedded | Isar + IsarLinks |
| ChromaDB | Custom HNSW + Isar |
| Ollama/llama.cpp | Cactus |
| Sentence Transformers | TFLite |
| Unstructured.io | Dart parsers + platform channels |
| LangChain | Custom Dart pipelines |

---

## Implementation Timeline: 22 Weeks

1. **Weeks 1-4:** Core (Isar graph, TFLite, Cactus)
2. **Weeks 5-8:** Ingestion (files, photos, extraction)
3. **Weeks 9-12:** Search (HNSW, GraphRAG)
4. **Weeks 13-18:** Intelligence (insights, curiosity, causal)
5. **Weeks 19-22:** UI, polish, launch

---

## Key Packages

| Package | Purpose |
|---------|---------|
| `isar` | Local graph/document storage |
| `cactus` | On-device LLM inference |
| `tflite_flutter` | Embeddings |
| `flutter_riverpod` | State management |
| `graphview` | PKG visualization |
