Summary: Outlines the vector embedding service feature with scope, testing, and review checkpoints.

# Feature · Vector Embedding Service

## Goal
Provide an on-device embedding pipeline that generates and manages vector representations for documents and graph entities, synchronizing updates with the PKG to enable hybrid search.

## Success Criteria
- Supports text embeddings for documents, code, and email content using state-of-the-art models (e.g., Instructor-large, CodeBERT) tuned for on-device execution.
- Embedding store (ChromaDB or Weaviate embedded) synchronized with PKG changes.
- Incremental updates handled efficiently with batching and deduplication.
- APIs expose similarity search and metadata filtering for hybrid search assembly.
- Telemetry captures embedding latency, size, and drift metrics.

## Functional Scope
- Embedding model manager with runtime selection based on resource profile.
- Job pipeline triggered by PKG or normalization events.
- Metadata storage linking embeddings to PKG node IDs and provenance.
- Rebuild tools for re-embedding when models update.
- Integrity checks to ensure store consistency.

## Non-Functional Guarantees
- On-device by default; optional cloud model usage requires consent.
- Memory footprint constrained; support streaming embedding for large corpora.
- Deterministic outputs given same model and input.
- Logging limited to metrics, no raw text.

## Dependencies
- Normalized documents and PKG nodes.
- Ingestion orchestrator for job scheduling.
- Privacy audit logging for optional cloud escalations.

## Implementation Guide
1. **Model Benchmarking:** Use @Web research to evaluate state-of-the-art embedding models for relevant domains (text, code); document trade-offs.
2. **Model Management:** Integrate with local model runner (Ollama/llama.cpp) or lightweight PyTorch runtime; provide configuration for quantized models.
3. **Pipeline Orchestration:** Build automata-style state machine for embedding jobs (pending → processing → stored → synced) ensuring reliability.
4. **Store Integration:** Evaluate embedded Chroma vs. Weaviate; implement adapter pattern for possible future swaps.
5. **Sync Hooks:** Subscribe to PKG mutation events; trigger embedding updates and deletions accordingly.
6. **Rebuild Utilities:** Provide CLI/UI to re-embed data when model versions change; track embeddings by model fingerprint.

## Testing Strategy
- **Unit Tests:** Pipeline state transitions, model selection logic, metadata associations.
- **Integration Tests:** End-to-end embedding generation for connectors; ensure store sync with PKG updates.
- **Performance Tests:** Throughput, latency, and memory usage on reference hardware.
- **Regression Tests:** Verify determinism and correct handling of model upgrades.

## Code Review Checklist
- Model downloads and execution remain on-device; cloud usage gated.
- Embeddings linked accurately to PKG entities with provenance.
- Store consistency checks implemented and tested.
- Tests cover re-embedding scenarios and failure recovery.
- Telemetry captures key metrics without leaking sensitive data.

## Documentation & Follow-up
- Publish embedding model catalog and configuration guidance.
- Update developer docs with instructions for adding domain-specific embedding models.
- Coordinate with hybrid search API for contract validation.


