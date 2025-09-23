Summary: Plans the entity and relationship extraction feature with scope, testing, and review checkpoints.

# Feature · Entity & Relationship Extraction

## Goal
Convert normalized documents into semantic triples (Subject, Predicate, Object) and supporting attributes using on-device LLMs to populate the Personal Knowledge Graph while maintaining privacy.

## Success Criteria
- Extraction pipeline produces high-precision triples across document types with configurable confidence thresholds.
- Pipeline runs locally via quantized models (Llama-3.1 8B, Mistral 7B) with optional cloud escalation per consent flow.
- Triples include provenance references back to source chunks and timestamps.
- Batch and streaming modes supported to handle both backfill and incremental updates.

## Functional Scope
- Prompt templates tuned for entity/relationship identification by document type.
- Post-processing heuristics to normalize entities (coreference resolution, canonical naming).
- Confidence scoring and filtering to prevent graph pollution.
- Feedback loop for manual corrections (capture for future model tuning).
- Export interface feeding PKG storage layer.

## Non-Functional Guarantees
- On-device inference default; cloud escalation only via explicit consent.
- Throughput aligned with ingestion pace; leverages GPU/ML accelerators when available.
- Deterministic reprocessing by storing model version + prompt signature.
- Logging redacts sensitive entity text, focusing on metrics.

## Dependencies
- Normalized documents ([feature-document-normalization](feature-document-normalization.md)).
- PKG schema defined in [feature-pkg-graph-storage](feature-pkg-graph-storage.md).
- Privacy audit logging for escalation tracking.

## Implementation Guide
1. **Model Selection:** Benchmark quantized models for extraction accuracy; reference @Web research for state-of-the-art prompt engineering and automata-inspired validation.
2. **Prompt Framework:** Implement modular prompt templates per document type with context windows tuned to chunk sizes.
3. **Post-Processing:** Apply spaCy/LLM hybrid for entity normalization; use heuristics for merging duplicates (case-insensitive, alias mapping).
4. **Confidence Scoring:** Combine model confidence with rule-based checks; enable configurable thresholds per connector.
5. **Feedback Capture:** Allow operators to accept/reject triples; log decisions for future training.
6. **Pipeline Orchestration:** Integrate with ingestion orchestrator; support batch reprocessing when schema evolves.

## Testing Strategy
- **Unit Tests:** Prompt selection logic, normalization routines, confidence scoring.
- **Integration Tests:** Full extraction from fixture corpora (markdown, email, GitHub issues) with golden triples.
- **Quality Evaluation:** Precision/recall measurement against labeled datasets; target ≥0.8 precision.
- **Performance Tests:** Throughput and latency benchmarks on reference hardware.

## Code Review Checklist
- Models run locally by default; cloud paths respect consent logging.
- Triples include full provenance metadata.
- Normalization avoids over-merging distinct entities.
- Tests cover diverse entity types and relationship predicates.
- Metrics collection supports future analytics.

## Documentation & Follow-up
- Document prompt templates and tuning tips in shared knowledge base.
- Update developer guide on extending predicate taxonomy.
- Coordinate with PKG team for schema adjustments based on extraction outputs.


