Summary: Roadmap and task breakdown to bring the Document Normalization Pipeline to production readiness for Ghost grounding.

# Document Normalization Pipeline · Production Plan

This folder tracks the work required to ship Feature 6 (Document Normalization Pipeline) with production-quality stability, observability, and privacy compliance—enabling the Ghost to learn from diverse document formats through consistent, high-quality normalization. Each task ensures the pipeline transforms raw connector outputs into standardized, chunked documents ready for entity extraction, embeddings, and PKG storage. Task documents define scope, acceptance criteria, test plans, and operational guidance aligned to the experiential learning architecture in [system-architecture.md](../../architecture/system-architecture.md).

## Task Index
- [01-normalized-document-schema.md](01-normalized-document-schema.md)
- [02-normalization-service-architecture.md](02-normalization-service-architecture.md)
- [03-format-adapter-registry.md](03-format-adapter-registry.md)
- [04-chunking-engine.md](04-chunking-engine.md)
- [05-metadata-enrichment-pipeline.md](05-metadata-enrichment-pipeline.md)
- [06-streaming-processor.md](06-streaming-processor.md)
- [07-unstructured-integration-bridge.md](07-unstructured-integration-bridge.md)
- [08-quarantine-error-routing.md](08-quarantine-error-routing.md)
- [09-versioning-provenance-tracking.md](09-versioning-provenance-tracking.md)
- [10-orchestrator-integration.md](10-orchestrator-integration.md)
- [11-performance-optimization.md](11-performance-optimization.md)
- [12-quality-gates-testing.md](12-quality-gates-testing.md)

## Technical Foundation

### Document Processing Library
**Unstructured.io v0.18.15** ([/unstructured-io/unstructured](https://github.com/unstructured-io/unstructured)) - Trust Score 8.7
- Production-ready parsing for 60+ file formats
- Advanced chunking strategies (by_title, by_page, basic)
- Metadata preservation during processing
- Element-based document representation
- Optimized for RAG and LLM workflows

### Language Detection
**fasttext-langdetect** ([/LlmKira/fast-langdetect](https://github.com/LlmKira/fast-langdetect))
- 80x faster than conventional langdetect
- 95% accuracy for language identification
- On-device execution (no network dependency)
- Support for 175+ languages

### Content Hashing
**SHA-256** - Cryptographic hashing for provenance
- Deterministic content fingerprinting
- Change detection for versioning
- Idempotency guarantees

### Chunking Strategies
From Unstructured.io best practices:
- **by_title**: Preserves section boundaries, optimal for structured documents
- **by_page**: Maintains page boundaries (PDFs, scanned documents)
- **basic**: Simple size-based chunking with overlap
- **custom**: Source-type specific semantic chunking

## Architectural Patterns

Following established patterns from Obsidian/Local Files/IMAP/GitHub connectors:

1. **Service + Adapter Pattern**
   - `NormalizationService`: Central orchestration interface
   - `FormatAdapterRegistry`: Pluggable format-specific handlers
   - Leverage existing normalizers (Obsidian, Email, GitHub)

2. **Privacy-First Design**
   - No document content in logs or audit trails
   - Metadata-only telemetry
   - Audit events track processing status without exposing content
   - Privacy-aware error messages

3. **Streaming Processing**
   - Chunked document processing for large files (>100MB)
   - Minimal memory footprint via iterative parsing
   - Progress tracking without holding full content in memory
   - Resource limits enforced

4. **Quarantine & Resilience**
   - Failed normalization → quarantine with detailed diagnostics
   - Format-specific error routing
   - Operator CLI for reprocessing
   - Retry policies per failure classification

5. **Orchestrator Integration**
   - Register with `IngestionOrchestrator`
   - Feed normalized chunks to `NormalizationSink`
   - StateStore for checkpoint tracking
   - AuditLogger for processing events

## Normalization Pipeline Flow

```
Raw Document (from connector)
  ↓
Format Detection & Adapter Selection
  ↓
Unstructured.io Parsing (format-specific strategy)
  ↓
Chunking Engine (configurable strategy)
  ↓
Metadata Enrichment (language, hash, type)
  ↓
Normalized Document Chunks
  ↓
NormalizationSink → PKG + Vector Store
```

## Supported Format Categories

### Text Documents
- Markdown (`.md`) - Already implemented via ObsidianNormalizer
- Plain text (`.txt`)
- Rich text (`.rtf`)
- PDF (`.pdf`) - Text extraction + OCR fallback

### Email & Communication
- Email (RFC822/MIME) - Already implemented via EmailNormalizer
- HTML (`.html`, `.htm`)
- XML (`.xml`)

### Code & Technical
- Source code (`.py`, `.js`, `.java`, etc.) - Comment extraction
- Jupyter notebooks (`.ipynb`)
- Configuration files (`.json`, `.yaml`, `.toml`)

### Office Documents
- Microsoft Word (`.docx`)
- PowerPoint (`.pptx`)
- Excel (`.xlsx`)
- LibreOffice formats

### Structured Data
- CSV (`.csv`)
- JSON (`.json`)
- YAML (`.yaml`)

## AI Learning Focus

Transform diverse document formats into consistent experiential memory:

- **Content Understanding**: Standardized text representation for semantic embeddings
- **Document Structure**: Preserve hierarchies, sections, and relationships
- **Temporal Patterns**: Track document creation, modification, and evolution
- **Metadata Signals**: Language, format, size, complexity metrics
- **Provenance Tracking**: Content hashing for change detection and versioning
- **Chunking Quality**: Semantic coherence for effective retrieval and understanding

## Schema Design Principles

1. **Standardization**: Consistent schema across all formats
2. **Extensibility**: Custom metadata per source type
3. **Versioning**: Content hash + timestamp for change tracking
4. **Hierarchy**: Parent-child relationships for chunked documents
5. **Privacy**: Separate content from metadata for audit logging

## Performance Targets

Based on system requirements and reference hardware (Apple Silicon):

- **Throughput**: Process ≥5 MB/s of mixed document types
- **Memory**: Peak usage <2 GB for large document processing
- **Determinism**: 100% identical outputs for unchanged inputs
- **Latency**: <500ms for small documents (<1MB), streaming for larger
- **Offline**: 100% functionality without network dependency

## Testing Strategy

### Determinism Tests
- Reprocess identical documents, verify byte-identical output
- Content hash stability across multiple runs
- Metadata consistency verification

### Format Coverage Tests
- Sample documents from all 60+ supported formats
- Edge cases: corrupted files, malformed content, encoding issues
- Round-trip fidelity where applicable

### Performance Benchmarks
- Throughput measurements per format category
- Memory profiling for large documents
- Streaming vs batch processing comparisons

### Integration Tests
- Full pipeline from connector → normalization → PKG storage
- Multi-format mixed batches
- Error recovery and quarantine workflows

## Quality Gates

Before marking normalization pipeline production-ready:

- ✅ All 60+ formats parse successfully with sample documents
- ✅ Determinism tests pass 100% (byte-identical outputs)
- ✅ Performance benchmarks meet ≥5 MB/s target
- ✅ Memory usage <2 GB for largest test documents
- ✅ Integration tests pass for all connector types
- ✅ Quarantine workflow handles all failure modes gracefully
- ✅ Privacy audit shows no content leakage in logs
- ✅ Streaming processor handles 1GB+ documents without OOM

## Usage

- Update these plans as tasks progress; each file captures scope, deliverables, and open questions.
- Cross-link implementation PRs and test evidence directly inside the relevant markdown files.
- When a task reaches completion, summarize learnings and move any follow-up work to the appropriate phase-2 documents.


