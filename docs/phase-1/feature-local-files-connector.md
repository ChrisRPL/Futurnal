Summary: Defines the local filesystem connector that grounds the Ghost in user's file-based experiential data.

# Feature · Local Files Connector

## Goal
**Ground the Ghost in user's file-based experiential data sources**, enabling the AI to develop understanding of the user's personal document universe. This connector transforms local directories into structured experiential memory within the Personal Knowledge Graph, providing the foundation for AI personalization and evolution. The Ghost learns from the user's file organization patterns, document relationships, and temporal evolution of content—building the high-fidelity memory required for true personalized intelligence.

## Success Criteria

### Experiential Memory Construction
- Ghost develops understanding of user's file organization patterns and document relationships.
- Temporal evolution of files is captured, enabling the AI to learn from content changes over time.
- Experiential metadata (provenance, temporal context, relational structure) enriches the PKG for AI learning.

### Technical Performance
- Users configure one or more local directory sources with inclusion/exclusion rules.
- Processing completes within target throughput (≥5 MB/s on reference hardware) with <1% parse failures.
- Documents carry complete provenance metadata (path, hash, timestamp) and temporal context.
- Scheduler supports incremental updates (create/update/delete) within minutes, preserving experiential history.

## Functional Scope
- Source registration via CLI or desktop UI stub.
- Recursive directory crawl honoring `.futurnalignore` rules.
- File batching and queuing into Unstructured.io parser jobs.
- Provenance metadata persistence and change detection via file hash + mtime.
- Retry/quarantine flow for unreadable files.

## Non-Functional Guarantees
- Offline-first operation; no network dependency.
- Respect OS-level permissions and sandbox boundaries.
- Logging limited to local audit trail with user-readable entries.
- Parallelization tuned for Apple Silicon big.LITTLE cores; expose config knob for concurrency.

## Dependencies
- [system-architecture.md](../architecture/system-architecture.md) ingestion layer design.
- Unstructured.io parsing library.
- Local scheduler primitives from [feature-ingestion-orchestrator](feature-ingestion-orchestrator.md).

## Implementation Guide

### Experiential Memory Construction Pipeline
1. **Source Definition Schema:** Implement `LocalDirectorySource` with include/exclude glob patterns to define which experiential data sources the Ghost should learn from.
2. **Change Detection Engine:** Use file hashes stored in lightweight SQLite to track temporal evolution of user's document universe; leverage macOS `FSEvents` for real-time experiential updates.
3. **Batch Planner:** Apply automata-based job state tracking to manage experiential data flow: discover → analyze → memorize. Link storage commits to Neo4j PKG nodes (Ghost's memory) and Chroma vectors (semantic understanding) in the same transaction boundary.
4. **Experiential Parsing Pipeline:** Stream files through Unstructured.io to extract semantic understanding while preserving temporal and relational context. Persist parsed artifacts into `workspace/parsed/` for Ghost's audit and learning replay.
5. **Error Handling:** Integrate resilient retry strategy for files the Ghost cannot initially understand. Route repeated failures to quarantine queue while maintaining experiential metadata.
6. **Learning Quality Metrics:** Emit processing duration, throughput, and **experiential context preservation** metrics to telemetry. Track Ghost's growing understanding of user's document patterns.
7. **Operator Tooling:** Expose CLI commands for viewing Ghost's learning progress (`futurnal local-sources telemetry`) and experiential audit events (`futurnal local-sources audit`).

## Testing Strategy
- **Unit Tests:** Change detection (hash diff), ignore rules, metadata serialization.
- **Integration Tests:** End-to-end crawl against fixture directories (nested, binary, large files) with golden outputs.
- **Performance Tests:** Benchmark ingestion throughput on M-series reference hardware and record metrics in `telemetry/` for roadmap KPIs.
- **Security Tests:** Validate permission errors handled gracefully without privilege escalation.
- **Operational Tests:** Exercise CLI telemetry/audit commands against fixtures to ensure workspace artifacts remain readable and privacy-compliant.

## Production Readiness Plan
- **Integration & Performance Suite:** Build automated scenarios covering large directories, symlinks, permission failures, and deletion flows; add throughput benchmarks validating the ≥5 MB/s target.
- **Quarantine Operations:** Extend CLI with commands to inspect, retry, or dismiss quarantined files; document operator remediation flow and surface counts in telemetry reports. Implemented via `futurnal local-sources quarantine` subcommands with telemetry summary generation and archived history.
- **Throughput & Concurrency Controls:** Expose configuration for scan intervals, worker concurrency, and batching; validate FSEvents-driven updates under heavy churn without overwhelming the queue.
- **Privacy & Audit Hardening:** Enrich audit entries with redacted metadata, integrate consent prompts for external parsers, and ensure telemetry omits sensitive paths while remaining user-auditable.
- **Scheduler UX Enhancements:** Provide CLI/desktop controls for scheduling (cron/interval), priority overrides, manual pause/resume, and job retry counters to keep orchestration deterministic.
- **Configuration & Security:** Add managed settings for Neo4j/Chroma credentials, encryption, backups, and workspace health checks; ensure parsed cache cleanup mirrors PKG/vector deletions.
- **Code Health:** Migrate lingering Pydantic V1 validators to `@field_validator`, expand docstrings/logging, and attach regression prompts from `prompts/phase-1-archivist.md` for reviewers.

## Code Review Checklist
- Change detection avoids redundant parsing and handles deletes.
- Provenance metadata complete and immutable post-ingestion.
- Error paths respect privacy logging requirements.
- Parallelization configurable and safe for resource constraints.
- Tests cover edge cases (symlinks, hidden files, large binaries).

## Documentation & Follow-up
- Update `DEVELOPMENT_GUIDE.md` with connector usage notes.
- Capture metrics baselines for ingestion throughput in shared telemetry doc.
- Feed lessons learned into other connector feature docs.
- Implementation reference: see `src/futurnal/ingestion/local/connector.py` and related tests in `tests/ingestion/local/`.
- CLI registration tool (`futurnal/cli/local_sources.py`) with Typer commands and tests in `tests/cli/test_local_sources.py`.
- Telemetry & normalization: instrumentation now writes to `telemetry/` via `TelemetryRecorder`; normalized PKG/vector outputs persist under `pkg/` and `vec/` directories for downstream consumption.


