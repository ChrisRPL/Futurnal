Summary: Defines the local filesystem ingestion feature including scope, testing, and review checkpoints.

# Feature · Local Files Connector

## Goal
Enable users to ingest local directories into Futurnal with deterministic scheduling, provenance capture, and privacy-first processing while laying the groundwork for additional connectors.

## Success Criteria
- Users configure one or more local directory sources with inclusion/exclusion rules.
- Ingestion completes within target throughput (≥5 MB/s on reference hardware) with <1% parse failures.
- Parsed documents carry provenance metadata (path, hash, timestamp) and are stored in normalized format ready for downstream pipelines.
- Scheduler supports incremental updates (create/update/delete) within minutes.

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
1. **Source Definition Schema:** Implement `LocalDirectorySource` with include/exclude glob patterns.
2. **Change Detection Engine:** Use file hashes stored in lightweight SQLite to detect diffs; leverage macOS `FSEvents` for optional real-time updates.
3. **Batch Planner:** Apply automata-based job state tracking (@Web Automata-Based Programming) to manage crawl → parse → persist transitions clearly.
4. **Parsing Pipeline:** Stream files through Unstructured.io with chunk-size heuristics aligned to Phase 1 prompts.
5. **Error Handling:** Integrate resilient retry strategy referencing modOpt modular patterns for configurable backoff.
6. **Metrics Hook:** Emit ingestion duration and throughput to telemetry baseline.

## Testing Strategy
- **Unit Tests:** Change detection (hash diff), ignore rules, metadata serialization.
- **Integration Tests:** End-to-end crawl against fixture directories (nested, binary, large files) with golden outputs.
- **Performance Tests:** Benchmark ingestion throughput on M-series reference hardware.
- **Security Tests:** Validate permission errors handled gracefully without privilege escalation.

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


