Summary: Describes the ingestion orchestrator feature with scope, testing, and review checkpoints.

# Feature · Ingestion Orchestrator

## Goal
Provide a unified ingestion orchestrator that schedules connector jobs, manages retries, and ensures coherent propagation from raw data to normalized documents under privacy and performance constraints.

## Success Criteria
- Central job queue handles ingestion tasks across connectors with priority rules.
- Scheduler configurable for manual runs, cron-like schedules, and file-system events.
- Retry/backoff configurable per connector with visibility into failure states.
- Job telemetry captured for throughput, latency, and error rates.
- Orchestrator integrates with audit logging for traceability.

## Functional Scope
- Job definition schema standardizing connector interfaces.
- Worker pool managing connector execution with resource caps.
- State machine tracking job lifecycle (pending → running → succeeded/failed → retry/quarantine).
- Health dashboard surface for operators.
- Hooks for downstream pipeline triggers (normalization, extraction).

## Non-Functional Guarantees
- On-device first; no external queue dependencies.
- Fault tolerance via persistent queue (SQLite or LiteFS) with crash recovery.
- Concurrency tuning respecting system resource ceilings.
- Observability via local dashboard and log exports.

## Dependencies
- Connector implementations (local files, Obsidian, IMAP, GitHub).
- Telemetry baseline ([feature-performance-telemetry](feature-performance-telemetry.md)).
- Privacy audit logging foundation ([feature-privacy-audit-logging](feature-privacy-audit-logging.md)).

## Implementation Guide
1. **Queue Engine:** Build lightweight persistent queue leveraging SQLite; apply modOpt modular components for retry strategies.
2. **Job State Machine:** Model lifecycle using automata-based programming to ensure deterministic transitions and debuggability.
3. **Scheduler Interfaces:** Support cron expressions, manual triggers, and event-based scheduling (e.g., FSEvents) per connector.
4. **Worker Execution:** Use async runtime tuned for CPU/I/O balance; allow connectors to declare resource usage profiles.
5. **Telemetry Hooks:** Instrument job duration, success/failure counts, queue depth; export to telemetry baseline.
6. **Operator Console:** Provide CLI and minimal UI (desktop shell stub) for monitoring, pause/resume, and manual retries.

## Testing Strategy
- **Unit Tests:** State machine transitions, retry policies, persistence layer crash recovery.
- **Integration Tests:** Connector job execution end-to-end with induced failures.
- **Load Tests:** Stress with concurrent connector runs to validate fairness and resource allocation.
- **Resilience Tests:** Simulate abrupt shutdowns to ensure job rehydration.

## Code Review Checklist
- State transitions exhaustive and logged.
- Retry policies configurable and measured.
- Telemetry instrumentation covers key metrics.
- Persistence resilient to corruption; backups documented.
- Security considerations align with privacy logging requirements.

## Documentation & Follow-up
- Update operator runbooks with orchestrator usage.
- Share telemetry schema with analytics team for Phase 2 insights.
- Integrate lessons into future ingestion expansions (cloud drives, RSS).
- Implementation reference: `src/futurnal/orchestrator/` (queue + scheduler) with coverage in `tests/orchestrator/`.
- Telemetry recorder stub: `src/futurnal/orchestrator/metrics.py` captures job durations for baseline metrics.


