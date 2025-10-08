Summary: Roadmap and task breakdown to bring the Ingestion Orchestrator to production readiness for Ghost experiential learning pipeline.

# Ingestion Orchestrator · Production Plan

This folder tracks the work required to ship Feature 5 (Ingestion Orchestrator) with production-quality stability, observability, and operational excellence—ensuring the Ghost's experiential learning pipeline operates with fault tolerance, performance, and operator visibility. Each task ensures the orchestrator not only schedules and executes connector jobs but provides comprehensive observability, resilience, and control for production operations. Task documents define scope, acceptance criteria, test plans, and operational guidance aligned to the experiential learning architecture in [system-architecture.md](../../architecture/system-architecture.md).

## Task Index
- [01-quarantine-system.md](01-quarantine-system.md)
- [02-per-connector-retry-policies.md](02-per-connector-retry-policies.md)
- [03-resource-usage-profiling.md](03-resource-usage-profiling.md)
- [04-operator-cli-commands.md](04-operator-cli-commands.md)
- [05-state-machine-hardening.md](05-state-machine-hardening.md)
- [06-crash-recovery-durability.md](06-crash-recovery-durability.md)
- [07-load-fairness-testing.md](07-load-fairness-testing.md)
- [08-integration-test-suite.md](08-integration-test-suite.md)
- [09-configuration-security.md](09-configuration-security.md)
- [10-operator-runbooks.md](10-operator-runbooks.md)

## Technical Foundation

### Queue Engine
**SQLite with WAL mode** - Persistent, crash-recoverable job queue
- ACID transactions with write-ahead logging
- Automatic crash recovery on database open after crash
- Thread-safe operations with connection-level locking
- Priority-based job ordering with scheduled execution support
- Sub-second fetch performance for pending jobs

### Scheduler
**APScheduler** (AsyncIOScheduler) - Context7 Trust Score 9.3
- Cron expressions for periodic scheduling (e.g., "0 * * * *" for hourly)
- Interval triggers for fixed-rate execution (e.g., every 5 minutes)
- Manual triggers for on-demand execution
- File system watcher integration for event-driven scheduling
- Job persistence across scheduler restarts
- Misfire handling and grace periods

### Worker Pool
**Async/await with semaphore** - Resource-controlled execution
- Configurable concurrency limits per hardware capabilities
- Hardware-aware worker caps (max 8 workers, adaptive to CPU count)
- Per-connector resource profiles for fine-grained control
- Fair scheduling across sources with priority support
- Non-blocking execution with async/await patterns

### State Machine
**Job lifecycle tracking** - Deterministic state transitions
- `pending` → `running` → `succeeded`/`failed`
- Retry with exponential backoff for transient failures
- Quarantine for persistent failures after max retries
- Audit trail for all state transitions
- Idempotent state operations for crash safety

### Telemetry & Observability
**TelemetryRecorder** - Comprehensive metrics collection
- Per-job duration, throughput (bytes/sec), and file counts
- Queue depth tracking and worker utilization
- Success/failure counts with error categorization
- Rolling telemetry summaries for operator dashboards
- Privacy-aware metrics (no sensitive data in telemetry)

## Architectural Patterns

Following established patterns from connector implementations:

1. **Persistent Queue Pattern**
   - SQLite-backed JobQueue with WAL journaling mode
   - Crash-recoverable state persistence (survives process crashes)
   - Priority-based job ordering (HIGH > NORMAL > LOW)
   - Scheduled job support with future execution times
   - Snapshot API for operator inspection

2. **Privacy-First Design**
   - Audit logging for all orchestrator operations
   - No sensitive data in logs (paths redacted via RedactionPolicy)
   - Consent-aware execution (check ConsentRegistry before ingestion)
   - Privacy audit events for job lifecycle

3. **Fault Tolerance**
   - Automatic retry with exponential backoff (60s base delay)
   - Quarantine for persistent failures (after 3 attempts)
   - Graceful degradation on connector failures
   - State consistency guarantees via SQLite transactions
   - Job rehydration on orchestrator restart

4. **Observability**
   - Comprehensive telemetry (job duration, throughput, queue depth)
   - Health checks for all subsystems (disk, state store, Neo4j, ChromaDB, IMAP, GitHub)
   - Operator CLI for job and queue inspection
   - Audit trail for all operations with structured logging

5. **Multi-Connector Integration**
   - Unified interface for Local Files, Obsidian, IMAP, GitHub connectors
   - Job type discrimination via JobType enum
   - Connector-specific payload schemas
   - Pluggable ElementSink for pipeline integration

## Current Implementation Status

### ✅ Implemented
- SQLite-backed JobQueue with WAL mode and priority ordering
- Basic state machine (pending/running/succeeded/failed)
- APScheduler integration (cron, interval, manual, watcher triggers)
- Async worker pool with semaphore-based concurrency control
- Basic retry mechanism with exponential backoff
- TelemetryRecorder capturing job metrics
- Audit logging integration via privacy module
- Health checks for all subsystems
- Multi-connector support (Local Files, Obsidian, IMAP, GitHub)
- Source registration with schedule configuration
- Debouncing for file watcher events

### ❌ Production Gaps
- **Quarantine system**: Mentioned in state machine but not fully implemented
- **Per-connector retry policies**: Currently uses global retry settings
- **Resource usage profiling**: No connector-specific resource declarations
- **Operator CLI commands**: Missing job management and queue inspection commands
- **State machine hardening**: Needs comprehensive validation and idempotency tests
- **Crash recovery validation**: SQLite persistence exists but not tested under failures
- **Load and fairness testing**: No validation of priority ordering under load
- **Integration test suite**: Limited end-to-end pipeline tests
- **Configuration security**: No validation schema or secure defaults
- **Operator documentation**: No runbooks or troubleshooting guides

## AI Learning Focus

Transform orchestration into experiential learning enabler:

- **Reliable Ingestion**: Ensure no experiential data loss through fault-tolerant queueing and crash recovery
- **Fair Scheduling**: Balance learning across multiple experiential sources (files, notes, emails, code) without starvation
- **Resource Management**: Optimize learning pipeline throughput to maximize experiential data processing
- **Visibility**: Provide operators insight into learning progress, bottlenecks, and failures for continuous improvement

The orchestrator is the Ghost's heartbeat—coordinating how experiential data flows from diverse sources into the unified PKG and vector stores that power personalized intelligence.

## Usage

- Update these plans as tasks progress; each file captures scope, deliverables, and open questions.
- Cross-link implementation PRs and test evidence directly inside the relevant markdown files.
- When a task reaches completion, summarize learnings and move any follow-up work to appropriate phase-2 documents.


