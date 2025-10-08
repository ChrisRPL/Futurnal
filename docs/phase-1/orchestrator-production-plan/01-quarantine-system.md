Summary: Implement comprehensive quarantine workflow for failed ingestion jobs with manual recovery operations and visibility into failure patterns.

# 01 · Quarantine System Implementation

## Purpose
Provide a robust quarantine system that isolates persistently failing jobs, classifies failure reasons, enables manual recovery operations, and gives operators visibility into failure patterns. Ensures the Ghost's experiential learning pipeline degrades gracefully rather than halting completely when encountering problematic data sources.

## Scope
- QuarantineStore with SQLite persistence for failed job metadata
- Failure classification taxonomy (parse errors, permissions, resources, timeouts, etc.)
- Automatic quarantine transition after max retry attempts exceeded
- Manual recovery operations (inspect, retry, purge)
- CLI commands for quarantine management
- Telemetry metrics for quarantine patterns
- Audit events for quarantine lifecycle operations
- Integration with existing retry mechanism

## Requirements Alignment
- **Fault Tolerance**: Graceful degradation via quarantine instead of pipeline blockage
- **State Machine**: Extends pending → running → failed → quarantine path
- **Observability**: Quarantine metrics expose failure patterns for operator intervention
- **Privacy-First**: No sensitive data in quarantine logs (redacted paths)
- **Reliability**: Failed jobs don't block other connectors or sources

## Data Model

### QuarantineReason Taxonomy
```python
class QuarantineReason(str, Enum):
    """Classification of why a job was quarantined."""
    PARSE_ERROR = "parse_error"              # Unstructured.io parsing failed
    PERMISSION_DENIED = "permission_denied"  # File/folder access denied
    RESOURCE_EXHAUSTED = "resource_exhausted" # Out of memory/disk
    CONNECTOR_ERROR = "connector_error"      # Connector-specific failure
    TIMEOUT = "timeout"                      # Job exceeded time limit
    INVALID_STATE = "invalid_state"          # State store corruption
    DEPENDENCY_FAILURE = "dependency_failure" # Neo4j/ChromaDB unavailable
    UNKNOWN = "unknown"                      # Uncategorized failure
```

### QuarantinedJob Schema
```python
class QuarantinedJob(BaseModel):
    """Represents a job in quarantine with recovery metadata."""
    job_id: str                               # Original job ID
    original_job: IngestionJob                # Full job details for retry
    reason: QuarantineReason                  # Why quarantined
    error_message: str                        # Exception message (redacted)
    error_traceback: Optional[str] = None     # Stack trace for debugging
    quarantined_at: datetime                  # When quarantined
    retry_count: int                          # Number of manual retry attempts
    last_retry_at: Optional[datetime] = None  # Last manual retry timestamp
    can_retry: bool = True                    # Whether retry is allowed
    operator_notes: Optional[str] = None      # Manual operator annotations
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Context
```

### QuarantineStore Implementation
```python
class QuarantineStore:
    """SQLite-backed persistent storage for quarantined jobs."""

    def __init__(self, path: Path) -> None:
        """Initialize quarantine store with SQLite database."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(QUARANTINE_SCHEMA)
        self._lock = threading.Lock()

    def quarantine(
        self,
        *,
        job: IngestionJob,
        reason: QuarantineReason,
        error_message: str,
        error_traceback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QuarantinedJob:
        """Move a failed job into quarantine."""

    def list(
        self,
        *,
        reason: Optional[QuarantineReason] = None,
        limit: Optional[int] = None,
    ) -> List[QuarantinedJob]:
        """List quarantined jobs with optional filtering."""

    def get(self, job_id: str) -> Optional[QuarantinedJob]:
        """Retrieve specific quarantined job by ID."""

    def mark_retry_attempted(
        self,
        job_id: str,
        *,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a manual retry attempt."""

    def remove(self, job_id: str) -> None:
        """Remove job from quarantine (after successful recovery or purge)."""

    def purge_old(self, days: int = 30) -> int:
        """Remove quarantined jobs older than specified days."""

    def statistics(self) -> Dict[str, Any]:
        """Compute quarantine statistics by reason."""
```

### SQLite Schema
```sql
CREATE TABLE IF NOT EXISTS quarantined_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    original_payload TEXT NOT NULL,
    reason TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_traceback TEXT,
    quarantined_at TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_retry_at TEXT,
    can_retry INTEGER NOT NULL DEFAULT 1,
    operator_notes TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quarantine_reason ON quarantined_jobs(reason);
CREATE INDEX IF NOT EXISTS idx_quarantine_timestamp ON quarantined_jobs(quarantined_at);
```

## Component Design

### Integration with IngestionOrchestrator
```python
class IngestionOrchestrator:
    def __init__(
        self,
        *,
        job_queue: JobQueue,
        quarantine_store: QuarantineStore,
        # ... existing params
    ) -> None:
        self._quarantine = quarantine_store
        # ... existing initialization

    async def _maybe_retry(self, job: IngestionJob) -> None:
        """Retry job or quarantine if max attempts exceeded."""
        record = DataRetryRecord(
            job_id=job.job_id,
            attempts=job.payload.get("attempts", 0)
        )

        if record.attempts >= self._max_retries:
            # Quarantine instead of abandoning
            await self._quarantine_job(job)
            return

        # Existing retry logic
        record.attempts += 1
        job.payload["attempts"] = record.attempts
        self._job_queue.reschedule(job.job_id, self._retry_backoff_seconds)

    async def _quarantine_job(self, job: IngestionJob) -> None:
        """Move failed job to quarantine with classification."""
        error_message = job.payload.get("error", "Unknown error")
        reason = self._classify_failure(error_message)

        quarantined = self._quarantine.quarantine(
            job=job,
            reason=reason,
            error_message=error_message,
            metadata={
                "source_name": job.payload.get("source_name"),
                "attempts": job.payload.get("attempts", 0),
            },
        )

        logger.error(
            "Job quarantined",
            extra={
                "ingestion_job_id": job.job_id,
                "quarantine_reason": reason.value,
                "ingestion_attempts": job.payload.get("attempts", 0),
            },
        )

        # Record audit event
        self._audit_logger.record(
            AuditEvent(
                job_id=job.job_id,
                source=job.payload.get("source_name", "unknown"),
                action="quarantine",
                status="quarantined",
                timestamp=datetime.utcnow(),
                metadata={
                    "reason": reason.value,
                    "attempts": job.payload.get("attempts", 0),
                },
            )
        )

        # Record telemetry
        if self._telemetry:
            self._telemetry.record(
                job_id=job.job_id,
                duration=0.0,
                status="quarantined",
                metadata={"reason": reason.value},
            )

    def _classify_failure(self, error_message: str) -> QuarantineReason:
        """Classify failure reason from error message."""
        error_lower = error_message.lower()

        if "permission denied" in error_lower or "access denied" in error_lower:
            return QuarantineReason.PERMISSION_DENIED
        elif "parse" in error_lower or "parsing" in error_lower:
            return QuarantineReason.PARSE_ERROR
        elif "memory" in error_lower or "disk" in error_lower:
            return QuarantineReason.RESOURCE_EXHAUSTED
        elif "timeout" in error_lower or "timed out" in error_lower:
            return QuarantineReason.TIMEOUT
        elif "neo4j" in error_lower or "chroma" in error_lower:
            return QuarantineReason.DEPENDENCY_FAILURE
        else:
            return QuarantineReason.UNKNOWN
```

## CLI Commands

### Quarantine List
```bash
futurnal orchestrator quarantine list [OPTIONS]

Options:
  --reason TEXT       Filter by quarantine reason
  --limit INTEGER     Limit results (default: 50)
  --format [table|json]  Output format (default: table)

Example:
  futurnal orchestrator quarantine list --reason parse_error
```

### Quarantine Show
```bash
futurnal orchestrator quarantine show JOB_ID

Displays detailed information about a quarantined job including:
- Original job configuration
- Error message and traceback
- Retry history
- Operator notes

Example:
  futurnal orchestrator quarantine show abc123-def456-789
```

### Quarantine Retry
```bash
futurnal orchestrator quarantine retry JOB_ID [OPTIONS]

Options:
  --force          Retry even if can_retry is False
  --note TEXT      Add operator note

Example:
  futurnal orchestrator quarantine retry abc123 --note "Fixed permissions"
```

### Quarantine Purge
```bash
futurnal orchestrator quarantine purge [OPTIONS]

Options:
  --older-than-days INTEGER  Remove jobs older than N days (default: 30)
  --reason TEXT              Purge specific reason
  --job-id TEXT              Purge specific job
  --all                      Purge all quarantined jobs
  --dry-run                  Show what would be purged

Example:
  futurnal orchestrator quarantine purge --older-than-days 90
```

## Acceptance Criteria

- ✅ QuarantineStore implemented with SQLite persistence
- ✅ Jobs automatically quarantined after exceeding max retry attempts
- ✅ Failure classification accurately categorizes common error types
- ✅ CLI command `quarantine list` displays quarantined jobs with filtering
- ✅ CLI command `quarantine show` displays detailed job information
- ✅ CLI command `quarantine retry` re-enqueues quarantined job
- ✅ CLI command `quarantine purge` removes old quarantined jobs
- ✅ Telemetry captures quarantine metrics (count by reason, age distribution)
- ✅ Audit events logged for quarantine, retry, and purge operations
- ✅ Quarantine doesn't block other jobs or connectors
- ✅ Quarantined job metadata includes redacted paths (privacy compliance)
- ✅ Manual retry updates retry_count and last_retry_at
- ✅ Statistics API provides quarantine patterns for operator dashboards

## Test Plan

### Unit Tests
- `test_quarantine_store_persistence.py`: SQLite CRUD operations
- `test_failure_classification.py`: Error message classification logic
- `test_quarantine_statistics.py`: Aggregation and filtering
- `test_quarantine_purge.py`: Old job removal logic
- `test_retry_tracking.py`: Retry counter updates

### Integration Tests
- `test_orchestrator_quarantine_integration.py`: End-to-end quarantine workflow
- `test_quarantine_after_max_retries.py`: Automatic quarantine trigger
- `test_manual_retry_from_quarantine.py`: Successful recovery flow
- `test_quarantine_telemetry_audit.py`: Telemetry and audit event generation

### Security Tests
- `test_quarantine_path_redaction.py`: Ensure paths redacted in logs
- `test_quarantine_error_sanitization.py`: No sensitive data in error messages

### Resilience Tests
- `test_quarantine_database_corruption.py`: Handle corrupted quarantine DB
- `test_concurrent_quarantine_operations.py`: Thread-safe quarantine access

## Implementation Notes

### Failure Classification Heuristics
Use pattern matching on exception types and messages:
```python
def _classify_failure(self, error_message: str, exception_type: Optional[Type[Exception]] = None) -> QuarantineReason:
    """Enhanced classification using exception type and message."""
    if exception_type:
        if issubclass(exception_type, PermissionError):
            return QuarantineReason.PERMISSION_DENIED
        elif issubclass(exception_type, MemoryError):
            return QuarantineReason.RESOURCE_EXHAUSTED
        elif issubclass(exception_type, TimeoutError):
            return QuarantineReason.TIMEOUT

    # Fallback to message pattern matching
    # ... existing logic
```

### Quarantine Metrics
Expose metrics for monitoring and alerting:
```python
{
    "total_quarantined": 42,
    "by_reason": {
        "parse_error": 15,
        "permission_denied": 10,
        "timeout": 8,
        "unknown": 9
    },
    "oldest_job_age_days": 12,
    "recent_quarantines_24h": 3,
    "retry_success_rate": 0.65
}
```

### Quarantine Dashboard Data
Provide data structure for operator UI/CLI dashboards:
```python
{
    "job_id": "abc-123",
    "source": "notes",
    "reason": "parse_error",
    "error": "Failed to parse markdown",
    "age_hours": 48,
    "retry_count": 2,
    "last_retry": "2 hours ago",
    "can_retry": true
}
```

## Open Questions

- Should quarantine have a TTL with automatic purge after N days?
- How to handle quarantine database corruption (fallback to file-based storage)?
- Should we support bulk retry operations (retry all parse_error jobs)?
- What's the appropriate default quarantine retention period (30/60/90 days)?
- Should quarantine statistics trigger alerts (e.g., >10 jobs with same reason)?
- How to export quarantine data for external analysis tools?
- Should operators be able to mark jobs as "do not retry" permanently?

## Dependencies

- Existing JobQueue for state machine integration
- AuditLogger for quarantine operation logging
- TelemetryRecorder for quarantine metrics
- CLI framework (Typer) for command implementation
- Privacy RedactionPolicy for path sanitization in error messages


