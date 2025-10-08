Summary: Validate SQLite WAL-based crash recovery and implement comprehensive durability testing for job queue persistence.

# 06 · Crash Recovery & Durability

## Purpose
Validate that the orchestrator's SQLite-backed job queue survives process crashes, power failures, and abrupt shutdowns without data loss or corruption. Ensure the Ghost's experiential learning pipeline maintains job state durability and automatically recovers from failures.

## Scope
- SQLite WAL checkpoint validation
- Job rehydration on orchestrator restart
- In-flight job recovery logic
- Power-loss simulation testing
- State consistency verification
- Database corruption detection and recovery
- Crash recovery performance benchmarking
- Operator documentation for disaster recovery

## Requirements Alignment
- **Fault Tolerance**: "Persistent queue (SQLite or LiteFS) with crash recovery" (non-functional requirement)
- **Durability**: Jobs survive process crashes and system failures
- **Automatic Recovery**: "Rehydration on orchestrator restart"
- **No Data Loss**: All enqueued jobs persist across failures

## Technical Foundation

### SQLite WAL Mode
The JobQueue uses Write-Ahead Logging (WAL) for durability:
- **ACID Transactions**: Atomic, consistent, isolated, durable
- **Crash Recovery**: Automatic on database open
- **No Application Intervention**: Recovery handled by SQLite
- **Concurrent Access**: Readers don't block writers

### WAL Mechanics
```sql
-- Enable WAL mode (already done in JobQueue.__init__)
PRAGMA journal_mode=WAL;

-- Checkpoint modes
PRAGMA wal_checkpoint(PASSIVE);  -- Non-blocking checkpoint
PRAGMA wal_checkpoint(FULL);     -- Block until WAL is empty
PRAGMA wal_checkpoint(RESTART);  -- Block and reset WAL
PRAGMA wal_checkpoint(TRUNCATE); -- Block and truncate WAL to zero bytes
```

### Recovery Process
1. **On database open**: SQLite checks for incomplete transactions in WAL
2. **Automatic replay**: Uncommitted changes rolled forward
3. **Checkpoint**: WAL changes merged into main database file
4. **Consistency**: Database returns to consistent state

## Data Model

### Crash Recovery Metadata
```python
@dataclass
class CrashRecoveryReport:
    """Report of crash recovery process."""
    recovered_at: datetime
    jobs_recovered: int
    jobs_pending: int
    jobs_running_before_crash: int
    jobs_reset_to_pending: int
    wal_size_before_recovery_bytes: int
    recovery_duration_seconds: float
    errors: List[str] = field(default_factory=list)

    def was_successful(self) -> bool:
        """Check if recovery completed without errors."""
        return len(self.errors) == 0
```

### Recovery State Tracking
```python
class RecoveryStateTracker:
    """Tracks recovery state for resuming interrupted jobs."""

    def __init__(self, workspace_dir: Path):
        self._recovery_marker = workspace_dir / ".orchestrator_recovery"

    def mark_crash(self) -> None:
        """Mark that orchestrator crashed (for testing)."""
        self._recovery_marker.write_text(
            json.dumps({
                "crashed_at": datetime.utcnow().isoformat(),
                "pid": os.getpid(),
            })
        )

    def is_recovering_from_crash(self) -> bool:
        """Check if orchestrator is recovering from crash."""
        return self._recovery_marker.exists()

    def get_crash_info(self) -> Optional[Dict[str, Any]]:
        """Get crash metadata."""
        if not self._recovery_marker.exists():
            return None
        return json.loads(self._recovery_marker.read_text())

    def clear_recovery_marker(self) -> None:
        """Clear recovery marker after successful recovery."""
        if self._recovery_marker.exists():
            self._recovery_marker.unlink()
```

## Component Design

### CrashRecoveryManager
```python
class CrashRecoveryManager:
    """Manages crash recovery process for orchestrator."""

    def __init__(
        self,
        *,
        job_queue: JobQueue,
        workspace_dir: Path,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self._queue = job_queue
        self._workspace = workspace_dir
        self._audit = audit_logger
        self._recovery_tracker = RecoveryStateTracker(workspace_dir)

    def recover_from_crash(self) -> CrashRecoveryReport:
        """Execute crash recovery procedure."""
        start_time = time.perf_counter()

        logger.info("Starting crash recovery")

        # Get crash info
        crash_info = self._recovery_tracker.get_crash_info()
        crashed_at = crash_info.get("crashed_at") if crash_info else None

        # Check WAL size
        wal_path = Path(str(self._queue._path) + "-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0

        logger.info(
            "WAL file status",
            extra={"wal_size_bytes": wal_size},
        )

        # Count jobs in various states
        all_jobs = self._queue.snapshot()
        jobs_pending = len([j for j in all_jobs if j["status"] == "pending"])
        jobs_running = len([j for j in all_jobs if j["status"] == "running"])

        # Reset RUNNING jobs to PENDING (they were interrupted)
        jobs_reset = self._reset_interrupted_jobs()

        # Re-count after reset
        jobs_recovered = len(self._queue.snapshot())

        # Checkpoint WAL to ensure all changes persisted
        self._checkpoint_wal()

        # Verify database integrity
        integrity_errors = self._verify_database_integrity()

        duration = time.perf_counter() - start_time

        report = CrashRecoveryReport(
            recovered_at=datetime.utcnow(),
            jobs_recovered=jobs_recovered,
            jobs_pending=jobs_pending + jobs_reset,
            jobs_running_before_crash=jobs_running,
            jobs_reset_to_pending=jobs_reset,
            wal_size_before_recovery_bytes=wal_size,
            recovery_duration_seconds=duration,
            errors=integrity_errors,
        )

        # Clear recovery marker
        self._recovery_tracker.clear_recovery_marker()

        # Audit log
        if self._audit:
            self._audit.record(
                AuditEvent(
                    job_id=f"recovery_{datetime.utcnow().isoformat()}",
                    source="crash_recovery",
                    action="recover_from_crash",
                    status="succeeded" if report.was_successful() else "failed",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "jobs_recovered": jobs_recovered,
                        "jobs_reset": jobs_reset,
                        "duration_seconds": duration,
                        "crashed_at": crashed_at,
                    },
                )
            )

        logger.info(
            "Crash recovery completed",
            extra={
                "jobs_recovered": jobs_recovered,
                "jobs_reset": jobs_reset,
                "duration_seconds": duration,
            },
        )

        return report

    def _reset_interrupted_jobs(self) -> int:
        """Reset RUNNING jobs to PENDING for retry."""
        running_jobs = self._queue.snapshot(status=JobStatus.RUNNING)

        for job in running_jobs:
            logger.info(
                "Resetting interrupted job",
                extra={"job_id": job["job_id"]},
            )
            # Reset to pending with no delay
            self._queue.mark_failed(job["job_id"])
            self._queue.reschedule(job["job_id"], retry_delay_seconds=0)

        return len(running_jobs)

    def _checkpoint_wal(self) -> None:
        """Force WAL checkpoint to persist all changes."""
        try:
            self._queue._conn.execute("PRAGMA wal_checkpoint(FULL)")
            logger.info("WAL checkpoint completed")
        except Exception as exc:
            logger.error("WAL checkpoint failed", exc_info=exc)

    def _verify_database_integrity(self) -> List[str]:
        """Verify database integrity after recovery."""
        errors = []

        try:
            # SQLite integrity check
            result = self._queue._conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] != "ok":
                errors.append(f"Integrity check failed: {result[0]}")

            # Foreign key check
            result = self._queue._conn.execute("PRAGMA foreign_key_check").fetchall()
            if result:
                errors.append(f"Foreign key violations: {result}")

        except Exception as exc:
            errors.append(f"Integrity verification failed: {exc}")

        return errors
```

### Integration with IngestionOrchestrator
```python
class IngestionOrchestrator:
    def __init__(
        self,
        *,
        job_queue: JobQueue,
        crash_recovery: Optional[CrashRecoveryManager] = None,
        # ... existing params
    ) -> None:
        self._crash_recovery = crash_recovery or CrashRecoveryManager(
            job_queue=job_queue,
            workspace_dir=self._workspace_dir,
            audit_logger=self._audit_logger,
        )
        # ... existing initialization

        # Check if recovering from crash
        if self._crash_recovery._recovery_tracker.is_recovering_from_crash():
            recovery_report = self._crash_recovery.recover_from_crash()

            if not recovery_report.was_successful():
                logger.error(
                    "Crash recovery completed with errors",
                    extra={"errors": recovery_report.errors},
                )

    def start(self) -> None:
        """Enhanced start with crash recovery marker."""
        if self._running:
            return

        # Set recovery marker (cleared on graceful shutdown)
        self._crash_recovery._recovery_tracker.mark_crash()

        # ... existing start logic

        self._running = True
        logger.info("Ingestion orchestrator started")

    async def shutdown(self) -> None:
        """Enhanced shutdown to clear recovery marker."""
        if not self._running:
            return

        # ... existing shutdown logic

        # Clear recovery marker (graceful shutdown)
        self._crash_recovery._recovery_tracker.clear_recovery_marker()

        self._running = False
        logger.info("Ingestion orchestrator stopped")
```

## Testing Strategy

### Crash Simulation
```python
class CrashSimulator:
    """Simulates various crash scenarios for testing."""

    @staticmethod
    def simulate_sigkill(orchestrator: IngestionOrchestrator) -> None:
        """Simulate SIGKILL (immediate termination)."""
        import signal
        os.kill(os.getpid(), signal.SIGKILL)

    @staticmethod
    def simulate_sigterm(orchestrator: IngestionOrchestrator) -> None:
        """Simulate SIGTERM (graceful shutdown)."""
        import signal
        os.kill(os.getpid(), signal.SIGTERM)

    @staticmethod
    def simulate_power_loss(orchestrator: IngestionOrchestrator) -> None:
        """Simulate power loss (no cleanup, no checkpoint)."""
        # Close database without checkpoint
        orchestrator._job_queue._conn.close()
        sys.exit(1)

    @staticmethod
    def simulate_oom_kill(orchestrator: IngestionOrchestrator) -> None:
        """Simulate OOM killer (immediate termination)."""
        # Trigger out-of-memory
        import signal
        os.kill(os.getpid(), signal.SIGKILL)
```

### Recovery Validation
```python
def validate_recovery(
    jobs_before_crash: List[Dict[str, Any]],
    jobs_after_recovery: List[Dict[str, Any]],
) -> bool:
    """Validate that all jobs survived crash."""
    jobs_before_ids = {j["job_id"] for j in jobs_before_crash}
    jobs_after_ids = {j["job_id"] for j in jobs_after_recovery}

    # All jobs must be present
    if jobs_before_ids != jobs_after_ids:
        missing = jobs_before_ids - jobs_after_ids
        logger.error(f"Jobs lost during crash: {missing}")
        return False

    # RUNNING jobs should be reset to PENDING
    running_before = {
        j["job_id"] for j in jobs_before_crash
        if j["status"] == "running"
    }
    running_after = {
        j["job_id"] for j in jobs_after_recovery
        if j["status"] == "running"
    }

    if running_before.intersection(running_after):
        logger.error("RUNNING jobs not reset after crash")
        return False

    return True
```

## Acceptance Criteria

- ✅ SQLite WAL mode enabled and validated
- ✅ Jobs survive process crashes (SIGKILL)
- ✅ Jobs survive graceful shutdowns (SIGTERM)
- ✅ Jobs survive power-loss simulation
- ✅ RUNNING jobs reset to PENDING on recovery
- ✅ Database integrity verified after recovery
- ✅ WAL checkpointed during recovery
- ✅ Crash recovery telemetry recorded
- ✅ Audit events logged for recovery
- ✅ Recovery completes in <5 seconds for 10K jobs
- ✅ No jobs lost during crash
- ✅ Operator documentation for disaster recovery

## Test Plan

### Unit Tests
- `test_wal_mode_enabled.py`: Verify WAL mode active
- `test_recovery_marker.py`: Recovery state tracking
- `test_reset_interrupted_jobs.py`: RUNNING → PENDING logic
- `test_database_integrity_check.py`: Integrity verification

### Integration Tests
- `test_crash_recovery_end_to_end.py`: Full recovery workflow
- `test_sigkill_recovery.py`: Recover from SIGKILL
- `test_power_loss_recovery.py`: Recover from power loss
- `test_graceful_shutdown.py`: No recovery needed for clean shutdown

### Resilience Tests
- `test_crash_during_job_enqueue.py`: Crash while enqueueing
- `test_crash_during_status_update.py`: Crash during state transition
- `test_crash_during_checkpoint.py`: Crash during WAL checkpoint
- `test_multiple_crashes.py`: Successive crashes and recoveries

### Performance Tests
- `test_recovery_performance_1k_jobs.py`: 1K jobs recovery time
- `test_recovery_performance_10k_jobs.py`: 10K jobs recovery time
- `test_recovery_performance_100k_jobs.py`: 100K jobs recovery time

### Database Corruption Tests
- `test_wal_corruption_detection.py`: Detect corrupted WAL file
- `test_database_corruption_recovery.py`: Recover from corruption
- `test_schema_migration_crash.py`: Crash during schema upgrade

## Implementation Notes

### SQLite Durability Guarantees
From SQLite documentation:
- **Application Crashes**: Transactions are durable across app crashes regardless of synchronous setting
- **Power Failures**: With `synchronous=NORMAL` and WAL mode, durability guaranteed across power failures after checkpoint
- **Recovery**: Automatic on database open, no application intervention needed

### WAL Checkpoint Frequency
```python
# Aggressive checkpointing for faster recovery
# Run after every N transactions
if self._transaction_count % 100 == 0:
    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
```

### Recovery Performance Optimization
```python
# Use indexes for faster recovery queries
CREATE INDEX IF NOT EXISTS idx_jobs_status_updated
ON jobs(status, updated_at);

# This speeds up finding RUNNING jobs during recovery
```

### Disaster Recovery Procedure
1. **Stop orchestrator**: `futurnal orchestrator stop`
2. **Backup database**: `cp queue.db queue.db.backup`
3. **Verify backup**: `sqlite3 queue.db.backup "PRAGMA integrity_check"`
4. **Start orchestrator**: `futurnal orchestrator start`
5. **Check recovery log**: `futurnal orchestrator health`

## Open Questions

- Should we support automatic backups before risky operations?
- How to handle database corruption beyond SQLite's repair capabilities?
- Should recovery be configurable (automatic vs. manual approval)?
- What's the appropriate WAL checkpoint frequency for production?
- Should we implement a "recovery mode" with reduced functionality?
- How to handle partial state corruption (some jobs lost)?
- Should we support point-in-time recovery from backups?

## Dependencies

- SQLite3 with WAL support
- JobQueue implementation
- AuditLogger for recovery events
- Telemetry for recovery metrics
- Pytest for crash simulation tests


