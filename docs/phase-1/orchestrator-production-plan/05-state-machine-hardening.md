Summary: Harden job lifecycle state machine with comprehensive validation, idempotent operations, and atomic transitions.

# 05 · State Machine Hardening

## Purpose
Ensure the job lifecycle state machine operates with deterministic transitions, idempotent operations, and comprehensive validation to prevent state corruption and provide reliable crash recovery. Guarantees the Ghost's experiential learning pipeline maintains consistent job state across all failure scenarios.

## Scope
- Comprehensive state transition validation
- Idempotent state operations (safe to retry)
- Atomic state updates with transaction guarantees
- Deadlock detection and recovery
- State corruption detection and repair
- State machine invariant checking
- Audit trail for all state transitions
- State validation on database load

## Requirements Alignment
- **Job State Machine**: "Model lifecycle using automata-based programming to ensure deterministic transitions and debuggability" (implementation guide)
- **Fault Tolerance**: Reliable state management for crash recovery
- **Observability**: State audit trail for debugging
- **Correctness**: Invariant checking prevents invalid states

## State Machine Specification

### Valid States
```python
class JobStatus(str, Enum):
    """Job lifecycle states."""
    PENDING = "pending"       # Queued, awaiting execution
    RUNNING = "running"       # Currently executing
    SUCCEEDED = "succeeded"   # Completed successfully
    FAILED = "failed"         # Failed (may retry)
    QUARANTINED = "quarantined"  # Failed permanently

### Valid Transitions
```python
VALID_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.PENDING: {
        JobStatus.RUNNING,    # Start execution
        JobStatus.PENDING,    # Reschedule (idempotent)
    },
    JobStatus.RUNNING: {
        JobStatus.SUCCEEDED,  # Complete successfully
        JobStatus.FAILED,     # Execution failed
        JobStatus.RUNNING,    # Update in-flight (idempotent)
    },
    JobStatus.FAILED: {
        JobStatus.PENDING,    # Retry
        JobStatus.QUARANTINED,  # Give up
        JobStatus.FAILED,     # Already failed (idempotent)
    },
    JobStatus.SUCCEEDED: {
        JobStatus.SUCCEEDED,  # Already succeeded (idempotent)
    },
    JobStatus.QUARANTINED: {
        JobStatus.PENDING,    # Manual retry from quarantine
        JobStatus.QUARANTINED,  # Already quarantined (idempotent)
    },
}
```

### State Transition Diagram
```
                    ┌─────────┐
                    │ PENDING │◄─────┐
                    └────┬────┘      │
                         │           │
                    start│           │retry
                         ▼           │
                    ┌─────────┐     │
              ┌─────┤ RUNNING ├─────┘
              │     └────┬────┘
         fail │          │ succeed
              │          │
              ▼          ▼
         ┌────────┐  ┌───────────┐
         │ FAILED │  │ SUCCEEDED │
         └───┬────┘  └───────────┘
             │
        quarantine
             │
             ▼
      ┌─────────────┐
      │ QUARANTINED │
      └─────────────┘
```

## Data Model

### State Transition Event
```python
@dataclass
class StateTransition:
    """Records a state transition attempt."""
    job_id: str
    from_status: JobStatus
    to_status: JobStatus
    timestamp: datetime
    operator: Optional[str] = None  # For manual transitions
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if transition is valid per state machine rules."""
        return self.to_status in VALID_TRANSITIONS.get(self.from_status, set())

    def is_idempotent(self) -> bool:
        """Check if transition is idempotent (same state)."""
        return self.from_status == self.to_status
```

### State Machine Invariants
```python
class StateMachineInvariants:
    """Defines invariants that must always hold."""

    @staticmethod
    def check_job_in_single_state(job_id: str, queue: JobQueue) -> bool:
        """A job must be in exactly one state."""
        # Query database to verify job appears in only one state
        pass

    @staticmethod
    def check_running_jobs_have_workers(orchestrator: IngestionOrchestrator) -> bool:
        """All RUNNING jobs must have active workers."""
        running_jobs = orchestrator._job_queue.snapshot(status=JobStatus.RUNNING)
        active_job_ids = set(orchestrator._active_jobs_map.keys())
        return all(job["job_id"] in active_job_ids for job in running_jobs)

    @staticmethod
    def check_succeeded_jobs_immutable(queue: JobQueue) -> bool:
        """SUCCEEDED jobs must not change state."""
        # Verify no succeeded jobs have been modified recently
        pass

    @staticmethod
    def check_attempts_monotonic(queue: JobQueue) -> bool:
        """Retry attempts must only increase."""
        # Verify attempts field never decreases
        pass
```

## Component Design

### StateMachineValidator
```python
class StateMachineValidator:
    """Validates and enforces state machine rules."""

    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self._audit = audit_logger
        self._transition_history: List[StateTransition] = []

    def validate_transition(
        self,
        job_id: str,
        from_status: JobStatus,
        to_status: JobStatus,
        *,
        operator: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> StateTransition:
        """Validate a state transition before execution."""
        transition = StateTransition(
            job_id=job_id,
            from_status=from_status,
            to_status=to_status,
            timestamp=datetime.utcnow(),
            operator=operator,
            reason=reason,
        )

        # Check if transition is valid
        if not transition.is_valid() and not transition.is_idempotent():
            logger.error(
                "Invalid state transition",
                extra={
                    "job_id": job_id,
                    "from_status": from_status.value,
                    "to_status": to_status.value,
                },
            )
            raise InvalidStateTransitionError(
                f"Invalid transition: {from_status.value} → {to_status.value}"
            )

        # Log idempotent transitions at debug level
        if transition.is_idempotent():
            logger.debug(
                "Idempotent state transition",
                extra={
                    "job_id": job_id,
                    "status": from_status.value,
                },
            )

        # Record transition
        self._transition_history.append(transition)

        # Audit log
        if self._audit:
            self._audit.record(
                AuditEvent(
                    job_id=job_id,
                    source="state_machine",
                    action=f"transition_{from_status.value}_to_{to_status.value}",
                    status="validated",
                    timestamp=transition.timestamp,
                    metadata={
                        "from_status": from_status.value,
                        "to_status": to_status.value,
                        "operator": operator,
                        "reason": reason,
                    },
                )
            )

        return transition

    def check_invariants(
        self,
        orchestrator: IngestionOrchestrator,
    ) -> List[str]:
        """Check all state machine invariants."""
        violations = []

        try:
            if not StateMachineInvariants.check_running_jobs_have_workers(orchestrator):
                violations.append("RUNNING jobs without active workers detected")
        except Exception as exc:
            violations.append(f"Invariant check failed: {exc}")

        return violations
```

### IdempotentJobQueue
```python
class IdempotentJobQueue(JobQueue):
    """Enhanced JobQueue with idempotent operations."""

    def __init__(self, path: Path, validator: Optional[StateMachineValidator] = None):
        super().__init__(path)
        self._validator = validator or StateMachineValidator()

    def mark_running(self, job_id: str, *, idempotent: bool = True) -> None:
        """Idempotently mark job as running."""
        with self._lock:
            # Get current status
            current_status = self._get_status(job_id)

            # Validate transition
            try:
                self._validator.validate_transition(
                    job_id=job_id,
                    from_status=current_status,
                    to_status=JobStatus.RUNNING,
                )
            except InvalidStateTransitionError:
                if not idempotent:
                    raise
                logger.warning(
                    "Skipping invalid transition (idempotent mode)",
                    extra={"job_id": job_id},
                )
                return

            # Already running - idempotent
            if current_status == JobStatus.RUNNING and idempotent:
                return

            # Execute state transition atomically
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'running',
                        attempts = attempts + 1,
                        updated_at = ?
                    WHERE job_id = ? AND status = ?
                    """,
                    (datetime.utcnow().isoformat(), job_id, current_status.value),
                )

                # Verify transition succeeded
                if self._conn.total_changes == 0:
                    raise StateTransitionRaceError(
                        f"Job {job_id} changed state during transition"
                    )

    def mark_completed(self, job_id: str, *, idempotent: bool = True) -> None:
        """Idempotently mark job as completed."""
        with self._lock:
            current_status = self._get_status(job_id)

            # Validate transition
            try:
                self._validator.validate_transition(
                    job_id=job_id,
                    from_status=current_status,
                    to_status=JobStatus.SUCCEEDED,
                )
            except InvalidStateTransitionError:
                if not idempotent:
                    raise
                return

            # Already succeeded - idempotent
            if current_status == JobStatus.SUCCEEDED and idempotent:
                return

            # Only succeed from RUNNING state
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'succeeded',
                        updated_at = ?
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (datetime.utcnow().isoformat(), job_id),
                )

                if self._conn.total_changes == 0 and not idempotent:
                    raise StateTransitionRaceError(
                        f"Job {job_id} not in RUNNING state"
                    )

    def mark_failed(self, job_id: str, *, idempotent: bool = True) -> None:
        """Idempotently mark job as failed."""
        with self._lock:
            current_status = self._get_status(job_id)

            # Validate transition
            try:
                self._validator.validate_transition(
                    job_id=job_id,
                    from_status=current_status,
                    to_status=JobStatus.FAILED,
                )
            except InvalidStateTransitionError:
                if not idempotent:
                    raise
                return

            # Already failed - idempotent
            if current_status == JobStatus.FAILED and idempotent:
                return

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        updated_at = ?
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (datetime.utcnow().isoformat(), job_id),
                )

    def _get_status(self, job_id: str) -> JobStatus:
        """Get current status of a job."""
        cur = self._conn.cursor()
        cur.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
        row = cur.fetchone()
        if not row:
            raise JobNotFoundError(f"Job {job_id} not found")
        return JobStatus(row[0])
```

### Deadlock Detection
```python
class DeadlockDetector:
    """Detects and resolves job processing deadlocks."""

    def __init__(self, queue: JobQueue, timeout_seconds: int = 600):
        self._queue = queue
        self._timeout_seconds = timeout_seconds

    def detect_stalled_jobs(self) -> List[str]:
        """Detect jobs stuck in RUNNING state."""
        stalled = []
        running_jobs = self._queue.snapshot(status=JobStatus.RUNNING)

        for job in running_jobs:
            updated_at = datetime.fromisoformat(job["updated_at"])
            age = (datetime.utcnow() - updated_at).total_seconds()

            if age > self._timeout_seconds:
                stalled.append(job["job_id"])
                logger.warning(
                    "Detected stalled job",
                    extra={
                        "job_id": job["job_id"],
                        "age_seconds": age,
                    },
                )

        return stalled

    def recover_stalled_job(self, job_id: str) -> None:
        """Recover a stalled job by resetting to PENDING."""
        logger.info(
            "Recovering stalled job",
            extra={"job_id": job_id},
        )

        # Reset to PENDING for retry
        self._queue.mark_failed(job_id)
        self._queue.reschedule(job_id, retry_delay_seconds=60)
```

## Acceptance Criteria

- ✅ StateMachineValidator enforces valid transitions
- ✅ Invalid transitions raise InvalidStateTransitionError
- ✅ Idempotent transitions (same-state) are allowed
- ✅ All state transitions logged to audit trail
- ✅ Atomic state updates with SQLite transactions
- ✅ Optimistic locking prevents race conditions
- ✅ State invariants checked periodically
- ✅ Deadlock detector identifies stalled jobs
- ✅ Stalled job recovery mechanism implemented
- ✅ State corruption detection on database load
- ✅ Comprehensive unit tests for all transitions
- ✅ Property-based testing for state machine invariants

## Test Plan

### Unit Tests
- `test_valid_transitions.py`: All valid transitions succeed
- `test_invalid_transitions.py`: Invalid transitions rejected
- `test_idempotent_transitions.py`: Same-state transitions allowed
- `test_atomic_updates.py`: Transaction guarantees
- `test_optimistic_locking.py`: Race condition prevention

### State Machine Tests
- `test_state_machine_exhaustive.py`: Test all state combinations
- `test_concurrent_transitions.py`: Thread-safe state updates
- `test_transition_audit_trail.py`: Audit logging complete

### Invariant Tests
- `test_invariant_single_state.py`: Job in exactly one state
- `test_invariant_running_workers.py`: RUNNING jobs have workers
- `test_invariant_succeeded_immutable.py`: SUCCEEDED jobs don't change
- `test_invariant_attempts_monotonic.py`: Attempts only increase

### Deadlock Tests
- `test_stalled_job_detection.py`: Detect jobs stuck in RUNNING
- `test_stalled_job_recovery.py`: Reset stalled jobs to PENDING
- `test_deadlock_timeout_tuning.py`: Configurable timeout

### Property-Based Tests
```python
from hypothesis import given, strategies as st

@given(
    from_status=st.sampled_from(list(JobStatus)),
    to_status=st.sampled_from(list(JobStatus)),
)
def test_transition_determinism(from_status: JobStatus, to_status: JobStatus):
    """State transitions are deterministic."""
    validator = StateMachineValidator()

    try:
        transition = validator.validate_transition(
            job_id="test",
            from_status=from_status,
            to_status=to_status,
        )
        # If allowed, it's in VALID_TRANSITIONS
        assert transition.is_valid() or transition.is_idempotent()
    except InvalidStateTransitionError:
        # If rejected, it's not in VALID_TRANSITIONS
        assert to_status not in VALID_TRANSITIONS.get(from_status, set())
```

## Implementation Notes

### Atomic State Updates with SQLite
```sql
-- Use WHERE clause to implement optimistic locking
UPDATE jobs
SET status = 'running',
    attempts = attempts + 1,
    updated_at = ?
WHERE job_id = ? AND status = 'pending';

-- Check affected rows to detect race conditions
-- If 0 rows updated, status changed concurrently
```

### State Machine Testing Strategy
1. **Unit tests**: Individual transition validation
2. **Integration tests**: End-to-end job lifecycle
3. **Property-based tests**: Invariant checking
4. **Fuzz testing**: Random transition sequences
5. **Concurrency tests**: Parallel transitions

### State Corruption Detection
```python
def validate_database_integrity(queue: JobQueue) -> List[str]:
    """Check database for state corruption."""
    issues = []

    # Check for duplicate job IDs
    duplicate_jobs = queue._conn.execute("""
        SELECT job_id, COUNT(*)
        FROM jobs
        GROUP BY job_id
        HAVING COUNT(*) > 1
    """).fetchall()
    if duplicate_jobs:
        issues.append(f"Duplicate job IDs: {duplicate_jobs}")

    # Check for invalid status values
    invalid_statuses = queue._conn.execute("""
        SELECT job_id, status
        FROM jobs
        WHERE status NOT IN ('pending', 'running', 'succeeded', 'failed', 'quarantined')
    """).fetchall()
    if invalid_statuses:
        issues.append(f"Invalid statuses: {invalid_statuses}")

    # Check for negative attempts
    negative_attempts = queue._conn.execute("""
        SELECT job_id, attempts
        FROM jobs
        WHERE attempts < 0
    """).fetchall()
    if negative_attempts:
        issues.append(f"Negative attempts: {negative_attempts}")

    return issues
```

## Open Questions

- Should we implement a state machine visualization tool for debugging?
- How to handle manual operator overrides of state (force transitions)?
- Should we support state snapshots for rollback?
- What's the appropriate timeout for deadlock detection (10min/30min)?
- Should stalled job recovery be automatic or require operator approval?
- How to handle state machine schema evolution (adding new states)?
- Should we implement state machine metrics (transition counts, timing)?

## Dependencies

- SQLite for atomic transaction support
- AuditLogger for transition logging
- Existing JobQueue implementation
- Hypothesis library for property-based testing


