"""Tests for the quarantine system."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.quarantine import (
    QuarantineReason,
    QuarantineStore,
    QuarantinedJob,
    classify_failure,
)


def make_job(job_id: str, payload: dict | None = None) -> IngestionJob:
    """Helper to create a test ingestion job."""
    return IngestionJob(
        job_id=job_id,
        job_type=JobType.LOCAL_FILES,
        payload=payload or {"source_name": "docs", "attempts": 3},
        priority=JobPriority.NORMAL,
        scheduled_for=datetime.utcnow(),
    )


def test_quarantine_store_persistence(tmp_path: Path) -> None:
    """Test basic quarantine storage and retrieval."""
    store = QuarantineStore(tmp_path / "quarantine.db")
    job = make_job("job1")

    # Quarantine a job
    quarantined = store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Failed to parse document",
    )

    assert quarantined.job_id == "job1"
    assert quarantined.reason == QuarantineReason.PARSE_ERROR
    assert quarantined.retry_count == 0

    # Retrieve the job
    retrieved = store.get("job1")
    assert retrieved is not None
    assert retrieved.job_id == "job1"
    assert retrieved.reason == QuarantineReason.PARSE_ERROR


def test_quarantine_list_filtering(tmp_path: Path) -> None:
    """Test listing quarantined jobs with filtering."""
    store = QuarantineStore(tmp_path / "quarantine.db")

    # Quarantine multiple jobs with different reasons
    store.quarantine(
        job=make_job("job1"),
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
    )
    store.quarantine(
        job=make_job("job2"),
        reason=QuarantineReason.PERMISSION_DENIED,
        error_message="Permission denied",
    )
    store.quarantine(
        job=make_job("job3"),
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Another parse error",
    )

    # List all jobs
    all_jobs = store.list()
    assert len(all_jobs) == 3

    # Filter by reason
    parse_errors = store.list(reason=QuarantineReason.PARSE_ERROR)
    assert len(parse_errors) == 2
    assert all(j.reason == QuarantineReason.PARSE_ERROR for j in parse_errors)

    # Test limit
    limited = store.list(limit=1)
    assert len(limited) == 1


def test_failure_classification_all_reasons(tmp_path: Path) -> None:
    """Test that all failure reasons are correctly classified."""
    test_cases = [
        ("Permission denied: /private/file.txt", QuarantineReason.PERMISSION_DENIED),
        ("Access denied to resource", QuarantineReason.PERMISSION_DENIED),
        ("Failed to parse markdown file", QuarantineReason.PARSE_ERROR),
        ("Parsing error in document", QuarantineReason.PARSE_ERROR),
        ("Out of memory", QuarantineReason.RESOURCE_EXHAUSTED),
        ("Disk space exhausted", QuarantineReason.RESOURCE_EXHAUSTED),
        ("Operation timed out", QuarantineReason.TIMEOUT),
        ("Request timed out after 30s", QuarantineReason.TIMEOUT),
        ("Neo4j connection failed", QuarantineReason.DEPENDENCY_FAILURE),
        ("ChromaDB is unavailable", QuarantineReason.DEPENDENCY_FAILURE),
        ("Connector-specific error", QuarantineReason.CONNECTOR_ERROR),
        ("State corruption detected", QuarantineReason.INVALID_STATE),
        ("Unknown failure type", QuarantineReason.UNKNOWN),
    ]

    for error_msg, expected_reason in test_cases:
        reason = classify_failure(error_msg)
        assert reason == expected_reason, f"Failed for: {error_msg}"


def test_failure_classification_exception_types(tmp_path: Path) -> None:
    """Test classification using exception types."""
    assert classify_failure("error", PermissionError) == QuarantineReason.PERMISSION_DENIED
    assert classify_failure("error", MemoryError) == QuarantineReason.RESOURCE_EXHAUSTED
    assert classify_failure("error", TimeoutError) == QuarantineReason.TIMEOUT


def test_quarantine_statistics(tmp_path: Path) -> None:
    """Test statistics aggregation and metrics."""
    store = QuarantineStore(tmp_path / "quarantine.db")

    # Quarantine several jobs
    store.quarantine(
        job=make_job("job1"),
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
    )
    store.quarantine(
        job=make_job("job2"),
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
    )
    store.quarantine(
        job=make_job("job3"),
        reason=QuarantineReason.PERMISSION_DENIED,
        error_message="Permission denied",
    )

    stats = store.statistics()

    assert stats["total_quarantined"] == 3
    assert stats["by_reason"]["parse_error"] == 2
    assert stats["by_reason"]["permission_denied"] == 1
    assert stats["recent_quarantines_24h"] == 3
    assert stats["retry_success_rate"] == 0.0  # No retries yet


def test_quarantine_purge_old(tmp_path: Path) -> None:
    """Test removal of old quarantined jobs."""
    store = QuarantineStore(tmp_path / "quarantine.db")

    # Create a job with old quarantine time
    job = make_job("old_job")
    quarantined = store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Old error",
    )

    # Manually update quarantine time to be 60 days old
    with store._lock:
        old_time = (datetime.utcnow() - timedelta(days=60)).isoformat()
        with store._conn:
            store._conn.execute(
                "UPDATE quarantined_jobs SET quarantined_at = ? WHERE job_id = ?",
                (old_time, "old_job"),
            )

    # Create a recent job
    store.quarantine(
        job=make_job("new_job"),
        reason=QuarantineReason.PARSE_ERROR,
        error_message="New error",
    )

    # Purge jobs older than 30 days
    purged_count = store.purge_old(days=30)
    assert purged_count == 1

    # Verify only the new job remains
    remaining = store.list()
    assert len(remaining) == 1
    assert remaining[0].job_id == "new_job"


def test_retry_tracking(tmp_path: Path) -> None:
    """Test retry counter updates and tracking."""
    store = QuarantineStore(tmp_path / "quarantine.db")
    job = make_job("job1")

    # Quarantine the job
    store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
    )

    # Mark first retry as failed
    store.mark_retry_attempted("job1", success=False, error_message="Still failing")

    # Verify counters updated
    retrieved = store.get("job1")
    assert retrieved is not None
    assert retrieved.retry_count == 1
    assert retrieved.retry_failure_count == 1
    assert retrieved.retry_success_count == 0
    assert retrieved.last_retry_at is not None

    # Mark second retry as success (should remove from quarantine)
    store.mark_retry_attempted("job1", success=True)

    # Job should be removed
    assert store.get("job1") is None


def test_path_redaction(tmp_path: Path) -> None:
    """Test privacy-aware path redaction in error messages."""
    store = QuarantineStore(tmp_path / "quarantine.db")
    job = make_job("job1")

    # Error message with sensitive path
    error_with_path = "Failed to read /Users/john/Documents/secret.md"

    quarantined = store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message=error_with_path,
    )

    # Error message should be redacted
    assert "/Users/john/Documents" not in quarantined.error_message
    # But should still contain some of the error context
    assert "Failed to read" in quarantined.error_message


def test_concurrent_access(tmp_path: Path) -> None:
    """Test thread-safe concurrent access."""
    import threading

    store = QuarantineStore(tmp_path / "quarantine.db")

    def quarantine_job(job_id: str) -> None:
        job = make_job(job_id)
        store.quarantine(
            job=job,
            reason=QuarantineReason.PARSE_ERROR,
            error_message="Concurrent error",
        )

    # Create multiple threads quarantining jobs
    threads = []
    for i in range(10):
        thread = threading.Thread(target=quarantine_job, args=(f"job{i}",))
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Verify all jobs were quarantined
    all_jobs = store.list()
    assert len(all_jobs) == 10


def test_metadata_storage(tmp_path: Path) -> None:
    """Test JSON metadata serialization and deserialization."""
    store = QuarantineStore(tmp_path / "quarantine.db")
    job = make_job("job1")

    metadata = {
        "source_name": "docs",
        "file_type": "markdown",
        "size_bytes": 1024,
        "nested": {"key": "value"},
    }

    quarantined = store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
        metadata=metadata,
    )

    # Verify metadata is preserved
    retrieved = store.get("job1")
    assert retrieved is not None
    assert retrieved.metadata == metadata
    assert retrieved.metadata["nested"]["key"] == "value"


def test_can_retry_flag(tmp_path: Path) -> None:
    """Test can_retry flag behavior."""
    store = QuarantineStore(tmp_path / "quarantine.db")
    job = make_job("job1")

    # Jobs should be retryable by default
    quarantined = store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
    )
    assert quarantined.can_retry is True

    # Manually disable retry
    with store._lock:
        with store._conn:
            store._conn.execute(
                "UPDATE quarantined_jobs SET can_retry = 0 WHERE job_id = ?",
                ("job1",),
            )

    retrieved = store.get("job1")
    assert retrieved is not None
    assert retrieved.can_retry is False


def test_quarantine_with_traceback(tmp_path: Path) -> None:
    """Test storing full error tracebacks."""
    store = QuarantineStore(tmp_path / "quarantine.db")
    job = make_job("job1")

    traceback_text = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("Test error")
ValueError: Test error"""

    quarantined = store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
        error_traceback=traceback_text,
    )

    retrieved = store.get("job1")
    assert retrieved is not None
    assert retrieved.error_traceback == traceback_text


def test_quarantine_remove(tmp_path: Path) -> None:
    """Test explicit job removal from quarantine."""
    store = QuarantineStore(tmp_path / "quarantine.db")
    job = make_job("job1")

    store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
    )

    # Verify job exists
    assert store.get("job1") is not None

    # Remove job
    store.remove("job1")

    # Verify job is gone
    assert store.get("job1") is None


def test_retry_success_rate_calculation(tmp_path: Path) -> None:
    """Test retry success rate statistics calculation."""
    store = QuarantineStore(tmp_path / "quarantine.db")

    # Create job with some successful and failed retries
    job1 = make_job("job1")
    store.quarantine(job=job1, reason=QuarantineReason.PARSE_ERROR, error_message="Error 1")

    # Manually update retry counts
    with store._lock:
        with store._conn:
            store._conn.execute(
                """UPDATE quarantined_jobs
                   SET retry_success_count = 3, retry_failure_count = 1
                   WHERE job_id = ?""",
                ("job1",),
            )

    # Create another job
    job2 = make_job("job2")
    store.quarantine(job=job2, reason=QuarantineReason.TIMEOUT, error_message="Error 2")

    with store._lock:
        with store._conn:
            store._conn.execute(
                """UPDATE quarantined_jobs
                   SET retry_success_count = 1, retry_failure_count = 1
                   WHERE job_id = ?""",
                ("job2",),
            )

    stats = store.statistics()
    # Total: 4 success, 2 failure = 4/6 = 0.666...
    assert abs(stats["retry_success_rate"] - 0.666666) < 0.001


def test_original_payload_preservation(tmp_path: Path) -> None:
    """Test that original job payload is preserved for retry."""
    store = QuarantineStore(tmp_path / "quarantine.db")

    original_payload = {
        "source_name": "docs",
        "path": "/test/path",
        "attempts": 3,
        "custom_field": "value",
    }

    job = make_job("job1", payload=original_payload)

    store.quarantine(
        job=job,
        reason=QuarantineReason.PARSE_ERROR,
        error_message="Parse error",
    )

    retrieved = store.get("job1")
    assert retrieved is not None
    assert retrieved.original_payload == original_payload
    assert retrieved.original_payload["custom_field"] == "value"
