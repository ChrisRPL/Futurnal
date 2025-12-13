"""Tests for orchestrator audit event helpers."""

from datetime import datetime
from pathlib import Path
import json
import pytest

from src.futurnal.orchestrator.audit_events import (
    OrchestratorAuditEvents,
    log_system_start,
    log_system_shutdown,
    log_crash_recovery,
    log_job_enqueued,
    log_job_started,
    log_job_completed,
    log_job_failed,
    log_retry_scheduled,
    log_job_quarantined,
    log_state_transition,
    log_invalid_state_transition,
    _hash_path,
    _sanitize_error_message,
)
from src.futurnal.privacy.audit import AuditLogger


@pytest.fixture
def audit_logger(tmp_path):
    """Create an audit logger for testing."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(exist_ok=True)
    return AuditLogger(output_dir=audit_dir)


def read_last_audit_event(audit_logger) -> dict:
    """Read the last audit event from the log."""
    log_path = audit_logger._path
    if not log_path.exists():
        return {}
    lines = log_path.read_text().strip().split("\n")
    if lines and lines[-1]:
        return json.loads(lines[-1])
    return {}


class TestOrchestratorAuditEventConstants:
    """Test event type constants."""

    def test_system_event_constants(self):
        assert OrchestratorAuditEvents.SYSTEM_STARTED == "orchestrator_started"
        assert OrchestratorAuditEvents.SYSTEM_SHUTDOWN == "orchestrator_shutdown"
        assert OrchestratorAuditEvents.SYSTEM_CRASH_RECOVERY == "orchestrator_crash_recovery"

    def test_job_event_constants(self):
        assert OrchestratorAuditEvents.JOB_ENQUEUED == "job_enqueued"
        assert OrchestratorAuditEvents.JOB_STARTED == "job_started"
        assert OrchestratorAuditEvents.JOB_COMPLETED == "job_completed"
        assert OrchestratorAuditEvents.JOB_FAILED == "job_failed"
        assert OrchestratorAuditEvents.JOB_RETRY_SCHEDULED == "job_retry_scheduled"
        assert OrchestratorAuditEvents.JOB_QUARANTINED == "job_quarantined"


class TestSystemAuditEvents:
    """Test system-level audit events."""

    def test_log_system_start(self, audit_logger):
        log_system_start(
            audit_logger,
            workspace_dir="/home/user/.futurnal",
            configured_workers=4,
            registered_sources=3,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == OrchestratorAuditEvents.SYSTEM_STARTED
        assert event["status"] == "success"
        assert event["source"] == "orchestrator"
        assert event["metadata"]["configured_workers"] == 4
        assert event["metadata"]["registered_sources"] == 3
        # Path should be hashed, not plain
        assert "workspace_hash" in event["metadata"]
        assert event["metadata"]["workspace_hash"] != "/home/user/.futurnal"

    def test_log_system_shutdown(self, audit_logger):
        log_system_shutdown(
            audit_logger,
            jobs_completed=100,
            jobs_failed=5,
            jobs_pending=2,
            uptime_seconds=3600.5,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == OrchestratorAuditEvents.SYSTEM_SHUTDOWN
        assert event["status"] == "success"
        assert event["metadata"]["jobs_completed"] == 100
        assert event["metadata"]["jobs_failed"] == 5
        assert event["metadata"]["jobs_pending"] == 2
        assert event["metadata"]["uptime_seconds"] == 3600.5

    def test_log_crash_recovery(self, audit_logger):
        log_crash_recovery(
            audit_logger,
            jobs_reset=5,
            recovery_duration_seconds=2.5,
            errors=1,
        )

        event = read_last_audit_event(audit_logger)
        assert event["action"] == OrchestratorAuditEvents.SYSTEM_CRASH_RECOVERY
        assert event["status"] == "partial_success"  # Because errors > 0
        assert event["metadata"]["jobs_reset"] == 5
        assert event["metadata"]["recovery_duration_seconds"] == 2.5
        assert event["metadata"]["errors"] == 1

    def test_log_crash_recovery_no_errors(self, audit_logger):
        log_crash_recovery(
            audit_logger,
            jobs_reset=3,
            recovery_duration_seconds=1.0,
            errors=0,
        )

        event = read_last_audit_event(audit_logger)
        assert event["status"] == "success"


class TestJobAuditEvents:
    """Test job lifecycle audit events."""

    def test_log_job_enqueued(self, audit_logger):
        log_job_enqueued(
            audit_logger,
            job_id="test-job-123",
            source_name="my-obsidian-vault",
            job_type="LOCAL_FILES",
            trigger="schedule",
            priority="normal",
        )

        event = read_last_audit_event(audit_logger)
        assert event["job_id"] == "test-job-123"
        assert event["source"] == "my-obsidian-vault"
        assert event["action"] == OrchestratorAuditEvents.JOB_ENQUEUED
        assert event["status"] == "pending"
        assert event["metadata"]["job_type"] == "LOCAL_FILES"
        assert event["metadata"]["trigger"] == "schedule"
        assert event["metadata"]["priority"] == "normal"

    def test_log_job_started(self, audit_logger):
        log_job_started(
            audit_logger,
            job_id="test-job-456",
            source_name="my-imap-mailbox",
            job_type="IMAP_MAILBOX",
            attempt=2,
        )

        event = read_last_audit_event(audit_logger)
        assert event["job_id"] == "test-job-456"
        assert event["action"] == OrchestratorAuditEvents.JOB_STARTED
        assert event["status"] == "running"
        assert event["attempt"] == 2
        assert event["metadata"]["job_type"] == "IMAP_MAILBOX"

    def test_log_job_completed(self, audit_logger):
        log_job_completed(
            audit_logger,
            job_id="test-job-789",
            source_name="github-repo",
            files_processed=50,
            bytes_processed=1024000,
            duration_seconds=10.5,
        )

        event = read_last_audit_event(audit_logger)
        assert event["job_id"] == "test-job-789"
        assert event["action"] == OrchestratorAuditEvents.JOB_COMPLETED
        assert event["status"] == "succeeded"
        assert event["metadata"]["files_processed"] == 50
        assert event["metadata"]["bytes_processed"] == 1024000
        assert event["metadata"]["duration_seconds"] == 10.5
        # Check throughput calculation
        expected_throughput = round(1024000 / 10.5, 2)
        assert event["metadata"]["throughput_bytes_per_sec"] == expected_throughput

    def test_log_job_failed(self, audit_logger):
        log_job_failed(
            audit_logger,
            job_id="test-job-fail",
            source_name="broken-source",
            error_type="network",
            error_message="Connection refused",
            attempt=3,
            files_processed=10,
            bytes_processed=5000,
        )

        event = read_last_audit_event(audit_logger)
        assert event["job_id"] == "test-job-fail"
        assert event["action"] == OrchestratorAuditEvents.JOB_FAILED
        assert event["status"] == "failed"
        assert event["attempt"] == 3
        assert event["metadata"]["error_type"] == "network"
        assert event["metadata"]["files_processed"] == 10


class TestRetryAndQuarantineEvents:
    """Test retry and quarantine audit events."""

    def test_log_retry_scheduled(self, audit_logger):
        log_retry_scheduled(
            audit_logger,
            job_id="retry-job-1",
            source_name="my-source",
            attempt=2,
            delay_seconds=60.0,
            failure_type="transient",
            retry_strategy="exponential",
        )

        event = read_last_audit_event(audit_logger)
        assert event["job_id"] == "retry-job-1"
        assert event["action"] == OrchestratorAuditEvents.JOB_RETRY_SCHEDULED
        assert event["status"] == "pending"
        assert event["attempt"] == 2
        assert event["metadata"]["delay_seconds"] == 60.0
        assert event["metadata"]["failure_type"] == "transient"
        assert event["metadata"]["retry_strategy"] == "exponential"

    def test_log_job_quarantined(self, audit_logger):
        log_job_quarantined(
            audit_logger,
            job_id="quarantine-job-1",
            source_name="bad-source",
            reason="permanent_error",
            total_attempts=5,
        )

        event = read_last_audit_event(audit_logger)
        assert event["job_id"] == "quarantine-job-1"
        assert event["action"] == OrchestratorAuditEvents.JOB_QUARANTINED
        assert event["status"] == "quarantined"
        assert event["metadata"]["reason"] == "permanent_error"
        assert event["metadata"]["total_attempts"] == 5


class TestStateTransitionEvents:
    """Test state transition audit events."""

    def test_log_state_transition(self, audit_logger):
        log_state_transition(
            audit_logger,
            job_id="state-job-1",
            from_status="pending",
            to_status="running",
            source_name="my-source",
        )

        event = read_last_audit_event(audit_logger)
        assert event["job_id"] == "state-job-1"
        assert event["action"] == OrchestratorAuditEvents.STATE_TRANSITION
        assert event["status"] == "success"
        assert event["metadata"]["from_status"] == "pending"
        assert event["metadata"]["to_status"] == "running"

    def test_log_invalid_state_transition(self, audit_logger):
        log_invalid_state_transition(
            audit_logger,
            job_id="invalid-state-job",
            from_status="succeeded",
            to_status="running",
            reason="Cannot run completed job",
        )

        event = read_last_audit_event(audit_logger)
        assert event["job_id"] == "invalid-state-job"
        assert event["action"] == OrchestratorAuditEvents.STATE_TRANSITION_INVALID
        assert event["status"] == "blocked"
        assert event["metadata"]["from_status"] == "succeeded"
        assert event["metadata"]["to_status"] == "running"
        assert event["metadata"]["reason"] == "Cannot run completed job"


class TestPrivacyHelpers:
    """Test privacy helper functions."""

    def test_hash_path(self):
        path1 = "/Users/john/Documents/secret.txt"
        path2 = "/Users/john/Documents/other.txt"

        hash1 = _hash_path(path1)
        hash2 = _hash_path(path2)

        # Hash should be different for different paths
        assert hash1 != hash2
        # Hash should be consistent
        assert hash1 == _hash_path(path1)
        # Hash should be 16 characters
        assert len(hash1) == 16

    def test_sanitize_error_message_paths(self):
        message = "Failed to open /Users/john/private/secret.txt"
        sanitized = _sanitize_error_message(message)

        assert "/Users/john" not in sanitized
        assert "[PATH]" in sanitized

    def test_sanitize_error_message_email(self):
        message = "Cannot send to john.doe@example.com"
        sanitized = _sanitize_error_message(message)

        assert "john.doe@example.com" not in sanitized
        assert "[EMAIL]" in sanitized

    def test_sanitize_error_message_tokens(self):
        message = "Invalid token: abc123def456789012345678901234567890"
        sanitized = _sanitize_error_message(message)

        assert "abc123def456789012345678901234567890" not in sanitized
        assert "[TOKEN]" in sanitized

    def test_sanitize_error_message_windows_path(self):
        message = "File not found: C:\\Users\\john\\Documents\\file.txt"
        sanitized = _sanitize_error_message(message)

        assert "C:\\Users\\john" not in sanitized
        assert "[PATH]" in sanitized


class TestChainIntegrity:
    """Test that audit events maintain chain integrity."""

    def test_multiple_events_have_chain_fields(self, audit_logger):
        """Verify events have chain fields for tamper detection."""
        log_system_start(
            audit_logger,
            workspace_dir="/tmp/test",
            configured_workers=2,
            registered_sources=1,
        )

        log_job_enqueued(
            audit_logger,
            job_id="job-1",
            source_name="source-1",
            job_type="LOCAL_FILES",
            trigger="manual",
            priority="normal",
        )

        log_job_started(
            audit_logger,
            job_id="job-1",
            source_name="source-1",
            job_type="LOCAL_FILES",
            attempt=1,
        )

        log_job_completed(
            audit_logger,
            job_id="job-1",
            source_name="source-1",
            files_processed=10,
            bytes_processed=1000,
            duration_seconds=1.0,
        )

        # Verify all events have chain fields
        log_path = audit_logger._path
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 4

        for line in lines:
            event = json.loads(line)
            # Each event should have chain_hash and chain_prev for tamper detection
            assert "chain_hash" in event
            assert "chain_prev" in event

        # Verify the chain_prev of each event matches chain_hash of previous
        events = [json.loads(line) for line in lines]
        assert events[0]["chain_prev"] is None  # First event has no previous
        for i in range(1, len(events)):
            assert events[i]["chain_prev"] == events[i - 1]["chain_hash"]


class TestNoContentLeakage:
    """Test privacy compliance - no content in logs."""

    def test_error_message_truncated(self, audit_logger):
        """Verify error messages are truncated to prevent content exposure."""
        long_error = "Error: " + "x" * 500

        log_job_failed(
            audit_logger,
            job_id="fail-job",
            source_name="source",
            error_type="parse",
            error_message=long_error,
            attempt=1,
        )

        event = read_last_audit_event(audit_logger)
        # Error summary should be truncated to 200 chars
        assert len(event["metadata"]["error_summary"]) <= 200
