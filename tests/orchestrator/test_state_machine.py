"""Tests for state machine validation and transitions."""

from datetime import datetime
from pathlib import Path

import pytest

from futurnal.orchestrator.exceptions import InvalidStateTransitionError
from futurnal.orchestrator.state_machine import (
    JobStatus,
    VALID_TRANSITIONS,
    StateTransition,
    StateMachineValidator,
)
from futurnal.privacy.audit import AuditLogger


def test_valid_transitions_pending_to_running():
    """Test valid transition from PENDING to RUNNING."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.PENDING,
        to_status=JobStatus.RUNNING,
    )

    assert transition.is_valid()
    assert not transition.is_idempotent()
    assert transition.from_status == JobStatus.PENDING
    assert transition.to_status == JobStatus.RUNNING


def test_valid_transitions_running_to_succeeded():
    """Test valid transition from RUNNING to SUCCEEDED."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.RUNNING,
        to_status=JobStatus.SUCCEEDED,
    )

    assert transition.is_valid()
    assert not transition.is_idempotent()


def test_valid_transitions_running_to_failed():
    """Test valid transition from RUNNING to FAILED."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.RUNNING,
        to_status=JobStatus.FAILED,
    )

    assert transition.is_valid()
    assert not transition.is_idempotent()


def test_valid_transitions_failed_to_pending():
    """Test valid transition from FAILED to PENDING (retry)."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.FAILED,
        to_status=JobStatus.PENDING,
    )

    assert transition.is_valid()
    assert not transition.is_idempotent()


def test_valid_transitions_failed_to_quarantined():
    """Test valid transition from FAILED to QUARANTINED."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.FAILED,
        to_status=JobStatus.QUARANTINED,
    )

    assert transition.is_valid()
    assert not transition.is_idempotent()


def test_valid_transitions_quarantined_to_pending():
    """Test valid transition from QUARANTINED to PENDING (manual retry)."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.QUARANTINED,
        to_status=JobStatus.PENDING,
        operator="admin",
        reason="manual_retry",
    )

    assert transition.is_valid()
    assert not transition.is_idempotent()
    assert transition.operator == "admin"
    assert transition.reason == "manual_retry"


def test_invalid_transitions_succeeded_to_running():
    """Test invalid transition from SUCCEEDED to RUNNING."""
    validator = StateMachineValidator()

    with pytest.raises(InvalidStateTransitionError):
        validator.validate_transition(
            job_id="test-job",
            from_status=JobStatus.SUCCEEDED,
            to_status=JobStatus.RUNNING,
        )


def test_invalid_transitions_succeeded_to_failed():
    """Test invalid transition from SUCCEEDED to FAILED."""
    validator = StateMachineValidator()

    with pytest.raises(InvalidStateTransitionError):
        validator.validate_transition(
            job_id="test-job",
            from_status=JobStatus.SUCCEEDED,
            to_status=JobStatus.FAILED,
        )


def test_invalid_transitions_pending_to_succeeded():
    """Test invalid transition from PENDING to SUCCEEDED."""
    validator = StateMachineValidator()

    with pytest.raises(InvalidStateTransitionError):
        validator.validate_transition(
            job_id="test-job",
            from_status=JobStatus.PENDING,
            to_status=JobStatus.SUCCEEDED,
        )


def test_idempotent_transitions_pending_to_pending():
    """Test idempotent transition PENDING to PENDING."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.PENDING,
        to_status=JobStatus.PENDING,
    )

    assert transition.is_idempotent()
    assert transition.is_valid()  # Idempotent transitions are valid


def test_idempotent_transitions_running_to_running():
    """Test idempotent transition RUNNING to RUNNING."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.RUNNING,
        to_status=JobStatus.RUNNING,
    )

    assert transition.is_idempotent()
    assert transition.is_valid()


def test_idempotent_transitions_succeeded_to_succeeded():
    """Test idempotent transition SUCCEEDED to SUCCEEDED."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.SUCCEEDED,
        to_status=JobStatus.SUCCEEDED,
    )

    assert transition.is_idempotent()
    assert transition.is_valid()


def test_idempotent_transitions_failed_to_failed():
    """Test idempotent transition FAILED to FAILED."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.FAILED,
        to_status=JobStatus.FAILED,
    )

    assert transition.is_idempotent()
    assert transition.is_valid()


def test_idempotent_transitions_quarantined_to_quarantined():
    """Test idempotent transition QUARANTINED to QUARANTINED."""
    validator = StateMachineValidator()
    transition = validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.QUARANTINED,
        to_status=JobStatus.QUARANTINED,
    )

    assert transition.is_idempotent()
    assert transition.is_valid()


def test_transition_audit_trail(tmp_path: Path):
    """Test that all transitions are logged to audit trail."""
    audit_logger = AuditLogger(tmp_path / "audit")
    validator = StateMachineValidator(audit_logger=audit_logger)

    # Perform a transition
    validator.validate_transition(
        job_id="test-job",
        from_status=JobStatus.PENDING,
        to_status=JobStatus.RUNNING,
    )

    # Verify audit log exists
    audit_file = tmp_path / "audit" / "audit.log"
    assert audit_file.exists()

    # Read and verify audit entry
    content = audit_file.read_text()
    assert "test-job" in content
    assert "state_machine" in content
    assert "transition_pending_to_running" in content


def test_transition_history_tracking():
    """Test that validator tracks transition history."""
    validator = StateMachineValidator()

    # Perform multiple transitions
    validator.validate_transition(
        job_id="job-1",
        from_status=JobStatus.PENDING,
        to_status=JobStatus.RUNNING,
    )
    validator.validate_transition(
        job_id="job-1",
        from_status=JobStatus.RUNNING,
        to_status=JobStatus.SUCCEEDED,
    )

    # Verify history
    assert len(validator._transition_history) == 2
    assert validator._transition_history[0].job_id == "job-1"
    assert validator._transition_history[0].from_status == JobStatus.PENDING
    assert validator._transition_history[1].to_status == JobStatus.SUCCEEDED


def test_valid_transitions_completeness():
    """Test that VALID_TRANSITIONS defines transitions for all states."""
    # Verify all states are present in VALID_TRANSITIONS
    for status in JobStatus:
        assert status in VALID_TRANSITIONS

    # Verify all states have at least idempotent transition
    for status in JobStatus:
        assert status in VALID_TRANSITIONS[status]


# Property-based test using hypothesis (requires hypothesis library)
try:
    from hypothesis import given, strategies as st

    @given(
        from_status=st.sampled_from(list(JobStatus)),
        to_status=st.sampled_from(list(JobStatus)),
    )
    def test_transition_determinism(from_status: JobStatus, to_status: JobStatus):
        """State transitions are deterministic (property-based test)."""
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

except ImportError:
    # Hypothesis not installed, skip property-based test
    pass
