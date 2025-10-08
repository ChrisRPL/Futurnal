"""Custom exceptions for state machine operations."""

from __future__ import annotations


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid state transition is attempted.

    This exception indicates a violation of the state machine's
    transition rules defined in VALID_TRANSITIONS.

    Example:
        Attempting to transition from SUCCEEDED to RUNNING would
        raise this exception since it's not a valid state flow.
    """

    pass


class StateTransitionRaceError(RuntimeError):
    """Raised when optimistic locking detects a concurrent state modification.

    This exception occurs when a job's state changes between the time
    it was read and the time an update was attempted, indicating a
    race condition.

    Example:
        Thread A reads job status as PENDING and attempts to mark it RUNNING,
        but Thread B already marked it RUNNING. The optimistic lock check
        (via SQL WHERE clause) detects this and raises this exception.
    """

    pass


class JobNotFoundError(KeyError):
    """Raised when a job doesn't exist in the queue.

    This exception indicates that an operation was attempted on a
    job_id that isn't present in the job queue database.

    Example:
        Attempting to mark a non-existent job as completed would
        raise this exception.
    """

    pass
