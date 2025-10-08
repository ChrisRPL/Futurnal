"""Database integrity checking and state corruption detection.

This module provides functions to validate the job queue database
for various forms of corruption, including duplicate entries, invalid
status values, and data constraint violations.
"""

from __future__ import annotations

import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .queue import JobQueue


logger = logging.getLogger(__name__)


def validate_database_integrity(queue: JobQueue) -> List[str]:
    """Check database for state corruption.

    Performs comprehensive integrity checks on the job queue database,
    including:
    - Duplicate job IDs (should be impossible with PRIMARY KEY)
    - Invalid status values outside the JobStatus enum
    - Negative attempt counts
    - Other constraint violations

    Args:
        queue: The job queue instance to validate

    Returns:
        List of issue descriptions (empty if database is clean)
    """
    issues = []

    # Check for duplicate job IDs
    duplicate_jobs = queue._conn.execute(
        """
        SELECT job_id, COUNT(*)
        FROM jobs
        GROUP BY job_id
        HAVING COUNT(*) > 1
        """
    ).fetchall()

    if duplicate_jobs:
        issues.append(f"Duplicate job IDs: {duplicate_jobs}")
        logger.error(
            "Database integrity violation: duplicate job IDs",
            extra={"duplicate_count": len(duplicate_jobs)},
        )

    # Check for invalid status values
    invalid_statuses = queue._conn.execute(
        """
        SELECT job_id, status
        FROM jobs
        WHERE status NOT IN ('pending', 'running', 'succeeded', 'failed', 'quarantined')
        """
    ).fetchall()

    if invalid_statuses:
        issues.append(f"Invalid statuses: {invalid_statuses}")
        logger.error(
            "Database integrity violation: invalid status values",
            extra={"invalid_count": len(invalid_statuses)},
        )

    # Check for negative attempts
    negative_attempts = queue._conn.execute(
        """
        SELECT job_id, attempts
        FROM jobs
        WHERE attempts < 0
        """
    ).fetchall()

    if negative_attempts:
        issues.append(f"Negative attempts: {negative_attempts}")
        logger.error(
            "Database integrity violation: negative attempt counts",
            extra={"negative_count": len(negative_attempts)},
        )

    # Check for NULL required fields
    null_required_fields = queue._conn.execute(
        """
        SELECT job_id
        FROM jobs
        WHERE job_type IS NULL
           OR payload IS NULL
           OR priority IS NULL
           OR status IS NULL
           OR created_at IS NULL
           OR updated_at IS NULL
        """
    ).fetchall()

    if null_required_fields:
        issues.append(f"NULL required fields: {null_required_fields}")
        logger.error(
            "Database integrity violation: NULL required fields",
            extra={"null_count": len(null_required_fields)},
        )

    # Check for invalid priority values
    invalid_priorities = queue._conn.execute(
        """
        SELECT job_id, priority
        FROM jobs
        WHERE priority NOT IN (1, 2, 3)
        """
    ).fetchall()

    if invalid_priorities:
        issues.append(f"Invalid priorities: {invalid_priorities}")
        logger.error(
            "Database integrity violation: invalid priority values",
            extra={"invalid_count": len(invalid_priorities)},
        )

    if not issues:
        logger.debug("Database integrity check passed")

    return issues
