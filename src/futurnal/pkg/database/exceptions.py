"""PKG Database Exception Hierarchy.

Custom exceptions for PKG database operations providing clear error classification
for connection failures, backup/restore issues, and schema initialization errors.

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md

Option B Compliance:
- Production-ready error handling with clear exception types
- Enables retry logic at appropriate levels
"""

from __future__ import annotations


class PKGDatabaseError(Exception):
    """Base exception for all PKG database operations.

    All PKG database-specific exceptions inherit from this class,
    allowing callers to catch all database errors with a single handler.

    Example:
        try:
            manager.connect()
        except PKGDatabaseError as e:
            logger.error(f"Database operation failed: {e}")
    """

    pass


class PKGConnectionError(PKGDatabaseError):
    """Raised when database connection fails.

    This includes:
    - Initial connection failures
    - Connection timeout
    - Authentication failures
    - Network errors

    Attributes:
        uri: The Neo4j URI that was attempted
        attempts: Number of connection attempts made
        last_error: The underlying error from the last attempt
    """

    def __init__(
        self,
        message: str,
        *,
        uri: str | None = None,
        attempts: int = 1,
        last_error: Exception | None = None,
    ):
        super().__init__(message)
        self.uri = uri
        self.attempts = attempts
        self.last_error = last_error

    def __str__(self) -> str:
        base = super().__str__()
        if self.uri:
            base = f"{base} (uri={self.uri}, attempts={self.attempts})"
        return base


class PKGBackupError(PKGDatabaseError):
    """Raised when backup operation fails.

    This includes:
    - Backup file creation failures
    - Export query failures
    - Disk space issues
    - Backup verification failures

    Attributes:
        backup_path: Path where backup was being written
        node_count: Number of nodes exported before failure
        relationship_count: Number of relationships exported before failure
    """

    def __init__(
        self,
        message: str,
        *,
        backup_path: str | None = None,
        node_count: int = 0,
        relationship_count: int = 0,
    ):
        super().__init__(message)
        self.backup_path = backup_path
        self.node_count = node_count
        self.relationship_count = relationship_count


class PKGRestoreError(PKGDatabaseError):
    """Raised when restore operation fails.

    This includes:
    - Backup file not found
    - Backup file corruption
    - Import query failures
    - Data validation failures post-restore

    Attributes:
        backup_path: Path of backup file being restored
        nodes_restored: Number of nodes restored before failure
        relationships_restored: Number of relationships restored before failure
        pre_restore_backup: Path to the pre-restore backup if one was created
    """

    def __init__(
        self,
        message: str,
        *,
        backup_path: str | None = None,
        nodes_restored: int = 0,
        relationships_restored: int = 0,
        pre_restore_backup: str | None = None,
    ):
        super().__init__(message)
        self.backup_path = backup_path
        self.nodes_restored = nodes_restored
        self.relationships_restored = relationships_restored
        self.pre_restore_backup = pre_restore_backup


class PKGSchemaInitializationError(PKGDatabaseError):
    """Raised when schema initialization fails.

    This includes:
    - Constraint creation failures
    - Index creation failures
    - Schema validation failures

    Attributes:
        failed_constraints: List of constraint names that failed
        failed_indices: List of index names that failed
    """

    def __init__(
        self,
        message: str,
        *,
        failed_constraints: list[str] | None = None,
        failed_indices: list[str] | None = None,
    ):
        super().__init__(message)
        self.failed_constraints = failed_constraints or []
        self.failed_indices = failed_indices or []

    def __str__(self) -> str:
        base = super().__str__()
        details = []
        if self.failed_constraints:
            details.append(f"constraints={self.failed_constraints}")
        if self.failed_indices:
            details.append(f"indices={self.failed_indices}")
        if details:
            base = f"{base} ({', '.join(details)})"
        return base


class PKGHealthCheckError(PKGDatabaseError):
    """Raised when health check fails.

    This indicates the database is not in a healthy state for operations.

    Attributes:
        check_name: Name of the health check that failed
        details: Additional details about the failure
    """

    def __init__(
        self,
        message: str,
        *,
        check_name: str | None = None,
        details: str | None = None,
    ):
        super().__init__(message)
        self.check_name = check_name
        self.details = details
