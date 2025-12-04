"""PKG Database Manager.

Manages Neo4j database lifecycle for PKG storage including connection
management, schema initialization, health checks, and session handling.

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md

Option B Compliance:
- Integrates with existing StorageSettings for credentials
- Uses init_schema() from Module 01 (temporal-first design)
- Privacy-first: credentials from keyring via SecretStore
- Production-ready with retry logic and error handling
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Tuple, TYPE_CHECKING

from futurnal.configuration.settings import StorageSettings
from futurnal.pkg.database.config import PKGDatabaseConfig
from futurnal.pkg.database.exceptions import (
    PKGConnectionError,
    PKGHealthCheckError,
    PKGSchemaInitializationError,
)
from futurnal.pkg.schema.constraints import (
    get_schema_statistics,
    init_schema,
    validate_schema,
)

if TYPE_CHECKING:
    from neo4j import Driver, Session
    from futurnal.privacy.audit import AuditLogger

# Runtime imports for neo4j
try:
    from neo4j import GraphDatabase, Driver as Neo4jDriver
    from neo4j.exceptions import AuthError, ServiceUnavailable, SessionExpired
except ImportError:
    # For testing environments without neo4j installed
    GraphDatabase = None  # type: ignore
    Neo4jDriver = None  # type: ignore
    AuthError = Exception  # type: ignore
    ServiceUnavailable = Exception  # type: ignore
    SessionExpired = Exception  # type: ignore

logger = logging.getLogger(__name__)


class PKGDatabaseManager:
    """Manages Neo4j database lifecycle for PKG storage.

    This class provides:
    - Connection management with retry logic
    - Schema initialization using Module 01 constraints
    - Health checks for monitoring
    - Session context manager for ACID transactions
    - Integration with audit logging

    The manager does NOT store credentials - it receives them from StorageSettings
    which retrieves passwords from the secure keyring.

    Example:
        >>> from futurnal.configuration.settings import bootstrap_settings
        >>> settings = bootstrap_settings()
        >>> manager = PKGDatabaseManager(settings.workspace.storage)
        >>> with manager:
        ...     with manager.session() as session:
        ...         session.run("MATCH (n) RETURN count(n)")

    Context Manager Usage:
        >>> with PKGDatabaseManager(storage_settings) as manager:
        ...     manager.initialize_schema()
        ...     with manager.session() as session:
        ...         # perform database operations
        ...         pass
    """

    def __init__(
        self,
        storage_settings: StorageSettings,
        config: Optional[PKGDatabaseConfig] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """Initialize the database manager.

        Args:
            storage_settings: Storage configuration containing Neo4j credentials.
                              Credentials are retrieved from keyring via SecretStr.
            config: Optional PKG-specific configuration for connection pooling,
                    backup settings, etc. If None, uses defaults.
            audit_logger: Optional audit logger for recording database operations.
        """
        self._storage = storage_settings
        self._config = config or PKGDatabaseConfig()
        self._audit = audit_logger
        self._driver: Optional["Driver"] = None
        self._connected = False

        # Log encryption info on initialization
        self._log_encryption_info()

    def _log_encryption_info(self) -> None:
        """Log information about encryption settings."""
        if self._storage.neo4j_encrypted:
            logger.info("PKG database connection configured with TLS encryption")
        else:
            logger.warning(
                "PKG database connection configured WITHOUT TLS encryption. "
                "For production, use bolt+s:// or neo4j+s:// URI schemes."
            )

        # Info about at-rest encryption
        logger.info(
            "For at-rest encryption, ensure workspace is on encrypted filesystem "
            "(FileVault/BitLocker/LUKS). Neo4j Community doesn't support native encryption."
        )

    def connect(self) -> "Driver":
        """Establish connection to Neo4j with retry logic.

        Creates a new driver if one doesn't exist, or returns the existing
        driver if already connected. Uses exponential backoff for retries.

        Returns:
            The Neo4j driver instance.

        Raises:
            PKGConnectionError: If connection fails after all retries.

        Example:
            >>> manager = PKGDatabaseManager(storage_settings)
            >>> driver = manager.connect()
            >>> # Use driver...
            >>> manager.disconnect()
        """
        if self._driver is not None and self._connected:
            return self._driver

        uri = self._storage.neo4j_uri
        username = self._storage.neo4j_username
        password = self._storage.neo4j_password.get_secret_value()
        encrypted = self._storage.neo4j_encrypted

        # Get driver configuration from PKGDatabaseConfig
        driver_config = self._config.to_driver_config()

        last_error: Optional[Exception] = None
        delay = self._config.retry_delay_seconds

        for attempt in range(1, self._config.max_connection_retries + 1):
            try:
                logger.debug(
                    f"Connecting to Neo4j at {uri} (attempt {attempt}/"
                    f"{self._config.max_connection_retries})"
                )

                self._driver = GraphDatabase.driver(
                    uri,
                    auth=(username, password),
                    encrypted=encrypted,
                    **driver_config,
                )

                # Verify connectivity
                self._driver.verify_connectivity()
                self._connected = True

                logger.info(f"Successfully connected to Neo4j at {uri}")
                self._audit_event("connect", "succeeded", {"uri": uri})

                return self._driver

            except AuthError as e:
                # Authentication errors should not be retried
                logger.error(f"Authentication failed for Neo4j at {uri}: {e}")
                self._audit_event(
                    "connect", "failed", {"uri": uri, "error": "authentication"}
                )
                raise PKGConnectionError(
                    f"Authentication failed for Neo4j at {uri}",
                    uri=uri,
                    attempts=attempt,
                    last_error=e,
                ) from e

            except (ServiceUnavailable, OSError) as e:
                last_error = e
                logger.warning(
                    f"Connection attempt {attempt} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                if attempt < self._config.max_connection_retries:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

        # All retries exhausted
        logger.error(
            f"Failed to connect to Neo4j at {uri} after "
            f"{self._config.max_connection_retries} attempts"
        )
        self._audit_event(
            "connect",
            "failed",
            {
                "uri": uri,
                "attempts": self._config.max_connection_retries,
                "error": str(last_error),
            },
        )

        raise PKGConnectionError(
            f"Failed to connect to Neo4j at {uri} after "
            f"{self._config.max_connection_retries} attempts",
            uri=uri,
            attempts=self._config.max_connection_retries,
            last_error=last_error,
        )

    def disconnect(self) -> None:
        """Close database connection and cleanup resources.

        Safe to call multiple times. If not connected, this is a no-op.

        Example:
            >>> manager.connect()
            >>> # Use database...
            >>> manager.disconnect()
        """
        if self._driver is not None:
            try:
                self._driver.close()
                logger.info("Disconnected from Neo4j")
                self._audit_event("disconnect", "succeeded", {})
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._driver = None
                self._connected = False

    def initialize_schema(self, database: Optional[str] = None) -> Dict[str, bool]:
        """Initialize PKG schema with constraints and indices.

        Calls init_schema() from Module 01 to create all required
        constraints and indices. This operation is idempotent.

        Args:
            database: Optional Neo4j database name. If None, uses default.

        Returns:
            Dictionary mapping constraint/index names to success status.

        Raises:
            PKGSchemaInitializationError: If schema initialization fails.
            PKGConnectionError: If not connected.

        Example:
            >>> with manager:
            ...     results = manager.initialize_schema()
            ...     assert all(results.values()), "Schema init failed"
        """
        if self._driver is None:
            raise PKGConnectionError("Not connected. Call connect() first.")

        try:
            logger.info("Initializing PKG schema...")
            results = init_schema(self._driver, database=database, skip_on_error=False)

            # Check for failures
            failed = [name for name, success in results.items() if not success]
            if failed:
                raise PKGSchemaInitializationError(
                    f"Failed to initialize {len(failed)} schema elements",
                    failed_constraints=[f for f in failed if "constraint" in f.lower()],
                    failed_indices=[f for f in failed if "index" in f.lower()],
                )

            logger.info(
                f"PKG schema initialized: {len(results)} constraints/indices created"
            )
            self._audit_event(
                "initialize_schema", "succeeded", {"elements": len(results)}
            )

            return results

        except PKGSchemaInitializationError:
            raise
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            self._audit_event(
                "initialize_schema", "failed", {"error": str(e)}
            )
            raise PKGSchemaInitializationError(
                f"Schema initialization failed: {e}"
            ) from e

    @contextmanager
    def session(
        self, database: Optional[str] = None
    ) -> Generator["Session", None, None]:
        """Context manager for database sessions.

        Provides ACID guarantees for database operations. Sessions should
        be used for related operations that form a logical unit of work.

        Args:
            database: Optional Neo4j database name. If None, uses default.

        Yields:
            A Neo4j Session for executing queries.

        Raises:
            PKGConnectionError: If not connected.

        Example:
            >>> with manager.session() as session:
            ...     result = session.run("MATCH (n:Person) RETURN n.name")
            ...     names = [record["n.name"] for record in result]
        """
        if self._driver is None:
            raise PKGConnectionError("Not connected. Call connect() first.")

        session = self._driver.session(
            database=database,
            fetch_size=self._config.fetch_size,
        )

        try:
            yield session
        finally:
            session.close()

    def health_check(self) -> Tuple[bool, str]:
        """Check database connectivity and health.

        Performs:
        1. Connectivity check
        2. Schema validation

        Returns:
            Tuple of (is_healthy, message).

        Example:
            >>> is_healthy, message = manager.health_check()
            >>> if not is_healthy:
            ...     logger.warning(f"Database unhealthy: {message}")
        """
        if self._driver is None:
            return False, "Not connected"

        try:
            # Check connectivity
            self._driver.verify_connectivity()

            # Validate schema
            validation = validate_schema(self._driver)
            missing = [name for name, exists in validation.items() if not exists]

            if missing:
                return False, f"Missing schema elements: {missing[:5]}..."

            return True, "Healthy"

        except (ServiceUnavailable, SessionExpired) as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Health check failed: {e}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics for monitoring.

        Returns node counts, relationship counts, and schema information.

        Returns:
            Dictionary with database statistics.

        Raises:
            PKGConnectionError: If not connected.

        Example:
            >>> stats = manager.get_statistics()
            >>> print(f"Nodes: {sum(stats['node_counts'].values())}")
        """
        if self._driver is None:
            raise PKGConnectionError("Not connected. Call connect() first.")

        return get_schema_statistics(self._driver)

    def get_driver(self) -> Optional["Driver"]:
        """Get the underlying Neo4j driver.

        Returns None if not connected. For most use cases, prefer
        using the session() context manager.

        Returns:
            The Neo4j driver or None if not connected.
        """
        return self._driver

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to the database."""
        return self._connected and self._driver is not None

    def _audit_event(
        self,
        action: str,
        status: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Record an audit event if audit logger is configured."""
        if self._audit is None:
            return

        try:
            # Import here to avoid circular dependencies
            from datetime import datetime

            self._audit.record(
                job_id=f"pkg_db_{action}_{datetime.utcnow().isoformat()}",
                source="pkg_database_manager",
                action=action,
                status=status,
                timestamp=datetime.utcnow(),
                metadata=metadata,
            )
        except Exception as e:
            # Don't fail operations due to audit logging issues
            logger.debug(f"Failed to record audit event: {e}")

    def __enter__(self) -> "PKGDatabaseManager":
        """Enter context manager, establishing connection."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, cleaning up connection."""
        self.disconnect()
