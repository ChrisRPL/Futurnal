"""PKG Database Module.

Provides database setup, configuration, and lifecycle management for the
Personal Knowledge Graph (PKG) storage layer.

This module implements Module 02 of the PKG Graph Storage production plan:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md

Components:
    - PKGDatabaseConfig: Pydantic configuration for connection pooling and backup
    - PKGDatabaseManager: Lifecycle management (connect, schema init, health)
    - PKGBackupManager: Backup and restore operations
    - Exception hierarchy for error handling

Example:
    >>> from futurnal.configuration.settings import bootstrap_settings
    >>> from futurnal.pkg.database import PKGDatabaseConfig, PKGDatabaseManager

    >>> settings = bootstrap_settings()
    >>> config = PKGDatabaseConfig.with_env_overrides()

    >>> with PKGDatabaseManager(settings.workspace.storage, config) as manager:
    ...     manager.initialize_schema()
    ...     with manager.session() as session:
    ...         result = session.run("MATCH (n) RETURN count(n)")
    ...         print(f"Nodes: {result.single()[0]}")

Option B Compliance:
    - Integrates with existing StorageSettings for credentials
    - Uses init_schema() from Module 01 (temporal-first design)
    - Privacy-first: credentials from keyring via SecretStore
    - No mockups - production-ready with retry logic
"""

from futurnal.pkg.database.backup import PKGBackupManager
from futurnal.pkg.database.config import PKGDatabaseConfig
from futurnal.pkg.database.exceptions import (
    PKGBackupError,
    PKGConnectionError,
    PKGDatabaseError,
    PKGHealthCheckError,
    PKGRestoreError,
    PKGSchemaInitializationError,
)
from futurnal.pkg.database.manager import PKGDatabaseManager

__all__ = [
    # Configuration
    "PKGDatabaseConfig",
    # Manager
    "PKGDatabaseManager",
    # Backup
    "PKGBackupManager",
    # Exceptions
    "PKGDatabaseError",
    "PKGConnectionError",
    "PKGBackupError",
    "PKGRestoreError",
    "PKGSchemaInitializationError",
    "PKGHealthCheckError",
]
