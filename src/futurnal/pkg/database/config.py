"""PKG Database Configuration.

Pydantic-based configuration for PKG database connection pooling, backup settings,
and health monitoring. This module does NOT duplicate StorageSettings fields -
connection credentials come from StorageSettings.

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md

Option B Compliance:
- No hardcoded values - all configurable via Pydantic models
- Environment variable overrides supported
- On-device optimization with sensible defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PKGDatabaseConfig(BaseModel):
    """PKG-specific database configuration.

    This configuration extends StorageSettings with PKG-specific tuning
    for connection pooling, backup management, and health monitoring.

    Connection credentials (uri, username, password, encrypted) are NOT
    duplicated here - they come from StorageSettings when creating
    a PKGDatabaseManager.

    Example:
        >>> config = PKGDatabaseConfig()
        >>> config = PKGDatabaseConfig.with_env_overrides()
        >>> config = PKGDatabaseConfig(max_connection_pool_size=20)
    """

    # Config version for future migrations
    version: int = Field(default=1, ge=1, description="Configuration version")

    # Connection pooling (on-device optimization)
    max_connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum connections in the pool (lower for on-device)",
    )
    connection_acquisition_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Seconds to wait for a connection from the pool",
    )
    max_transaction_retry_time: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Maximum time to retry transient transaction failures",
    )
    connection_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Seconds to wait for initial connection",
    )

    # Query optimization
    fetch_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of records to fetch per batch",
    )

    # Backup configuration
    backup_path: Optional[Path] = Field(
        default=None,
        description="Backup directory path. Default: workspace_path/pkg/backups",
    )
    backup_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days to retain backups before purging",
    )
    max_backups: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of backups to retain",
    )

    # Health monitoring
    health_check_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Seconds between health checks",
    )

    # Retry configuration for connection
    max_connection_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retries for initial connection",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Initial delay between retries (exponential backoff)",
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }

    @field_validator("backup_path", mode="before")
    @classmethod
    def _validate_backup_path(cls, value: Any) -> Optional[Path]:
        """Convert string paths to Path objects."""
        if value is None:
            return None
        if isinstance(value, str):
            return Path(value)
        return value

    @model_validator(mode="after")
    def _validate_retry_config(self) -> "PKGDatabaseConfig":
        """Ensure retry configuration is sensible."""
        # Total retry time should not exceed connection timeout * retries
        max_retry_time = self.connection_timeout * self.max_connection_retries
        if max_retry_time < self.retry_delay_seconds:
            raise ValueError(
                f"retry_delay_seconds ({self.retry_delay_seconds}) should be less than "
                f"connection_timeout * max_connection_retries ({max_retry_time})"
            )
        return self

    @classmethod
    def with_env_overrides(
        cls, base: Optional["PKGDatabaseConfig"] = None
    ) -> "PKGDatabaseConfig":
        """Create configuration with environment variable overrides.

        Environment variables:
        - FUTURNAL_PKG_MAX_POOL_SIZE: max_connection_pool_size
        - FUTURNAL_PKG_CONNECTION_TIMEOUT: connection_timeout
        - FUTURNAL_PKG_BACKUP_PATH: backup_path
        - FUTURNAL_PKG_BACKUP_RETENTION_DAYS: backup_retention_days
        - FUTURNAL_PKG_MAX_BACKUPS: max_backups
        - FUTURNAL_PKG_FETCH_SIZE: fetch_size
        - FUTURNAL_PKG_HEALTH_CHECK_INTERVAL: health_check_interval_seconds

        Args:
            base: Optional base configuration to apply overrides to.
                  If None, starts with default configuration.

        Returns:
            Configuration with environment overrides applied.

        Example:
            >>> os.environ["FUTURNAL_PKG_MAX_POOL_SIZE"] = "20"
            >>> config = PKGDatabaseConfig.with_env_overrides()
            >>> assert config.max_connection_pool_size == 20
        """
        if base is None:
            data: dict[str, Any] = {}
        else:
            data = base.model_dump()

        # Apply environment overrides
        env_mappings = {
            "FUTURNAL_PKG_MAX_POOL_SIZE": ("max_connection_pool_size", int),
            "FUTURNAL_PKG_CONNECTION_TIMEOUT": ("connection_timeout", float),
            "FUTURNAL_PKG_BACKUP_PATH": ("backup_path", str),
            "FUTURNAL_PKG_BACKUP_RETENTION_DAYS": ("backup_retention_days", int),
            "FUTURNAL_PKG_MAX_BACKUPS": ("max_backups", int),
            "FUTURNAL_PKG_FETCH_SIZE": ("fetch_size", int),
            "FUTURNAL_PKG_HEALTH_CHECK_INTERVAL": (
                "health_check_interval_seconds",
                int,
            ),
            "FUTURNAL_PKG_MAX_CONNECTION_RETRIES": ("max_connection_retries", int),
            "FUTURNAL_PKG_RETRY_DELAY": ("retry_delay_seconds", float),
            "FUTURNAL_PKG_CONNECTION_ACQUISITION_TIMEOUT": (
                "connection_acquisition_timeout",
                float,
            ),
            "FUTURNAL_PKG_MAX_TRANSACTION_RETRY_TIME": (
                "max_transaction_retry_time",
                float,
            ),
        }

        for env_var, (field_name, cast_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    data[field_name] = cast_type(env_value)
                except (ValueError, TypeError):
                    # Skip invalid environment values
                    pass

        return cls.model_validate(data)

    def get_backup_path(self, workspace_path: Path) -> Path:
        """Get the effective backup path.

        If backup_path is not set, returns workspace_path/pkg/backups.

        Args:
            workspace_path: The workspace root path.

        Returns:
            The path where backups should be stored.
        """
        if self.backup_path is not None:
            return self.backup_path
        return workspace_path / "pkg" / "backups"

    def to_driver_config(self) -> dict[str, Any]:
        """Convert to Neo4j driver configuration dictionary.

        Returns configuration options suitable for passing to
        neo4j.GraphDatabase.driver().

        Returns:
            Dictionary of Neo4j driver configuration options.

        Example:
            >>> config = PKGDatabaseConfig()
            >>> driver_config = config.to_driver_config()
            >>> driver = GraphDatabase.driver(uri, auth=auth, **driver_config)
        """
        return {
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_acquisition_timeout": self.connection_acquisition_timeout,
            "max_transaction_retry_time": self.max_transaction_retry_time,
            "connection_timeout": self.connection_timeout,
        }
