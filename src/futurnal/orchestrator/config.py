"""Orchestrator configuration management with validation, security, and migration.

Implements secure configuration management for the orchestrator with Pydantic validation,
secure defaults, secrets management integration, and configuration migration support.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..privacy.audit import AuditEvent, AuditLogger

try:
    import keyring
except ModuleNotFoundError:
    keyring = None  # type: ignore[assignment]


class WorkerConfig(BaseModel):
    """Worker pool configuration.

    Attributes:
        max_workers: Maximum concurrent workers (1-32)
        hardware_cap_enabled: Respect hardware CPU count limits
    """

    max_workers: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum concurrent workers"
    )
    hardware_cap_enabled: bool = Field(
        default=True,
        description="Respect hardware CPU count limits"
    )


class QueueConfig(BaseModel):
    """Job queue configuration.

    Attributes:
        database_path: SQLite database path
        wal_mode: Enable SQLite WAL mode for durability
        checkpoint_interval: WAL checkpoint frequency (transactions)
    """

    database_path: Path = Field(
        default=Path.home() / ".futurnal" / "queue.db",
        description="SQLite database path"
    )
    wal_mode: bool = Field(
        default=True,
        description="Enable SQLite WAL mode for durability"
    )
    checkpoint_interval: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="WAL checkpoint frequency (transactions)"
    )

    @field_validator("database_path")
    @classmethod
    def validate_database_path(cls, v: Path) -> Path:
        """Ensure database parent directory exists."""
        resolved = v.expanduser().resolve()
        if not resolved.parent.exists():
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved


class RetryConfig(BaseModel):
    """Retry policy configuration.

    Attributes:
        max_attempts: Maximum retry attempts (1-10)
        base_delay_seconds: Base retry delay in seconds (1-3600)
        backoff_multiplier: Exponential backoff multiplier (1.0-10.0)
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )
    base_delay_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Base retry delay"
    )
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff multiplier"
    )


class QuarantineConfig(BaseModel):
    """Quarantine configuration.

    Attributes:
        enabled: Enable quarantine system
        retention_days: Days to retain quarantined jobs (1-365)
        auto_purge: Automatically purge old quarantined jobs
    """

    enabled: bool = Field(
        default=True,
        description="Enable quarantine system"
    )
    retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days to retain quarantined jobs"
    )
    auto_purge: bool = Field(
        default=False,
        description="Automatically purge old quarantined jobs"
    )


class TelemetryConfig(BaseModel):
    """Telemetry configuration.

    Attributes:
        enabled: Enable telemetry collection
        output_dir: Telemetry output directory
        retention_days: Days to retain telemetry (1-365)
    """

    enabled: bool = Field(
        default=True,
        description="Enable telemetry collection"
    )
    output_dir: Path = Field(
        default=Path.home() / ".futurnal" / "telemetry",
        description="Telemetry output directory"
    )
    retention_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Days to retain telemetry"
    )

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure telemetry directory exists."""
        resolved = v.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


class SecurityConfig(BaseModel):
    """Security configuration.

    Attributes:
        audit_logging: Enable audit logging
        path_redaction: Redact paths in logs
        consent_required: Require explicit consent for data processing
    """

    audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    path_redaction: bool = Field(
        default=True,
        description="Redact paths in logs"
    )
    consent_required: bool = Field(
        default=True,
        description="Require explicit consent for data processing"
    )


class OrchestratorConfig(BaseModel):
    """Main orchestrator configuration.

    Combines all orchestrator subsystem configurations with validation
    and secure defaults.

    Attributes:
        version: Configuration schema version
        workers: Worker pool configuration
        queue: Job queue configuration
        retry: Retry policy configuration
        quarantine: Quarantine configuration
        telemetry: Telemetry configuration
        security: Security configuration
    """

    version: int = Field(
        default=2,
        description="Configuration schema version"
    )
    workers: WorkerConfig = Field(default_factory=WorkerConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    quarantine: QuarantineConfig = Field(default_factory=QuarantineConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    class Config:
        """Pydantic configuration."""
        validate_assignment = True  # Validate on field assignment
        extra = "forbid"  # Reject unknown fields


class ConfigurationManager:
    """Manages orchestrator configuration with validation and audit logging.

    Provides load, save, and validation operations for orchestrator configuration
    with automatic audit trail generation.

    Attributes:
        config_path: Path to configuration file
        audit_logger: Optional audit logger for configuration changes
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """Initialize configuration manager.

        Args:
            config_path: Path to config file (default: ~/.futurnal/config/orchestrator.yaml)
            audit_logger: Optional audit logger for tracking config changes
        """
        self._config_path = config_path or (
            Path.home() / ".futurnal" / "config" / "orchestrator.yaml"
        )
        self._audit = audit_logger
        self._config: Optional[OrchestratorConfig] = None

    def load(self) -> OrchestratorConfig:
        """Load and validate configuration.

        Returns:
            Validated orchestrator configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self._config_path.exists():
            # Load from file
            with open(self._config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Validate and parse
            try:
                self._config = OrchestratorConfig(**data)
            except ValidationError as exc:
                error_details = [
                    f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                    for err in exc.errors()
                ]
                raise ConfigurationError(
                    f"Invalid configuration: {'; '.join(error_details)}"
                ) from exc
        else:
            # Use defaults
            self._config = OrchestratorConfig()

        # Audit log
        if self._audit:
            self._audit.record(
                AuditEvent(
                    job_id=f"config_load_{datetime.utcnow().isoformat()}",
                    source="configuration_manager",
                    action="load_configuration",
                    status="succeeded",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "config_path": str(self._config_path),
                        "using_defaults": not self._config_path.exists(),
                    },
                )
            )

        return self._config

    def save(self, config: OrchestratorConfig) -> None:
        """Save configuration to file.

        Args:
            config: Configuration to save
        """
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and serialize Path objects as strings
        data = config.model_dump(mode="json")

        # Write to file
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        # Audit log
        if self._audit:
            self._audit.record(
                AuditEvent(
                    job_id=f"config_save_{datetime.utcnow().isoformat()}",
                    source="configuration_manager",
                    action="save_configuration",
                    status="succeeded",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "config_path": str(self._config_path),
                    },
                )
            )

    def validate(self, config_path: Optional[Path] = None) -> List[str]:
        """Validate configuration without loading.

        Args:
            config_path: Optional path to config file to validate

        Returns:
            List of validation errors (empty if valid)
        """
        path = config_path or self._config_path
        errors = []

        if not path.exists():
            errors.append(f"Configuration file not found: {path}")
            return errors

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            OrchestratorConfig(**data)
        except ValidationError as exc:
            for error in exc.errors():
                loc = ".".join(str(l) for l in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")
        except Exception as exc:
            errors.append(f"Failed to load configuration: {exc}")

        return errors


class ConfigMigrationManager:
    """Handles configuration schema migrations.

    Provides versioned migration support with automatic backups.
    """

    CURRENT_VERSION = 2

    def migrate(self, config_path: Path) -> OrchestratorConfig:
        """Migrate configuration to current version.

        Args:
            config_path: Path to configuration file

        Returns:
            Migrated configuration

        Raises:
            ConfigurationError: If migration fails
        """
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        version = data.get("version", 1)

        if version == self.CURRENT_VERSION:
            return OrchestratorConfig(**data)

        # Apply migrations
        if version == 1:
            data = self._migrate_v1_to_v2(data)
            version = 2

        # Update version
        data["version"] = self.CURRENT_VERSION

        # Backup old config
        backup_path = config_path.with_suffix(".yaml.backup")
        shutil.copy(config_path, backup_path)

        # Save migrated config
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        return OrchestratorConfig(**data)

    def _migrate_v1_to_v2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1 to 2.

        Args:
            data: Configuration data

        Returns:
            Migrated configuration data
        """
        # Add quarantine configuration if missing
        if "quarantine" not in data:
            data["quarantine"] = {
                "enabled": True,
                "retention_days": 30,
                "auto_purge": False,
            }

        # Migrate legacy retry settings
        if "max_retries" in data:
            data.setdefault("retry", {})
            data["retry"]["max_attempts"] = data.pop("max_retries")

        # Add security configuration if missing
        if "security" not in data:
            data["security"] = {
                "audit_logging": True,
                "path_redaction": True,
                "consent_required": True,
            }

        return data


class SecretsManager:
    """Manages secrets with secure storage.

    Provides secure secret storage with environment variable priority
    and keyring fallback.

    Attributes:
        keyring_service: Service name for keyring storage
    """

    def __init__(self, keyring_service: str = "futurnal.orchestrator"):
        """Initialize secrets manager.

        Args:
            keyring_service: Service name for keyring storage
        """
        self._keyring_service = keyring_service

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve secret from keyring or environment.

        Environment variables take priority over keyring storage.

        Args:
            key: Secret key

        Returns:
            Secret value or None if not found
        """
        # Try environment variable first (with FUTURNAL_ prefix)
        env_value = os.getenv(f"FUTURNAL_{key.upper()}")
        if env_value:
            return env_value

        # Try keyring
        if keyring is not None:
            try:
                value = keyring.get_password(self._keyring_service, key)
                return value
            except Exception:
                # Silently fall through on keyring errors
                pass

        return None

    def set_secret(self, key: str, value: str) -> None:
        """Store secret in keyring.

        Args:
            key: Secret key
            value: Secret value

        Raises:
            RuntimeError: If keyring is not available or storage fails
        """
        if keyring is None:
            raise RuntimeError("Keyring not available for secret storage")

        try:
            keyring.set_password(self._keyring_service, key, value)
        except Exception as exc:
            raise RuntimeError(f"Failed to store secret in keyring: {exc}") from exc

    def delete_secret(self, key: str) -> None:
        """Delete secret from keyring.

        Args:
            key: Secret key
        """
        if keyring is None:
            return

        try:
            keyring.delete_password(self._keyring_service, key)
        except Exception:
            # Silently ignore deletion errors (secret may not exist)
            pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


__all__ = [
    "WorkerConfig",
    "QueueConfig",
    "RetryConfig",
    "QuarantineConfig",
    "TelemetryConfig",
    "SecurityConfig",
    "OrchestratorConfig",
    "ConfigurationManager",
    "ConfigMigrationManager",
    "SecretsManager",
    "ConfigurationError",
]
