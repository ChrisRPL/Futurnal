Summary: Implement secure configuration management with validation, secure defaults, and migration support for orchestrator settings.

# 09 · Configuration & Security

## Purpose
Establish secure configuration management for the orchestrator with validation schemas, secure defaults, secrets management integration, and configuration migration support. Ensures the Ghost's experiential learning pipeline operates with security best practices and prevents misconfiguration.

## Scope
- Configuration schema with Pydantic validation
- Secure default values for all settings
- Secrets management integration (environment variables, keyring)
- Configuration validation on load
- Configuration migration for schema evolution
- Audit logging for configuration changes
- Configuration documentation and examples
- Security hardening guidelines

## Requirements Alignment
- **Configuration Security**: Validate secure configuration patterns
- **Secrets Management**: Never store secrets in plain files
- **Audit Trail**: Log configuration changes
- **Best Practices**: Enforce secure defaults

## Configuration Schema

### OrchestratorConfig
```python
from pydantic import BaseModel, Field, SecretStr, validator

class WorkerConfig(BaseModel):
    """Worker pool configuration."""
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
    """Job queue configuration."""
    database_path: Path = Field(
        default=Path("~/.futurnal/queue.db").expanduser(),
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

class RetryConfig(BaseModel):
    """Retry policy configuration."""
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
    """Quarantine configuration."""
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
    """Telemetry configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable telemetry collection"
    )
    output_dir: Path = Field(
        default=Path("~/.futurnal/telemetry").expanduser(),
        description="Telemetry output directory"
    )
    retention_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Days to retain telemetry"
    )

class SecurityConfig(BaseModel):
    """Security configuration."""
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
    """Main orchestrator configuration."""
    workers: WorkerConfig = Field(default_factory=WorkerConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    quarantine: QuarantineConfig = Field(default_factory=QuarantineConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @validator("workers")
    def validate_workers(cls, v):
        """Validate worker configuration."""
        if v.max_workers > 32:
            raise ValueError("max_workers must not exceed 32")
        return v

    @validator("queue")
    def validate_queue(cls, v):
        """Validate queue configuration."""
        if not v.database_path.parent.exists():
            v.database_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        """Pydantic configuration."""
        validate_assignment = True  # Validate on field assignment
        extra = "forbid"  # Reject unknown fields
```

## Configuration Loading

### ConfigurationManager
```python
class ConfigurationManager:
    """Manages orchestrator configuration with validation."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self._config_path = config_path or Path("~/.futurnal/config/orchestrator.yaml").expanduser()
        self._audit = audit_logger
        self._config: Optional[OrchestratorConfig] = None

    def load(self) -> OrchestratorConfig:
        """Load and validate configuration."""
        if self._config_path.exists():
            # Load from file
            with open(self._config_path) as f:
                data = yaml.safe_load(f)

            # Validate and parse
            try:
                self._config = OrchestratorConfig(**data)
                logger.info(
                    "Configuration loaded",
                    extra={"config_path": str(self._config_path)},
                )
            except ValidationError as exc:
                logger.error(
                    "Configuration validation failed",
                    extra={"errors": exc.errors()},
                )
                raise ConfigurationError(f"Invalid configuration: {exc}")

        else:
            # Use defaults
            logger.info("Using default configuration")
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
        """Save configuration to file."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        data = config.dict()

        # Write to file
        with open(self._config_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

        logger.info(
            "Configuration saved",
            extra={"config_path": str(self._config_path)},
        )

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
        """Validate configuration without loading."""
        path = config_path or self._config_path
        errors = []

        if not path.exists():
            errors.append(f"Configuration file not found: {path}")
            return errors

        try:
            with open(path) as f:
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
```

## Secrets Management

### SecretsManager
```python
class SecretsManager:
    """Manages secrets with secure storage."""

    def __init__(self):
        self._keyring_service = "futurnal.orchestrator"

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve secret from keyring or environment."""
        # Try environment variable first
        env_value = os.getenv(f"FUTURNAL_{key.upper()}")
        if env_value:
            return env_value

        # Try keyring
        try:
            import keyring
            value = keyring.get_password(self._keyring_service, key)
            return value
        except Exception as exc:
            logger.warning(
                "Failed to retrieve secret from keyring",
                extra={"key": key, "error": str(exc)},
            )
            return None

    def set_secret(self, key: str, value: str) -> None:
        """Store secret in keyring."""
        try:
            import keyring
            keyring.set_password(self._keyring_service, key, value)
            logger.info(
                "Secret stored in keyring",
                extra={"key": key},
            )
        except Exception as exc:
            logger.error(
                "Failed to store secret in keyring",
                extra={"key": key, "error": str(exc)},
            )
            raise

    def delete_secret(self, key: str) -> None:
        """Delete secret from keyring."""
        try:
            import keyring
            keyring.delete_password(self._keyring_service, key)
            logger.info(
                "Secret deleted from keyring",
                extra={"key": key},
            )
        except Exception as exc:
            logger.warning(
                "Failed to delete secret from keyring",
                extra={"key": key, "error": str(exc)},
            )
```

## Configuration Migration

### ConfigMigrationManager
```python
class ConfigMigrationManager:
    """Handles configuration schema migrations."""

    CURRENT_VERSION = 2

    def migrate(self, config_path: Path) -> OrchestratorConfig:
        """Migrate configuration to current version."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        version = data.get("version", 1)

        if version == self.CURRENT_VERSION:
            return OrchestratorConfig(**data)

        logger.info(
            "Migrating configuration",
            extra={"from_version": version, "to_version": self.CURRENT_VERSION},
        )

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
        with open(config_path, "w") as f:
            yaml.safe_dump(data, f)

        logger.info(
            "Configuration migrated",
            extra={"backup_path": str(backup_path)},
        )

        return OrchestratorConfig(**data)

    def _migrate_v1_to_v2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1 to 2."""
        # Example: Add new fields with defaults
        if "quarantine" not in data:
            data["quarantine"] = {
                "enabled": True,
                "retention_days": 30,
                "auto_purge": False,
            }

        # Example: Rename fields
        if "max_retries" in data:
            data["retry"] = data.get("retry", {})
            data["retry"]["max_attempts"] = data.pop("max_retries")

        return data
```

## Secure Defaults

### Security Hardening
```yaml
# ~/.futurnal/config/orchestrator.yaml
version: 2

workers:
  max_workers: 8
  hardware_cap_enabled: true  # Respect system limits

queue:
  database_path: ~/.futurnal/queue.db
  wal_mode: true  # Durability
  checkpoint_interval: 100

retry:
  max_attempts: 3
  base_delay_seconds: 60
  backoff_multiplier: 2.0

quarantine:
  enabled: true  # Isolate persistent failures
  retention_days: 30
  auto_purge: false  # Manual purge only

telemetry:
  enabled: true
  output_dir: ~/.futurnal/telemetry
  retention_days: 90

security:
  audit_logging: true  # Always audit
  path_redaction: true  # Privacy-aware logs
  consent_required: true  # Explicit consent
```

## CLI Commands

### Config Validation
```bash
futurnal config validate [OPTIONS]

Validate orchestrator configuration.

Options:
  --config PATH    Configuration file path (default: ~/.futurnal/config/orchestrator.yaml)
  --verbose        Show detailed validation output

Example:
  futurnal config validate
  futurnal config validate --config custom-config.yaml
```

### Config Display
```bash
futurnal config show [OPTIONS]

Display current configuration.

Options:
  --section TEXT   Show specific section (workers, queue, retry, etc.)
  --format [yaml|json]  Output format (default: yaml)

Example:
  futurnal config show
  futurnal config show --section retry
```

### Config Migration
```bash
futurnal config migrate [OPTIONS]

Migrate configuration to current schema version.

Options:
  --config PATH    Configuration file path
  --dry-run        Show migration without applying

Example:
  futurnal config migrate
  futurnal config migrate --dry-run
```

## Acceptance Criteria

- ✅ Configuration schema with Pydantic validation
- ✅ Secure defaults for all settings
- ✅ Secrets management via keyring/environment
- ✅ Configuration validation on load
- ✅ Configuration migration support
- ✅ Audit logging for config changes
- ✅ CLI commands for config management
- ✅ Configuration documentation
- ✅ Security hardening guidelines
- ✅ Example configurations provided

## Test Plan

### Unit Tests
- `test_config_validation.py`: Schema validation
- `test_secure_defaults.py`: Default values
- `test_secrets_management.py`: Keyring integration
- `test_config_migration.py`: Schema evolution

### Integration Tests
- `test_config_loading.py`: End-to-end config load
- `test_invalid_config.py`: Error handling
- `test_config_audit.py`: Audit event logging

### Security Tests
- `test_no_secrets_in_files.py`: Secrets not persisted
- `test_path_redaction.py`: Paths redacted in logs
- `test_secure_defaults.py`: Security settings enforced

## Implementation Notes

### Configuration Validation Examples
```python
# Valid configuration
config = OrchestratorConfig(
    workers=WorkerConfig(max_workers=4),
    retry=RetryConfig(max_attempts=5),
)

# Invalid configuration (raises ValidationError)
try:
    config = OrchestratorConfig(
        workers=WorkerConfig(max_workers=100),  # Exceeds limit
    )
except ValidationError as exc:
    print(exc.errors())
```

### Environment Variable Override
```bash
# Override configuration with environment variables
export FUTURNAL_MAX_WORKERS=4
export FUTURNAL_RETRY_MAX_ATTEMPTS=5

futurnal orchestrator start
```

## Open Questions

- Should configuration be hot-reloadable (no restart required)?
- How to handle configuration conflicts between file and environment?
- Should we provide configuration templates for common scenarios?
- What's the appropriate validation level (strict/lenient)?
- Should we implement configuration encryption at rest?
- How to handle multi-environment configurations (dev/staging/prod)?

## Dependencies

- Pydantic for schema validation
- PyYAML for configuration parsing
- Keyring library for secrets storage
- AuditLogger for configuration changes


