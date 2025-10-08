"""Tests for orchestrator configuration management."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from pydantic import ValidationError

from futurnal.orchestrator.config import (
    ConfigurationError,
    ConfigurationManager,
    ConfigMigrationManager,
    OrchestratorConfig,
    QuarantineConfig,
    QueueConfig,
    RetryConfig,
    SecretsManager,
    SecurityConfig,
    TelemetryConfig,
    WorkerConfig,
)
from futurnal.privacy.audit import AuditLogger


class TestWorkerConfig:
    """Tests for WorkerConfig validation."""

    def test_default_values(self) -> None:
        """Test default worker configuration values."""
        config = WorkerConfig()
        assert config.max_workers == 8
        assert config.hardware_cap_enabled is True

    def test_valid_worker_counts(self) -> None:
        """Test valid worker count range."""
        config = WorkerConfig(max_workers=1)
        assert config.max_workers == 1

        config = WorkerConfig(max_workers=32)
        assert config.max_workers == 32

    def test_invalid_worker_count_too_high(self) -> None:
        """Test validation rejects worker count > 32."""
        with pytest.raises(ValidationError) as exc_info:
            WorkerConfig(max_workers=33)
        assert "less than or equal to 32" in str(exc_info.value)

    def test_invalid_worker_count_too_low(self) -> None:
        """Test validation rejects worker count < 1."""
        with pytest.raises(ValidationError):
            WorkerConfig(max_workers=0)


class TestQueueConfig:
    """Tests for QueueConfig validation."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default queue configuration values."""
        config = QueueConfig()
        assert config.database_path.name == "queue.db"
        assert config.wal_mode is True
        assert config.checkpoint_interval == 100

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that queue config creates parent directory."""
        db_path = tmp_path / "subdir" / "queue.db"
        config = QueueConfig(database_path=db_path)
        assert config.database_path.parent.exists()

    def test_valid_checkpoint_interval(self) -> None:
        """Test valid checkpoint interval range."""
        config = QueueConfig(checkpoint_interval=1)
        assert config.checkpoint_interval == 1

        config = QueueConfig(checkpoint_interval=10000)
        assert config.checkpoint_interval == 10000

    def test_invalid_checkpoint_interval(self) -> None:
        """Test validation rejects invalid checkpoint intervals."""
        with pytest.raises(ValidationError):
            QueueConfig(checkpoint_interval=0)

        with pytest.raises(ValidationError):
            QueueConfig(checkpoint_interval=10001)


class TestRetryConfig:
    """Tests for RetryConfig validation."""

    def test_default_values(self) -> None:
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay_seconds == 60
        assert config.backoff_multiplier == 2.0

    def test_valid_ranges(self) -> None:
        """Test valid configuration ranges."""
        config = RetryConfig(
            max_attempts=10,
            base_delay_seconds=3600,
            backoff_multiplier=10.0
        )
        assert config.max_attempts == 10
        assert config.base_delay_seconds == 3600
        assert config.backoff_multiplier == 10.0

    def test_invalid_max_attempts(self) -> None:
        """Test validation rejects invalid max attempts."""
        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=0)

        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=11)

    def test_invalid_backoff_multiplier(self) -> None:
        """Test validation rejects invalid backoff multiplier."""
        with pytest.raises(ValidationError):
            RetryConfig(backoff_multiplier=0.5)

        with pytest.raises(ValidationError):
            RetryConfig(backoff_multiplier=11.0)


class TestQuarantineConfig:
    """Tests for QuarantineConfig validation."""

    def test_default_values(self) -> None:
        """Test default quarantine configuration values."""
        config = QuarantineConfig()
        assert config.enabled is True
        assert config.retention_days == 30
        assert config.auto_purge is False

    def test_valid_retention_days(self) -> None:
        """Test valid retention day range."""
        config = QuarantineConfig(retention_days=1)
        assert config.retention_days == 1

        config = QuarantineConfig(retention_days=365)
        assert config.retention_days == 365

    def test_invalid_retention_days(self) -> None:
        """Test validation rejects invalid retention days."""
        with pytest.raises(ValidationError):
            QuarantineConfig(retention_days=0)

        with pytest.raises(ValidationError):
            QuarantineConfig(retention_days=366)


class TestTelemetryConfig:
    """Tests for TelemetryConfig validation."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default telemetry configuration values."""
        config = TelemetryConfig()
        assert config.enabled is True
        assert config.retention_days == 90

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that telemetry config creates output directory."""
        output_dir = tmp_path / "telemetry"
        config = TelemetryConfig(output_dir=output_dir)
        assert config.output_dir.exists()

    def test_valid_retention_days(self) -> None:
        """Test valid retention day range."""
        config = TelemetryConfig(retention_days=1)
        assert config.retention_days == 1

        config = TelemetryConfig(retention_days=365)
        assert config.retention_days == 365

    def test_invalid_retention_days(self) -> None:
        """Test validation rejects invalid retention days."""
        with pytest.raises(ValidationError):
            TelemetryConfig(retention_days=0)

        with pytest.raises(ValidationError):
            TelemetryConfig(retention_days=366)


class TestSecurityConfig:
    """Tests for SecurityConfig validation."""

    def test_default_values(self) -> None:
        """Test default security configuration values."""
        config = SecurityConfig()
        assert config.audit_logging is True
        assert config.path_redaction is True
        assert config.consent_required is True

    def test_all_features_enabled_by_default(self) -> None:
        """Test secure defaults - all security features enabled."""
        config = SecurityConfig()
        assert config.audit_logging is True
        assert config.path_redaction is True
        assert config.consent_required is True


class TestOrchestratorConfig:
    """Tests for main OrchestratorConfig."""

    def test_default_configuration(self) -> None:
        """Test default orchestrator configuration."""
        config = OrchestratorConfig()
        assert config.version == 2
        assert isinstance(config.workers, WorkerConfig)
        assert isinstance(config.queue, QueueConfig)
        assert isinstance(config.retry, RetryConfig)
        assert isinstance(config.quarantine, QuarantineConfig)
        assert isinstance(config.telemetry, TelemetryConfig)
        assert isinstance(config.security, SecurityConfig)

    def test_custom_configuration(self) -> None:
        """Test custom orchestrator configuration."""
        config = OrchestratorConfig(
            workers=WorkerConfig(max_workers=4),
            retry=RetryConfig(max_attempts=5),
        )
        assert config.workers.max_workers == 4
        assert config.retry.max_attempts == 5

    def test_rejects_unknown_fields(self) -> None:
        """Test that unknown fields are rejected."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(unknown_field="value")  # type: ignore

    def test_validates_on_assignment(self) -> None:
        """Test that validation occurs on field assignment."""
        config = OrchestratorConfig()
        with pytest.raises(ValidationError):
            config.workers = "invalid"  # type: ignore


class TestConfigurationManager:
    """Tests for ConfigurationManager."""

    def test_load_default_config(self, tmp_path: Path) -> None:
        """Test loading default configuration when file doesn't exist."""
        config_path = tmp_path / "config.yaml"
        manager = ConfigurationManager(config_path=config_path)
        config = manager.load()

        assert isinstance(config, OrchestratorConfig)
        assert config.version == 2
        assert config.workers.max_workers == 8

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading configuration from YAML file."""
        config_path = tmp_path / "config.yaml"
        data = {
            "version": 2,
            "workers": {"max_workers": 4, "hardware_cap_enabled": True},
            "retry": {"max_attempts": 5, "base_delay_seconds": 120, "backoff_multiplier": 2.0},
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        manager = ConfigurationManager(config_path=config_path)
        config = manager.load()

        assert config.workers.max_workers == 4
        assert config.retry.max_attempts == 5
        assert config.retry.base_delay_seconds == 120

    def test_load_invalid_config(self, tmp_path: Path) -> None:
        """Test loading invalid configuration raises error."""
        config_path = tmp_path / "config.yaml"
        data = {
            "version": 2,
            "workers": {"max_workers": 100},  # Invalid: exceeds max
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        manager = ConfigurationManager(config_path=config_path)
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load()
        assert "Invalid configuration" in str(exc_info.value)

    def test_save_configuration(self, tmp_path: Path) -> None:
        """Test saving configuration to file."""
        config_path = tmp_path / "config.yaml"
        manager = ConfigurationManager(config_path=config_path)

        config = OrchestratorConfig(
            workers=WorkerConfig(max_workers=16)
        )
        manager.save(config)

        assert config_path.exists()

        # Reload and verify
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["workers"]["max_workers"] == 16

    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        """Test validation of nonexistent file."""
        config_path = tmp_path / "nonexistent.yaml"
        manager = ConfigurationManager(config_path=config_path)
        errors = manager.validate()

        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_validate_valid_file(self, tmp_path: Path) -> None:
        """Test validation of valid configuration file."""
        config_path = tmp_path / "config.yaml"
        config = OrchestratorConfig()
        manager = ConfigurationManager(config_path=config_path)
        manager.save(config)

        errors = manager.validate()
        assert len(errors) == 0

    def test_validate_invalid_file(self, tmp_path: Path) -> None:
        """Test validation of invalid configuration file."""
        config_path = tmp_path / "config.yaml"
        data = {
            "version": 2,
            "workers": {"max_workers": 100},  # Invalid
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        manager = ConfigurationManager(config_path=config_path)
        errors = manager.validate()

        assert len(errors) > 0
        assert any("max_workers" in err for err in errors)

    def test_audit_logging_on_load(self, tmp_path: Path) -> None:
        """Test that configuration loads are audit logged."""
        config_path = tmp_path / "config.yaml"
        audit_logger = AuditLogger(tmp_path / "audit")

        manager = ConfigurationManager(
            config_path=config_path,
            audit_logger=audit_logger
        )
        manager.load()

        # Verify audit log was created
        audit_file = tmp_path / "audit" / "audit.log"
        assert audit_file.exists()

        with open(audit_file, encoding="utf-8") as f:
            log_entry = json.loads(f.readline())
        assert log_entry["action"] == "load_configuration"
        assert log_entry["status"] == "succeeded"

    def test_audit_logging_on_save(self, tmp_path: Path) -> None:
        """Test that configuration saves are audit logged."""
        config_path = tmp_path / "config.yaml"
        audit_logger = AuditLogger(tmp_path / "audit")

        manager = ConfigurationManager(
            config_path=config_path,
            audit_logger=audit_logger
        )
        config = OrchestratorConfig()
        manager.save(config)

        # Verify audit log contains save event
        audit_file = tmp_path / "audit" / "audit.log"
        with open(audit_file, encoding="utf-8") as f:
            lines = f.readlines()

        save_event = json.loads(lines[-1])
        assert save_event["action"] == "save_configuration"
        assert save_event["status"] == "succeeded"


class TestConfigMigrationManager:
    """Tests for ConfigMigrationManager."""

    def test_no_migration_needed(self, tmp_path: Path) -> None:
        """Test migration when config is already at current version."""
        config_path = tmp_path / "config.yaml"
        data = {"version": ConfigMigrationManager.CURRENT_VERSION}

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        migrator = ConfigMigrationManager()
        config = migrator.migrate(config_path)

        assert config.version == ConfigMigrationManager.CURRENT_VERSION

    def test_migrate_v1_to_v2(self, tmp_path: Path) -> None:
        """Test migration from version 1 to version 2."""
        config_path = tmp_path / "config.yaml"
        data = {
            "version": 1,
            "max_retries": 5,  # Old field name
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        migrator = ConfigMigrationManager()
        config = migrator.migrate(config_path)

        # Check migration occurred
        assert config.version == 2
        assert config.retry.max_attempts == 5

        # Check quarantine was added
        assert config.quarantine.enabled is True
        assert config.quarantine.retention_days == 30

        # Check security was added
        assert config.security.audit_logging is True

    def test_migration_creates_backup(self, tmp_path: Path) -> None:
        """Test that migration creates a backup file."""
        config_path = tmp_path / "config.yaml"
        data = {"version": 1}

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        migrator = ConfigMigrationManager()
        migrator.migrate(config_path)

        backup_path = config_path.with_suffix(".yaml.backup")
        assert backup_path.exists()

        # Verify backup contains original data
        with open(backup_path, encoding="utf-8") as f:
            backup_data = yaml.safe_load(f)
        assert backup_data["version"] == 1


class TestSecretsManager:
    """Tests for SecretsManager."""

    def test_get_secret_from_environment(self) -> None:
        """Test retrieving secret from environment variable."""
        os.environ["FUTURNAL_TEST_SECRET"] = "env_value"
        try:
            manager = SecretsManager()
            value = manager.get_secret("test_secret")
            assert value == "env_value"
        finally:
            del os.environ["FUTURNAL_TEST_SECRET"]

    def test_get_nonexistent_secret(self) -> None:
        """Test retrieving nonexistent secret returns None."""
        manager = SecretsManager()
        value = manager.get_secret("nonexistent_secret")
        assert value is None

    def test_environment_takes_priority(self) -> None:
        """Test that environment variables take priority over keyring."""
        os.environ["FUTURNAL_PRIORITY_TEST"] = "env_value"
        try:
            manager = SecretsManager()
            # Even if keyring has a value, env should take priority
            value = manager.get_secret("priority_test")
            assert value == "env_value"
        finally:
            del os.environ["FUTURNAL_PRIORITY_TEST"]


class TestSecureDefaults:
    """Security tests for configuration defaults."""

    def test_all_security_features_enabled(self) -> None:
        """Test that all security features are enabled by default."""
        config = OrchestratorConfig()

        assert config.security.audit_logging is True
        assert config.security.path_redaction is True
        assert config.security.consent_required is True

    def test_quarantine_enabled_by_default(self) -> None:
        """Test that quarantine is enabled by default."""
        config = OrchestratorConfig()
        assert config.quarantine.enabled is True

    def test_manual_purge_by_default(self) -> None:
        """Test that auto-purge is disabled by default for safety."""
        config = OrchestratorConfig()
        assert config.quarantine.auto_purge is False

    def test_wal_mode_enabled(self) -> None:
        """Test that WAL mode is enabled for durability."""
        config = OrchestratorConfig()
        assert config.queue.wal_mode is True

    def test_hardware_cap_enabled(self) -> None:
        """Test that hardware cap is enabled by default."""
        config = OrchestratorConfig()
        assert config.workers.hardware_cap_enabled is True


class TestIntegration:
    """Integration tests for full configuration workflow."""

    def test_full_configuration_lifecycle(self, tmp_path: Path) -> None:
        """Test complete configuration lifecycle: create, save, load, validate."""
        config_path = tmp_path / "config.yaml"
        audit_logger = AuditLogger(tmp_path / "audit")

        # Create and save
        manager = ConfigurationManager(
            config_path=config_path,
            audit_logger=audit_logger
        )
        config = OrchestratorConfig(
            workers=WorkerConfig(max_workers=16),
            retry=RetryConfig(max_attempts=5)
        )
        manager.save(config)

        # Validate
        errors = manager.validate()
        assert len(errors) == 0

        # Load and verify
        loaded_config = manager.load()
        assert loaded_config.workers.max_workers == 16
        assert loaded_config.retry.max_attempts == 5

        # Verify audit trail
        audit_file = tmp_path / "audit" / "audit.log"
        with open(audit_file, encoding="utf-8") as f:
            events = [json.loads(line) for line in f]

        assert len(events) >= 2
        assert events[0]["action"] == "save_configuration"
        assert events[-1]["action"] == "load_configuration"

    def test_configuration_error_handling(self, tmp_path: Path) -> None:
        """Test error handling for invalid configurations."""
        config_path = tmp_path / "config.yaml"

        # Write malformed YAML
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("invalid: yaml: content:")

        manager = ConfigurationManager(config_path=config_path)
        errors = manager.validate()

        assert len(errors) > 0

    def test_migration_preserves_valid_settings(self, tmp_path: Path) -> None:
        """Test that migration preserves valid existing settings."""
        config_path = tmp_path / "config.yaml"

        # Create v1 config with custom settings
        data = {
            "version": 1,
            "workers": {"max_workers": 12},
            "queue": {"wal_mode": False},
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        # Migrate
        migrator = ConfigMigrationManager()
        config = migrator.migrate(config_path)

        # Verify custom settings preserved
        assert config.workers.max_workers == 12
        assert config.queue.wal_mode is False

        # Verify new fields added
        assert config.quarantine.enabled is True
        assert config.security.audit_logging is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
