"""Tests for PKG Database Configuration.

Tests PKGDatabaseConfig Pydantic model including:
- Default values for on-device operation
- Boundary value validation
- Environment variable overrides
- Invalid configuration rejection

Follows production plan testing strategy:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from futurnal.pkg.database.config import PKGDatabaseConfig


class TestPKGDatabaseConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Verify sensible defaults for on-device operation."""
        config = PKGDatabaseConfig()

        # Connection pooling defaults optimized for on-device
        assert config.max_connection_pool_size == 10
        assert config.connection_acquisition_timeout == 60.0
        assert config.max_transaction_retry_time == 30.0
        assert config.connection_timeout == 30.0

        # Query optimization
        assert config.fetch_size == 1000

        # Backup defaults
        assert config.backup_path is None  # Uses workspace default
        assert config.backup_retention_days == 30
        assert config.max_backups == 10

        # Health monitoring
        assert config.health_check_interval_seconds == 60

        # Retry configuration
        assert config.max_connection_retries == 3
        assert config.retry_delay_seconds == 1.0

    def test_version_default(self):
        """Config version should be 1."""
        config = PKGDatabaseConfig()
        assert config.version == 1


class TestPKGDatabaseConfigBoundaryValues:
    """Test boundary value constraints."""

    def test_max_connection_pool_size_boundaries(self):
        """Pool size must be between 1 and 50."""
        # Valid boundaries
        PKGDatabaseConfig(max_connection_pool_size=1)
        PKGDatabaseConfig(max_connection_pool_size=50)

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(max_connection_pool_size=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(max_connection_pool_size=51)

    def test_connection_timeout_boundaries(self):
        """Connection timeout must be between 5 and 120 seconds."""
        # Valid boundaries
        PKGDatabaseConfig(connection_timeout=5.0)
        PKGDatabaseConfig(connection_timeout=120.0)

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(connection_timeout=4.9)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(connection_timeout=120.1)

    def test_fetch_size_boundaries(self):
        """Fetch size must be between 100 and 10000."""
        # Valid boundaries
        PKGDatabaseConfig(fetch_size=100)
        PKGDatabaseConfig(fetch_size=10000)

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(fetch_size=99)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(fetch_size=10001)

    def test_backup_retention_days_boundaries(self):
        """Backup retention must be between 1 and 365 days."""
        # Valid boundaries
        PKGDatabaseConfig(backup_retention_days=1)
        PKGDatabaseConfig(backup_retention_days=365)

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(backup_retention_days=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(backup_retention_days=366)

    def test_max_backups_boundaries(self):
        """Max backups must be between 1 and 100."""
        # Valid boundaries
        PKGDatabaseConfig(max_backups=1)
        PKGDatabaseConfig(max_backups=100)

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(max_backups=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(max_backups=101)

    def test_health_check_interval_boundaries(self):
        """Health check interval must be between 10 and 600 seconds."""
        # Valid boundaries
        PKGDatabaseConfig(health_check_interval_seconds=10)
        PKGDatabaseConfig(health_check_interval_seconds=600)

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(health_check_interval_seconds=9)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(health_check_interval_seconds=601)


class TestPKGDatabaseConfigEnvironmentOverrides:
    """Test environment variable override support."""

    def test_env_override_max_pool_size(self):
        """FUTURNAL_PKG_MAX_POOL_SIZE overrides max_connection_pool_size."""
        with mock.patch.dict(os.environ, {"FUTURNAL_PKG_MAX_POOL_SIZE": "25"}):
            config = PKGDatabaseConfig.with_env_overrides()
            assert config.max_connection_pool_size == 25

    def test_env_override_connection_timeout(self):
        """FUTURNAL_PKG_CONNECTION_TIMEOUT overrides connection_timeout."""
        with mock.patch.dict(os.environ, {"FUTURNAL_PKG_CONNECTION_TIMEOUT": "45.0"}):
            config = PKGDatabaseConfig.with_env_overrides()
            assert config.connection_timeout == 45.0

    def test_env_override_backup_path(self):
        """FUTURNAL_PKG_BACKUP_PATH overrides backup_path."""
        with mock.patch.dict(os.environ, {"FUTURNAL_PKG_BACKUP_PATH": "/custom/backups"}):
            config = PKGDatabaseConfig.with_env_overrides()
            assert config.backup_path == Path("/custom/backups")

    def test_env_override_backup_retention_days(self):
        """FUTURNAL_PKG_BACKUP_RETENTION_DAYS overrides backup_retention_days."""
        with mock.patch.dict(os.environ, {"FUTURNAL_PKG_BACKUP_RETENTION_DAYS": "60"}):
            config = PKGDatabaseConfig.with_env_overrides()
            assert config.backup_retention_days == 60

    def test_env_override_fetch_size(self):
        """FUTURNAL_PKG_FETCH_SIZE overrides fetch_size."""
        with mock.patch.dict(os.environ, {"FUTURNAL_PKG_FETCH_SIZE": "500"}):
            config = PKGDatabaseConfig.with_env_overrides()
            assert config.fetch_size == 500

    def test_env_override_health_check_interval(self):
        """FUTURNAL_PKG_HEALTH_CHECK_INTERVAL overrides health_check_interval_seconds."""
        with mock.patch.dict(os.environ, {"FUTURNAL_PKG_HEALTH_CHECK_INTERVAL": "120"}):
            config = PKGDatabaseConfig.with_env_overrides()
            assert config.health_check_interval_seconds == 120

    def test_env_override_with_base_config(self):
        """Environment overrides apply on top of base config."""
        base = PKGDatabaseConfig(max_connection_pool_size=15, fetch_size=2000)

        with mock.patch.dict(os.environ, {"FUTURNAL_PKG_MAX_POOL_SIZE": "20"}):
            config = PKGDatabaseConfig.with_env_overrides(base)
            # Overridden value
            assert config.max_connection_pool_size == 20
            # Base value preserved
            assert config.fetch_size == 2000

    def test_invalid_env_value_ignored(self):
        """Invalid environment values are silently ignored."""
        with mock.patch.dict(os.environ, {"FUTURNAL_PKG_MAX_POOL_SIZE": "not_a_number"}):
            # Should not raise, uses default instead
            config = PKGDatabaseConfig.with_env_overrides()
            assert config.max_connection_pool_size == 10  # Default

    def test_multiple_env_overrides(self):
        """Multiple environment overrides work together."""
        env = {
            "FUTURNAL_PKG_MAX_POOL_SIZE": "30",
            "FUTURNAL_PKG_CONNECTION_TIMEOUT": "60.0",
            "FUTURNAL_PKG_FETCH_SIZE": "5000",
        }
        with mock.patch.dict(os.environ, env):
            config = PKGDatabaseConfig.with_env_overrides()
            assert config.max_connection_pool_size == 30
            assert config.connection_timeout == 60.0
            assert config.fetch_size == 5000


class TestPKGDatabaseConfigValidation:
    """Test configuration validation."""

    def test_extra_fields_rejected(self):
        """Extra fields should be rejected."""
        with pytest.raises(ValidationError):
            PKGDatabaseConfig(unknown_field="value")

    def test_backup_path_string_converted(self):
        """String backup_path should be converted to Path."""
        config = PKGDatabaseConfig(backup_path="/some/path")
        assert isinstance(config.backup_path, Path)
        assert config.backup_path == Path("/some/path")

    def test_validate_assignment(self):
        """Assignment validation should work."""
        config = PKGDatabaseConfig()

        # Valid assignment
        config.max_connection_pool_size = 25
        assert config.max_connection_pool_size == 25

        # Invalid assignment should raise
        with pytest.raises(ValidationError):
            config.max_connection_pool_size = 0


class TestPKGDatabaseConfigMethods:
    """Test configuration methods."""

    def test_get_backup_path_with_explicit_path(self):
        """get_backup_path returns explicit path when set."""
        config = PKGDatabaseConfig(backup_path="/explicit/path")
        workspace = Path("/workspace")

        result = config.get_backup_path(workspace)
        assert result == Path("/explicit/path")

    def test_get_backup_path_default(self):
        """get_backup_path returns workspace/pkg/backups by default."""
        config = PKGDatabaseConfig()
        workspace = Path("/workspace")

        result = config.get_backup_path(workspace)
        assert result == Path("/workspace/pkg/backups")

    def test_to_driver_config(self):
        """to_driver_config returns Neo4j driver options."""
        config = PKGDatabaseConfig(
            max_connection_pool_size=20,
            connection_acquisition_timeout=90.0,
            max_transaction_retry_time=45.0,
            connection_timeout=60.0,
        )

        driver_config = config.to_driver_config()

        assert driver_config == {
            "max_connection_pool_size": 20,
            "connection_acquisition_timeout": 90.0,
            "max_transaction_retry_time": 45.0,
            "connection_timeout": 60.0,
        }
