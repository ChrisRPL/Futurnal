"""Integration tests for quality gate system with vault descriptors."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from futurnal.ingestion.obsidian.descriptor import (
    ObsidianVaultDescriptor,
    VaultQualityGateSettings,
    VaultRegistry
)
from futurnal.ingestion.obsidian.quality_gate import (
    QualityGateConfig,
    QualityGateEvaluator,
    QualityGateStatus
)
from futurnal.ingestion.obsidian.sync_metrics import SyncMetricsSummary


class TestVaultQualityGateSettings:
    """Test vault quality gate settings configuration."""

    def test_default_settings(self):
        """Test default quality gate settings."""
        settings = VaultQualityGateSettings()

        assert settings.enable_quality_gates is True
        assert settings.strict_mode is False
        assert settings.max_error_rate == 0.05
        assert settings.max_critical_error_rate == 0.10
        assert settings.max_parse_failure_rate == 0.02
        assert settings.max_broken_link_rate == 0.03
        assert settings.min_throughput_events_per_second == 1.0
        assert settings.max_avg_processing_time_seconds == 5.0
        assert settings.min_consent_coverage_rate == 0.95
        assert settings.min_asset_processing_success_rate == 0.90
        assert settings.max_quarantine_rate == 0.02
        assert settings.evaluation_time_window_hours == 1
        assert settings.require_minimum_sample_size == 10

    def test_custom_settings(self):
        """Test custom quality gate settings."""
        settings = VaultQualityGateSettings(
            enable_quality_gates=False,
            strict_mode=True,
            max_error_rate=0.08,
            min_throughput_events_per_second=2.0,
            evaluation_time_window_hours=24,
            require_minimum_sample_size=50
        )

        assert settings.enable_quality_gates is False
        assert settings.strict_mode is True
        assert settings.max_error_rate == 0.08
        assert settings.min_throughput_events_per_second == 2.0
        assert settings.evaluation_time_window_hours == 24
        assert settings.require_minimum_sample_size == 50

    def test_settings_validation(self):
        """Test settings validation constraints."""
        # Test valid settings
        settings = VaultQualityGateSettings(
            max_error_rate=0.5,
            min_consent_coverage_rate=0.8,
            evaluation_time_window_hours=48
        )
        assert settings.max_error_rate == 0.5
        assert settings.min_consent_coverage_rate == 0.8
        assert settings.evaluation_time_window_hours == 48

        # Test invalid settings (should raise validation errors)
        with pytest.raises(ValueError):
            VaultQualityGateSettings(max_error_rate=1.5)  # > 1.0

        with pytest.raises(ValueError):
            VaultQualityGateSettings(max_error_rate=-0.1)  # < 0.0

        with pytest.raises(ValueError):
            VaultQualityGateSettings(evaluation_time_window_hours=0)  # < 1

        with pytest.raises(ValueError):
            VaultQualityGateSettings(evaluation_time_window_hours=200)  # > 168


class TestObsidianVaultDescriptorQualityGate:
    """Test quality gate integration with vault descriptors."""

    def create_test_vault_path(self):
        """Create a temporary test vault directory."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir)
        obsidian_dir = vault_path / ".obsidian"
        obsidian_dir.mkdir()
        return vault_path

    def test_descriptor_default_quality_gate_settings(self):
        """Test descriptor creation with default quality gate settings."""
        vault_path = self.create_test_vault_path()

        try:
            descriptor = ObsidianVaultDescriptor.from_path(vault_path, name="Test Vault")

            assert descriptor.quality_gate_settings is not None
            assert isinstance(descriptor.quality_gate_settings, VaultQualityGateSettings)
            assert descriptor.quality_gate_settings.enable_quality_gates is True
            assert descriptor.quality_gate_settings.max_error_rate == 0.05

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(vault_path)

    def test_descriptor_custom_quality_gate_settings(self):
        """Test descriptor creation with custom quality gate settings."""
        vault_path = self.create_test_vault_path()

        try:
            custom_settings = VaultQualityGateSettings(
                enable_quality_gates=False,
                strict_mode=True,
                max_error_rate=0.08,
                min_throughput_events_per_second=2.5
            )

            descriptor = ObsidianVaultDescriptor.from_path(
                vault_path,
                name="Test Vault",
                quality_gate_settings=custom_settings
            )

            assert descriptor.quality_gate_settings == custom_settings
            assert descriptor.quality_gate_settings.enable_quality_gates is False
            assert descriptor.quality_gate_settings.strict_mode is True
            assert descriptor.quality_gate_settings.max_error_rate == 0.08
            assert descriptor.quality_gate_settings.min_throughput_events_per_second == 2.5

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(vault_path)

    def test_vault_registry_quality_gate_settings(self):
        """Test vault registry with quality gate settings."""
        vault_path = self.create_test_vault_path()

        try:
            with tempfile.TemporaryDirectory() as registry_dir:
                registry = VaultRegistry(registry_root=Path(registry_dir))

                custom_settings = VaultQualityGateSettings(
                    max_error_rate=0.12,
                    strict_mode=True
                )

                descriptor = registry.register_path(
                    vault_path,
                    name="Registry Test Vault",
                    quality_gate_settings=custom_settings,
                    operator="test_operator"
                )

                assert descriptor.quality_gate_settings.max_error_rate == 0.12
                assert descriptor.quality_gate_settings.strict_mode is True

                # Test retrieval
                retrieved = registry.get(descriptor.id)
                assert retrieved is not None
                assert retrieved.quality_gate_settings.max_error_rate == 0.12
                assert retrieved.quality_gate_settings.strict_mode is True

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(vault_path)


class TestQualityGateConfigFromVaultSettings:
    """Test converting vault settings to quality gate config."""

    def test_config_from_vault_settings(self):
        """Test creating quality gate config from vault settings."""
        vault_settings = VaultQualityGateSettings(
            strict_mode=True,
            max_error_rate=0.08,
            max_critical_error_rate=0.15,
            max_parse_failure_rate=0.03,
            min_throughput_events_per_second=2.0,
            max_avg_processing_time_seconds=8.0,
            min_consent_coverage_rate=0.90,
            min_asset_processing_success_rate=0.85,
            max_quarantine_rate=0.04,
            evaluation_time_window_hours=6,
            require_minimum_sample_size=25
        )

        config = QualityGateConfig(
            enable_strict_mode=vault_settings.strict_mode,
            max_error_rate=vault_settings.max_error_rate,
            max_critical_error_rate=vault_settings.max_critical_error_rate,
            max_parse_failure_rate=vault_settings.max_parse_failure_rate,
            min_throughput_events_per_second=vault_settings.min_throughput_events_per_second,
            max_avg_processing_time_seconds=vault_settings.max_avg_processing_time_seconds,
            min_consent_coverage_rate=vault_settings.min_consent_coverage_rate,
            min_asset_processing_success_rate=vault_settings.min_asset_processing_success_rate,
            max_quarantine_rate=vault_settings.max_quarantine_rate,
            evaluation_time_window_hours=vault_settings.evaluation_time_window_hours,
            require_minimum_sample_size=vault_settings.require_minimum_sample_size
        )

        assert config.enable_strict_mode is True
        assert config.max_error_rate == 0.08
        assert config.max_critical_error_rate == 0.15
        assert config.max_parse_failure_rate == 0.03
        assert config.min_throughput_events_per_second == 2.0
        assert config.max_avg_processing_time_seconds == 8.0
        assert config.min_consent_coverage_rate == 0.90
        assert config.min_asset_processing_success_rate == 0.85
        assert config.max_quarantine_rate == 0.04
        assert config.evaluation_time_window_hours == 6
        assert config.require_minimum_sample_size == 25

    def test_config_inheritance_with_overrides(self):
        """Test config inheritance with CLI overrides."""
        vault_settings = VaultQualityGateSettings(
            max_error_rate=0.05,
            min_throughput_events_per_second=1.0
        )

        # Simulate CLI overrides
        cli_max_error_rate = 0.10
        cli_min_throughput = 2.0

        config = QualityGateConfig(
            max_error_rate=cli_max_error_rate or vault_settings.max_error_rate,
            min_throughput_events_per_second=cli_min_throughput or vault_settings.min_throughput_events_per_second,
            # Other settings from vault
            max_critical_error_rate=vault_settings.max_critical_error_rate,
            max_parse_failure_rate=vault_settings.max_parse_failure_rate,
        )

        # CLI overrides should take precedence
        assert config.max_error_rate == 0.10
        assert config.min_throughput_events_per_second == 2.0

        # Vault settings should be used for non-overridden values
        assert config.max_critical_error_rate == vault_settings.max_critical_error_rate
        assert config.max_parse_failure_rate == vault_settings.max_parse_failure_rate


class TestQualityGateVaultIntegration:
    """Test end-to-end quality gate integration with vaults."""

    def create_mock_metrics_collector(self, vault_id="test_vault"):
        """Create a mock metrics collector."""
        from unittest.mock import Mock
        from datetime import datetime, timedelta

        collector = Mock()

        summary = SyncMetricsSummary(
            vault_id=vault_id,
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=100,
            failed_events=3,
            events_per_second=2.5,
            average_event_processing_time=2.0
        )

        collector.generate_summary.return_value = summary
        collector.get_counter.return_value = 1

        return collector

    def test_quality_gate_evaluation_with_vault_settings(self):
        """Test quality gate evaluation using vault-specific settings."""
        # Create vault with custom settings
        vault_settings = VaultQualityGateSettings(
            enable_quality_gates=True,
            max_error_rate=0.02,  # Stricter than default
            strict_mode=True
        )

        # Create quality gate config from vault settings
        config = QualityGateConfig(
            enable_strict_mode=vault_settings.strict_mode,
            max_error_rate=vault_settings.max_error_rate,
            max_critical_error_rate=vault_settings.max_critical_error_rate
        )

        # Create evaluator
        collector = self.create_mock_metrics_collector()
        evaluator = QualityGateEvaluator(config=config, metrics_collector=collector)

        # Evaluate quality gate
        result = evaluator.evaluate_vault_quality("test_vault", collector)

        assert result is not None
        assert result.vault_id == "test_vault"
        assert result.config.max_error_rate == 0.02
        assert result.config.enable_strict_mode is True

    def test_quality_gate_disabled_in_vault(self):
        """Test behavior when quality gates are disabled in vault settings."""
        vault_settings = VaultQualityGateSettings(
            enable_quality_gates=False
        )

        # This would typically be handled at the CLI level
        assert vault_settings.enable_quality_gates is False

    def test_different_vault_configurations(self):
        """Test multiple vaults with different quality gate configurations."""
        # Vault 1: Strict settings
        vault1_settings = VaultQualityGateSettings(
            max_error_rate=0.01,
            strict_mode=True,
            min_throughput_events_per_second=5.0
        )

        # Vault 2: Permissive settings
        vault2_settings = VaultQualityGateSettings(
            max_error_rate=0.10,
            strict_mode=False,
            min_throughput_events_per_second=0.5
        )

        # Vault 3: Disabled quality gates
        vault3_settings = VaultQualityGateSettings(
            enable_quality_gates=False
        )

        # Test that each vault has different configurations
        assert vault1_settings.max_error_rate == 0.01
        assert vault1_settings.strict_mode is True

        assert vault2_settings.max_error_rate == 0.10
        assert vault2_settings.strict_mode is False

        assert vault3_settings.enable_quality_gates is False

    def test_vault_settings_persistence(self):
        """Test that vault quality gate settings persist across registry operations."""
        vault_path = None

        try:
            # Create test vault
            vault_path = Path(tempfile.mkdtemp())
            obsidian_dir = vault_path / ".obsidian"
            obsidian_dir.mkdir()

            with tempfile.TemporaryDirectory() as registry_dir:
                registry = VaultRegistry(registry_root=Path(registry_dir))

                # Register vault with custom settings
                custom_settings = VaultQualityGateSettings(
                    max_error_rate=0.07,
                    strict_mode=True,
                    evaluation_time_window_hours=12
                )

                descriptor1 = registry.register_path(
                    vault_path,
                    name="Persistence Test Vault",
                    quality_gate_settings=custom_settings
                )

                vault_id = descriptor1.id

                # Retrieve vault and verify settings
                descriptor2 = registry.get(vault_id)
                assert descriptor2 is not None
                assert descriptor2.quality_gate_settings.max_error_rate == 0.07
                assert descriptor2.quality_gate_settings.strict_mode is True
                assert descriptor2.quality_gate_settings.evaluation_time_window_hours == 12

                # Update vault with new settings
                updated_settings = VaultQualityGateSettings(
                    max_error_rate=0.09,
                    strict_mode=False,
                    min_throughput_events_per_second=3.0
                )

                descriptor2.quality_gate_settings = updated_settings
                registry.add_or_update(descriptor2)

                # Retrieve again and verify updates
                descriptor3 = registry.get(vault_id)
                assert descriptor3 is not None
                assert descriptor3.quality_gate_settings.max_error_rate == 0.09
                assert descriptor3.quality_gate_settings.strict_mode is False
                assert descriptor3.quality_gate_settings.min_throughput_events_per_second == 3.0

        finally:
            # Cleanup
            if vault_path:
                import shutil
                shutil.rmtree(vault_path)


class TestQualityGateValidationEdgeCases:
    """Test edge cases and validation scenarios."""

    def test_extreme_threshold_values(self):
        """Test extreme but valid threshold values."""
        # Test very permissive settings
        permissive_settings = VaultQualityGateSettings(
            max_error_rate=1.0,  # 100% error rate allowed
            max_critical_error_rate=1.0,
            min_throughput_events_per_second=0.01,  # Very low throughput
            max_avg_processing_time_seconds=3600.0,  # 1 hour per file
            min_consent_coverage_rate=0.0,  # No consent required
            min_asset_processing_success_rate=0.0,  # No asset success required
            evaluation_time_window_hours=168,  # 1 week
            require_minimum_sample_size=1  # Minimum sample size
        )

        # Should not raise validation errors
        assert permissive_settings.max_error_rate == 1.0
        assert permissive_settings.min_throughput_events_per_second == 0.01
        assert permissive_settings.evaluation_time_window_hours == 168

        # Test very strict settings
        strict_settings = VaultQualityGateSettings(
            max_error_rate=0.0,  # 0% error rate
            max_critical_error_rate=0.0,
            min_throughput_events_per_second=100.0,  # Very high throughput
            max_avg_processing_time_seconds=0.1,  # Very fast processing
            min_consent_coverage_rate=1.0,  # 100% consent required
            min_asset_processing_success_rate=1.0,  # 100% asset success required
            evaluation_time_window_hours=1,  # Minimum time window
            require_minimum_sample_size=10000  # Large sample size
        )

        assert strict_settings.max_error_rate == 0.0
        assert strict_settings.min_throughput_events_per_second == 100.0
        assert strict_settings.require_minimum_sample_size == 10000

    def test_quality_gate_with_no_metrics(self):
        """Test quality gate evaluation with no metrics available."""
        from unittest.mock import Mock

        # Create a collector that returns empty metrics
        collector = Mock()
        empty_summary = SyncMetricsSummary(
            vault_id="empty_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=0,
            failed_events=0,
            events_per_second=0.0,
            average_event_processing_time=0.0
        )

        collector.generate_summary.return_value = empty_summary
        collector.get_counter.return_value = 0

        config = QualityGateConfig(require_minimum_sample_size=10)
        evaluator = QualityGateEvaluator(config=config, metrics_collector=collector)

        # Should handle empty metrics gracefully
        result = evaluator.evaluate_vault_quality("empty_vault", collector)
        assert result is not None
        assert result.vault_id == "empty_vault"

    def test_quality_gate_config_validation_edge_cases(self):
        """Test quality gate config validation edge cases."""
        from datetime import datetime, timedelta

        # Test with valid boundary values
        config = QualityGateConfig(
            max_error_rate=0.0,
            max_critical_error_rate=1.0,
            evaluation_time_window_hours=1,
            require_minimum_sample_size=1
        )

        assert config.max_error_rate == 0.0
        assert config.max_critical_error_rate == 1.0
        assert config.evaluation_time_window_hours == 1
        assert config.require_minimum_sample_size == 1