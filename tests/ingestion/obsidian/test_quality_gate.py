"""Comprehensive tests for Obsidian quality gate system."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from futurnal.ingestion.obsidian.quality_gate import (
    QualityGateConfig,
    QualityGateEvaluator,
    QualityGateResult,
    QualityGateStatus,
    QualityMetricType,
    QualityThreshold,
    QualityMetricResult,
    create_quality_gate_evaluator
)
from futurnal.ingestion.obsidian.sync_metrics import (
    SyncMetricsCollector,
    SyncMetricsSummary,
    create_metrics_collector
)


class TestQualityGateConfig:
    """Test quality gate configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QualityGateConfig()

        assert config.max_error_rate == 0.05
        assert config.max_critical_error_rate == 0.10
        assert config.max_parse_failure_rate == 0.02
        assert config.max_broken_link_rate == 0.03
        assert config.min_throughput_events_per_second == 1.0
        assert config.max_avg_processing_time_seconds == 5.0
        assert config.min_consent_coverage_rate == 0.95
        assert config.min_asset_processing_success_rate == 0.90
        assert config.max_quarantine_rate == 0.02
        assert config.enable_strict_mode is False
        assert config.evaluation_time_window_hours == 1
        assert config.require_minimum_sample_size == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QualityGateConfig(
            max_error_rate=0.10,
            enable_strict_mode=True,
            evaluation_time_window_hours=24,
            require_minimum_sample_size=100
        )

        assert config.max_error_rate == 0.10
        assert config.enable_strict_mode is True
        assert config.evaluation_time_window_hours == 24
        assert config.require_minimum_sample_size == 100

    def test_get_thresholds(self):
        """Test threshold generation from config."""
        config = QualityGateConfig()
        thresholds = config.get_thresholds()

        assert len(thresholds) == 8  # Should have 8 different threshold types

        # Check that all metric types are covered
        metric_types = {threshold.metric_type for threshold in thresholds}
        expected_types = {
            QualityMetricType.ERROR_RATE,
            QualityMetricType.PARSE_FAILURE_RATE,
            QualityMetricType.BROKEN_LINK_RATE,
            QualityMetricType.THROUGHPUT_MINIMUM,
            QualityMetricType.PROCESSING_TIME_MAXIMUM,
            QualityMetricType.CONSENT_COVERAGE,
            QualityMetricType.ASSET_PROCESSING_SUCCESS_RATE,
            QualityMetricType.QUARANTINE_RATE
        }
        assert metric_types == expected_types


class TestQualityThreshold:
    """Test quality threshold evaluation."""

    def test_threshold_evaluate_lower_is_better(self):
        """Test threshold evaluation for metrics where lower is better."""
        threshold = QualityThreshold(
            metric_type=QualityMetricType.ERROR_RATE,
            warn_threshold=0.05,
            fail_threshold=0.10,
            description="Error rate",
            unit="%",
            higher_is_better=False
        )

        # Test passing values
        assert threshold.evaluate(0.01) == QualityGateStatus.PASS
        assert threshold.evaluate(0.04) == QualityGateStatus.PASS

        # Test warning values
        assert threshold.evaluate(0.06) == QualityGateStatus.WARN
        assert threshold.evaluate(0.08) == QualityGateStatus.WARN

        # Test failing values
        assert threshold.evaluate(0.12) == QualityGateStatus.FAIL
        assert threshold.evaluate(0.20) == QualityGateStatus.FAIL

    def test_threshold_evaluate_higher_is_better(self):
        """Test threshold evaluation for metrics where higher is better."""
        threshold = QualityThreshold(
            metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
            warn_threshold=1.0,
            fail_threshold=0.5,
            description="Throughput",
            unit="events/sec",
            higher_is_better=True
        )

        # Test passing values
        assert threshold.evaluate(2.0) == QualityGateStatus.PASS
        assert threshold.evaluate(1.5) == QualityGateStatus.PASS

        # Test warning values
        assert threshold.evaluate(0.8) == QualityGateStatus.WARN
        assert threshold.evaluate(0.6) == QualityGateStatus.WARN

        # Test failing values
        assert threshold.evaluate(0.3) == QualityGateStatus.FAIL
        assert threshold.evaluate(0.1) == QualityGateStatus.FAIL


class TestQualityMetricResult:
    """Test quality metric result formatting."""

    def test_formatted_value_percentage(self):
        """Test percentage formatting."""
        threshold = QualityThreshold(
            metric_type=QualityMetricType.ERROR_RATE,
            warn_threshold=0.05,
            fail_threshold=0.10,
            description="Error rate",
            unit="%"
        )

        result = QualityMetricResult(
            metric_type=QualityMetricType.ERROR_RATE,
            status=QualityGateStatus.WARN,
            value=0.067,
            threshold=threshold,
            message="Test message"
        )

        assert result.formatted_value == "6.7%"

    def test_formatted_value_with_unit(self):
        """Test formatting with custom unit."""
        threshold = QualityThreshold(
            metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
            warn_threshold=1.0,
            fail_threshold=0.5,
            description="Throughput",
            unit="events/sec"
        )

        result = QualityMetricResult(
            metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
            status=QualityGateStatus.PASS,
            value=2.45,
            threshold=threshold,
            message="Test message"
        )

        assert result.formatted_value == "2.45 events/sec"

    def test_formatted_value_no_unit(self):
        """Test formatting without unit."""
        threshold = QualityThreshold(
            metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
            warn_threshold=1.0,
            fail_threshold=0.5,
            description="Throughput",
            unit=""
        )

        result = QualityMetricResult(
            metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
            status=QualityGateStatus.PASS,
            value=2.45,
            threshold=threshold,
            message="Test message"
        )

        assert result.formatted_value == "2.45"


class TestQualityGateResult:
    """Test quality gate result properties."""

    def create_sample_result(self, status=QualityGateStatus.PASS, strict_mode=False):
        """Create a sample quality gate result for testing."""
        config = QualityGateConfig(enable_strict_mode=strict_mode)
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow()
        )

        return QualityGateResult(
            vault_id="test_vault",
            status=status,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[],
            summary_metrics=summary
        )

    def test_passed_property(self):
        """Test passed property calculation."""
        result_pass = self.create_sample_result(QualityGateStatus.PASS)
        result_warn = self.create_sample_result(QualityGateStatus.WARN)
        result_fail = self.create_sample_result(QualityGateStatus.FAIL)

        assert result_pass.passed is True
        assert result_warn.passed is True
        assert result_fail.passed is False

    def test_has_warnings(self):
        """Test warning detection."""
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow()
        )

        warning_result = QualityMetricResult(
            metric_type=QualityMetricType.ERROR_RATE,
            status=QualityGateStatus.WARN,
            value=0.06,
            threshold=QualityThreshold(
                metric_type=QualityMetricType.ERROR_RATE,
                warn_threshold=0.05,
                fail_threshold=0.10,
                description="Error rate"
            ),
            message="Warning"
        )

        result = QualityGateResult(
            vault_id="test_vault",
            status=QualityGateStatus.WARN,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[warning_result],
            summary_metrics=summary
        )

        assert result.has_warnings is True

    def test_has_failures(self):
        """Test failure detection."""
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow()
        )

        failure_result = QualityMetricResult(
            metric_type=QualityMetricType.ERROR_RATE,
            status=QualityGateStatus.FAIL,
            value=0.15,
            threshold=QualityThreshold(
                metric_type=QualityMetricType.ERROR_RATE,
                warn_threshold=0.05,
                fail_threshold=0.10,
                description="Error rate"
            ),
            message="Failure"
        )

        result = QualityGateResult(
            vault_id="test_vault",
            status=QualityGateStatus.FAIL,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[failure_result],
            summary_metrics=summary
        )

        assert result.has_failures is True

    def test_get_exit_code(self):
        """Test exit code calculation."""
        # Test passing result
        result_pass = self.create_sample_result(QualityGateStatus.PASS)
        assert result_pass.get_exit_code() == 0

        # Test warning result (normal mode)
        result_warn = self.create_sample_result(QualityGateStatus.WARN, strict_mode=False)
        assert result_warn.get_exit_code() == 1

        # Test warning result (strict mode)
        result_warn_strict = self.create_sample_result(QualityGateStatus.WARN, strict_mode=True)
        assert result_warn_strict.get_exit_code() == 2

        # Test failure result
        result_fail = self.create_sample_result(QualityGateStatus.FAIL)
        assert result_fail.get_exit_code() == 2


class TestQualityGateEvaluator:
    """Test quality gate evaluator."""

    def create_mock_metrics_collector(self, vault_id="test_vault"):
        """Create a mock metrics collector with sample data."""
        collector = Mock(spec=SyncMetricsCollector)

        # Mock summary data
        summary = SyncMetricsSummary(
            vault_id=vault_id,
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=100,
            failed_events=5,
            events_per_second=2.5,
            average_event_processing_time=3.0
        )

        collector.generate_summary.return_value = summary
        collector.get_counter.return_value = 2  # Mock quarantine count

        return collector

    def test_evaluator_creation(self):
        """Test evaluator creation with different configurations."""
        config = QualityGateConfig()
        collector = self.create_mock_metrics_collector()

        evaluator = QualityGateEvaluator(config=config, metrics_collector=collector)
        assert evaluator.config == config
        assert evaluator.metrics_collector == collector

    def test_evaluate_vault_quality_success(self):
        """Test successful quality evaluation."""
        config = QualityGateConfig()
        collector = self.create_mock_metrics_collector()

        evaluator = QualityGateEvaluator(config=config, metrics_collector=collector)
        result = evaluator.evaluate_vault_quality("test_vault", collector)

        assert result.vault_id == "test_vault"
        assert result.status in [QualityGateStatus.PASS, QualityGateStatus.WARN, QualityGateStatus.FAIL]
        assert len(result.metric_results) > 0
        assert result.summary_metrics is not None

    def test_evaluate_vault_quality_insufficient_sample_size(self):
        """Test evaluation with insufficient sample size."""
        config = QualityGateConfig(require_minimum_sample_size=200)
        collector = self.create_mock_metrics_collector()

        evaluator = QualityGateEvaluator(config=config, metrics_collector=collector)

        # Should not raise an exception but may log warnings
        result = evaluator.evaluate_vault_quality("test_vault", collector)
        assert result is not None

    def test_evaluate_vault_quality_no_collector(self):
        """Test evaluation without metrics collector."""
        config = QualityGateConfig()
        evaluator = QualityGateEvaluator(config=config)

        with pytest.raises(ValueError, match="Metrics collector is required"):
            evaluator.evaluate_vault_quality("test_vault")

    def test_determine_overall_status_pass(self):
        """Test overall status determination for passing metrics."""
        config = QualityGateConfig()
        evaluator = QualityGateEvaluator(config=config)

        metric_results = [
            QualityMetricResult(
                metric_type=QualityMetricType.ERROR_RATE,
                status=QualityGateStatus.PASS,
                value=0.02,
                threshold=Mock(),
                message="Pass"
            ),
            QualityMetricResult(
                metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
                status=QualityGateStatus.PASS,
                value=2.0,
                threshold=Mock(),
                message="Pass"
            )
        ]

        status = evaluator._determine_overall_status(metric_results)
        assert status == QualityGateStatus.PASS

    def test_determine_overall_status_warn(self):
        """Test overall status determination for warning metrics."""
        config = QualityGateConfig(enable_strict_mode=False)
        evaluator = QualityGateEvaluator(config=config)

        metric_results = [
            QualityMetricResult(
                metric_type=QualityMetricType.ERROR_RATE,
                status=QualityGateStatus.PASS,
                value=0.02,
                threshold=Mock(),
                message="Pass"
            ),
            QualityMetricResult(
                metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
                status=QualityGateStatus.WARN,
                value=0.8,
                threshold=Mock(),
                message="Warning"
            )
        ]

        status = evaluator._determine_overall_status(metric_results)
        assert status == QualityGateStatus.WARN

    def test_determine_overall_status_warn_strict_mode(self):
        """Test overall status determination for warning metrics in strict mode."""
        config = QualityGateConfig(enable_strict_mode=True)
        evaluator = QualityGateEvaluator(config=config)

        metric_results = [
            QualityMetricResult(
                metric_type=QualityMetricType.ERROR_RATE,
                status=QualityGateStatus.PASS,
                value=0.02,
                threshold=Mock(),
                message="Pass"
            ),
            QualityMetricResult(
                metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
                status=QualityGateStatus.WARN,
                value=0.8,
                threshold=Mock(),
                message="Warning"
            )
        ]

        status = evaluator._determine_overall_status(metric_results)
        assert status == QualityGateStatus.FAIL

    def test_determine_overall_status_fail(self):
        """Test overall status determination for failing metrics."""
        config = QualityGateConfig()
        evaluator = QualityGateEvaluator(config=config)

        metric_results = [
            QualityMetricResult(
                metric_type=QualityMetricType.ERROR_RATE,
                status=QualityGateStatus.FAIL,
                value=0.15,
                threshold=Mock(),
                message="Failure"
            )
        ]

        status = evaluator._determine_overall_status(metric_results)
        assert status == QualityGateStatus.FAIL

    def test_generate_insights(self):
        """Test insight generation."""
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=100,
            failed_events=10
        )

        result = QualityGateResult(
            vault_id="test_vault",
            status=QualityGateStatus.WARN,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[],
            summary_metrics=summary,
            total_files_failed=10,
            throughput_events_per_second=0.5
        )

        evaluator = QualityGateEvaluator(config=config)
        evaluator._generate_insights(result)

        # Should generate recommendations based on the metrics
        assert len(result.recommendations) > 0


class TestQualityGateFactory:
    """Test quality gate factory functions."""

    def test_create_quality_gate_evaluator_default(self):
        """Test factory function with default parameters."""
        evaluator = create_quality_gate_evaluator()

        assert isinstance(evaluator, QualityGateEvaluator)
        assert isinstance(evaluator.config, QualityGateConfig)
        assert evaluator.metrics_collector is None

    def test_create_quality_gate_evaluator_with_config(self):
        """Test factory function with custom config."""
        config = QualityGateConfig(enable_strict_mode=True)
        evaluator = create_quality_gate_evaluator(config=config)

        assert evaluator.config == config
        assert evaluator.config.enable_strict_mode is True

    def test_create_quality_gate_evaluator_with_collector(self):
        """Test factory function with metrics collector."""
        collector = self.create_mock_metrics_collector()
        evaluator = create_quality_gate_evaluator(metrics_collector=collector)

        assert evaluator.metrics_collector == collector


class TestQualityGateIntegration:
    """Integration tests for quality gate system."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Create real metrics collector
        collector = create_metrics_collector()

        # Add some sample metrics
        collector.increment_counter("sync_events_total", 100)
        collector.increment_counter("sync_errors_total", 5)
        collector.set_gauge("pending_events", 0)
        collector.record_timer("batch_processing_time", 2.5)

        # Create evaluator and run evaluation
        config = QualityGateConfig()
        evaluator = QualityGateEvaluator(config=config, metrics_collector=collector)

        # This should work even with minimal metrics
        result = evaluator.evaluate_vault_quality("test_vault", collector)

        assert result is not None
        assert result.vault_id == "test_vault"
        assert isinstance(result.status, QualityGateStatus)

    @patch('futurnal.ingestion.obsidian.quality_gate.logger')
    def test_error_handling_in_evaluation(self, mock_logger):
        """Test error handling during metric evaluation."""
        # Create a collector that throws errors
        collector = Mock(spec=SyncMetricsCollector)
        collector.generate_summary.side_effect = Exception("Test error")

        config = QualityGateConfig()
        evaluator = QualityGateEvaluator(config=config, metrics_collector=collector)

        with pytest.raises(Exception):
            evaluator.evaluate_vault_quality("test_vault", collector)

    def test_evaluation_with_different_vault_ids(self):
        """Test evaluation with different vault IDs."""
        collector = self.create_mock_metrics_collector()
        config = QualityGateConfig()
        evaluator = QualityGateEvaluator(config=config, metrics_collector=collector)

        # Test with different vault IDs
        vault_ids = ["vault1", "vault2", "test-vault-123"]

        for vault_id in vault_ids:
            result = evaluator.evaluate_vault_quality(vault_id, collector)
            assert result.vault_id == vault_id

    def create_mock_metrics_collector(self, vault_id="test_vault"):
        """Helper method to create mock metrics collector."""
        collector = Mock(spec=SyncMetricsCollector)

        summary = SyncMetricsSummary(
            vault_id=vault_id,
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=100,
            failed_events=5,
            events_per_second=2.5,
            average_event_processing_time=3.0
        )

        collector.generate_summary.return_value = summary
        collector.get_counter.return_value = 2

        return collector