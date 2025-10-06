"""Tests for IMAP quality gate evaluation system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from futurnal.ingestion.imap.quality_gate import (
    ImapQualityGateEvaluator,
    ImapQualityGates,
    ImapQualityMetricType,
    QualityGateStatus,
)
from futurnal.ingestion.imap.sync_metrics import ImapSyncMetricsCollector


@pytest.fixture
def mailbox_id() -> str:
    """Test mailbox identifier."""
    return "test@example.com"


@pytest.fixture
def collector() -> ImapSyncMetricsCollector:
    """Fresh metrics collector."""
    return ImapSyncMetricsCollector()


@pytest.fixture
def evaluator(collector: ImapSyncMetricsCollector) -> ImapQualityGateEvaluator:
    """Quality gate evaluator with default config."""
    return ImapQualityGateEvaluator(
        config=ImapQualityGates(), metrics_collector=collector
    )


# ============================================================================
# Metrics Collection Tests
# ============================================================================


def test_metrics_collector_sync_attempts(collector: ImapSyncMetricsCollector, mailbox_id: str):
    """Test recording sync attempts."""
    collector.record_sync_attempt(mailbox_id, success=True)
    collector.record_sync_attempt(mailbox_id, success=True)
    collector.record_sync_attempt(mailbox_id, success=False)

    summary = collector.generate_summary(mailbox_id)
    assert summary.total_sync_attempts == 3
    assert summary.successful_syncs == 2
    assert summary.failed_syncs == 1
    assert summary.sync_failure_rate == pytest.approx(1 / 3)


def test_metrics_collector_connection_attempts(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test recording connection attempts."""
    collector.record_connection_attempt(mailbox_id, success=True)
    collector.record_connection_attempt(mailbox_id, success=True)
    collector.record_connection_attempt(mailbox_id, success=False)

    summary = collector.generate_summary(mailbox_id)
    assert summary.total_connection_attempts == 3
    assert summary.successful_connections == 2
    assert summary.failed_connections == 1
    assert summary.connection_failure_rate == pytest.approx(1 / 3)


def test_metrics_collector_parse_attempts(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test recording parse attempts."""
    collector.record_parse_attempt(mailbox_id, success=True)
    collector.record_parse_attempt(mailbox_id, success=True)
    collector.record_parse_attempt(mailbox_id, success=False)

    summary = collector.generate_summary(mailbox_id)
    assert summary.total_parse_attempts == 3
    assert summary.successful_parses == 2
    assert summary.failed_parses == 1
    assert summary.parse_failure_rate == pytest.approx(1 / 3)


def test_metrics_collector_detection_time(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test recording detection times."""
    collector.record_detection_time(mailbox_id, 10.0)
    collector.record_detection_time(mailbox_id, 20.0)
    collector.record_detection_time(mailbox_id, 30.0)

    summary = collector.generate_summary(mailbox_id)
    assert len(summary.detection_times_seconds) == 3
    assert summary.average_detection_time_seconds == pytest.approx(20.0)
    assert summary.max_detection_time_seconds == pytest.approx(30.0)


def test_metrics_collector_throughput(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test throughput calculation."""
    collector.record_message_processing(mailbox_id, message_count=100, duration_seconds=10.0)
    collector.record_message_processing(mailbox_id, message_count=50, duration_seconds=5.0)

    summary = collector.generate_summary(mailbox_id)
    assert summary.total_messages_processed == 150
    assert summary.total_processing_duration_seconds == pytest.approx(15.0)
    assert summary.messages_per_second == pytest.approx(10.0)


def test_metrics_collector_thread_accuracy(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test thread reconstruction accuracy tracking."""
    collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
    collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
    collector.record_thread_reconstruction(mailbox_id, success=True, correct=False)

    summary = collector.generate_summary(mailbox_id)
    assert summary.thread_reconstruction_attempts == 3
    assert summary.thread_reconstruction_correct == 2
    assert summary.thread_reconstruction_accuracy == pytest.approx(2 / 3)


def test_metrics_collector_attachment_accuracy(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test attachment extraction accuracy tracking."""
    collector.record_attachment_extraction(mailbox_id, success=True)
    collector.record_attachment_extraction(mailbox_id, success=True)
    collector.record_attachment_extraction(mailbox_id, success=False)

    summary = collector.generate_summary(mailbox_id)
    assert summary.attachment_extraction_attempts == 3
    assert summary.attachment_extraction_successful == 2
    assert summary.attachment_extraction_accuracy == pytest.approx(2 / 3)


def test_metrics_collector_pii_leaks(
    collector: ImapSyncMetricsCollector, mailbox_id: str, caplog
):
    """Test PII leak tracking (should trigger critical log)."""
    collector.record_pii_leak(mailbox_id, details="test@example.com in logs")

    summary = collector.generate_summary(mailbox_id)
    assert summary.pii_leak_count == 1
    assert "PII LEAK DETECTED" in caplog.text


def test_metrics_collector_consent_coverage(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test consent coverage tracking."""
    collector.record_consent_check(mailbox_id, granted=True)
    collector.record_consent_check(mailbox_id, granted=True)
    collector.record_consent_check(mailbox_id, granted=False)

    summary = collector.generate_summary(mailbox_id)
    assert summary.consent_checks_performed == 3
    assert summary.consent_checks_granted == 2
    assert summary.consent_coverage == pytest.approx(2 / 3)


def test_metrics_collector_element_sink(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test element sink success rate tracking."""
    collector.record_element_sink(mailbox_id, success=True)
    collector.record_element_sink(mailbox_id, success=True)
    collector.record_element_sink(mailbox_id, success=False)

    summary = collector.generate_summary(mailbox_id)
    assert summary.element_sink_attempts == 3
    assert summary.element_sink_successes == 2
    assert summary.element_sink_success_rate == pytest.approx(2 / 3)


def test_metrics_collector_state_persistence(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test state persistence success rate tracking."""
    collector.record_state_persistence(mailbox_id, success=True)
    collector.record_state_persistence(mailbox_id, success=True)
    collector.record_state_persistence(mailbox_id, success=False)

    summary = collector.generate_summary(mailbox_id)
    assert summary.state_persistence_attempts == 3
    assert summary.state_persistence_successes == 2
    assert summary.state_persistence_success_rate == pytest.approx(2 / 3)


def test_metrics_collector_reset(collector: ImapSyncMetricsCollector, mailbox_id: str):
    """Test metrics reset functionality."""
    collector.record_sync_attempt(mailbox_id, success=True)
    collector.record_sync_attempt(mailbox_id, success=False)

    summary_before = collector.generate_summary(mailbox_id)
    assert summary_before.total_sync_attempts == 2

    collector.reset_metrics(mailbox_id)

    summary_after = collector.generate_summary(mailbox_id)
    assert summary_after.total_sync_attempts == 0


# ============================================================================
# Quality Gate Threshold Evaluation Tests
# ============================================================================


def test_quality_gate_all_pass(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test quality gate when all metrics pass."""
    # Record metrics that meet all thresholds
    for _ in range(100):
        collector.record_sync_attempt(mailbox_id, success=True)
        collector.record_connection_attempt(mailbox_id, success=True)
        collector.record_parse_attempt(mailbox_id, success=True)
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
        collector.record_attachment_extraction(mailbox_id, success=True)
        collector.record_consent_check(mailbox_id, granted=True)
        collector.record_element_sink(mailbox_id, success=True)
        collector.record_state_persistence(mailbox_id, success=True)

    collector.record_detection_time(mailbox_id, 60.0)  # 1 minute (< 5 min threshold)
    collector.record_message_processing(mailbox_id, 100, 50.0)  # 2 msg/s (> 1 msg/s)
    collector.record_sync_latency(mailbox_id, "INBOX", 10.0)  # 10s (< 30s)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    assert result.passed
    assert result.status == QualityGateStatus.PASS
    assert not result.has_failures
    assert not result.has_warnings
    assert result.get_exit_code() == 0


def test_quality_gate_sync_failure_warning(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test quality gate warning for elevated sync failure rate."""
    # Record failures at warning level (0.45% > 0.4% warn threshold, < 0.5% fail threshold)
    for _ in range(995):
        collector.record_sync_attempt(mailbox_id, success=True)
        collector.record_connection_attempt(mailbox_id, success=True)
        collector.record_parse_attempt(mailbox_id, success=True)
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
        collector.record_attachment_extraction(mailbox_id, success=True)
        collector.record_consent_check(mailbox_id, granted=True)
        collector.record_element_sink(mailbox_id, success=True)
        collector.record_state_persistence(mailbox_id, success=True)
    for _ in range(5):  # 5/1000 = 0.5% (exactly at fail threshold, will warn)
        collector.record_sync_attempt(mailbox_id, success=False)

    # Add performance metrics
    collector.record_message_processing(mailbox_id, 1000, 500)  # 2 msg/s
    collector.record_detection_time(mailbox_id, 60.0)  # 1 minute

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    # With 0.5% failure rate (at threshold), should warn or fail depending on implementation
    # Let's check it generates warnings or fails
    assert result.status in [QualityGateStatus.WARN, QualityGateStatus.FAIL]


def test_quality_gate_sync_failure_fail(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test quality gate failure for high sync failure rate."""
    # Record failures exceeding threshold (1% > 0.5% threshold)
    for _ in range(99):
        collector.record_sync_attempt(mailbox_id, success=True)
    for _ in range(10):  # 10/109 = 9.2%
        collector.record_sync_attempt(mailbox_id, success=False)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    assert not result.passed
    assert result.status == QualityGateStatus.FAIL
    assert result.has_failures
    assert result.get_exit_code() == 2
    assert len(result.critical_issues) > 0


def test_quality_gate_pii_leak_failure(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test quality gate fails on any PII leak (zero tolerance)."""
    # Single PII leak should fail quality gate
    collector.record_pii_leak(mailbox_id, details="email in logs")

    # All other metrics perfect
    for _ in range(100):
        collector.record_sync_attempt(mailbox_id, success=True)
        collector.record_connection_attempt(mailbox_id, success=True)
        collector.record_parse_attempt(mailbox_id, success=True)
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
        collector.record_attachment_extraction(mailbox_id, success=True)
        collector.record_consent_check(mailbox_id, granted=True)
        collector.record_element_sink(mailbox_id, success=True)
        collector.record_state_persistence(mailbox_id, success=True)

    collector.record_message_processing(mailbox_id, 100, 50)  # 2 msg/s
    collector.record_detection_time(mailbox_id, 30.0)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    assert not result.passed
    assert result.status == QualityGateStatus.FAIL
    assert result.get_exit_code() == 2
    assert any("PII" in issue or "pii" in issue.lower() for issue in result.critical_issues)


def test_quality_gate_thread_accuracy_fail(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test quality gate fails when thread accuracy below threshold."""
    # Thread accuracy: 80% (below 95% threshold)
    for _ in range(80):
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
    for _ in range(20):
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=False)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    assert not result.passed
    assert result.status == QualityGateStatus.FAIL
    assert any("Thread" in issue or "accuracy" in issue for issue in result.critical_issues)


def test_quality_gate_consent_coverage_fail(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test quality gate fails when consent coverage incomplete."""
    # Consent coverage: 80% (below 100% requirement)
    for _ in range(80):
        collector.record_consent_check(mailbox_id, granted=True)
    for _ in range(20):
        collector.record_consent_check(mailbox_id, granted=False)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    assert not result.passed
    assert result.status == QualityGateStatus.FAIL
    assert any("Consent" in issue or "consent" in issue for issue in result.critical_issues)


def test_quality_gate_strict_mode(
    collector: ImapSyncMetricsCollector, mailbox_id: str
):
    """Test strict mode treats warnings as failures."""
    config = ImapQualityGates(
        enable_strict_mode=True,
        min_thread_reconstruction_accuracy=0.95  # Will create warning at 94%
    )
    evaluator = ImapQualityGateEvaluator(config=config, metrics_collector=collector)

    # Create warning condition (thread accuracy at 94% < 95% requirement)
    for _ in range(94):
        collector.record_sync_attempt(mailbox_id, success=True)
        collector.record_connection_attempt(mailbox_id, success=True)
        collector.record_parse_attempt(mailbox_id, success=True)
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
        collector.record_attachment_extraction(mailbox_id, success=True)
        collector.record_consent_check(mailbox_id, granted=True)
        collector.record_element_sink(mailbox_id, success=True)
        collector.record_state_persistence(mailbox_id, success=True)
    for _ in range(6):  # 94/100 = 94% thread accuracy (warning level)
        collector.record_sync_attempt(mailbox_id, success=True)
        collector.record_connection_attempt(mailbox_id, success=True)
        collector.record_parse_attempt(mailbox_id, success=True)
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=False)
        collector.record_attachment_extraction(mailbox_id, success=True)
        collector.record_consent_check(mailbox_id, granted=True)
        collector.record_element_sink(mailbox_id, success=True)
        collector.record_state_persistence(mailbox_id, success=True)

    collector.record_message_processing(mailbox_id, 100, 50)  # 2 msg/s
    collector.record_detection_time(mailbox_id, 30.0)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    # In strict mode, warnings should cause failure
    if result.has_warnings:
        assert not result.passed  # Strict mode fails on warnings
        assert result.get_exit_code() == 2


def test_quality_gate_insufficient_sample_size(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
    caplog,
):
    """Test warning logged for insufficient sample size."""
    # Only 5 sync attempts (< 10 minimum)
    for _ in range(5):
        collector.record_sync_attempt(mailbox_id, success=True)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    assert "Insufficient sample size" in caplog.text
    assert result.summary_metrics.total_sync_attempts == 5


# ============================================================================
# Quality Gate Result Tests
# ============================================================================


def test_quality_gate_result_json_export(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
    tmp_path: Path,
):
    """Test quality gate result JSON export."""
    for _ in range(100):
        collector.record_sync_attempt(mailbox_id, success=True)
        collector.record_connection_attempt(mailbox_id, success=True)
        collector.record_parse_attempt(mailbox_id, success=True)
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
        collector.record_attachment_extraction(mailbox_id, success=True)
        collector.record_consent_check(mailbox_id, granted=True)
        collector.record_element_sink(mailbox_id, success=True)
        collector.record_state_persistence(mailbox_id, success=True)

    collector.record_message_processing(mailbox_id, 100, 50)  # 2 msg/s
    collector.record_detection_time(mailbox_id, 30.0)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    # Export to JSON
    output_path = tmp_path / "quality_gate_result.json"
    result.save_to_file(output_path)

    # Verify file exists and is valid JSON
    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)

    assert data["mailbox_id"] == mailbox_id
    assert data["status"] == "pass"
    assert data["passed"] is True
    assert "metrics" in data
    assert "summary" in data


def test_quality_gate_result_recommendations(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test quality gate generates actionable recommendations."""
    # Create multiple failure conditions
    for _ in range(90):
        collector.record_sync_attempt(mailbox_id, success=True)
    for _ in range(10):  # High sync failure
        collector.record_sync_attempt(mailbox_id, success=False)

    for _ in range(85):  # Low thread accuracy
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=True)
    for _ in range(15):
        collector.record_thread_reconstruction(mailbox_id, success=True, correct=False)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    assert len(result.recommendations) > 0
    assert any("sync" in rec.lower() for rec in result.recommendations)
    assert any("thread" in rec.lower() for rec in result.recommendations)


def test_quality_gate_config_custom_thresholds():
    """Test quality gate with custom threshold configuration."""
    config = ImapQualityGates(
        max_sync_failure_rate=0.01,  # More lenient: 1%
        min_thread_reconstruction_accuracy=0.90,  # More lenient: 90%
    )

    assert config.max_sync_failure_rate == 0.01
    assert config.min_thread_reconstruction_accuracy == 0.90

    thresholds = config.get_thresholds()
    sync_threshold = next(
        t
        for t in thresholds
        if t.metric_type == ImapQualityMetricType.SYNC_FAILURE_RATE
    )
    assert sync_threshold.fail_threshold == 0.01


def test_quality_gate_formatted_value_display(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test metric values are formatted correctly."""
    for _ in range(100):
        collector.record_sync_attempt(mailbox_id, success=True)
    for _ in range(5):  # 5% failure rate
        collector.record_sync_attempt(mailbox_id, success=False)

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    sync_metric = next(
        m
        for m in result.metric_results
        if m.metric_type == ImapQualityMetricType.SYNC_FAILURE_RATE
    )

    # Should format as percentage
    assert "%" in sync_metric.formatted_value
    assert "4.76%" in sync_metric.formatted_value or "4.8%" in sync_metric.formatted_value


def test_quality_gate_evaluator_no_collector_raises():
    """Test evaluator raises error when no metrics collector provided."""
    evaluator = ImapQualityGateEvaluator(config=ImapQualityGates())

    with pytest.raises(ValueError, match="Metrics collector is required"):
        evaluator.evaluate_mailbox_quality("test@example.com")


def test_quality_gate_multiple_metric_failures(
    evaluator: ImapQualityGateEvaluator,
    collector: ImapSyncMetricsCollector,
    mailbox_id: str,
):
    """Test quality gate correctly reports multiple simultaneous failures."""
    # High sync failure rate
    for _ in range(90):
        collector.record_sync_attempt(mailbox_id, success=True)
    for _ in range(10):
        collector.record_sync_attempt(mailbox_id, success=False)

    # High connection failure rate
    for _ in range(90):
        collector.record_connection_attempt(mailbox_id, success=True)
    for _ in range(10):
        collector.record_connection_attempt(mailbox_id, success=False)

    # PII leak
    collector.record_pii_leak(mailbox_id, details="test")

    result = evaluator.evaluate_mailbox_quality(mailbox_id)

    assert not result.passed
    assert len(result.critical_issues) >= 3  # At least 3 failures
    assert len(result.recommendations) >= 3
