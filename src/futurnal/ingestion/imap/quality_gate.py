"""Quality gate system for IMAP connector with configurable thresholds.

This module implements the quality gates defined in
``docs/phase-1/imap-connector-production-plan/10-quality-gates-testing.md``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .sync_metrics import ImapSyncMetricsCollector, ImapSyncMetricsSummary

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate evaluation status."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ImapQualityMetricType(Enum):
    """Types of quality metrics for IMAP connector."""

    # Reliability
    SYNC_FAILURE_RATE = "sync_failure_rate"
    CONNECTION_FAILURE_RATE = "connection_failure_rate"
    PARSE_FAILURE_RATE = "parse_failure_rate"

    # Performance
    DETECTION_WINDOW_MINUTES = "detection_window_minutes"
    THROUGHPUT_MINIMUM = "throughput_minimum"
    SYNC_LATENCY_MAXIMUM = "sync_latency_maximum"

    # Accuracy
    THREAD_RECONSTRUCTION_ACCURACY = "thread_reconstruction_accuracy"
    ATTACHMENT_EXTRACTION_ACCURACY = "attachment_extraction_accuracy"

    # Privacy
    PII_LEAKS = "pii_leaks"
    CONSENT_COVERAGE = "consent_coverage"

    # Integration
    ELEMENT_SINK_SUCCESS_RATE = "element_sink_success_rate"
    STATE_PERSISTENCE_SUCCESS_RATE = "state_persistence_success_rate"


@dataclass
class QualityThreshold:
    """Defines a quality threshold with warning and failure levels."""

    metric_type: ImapQualityMetricType
    warn_threshold: float
    fail_threshold: float
    description: str
    unit: str = ""
    higher_is_better: bool = False

    def evaluate(self, value: float) -> QualityGateStatus:
        """Evaluate a value against this threshold."""
        if self.higher_is_better:
            if value < self.fail_threshold:
                return QualityGateStatus.FAIL
            elif value < self.warn_threshold:
                return QualityGateStatus.WARN
            else:
                return QualityGateStatus.PASS
        else:
            if value > self.fail_threshold:
                return QualityGateStatus.FAIL
            elif value > self.warn_threshold:
                return QualityGateStatus.WARN
            else:
                return QualityGateStatus.PASS


class ImapQualityGates(BaseModel):
    """Quality gate thresholds for IMAP connector.

    These thresholds match the requirements from the production plan:
    - <0.5% sync failure rate
    - <1% connection failure rate
    - <2% parse failure rate
    - 5-minute detection window
    - ≥1 msg/s throughput
    - <30s sync latency per folder
    - ≥95% thread reconstruction accuracy
    - ≥98% attachment extraction accuracy
    - Zero PII leaks (zero tolerance)
    - 100% consent coverage
    - ≥99% element sink success
    - 100% state persistence success
    """

    # Reliability thresholds (lower is better)
    max_sync_failure_rate: float = 0.005  # <0.5%
    max_connection_failure_rate: float = 0.01  # <1%
    max_parse_failure_rate: float = 0.02  # <2%

    # Performance thresholds
    max_detection_window_minutes: float = 5.0  # 5-minute detection
    min_throughput_messages_per_second: float = 1.0  # ≥1 msg/s
    max_sync_latency_seconds: float = 30.0  # <30s per folder

    # Accuracy thresholds (higher is better)
    min_thread_reconstruction_accuracy: float = 0.95  # 95% correct
    min_attachment_extraction_accuracy: float = 0.98  # 98% correct

    # Privacy thresholds
    zero_pii_in_logs: bool = True  # No PII leaked (zero tolerance)
    require_consent_coverage: float = 1.0  # 100% consent checks

    # Integration thresholds (higher is better)
    min_element_sink_success_rate: float = 0.99  # 99% elements processed
    min_state_persistence_success_rate: float = 1.0  # 100% state saved

    # Evaluation settings
    enable_strict_mode: bool = False  # Treat warnings as failures
    evaluation_time_window_hours: int = 1  # Time window for metrics
    require_minimum_sample_size: int = 10  # Minimum operations for evaluation

    def get_thresholds(self) -> List[QualityThreshold]:
        """Get all configured quality thresholds."""
        return [
            # Reliability
            QualityThreshold(
                metric_type=ImapQualityMetricType.SYNC_FAILURE_RATE,
                warn_threshold=self.max_sync_failure_rate * 0.8,  # 80% of max
                fail_threshold=self.max_sync_failure_rate,
                description="Sync failure rate",
                unit="%",
            ),
            QualityThreshold(
                metric_type=ImapQualityMetricType.CONNECTION_FAILURE_RATE,
                warn_threshold=self.max_connection_failure_rate * 0.8,
                fail_threshold=self.max_connection_failure_rate,
                description="Connection failure rate",
                unit="%",
            ),
            QualityThreshold(
                metric_type=ImapQualityMetricType.PARSE_FAILURE_RATE,
                warn_threshold=self.max_parse_failure_rate * 0.8,
                fail_threshold=self.max_parse_failure_rate,
                description="Email parse failure rate",
                unit="%",
            ),
            # Performance
            QualityThreshold(
                metric_type=ImapQualityMetricType.DETECTION_WINDOW_MINUTES,
                warn_threshold=self.max_detection_window_minutes * 0.8,  # 4 minutes
                fail_threshold=self.max_detection_window_minutes,  # 5 minutes
                description="Message detection window",
                unit="minutes",
            ),
            QualityThreshold(
                metric_type=ImapQualityMetricType.THROUGHPUT_MINIMUM,
                warn_threshold=self.min_throughput_messages_per_second,
                fail_threshold=self.min_throughput_messages_per_second * 0.5,  # 0.5 msg/s
                description="Message processing throughput",
                unit="msg/s",
                higher_is_better=True,
            ),
            QualityThreshold(
                metric_type=ImapQualityMetricType.SYNC_LATENCY_MAXIMUM,
                warn_threshold=self.max_sync_latency_seconds * 0.8,  # 24 seconds
                fail_threshold=self.max_sync_latency_seconds,  # 30 seconds
                description="Sync latency per folder",
                unit="seconds",
            ),
            # Accuracy
            QualityThreshold(
                metric_type=ImapQualityMetricType.THREAD_RECONSTRUCTION_ACCURACY,
                warn_threshold=self.min_thread_reconstruction_accuracy,
                fail_threshold=self.min_thread_reconstruction_accuracy * 0.9,  # 85.5%
                description="Thread reconstruction accuracy",
                unit="%",
                higher_is_better=True,
            ),
            QualityThreshold(
                metric_type=ImapQualityMetricType.ATTACHMENT_EXTRACTION_ACCURACY,
                warn_threshold=self.min_attachment_extraction_accuracy,
                fail_threshold=self.min_attachment_extraction_accuracy * 0.9,  # 88.2%
                description="Attachment extraction accuracy",
                unit="%",
                higher_is_better=True,
            ),
            # Privacy (special handling: 0 is pass, ≥1 is fail, no warning level)
            QualityThreshold(
                metric_type=ImapQualityMetricType.PII_LEAKS,
                warn_threshold=0.5,  # Threshold between 0 and 1 to trigger fail on any leak
                fail_threshold=0.5,  # Same as warn: any value >= 0.5 fails (so 1+ fails)
                description="PII leaks in logs (zero tolerance)",
                unit="count",
            ),
            QualityThreshold(
                metric_type=ImapQualityMetricType.CONSENT_COVERAGE,
                warn_threshold=self.require_consent_coverage,
                fail_threshold=self.require_consent_coverage * 0.9,  # 90%
                description="Consent coverage",
                unit="%",
                higher_is_better=True,
            ),
            # Integration
            QualityThreshold(
                metric_type=ImapQualityMetricType.ELEMENT_SINK_SUCCESS_RATE,
                warn_threshold=self.min_element_sink_success_rate,
                fail_threshold=self.min_element_sink_success_rate * 0.95,  # 94.05%
                description="Element sink success rate",
                unit="%",
                higher_is_better=True,
            ),
            QualityThreshold(
                metric_type=ImapQualityMetricType.STATE_PERSISTENCE_SUCCESS_RATE,
                warn_threshold=self.min_state_persistence_success_rate,
                fail_threshold=self.min_state_persistence_success_rate * 0.95,  # 95%
                description="State persistence success rate",
                unit="%",
                higher_is_better=True,
            ),
        ]


@dataclass
class QualityMetricResult:
    """Result of evaluating a single quality metric."""

    metric_type: ImapQualityMetricType
    status: QualityGateStatus
    value: float
    threshold: QualityThreshold
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def formatted_value(self) -> str:
        """Get formatted value with appropriate unit."""
        if self.threshold.unit == "%":
            return f"{self.value * 100:.2f}%"
        elif self.threshold.unit:
            return f"{self.value:.2f} {self.threshold.unit}"
        else:
            return f"{self.value:.2f}"


@dataclass
class ImapQualityGateResult:
    """Comprehensive result of quality gate evaluation."""

    mailbox_id: str
    status: QualityGateStatus
    evaluated_at: datetime
    config: ImapQualityGates
    metric_results: List[QualityMetricResult]
    summary_metrics: ImapSyncMetricsSummary

    # Error and warning details
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if quality gate passed (no failures)."""
        return self.status != QualityGateStatus.FAIL

    @property
    def has_warnings(self) -> bool:
        """Check if quality gate has warnings."""
        return any(
            result.status == QualityGateStatus.WARN for result in self.metric_results
        )

    @property
    def has_failures(self) -> bool:
        """Check if quality gate has failures."""
        return any(
            result.status == QualityGateStatus.FAIL for result in self.metric_results
        )

    def get_exit_code(self) -> int:
        """Get appropriate exit code for CI/CD integration."""
        if self.status == QualityGateStatus.FAIL:
            return 2  # Failure - should block CI/CD
        elif self.status == QualityGateStatus.WARN and self.config.enable_strict_mode:
            return 2  # Treat warnings as failures in strict mode
        elif self.status == QualityGateStatus.WARN:
            return 1  # Warnings present but not blocking
        else:
            return 0  # Success

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "mailbox_id": self.mailbox_id,
            "status": self.status.value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "passed": self.passed,
            "exit_code": self.get_exit_code(),
            "metrics": [
                {
                    "type": result.metric_type.value,
                    "status": result.status.value,
                    "value": result.value,
                    "formatted_value": result.formatted_value,
                    "message": result.message,
                    "details": result.details,
                }
                for result in self.metric_results
            ],
            "summary": {
                "sync_failure_rate": self.summary_metrics.sync_failure_rate,
                "connection_failure_rate": self.summary_metrics.connection_failure_rate,
                "parse_failure_rate": self.summary_metrics.parse_failure_rate,
                "average_detection_time_seconds": self.summary_metrics.average_detection_time_seconds,
                "messages_per_second": self.summary_metrics.messages_per_second,
                "thread_reconstruction_accuracy": self.summary_metrics.thread_reconstruction_accuracy,
                "pii_leak_count": self.summary_metrics.pii_leak_count,
            },
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def save_to_file(self, file_path: Path) -> None:
        """Save quality gate result to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Quality gate result saved to {file_path}")


class ImapQualityGateEvaluator:
    """Evaluates IMAP connector quality against configurable thresholds."""

    def __init__(
        self,
        config: Optional[ImapQualityGates] = None,
        metrics_collector: Optional[ImapSyncMetricsCollector] = None,
    ):
        self.config = config or ImapQualityGates()
        self.metrics_collector = metrics_collector

    def evaluate_mailbox_quality(
        self,
        mailbox_id: str,
        metrics_collector: Optional[ImapSyncMetricsCollector] = None,
    ) -> ImapQualityGateResult:
        """Evaluate quality gate for a specific mailbox.

        Args:
            mailbox_id: Mailbox identifier
            metrics_collector: Optional metrics collector (uses instance default if not provided)

        Returns:
            Quality gate evaluation result

        Raises:
            ValueError: If no metrics collector available
        """
        collector = metrics_collector or self.metrics_collector
        if not collector:
            raise ValueError("Metrics collector is required for quality evaluation")

        logger.info(f"Evaluating quality gate for mailbox {mailbox_id}")

        # Generate metrics summary for evaluation period
        summary = collector.generate_summary(
            mailbox_id=mailbox_id, hours=self.config.evaluation_time_window_hours
        )

        # Check minimum sample size
        if summary.total_sync_attempts < self.config.require_minimum_sample_size:
            logger.warning(
                f"Insufficient sample size for quality evaluation: "
                f"{summary.total_sync_attempts} < {self.config.require_minimum_sample_size}"
            )

        # Evaluate each quality threshold
        metric_results = []
        thresholds = self.config.get_thresholds()

        for threshold in thresholds:
            result = self._evaluate_metric(threshold, summary)
            metric_results.append(result)
            logger.debug(
                f"Quality metric {threshold.metric_type.value}: {result.status.value} ({result.formatted_value})"
            )

        # Determine overall status
        overall_status = self._determine_overall_status(metric_results)

        # Create comprehensive result
        result = ImapQualityGateResult(
            mailbox_id=mailbox_id,
            status=overall_status,
            evaluated_at=datetime.utcnow(),
            config=self.config,
            metric_results=metric_results,
            summary_metrics=summary,
        )

        # Generate actionable insights
        self._generate_insights(result)

        logger.info(
            f"Quality gate evaluation completed for mailbox {mailbox_id}: {overall_status.value}"
        )
        return result

    def _evaluate_metric(
        self, threshold: QualityThreshold, summary: ImapSyncMetricsSummary
    ) -> QualityMetricResult:
        """Evaluate a single quality metric against its threshold."""
        metric_type = threshold.metric_type
        value = 0.0
        details = {}
        message = ""

        try:
            if metric_type == ImapQualityMetricType.SYNC_FAILURE_RATE:
                value = summary.sync_failure_rate
                details = {
                    "failed_syncs": summary.failed_syncs,
                    "total_syncs": summary.total_sync_attempts,
                }
                message = f"Sync failure rate: {value*100:.2f}% ({summary.failed_syncs}/{summary.total_sync_attempts})"

            elif metric_type == ImapQualityMetricType.CONNECTION_FAILURE_RATE:
                value = summary.connection_failure_rate
                details = {
                    "failed_connections": summary.failed_connections,
                    "total_connections": summary.total_connection_attempts,
                }
                message = f"Connection failure rate: {value*100:.2f}% ({summary.failed_connections}/{summary.total_connection_attempts})"

            elif metric_type == ImapQualityMetricType.PARSE_FAILURE_RATE:
                value = summary.parse_failure_rate
                details = {
                    "failed_parses": summary.failed_parses,
                    "total_parses": summary.total_parse_attempts,
                }
                message = f"Parse failure rate: {value*100:.2f}% ({summary.failed_parses}/{summary.total_parse_attempts})"

            elif metric_type == ImapQualityMetricType.DETECTION_WINDOW_MINUTES:
                value = summary.average_detection_time_seconds / 60.0  # Convert to minutes
                details = {
                    "average_seconds": summary.average_detection_time_seconds,
                    "max_seconds": summary.max_detection_time_seconds,
                    "sample_count": len(summary.detection_times_seconds),
                }
                message = f"Detection window: {value:.2f} minutes (avg)"

            elif metric_type == ImapQualityMetricType.THROUGHPUT_MINIMUM:
                value = summary.messages_per_second
                details = {
                    "messages_processed": summary.total_messages_processed,
                    "duration_seconds": summary.total_processing_duration_seconds,
                }
                message = f"Throughput: {value:.2f} msg/s"

            elif metric_type == ImapQualityMetricType.SYNC_LATENCY_MAXIMUM:
                value = summary.average_sync_latency_seconds
                details = {
                    "average_latency": summary.average_sync_latency_seconds,
                    "sample_count": len(summary.sync_latencies),
                }
                message = f"Sync latency: {value:.2f} seconds (avg)"

            elif metric_type == ImapQualityMetricType.THREAD_RECONSTRUCTION_ACCURACY:
                value = summary.thread_reconstruction_accuracy
                details = {
                    "correct": summary.thread_reconstruction_correct,
                    "total": summary.thread_reconstruction_attempts,
                }
                message = f"Thread accuracy: {value*100:.2f}% ({summary.thread_reconstruction_correct}/{summary.thread_reconstruction_attempts})"

            elif metric_type == ImapQualityMetricType.ATTACHMENT_EXTRACTION_ACCURACY:
                value = summary.attachment_extraction_accuracy
                details = {
                    "successful": summary.attachment_extraction_successful,
                    "total": summary.attachment_extraction_attempts,
                }
                message = f"Attachment accuracy: {value*100:.2f}% ({summary.attachment_extraction_successful}/{summary.attachment_extraction_attempts})"

            elif metric_type == ImapQualityMetricType.PII_LEAKS:
                value = float(summary.pii_leak_count)
                details = {"pii_leak_count": summary.pii_leak_count}
                message = f"PII leaks: {summary.pii_leak_count} (CRITICAL if > 0)"

            elif metric_type == ImapQualityMetricType.CONSENT_COVERAGE:
                value = summary.consent_coverage
                details = {
                    "granted": summary.consent_checks_granted,
                    "total": summary.consent_checks_performed,
                }
                message = f"Consent coverage: {value*100:.2f}% ({summary.consent_checks_granted}/{summary.consent_checks_performed})"

            elif metric_type == ImapQualityMetricType.ELEMENT_SINK_SUCCESS_RATE:
                value = summary.element_sink_success_rate
                details = {
                    "successes": summary.element_sink_successes,
                    "total": summary.element_sink_attempts,
                }
                message = f"Element sink success: {value*100:.2f}% ({summary.element_sink_successes}/{summary.element_sink_attempts})"

            elif metric_type == ImapQualityMetricType.STATE_PERSISTENCE_SUCCESS_RATE:
                value = summary.state_persistence_success_rate
                details = {
                    "successes": summary.state_persistence_successes,
                    "total": summary.state_persistence_attempts,
                }
                message = f"State persistence success: {value*100:.2f}% ({summary.state_persistence_successes}/{summary.state_persistence_attempts})"

            else:
                message = f"Unknown metric type: {metric_type.value}"
                logger.warning(message)

        except Exception as e:
            logger.error(f"Failed to evaluate metric {metric_type.value}: {e}")
            value = 0.0
            message = f"Evaluation failed: {str(e)}"
            details["error"] = str(e)

        status = threshold.evaluate(value)

        return QualityMetricResult(
            metric_type=metric_type,
            status=status,
            value=value,
            threshold=threshold,
            message=message,
            details=details,
        )

    def _determine_overall_status(
        self, metric_results: List[QualityMetricResult]
    ) -> QualityGateStatus:
        """Determine overall quality gate status from individual metric results."""
        has_failures = any(
            result.status == QualityGateStatus.FAIL for result in metric_results
        )
        has_warnings = any(
            result.status == QualityGateStatus.WARN for result in metric_results
        )

        if has_failures:
            return QualityGateStatus.FAIL
        elif has_warnings:
            if self.config.enable_strict_mode:
                return QualityGateStatus.FAIL
            else:
                return QualityGateStatus.WARN
        else:
            return QualityGateStatus.PASS

    def _generate_insights(self, result: ImapQualityGateResult) -> None:
        """Generate actionable insights and recommendations."""
        result.critical_issues.clear()
        result.warnings.clear()
        result.recommendations.clear()

        for metric_result in result.metric_results:
            if metric_result.status == QualityGateStatus.FAIL:
                result.critical_issues.append(
                    f"{metric_result.threshold.description}: {metric_result.message}"
                )
            elif metric_result.status == QualityGateStatus.WARN:
                result.warnings.append(
                    f"{metric_result.threshold.description}: {metric_result.message}"
                )

        # Generate recommendations based on issues found
        if result.has_failures or result.has_warnings:
            for metric_result in result.metric_results:
                if metric_result.status in [
                    QualityGateStatus.FAIL,
                    QualityGateStatus.WARN,
                ]:
                    recommendation = self._get_metric_recommendation(metric_result)
                    if recommendation:
                        result.recommendations.append(recommendation)

    def _get_metric_recommendation(
        self, metric_result: QualityMetricResult
    ) -> Optional[str]:
        """Get actionable recommendation for a specific metric issue."""
        recommendations = {
            ImapQualityMetricType.SYNC_FAILURE_RATE: "Review sync error logs and address common failure patterns",
            ImapQualityMetricType.CONNECTION_FAILURE_RATE: "Check network stability and IMAP server health",
            ImapQualityMetricType.PARSE_FAILURE_RATE: "Investigate malformed email formats causing parse errors",
            ImapQualityMetricType.DETECTION_WINDOW_MINUTES: "Optimize IDLE monitoring or reduce polling interval",
            ImapQualityMetricType.THROUGHPUT_MINIMUM: "Review processing pipeline for performance bottlenecks",
            ImapQualityMetricType.SYNC_LATENCY_MAXIMUM: "Optimize folder sync strategy or reduce message batch sizes",
            ImapQualityMetricType.THREAD_RECONSTRUCTION_ACCURACY: "Review thread reconstruction algorithm and test with complex thread structures",
            ImapQualityMetricType.ATTACHMENT_EXTRACTION_ACCURACY: "Investigate attachment extraction failures and edge cases",
            ImapQualityMetricType.PII_LEAKS: "CRITICAL: Review all logging statements and apply redaction policies",
            ImapQualityMetricType.CONSENT_COVERAGE: "Ensure consent checks are performed for all mailbox operations",
            ImapQualityMetricType.ELEMENT_SINK_SUCCESS_RATE: "Review element sink integration and error handling",
            ImapQualityMetricType.STATE_PERSISTENCE_SUCCESS_RATE: "Investigate state persistence failures and database health",
        }
        return recommendations.get(metric_result.metric_type)


def create_quality_gate_evaluator(
    config: Optional[ImapQualityGates] = None,
    metrics_collector: Optional[ImapSyncMetricsCollector] = None,
) -> ImapQualityGateEvaluator:
    """Factory function to create a quality gate evaluator with sensible defaults."""
    return ImapQualityGateEvaluator(
        config=config or ImapQualityGates(), metrics_collector=metrics_collector
    )
