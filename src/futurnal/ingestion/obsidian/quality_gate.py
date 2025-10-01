"""Quality gate system for Obsidian vault ingestion with configurable thresholds and comprehensive reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from .sync_metrics import SyncMetricsCollector, SyncMetricsSummary

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate evaluation status."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class QualityMetricType(Enum):
    """Types of quality metrics that can be evaluated."""
    ERROR_RATE = "error_rate"
    PARSE_FAILURE_RATE = "parse_failure_rate"
    BROKEN_LINK_RATE = "broken_link_rate"
    MISSING_REFERENCE_RATE = "missing_reference_rate"
    THROUGHPUT_MINIMUM = "throughput_minimum"
    PROCESSING_TIME_MAXIMUM = "processing_time_maximum"
    CONSENT_COVERAGE = "consent_coverage"
    ASSET_PROCESSING_SUCCESS_RATE = "asset_processing_success_rate"
    QUARANTINE_RATE = "quarantine_rate"


@dataclass
class QualityThreshold:
    """Defines a quality threshold with warning and failure levels."""
    metric_type: QualityMetricType
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


@dataclass
class QualityGateConfig:
    """Configuration for quality gate evaluation with customizable thresholds."""

    # Error rate thresholds (lower is better)
    max_error_rate: float = 0.05  # 5% error rate triggers warning
    max_critical_error_rate: float = 0.10  # 10% error rate triggers failure

    # Parse failure thresholds (lower is better)
    max_parse_failure_rate: float = 0.02  # 2% parse failure rate triggers warning
    max_critical_parse_failure_rate: float = 0.05  # 5% parse failure rate triggers failure

    # Link quality thresholds (lower is better)
    max_broken_link_rate: float = 0.03  # 3% broken links triggers warning
    max_critical_broken_link_rate: float = 0.08  # 8% broken links triggers failure

    # Performance thresholds
    min_throughput_events_per_second: float = 1.0  # Minimum 1 event/second (higher is better)
    min_critical_throughput_events_per_second: float = 0.5  # Critical minimum 0.5 events/second

    max_avg_processing_time_seconds: float = 5.0  # Maximum 5 seconds average processing time
    max_critical_processing_time_seconds: float = 10.0  # Critical maximum 10 seconds

    # Privacy and consent thresholds (higher is better)
    min_consent_coverage_rate: float = 0.95  # 95% consent coverage required
    min_critical_consent_coverage_rate: float = 0.90  # 90% critical minimum

    # Asset processing thresholds (higher is better)
    min_asset_processing_success_rate: float = 0.90  # 90% asset processing success
    min_critical_asset_success_rate: float = 0.80  # 80% critical minimum

    # Quarantine thresholds (lower is better)
    max_quarantine_rate: float = 0.02  # 2% quarantine rate triggers warning
    max_critical_quarantine_rate: float = 0.05  # 5% quarantine rate triggers failure

    # Evaluation settings
    enable_strict_mode: bool = False  # In strict mode, warnings are treated as failures
    evaluation_time_window_hours: int = 1  # Time window for metrics evaluation
    require_minimum_sample_size: int = 10  # Minimum events required for meaningful evaluation

    def get_thresholds(self) -> List[QualityThreshold]:
        """Get all configured quality thresholds."""
        return [
            QualityThreshold(
                metric_type=QualityMetricType.ERROR_RATE,
                warn_threshold=self.max_error_rate,
                fail_threshold=self.max_critical_error_rate,
                description="Overall error rate during ingestion",
                unit="%"
            ),
            QualityThreshold(
                metric_type=QualityMetricType.PARSE_FAILURE_RATE,
                warn_threshold=self.max_parse_failure_rate,
                fail_threshold=self.max_critical_parse_failure_rate,
                description="Rate of document parsing failures",
                unit="%"
            ),
            QualityThreshold(
                metric_type=QualityMetricType.BROKEN_LINK_RATE,
                warn_threshold=self.max_broken_link_rate,
                fail_threshold=self.max_critical_broken_link_rate,
                description="Rate of broken wikilinks and references",
                unit="%"
            ),
            QualityThreshold(
                metric_type=QualityMetricType.THROUGHPUT_MINIMUM,
                warn_threshold=self.min_throughput_events_per_second,
                fail_threshold=self.min_critical_throughput_events_per_second,
                description="Minimum ingestion throughput",
                unit="events/sec",
                higher_is_better=True
            ),
            QualityThreshold(
                metric_type=QualityMetricType.PROCESSING_TIME_MAXIMUM,
                warn_threshold=self.max_avg_processing_time_seconds,
                fail_threshold=self.max_critical_processing_time_seconds,
                description="Maximum average processing time per event",
                unit="seconds"
            ),
            QualityThreshold(
                metric_type=QualityMetricType.CONSENT_COVERAGE,
                warn_threshold=self.min_consent_coverage_rate,
                fail_threshold=self.min_critical_consent_coverage_rate,
                description="Consent coverage for processed files",
                unit="%",
                higher_is_better=True
            ),
            QualityThreshold(
                metric_type=QualityMetricType.ASSET_PROCESSING_SUCCESS_RATE,
                warn_threshold=self.min_asset_processing_success_rate,
                fail_threshold=self.min_critical_asset_success_rate,
                description="Asset processing success rate",
                unit="%",
                higher_is_better=True
            ),
            QualityThreshold(
                metric_type=QualityMetricType.QUARANTINE_RATE,
                warn_threshold=self.max_quarantine_rate,
                fail_threshold=self.max_critical_quarantine_rate,
                description="Rate of files quarantined due to processing errors",
                unit="%"
            ),
        ]


@dataclass
class QualityMetricResult:
    """Result of evaluating a single quality metric."""
    metric_type: QualityMetricType
    status: QualityGateStatus
    value: float
    threshold: QualityThreshold
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def formatted_value(self) -> str:
        """Get formatted value with appropriate unit."""
        if self.threshold.unit == "%":
            return f"{self.value * 100:.1f}%"
        elif self.threshold.unit:
            return f"{self.value:.2f} {self.threshold.unit}"
        else:
            return f"{self.value:.2f}"


@dataclass
class QualityGateResult:
    """Comprehensive result of quality gate evaluation."""
    vault_id: str
    status: QualityGateStatus
    evaluated_at: datetime
    config: QualityGateConfig
    metric_results: List[QualityMetricResult]
    summary_metrics: SyncMetricsSummary

    # Aggregate statistics
    total_files_processed: int = 0
    total_files_failed: int = 0
    total_links_detected: int = 0
    total_broken_links: int = 0
    total_assets_processed: int = 0
    total_assets_failed: int = 0

    # Privacy and consent statistics
    consent_granted_files: int = 0
    redactions_applied: int = 0
    privacy_policy_violations: int = 0

    # Performance statistics
    average_processing_time: float = 0.0
    throughput_events_per_second: float = 0.0

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
        return any(result.status == QualityGateStatus.WARN for result in self.metric_results)

    @property
    def has_failures(self) -> bool:
        """Check if quality gate has failures."""
        return any(result.status == QualityGateStatus.FAIL for result in self.metric_results)

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


class QualityGateEvaluator:
    """Evaluates vault ingestion quality against configurable thresholds."""

    def __init__(
        self,
        config: Optional[QualityGateConfig] = None,
        metrics_collector: Optional[SyncMetricsCollector] = None
    ):
        self.config = config or QualityGateConfig()
        self.metrics_collector = metrics_collector

    def evaluate_vault_quality(
        self,
        vault_id: str,
        metrics_collector: Optional[SyncMetricsCollector] = None
    ) -> QualityGateResult:
        """Evaluate quality gate for a specific vault."""
        collector = metrics_collector or self.metrics_collector
        if not collector:
            raise ValueError("Metrics collector is required for quality evaluation")

        logger.info(f"Evaluating quality gate for vault {vault_id}")

        # Generate metrics summary for evaluation period
        summary = collector.generate_summary(
            vault_id=vault_id,
            hours=self.config.evaluation_time_window_hours
        )

        # Check minimum sample size
        if summary.total_events_processed < self.config.require_minimum_sample_size:
            logger.warning(
                f"Insufficient sample size for quality evaluation: "
                f"{summary.total_events_processed} < {self.config.require_minimum_sample_size}"
            )

        # Evaluate each quality threshold
        metric_results = []
        thresholds = self.config.get_thresholds()

        for threshold in thresholds:
            result = self._evaluate_metric(threshold, summary, collector)
            metric_results.append(result)
            logger.debug(f"Quality metric {threshold.metric_type.value}: {result.status.value} ({result.formatted_value})")

        # Determine overall status
        overall_status = self._determine_overall_status(metric_results)

        # Create comprehensive result
        result = QualityGateResult(
            vault_id=vault_id,
            status=overall_status,
            evaluated_at=datetime.utcnow(),
            config=self.config,
            metric_results=metric_results,
            summary_metrics=summary
        )

        # Populate aggregate statistics
        self._populate_aggregate_statistics(result, summary, collector)

        # Generate actionable insights
        self._generate_insights(result)

        logger.info(f"Quality gate evaluation completed for vault {vault_id}: {overall_status.value}")
        return result

    def _evaluate_metric(
        self,
        threshold: QualityThreshold,
        summary: SyncMetricsSummary,
        collector: SyncMetricsCollector
    ) -> QualityMetricResult:
        """Evaluate a single quality metric against its threshold."""

        metric_type = threshold.metric_type
        value = 0.0
        details = {}
        message = ""

        try:
            if metric_type == QualityMetricType.ERROR_RATE:
                if summary.total_events_processed > 0:
                    value = summary.failed_events / summary.total_events_processed
                    details = {
                        "failed_events": summary.failed_events,
                        "total_events": summary.total_events_processed
                    }
                message = f"Error rate: {value*100:.1f}% ({summary.failed_events}/{summary.total_events_processed})"

            elif metric_type == QualityMetricType.THROUGHPUT_MINIMUM:
                value = summary.events_per_second
                details = {
                    "events_per_second": value,
                    "total_events": summary.total_events_processed,
                    "time_window_hours": self.config.evaluation_time_window_hours
                }
                message = f"Throughput: {value:.2f} events/second"

            elif metric_type == QualityMetricType.PROCESSING_TIME_MAXIMUM:
                value = summary.average_event_processing_time
                details = {
                    "average_processing_time": value,
                    "batch_processing_time": summary.average_batch_processing_time
                }
                message = f"Average processing time: {value:.2f} seconds"

            elif metric_type == QualityMetricType.QUARANTINE_RATE:
                # Get quarantine statistics from collector
                quarantine_count = collector.get_counter("quarantine_events", labels={"vault_id": summary.vault_id})
                if summary.total_events_processed > 0:
                    value = quarantine_count / summary.total_events_processed
                details = {
                    "quarantine_count": quarantine_count,
                    "total_events": summary.total_events_processed
                }
                message = f"Quarantine rate: {value*100:.1f}% ({quarantine_count}/{summary.total_events_processed})"

            elif metric_type == QualityMetricType.PARSE_FAILURE_RATE:
                # Get parse failure and success counts
                parse_failures = collector.get_counter("parse_failures", labels={"vault_id": summary.vault_id})
                parse_successes = collector.get_counter("parse_successes", labels={"vault_id": summary.vault_id})
                total_parse_attempts = parse_failures + parse_successes

                if total_parse_attempts > 0:
                    value = parse_failures / total_parse_attempts
                    details = {
                        "parse_failures": parse_failures,
                        "parse_successes": parse_successes,
                        "total_attempts": total_parse_attempts
                    }
                    message = f"Parse failure rate: {value*100:.1f}% ({parse_failures}/{total_parse_attempts})"
                else:
                    value = 0.0
                    details = {"parse_failures": 0, "parse_successes": 0, "total_attempts": 0}
                    message = "Parse failure rate: No parse attempts recorded"

            elif metric_type == QualityMetricType.BROKEN_LINK_RATE:
                # Get broken link and total link counts
                broken_links = collector.get_counter("broken_links", labels={"vault_id": summary.vault_id})
                total_links = collector.get_counter("total_links", labels={"vault_id": summary.vault_id})

                if total_links > 0:
                    value = broken_links / total_links
                    details = {
                        "broken_links": broken_links,
                        "total_links": total_links
                    }
                    message = f"Broken link rate: {value*100:.1f}% ({broken_links}/{total_links})"
                else:
                    value = 0.0
                    details = {"broken_links": 0, "total_links": 0}
                    message = "Broken link rate: No links found"

            elif metric_type == QualityMetricType.MISSING_REFERENCE_RATE:
                # Use broken links as proxy for missing references
                broken_links = collector.get_counter("broken_links", labels={"vault_id": summary.vault_id})
                total_links = collector.get_counter("total_links", labels={"vault_id": summary.vault_id})

                if total_links > 0:
                    value = broken_links / total_links
                    details = {
                        "missing_references": broken_links,
                        "total_references": total_links
                    }
                    message = f"Missing reference rate: {value*100:.1f}% ({broken_links}/{total_links})"
                else:
                    value = 0.0
                    details = {"missing_references": 0, "total_references": 0}
                    message = "Missing reference rate: No references found"

            elif metric_type == QualityMetricType.ASSET_PROCESSING_SUCCESS_RATE:
                # Get asset processing statistics
                processable_assets = collector.get_counter("processable_assets", labels={"vault_id": summary.vault_id})
                broken_assets = collector.get_counter("broken_assets", labels={"vault_id": summary.vault_id})
                total_assets = collector.get_counter("total_assets", labels={"vault_id": summary.vault_id})

                if processable_assets > 0:
                    successful_assets = processable_assets - broken_assets
                    value = successful_assets / processable_assets
                    details = {
                        "successful_assets": successful_assets,
                        "failed_assets": broken_assets,
                        "processable_assets": processable_assets,
                        "total_assets": total_assets
                    }
                    message = f"Asset processing success rate: {value*100:.1f}% ({successful_assets}/{processable_assets})"
                else:
                    value = 1.0  # If no processable assets, consider it 100% success
                    details = {"successful_assets": 0, "failed_assets": 0, "processable_assets": 0, "total_assets": total_assets}
                    message = "Asset processing success rate: No processable assets"

            elif metric_type == QualityMetricType.CONSENT_COVERAGE:
                # Get consent-related metrics
                consent_granted = collector.get_counter("consent_granted_files", labels={"vault_id": summary.vault_id})
                files_processed = summary.total_events_processed

                if files_processed > 0:
                    value = consent_granted / files_processed
                    details = {
                        "consent_granted_files": consent_granted,
                        "total_files_processed": files_processed
                    }
                    message = f"Consent coverage: {value*100:.1f}% ({consent_granted}/{files_processed})"
                else:
                    value = 1.0  # If no files processed, consider it 100% coverage
                    details = {"consent_granted_files": 0, "total_files_processed": 0}
                    message = "Consent coverage: No files processed"

            else:
                # Fallback for unknown metrics
                value = 0.0
                message = f"Unknown metric type: {metric_type.value}"
                details = {"error": "Unknown metric type"}
                logger.warning(f"Unknown quality metric type: {metric_type.value}")

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
            details=details
        )

    def _determine_overall_status(self, metric_results: List[QualityMetricResult]) -> QualityGateStatus:
        """Determine overall quality gate status from individual metric results."""
        has_failures = any(result.status == QualityGateStatus.FAIL for result in metric_results)
        has_warnings = any(result.status == QualityGateStatus.WARN for result in metric_results)

        if has_failures:
            return QualityGateStatus.FAIL
        elif has_warnings:
            if self.config.enable_strict_mode:
                return QualityGateStatus.FAIL
            else:
                return QualityGateStatus.WARN
        else:
            return QualityGateStatus.PASS

    def _populate_aggregate_statistics(
        self,
        result: QualityGateResult,
        summary: SyncMetricsSummary,
        collector: SyncMetricsCollector
    ) -> None:
        """Populate aggregate statistics in the quality gate result."""
        result.total_files_processed = summary.total_events_processed
        result.total_files_failed = summary.failed_events
        result.average_processing_time = summary.average_event_processing_time
        result.throughput_events_per_second = summary.events_per_second

        # Get additional statistics from metrics collector
        try:
            # Asset processing statistics
            asset_success_count = collector.get_counter("asset_processing_success", labels={"vault_id": result.vault_id})
            asset_failure_count = collector.get_counter("asset_processing_failure", labels={"vault_id": result.vault_id})
            result.total_assets_processed = asset_success_count + asset_failure_count
            result.total_assets_failed = asset_failure_count

            # Privacy and consent statistics
            result.consent_granted_files = collector.get_counter("consent_granted", labels={"vault_id": result.vault_id}) or 0
            result.redactions_applied = collector.get_counter("redactions_applied", labels={"vault_id": result.vault_id}) or 0

        except Exception as e:
            logger.debug(f"Failed to get additional statistics: {e}")

    def _generate_insights(self, result: QualityGateResult) -> None:
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
                if metric_result.status in [QualityGateStatus.FAIL, QualityGateStatus.WARN]:
                    recommendation = self._get_metric_recommendation(metric_result)
                    if recommendation:
                        result.recommendations.append(recommendation)

        # Add general recommendations
        if result.total_files_failed > 0:
            result.recommendations.append(
                "Review quarantined files and address common parsing errors"
            )

        if result.throughput_events_per_second < 1.0:
            result.recommendations.append(
                "Consider optimizing vault structure or increasing processing resources"
            )

    def _get_metric_recommendation(self, metric_result: QualityMetricResult) -> Optional[str]:
        """Get actionable recommendation for a specific metric issue."""
        metric_type = metric_result.metric_type

        recommendations = {
            QualityMetricType.ERROR_RATE: "Investigate error patterns in audit logs and address common failure causes",
            QualityMetricType.PARSE_FAILURE_RATE: "Review markdown syntax and fix malformed documents",
            QualityMetricType.BROKEN_LINK_RATE: "Update broken wikilinks and fix missing file references",
            QualityMetricType.THROUGHPUT_MINIMUM: "Optimize vault organization or increase processing resources",
            QualityMetricType.PROCESSING_TIME_MAXIMUM: "Review large files and complex documents for optimization opportunities",
            QualityMetricType.CONSENT_COVERAGE: "Grant necessary consents for file processing",
            QualityMetricType.ASSET_PROCESSING_SUCCESS_RATE: "Check asset file formats and processing configurations",
            QualityMetricType.QUARANTINE_RATE: "Address root causes of file processing failures"
        }

        return recommendations.get(metric_type)


def create_quality_gate_evaluator(
    config: Optional[QualityGateConfig] = None,
    metrics_collector: Optional[SyncMetricsCollector] = None
) -> QualityGateEvaluator:
    """Factory function to create a quality gate evaluator with sensible defaults."""
    return QualityGateEvaluator(
        config=config or QualityGateConfig(),
        metrics_collector=metrics_collector
    )