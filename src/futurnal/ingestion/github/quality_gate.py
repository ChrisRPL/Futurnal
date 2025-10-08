"""Quality gate evaluation for GitHub connector.

Implements automated quality checks according to the production readiness
requirements defined in the quality gates documentation.

Quality Gates:
- Test coverage >90%
- Failure rate <0.5%
- Performance targets met
- Security compliance 100%
- No credential leakage
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Quality Gate Models
# ---------------------------------------------------------------------------


class QualityGateStatus(str, Enum):
    """Quality gate evaluation status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class GitHubQualityMetrics:
    """Metrics collection for GitHub connector quality evaluation."""

    # Test metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0

    # Coverage metrics
    coverage_percentage: float = 0.0
    lines_covered: int = 0
    lines_total: int = 0

    # Performance metrics
    small_repo_sync_time: Optional[float] = None  # Target: <10s
    medium_repo_sync_time: Optional[float] = None  # Target: <60s
    incremental_sync_time: Optional[float] = None  # Target: <5s
    api_requests_per_sync: Optional[int] = None  # Target: <100 for medium repo
    memory_peak_mb: Optional[float] = None  # Target: <500MB

    # Security metrics
    security_tests_passed: int = 0
    security_tests_total: int = 0
    credential_leaks_found: int = 0

    # Integration metrics
    integration_tests_passed: int = 0
    integration_tests_total: int = 0

    # Load test metrics
    rate_limit_violations: int = 0
    circuit_breaker_triggers: int = 0

    # Operational metrics
    sync_attempts: int = 0
    sync_successes: int = 0
    sync_failures: int = 0

    def failure_rate(self) -> float:
        """Calculate sync failure rate."""
        if self.sync_attempts == 0:
            return 0.0
        return (self.sync_failures / self.sync_attempts) * 100

    def success_rate(self) -> float:
        """Calculate sync success rate."""
        return 100.0 - self.failure_rate()

    def security_pass_rate(self) -> float:
        """Calculate security test pass rate."""
        if self.security_tests_total == 0:
            return 0.0
        return (self.security_tests_passed / self.security_tests_total) * 100


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation."""

    status: QualityGateStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: GitHubQualityMetrics = field(default_factory=GitHubQualityMetrics)
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def get_exit_code(self) -> int:
        """Get exit code for CI/CD integration."""
        if self.status == QualityGateStatus.FAIL:
            return 1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "metrics": {
                "test_coverage": self.metrics.coverage_percentage,
                "failure_rate": self.metrics.failure_rate(),
                "security_pass_rate": self.metrics.security_pass_rate(),
                "total_tests": self.metrics.total_tests,
                "passed_tests": self.metrics.passed_tests,
                "failed_tests": self.metrics.failed_tests,
            },
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def save_to_file(self, file_path: Path):
        """Save result to JSON file."""
        file_path.write_text(json.dumps(self.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Quality Gate Evaluator
# ---------------------------------------------------------------------------


class GitHubQualityGateEvaluator:
    """Evaluates quality gates for GitHub connector."""

    # Quality gate thresholds
    COVERAGE_THRESHOLD = 90.0  # >90% coverage required
    FAILURE_RATE_THRESHOLD = 0.5  # <0.5% failure rate required
    SECURITY_PASS_RATE_THRESHOLD = 100.0  # 100% security tests must pass

    # Performance targets (from quality gates doc)
    SMALL_REPO_SYNC_TARGET = 10.0  # seconds
    MEDIUM_REPO_SYNC_TARGET = 60.0  # seconds
    INCREMENTAL_SYNC_TARGET = 5.0  # seconds
    API_REQUESTS_TARGET = 100  # requests for medium repo
    MEMORY_TARGET_MB = 500.0  # MB

    def __init__(self, metrics: Optional[GitHubQualityMetrics] = None):
        """Initialize evaluator with metrics."""
        self.metrics = metrics or GitHubQualityMetrics()

    def evaluate(self) -> QualityGateResult:
        """Evaluate all quality gates.

        Returns:
            QualityGateResult with overall status and detailed findings
        """
        result = QualityGateResult(
            status=QualityGateStatus.PASS,
            metrics=self.metrics,
        )

        # Evaluate each gate
        self._evaluate_coverage(result)
        self._evaluate_failure_rate(result)
        self._evaluate_security(result)
        self._evaluate_performance(result)
        self._evaluate_integration(result)

        # Determine overall status
        if result.critical_issues:
            result.status = QualityGateStatus.FAIL
        elif result.warnings:
            result.status = QualityGateStatus.WARNING

        return result

    def _evaluate_coverage(self, result: QualityGateResult):
        """Evaluate test coverage gate."""
        coverage = self.metrics.coverage_percentage

        if coverage < self.COVERAGE_THRESHOLD:
            result.critical_issues.append(
                f"Test coverage {coverage:.1f}% is below threshold {self.COVERAGE_THRESHOLD}%"
            )
        elif coverage < self.COVERAGE_THRESHOLD + 2:
            result.warnings.append(
                f"Test coverage {coverage:.1f}% is close to threshold {self.COVERAGE_THRESHOLD}%"
            )
        else:
            result.recommendations.append(
                f"Test coverage {coverage:.1f}% exceeds target - excellent!"
            )

    def _evaluate_failure_rate(self, result: QualityGateResult):
        """Evaluate failure rate gate."""
        failure_rate = self.metrics.failure_rate()

        if failure_rate > self.FAILURE_RATE_THRESHOLD:
            result.critical_issues.append(
                f"Failure rate {failure_rate:.2f}% exceeds threshold {self.FAILURE_RATE_THRESHOLD}%"
            )
        elif failure_rate > self.FAILURE_RATE_THRESHOLD * 0.8:
            result.warnings.append(
                f"Failure rate {failure_rate:.2f}% is approaching threshold"
            )

    def _evaluate_security(self, result: QualityGateResult):
        """Evaluate security compliance gate."""
        # Credential leakage check
        if self.metrics.credential_leaks_found > 0:
            result.critical_issues.append(
                f"CRITICAL: {self.metrics.credential_leaks_found} credential leaks detected!"
            )

        # Security test pass rate
        security_pass_rate = self.metrics.security_pass_rate()
        if security_pass_rate < self.SECURITY_PASS_RATE_THRESHOLD:
            result.critical_issues.append(
                f"Security test pass rate {security_pass_rate:.1f}% below required {self.SECURITY_PASS_RATE_THRESHOLD}%"
            )

    def _evaluate_performance(self, result: QualityGateResult):
        """Evaluate performance benchmarks gate."""
        # Small repo sync time
        if self.metrics.small_repo_sync_time is not None:
            if self.metrics.small_repo_sync_time > self.SMALL_REPO_SYNC_TARGET:
                result.warnings.append(
                    f"Small repo sync time {self.metrics.small_repo_sync_time:.1f}s "
                    f"exceeds target {self.SMALL_REPO_SYNC_TARGET}s"
                )

        # Medium repo sync time
        if self.metrics.medium_repo_sync_time is not None:
            if self.metrics.medium_repo_sync_time > self.MEDIUM_REPO_SYNC_TARGET:
                result.warnings.append(
                    f"Medium repo sync time {self.metrics.medium_repo_sync_time:.1f}s "
                    f"exceeds target {self.MEDIUM_REPO_SYNC_TARGET}s"
                )

        # Incremental sync time
        if self.metrics.incremental_sync_time is not None:
            if self.metrics.incremental_sync_time > self.INCREMENTAL_SYNC_TARGET:
                result.warnings.append(
                    f"Incremental sync time {self.metrics.incremental_sync_time:.1f}s "
                    f"exceeds target {self.INCREMENTAL_SYNC_TARGET}s"
                )

        # API request efficiency
        if self.metrics.api_requests_per_sync is not None:
            if self.metrics.api_requests_per_sync > self.API_REQUESTS_TARGET:
                result.warnings.append(
                    f"API requests {self.metrics.api_requests_per_sync} "
                    f"exceeds target {self.API_REQUESTS_TARGET}"
                )

        # Memory usage
        if self.metrics.memory_peak_mb is not None:
            if self.metrics.memory_peak_mb > self.MEMORY_TARGET_MB:
                result.critical_issues.append(
                    f"Memory usage {self.metrics.memory_peak_mb:.1f}MB "
                    f"exceeds target {self.MEMORY_TARGET_MB}MB"
                )

    def _evaluate_integration(self, result: QualityGateResult):
        """Evaluate integration tests gate."""
        if self.metrics.integration_tests_total > 0:
            pass_rate = (
                self.metrics.integration_tests_passed
                / self.metrics.integration_tests_total
                * 100
            )

            if pass_rate < 100:
                result.critical_issues.append(
                    f"Integration test pass rate {pass_rate:.1f}% - all must pass"
                )

        # Rate limit violations
        if self.metrics.rate_limit_violations > 0:
            result.critical_issues.append(
                f"Rate limit violations detected: {self.metrics.rate_limit_violations}"
            )


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_quality_gate_evaluator(
    metrics: Optional[GitHubQualityMetrics] = None,
) -> GitHubQualityGateEvaluator:
    """Create quality gate evaluator instance.

    Args:
        metrics: Optional pre-populated metrics

    Returns:
        Configured evaluator instance
    """
    return GitHubQualityGateEvaluator(metrics=metrics)


def evaluate_from_pytest_results(pytest_json_path: Path) -> QualityGateResult:
    """Evaluate quality gates from pytest JSON results.

    Args:
        pytest_json_path: Path to pytest --json-report output

    Returns:
        Quality gate evaluation result
    """
    # Load pytest results
    with open(pytest_json_path) as f:
        pytest_data = json.load(f)

    # Extract metrics
    metrics = GitHubQualityMetrics()

    # Test results
    summary = pytest_data.get("summary", {})
    metrics.total_tests = summary.get("total", 0)
    metrics.passed_tests = summary.get("passed", 0)
    metrics.failed_tests = summary.get("failed", 0)
    metrics.skipped_tests = summary.get("skipped", 0)

    # Coverage (if available)
    if "coverage" in pytest_data:
        metrics.coverage_percentage = pytest_data["coverage"].get("percent_covered", 0.0)

    # Create evaluator and evaluate
    evaluator = create_quality_gate_evaluator(metrics)
    return evaluator.evaluate()


def print_quality_gate_report(result: QualityGateResult):
    """Print quality gate report to console.

    Args:
        result: Quality gate evaluation result
    """
    print("=" * 80)
    print("GitHub Connector Quality Gate Evaluation")
    print("=" * 80)
    print(f"\nStatus: {result.status.value.upper()}")
    print(f"Timestamp: {result.timestamp.isoformat()}")

    print("\n" + "=" * 80)
    print("Metrics Summary")
    print("=" * 80)
    print(f"Test Coverage: {result.metrics.coverage_percentage:.1f}%")
    print(f"Total Tests: {result.metrics.total_tests}")
    print(f"Passed Tests: {result.metrics.passed_tests}")
    print(f"Failed Tests: {result.metrics.failed_tests}")
    print(f"Failure Rate: {result.metrics.failure_rate():.2f}%")
    print(f"Security Pass Rate: {result.metrics.security_pass_rate():.1f}%")

    if result.critical_issues:
        print("\n" + "=" * 80)
        print("CRITICAL ISSUES")
        print("=" * 80)
        for issue in result.critical_issues:
            print(f"  ❌ {issue}")

    if result.warnings:
        print("\n" + "=" * 80)
        print("WARNINGS")
        print("=" * 80)
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")

    if result.recommendations:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        for rec in result.recommendations:
            print(f"  ✅ {rec}")

    print("\n" + "=" * 80)
    print(f"Exit Code: {result.get_exit_code()}")
    print("=" * 80)
