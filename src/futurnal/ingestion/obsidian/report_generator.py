"""Comprehensive ingestion reporting system for Obsidian vaults with privacy-aware output formatting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from futurnal.privacy.redaction import RedactionPolicy, build_policy

from .quality_gate import QualityGateResult, QualityGateStatus, QualityMetricResult
from .sync_metrics import SyncMetricsSummary

logger = logging.getLogger(__name__)


@dataclass
class IngestionTotals:
    """Aggregated totals from ingestion process."""
    notes_scanned: int = 0
    notes_ingested: int = 0
    notes_updated: int = 0
    notes_failed: int = 0

    assets_discovered: int = 0
    assets_processed: int = 0
    assets_deduped: int = 0
    assets_failed: int = 0

    edges_created: int = 0
    edges_updated: int = 0

    wikilinks_resolved: int = 0
    wikilinks_broken: int = 0

    tags_extracted: int = 0
    callouts_processed: int = 0


@dataclass
class IngestionWarnings:
    """Collection of warnings and issues found during ingestion."""
    missing_references: List[Dict[str, str]] = field(default_factory=list)
    unresolved_embeds: List[Dict[str, str]] = field(default_factory=list)
    parse_failures: List[Dict[str, str]] = field(default_factory=list)
    broken_links: List[Dict[str, str]] = field(default_factory=list)
    asset_processing_failures: List[Dict[str, str]] = field(default_factory=list)
    consent_issues: List[Dict[str, str]] = field(default_factory=list)

    @property
    def total_warnings(self) -> int:
        """Total number of warnings across all categories."""
        return (
            len(self.missing_references) +
            len(self.unresolved_embeds) +
            len(self.parse_failures) +
            len(self.broken_links) +
            len(self.asset_processing_failures) +
            len(self.consent_issues)
        )

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return self.total_warnings > 0


@dataclass
class IngestionPrivacyReport:
    """Privacy-related statistics and compliance information."""
    redactions_applied: int = 0
    consent_granted_files: int = 0
    consent_denied_files: int = 0
    consent_pending_files: int = 0
    privacy_policy_violations: int = 0

    consent_scopes: Dict[str, int] = field(default_factory=dict)
    redaction_types: Dict[str, int] = field(default_factory=dict)

    @property
    def consent_coverage_rate(self) -> float:
        """Calculate consent coverage rate."""
        total_files = self.consent_granted_files + self.consent_denied_files + self.consent_pending_files
        if total_files == 0:
            return 1.0
        return self.consent_granted_files / total_files

    @property
    def consent_status_summary(self) -> str:
        """Get human-readable consent status summary."""
        total = self.consent_granted_files + self.consent_denied_files + self.consent_pending_files
        if total == 0:
            return "No consent requirements"

        granted_pct = (self.consent_granted_files / total) * 100
        return f"{granted_pct:.1f}% consent coverage ({self.consent_granted_files}/{total} files)"


@dataclass
class IngestionPerformance:
    """Performance metrics and statistics."""
    total_processing_time_seconds: float = 0.0
    average_file_processing_time_seconds: float = 0.0
    throughput_events_per_second: float = 0.0
    throughput_files_per_minute: float = 0.0

    queue_latency_average_seconds: float = 0.0
    queue_latency_p95_seconds: float = 0.0
    queue_depth_max: int = 0
    queue_depth_average: float = 0.0

    batch_processing_time_average: float = 0.0
    batch_size_average: float = 0.0

    watchdog_events: int = 0
    timeout_events: int = 0
    retry_events: int = 0

    @property
    def performance_summary(self) -> str:
        """Get human-readable performance summary."""
        return (
            f"Processed {self.throughput_events_per_second:.1f} events/sec, "
            f"avg {self.average_file_processing_time_seconds:.2f}s per file"
        )


@dataclass
class IngestionReport:
    """Comprehensive ingestion report combining quality gate results with detailed metrics."""

    # Report metadata
    vault_id: str
    vault_name: Optional[str] = None
    report_generated_at: datetime = field(default_factory=datetime.utcnow)
    evaluation_period_hours: int = 1

    # Quality gate results
    quality_gate_result: Optional[QualityGateResult] = None

    # Detailed metrics
    totals: IngestionTotals = field(default_factory=IngestionTotals)
    warnings: IngestionWarnings = field(default_factory=IngestionWarnings)
    privacy: IngestionPrivacyReport = field(default_factory=IngestionPrivacyReport)
    performance: IngestionPerformance = field(default_factory=IngestionPerformance)

    # Additional context
    recommendations: List[str] = field(default_factory=list)
    vault_statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_status(self) -> QualityGateStatus:
        """Get overall status from quality gate result."""
        if self.quality_gate_result:
            return self.quality_gate_result.status
        return QualityGateStatus.PASS

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        total_processed = self.totals.notes_scanned + self.totals.assets_discovered
        if total_processed == 0:
            return 1.0

        total_failed = self.totals.notes_failed + self.totals.assets_failed
        return 1.0 - (total_failed / total_processed)

    @property
    def exit_code(self) -> int:
        """Get appropriate exit code for CI/CD integration."""
        if self.quality_gate_result:
            return self.quality_gate_result.get_exit_code()

        # Fallback logic if no quality gate result
        if self.warnings.has_warnings or self.success_rate < 0.9:
            return 1  # Warnings present

        return 0  # Success


class ReportGenerator:
    """Generates comprehensive ingestion reports from quality gate results and metrics."""

    def __init__(self, redaction_policy: Optional[RedactionPolicy] = None):
        self.redaction_policy = redaction_policy or build_policy()

    def generate_report(
        self,
        vault_id: str,
        vault_name: Optional[str] = None,
        quality_gate_result: Optional[QualityGateResult] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> IngestionReport:
        """Generate comprehensive ingestion report."""

        logger.info(f"Generating ingestion report for vault {vault_id}")

        # Create base report
        report = IngestionReport(
            vault_id=vault_id,
            vault_name=vault_name,
            quality_gate_result=quality_gate_result
        )

        # Populate report sections from quality gate result
        if quality_gate_result:
            self._populate_from_quality_gate(report, quality_gate_result)

        # Add any additional metrics
        if additional_metrics:
            report.vault_statistics.update(additional_metrics)

        # Generate recommendations
        self._generate_recommendations(report)

        logger.info(f"Report generated for vault {vault_id}: {report.overall_status.value}")
        return report

    def _populate_from_quality_gate(self, report: IngestionReport, quality_result: QualityGateResult) -> None:
        """Populate report sections from quality gate evaluation result."""
        summary = quality_result.summary_metrics

        # Populate totals
        report.totals.notes_scanned = summary.total_events_processed
        report.totals.notes_failed = summary.failed_events
        report.totals.notes_ingested = summary.total_events_processed - summary.failed_events

        # Populate performance metrics
        report.performance.throughput_events_per_second = summary.events_per_second
        report.performance.average_file_processing_time_seconds = summary.average_event_processing_time
        report.performance.batch_processing_time_average = summary.average_batch_processing_time
        report.performance.batch_size_average = summary.average_batch_size

        # Populate additional statistics from quality gate result
        report.totals.assets_processed = quality_result.total_assets_processed
        report.totals.assets_failed = quality_result.total_assets_failed
        report.totals.assets_discovered = report.totals.assets_processed + report.totals.assets_failed

        # Privacy statistics
        report.privacy.consent_granted_files = quality_result.consent_granted_files
        report.privacy.redactions_applied = quality_result.redactions_applied
        report.privacy.privacy_policy_violations = quality_result.privacy_policy_violations

        # Populate warnings from quality gate critical issues
        for issue in quality_result.critical_issues:
            report.warnings.parse_failures.append({
                "description": issue,
                "severity": "critical"
            })

        for warning in quality_result.warnings:
            report.warnings.missing_references.append({
                "description": warning,
                "severity": "warning"
            })

    def _generate_recommendations(self, report: IngestionReport) -> None:
        """Generate actionable recommendations based on report data."""
        recommendations = []

        # Quality-based recommendations
        if report.quality_gate_result:
            recommendations.extend(report.quality_gate_result.recommendations)

        # Performance-based recommendations
        if report.performance.throughput_events_per_second < 1.0:
            recommendations.append(
                "Consider optimizing vault structure or processing configuration to improve throughput"
            )

        if report.performance.average_file_processing_time_seconds > 5.0:
            recommendations.append(
                "Review large files and complex documents that may be slowing processing"
            )

        # Error-based recommendations
        if report.totals.notes_failed > 0:
            recommendations.append(
                f"Address {report.totals.notes_failed} failed notes by reviewing quarantine entries"
            )

        if report.totals.assets_failed > 0:
            recommendations.append(
                f"Review {report.totals.assets_failed} failed assets for format compatibility"
            )

        # Privacy-based recommendations
        if report.privacy.consent_coverage_rate < 0.95:
            recommendations.append(
                "Improve consent coverage by granting necessary permissions for file processing"
            )

        # Warning-based recommendations
        if report.warnings.has_warnings:
            recommendations.append(
                f"Address {report.warnings.total_warnings} warnings to improve vault quality"
            )

        report.recommendations = recommendations


class JSONReportFormatter:
    """Formats ingestion reports as machine-readable JSON for CI/CD consumption."""

    def __init__(self, redaction_policy: Optional[RedactionPolicy] = None):
        self.redaction_policy = redaction_policy or build_policy()

    def format_report(self, report: IngestionReport) -> str:
        """Format report as JSON string."""
        # Create privacy-safe representation
        report_dict = self._create_safe_dict(report)

        return json.dumps(report_dict, indent=2, default=self._json_serializer)

    def write_report(self, report: IngestionReport, output_path: Path) -> None:
        """Write report to JSON file."""
        json_content = self.format_report(report)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_content, encoding='utf-8')

        logger.info(f"JSON report written to {output_path}")

    def _create_safe_dict(self, report: IngestionReport) -> Dict[str, Any]:
        """Create privacy-safe dictionary representation of report."""
        # Convert to dict using dataclass fields
        report_dict = asdict(report)

        # Add computed properties
        report_dict['overall_status'] = report.overall_status.value
        report_dict['success_rate'] = report.success_rate
        report_dict['exit_code'] = report.exit_code

        # Add privacy-safe vault information
        report_dict['vault_info'] = {
            'vault_id': report.vault_id,
            'vault_name': report.vault_name,
            'evaluation_period_hours': report.evaluation_period_hours
        }

        # Add quality gate summary if available
        if report.quality_gate_result:
            report_dict['quality_gate_summary'] = {
                'status': report.quality_gate_result.status.value,
                'passed': report.quality_gate_result.passed,
                'has_warnings': report.quality_gate_result.has_warnings,
                'has_failures': report.quality_gate_result.has_failures,
                'metric_count': len(report.quality_gate_result.metric_results),
                'evaluated_at': report.quality_gate_result.evaluated_at.isoformat()
            }

        return report_dict

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, 'value'):  # Handle enums
            return obj.value

        return str(obj)


class MarkdownReportFormatter:
    """Formats ingestion reports as human-readable Markdown for operators."""

    def __init__(self, redaction_policy: Optional[RedactionPolicy] = None):
        self.redaction_policy = redaction_policy or build_policy()

    def format_report(self, report: IngestionReport) -> str:
        """Format report as Markdown string."""
        sections = []

        # Header
        sections.append(self._format_header(report))

        # Summary
        sections.append(self._format_summary(report))

        # Totals
        sections.append(self._format_totals(report))

        # Quality Gate Results
        if report.quality_gate_result:
            sections.append(self._format_quality_gate(report))

        # Warnings
        if report.warnings.has_warnings:
            sections.append(self._format_warnings(report))

        # Privacy
        sections.append(self._format_privacy(report))

        # Performance
        sections.append(self._format_performance(report))

        # Recommendations
        if report.recommendations:
            sections.append(self._format_recommendations(report))

        return "\n\n".join(sections)

    def write_report(self, report: IngestionReport, output_path: Path) -> None:
        """Write report to Markdown file."""
        markdown_content = self.format_report(report)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_content, encoding='utf-8')

        logger.info(f"Markdown report written to {output_path}")

    def _format_header(self, report: IngestionReport) -> str:
        """Format report header."""
        status_emoji = {
            QualityGateStatus.PASS: "✅",
            QualityGateStatus.WARN: "⚠️",
            QualityGateStatus.FAIL: "❌"
        }

        emoji = status_emoji.get(report.overall_status, "ℹ️")
        vault_display = report.vault_name or report.vault_id

        return f"""# {emoji} Obsidian Ingestion Report

**Vault:** {vault_display}
**Generated:** {report.report_generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Status:** {report.overall_status.value.upper()}
**Success Rate:** {report.success_rate*100:.1f}%"""

    def _format_summary(self, report: IngestionReport) -> str:
        """Format executive summary."""
        lines = ["## Summary"]

        # Overall status
        if report.overall_status == QualityGateStatus.PASS:
            lines.append("✅ **Quality gate PASSED** - All quality criteria met successfully.")
        elif report.overall_status == QualityGateStatus.WARN:
            lines.append("⚠️ **Quality gate PASSED with warnings** - Some quality thresholds approached.")
        else:
            lines.append("❌ **Quality gate FAILED** - Quality criteria not met, review required.")

        # Key metrics
        lines.extend([
            "",
            f"- **Files processed:** {report.totals.notes_ingested:,} successful, {report.totals.notes_failed:,} failed",
            f"- **Assets processed:** {report.totals.assets_processed:,} successful, {report.totals.assets_failed:,} failed",
            f"- **Performance:** {report.performance.performance_summary}",
            f"- **Privacy:** {report.privacy.consent_status_summary}"
        ])

        if report.warnings.has_warnings:
            lines.append(f"- **Warnings:** {report.warnings.total_warnings:,} issues requiring attention")

        return "\n".join(lines)

    def _format_totals(self, report: IngestionReport) -> str:
        """Format totals section."""
        totals = report.totals

        return f"""## Ingestion Totals

| Metric | Count |
|--------|-------|
| Notes scanned | {totals.notes_scanned:,} |
| Notes ingested | {totals.notes_ingested:,} |
| Notes failed | {totals.notes_failed:,} |
| Assets discovered | {totals.assets_discovered:,} |
| Assets processed | {totals.assets_processed:,} |
| Assets failed | {totals.assets_failed:,} |
| Edges created | {totals.edges_created:,} |
| Wikilinks resolved | {totals.wikilinks_resolved:,} |
| Wikilinks broken | {totals.wikilinks_broken:,} |"""

    def _format_quality_gate(self, report: IngestionReport) -> str:
        """Format quality gate results."""
        if not report.quality_gate_result:
            return ""

        qg = report.quality_gate_result
        lines = ["## Quality Gate Results"]

        # Summary table
        lines.extend([
            "",
            "| Metric | Status | Value | Threshold |",
            "|--------|--------|-------|-----------|"
        ])

        for metric in qg.metric_results:
            status_emoji = {
                QualityGateStatus.PASS: "✅",
                QualityGateStatus.WARN: "⚠️",
                QualityGateStatus.FAIL: "❌"
            }

            emoji = status_emoji.get(metric.status, "ℹ️")
            lines.append(
                f"| {metric.threshold.description} | {emoji} {metric.status.value} | "
                f"{metric.formatted_value} | {metric.threshold.warn_threshold} |"
            )

        return "\n".join(lines)

    def _format_warnings(self, report: IngestionReport) -> str:
        """Format warnings section."""
        warnings = report.warnings
        if not warnings.has_warnings:
            return ""

        lines = [f"## Warnings ({warnings.total_warnings:,} total)"]

        warning_sections = [
            ("Parse Failures", warnings.parse_failures),
            ("Missing References", warnings.missing_references),
            ("Broken Links", warnings.broken_links),
            ("Unresolved Embeds", warnings.unresolved_embeds),
            ("Asset Processing Failures", warnings.asset_processing_failures),
            ("Consent Issues", warnings.consent_issues)
        ]

        for section_name, section_warnings in warning_sections:
            if section_warnings:
                lines.extend([
                    "",
                    f"### {section_name} ({len(section_warnings):,})",
                    ""
                ])

                for warning in section_warnings[:10]:  # Limit to first 10
                    lines.append(f"- {warning.get('description', 'Unknown warning')}")

                if len(section_warnings) > 10:
                    lines.append(f"- ... and {len(section_warnings) - 10:,} more")

        return "\n".join(lines)

    def _format_privacy(self, report: IngestionReport) -> str:
        """Format privacy section."""
        privacy = report.privacy

        return f"""## Privacy & Consent

| Metric | Value |
|--------|-------|
| Consent coverage | {privacy.consent_coverage_rate*100:.1f}% |
| Files with consent | {privacy.consent_granted_files:,} |
| Files without consent | {privacy.consent_denied_files:,} |
| Redactions applied | {privacy.redactions_applied:,} |
| Privacy violations | {privacy.privacy_policy_violations:,} |"""

    def _format_performance(self, report: IngestionReport) -> str:
        """Format performance section."""
        perf = report.performance

        return f"""## Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | {perf.throughput_events_per_second:.1f} events/sec |
| Avg processing time | {perf.average_file_processing_time_seconds:.2f}s |
| Batch processing time | {perf.batch_processing_time_average:.2f}s |
| Queue latency (avg) | {perf.queue_latency_average_seconds:.2f}s |
| Queue depth (max) | {perf.queue_depth_max:,} |
| Timeout events | {perf.timeout_events:,} |
| Retry events | {perf.retry_events:,} |"""

    def _format_recommendations(self, report: IngestionReport) -> str:
        """Format recommendations section."""
        if not report.recommendations:
            return ""

        lines = ["## Recommendations"]

        for i, recommendation in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {recommendation}")

        return "\n".join(lines)


def create_report_generator(redaction_policy: Optional[RedactionPolicy] = None) -> ReportGenerator:
    """Factory function to create a report generator with privacy-aware defaults."""
    return ReportGenerator(redaction_policy=redaction_policy)


def create_json_formatter(redaction_policy: Optional[RedactionPolicy] = None) -> JSONReportFormatter:
    """Factory function to create a JSON report formatter."""
    return JSONReportFormatter(redaction_policy=redaction_policy)


def create_markdown_formatter(redaction_policy: Optional[RedactionPolicy] = None) -> MarkdownReportFormatter:
    """Factory function to create a Markdown report formatter."""
    return MarkdownReportFormatter(redaction_policy=redaction_policy)