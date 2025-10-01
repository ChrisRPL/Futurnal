"""Comprehensive tests for Obsidian ingestion report generation system."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from futurnal.ingestion.obsidian.report_generator import (
    IngestionTotals,
    IngestionWarnings,
    IngestionPrivacyReport,
    IngestionPerformance,
    IngestionReport,
    ReportGenerator,
    JSONReportFormatter,
    MarkdownReportFormatter,
    create_report_generator,
    create_json_formatter,
    create_markdown_formatter
)
from futurnal.ingestion.obsidian.quality_gate import (
    QualityGateResult,
    QualityGateStatus,
    QualityGateConfig,
    QualityMetricResult,
    QualityMetricType,
    QualityThreshold
)
from futurnal.ingestion.obsidian.sync_metrics import SyncMetricsSummary
from futurnal.privacy.redaction import build_policy


class TestIngestionTotals:
    """Test ingestion totals data structure."""

    def test_default_totals(self):
        """Test default totals initialization."""
        totals = IngestionTotals()

        assert totals.notes_scanned == 0
        assert totals.notes_ingested == 0
        assert totals.notes_updated == 0
        assert totals.notes_failed == 0
        assert totals.assets_discovered == 0
        assert totals.assets_processed == 0
        assert totals.assets_deduped == 0
        assert totals.assets_failed == 0
        assert totals.edges_created == 0
        assert totals.wikilinks_resolved == 0
        assert totals.wikilinks_broken == 0
        assert totals.tags_extracted == 0

    def test_custom_totals(self):
        """Test custom totals values."""
        totals = IngestionTotals(
            notes_scanned=100,
            notes_ingested=95,
            notes_failed=5,
            assets_processed=50,
            wikilinks_resolved=200,
            wikilinks_broken=10
        )

        assert totals.notes_scanned == 100
        assert totals.notes_ingested == 95
        assert totals.notes_failed == 5
        assert totals.assets_processed == 50
        assert totals.wikilinks_resolved == 200
        assert totals.wikilinks_broken == 10


class TestIngestionWarnings:
    """Test ingestion warnings data structure."""

    def test_empty_warnings(self):
        """Test empty warnings state."""
        warnings = IngestionWarnings()

        assert warnings.total_warnings == 0
        assert warnings.has_warnings is False
        assert len(warnings.missing_references) == 0
        assert len(warnings.parse_failures) == 0

    def test_warnings_with_data(self):
        """Test warnings with sample data."""
        warnings = IngestionWarnings(
            missing_references=[
                {"file": "note1.md", "reference": "missing_note.md"},
                {"file": "note2.md", "reference": "another_missing.md"}
            ],
            parse_failures=[
                {"file": "broken.md", "error": "Invalid syntax"}
            ],
            broken_links=[
                {"file": "note3.md", "link": "[[broken]]"}
            ]
        )

        assert warnings.total_warnings == 4
        assert warnings.has_warnings is True
        assert len(warnings.missing_references) == 2
        assert len(warnings.parse_failures) == 1
        assert len(warnings.broken_links) == 1

    def test_total_warnings_calculation(self):
        """Test total warnings calculation across all categories."""
        warnings = IngestionWarnings(
            missing_references=[{"ref": "1"}, {"ref": "2"}],
            unresolved_embeds=[{"embed": "1"}],
            parse_failures=[{"fail": "1"}],
            broken_links=[{"link": "1"}, {"link": "2"}, {"link": "3"}],
            asset_processing_failures=[{"asset": "1"}],
            consent_issues=[{"consent": "1"}, {"consent": "2"}]
        )

        # 2 + 1 + 1 + 3 + 1 + 2 = 10
        assert warnings.total_warnings == 10


class TestIngestionPrivacyReport:
    """Test ingestion privacy report data structure."""

    def test_default_privacy_report(self):
        """Test default privacy report values."""
        privacy = IngestionPrivacyReport()

        assert privacy.redactions_applied == 0
        assert privacy.consent_granted_files == 0
        assert privacy.consent_denied_files == 0
        assert privacy.consent_pending_files == 0
        assert privacy.privacy_policy_violations == 0
        assert len(privacy.consent_scopes) == 0
        assert len(privacy.redaction_types) == 0

    def test_consent_coverage_rate_calculation(self):
        """Test consent coverage rate calculation."""
        # Test with no files
        privacy_empty = IngestionPrivacyReport()
        assert privacy_empty.consent_coverage_rate == 1.0

        # Test with some files
        privacy = IngestionPrivacyReport(
            consent_granted_files=80,
            consent_denied_files=15,
            consent_pending_files=5
        )
        assert privacy.consent_coverage_rate == 0.8  # 80/100

    def test_consent_status_summary(self):
        """Test consent status summary generation."""
        # Test with no files
        privacy_empty = IngestionPrivacyReport()
        assert privacy_empty.consent_status_summary == "No consent requirements"

        # Test with files
        privacy = IngestionPrivacyReport(
            consent_granted_files=90,
            consent_denied_files=10,
            consent_pending_files=0
        )
        summary = privacy.consent_status_summary
        assert "90.0%" in summary
        assert "(90/100 files)" in summary


class TestIngestionPerformance:
    """Test ingestion performance data structure."""

    def test_default_performance(self):
        """Test default performance values."""
        perf = IngestionPerformance()

        assert perf.total_processing_time_seconds == 0.0
        assert perf.average_file_processing_time_seconds == 0.0
        assert perf.throughput_events_per_second == 0.0
        assert perf.queue_depth_max == 0
        assert perf.watchdog_events == 0

    def test_performance_summary(self):
        """Test performance summary generation."""
        perf = IngestionPerformance(
            throughput_events_per_second=2.5,
            average_file_processing_time_seconds=1.8
        )

        summary = perf.performance_summary
        assert "2.5 events/sec" in summary
        assert "1.80s per file" in summary


class TestIngestionReport:
    """Test comprehensive ingestion report."""

    def create_sample_quality_gate_result(self, status=QualityGateStatus.PASS):
        """Create a sample quality gate result for testing."""
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=100,
            failed_events=5
        )

        return QualityGateResult(
            vault_id="test_vault",
            status=status,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[],
            summary_metrics=summary
        )

    def test_default_report(self):
        """Test default report initialization."""
        report = IngestionReport(vault_id="test_vault")

        assert report.vault_id == "test_vault"
        assert report.vault_name is None
        assert report.evaluation_period_hours == 1
        assert report.quality_gate_result is None
        assert isinstance(report.totals, IngestionTotals)
        assert isinstance(report.warnings, IngestionWarnings)
        assert isinstance(report.privacy, IngestionPrivacyReport)
        assert isinstance(report.performance, IngestionPerformance)

    def test_overall_status_from_quality_gate(self):
        """Test overall status from quality gate result."""
        quality_result = self.create_sample_quality_gate_result(QualityGateStatus.WARN)
        report = IngestionReport(
            vault_id="test_vault",
            quality_gate_result=quality_result
        )

        assert report.overall_status == QualityGateStatus.WARN

    def test_overall_status_without_quality_gate(self):
        """Test overall status without quality gate result."""
        report = IngestionReport(vault_id="test_vault")
        assert report.overall_status == QualityGateStatus.PASS

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Test with no processed files
        report_empty = IngestionReport(vault_id="test_vault")
        assert report_empty.success_rate == 1.0

        # Test with processed files
        report = IngestionReport(vault_id="test_vault")
        report.totals.notes_scanned = 100
        report.totals.assets_discovered = 50
        report.totals.notes_failed = 5
        report.totals.assets_failed = 5

        # (100 + 50 - 5 - 5) / (100 + 50) = 140/150 = 0.933...
        assert abs(report.success_rate - 0.9333333333333333) < 0.0001

    def test_exit_code_from_quality_gate(self):
        """Test exit code from quality gate result."""
        quality_result = self.create_sample_quality_gate_result(QualityGateStatus.WARN)
        report = IngestionReport(
            vault_id="test_vault",
            quality_gate_result=quality_result
        )

        assert report.exit_code == quality_result.get_exit_code()

    def test_exit_code_fallback(self):
        """Test exit code fallback logic without quality gate."""
        # Test success case
        report_success = IngestionReport(vault_id="test_vault")
        assert report_success.exit_code == 0

        # Test warning case (with warnings)
        report_warn = IngestionReport(vault_id="test_vault")
        report_warn.warnings.parse_failures = [{"error": "test"}]
        assert report_warn.exit_code == 1


class TestReportGenerator:
    """Test report generator functionality."""

    def test_generator_initialization(self):
        """Test report generator initialization."""
        generator = ReportGenerator()
        assert generator.redaction_policy is not None

        # Test with custom redaction policy
        custom_policy = build_policy()
        generator_custom = ReportGenerator(redaction_policy=custom_policy)
        assert generator_custom.redaction_policy == custom_policy

    def test_generate_report_basic(self):
        """Test basic report generation."""
        generator = ReportGenerator()
        report = generator.generate_report(
            vault_id="test_vault",
            vault_name="Test Vault"
        )

        assert report.vault_id == "test_vault"
        assert report.vault_name == "Test Vault"
        assert isinstance(report.totals, IngestionTotals)
        assert isinstance(report.warnings, IngestionWarnings)
        assert isinstance(report.privacy, IngestionPrivacyReport)
        assert isinstance(report.performance, IngestionPerformance)

    def test_generate_report_with_quality_gate(self):
        """Test report generation with quality gate result."""
        generator = ReportGenerator()
        quality_result = self.create_sample_quality_gate_result()

        report = generator.generate_report(
            vault_id="test_vault",
            quality_gate_result=quality_result
        )

        assert report.quality_gate_result == quality_result
        assert report.overall_status == quality_result.status

    def test_populate_from_quality_gate(self):
        """Test populating report from quality gate result."""
        generator = ReportGenerator()

        # Create quality gate result with sample data
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=150,
            failed_events=8,
            events_per_second=3.2,
            average_event_processing_time=2.1,
            average_batch_processing_time=5.5,
            average_batch_size=10.0
        )

        quality_result = QualityGateResult(
            vault_id="test_vault",
            status=QualityGateStatus.PASS,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[],
            summary_metrics=summary,
            total_assets_processed=75,
            total_assets_failed=3,
            consent_granted_files=140,
            redactions_applied=25,
            critical_issues=["Critical issue 1"],
            warnings=["Warning 1", "Warning 2"]
        )

        report = IngestionReport(vault_id="test_vault")
        generator._populate_from_quality_gate(report, quality_result)

        # Check totals
        assert report.totals.notes_scanned == 150
        assert report.totals.notes_failed == 8
        assert report.totals.notes_ingested == 142
        assert report.totals.assets_processed == 75
        assert report.totals.assets_failed == 3

        # Check performance
        assert report.performance.throughput_events_per_second == 3.2
        assert report.performance.average_file_processing_time_seconds == 2.1
        assert report.performance.batch_processing_time_average == 5.5

        # Check privacy
        assert report.privacy.consent_granted_files == 140
        assert report.privacy.redactions_applied == 25

        # Check warnings
        assert len(report.warnings.parse_failures) == 1
        assert len(report.warnings.missing_references) == 2

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        generator = ReportGenerator()
        report = IngestionReport(vault_id="test_vault")

        # Set up conditions that should trigger recommendations
        report.performance.throughput_events_per_second = 0.5  # Low throughput
        report.performance.average_file_processing_time_seconds = 8.0  # High processing time
        report.totals.notes_failed = 5  # Failed notes
        report.totals.assets_failed = 3  # Failed assets
        report.privacy.consent_granted_files = 80
        report.privacy.consent_denied_files = 20  # Low consent coverage
        report.warnings.parse_failures = [{"error": "test"}]  # Has warnings

        generator._generate_recommendations(report)

        # Should generate multiple recommendations
        assert len(report.recommendations) > 0

        # Check for specific recommendation types
        recommendation_text = " ".join(report.recommendations).lower()
        assert "throughput" in recommendation_text or "processing" in recommendation_text
        assert "failed" in recommendation_text or "notes" in recommendation_text

    def create_sample_quality_gate_result(self, status=QualityGateStatus.PASS):
        """Helper method to create sample quality gate result."""
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=100,
            failed_events=5
        )

        return QualityGateResult(
            vault_id="test_vault",
            status=status,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[],
            summary_metrics=summary
        )


class TestJSONReportFormatter:
    """Test JSON report formatter."""

    def test_formatter_initialization(self):
        """Test JSON formatter initialization."""
        formatter = JSONReportFormatter()
        assert formatter.redaction_policy is not None

    def test_format_report_json_structure(self):
        """Test JSON report formatting structure."""
        formatter = JSONReportFormatter()
        report = IngestionReport(
            vault_id="test_vault",
            vault_name="Test Vault"
        )

        json_str = formatter.format_report(report)
        data = json.loads(json_str)

        assert data["vault_id"] == "test_vault"
        assert data["vault_name"] == "Test Vault"
        assert "overall_status" in data
        assert "success_rate" in data
        assert "exit_code" in data
        assert "vault_info" in data
        assert "totals" in data
        assert "warnings" in data
        assert "privacy" in data
        assert "performance" in data

    def test_format_report_with_quality_gate(self):
        """Test JSON formatting with quality gate result."""
        formatter = JSONReportFormatter()

        # Create quality gate result
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow()
        )
        quality_result = QualityGateResult(
            vault_id="test_vault",
            status=QualityGateStatus.WARN,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[],
            summary_metrics=summary
        )

        report = IngestionReport(
            vault_id="test_vault",
            quality_gate_result=quality_result
        )

        json_str = formatter.format_report(report)
        data = json.loads(json_str)

        assert "quality_gate_summary" in data
        assert data["quality_gate_summary"]["status"] == "warn"
        assert data["quality_gate_summary"]["has_warnings"] is False
        assert data["quality_gate_summary"]["has_failures"] is False

    def test_write_report_to_file(self):
        """Test writing JSON report to file."""
        formatter = JSONReportFormatter()
        report = IngestionReport(vault_id="test_vault")

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.json"
            formatter.write_report(report, output_path)

            assert output_path.exists()

            # Verify the file contains valid JSON
            with open(output_path, 'r') as f:
                data = json.load(f)
                assert data["vault_id"] == "test_vault"

    def test_json_serializer(self):
        """Test custom JSON serializer."""
        formatter = JSONReportFormatter()

        # Test datetime serialization
        dt = datetime(2023, 1, 1, 12, 0, 0)
        assert formatter._json_serializer(dt) == "2023-01-01T12:00:00"

        # Test Path serialization
        path = Path("/test/path")
        assert formatter._json_serializer(path) == "/test/path"

        # Test enum serialization (mock object with value attribute)
        class MockEnum:
            value = "test_value"

        enum_obj = MockEnum()
        assert formatter._json_serializer(enum_obj) == "test_value"

        # Test fallback string conversion
        assert formatter._json_serializer(123) == "123"


class TestMarkdownReportFormatter:
    """Test Markdown report formatter."""

    def test_formatter_initialization(self):
        """Test Markdown formatter initialization."""
        formatter = MarkdownReportFormatter()
        assert formatter.redaction_policy is not None

    def test_format_report_structure(self):
        """Test Markdown report formatting structure."""
        formatter = MarkdownReportFormatter()
        report = IngestionReport(
            vault_id="test_vault",
            vault_name="Test Vault"
        )

        markdown = formatter.format_report(report)

        # Check for key sections
        assert "# ✅ Obsidian Ingestion Report" in markdown
        assert "**Vault:** Test Vault" in markdown
        assert "**Status:** PASS" in markdown
        assert "## Summary" in markdown
        assert "## Ingestion Totals" in markdown
        assert "## Privacy & Consent" in markdown
        assert "## Performance Metrics" in markdown

    def test_format_report_with_warnings(self):
        """Test Markdown formatting with warnings."""
        formatter = MarkdownReportFormatter()
        report = IngestionReport(vault_id="test_vault")

        # Add warnings
        report.warnings.parse_failures = [
            {"description": "Parse error 1", "severity": "warning"},
            {"description": "Parse error 2", "severity": "warning"}
        ]
        report.warnings.broken_links = [
            {"description": "Broken link", "severity": "warning"}
        ]

        markdown = formatter.format_report(report)

        assert "## Warnings" in markdown
        assert "### Parse Failures (2)" in markdown
        assert "### Broken Links (1)" in markdown
        assert "Parse error 1" in markdown
        assert "Broken link" in markdown

    def test_format_report_with_quality_gate(self):
        """Test Markdown formatting with quality gate results."""
        formatter = MarkdownReportFormatter()

        # Create quality gate result with metrics
        threshold = QualityThreshold(
            metric_type=QualityMetricType.ERROR_RATE,
            warn_threshold=0.05,
            fail_threshold=0.10,
            description="Error rate threshold",
            unit="%"
        )

        metric_result = QualityMetricResult(
            metric_type=QualityMetricType.ERROR_RATE,
            status=QualityGateStatus.WARN,
            value=0.06,
            threshold=threshold,
            message="Warning: error rate exceeded"
        )

        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow()
        )

        quality_result = QualityGateResult(
            vault_id="test_vault",
            status=QualityGateStatus.WARN,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[metric_result],
            summary_metrics=summary
        )

        report = IngestionReport(
            vault_id="test_vault",
            quality_gate_result=quality_result
        )

        markdown = formatter.format_report(report)

        assert "## Quality Gate Results" in markdown
        assert "Error rate threshold" in markdown
        assert "⚠️ warn" in markdown

    def test_format_report_with_recommendations(self):
        """Test Markdown formatting with recommendations."""
        formatter = MarkdownReportFormatter()
        report = IngestionReport(vault_id="test_vault")
        report.recommendations = [
            "Improve vault organization",
            "Address parsing errors",
            "Optimize processing performance"
        ]

        markdown = formatter.format_report(report)

        assert "## Recommendations" in markdown
        assert "1. Improve vault organization" in markdown
        assert "2. Address parsing errors" in markdown
        assert "3. Optimize processing performance" in markdown

    def test_write_report_to_file(self):
        """Test writing Markdown report to file."""
        formatter = MarkdownReportFormatter()
        report = IngestionReport(vault_id="test_vault", vault_name="Test Vault")

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.md"
            formatter.write_report(report, output_path)

            assert output_path.exists()

            # Verify the file contains expected content
            content = output_path.read_text()
            assert "# ✅ Obsidian Ingestion Report" in content
            assert "**Vault:** Test Vault" in content

    def test_format_header_different_statuses(self):
        """Test header formatting with different status values."""
        formatter = MarkdownReportFormatter()

        # Test PASS status
        report_pass = IngestionReport(vault_id="test_vault")
        header_pass = formatter._format_header(report_pass)
        assert "# ✅ Obsidian Ingestion Report" in header_pass

        # Test WARN status (mock quality gate result)
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow()
        )
        quality_result = QualityGateResult(
            vault_id="test_vault",
            status=QualityGateStatus.WARN,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[],
            summary_metrics=summary
        )

        report_warn = IngestionReport(vault_id="test_vault", quality_gate_result=quality_result)
        header_warn = formatter._format_header(report_warn)
        assert "# ⚠️ Obsidian Ingestion Report" in header_warn

        # Test FAIL status
        quality_result.status = QualityGateStatus.FAIL
        report_fail = IngestionReport(vault_id="test_vault", quality_gate_result=quality_result)
        header_fail = formatter._format_header(report_fail)
        assert "# ❌ Obsidian Ingestion Report" in header_fail


class TestReportGeneratorFactories:
    """Test report generator factory functions."""

    def test_create_report_generator(self):
        """Test report generator factory."""
        generator = create_report_generator()
        assert isinstance(generator, ReportGenerator)
        assert generator.redaction_policy is not None

        # Test with custom policy
        custom_policy = build_policy()
        generator_custom = create_report_generator(redaction_policy=custom_policy)
        assert generator_custom.redaction_policy == custom_policy

    def test_create_json_formatter(self):
        """Test JSON formatter factory."""
        formatter = create_json_formatter()
        assert isinstance(formatter, JSONReportFormatter)
        assert formatter.redaction_policy is not None

    def test_create_markdown_formatter(self):
        """Test Markdown formatter factory."""
        formatter = create_markdown_formatter()
        assert isinstance(formatter, MarkdownReportFormatter)
        assert formatter.redaction_policy is not None


class TestReportGeneratorIntegration:
    """Integration tests for report generation system."""

    def test_complete_report_generation_pipeline(self):
        """Test complete report generation pipeline."""
        # Create components
        generator = create_report_generator()
        json_formatter = create_json_formatter()
        md_formatter = create_markdown_formatter()

        # Create quality gate result
        config = QualityGateConfig()
        summary = SyncMetricsSummary(
            vault_id="integration_test_vault",
            time_period_start=datetime.utcnow() - timedelta(hours=1),
            time_period_end=datetime.utcnow(),
            total_events_processed=250,
            failed_events=12,
            events_per_second=5.2,
            average_event_processing_time=1.8
        )

        quality_result = QualityGateResult(
            vault_id="integration_test_vault",
            status=QualityGateStatus.WARN,
            evaluated_at=datetime.utcnow(),
            config=config,
            metric_results=[],
            summary_metrics=summary,
            total_assets_processed=120,
            total_assets_failed=8,
            consent_granted_files=230,
            redactions_applied=45
        )

        # Generate report
        report = generator.generate_report(
            vault_id="integration_test_vault",
            vault_name="Integration Test Vault",
            quality_gate_result=quality_result
        )

        # Verify report content
        assert report.vault_id == "integration_test_vault"
        assert report.vault_name == "Integration Test Vault"
        assert report.overall_status == QualityGateStatus.WARN
        assert report.totals.notes_scanned == 250
        assert report.totals.notes_failed == 12

        # Test JSON formatting
        json_output = json_formatter.format_report(report)
        json_data = json.loads(json_output)
        assert json_data["vault_id"] == "integration_test_vault"
        assert json_data["overall_status"] == "warn"

        # Test Markdown formatting
        md_output = md_formatter.format_report(report)
        assert "# ⚠️ Obsidian Ingestion Report" in md_output
        assert "Integration Test Vault" in md_output
        assert "## Summary" in md_output

        # Test file output
        with TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "report.json"
            md_path = Path(temp_dir) / "report.md"

            json_formatter.write_report(report, json_path)
            md_formatter.write_report(report, md_path)

            assert json_path.exists()
            assert md_path.exists()

            # Verify file contents
            json_content = json.loads(json_path.read_text())
            assert json_content["vault_id"] == "integration_test_vault"

            md_content = md_path.read_text()
            assert "Integration Test Vault" in md_content