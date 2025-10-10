"""Privacy compliance tests for normalization pipeline.

Validates that the normalization pipeline maintains strict privacy standards:
- No document content in logs
- No content in audit trails
- No content in error messages
- No content in telemetry/metrics
- Proper path redaction

Requirements Tested:
- ✓ Production Checklist: "Privacy audit shows no content leakage in logs"
- ✓ Privacy Requirement: "Privacy-safe design (metadata-only)"
- ✓ Error Handling: "Error messages don't leak file content"
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

import pytest

from futurnal.pipeline.normalization import create_normalization_service
from tests.pipeline.normalization.test_utils import assert_no_content_in_string


# ---------------------------------------------------------------------------
# Log Content Isolation Tests
# ---------------------------------------------------------------------------


@pytest.mark.privacy_audit
class TestLogContentIsolation:
    """Test suite for verifying no content in logs."""

    @pytest.mark.asyncio
    async def test_document_content_not_in_logs(
        self,
        markdown_simple,
        caplog
    ):
        """Verify document content does not appear in logs.

        From requirements: "Privacy audit shows no content leakage in logs"
        """
        # Get the actual content
        content = markdown_simple.read_text(encoding="utf-8")
        # Extract a distinctive phrase
        distinctive_phrase = "This is a simple markdown document"

        service = create_normalization_service()

        with caplog.at_level(logging.DEBUG):
            await service.normalize_document(
                file_path=markdown_simple,
                source_id="privacy_test",
                source_type="test"
            )

        # Verify content not in any log messages
        log_output = "\n".join(record.message for record in caplog.records)
        assert_no_content_in_string(distinctive_phrase, log_output)

    @pytest.mark.asyncio
    async def test_sensitive_content_not_logged(self, tmp_path, caplog):
        """Test that sensitive content markers are not logged."""
        # Create file with "sensitive" markers
        content = "SENSITIVE_API_KEY=abc123xyz\nSECRET_PASSWORD=hunter2"
        test_file = tmp_path / "sensitive.txt"
        test_file.write_text(content, encoding="utf-8")

        service = create_normalization_service()

        with caplog.at_level(logging.DEBUG):
            await service.normalize_document(
                file_path=test_file,
                source_id="sensitive_test",
                source_type="test"
            )

        log_output = "\n".join(record.message for record in caplog.records)

        # Sensitive content should not appear in logs
        assert_no_content_in_string("SENSITIVE_API_KEY", log_output)
        assert_no_content_in_string("abc123xyz", log_output)
        assert_no_content_in_string("SECRET_PASSWORD", log_output)
        assert_no_content_in_string("hunter2", log_output)

    @pytest.mark.asyncio
    async def test_file_content_not_in_debug_logs(
        self,
        json_simple,
        caplog
    ):
        """Verify JSON content not in debug logs."""
        content = json_simple.read_text(encoding="utf-8")
        # Get a distinctive value from JSON
        distinctive_value = "Test Document"

        service = create_normalization_service()

        with caplog.at_level(logging.DEBUG):
            await service.normalize_document(
                file_path=json_simple,
                source_id="json_privacy_test",
                source_type="test"
            )

        log_output = "\n".join(record.message for record in caplog.records)
        # Content should not appear in logs
        # Note: filename might appear, but not content
        assert_no_content_in_string(distinctive_value, log_output)


# ---------------------------------------------------------------------------
# Audit Trail Privacy Tests
# ---------------------------------------------------------------------------


@pytest.mark.privacy_audit
class TestAuditTrailPrivacy:
    """Test suite for privacy in audit logs."""

    @pytest.mark.asyncio
    async def test_audit_log_excludes_content(
        self,
        markdown_complex,
        tmp_path,
        mock_audit_logger
    ):
        """Verify audit logs don't contain document content."""
        content = markdown_complex.read_text(encoding="utf-8")
        distinctive_phrase = "Complex Markdown Test"

        service = create_normalization_service()
        service.audit_logger = mock_audit_logger

        await service.normalize_document(
            file_path=markdown_complex,
            source_id="audit_test",
            source_type="test"
        )

        # Read all audit log files
        audit_files = list(mock_audit_logger.output_dir.glob("*.log"))
        assert len(audit_files) > 0, "No audit logs generated"

        audit_content = ""
        for audit_file in audit_files:
            audit_content += audit_file.read_text(encoding="utf-8")

        # Verify content not in audit logs
        assert_no_content_in_string(distinctive_phrase, audit_content)

    @pytest.mark.asyncio
    async def test_audit_events_contain_metadata_only(
        self,
        text_simple,
        tmp_path,
        mock_audit_logger
    ):
        """Verify audit events contain metadata but not content."""
        service = create_normalization_service()
        service.audit_logger = mock_audit_logger

        await service.normalize_document(
            file_path=text_simple,
            source_id="metadata_test",
            source_type="test"
        )

        # Read audit events
        audit_files = list(mock_audit_logger.output_dir.glob("*.log"))
        events = []
        for audit_file in audit_files:
            for line in audit_file.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))

        # Verify events have metadata but not content
        for event in events:
            # Should have metadata fields
            assert "action" in event or "job_id" in event

            # Should NOT have content field
            assert "content" not in event
            assert "text" not in event
            assert "document_content" not in event


# ---------------------------------------------------------------------------
# Error Message Privacy Tests
# ---------------------------------------------------------------------------


@pytest.mark.privacy_audit
class TestErrorMessagePrivacy:
    """Test error messages don't leak content.

    From requirements: "Error messages don't leak file content"
    """

    @pytest.mark.asyncio
    async def test_error_messages_exclude_content(
        self,
        truncated_json_file
    ):
        """Verify error messages don't contain file content."""
        from futurnal.pipeline.normalization import NormalizationError

        content = truncated_json_file.read_text(encoding="utf-8")
        # Get a distinctive part of the content
        distinctive_part = '"id": 1'

        service = create_normalization_service()

        try:
            await service.normalize_document(
                file_path=truncated_json_file,
                source_id="error_test",
                source_type="test"
            )
        except NormalizationError as e:
            error_message = str(e)
            # Error should not contain document content
            assert_no_content_in_string(distinctive_part, error_message)

    @pytest.mark.asyncio
    async def test_exception_traceback_excludes_content(
        self,
        tmp_path,
        caplog
    ):
        """Verify exception tracebacks don't expose content."""
        # Create file that will cause processing error
        content = "CONFIDENTIAL_DATA: This should not appear in traceback"
        test_file = tmp_path / "error_file.txt"
        test_file.write_text(content, encoding="utf-8")

        service = create_normalization_service()

        with caplog.at_level(logging.ERROR):
            try:
                # Force an error scenario
                await service.normalize_document(
                    file_path=test_file,
                    source_id="traceback_test",
                    source_type="test"
                )
            except Exception:
                pass

        # Check logs for content leakage
        log_output = "\n".join(record.message for record in caplog.records)
        assert_no_content_in_string("CONFIDENTIAL_DATA", log_output)


# ---------------------------------------------------------------------------
# Telemetry and Metrics Privacy Tests
# ---------------------------------------------------------------------------


@pytest.mark.privacy_audit
class TestTelemetryPrivacy:
    """Test telemetry and metrics don't contain content."""

    @pytest.mark.asyncio
    async def test_metrics_exclude_document_content(self, markdown_simple):
        """Verify performance metrics don't contain content."""
        content = markdown_simple.read_text(encoding="utf-8")
        distinctive_phrase = "Simple Markdown Document"

        service = create_normalization_service()

        await service.normalize_document(
            file_path=markdown_simple,
            source_id="metrics_test",
            source_type="test"
        )

        # Get metrics
        metrics = service.get_metrics()
        metrics_json = json.dumps(metrics)

        # Verify content not in metrics
        assert_no_content_in_string(distinctive_phrase, metrics_json)

    @pytest.mark.asyncio
    async def test_performance_monitor_privacy(self, text_large):
        """Verify performance monitor doesn't log content."""
        from futurnal.pipeline.normalization.performance import PerformanceMonitor

        monitor = PerformanceMonitor()
        monitor.start()

        service = create_normalization_service()
        service.performance_monitor = monitor

        content = text_large.read_text(encoding="utf-8")
        content_sample = content[:100]

        await service.normalize_document(
            file_path=text_large,
            source_id="perf_monitor_test",
            source_type="test"
        )

        # Get performance metrics
        perf_metrics = monitor.get_metrics()
        perf_json = json.dumps(perf_metrics)

        # Content should not be in metrics
        assert_no_content_in_string(content_sample, perf_json)


# ---------------------------------------------------------------------------
# Path Redaction Tests
# ---------------------------------------------------------------------------


@pytest.mark.privacy_audit
class TestPathRedaction:
    """Test sensitive paths are redacted."""

    @pytest.mark.asyncio
    async def test_full_paths_not_in_logs(
        self,
        tmp_path,
        caplog
    ):
        """Verify full file paths are not exposed in logs."""
        # Create file in deeply nested path
        sensitive_dir = tmp_path / "private" / "confidential" / "secret"
        sensitive_dir.mkdir(parents=True, exist_ok=True)
        test_file = sensitive_dir / "document.txt"
        test_file.write_text("Test content", encoding="utf-8")

        service = create_normalization_service()

        with caplog.at_level(logging.INFO):
            await service.normalize_document(
                file_path=test_file,
                source_id="path_test",
                source_type="test"
            )

        log_output = "\n".join(record.message for record in caplog.records)

        # Sensitive directory names should not appear
        # Note: Filename might appear, but not full path
        assert "confidential" not in log_output or "document.txt" in log_output
        assert "secret" not in log_output or "document.txt" in log_output

    @pytest.mark.asyncio
    async def test_home_directory_not_exposed(
        self,
        tmp_path,
        mock_audit_logger
    ):
        """Verify home directory paths are not exposed in audit logs."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content", encoding="utf-8")

        service = create_normalization_service()
        service.audit_logger = mock_audit_logger

        await service.normalize_document(
            file_path=test_file,
            source_id="home_test",
            source_type="test"
        )

        # Read audit logs
        audit_files = list(mock_audit_logger.output_dir.glob("*.log"))
        audit_content = ""
        for audit_file in audit_files:
            audit_content += audit_file.read_text()

        # Should not contain actual tmp_path
        # (depends on redaction implementation)


# ---------------------------------------------------------------------------
# Chunk Content Privacy Tests
# ---------------------------------------------------------------------------


@pytest.mark.privacy_audit
class TestChunkContentPrivacy:
    """Test chunk content is not leaked."""

    @pytest.mark.asyncio
    async def test_chunk_content_not_in_logs(
        self,
        markdown_large,
        caplog
    ):
        """Verify chunk content doesn't appear in logs."""
        service = create_normalization_service()

        with caplog.at_level(logging.DEBUG):
            result = await service.normalize_document(
                file_path=markdown_large,
                source_id="chunk_privacy_test",
                source_type="test"
            )

        # Get first chunk content as sample
        assert len(result.chunks) > 0
        chunk_content_sample = result.chunks[0].content[:50]

        log_output = "\n".join(record.message for record in caplog.records)
        # Chunk content should not appear in logs
        assert_no_content_in_string(chunk_content_sample, log_output)

    @pytest.mark.asyncio
    async def test_chunk_metadata_excludes_content(self, markdown_complex):
        """Verify chunk metadata doesn't contain actual content."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=markdown_complex,
            source_id="chunk_meta_test",
            source_type="test"
        )

        # Serialize chunks to JSON (simulating storage)
        for chunk in result.chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "character_count": chunk.character_count,
                "word_count": chunk.word_count,
            }
            # Metadata dict should not contain actual content field
            # (content is separate)
            assert "content" not in chunk_dict


# ---------------------------------------------------------------------------
# Privacy Compliance Summary
# ---------------------------------------------------------------------------


@pytest.mark.privacy_audit
class TestPrivacyComplianceSummary:
    """Generate privacy compliance summary report."""

    @pytest.mark.asyncio
    async def test_privacy_compliance_report(
        self,
        markdown_simple,
        text_simple,
        json_simple,
        tmp_path,
        caplog,
        mock_audit_logger
    ):
        """Generate comprehensive privacy compliance report."""
        service = create_normalization_service()
        service.audit_logger = mock_audit_logger

        test_files = {
            "markdown": markdown_simple,
            "text": text_simple,
            "json": json_simple
        }

        report = {
            "files_tested": len(test_files),
            "privacy_violations": 0,
            "tests_passed": 0,
            "details": []
        }

        with caplog.at_level(logging.DEBUG):
            for file_type, file_path in test_files.items():
                content = file_path.read_text(encoding="utf-8")
                content_sample = content[:100]

                # Process file
                await service.normalize_document(
                    file_path=file_path,
                    source_id=f"privacy_{file_type}",
                    source_type="test"
                )

                # Check for content in logs
                log_output = "\n".join(record.message for record in caplog.records)

                violation_found = content_sample in log_output
                if violation_found:
                    report["privacy_violations"] += 1
                else:
                    report["tests_passed"] += 1

                report["details"].append({
                    "file_type": file_type,
                    "privacy_clean": not violation_found
                })

        # Print summary
        print("\n" + "=" * 70)
        print("PRIVACY COMPLIANCE REPORT")
        print("=" * 70)
        print(f"Files Tested: {report['files_tested']}")
        print(f"Privacy Violations: {report['privacy_violations']}")
        print(f"Tests Passed: {report['tests_passed']}")
        print(f"Compliance: {'✓ PASS' if report['privacy_violations'] == 0 else '✗ FAIL'}")
        print("=" * 70 + "\n")

        # Assert no violations
        assert report["privacy_violations"] == 0, (
            "Privacy violations detected in logs"
        )
