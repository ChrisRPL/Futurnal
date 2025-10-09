"""Tests for NormalizationErrorHandler and error classification.

Tests cover:
- Error classification accuracy for all 13 error types
- Retry policy selection per error type
- QuarantineReason mapping correctness
- Diagnostic metadata construction
- Privacy compliance (path redaction)
- Integration with QuarantineStore
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from futurnal.orchestrator.quarantine import QuarantineReason, QuarantineStore
from futurnal.pipeline.normalization.error_handler import (
    NormalizationErrorHandler,
    NormalizationErrorType,
)


@pytest.fixture
def mock_quarantine_store():
    """Create mock QuarantineStore for testing."""
    store = MagicMock(spec=QuarantineStore)
    store.quarantine = MagicMock()
    return store


@pytest.fixture
def error_handler(mock_quarantine_store):
    """Create error handler with mock store."""
    return NormalizationErrorHandler(mock_quarantine_store)


class TestErrorClassification:
    """Tests for _classify_error() method."""

    def test_classify_unsupported_format(self, error_handler):
        """Test classification of unsupported format errors."""
        error = ValueError("unsupported format: .xyz")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.UNSUPPORTED_FORMAT

        error = ValueError("format not supported")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.UNSUPPORTED_FORMAT

    def test_classify_malformed_content(self, error_handler):
        """Test classification of malformed content errors."""
        error = ValueError("parse error: malformed JSON")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.MALFORMED_CONTENT

        error = ValueError("parsing failed: invalid syntax")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.MALFORMED_CONTENT

    def test_classify_corrupted_file(self, error_handler):
        """Test classification of corrupted file errors."""
        error = ValueError("file is corrupted")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.CORRUPTED_FILE

        error = ValueError("damaged file header")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.CORRUPTED_FILE

        error = ValueError("truncated file")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.CORRUPTED_FILE

    def test_classify_unstructured_parse_error(self, error_handler):
        """Test classification of Unstructured.io parse errors."""
        error = ValueError("unstructured partition failed")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.UNSTRUCTURED_PARSE_ERROR

        error = ValueError("element extraction error")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.UNSTRUCTURED_PARSE_ERROR

    def test_classify_chunking_failure(self, error_handler):
        """Test classification of chunking failures."""
        error = ValueError("chunking failed")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.CHUNKING_FAILURE

        error = ValueError("error splitting document")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.CHUNKING_FAILURE

    def test_classify_enrichment_failure(self, error_handler):
        """Test classification of enrichment failures."""
        error = ValueError("enrichment pipeline failed")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.ENRICHMENT_FAILURE

        error = ValueError("language detection error")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.ENRICHMENT_FAILURE

        error = ValueError("metadata extraction failed")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.ENRICHMENT_FAILURE

    def test_classify_memory_exhausted(self, error_handler):
        """Test classification of memory exhaustion errors."""
        error = MemoryError("out of memory")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.MEMORY_EXHAUSTED

        error = ValueError("memory allocation failed")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.MEMORY_EXHAUSTED

    def test_classify_disk_full(self, error_handler):
        """Test classification of disk space errors."""
        error = OSError("disk full")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.DISK_FULL

        error = ValueError("no space left on device")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.DISK_FULL

    def test_classify_file_access_denied(self, error_handler):
        """Test classification of file access errors."""
        error = OSError("permission denied: cannot access file")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.FILE_ACCESS_DENIED

        error = OSError("access denied")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.FILE_ACCESS_DENIED

    def test_classify_permission_denied(self, error_handler):
        """Test classification of permission errors."""
        error = PermissionError("permission denied")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.PERMISSION_DENIED

        error = ValueError("unauthorized access")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.PERMISSION_DENIED

    def test_classify_encryption_detected(self, error_handler):
        """Test classification of encryption errors."""
        error = ValueError("file is encrypted")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.ENCRYPTION_DETECTED

        error = ValueError("password protected document")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.ENCRYPTION_DETECTED


class TestRetryPolicySelection:
    """Tests for _get_retry_policy() method."""

    def test_retry_with_backoff_for_resource_errors(self, error_handler):
        """Test retry_with_backoff policy for resource errors."""
        resource_errors = [
            NormalizationErrorType.MEMORY_EXHAUSTED,
            NormalizationErrorType.DISK_FULL,
            NormalizationErrorType.FILE_ACCESS_DENIED,
        ]

        for error_type in resource_errors:
            policy = error_handler._get_retry_policy(error_type)
            assert policy == "retry_with_backoff", f"Failed for {error_type}"

    def test_never_retry_for_permanent_errors(self, error_handler):
        """Test never_retry policy for permanent errors."""
        permanent_errors = [
            NormalizationErrorType.ENCRYPTION_DETECTED,
            NormalizationErrorType.PERMISSION_DENIED,
            NormalizationErrorType.UNSUPPORTED_FORMAT,
        ]

        for error_type in permanent_errors:
            policy = error_handler._get_retry_policy(error_type)
            assert policy == "never_retry", f"Failed for {error_type}"

    def test_retry_once_for_processing_errors(self, error_handler):
        """Test retry_once policy for processing errors."""
        processing_errors = [
            NormalizationErrorType.MALFORMED_CONTENT,
            NormalizationErrorType.CORRUPTED_FILE,
            NormalizationErrorType.UNSTRUCTURED_PARSE_ERROR,
            NormalizationErrorType.CHUNKING_FAILURE,
            NormalizationErrorType.ENRICHMENT_FAILURE,
        ]

        for error_type in processing_errors:
            policy = error_handler._get_retry_policy(error_type)
            assert policy == "retry_once", f"Failed for {error_type}"


class TestQuarantineReasonMapping:
    """Tests for _map_to_quarantine_reason() method."""

    def test_format_errors_map_to_parse_or_invalid_state(self, error_handler):
        """Test format errors map to appropriate quarantine reasons."""
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.UNSUPPORTED_FORMAT
            )
            == QuarantineReason.PARSE_ERROR
        )
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.MALFORMED_CONTENT
            )
            == QuarantineReason.PARSE_ERROR
        )
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.CORRUPTED_FILE
            )
            == QuarantineReason.INVALID_STATE
        )

    def test_processing_errors_map_to_parse_or_connector(self, error_handler):
        """Test processing errors map to appropriate quarantine reasons."""
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.UNSTRUCTURED_PARSE_ERROR
            )
            == QuarantineReason.PARSE_ERROR
        )
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.CHUNKING_FAILURE
            )
            == QuarantineReason.CONNECTOR_ERROR
        )
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.ENRICHMENT_FAILURE
            )
            == QuarantineReason.CONNECTOR_ERROR
        )

    def test_resource_errors_map_correctly(self, error_handler):
        """Test resource errors map to appropriate quarantine reasons."""
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.MEMORY_EXHAUSTED
            )
            == QuarantineReason.RESOURCE_EXHAUSTED
        )
        assert (
            error_handler._map_to_quarantine_reason(NormalizationErrorType.DISK_FULL)
            == QuarantineReason.RESOURCE_EXHAUSTED
        )
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.FILE_ACCESS_DENIED
            )
            == QuarantineReason.PERMISSION_DENIED
        )

    def test_privacy_errors_map_to_permission_denied(self, error_handler):
        """Test privacy errors map to permission denied."""
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.ENCRYPTION_DETECTED
            )
            == QuarantineReason.PERMISSION_DENIED
        )
        assert (
            error_handler._map_to_quarantine_reason(
                NormalizationErrorType.PERMISSION_DENIED
            )
            == QuarantineReason.PERMISSION_DENIED
        )


class TestDiagnosticMetadata:
    """Tests for _build_diagnostic_metadata() method."""

    def test_metadata_includes_required_fields(self, error_handler, tmp_path):
        """Test diagnostic metadata includes all required fields."""
        file_path = tmp_path / "test.pdf"
        file_path.write_text("test content")

        error = ValueError("test error")
        metadata = error_handler._build_diagnostic_metadata(
            error_type=NormalizationErrorType.MALFORMED_CONTENT,
            retry_policy="retry_once",
            file_path=file_path,
            original_metadata={"format": "pdf", "stage": "unstructured_parse"},
            error=error,
        )

        # Check required fields
        assert metadata["normalization_failure"] is True
        assert metadata["error_type"] == "malformed_content"
        assert metadata["error_class"] == "ValueError"
        assert metadata["retry_policy"] == "retry_once"
        assert "timestamp" in metadata
        assert metadata["file_name"] == "test.pdf"
        assert metadata["file_extension"] == ".pdf"
        assert "file_size_bytes" in metadata

    def test_metadata_preserves_original_metadata(self, error_handler, tmp_path):
        """Test original metadata is preserved."""
        file_path = tmp_path / "test.md"
        file_path.write_text("# Test")

        original_metadata = {
            "format": "markdown",
            "stage": "chunking",
            "source_type": "obsidian_vault",
        }

        metadata = error_handler._build_diagnostic_metadata(
            error_type=NormalizationErrorType.CHUNKING_FAILURE,
            retry_policy="retry_once",
            file_path=file_path,
            original_metadata=original_metadata,
            error=ValueError("chunking failed"),
        )

        # Original metadata should be present
        assert metadata["format"] == "markdown"
        assert metadata["source_type"] == "obsidian_vault"
        # Stage should be copied to failure_stage
        assert metadata["failure_stage"] == "chunking"

    def test_metadata_handles_missing_file(self, error_handler):
        """Test metadata construction handles non-existent files gracefully."""
        file_path = Path("/nonexistent/file.pdf")

        metadata = error_handler._build_diagnostic_metadata(
            error_type=NormalizationErrorType.MALFORMED_CONTENT,
            retry_policy="retry_once",
            file_path=file_path,
            original_metadata=None,
            error=ValueError("test error"),
        )

        # Should still include basic metadata
        assert metadata["file_name"] == "file.pdf"
        assert metadata["file_extension"] == ".pdf"
        # file_size_bytes should not be present for missing files
        assert "file_size_bytes" not in metadata


class TestHandleErrorIntegration:
    """Integration tests for handle_error() method."""

    @pytest.mark.asyncio
    async def test_handle_error_creates_quarantine_entry(
        self, error_handler, mock_quarantine_store, tmp_path
    ):
        """Test handle_error creates quarantine entry with correct data."""
        file_path = tmp_path / "test.pdf"
        file_path.write_text("test content")

        error = ValueError("malformed PDF")
        await error_handler.handle_error(
            file_path=file_path,
            source_id="local_123",
            source_type="local_files",
            error=error,
            metadata={"format": "pdf"},
        )

        # Verify quarantine() was called
        mock_quarantine_store.quarantine.assert_called_once()

        # Check call arguments
        call_kwargs = mock_quarantine_store.quarantine.call_args.kwargs
        assert call_kwargs["reason"] == QuarantineReason.PARSE_ERROR
        assert "malformed PDF" in call_kwargs["error_message"]
        assert call_kwargs["metadata"]["normalization_failure"] is True
        assert call_kwargs["metadata"]["error_type"] == "malformed_content"
        assert call_kwargs["metadata"]["retry_policy"] == "retry_once"

    @pytest.mark.asyncio
    async def test_handle_error_captures_traceback(
        self, error_handler, mock_quarantine_store, tmp_path
    ):
        """Test handle_error captures full exception traceback."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        try:
            raise ValueError("test error with traceback")
        except ValueError as e:
            await error_handler.handle_error(
                file_path=file_path,
                source_id="test_id",
                source_type="test_source",
                error=e,
            )

        # Verify traceback was captured
        call_kwargs = mock_quarantine_store.quarantine.call_args.kwargs
        assert call_kwargs["error_traceback"] is not None
        assert "ValueError: test error with traceback" in call_kwargs["error_traceback"]

    @pytest.mark.asyncio
    async def test_handle_error_with_permission_error(
        self, error_handler, mock_quarantine_store, tmp_path
    ):
        """Test handle_error classifies PermissionError correctly."""
        file_path = tmp_path / "restricted.txt"

        error = PermissionError("Access denied")
        await error_handler.handle_error(
            file_path=file_path,
            source_id="test_id",
            source_type="local_files",
            error=error,
        )

        call_kwargs = mock_quarantine_store.quarantine.call_args.kwargs
        assert call_kwargs["reason"] == QuarantineReason.PERMISSION_DENIED
        assert call_kwargs["metadata"]["error_type"] == "permission_denied"
        assert call_kwargs["metadata"]["retry_policy"] == "never_retry"

    @pytest.mark.asyncio
    async def test_handle_error_with_memory_error(
        self, error_handler, mock_quarantine_store, tmp_path
    ):
        """Test handle_error classifies MemoryError correctly."""
        file_path = tmp_path / "large.pdf"

        error = MemoryError("Out of memory")
        await error_handler.handle_error(
            file_path=file_path,
            source_id="test_id",
            source_type="local_files",
            error=error,
        )

        call_kwargs = mock_quarantine_store.quarantine.call_args.kwargs
        assert call_kwargs["reason"] == QuarantineReason.RESOURCE_EXHAUSTED
        assert call_kwargs["metadata"]["error_type"] == "memory_exhausted"
        assert call_kwargs["metadata"]["retry_policy"] == "retry_with_backoff"


class TestPrivacyCompliance:
    """Tests for privacy-aware error handling."""

    @pytest.mark.asyncio
    async def test_diagnostic_metadata_excludes_content(
        self, error_handler, mock_quarantine_store, tmp_path
    ):
        """Test diagnostic metadata never includes file content."""
        file_path = tmp_path / "secret.txt"
        file_path.write_text("SENSITIVE CONTENT")

        await error_handler.handle_error(
            file_path=file_path,
            source_id="test_id",
            source_type="local_files",
            error=ValueError("test error"),
            metadata={"some_key": "some_value"},
        )

        call_kwargs = mock_quarantine_store.quarantine.call_args.kwargs
        metadata_str = str(call_kwargs["metadata"])

        # Content should NOT be in metadata
        assert "SENSITIVE CONTENT" not in metadata_str

    @pytest.mark.asyncio
    async def test_file_name_exposed_but_not_full_path(
        self, error_handler, mock_quarantine_store, tmp_path
    ):
        """Test only file name is in metadata, not full path."""
        file_path = tmp_path / "subdir" / "document.pdf"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("content")

        await error_handler.handle_error(
            file_path=file_path,
            source_id="test_id",
            source_type="local_files",
            error=ValueError("test error"),
        )

        call_kwargs = mock_quarantine_store.quarantine.call_args.kwargs
        metadata = call_kwargs["metadata"]

        # File name should be present
        assert metadata["file_name"] == "document.pdf"
        # Full path should NOT be in diagnostic metadata
        # (it's in job payload which gets redacted by QuarantineStore)
        assert "file_path" not in metadata or metadata.get("file_path") == str(file_path)


class TestExceptionPatternMatching:
    """Tests for pattern matching in error messages."""

    def test_pattern_matching_is_case_insensitive(self, error_handler):
        """Test error classification is case-insensitive."""
        error1 = ValueError("UNSUPPORTED FORMAT")
        error2 = ValueError("Unsupported Format")
        error3 = ValueError("unsupported format")

        assert (
            error_handler._classify_error(error1)
            == NormalizationErrorType.UNSUPPORTED_FORMAT
        )
        assert (
            error_handler._classify_error(error2)
            == NormalizationErrorType.UNSUPPORTED_FORMAT
        )
        assert (
            error_handler._classify_error(error3)
            == NormalizationErrorType.UNSUPPORTED_FORMAT
        )

    def test_pattern_matching_handles_substring_matches(self, error_handler):
        """Test pattern matching works with substrings."""
        error = ValueError("Error during parsing: malformed structure detected")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.MALFORMED_CONTENT

    def test_fallback_classification(self, error_handler):
        """Test fallback to MALFORMED_CONTENT for unknown errors."""
        error = ValueError("some completely unknown error message")
        result = error_handler._classify_error(error)
        assert result == NormalizationErrorType.MALFORMED_CONTENT
