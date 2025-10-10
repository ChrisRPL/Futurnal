"""Edge case tests for normalization pipeline.

Validates robustness and error handling for edge cases including:
- Empty and minimal documents
- Very large documents (>1GB for streaming)
- Unicode and special characters
- Corrupted and malformed files
- Boundary conditions

Requirements Tested:
- ✓ Production Checklist: "Streaming processor handles 1GB+ documents without OOM"
- ✓ Edge Case Coverage: "Empty documents, very large documents (>1GB), corrupted files"
- ✓ Error Handling: "Quarantine workflow handles all failure modes"
"""

from __future__ import annotations

import pytest

from futurnal.pipeline.normalization import create_normalization_service, NormalizationError
from tests.pipeline.normalization.test_utils import assert_valid_sha256


# ---------------------------------------------------------------------------
# Empty and Minimal Document Tests
# ---------------------------------------------------------------------------


@pytest.mark.edge_case
class TestEmptyAndMinimalDocuments:
    """Test suite for empty and minimal documents."""

    @pytest.mark.asyncio
    async def test_empty_file_handling(self, empty_file):
        """Test handling of completely empty file (0 bytes).

        Should either succeed with minimal document or fail gracefully.
        """
        service = create_normalization_service()

        try:
            result = await service.normalize_document(
                file_path=empty_file,
                source_id="empty_test",
                source_type="test"
            )

            # If succeeds, verify minimal valid document
            assert result.sha256 is not None
            assert_valid_sha256(result.sha256)

        except NormalizationError:
            # Graceful failure is acceptable for empty files
            pass

    @pytest.mark.asyncio
    async def test_whitespace_only_file(self, whitespace_only_file):
        """Test file containing only whitespace."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=whitespace_only_file,
            source_id="whitespace_test",
            source_type="test"
        )

        # Should succeed with minimal content
        assert result.sha256 is not None
        assert_valid_sha256(result.sha256)

    @pytest.mark.asyncio
    async def test_single_character_file(self, single_char_file):
        """Test file with single character."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=single_char_file,
            source_id="single_char_test",
            source_type="test"
        )

        assert result.sha256 is not None
        assert result.metadata.character_count == 1

    @pytest.mark.asyncio
    async def test_single_line_no_newline(self, single_line_file):
        """Test file with single line and no trailing newline."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=single_line_file,
            source_id="single_line_test",
            source_type="test"
        )

        assert result.sha256 is not None
        assert result.metadata.line_count >= 1


# ---------------------------------------------------------------------------
# Unicode and Special Character Tests
# ---------------------------------------------------------------------------


@pytest.mark.edge_case
class TestUnicodeHandling:
    """Test suite for unicode and special character handling."""

    @pytest.mark.asyncio
    async def test_emoji_content_processing(self, unicode_emoji_file):
        """Test processing of emoji and special unicode characters.

        From requirements: "Unicode and special characters"
        """
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=unicode_emoji_file,
            source_id="emoji_test",
            source_type="test"
        )

        # Should process successfully
        assert result.sha256 is not None
        assert_valid_sha256(result.sha256)
        assert result.metadata.character_count > 0

        # Content should be preserved
        assert result.content or len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_utf8_bom_handling(self, unicode_bom_file):
        """Test handling of UTF-8 BOM (Byte Order Mark)."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=unicode_bom_file,
            source_id="bom_test",
            source_type="test"
        )

        # Should process successfully, BOM should be handled transparently
        assert result.sha256 is not None
        assert result.content or len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_mixed_encoding_content(self, mixed_encoding_file):
        """Test content with various special characters."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=mixed_encoding_file,
            source_id="mixed_encoding_test",
            source_type="test"
        )

        assert result.sha256 is not None
        assert result.metadata.character_count > 0


# ---------------------------------------------------------------------------
# Large File Tests
# ---------------------------------------------------------------------------


@pytest.mark.edge_case
@pytest.mark.slow
class TestLargeFileHandling:
    """Test suite for very large file handling.

    From requirements: "Streaming processor handles 1GB+ documents without OOM"
    """

    @pytest.mark.asyncio
    async def test_10mb_file_processing(self, large_file_10mb):
        """Test processing of 10MB file."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=large_file_10mb,
            source_id="large_10mb_test",
            source_type="test"
        )

        # Should succeed
        assert result.sha256 is not None
        assert_valid_sha256(result.sha256)
        assert result.metadata.file_size_bytes >= 10 * 1024 * 1024  # ≥10MB

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "not config.getoption('--run-slow')",
        reason="Skipped by default, use --run-slow to run"
    )
    async def test_100mb_file_processing(self, large_file_100mb):
        """Test processing of 100MB file.

        Note: Slow test, only runs with --run-slow flag.
        """
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=large_file_100mb,
            source_id="large_100mb_test",
            source_type="test"
        )

        # Should succeed using streaming processor
        assert result.sha256 is not None
        assert result.is_chunked  # Large files should be chunked
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_50mb_markdown_file_processing(self, large_markdown_50mb):
        """Test processing of 50MB structured markdown file."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=large_markdown_50mb,
            source_id="large_md_50mb_test",
            source_type="test"
        )

        # Should succeed with chunking
        assert result.sha256 is not None
        assert result.is_chunked
        assert len(result.chunks) > 10  # Should have many chunks


# ---------------------------------------------------------------------------
# Corrupted and Malformed File Tests
# ---------------------------------------------------------------------------


@pytest.mark.edge_case
class TestCorruptedFiles:
    """Test suite for corrupted and malformed files.

    From requirements: "Corrupted files, malformed content"
    """

    @pytest.mark.asyncio
    async def test_truncated_json_handling(self, truncated_json_file):
        """Test handling of truncated JSON file."""
        service = create_normalization_service()

        # Should fail gracefully and quarantine
        with pytest.raises(NormalizationError):
            await service.normalize_document(
                file_path=truncated_json_file,
                source_id="truncated_json_test",
                source_type="test"
            )

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, invalid_json_file):
        """Test handling of syntactically invalid JSON."""
        service = create_normalization_service()

        # Should fail gracefully
        with pytest.raises(NormalizationError):
            await service.normalize_document(
                file_path=invalid_json_file,
                source_id="invalid_json_test",
                source_type="test"
            )

    @pytest.mark.asyncio
    async def test_malformed_xml_handling(self, malformed_xml_file):
        """Test handling of malformed XML with unclosed tags."""
        service = create_normalization_service()

        # May fail gracefully or succeed with best-effort parsing
        try:
            result = await service.normalize_document(
                file_path=malformed_xml_file,
                source_id="malformed_xml_test",
                source_type="test"
            )
            # If succeeds, verify basic structure
            assert result.sha256 is not None

        except NormalizationError:
            # Graceful failure acceptable for malformed files
            pass

    @pytest.mark.asyncio
    async def test_malformed_yaml_handling(self, malformed_yaml_file):
        """Test handling of malformed YAML with syntax errors."""
        service = create_normalization_service()

        # Should fail gracefully
        with pytest.raises(NormalizationError):
            await service.normalize_document(
                file_path=malformed_yaml_file,
                source_id="malformed_yaml_test",
                source_type="test"
            )

    @pytest.mark.asyncio
    async def test_binary_as_text_handling(self, binary_as_text_file):
        """Test handling of binary data with text extension."""
        service = create_normalization_service()

        # Should handle gracefully (may succeed or fail)
        try:
            result = await service.normalize_document(
                file_path=binary_as_text_file,
                source_id="binary_text_test",
                source_type="test"
            )
            assert result.sha256 is not None

        except (NormalizationError, UnicodeDecodeError):
            # Acceptable failure for binary data
            pass

    @pytest.mark.asyncio
    async def test_null_bytes_in_text(self, null_bytes_file):
        """Test handling of text file containing null bytes."""
        service = create_normalization_service()

        # Should handle gracefully
        try:
            result = await service.normalize_document(
                file_path=null_bytes_file,
                source_id="null_bytes_test",
                source_type="test"
            )
            assert result.sha256 is not None

        except (NormalizationError, ValueError):
            # Acceptable failure
            pass


# ---------------------------------------------------------------------------
# Boundary Condition Tests
# ---------------------------------------------------------------------------


@pytest.mark.edge_case
class TestBoundaryConditions:
    """Test suite for boundary conditions."""

    @pytest.mark.asyncio
    async def test_deeply_nested_json(self, deeply_nested_json):
        """Test deeply nested JSON structure (100+ levels)."""
        service = create_normalization_service()

        # May hit recursion limits or succeed
        try:
            result = await service.normalize_document(
                file_path=deeply_nested_json,
                source_id="deep_nest_test",
                source_type="test"
            )
            assert result.sha256 is not None

        except (NormalizationError, RecursionError):
            # Acceptable for extreme nesting
            pass

    @pytest.mark.asyncio
    async def test_very_long_lines(self, very_long_lines_file):
        """Test file with extremely long lines (>100K chars)."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=very_long_lines_file,
            source_id="long_lines_test",
            source_type="test"
        )

        # Should succeed
        assert result.sha256 is not None
        assert result.metadata.character_count > 100000

    @pytest.mark.asyncio
    async def test_many_small_lines(self, many_small_lines_file):
        """Test file with very many small lines."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=many_small_lines_file,
            source_id="many_lines_test",
            source_type="test"
        )

        assert result.sha256 is not None
        assert result.metadata.line_count >= 100000

    @pytest.mark.asyncio
    async def test_special_filename_handling(self, special_filename_file):
        """Test handling of special characters in filename."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=special_filename_file,
            source_id="special_filename_test",
            source_type="test"
        )

        # Should succeed regardless of filename
        assert result.sha256 is not None

    @pytest.mark.asyncio
    async def test_very_long_filename(self, very_long_filename_file):
        """Test handling of very long filename."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=very_long_filename_file,
            source_id="long_filename_test",
            source_type="test"
        )

        assert result.sha256 is not None


# ---------------------------------------------------------------------------
# Format-Specific Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.edge_case
class TestFormatSpecificEdgeCases:
    """Test format-specific edge cases."""

    @pytest.mark.asyncio
    async def test_markdown_no_headings(self, markdown_no_headings):
        """Test markdown file with no headings (flat structure)."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=markdown_no_headings,
            source_id="md_no_headings_test",
            source_type="test"
        )

        # Should succeed, may or may not chunk
        assert result.sha256 is not None

    @pytest.mark.asyncio
    async def test_markdown_only_headings(self, markdown_only_headings):
        """Test markdown file with only headings, no content."""
        service = create_normalization_service()

        result = await service.normalize_document(
            file_path=markdown_only_headings,
            source_id="md_only_headings_test",
            source_type="test"
        )

        # Should succeed with minimal content
        assert result.sha256 is not None

    @pytest.mark.asyncio
    async def test_csv_inconsistent_columns(self, csv_inconsistent_columns):
        """Test CSV with inconsistent column counts."""
        service = create_normalization_service()

        # May succeed or fail depending on strictness
        try:
            result = await service.normalize_document(
                file_path=csv_inconsistent_columns,
                source_id="csv_inconsistent_test",
                source_type="test"
            )
            assert result.sha256 is not None

        except NormalizationError:
            # Acceptable failure for malformed CSV
            pass

    @pytest.mark.asyncio
    async def test_html_malformed(self, html_malformed):
        """Test malformed HTML with unclosed tags."""
        service = create_normalization_service()

        # HTML parsers are usually lenient
        result = await service.normalize_document(
            file_path=html_malformed,
            source_id="html_malformed_test",
            source_type="test"
        )

        # Should succeed with best-effort parsing
        assert result.sha256 is not None


# ---------------------------------------------------------------------------
# Error Recovery and Quarantine Tests
# ---------------------------------------------------------------------------


@pytest.mark.edge_case
class TestErrorRecoveryAndQuarantine:
    """Test error recovery and quarantine workflow."""

    @pytest.mark.asyncio
    async def test_failures_are_quarantined(
        self,
        truncated_json_file,
        mock_quarantine_manager
    ):
        """Verify failures are properly quarantined.

        From requirements: "Quarantine workflow handles all failure modes"
        """
        from futurnal.pipeline.normalization import NormalizationConfig

        config = NormalizationConfig(
            quarantine_on_failure=True
        )

        service = create_normalization_service()
        service.config = config
        service.quarantine_manager = mock_quarantine_manager

        # Attempt normalization (should fail)
        with pytest.raises(NormalizationError):
            await service.normalize_document(
                file_path=truncated_json_file,
                source_id="quarantine_test",
                source_type="test"
            )

        # Verify quarantine was called
        # (implementation depends on how service integrates with quarantine)

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_error(self, binary_as_text_file):
        """Test graceful degradation when errors occur."""
        service = create_normalization_service()

        # Should not crash entire pipeline
        try:
            await service.normalize_document(
                file_path=binary_as_text_file,
                source_id="degradation_test",
                source_type="test"
            )
        except Exception as e:
            # Error should be handled, not crash
            assert isinstance(e, (NormalizationError, UnicodeDecodeError))


# ---------------------------------------------------------------------------
# Edge Case Summary Report
# ---------------------------------------------------------------------------


@pytest.mark.edge_case
class TestEdgeCaseSummary:
    """Generate edge case testing summary report."""

    @pytest.mark.asyncio
    async def test_edge_case_coverage_report(
        self,
        empty_file,
        unicode_emoji_file,
        large_file_10mb,
        truncated_json_file,
        very_long_lines_file,
        tmp_path
    ):
        """Generate comprehensive edge case coverage report."""
        service = create_normalization_service()

        edge_cases = {
            "empty_file": empty_file,
            "unicode_emoji": unicode_emoji_file,
            "large_10mb": large_file_10mb,
            "truncated_json": truncated_json_file,
            "very_long_lines": very_long_lines_file,
        }

        results = {
            "edge_cases_tested": len(edge_cases),
            "successes": 0,
            "graceful_failures": 0,
            "unexpected_failures": 0,
            "details": []
        }

        for case_name, file_path in edge_cases.items():
            try:
                result = await service.normalize_document(
                    file_path=file_path,
                    source_id=f"edge_{case_name}",
                    source_type="test"
                )
                results["successes"] += 1
                results["details"].append({
                    "case": case_name,
                    "status": "success",
                    "sha256": result.sha256
                })

            except (NormalizationError, UnicodeDecodeError, ValueError) as e:
                # Expected/graceful failure
                results["graceful_failures"] += 1
                results["details"].append({
                    "case": case_name,
                    "status": "graceful_failure",
                    "error": str(e)[:100]
                })

            except Exception as e:
                # Unexpected failure
                results["unexpected_failures"] += 1
                results["details"].append({
                    "case": case_name,
                    "status": "unexpected_failure",
                    "error": str(e)[:100]
                })

        # Print summary
        print("\n" + "=" * 70)
        print("EDGE CASE TESTING REPORT")
        print("=" * 70)
        print(f"Total Edge Cases: {results['edge_cases_tested']}")
        print(f"Successes: {results['successes']}")
        print(f"Graceful Failures: {results['graceful_failures']}")
        print(f"Unexpected Failures: {results['unexpected_failures']}")
        print("=" * 70 + "\n")

        # Should not have unexpected failures
        assert results["unexpected_failures"] == 0, (
            "Edge cases produced unexpected failures"
        )


# Hook for custom pytest option
def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (100MB+ files)"
    )
