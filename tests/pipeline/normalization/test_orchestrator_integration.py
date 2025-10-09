"""Integration tests for NormalizationProcessor with orchestrator components.

Tests the complete integration between NormalizationService and orchestrator
components including state checkpointing, audit logging, and metrics collection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.quarantine import QuarantineStore
from futurnal.pipeline.models import DocumentFormat
from futurnal.pipeline.normalization import (
    create_normalization_processor,
    create_normalization_processor_with_workspace,
    NormalizationProcessor,
    ProcessingResult,
)
from futurnal.privacy.audit import AuditLogger


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def state_store(temp_workspace):
    """Create state store for tests."""
    state_db = temp_workspace / "state" / "state.db"
    state_db.parent.mkdir(parents=True, exist_ok=True)
    return StateStore(state_db)


@pytest.fixture
def audit_logger(temp_workspace):
    """Create audit logger for tests."""
    audit_dir = temp_workspace / "audit"
    return AuditLogger(output_dir=audit_dir)


@pytest.fixture
def normalization_processor(
    mock_normalization_sink,
    state_store,
    audit_logger,
    mock_unstructured_partition,
    mock_language_detector,
):
    """Create configured normalization processor for tests."""
    return create_normalization_processor(
        state_store=state_store,
        audit_logger=audit_logger,
        sink=mock_normalization_sink,
        enable_state_checkpointing=True,
    )


class TestNormalizationProcessor:
    """Tests for NormalizationProcessor."""

    @pytest.mark.asyncio
    async def test_process_file_success(
        self, normalization_processor, temp_file, mock_normalization_sink
    ):
        """Test successful file processing."""
        content = "# Test Document\n\nThis is test content."
        test_file = temp_file(content, "test.md")

        result = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        # Verify result
        assert result.success is True
        assert result.file_path == test_file
        assert result.source_id == "test-001"
        assert result.document_id is not None
        assert result.processing_duration_ms is not None
        assert result.processing_duration_ms > 0
        assert result.was_cached is False

        # Verify sink received document
        assert len(mock_normalization_sink.handled_documents) == 1

        # Verify metrics updated
        metrics = normalization_processor.get_metrics()
        assert metrics["files_processed"] == 1
        assert metrics["files_failed"] == 0
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_state_checkpointing_prevents_reprocessing(
        self, normalization_processor, temp_file, mock_normalization_sink
    ):
        """Test that state checkpointing prevents duplicate processing."""
        content = "Test content for state checkpointing"
        test_file = temp_file(content, "test.txt")

        # First processing - should succeed
        result1 = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        assert result1.success is True
        assert result1.was_cached is False

        # Second processing - should be cached
        result2 = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        assert result2.success is True
        assert result2.was_cached is True

        # Verify only processed once
        metrics = normalization_processor.get_metrics()
        assert metrics["files_processed"] == 1  # Only first run
        assert metrics["files_skipped_cached"] == 1
        assert metrics["cache_hit_rate"] == 0.5  # 1 cached out of 2 total

        # Verify sink received document only once
        assert len(mock_normalization_sink.handled_documents) == 1

    @pytest.mark.asyncio
    async def test_force_reprocess_bypasses_cache(
        self, normalization_processor, temp_file, mock_normalization_sink
    ):
        """Test that force_reprocess bypasses state cache."""
        content = "Test content"
        test_file = temp_file(content, "test.txt")

        # First processing
        result1 = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        assert result1.success is True
        assert result1.was_cached is False

        # Force reprocessing - should bypass cache
        result2 = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-002",
            source_type="local_files",
            force_reprocess=True,
        )

        assert result2.success is True
        assert result2.was_cached is False

        # Verify processed twice
        metrics = normalization_processor.get_metrics()
        assert metrics["files_processed"] == 2
        assert metrics["files_skipped_cached"] == 0

        # Verify sink received document twice
        assert len(mock_normalization_sink.handled_documents) == 2

    @pytest.mark.asyncio
    async def test_audit_logging_completeness(
        self, normalization_processor, temp_file, audit_logger, temp_workspace
    ):
        """Test comprehensive audit logging for normalization events."""
        content = "Test content for audit logging"
        test_file = temp_file(content, "test.txt")

        await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        # Verify audit log files created
        audit_dir = temp_workspace / "audit"
        audit_files = list(audit_dir.glob("*.log"))
        assert len(audit_files) > 0

        # Read audit events
        events = []
        for audit_file in audit_files:
            import json
            for line in audit_file.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))

        # Verify expected events
        assert len(events) >= 2  # At least start and success events
        event_actions = {e["action"] for e in events}
        assert "normalization_processing" in event_actions
        assert "normalization_completed" in event_actions

        # Verify no content exposure in audit
        audit_content = " ".join(str(e) for e in events).lower()
        assert "test content" not in audit_content

    @pytest.mark.asyncio
    async def test_error_handling_and_metrics(
        self, normalization_processor, temp_file
    ):
        """Test error handling updates metrics correctly."""
        # Create a file that will cause normalization error (non-existent)
        fake_file = Path("/nonexistent/file.txt")

        result = await normalization_processor.process_file(
            file_path=fake_file,
            source_id="test-error",
            source_type="local_files",
        )

        # Verify error result
        assert result.success is False
        assert result.error_message is not None
        assert result.processing_duration_ms is not None

        # Verify metrics updated
        metrics = normalization_processor.get_metrics()
        assert metrics["files_failed"] == 1
        assert metrics["files_processed"] == 0
        assert metrics["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_process_batch(
        self, normalization_processor, temp_file, mock_normalization_sink
    ):
        """Test batch processing of multiple files."""
        files = [
            (temp_file("Content 1", "file1.txt"), "id-001"),
            (temp_file("Content 2", "file2.txt"), "id-002"),
            (temp_file("Content 3", "file3.txt"), "id-003"),
        ]

        results = await normalization_processor.process_batch(
            files=files,
            source_type="local_files",
        )

        # Verify all files processed
        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify metrics
        metrics = normalization_processor.get_metrics()
        assert metrics["files_processed"] == 3
        assert metrics["success_rate"] == 1.0

        # Verify all delivered to sink
        assert len(mock_normalization_sink.handled_documents) == 3

    @pytest.mark.asyncio
    async def test_different_file_formats(
        self, normalization_processor, temp_file
    ):
        """Test processing different file formats."""
        test_files = {
            "markdown": ("# Markdown\nContent", "test.md", DocumentFormat.MARKDOWN),
            "text": ("Plain text", "test.txt", DocumentFormat.TEXT),
            "python": ("def hello():\n    pass", "test.py", DocumentFormat.CODE),
        }

        for name, (content, filename, expected_format) in test_files.items():
            test_file = temp_file(content, filename)
            result = await normalization_processor.process_file(
                file_path=test_file,
                source_id=f"test-{name}",
                source_type="local_files",
            )
            assert result.success is True, f"Failed to process {name}"

        # Verify all processed
        metrics = normalization_processor.get_metrics()
        assert metrics["files_processed"] == len(test_files)

    @pytest.mark.asyncio
    async def test_state_cache_invalidation_on_file_change(
        self, normalization_processor, tmp_path
    ):
        """Test that state cache is invalidated when file changes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # First processing
        result1 = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )
        assert result1.success is True
        assert result1.was_cached is False

        # Modify file
        import time
        time.sleep(0.1)  # Ensure mtime changes
        test_file.write_text("Modified content")

        # Second processing - should reprocess due to change
        result2 = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )
        assert result2.success is True
        assert result2.was_cached is False  # Not cached because file changed

        # Verify processed twice
        metrics = normalization_processor.get_metrics()
        assert metrics["files_processed"] == 2


class TestNormalizationProcessorFactory:
    """Tests for processor factory functions."""

    @pytest.mark.asyncio
    async def test_create_processor_with_workspace(
        self, temp_workspace, state_store, mock_normalization_sink,
        mock_unstructured_partition, mock_language_detector
    ):
        """Test creating processor with workspace factory."""
        processor = create_normalization_processor_with_workspace(
            workspace_path=temp_workspace,
            state_store=state_store,
            sink=mock_normalization_sink,
        )

        assert isinstance(processor, NormalizationProcessor)
        assert processor.state_store is not None
        assert processor.audit_logger is not None

        # Verify it works
        content = "Test content"
        test_file = temp_workspace / "test.txt"
        test_file.write_text(content)

        result = await processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        assert result.success is True


class TestMetricsCollection:
    """Tests for metrics collection and reporting."""

    @pytest.mark.asyncio
    async def test_metrics_accuracy(
        self, normalization_processor, temp_file
    ):
        """Test that metrics are accurately collected."""
        # Process successful files
        for i in range(3):
            test_file = temp_file(f"Content {i}", f"file{i}.txt")
            await normalization_processor.process_file(
                file_path=test_file,
                source_id=f"id-{i}",
                source_type="local_files",
            )

        # Process failed file
        await normalization_processor.process_file(
            file_path=Path("/nonexistent.txt"),
            source_id="error-id",
            source_type="local_files",
        )

        metrics = normalization_processor.get_metrics()

        # Verify counts
        assert metrics["files_processed"] == 3
        assert metrics["files_failed"] == 1
        assert metrics["total_files"] == 4

        # Verify rates
        assert metrics["success_rate"] == 0.75  # 3/4
        assert metrics["average_processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_processing_time_tracking(
        self, normalization_processor, temp_file
    ):
        """Test that processing time is accurately tracked."""
        test_file = temp_file("Test content", "test.txt")

        result = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        # Verify duration tracked in result
        assert result.processing_duration_ms is not None
        assert result.processing_duration_ms > 0

        # Verify duration in metrics
        metrics = normalization_processor.get_metrics()
        assert metrics["total_processing_time_ms"] > 0
        assert metrics["average_processing_time_ms"] > 0
        assert (
            metrics["average_processing_time_ms"]
            == metrics["total_processing_time_ms"]
        )  # Only one file


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_file(self, normalization_processor, temp_file):
        """Test processing empty file."""
        test_file = temp_file("", "empty.txt")

        result = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-empty",
            source_type="local_files",
        )

        # Should succeed even for empty file
        assert result.success is True

    @pytest.mark.asyncio
    async def test_very_large_filename(self, normalization_processor, tmp_path):
        """Test processing file with very long filename."""
        long_name = "x" * 200 + ".txt"
        test_file = tmp_path / long_name
        test_file.write_text("Content")

        result = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-long",
            source_type="local_files",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_special_characters_in_filename(
        self, normalization_processor, tmp_path
    ):
        """Test processing file with special characters in name."""
        special_name = "test file [2023] (draft).txt"
        test_file = tmp_path / special_name
        test_file.write_text("Content with special filename")

        result = await normalization_processor.process_file(
            file_path=test_file,
            source_id="test-special",
            source_type="local_files",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_processing_without_state_store(
        self, mock_normalization_sink,
        audit_logger,
        mock_unstructured_partition,
        mock_language_detector,
        temp_file,
    ):
        """Test processor works without state store."""
        processor = create_normalization_processor(
            state_store=None,  # No state store
            audit_logger=audit_logger,
            sink=mock_normalization_sink,
            enable_state_checkpointing=False,
        )

        test_file = temp_file("Content", "test.txt")

        result = await processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        # Should still work without state store
        assert result.success is True
        assert result.was_cached is False

    @pytest.mark.asyncio
    async def test_processing_without_audit_logger(
        self, mock_normalization_sink,
        state_store,
        mock_unstructured_partition,
        mock_language_detector,
        temp_file,
    ):
        """Test processor works without audit logger."""
        processor = create_normalization_processor(
            state_store=state_store,
            audit_logger=None,  # No audit logger
            sink=mock_normalization_sink,
        )

        test_file = temp_file("Content", "test.txt")

        result = await processor.process_file(
            file_path=test_file,
            source_id="test-001",
            source_type="local_files",
        )

        # Should still work without audit logger
        assert result.success is True
