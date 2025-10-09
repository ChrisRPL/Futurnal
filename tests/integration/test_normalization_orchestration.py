"""End-to-end integration tests for normalization orchestration.

Tests the complete pipeline from file ingestion through normalization to
PKG/Vector storage, validating the full orchestrator integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from futurnal.ingestion.local.state import StateStore
from futurnal.pipeline.normalization import create_normalization_processor_with_workspace
from futurnal.pipeline.stubs import NormalizationSink
from futurnal.privacy.audit import AuditLogger


# ============================================================================
# Mock Storage Classes
# ============================================================================


class MockPKGWriter:
    """Mock PKG writer for testing."""

    def __init__(self):
        self.documents_written = 0
        self.documents: List[Dict[str, Any]] = []

    def write_document(self, payload: Dict[str, Any]) -> None:
        """Write document to mock storage."""
        self.documents_written += 1
        self.documents.append(payload)

    def remove_document(self, sha256: str) -> None:
        """Remove document by SHA256."""
        self.documents = [d for d in self.documents if d.get("sha256") != sha256]


class MockVectorWriter:
    """Mock vector writer for testing."""

    def __init__(self):
        self.embeddings_written = 0
        self.embeddings: List[Dict[str, Any]] = []

    def write_embedding(self, payload: Dict[str, Any]) -> None:
        """Write embedding to mock storage."""
        self.embeddings_written += 1
        self.embeddings.append(payload)

    def remove_embedding(self, sha256: str) -> None:
        """Remove embedding by SHA256."""
        self.embeddings = [e for e in self.embeddings if e.get("sha256") != sha256]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def mock_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock Unstructured.io partition function."""

    def _fake_partition(
        *, filename: str, strategy: str, include_metadata: bool, content_type: str = None
    ):
        """Fake partition that returns simple elements from file content."""
        path = Path(filename)
        try:
            content = path.read_text()
        except Exception:
            return [
                {
                    "text": "Binary content",
                    "type": "Unknown",
                    "metadata": {"filename": filename},
                }
            ]

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [content.strip()]

        elements = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph.startswith("#"):
                element_type = "Title"
            else:
                element_type = "NarrativeText"

            elements.append(
                {
                    "text": paragraph,
                    "type": element_type,
                    "metadata": {
                        "filename": filename,
                        "element_id": f"elem-{i}",
                        "languages": ["en"],
                        "file_size": path.stat().st_size if path.exists() else 0,
                    },
                }
            )

        return elements

    # Monkeypatch partition
    try:
        monkeypatch.setattr(
            "futurnal.pipeline.normalization.unstructured_bridge.partition",
            _fake_partition,
        )
    except (AttributeError, ImportError):
        pass


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================


class TestNormalizationOrchestrationE2E:
    """End-to-end tests for normalization orchestration."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_local_files(self, tmp_path: Path):
        """Test complete pipeline from local files to PKG storage."""
        # Setup workspace
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create test documents
        (source_dir / "note1.md").write_text("# Note 1\nThis is the first note.")
        (source_dir / "note2.md").write_text("# Note 2\nThis is the second note.")
        (source_dir / "doc.txt").write_text("Plain text document content.")

        # Setup state store
        state_store = StateStore(workspace / "state" / "state.db")

        # Setup mock storage
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        # Create normalization processor
        processor = create_normalization_processor_with_workspace(
            workspace_path=workspace,
            state_store=state_store,
            sink=sink,
        )

        # Process all files
        files = list(source_dir.glob("*"))
        for file in files:
            result = await processor.process_file(
                file_path=file,
                source_id=f"local-{file.name}",
                source_type="local_files",
            )
            assert result.success, f"Failed to process {file.name}: {result.error_message}"

        # Verify all documents written to PKG
        assert pkg_writer.documents_written == 3, f"Expected 3 documents, got {pkg_writer.documents_written}"
        assert vector_writer.embeddings_written == 3, f"Expected 3 embeddings, got {vector_writer.embeddings_written}"

        # Verify state store updated for all files
        for file in files:
            state_record = state_store.fetch(file)
            assert state_record is not None, f"State not stored for {file.name}"
            assert state_record.sha256 is not None

        # Verify metrics
        metrics = processor.get_metrics()
        assert metrics["files_processed"] == 3
        assert metrics["files_failed"] == 0
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_with_state_checkpointing(self, tmp_path: Path):
        """Test pipeline respects state checkpointing."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create test file
        test_file = source_dir / "test.md"
        test_file.write_text("# Test\nContent")

        # Setup components
        state_store = StateStore(workspace / "state" / "state.db")
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        processor = create_normalization_processor_with_workspace(
            workspace_path=workspace,
            state_store=state_store,
            sink=sink,
        )

        # First run - should process
        result1 = await processor.process_file(
            file_path=test_file,
            source_id="test-1",
            source_type="local_files",
        )
        assert result1.success
        assert not result1.was_cached

        # Second run - should be cached
        result2 = await processor.process_file(
            file_path=test_file,
            source_id="test-1",
            source_type="local_files",
        )
        assert result2.success
        assert result2.was_cached

        # Verify only processed once
        assert pkg_writer.documents_written == 1
        metrics = processor.get_metrics()
        assert metrics["files_processed"] == 1
        assert metrics["files_skipped_cached"] == 1

    @pytest.mark.asyncio
    async def test_pipeline_with_audit_trail(self, tmp_path: Path):
        """Test pipeline creates comprehensive audit trail."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create test file
        test_file = source_dir / "test.txt"
        test_file.write_text("Test content")

        # Setup components
        state_store = StateStore(workspace / "state" / "state.db")
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        processor = create_normalization_processor_with_workspace(
            workspace_path=workspace,
            state_store=state_store,
            sink=sink,
        )

        # Process file
        result = await processor.process_file(
            file_path=test_file,
            source_id="test-1",
            source_type="local_files",
        )
        assert result.success

        # Verify audit log created
        audit_dir = workspace / "audit"
        assert audit_dir.exists()

        audit_files = list(audit_dir.glob("*.log"))
        assert len(audit_files) > 0

        # Read audit events
        events = []
        for audit_file in audit_files:
            for line in audit_file.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))

        # Verify events captured
        assert len(events) >= 2  # Start and completion events

        event_actions = [e["action"] for e in events]
        assert "normalization_processing" in event_actions
        assert "normalization_completed" in event_actions

        # Verify no content in audit
        audit_text = json.dumps(events).lower()
        assert "test content" not in audit_text

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, tmp_path: Path):
        """Test pipeline handles errors gracefully."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Setup components
        state_store = StateStore(workspace / "state" / "state.db")
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        processor = create_normalization_processor_with_workspace(
            workspace_path=workspace,
            state_store=state_store,
            sink=sink,
        )

        # Try to process non-existent file
        result = await processor.process_file(
            file_path=Path("/nonexistent/file.txt"),
            source_id="error-test",
            source_type="local_files",
        )

        # Verify error handling
        assert not result.success
        assert result.error_message is not None

        # Verify metrics
        metrics = processor.get_metrics()
        assert metrics["files_failed"] == 1
        assert metrics["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_file_types(self, tmp_path: Path):
        """Test pipeline handles different file types correctly."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create files of different types
        (source_dir / "note.md").write_text("# Markdown\nContent")
        (source_dir / "doc.txt").write_text("Plain text")
        (source_dir / "script.py").write_text("def hello():\n    pass")

        # Setup components
        state_store = StateStore(workspace / "state" / "state.db")
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        processor = create_normalization_processor_with_workspace(
            workspace_path=workspace,
            state_store=state_store,
            sink=sink,
        )

        # Process all files
        files = list(source_dir.glob("*"))
        for file in files:
            result = await processor.process_file(
                file_path=file,
                source_id=f"test-{file.name}",
                source_type="local_files",
            )
            assert result.success, f"Failed to process {file.name}"

        # Verify all processed
        assert pkg_writer.documents_written == 3
        metrics = processor.get_metrics()
        assert metrics["files_processed"] == 3
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_batch_processing(self, tmp_path: Path):
        """Test pipeline batch processing efficiency."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create multiple files
        num_files = 10
        for i in range(num_files):
            (source_dir / f"file{i}.txt").write_text(f"Content {i}")

        # Setup components
        state_store = StateStore(workspace / "state" / "state.db")
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        processor = create_normalization_processor_with_workspace(
            workspace_path=workspace,
            state_store=state_store,
            sink=sink,
        )

        # Process batch
        files = [(f, f"id-{i}") for i, f in enumerate(source_dir.glob("*"))]
        results = await processor.process_batch(
            files=files,
            source_type="local_files",
        )

        # Verify all processed
        assert len(results) == num_files
        assert all(r.success for r in results)

        # Verify storage
        assert pkg_writer.documents_written == num_files
        assert vector_writer.embeddings_written == num_files

        # Verify metrics
        metrics = processor.get_metrics()
        assert metrics["files_processed"] == num_files
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_with_file_updates(self, tmp_path: Path):
        """Test pipeline handles file updates correctly."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        test_file = source_dir / "test.txt"
        test_file.write_text("Original content")

        # Setup components
        state_store = StateStore(workspace / "state" / "state.db")
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        processor = create_normalization_processor_with_workspace(
            workspace_path=workspace,
            state_store=state_store,
            sink=sink,
        )

        # Process original file
        result1 = await processor.process_file(
            file_path=test_file,
            source_id="test-1",
            source_type="local_files",
        )
        assert result1.success
        original_doc_id = result1.document_id

        # Update file
        import time
        time.sleep(0.1)  # Ensure mtime changes
        test_file.write_text("Updated content")

        # Process updated file
        result2 = await processor.process_file(
            file_path=test_file,
            source_id="test-1",
            source_type="local_files",
        )
        assert result2.success
        updated_doc_id = result2.document_id

        # Verify different document IDs (content changed)
        assert updated_doc_id != original_doc_id

        # Verify both versions processed
        assert pkg_writer.documents_written == 2
        metrics = processor.get_metrics()
        assert metrics["files_processed"] == 2


class TestPerformanceMetrics:
    """Tests for performance metrics collection."""

    @pytest.mark.asyncio
    async def test_processing_time_metrics(self, tmp_path: Path):
        """Test processing time metrics are accurate."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create test files
        for i in range(5):
            (source_dir / f"file{i}.txt").write_text(f"Content {i}")

        # Setup components
        state_store = StateStore(workspace / "state" / "state.db")
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        processor = create_normalization_processor_with_workspace(
            workspace_path=workspace,
            state_store=state_store,
            sink=sink,
        )

        # Process files
        for file in source_dir.glob("*"):
            await processor.process_file(
                file_path=file,
                source_id=f"test-{file.name}",
                source_type="local_files",
            )

        # Verify metrics
        metrics = processor.get_metrics()
        assert metrics["total_processing_time_ms"] > 0
        assert metrics["average_processing_time_ms"] > 0
        assert (
            metrics["average_processing_time_ms"]
            == metrics["total_processing_time_ms"] / metrics["files_processed"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
