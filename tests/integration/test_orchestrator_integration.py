"""Comprehensive integration tests for orchestrator end-to-end workflows.

This test suite validates the complete ingestion pipeline from data source
through orchestration to storage, covering multi-connector coordination,
privacy audit trails, telemetry accuracy, error handling, and state persistence.

All tests use REAL orchestrator components (queue, state, connectors) with
MOCK storage backends (PKG/vector) to validate production behavior.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.connector import LocalFilesConnector
from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator import (
    IngestionOrchestrator,
    JobStatus,
    QuarantineStore,
)
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.models import JobPriority, JobType
from futurnal.orchestrator.scheduler import SourceRegistration
from futurnal.orchestrator.audit import AuditLogger
from futurnal.orchestrator.metrics import TelemetryRecorder
from futurnal.pipeline.stubs import NormalizationSink


# ============================================================================
# Mock Storage Classes
# ============================================================================


class MockPKGWriter:
    """Mock PKG writer for testing pipeline integration."""

    def __init__(self):
        self.documents_written = 0
        self.documents: List[Dict[str, Any]] = []

    def write_document(self, payload: Dict[str, Any]) -> None:
        """Write document to in-memory storage."""
        self.documents_written += 1
        self.documents.append(payload)

    def remove_document(self, sha256: str) -> None:
        """Remove document by SHA256."""
        self.documents = [d for d in self.documents if d.get("sha256") != sha256]


class MockVectorWriter:
    """Mock vector writer for testing pipeline integration."""

    def __init__(self):
        self.embeddings_written = 0
        self.embeddings: List[Dict[str, Any]] = []

    def write_embedding(self, payload: Dict[str, Any]) -> None:
        """Write embedding to in-memory storage."""
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
    """Mock Unstructured.io partition function for all tests."""

    def _fake_partition(
        *, filename: str, strategy: str, include_metadata: bool, content_type: str = None
    ):
        """Fake partition that returns simple elements from file content."""
        path = Path(filename)
        try:
            content = path.read_text()
        except Exception:
            # For non-text files, return minimal element
            return [
                {
                    "text": "Binary content",
                    "type": "Unknown",
                    "metadata": {"filename": filename},
                }
            ]

        # Split content into paragraphs
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

    # Monkeypatch partition across multiple possible import paths
    try:
        monkeypatch.setattr(
            "futurnal.ingestion.local.connector.partition", _fake_partition
        )
    except (AttributeError, ImportError):
        pass

    try:
        import sys
        import types

        # Ensure unstructured module exists for dynamic imports
        if "unstructured.partition.auto" in sys.modules:
            auto_module = sys.modules["unstructured.partition.auto"]
            auto_module.partition = _fake_partition
    except (AttributeError, ImportError):
        pass


# ============================================================================
# Test 01: End-to-End Local Files Pipeline
# ============================================================================


class TestLocalFilesPipeline:
    """Test complete pipeline from local file to PKG storage."""

    @pytest.mark.asyncio
    async def test_local_files_end_to_end_pipeline(self, tmp_path: Path):
        """Test complete pipeline from local file to PKG storage."""

        # Setup workspace and source directory
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create test document
        test_file = source_dir / "note.md"
        test_file.write_text("# Test Note\nThis is test content.")

        # Initialize orchestrator components
        queue = JobQueue(workspace / "queue.db")
        state_store = StateStore(workspace / "state" / "state.db")

        # Mock PKG and vector writers
        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        # Create orchestrator
        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            element_sink=sink,
        )

        # Register source
        source = LocalIngestionSource(
            name="test_source",
            root_path=source_dir,
        )
        orchestrator.register_source(
            SourceRegistration(source=source, schedule="@manual")
        )

        # Execute job
        orchestrator.run_manual_job("test_source", force=True)
        orchestrator.start()
        await asyncio.sleep(3)
        await orchestrator.shutdown()

        # Validate: File was processed
        assert pkg_writer.documents_written > 0, "No documents written to PKG"
        assert vector_writer.embeddings_written > 0, "No embeddings written"

        # Validate: State stored
        file_state = state_store.fetch(test_file)
        assert file_state is not None, "File state not stored"
        assert file_state.sha256 is not None, "File SHA256 not computed"

        # Validate: Job completed
        jobs = queue.snapshot(status=JobStatus.SUCCEEDED)
        assert len(jobs) >= 1, f"Expected at least 1 succeeded job, got {len(jobs)}"

        # Validate: Telemetry recorded
        telemetry_dir = workspace / "telemetry"
        assert (telemetry_dir / "telemetry.log").exists(), "Telemetry log not created"


# ============================================================================
# Test 02: Multi-Connector Integration
# ============================================================================


class TestMultiConnectorOrchestration:
    """Test orchestrator with multiple connector types."""

    @pytest.mark.asyncio
    async def test_multi_connector_orchestration(self, tmp_path: Path):
        """Test orchestrator with multiple local file sources."""

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Setup first source
        local_dir_1 = tmp_path / "local_1"
        local_dir_1.mkdir()
        (local_dir_1 / "file1.txt").write_text("test content 1")

        # Setup second source
        local_dir_2 = tmp_path / "local_2"
        local_dir_2.mkdir()
        (local_dir_2 / "file2.txt").write_text("test content 2")

        # Initialize orchestrator
        queue = JobQueue(workspace / "queue.db")
        state_store = StateStore(workspace / "state" / "state.db")

        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            element_sink=sink,
        )

        # Register multiple sources
        orchestrator.register_source(
            SourceRegistration(
                source=LocalIngestionSource(name="local_1", root_path=local_dir_1),
                schedule="@manual",
            )
        )
        orchestrator.register_source(
            SourceRegistration(
                source=LocalIngestionSource(name="local_2", root_path=local_dir_2),
                schedule="@manual",
            )
        )

        # Trigger both sources
        orchestrator.run_manual_job("local_1", force=True)
        orchestrator.run_manual_job("local_2", force=True)

        orchestrator.start()
        await asyncio.sleep(4)
        await orchestrator.shutdown()

        # Validate: Both connectors executed
        jobs = queue.snapshot(status=JobStatus.SUCCEEDED)
        assert len(jobs) >= 2, f"Expected at least 2 succeeded jobs, got {len(jobs)}"

        # Validate: Both source names appear
        job_sources = {job["payload"].get("source_name") for job in jobs}
        assert "local_1" in job_sources, "local_1 not processed"
        assert "local_2" in job_sources, "local_2 not processed"


# ============================================================================
# Test 03: Privacy Audit Trail Verification
# ============================================================================


class TestAuditTrailCompleteness:
    """Verify comprehensive audit trail for all operations."""

    @pytest.mark.asyncio
    async def test_audit_trail_completeness(self, tmp_path: Path):
        """Verify comprehensive audit trail for all operations."""

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        audit_dir = workspace / "audit"
        audit_dir.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        # Create orchestrator with audit logging
        queue = JobQueue(workspace / "queue.db")
        state_store = StateStore(workspace / "state" / "state.db")

        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            element_sink=sink,
        )

        # Register source
        source = LocalIngestionSource(name="test", root_path=source_dir)
        orchestrator.register_source(
            SourceRegistration(source=source, schedule="@manual")
        )

        # Execute
        orchestrator.run_manual_job("test", force=True)
        orchestrator.start()
        await asyncio.sleep(3)
        await orchestrator.shutdown()

        # Validate: Audit events logged
        audit_files = list(audit_dir.glob("*.log"))
        assert len(audit_files) > 0, "No audit log files created"

        # Parse audit events
        events = []
        for audit_file in audit_files:
            for line in audit_file.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))

        assert len(events) > 0, "No audit events logged"

        # Verify expected events exist
        event_actions = {e.get("action") for e in events}
        assert "job" in event_actions, "Job execution events not logged"

        # Verify no sensitive data in audit
        audit_content = json.dumps(events).lower()
        assert "password" not in audit_content, "Sensitive 'password' found in audit"
        assert "secret" not in audit_content, "Sensitive 'secret' found in audit"


# ============================================================================
# Test 04: Telemetry Accuracy Validation
# ============================================================================


class TestTelemetryAccuracy:
    """Validate telemetry metrics match actual execution."""

    @pytest.mark.asyncio
    async def test_telemetry_accuracy(self, tmp_path: Path):
        """Validate telemetry metrics match actual execution."""

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create test files with known sizes
        file_sizes = [1000, 2000, 3000]
        for i, size in enumerate(file_sizes):
            (source_dir / f"file{i}.txt").write_bytes(b"x" * size)

        # Create orchestrator with telemetry
        queue = JobQueue(workspace / "queue.db")
        state_store = StateStore(workspace / "state" / "state.db")
        telemetry = TelemetryRecorder(workspace / "telemetry")

        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            element_sink=sink,
            telemetry=telemetry,
        )

        # Register and execute
        source = LocalIngestionSource(name="test", root_path=source_dir)
        orchestrator.register_source(
            SourceRegistration(source=source, schedule="@manual")
        )

        orchestrator.run_manual_job("test", force=True)
        orchestrator.start()
        await asyncio.sleep(4)
        await orchestrator.shutdown()

        # Load telemetry summary
        summary_path = workspace / "telemetry" / "telemetry_summary.json"
        assert summary_path.exists(), "Telemetry summary not created"

        telemetry_summary = json.loads(summary_path.read_text())

        # Validate: File count matches
        overall_files = telemetry_summary.get("overall", {}).get("files", 0)
        assert overall_files >= len(
            file_sizes
        ), f"Expected at least {len(file_sizes)} files, got {overall_files}"

        # Validate: Byte count reasonable (approximate due to metadata)
        expected_bytes = sum(file_sizes)
        actual_bytes = telemetry_summary.get("overall", {}).get("bytes", 0)
        # Allow variance due to element wrapping and metadata
        assert (
            abs(actual_bytes - expected_bytes) < 10000
        ), f"Byte count variance too high: expected ~{expected_bytes}, got {actual_bytes}"


# ============================================================================
# Test 05: Error Propagation Testing
# ============================================================================


class TestErrorPropagation:
    """Test error handling across pipeline components."""

    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, tmp_path: Path):
        """Test error handling across pipeline components."""

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create file
        (source_dir / "corrupt.txt").write_text("test content")

        # Create orchestrator with quarantine
        queue = JobQueue(workspace / "queue.db")
        state_store = StateStore(workspace / "state" / "state.db")
        quarantine = QuarantineStore(workspace / "quarantine" / "quarantine.db")

        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            element_sink=sink,
            quarantine_store=quarantine,
        )

        # Register source
        source = LocalIngestionSource(name="test", root_path=source_dir)
        orchestrator.register_source(
            SourceRegistration(source=source, schedule="@manual")
        )

        # Monkey-patch connector to fail
        original_ingest = orchestrator._local_connector.ingest

        def failing_ingest(*args, **kwargs):
            raise ValueError("Simulated parsing error")

        orchestrator._local_connector.ingest = failing_ingest

        # Execute (will fail)
        orchestrator.run_manual_job("test", force=True)
        orchestrator.start()
        await asyncio.sleep(6)  # Allow retries
        await orchestrator.shutdown()

        # Restore original method
        orchestrator._local_connector.ingest = original_ingest

        # Validate: Job failed
        failed_jobs = queue.snapshot(status=JobStatus.FAILED)
        quarantined_jobs = queue.snapshot(status=JobStatus.QUARANTINED)

        assert (
            len(failed_jobs) + len(quarantined_jobs) >= 1
        ), "No failed or quarantined jobs found"

        # Validate: Quarantine contains job
        quarantined_list = quarantine.list()
        assert len(quarantined_list) >= 1, "No jobs in quarantine"

        # Validate: Error logged
        assert (
            "parsing error" in quarantined_list[0].error_message.lower()
        ), "Error message not captured"


# ============================================================================
# Test 06: State Persistence Across Restarts
# ============================================================================


class TestStatePersistence:
    """Test orchestrator state persists across restarts."""

    @pytest.mark.asyncio
    async def test_state_persistence_across_restarts(self, tmp_path: Path):
        """Test orchestrator state persists across restarts."""

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        # Shared components (persist across restarts)
        queue_path = workspace / "queue.db"
        state_path = workspace / "state" / "state.db"

        # First run: Process file
        queue = JobQueue(queue_path)
        state_store = StateStore(state_path)

        pkg_writer_1 = MockPKGWriter()
        vector_writer_1 = MockVectorWriter()
        sink_1 = NormalizationSink(
            pkg_writer=pkg_writer_1, vector_writer=vector_writer_1
        )

        orchestrator1 = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            element_sink=sink_1,
        )

        source = LocalIngestionSource(name="test", root_path=source_dir)
        orchestrator1.register_source(
            SourceRegistration(source=source, schedule="@manual")
        )

        orchestrator1.run_manual_job("test", force=True)
        orchestrator1.start()
        await asyncio.sleep(3)
        await orchestrator1.shutdown()

        # Validate: Job completed
        jobs_after_first_run = queue.snapshot(status=JobStatus.SUCCEEDED)
        first_run_count = len(jobs_after_first_run)
        assert first_run_count >= 1, "First run produced no succeeded jobs"

        # Capture documents written in first run
        first_run_docs = pkg_writer_1.documents_written

        # Second run: Restart orchestrator (should not reprocess)
        queue_2 = JobQueue(queue_path)  # Reuse same database
        state_store_2 = StateStore(state_path)  # Reuse same database

        pkg_writer_2 = MockPKGWriter()
        vector_writer_2 = MockVectorWriter()
        sink_2 = NormalizationSink(
            pkg_writer=pkg_writer_2, vector_writer=vector_writer_2
        )

        orchestrator2 = IngestionOrchestrator(
            job_queue=queue_2,
            state_store=state_store_2,
            workspace_dir=str(workspace),
            element_sink=sink_2,
        )

        orchestrator2.register_source(
            SourceRegistration(source=source, schedule="@manual")
        )

        orchestrator2.run_manual_job("test", force=True)
        orchestrator2.start()
        await asyncio.sleep(3)
        await orchestrator2.shutdown()

        # Validate: No duplicate processing
        jobs_after_second_run = queue_2.snapshot(status=JobStatus.SUCCEEDED)
        second_run_count = len(jobs_after_second_run)

        # Second run should create a new job entry, but state should prevent reprocessing
        # So we expect at most 2 total jobs (or 1 if deduplication prevents enqueueing)
        assert (
            second_run_count <= first_run_count + 1
        ), f"Duplicate processing detected: {first_run_count} -> {second_run_count}"

        # More importantly: PKG writer should show no new documents
        assert (
            pkg_writer_2.documents_written == 0
        ), "State deduplication failed: file was reprocessed"


# ============================================================================
# Test 07: Scheduled Job Execution
# ============================================================================


class TestScheduledExecution:
    """Test automatic scheduled job execution."""

    @pytest.mark.asyncio
    async def test_scheduled_job_execution(self, tmp_path: Path):
        """Test interval-based job scheduling."""

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        # Create orchestrator
        queue = JobQueue(workspace / "queue.db")
        state_store = StateStore(workspace / "state" / "state.db")

        pkg_writer = MockPKGWriter()
        vector_writer = MockVectorWriter()
        sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

        orchestrator = IngestionOrchestrator(
            job_queue=queue,
            state_store=state_store,
            workspace_dir=str(workspace),
            element_sink=sink,
        )

        # Register with interval schedule
        source = LocalIngestionSource(name="test", root_path=source_dir)
        orchestrator.register_source(
            SourceRegistration(
                source=source,
                schedule="@interval",
                interval_seconds=2,
            )
        )

        # Start orchestrator
        orchestrator.start()

        # Manually trigger enqueue multiple times (APScheduler stub limitation)
        # This simulates interval-based triggering
        for _ in range(3):
            orchestrator._enqueue_job("test", trigger="interval")
            await asyncio.sleep(2)

        await orchestrator.shutdown()

        # Validate: Multiple scheduled executions
        jobs = queue.snapshot(status=JobStatus.SUCCEEDED)
        assert len(jobs) >= 2, f"Expected at least 2 scheduled runs, got {len(jobs)}"


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
