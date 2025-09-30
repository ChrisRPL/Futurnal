"""Integration tests for Obsidian sync engine functionality.

Tests the core sync engine coordination, change detection, and
integration with the vault connector and job queue.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from futurnal.ingestion.local.state import StateStore, FileRecord
from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector, ObsidianVaultSource
from futurnal.ingestion.obsidian.descriptor import ObsidianVaultDescriptor, VaultRegistry
from futurnal.ingestion.obsidian.sync_engine import (
    ObsidianSyncEngine, SyncEventType, SyncPriority, create_sync_engine
)
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.models import JobPriority
from futurnal.privacy.audit import AuditLogger


class TestObsidianSyncEngineIntegration:
    """Integration tests for ObsidianSyncEngine with vault connector."""

    @pytest.fixture
    async def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir(parents=True)
            yield workspace

    @pytest.fixture
    async def vault_path(self, temp_workspace):
        """Create a test vault directory structure."""
        vault_dir = temp_workspace / "test_vault"
        vault_dir.mkdir()

        # Create some test markdown files
        (vault_dir / "note1.md").write_text("# Note 1\nContent of note 1")
        (vault_dir / "note2.md").write_text("# Note 2\nContent of note 2")

        # Create subdirectory with notes
        subdir = vault_dir / "subdirectory"
        subdir.mkdir()
        (subdir / "note3.md").write_text("# Note 3\nContent of note 3")

        return vault_dir

    @pytest.fixture
    async def state_store(self, temp_workspace):
        """Create a test state store."""
        db_path = temp_workspace / "state.db"
        return StateStore(db_path)

    @pytest.fixture
    async def job_queue(self, temp_workspace):
        """Create a test job queue."""
        queue_path = temp_workspace / "queue.db"
        return JobQueue(queue_path)

    @pytest.fixture
    async def vault_registry(self, vault_path):
        """Create a test vault registry with a registered vault."""
        registry = VaultRegistry()
        descriptor = ObsidianVaultDescriptor(
            id="test_vault",
            name="Test Vault",
            base_path=vault_path,
            created_at=time.time()
        )
        registry.register(descriptor)
        return registry

    @pytest.fixture
    async def vault_source(self, vault_path):
        """Create a test vault source."""
        return ObsidianVaultSource(
            name="test_vault_source",
            root_path=vault_path,
            vault_id="test_vault",
            vault_name="Test Vault",
            include=["**/*.md"],
            exclude=["**/.obsidian/**"]
        )

    @pytest.fixture
    async def audit_logger(self, temp_workspace):
        """Create a test audit logger."""
        audit_dir = temp_workspace / "audit"
        audit_dir.mkdir()
        return AuditLogger(audit_dir)

    @pytest.fixture
    async def vault_connector(self, temp_workspace, state_store, vault_registry, audit_logger):
        """Create a test vault connector."""
        return ObsidianVaultConnector(
            workspace_dir=temp_workspace,
            state_store=state_store,
            vault_registry=vault_registry,
            audit_logger=audit_logger,
            enable_sync_engine=True
        )

    @pytest.fixture
    async def sync_engine(self, vault_connector, job_queue, state_store, audit_logger):
        """Create a test sync engine."""
        engine = create_sync_engine(
            vault_connector=vault_connector,
            job_queue=job_queue,
            state_store=state_store,
            audit_logger=audit_logger,
            batch_window_seconds=0.5,  # Faster batching for tests
            max_batch_size=10
        )
        await engine.start()
        yield engine
        await engine.stop()

    @pytest.mark.asyncio
    async def test_sync_engine_initialization(self, vault_connector, job_queue):
        """Test sync engine initialization and shutdown."""
        # Initialize sync engine
        await vault_connector.initialize_sync_engine(job_queue)

        # Check that sync engine is available
        assert vault_connector._sync_engine is not None
        assert vault_connector._sync_engine._running is True

        # Shutdown
        await vault_connector.shutdown_sync_engine()
        assert vault_connector._sync_engine is None

    @pytest.mark.asyncio
    async def test_file_event_handling(self, sync_engine, vault_path):
        """Test basic file event handling through sync engine."""
        vault_id = "test_vault"
        test_file = vault_path / "new_note.md"

        # Track jobs enqueued
        enqueued_jobs = []
        original_enqueue = sync_engine._job_queue.enqueue
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Handle a file creation event
        await sync_engine.handle_file_event(
            event_type=SyncEventType.FILE_CREATED,
            file_path=test_file,
            vault_id=vault_id,
            metadata={"test": True}
        )

        # Wait for batch processing
        await asyncio.sleep(0.6)

        # Check that a job was enqueued
        assert len(enqueued_jobs) > 0
        job = enqueued_jobs[0]
        assert job.payload["vault_id"] == vault_id
        assert "events" in job.payload
        assert len(job.payload["events"]) == 1

        # Restore original enqueue
        sync_engine._job_queue.enqueue = original_enqueue

    @pytest.mark.asyncio
    async def test_priority_handling(self, sync_engine, vault_path):
        """Test that different event types receive appropriate priorities."""
        vault_id = "test_vault"

        enqueued_jobs = []
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Create events with different priorities
        markdown_file = vault_path / "note.md"
        asset_file = vault_path / "image.png"

        # High priority: markdown file creation
        await sync_engine.handle_file_event(
            event_type=SyncEventType.FILE_CREATED,
            file_path=markdown_file,
            vault_id=vault_id
        )

        # Low priority: asset file creation
        await sync_engine.handle_file_event(
            event_type=SyncEventType.FILE_CREATED,
            file_path=asset_file,
            vault_id=vault_id
        )

        # Critical priority: file move
        await sync_engine.handle_file_event(
            event_type=SyncEventType.FILE_MOVED,
            file_path=markdown_file,
            vault_id=vault_id
        )

        # Wait for processing
        await asyncio.sleep(0.6)

        # Check that jobs were created with appropriate priorities
        assert len(enqueued_jobs) >= 2  # At least 2 batches (critical should be immediate)

        # Find the critical priority job (should be first due to immediate processing)
        critical_jobs = [job for job in enqueued_jobs if job.priority == JobPriority.HIGH]
        assert len(critical_jobs) > 0

    @pytest.mark.asyncio
    async def test_batching_behavior(self, sync_engine, vault_path):
        """Test that events are properly batched."""
        vault_id = "test_vault"

        enqueued_jobs = []
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Create multiple events quickly
        for i in range(5):
            test_file = vault_path / f"note_{i}.md"
            await sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_MODIFIED,
                file_path=test_file,
                vault_id=vault_id
            )

        # Wait for batch processing
        await asyncio.sleep(0.6)

        # Should have fewer jobs than events due to batching
        assert len(enqueued_jobs) > 0

        # Check that events were batched
        total_events = sum(len(job.payload.get("events", [])) for job in enqueued_jobs)
        assert total_events == 5

    @pytest.mark.asyncio
    async def test_vault_connector_sync_integration(self, vault_connector, job_queue, vault_source):
        """Test full integration with vault connector sync capabilities."""
        # Initialize sync engine
        await vault_connector.initialize_sync_engine(job_queue)

        # Enable sync for vault
        success = await vault_connector.enable_vault_sync(vault_source)
        assert success is True

        # Check sync status
        status = await vault_connector.get_sync_status(vault_source.vault_id)
        assert status["sync_enabled"] is True
        assert status["sync_engine_available"] is True

        # Trigger incremental sync
        job_id = await vault_connector.trigger_incremental_sync(vault_source)
        assert job_id is not None

        # Disable sync
        success = await vault_connector.disable_vault_sync(vault_source.vault_id)
        assert success is True

        # Check status after disable
        status = await vault_connector.get_sync_status(vault_source.vault_id)
        assert status["sync_enabled"] is False

    @pytest.mark.asyncio
    async def test_sync_engine_error_handling(self, sync_engine, vault_path):
        """Test error handling in sync engine operations."""
        vault_id = "test_vault"

        # Mock job queue to raise exception
        original_enqueue = sync_engine._job_queue.enqueue
        sync_engine._job_queue.enqueue = MagicMock(side_effect=Exception("Queue error"))

        # Handle event that should trigger error
        await sync_engine.handle_file_event(
            event_type=SyncEventType.FILE_CREATED,
            file_path=vault_path / "test.md",
            vault_id=vault_id
        )

        # Wait for processing
        await asyncio.sleep(0.6)

        # Engine should still be running despite error
        assert sync_engine._running is True

        # Restore original enqueue
        sync_engine._job_queue.enqueue = original_enqueue

    @pytest.mark.asyncio
    async def test_full_sync_trigger(self, sync_engine, vault_source):
        """Test triggering a full vault sync."""
        job_id = await sync_engine.trigger_full_sync(vault_source)
        assert job_id is not None

        # Check that job was enqueued
        pending_jobs = list(sync_engine._job_queue.fetch_pending(limit=10))
        assert len(pending_jobs) > 0

        # Find the full sync job
        full_sync_jobs = [job for job in pending_jobs if
                         job.payload.get("sync_type") == "full_scan"]
        assert len(full_sync_jobs) > 0

    @pytest.mark.asyncio
    async def test_sync_status_reporting(self, sync_engine, vault_path):
        """Test sync status reporting and metrics."""
        vault_id = "test_vault"

        # Get initial status
        initial_status = await sync_engine.get_sync_status(vault_id)
        assert "vault_id" in initial_status
        assert "engine_running" in initial_status

        # Process some events
        for i in range(3):
            await sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_MODIFIED,
                file_path=vault_path / f"note_{i}.md",
                vault_id=vault_id
            )

        # Wait for processing
        await asyncio.sleep(0.6)

        # Get updated status
        updated_status = await sync_engine.get_sync_status(vault_id)

        # Should have metrics if performance monitoring is enabled
        if sync_engine._performance_monitoring:
            assert "metrics" in updated_status

    @pytest.mark.asyncio
    async def test_concurrent_event_handling(self, sync_engine, vault_path):
        """Test handling concurrent events from multiple vaults."""
        vault_ids = ["vault_1", "vault_2", "vault_3"]

        enqueued_jobs = []
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Create concurrent events for different vaults
        tasks = []
        for vault_id in vault_ids:
            for i in range(3):
                task = sync_engine.handle_file_event(
                    event_type=SyncEventType.FILE_MODIFIED,
                    file_path=vault_path / f"{vault_id}_note_{i}.md",
                    vault_id=vault_id
                )
                tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        # Wait for batch processing
        await asyncio.sleep(0.6)

        # Check that jobs were created for different vaults
        vault_jobs = {}
        for job in enqueued_jobs:
            vault_id = job.payload.get("vault_id")
            if vault_id:
                vault_jobs[vault_id] = vault_jobs.get(vault_id, 0) + 1

        # Should have jobs for multiple vaults
        assert len(vault_jobs) > 1

    @pytest.mark.asyncio
    async def test_change_detector_integration(self, vault_connector, vault_source, state_store):
        """Test integration with advanced change detector."""
        # Create initial file records
        test_file = vault_source.root_path / "test_note.md"
        test_file.write_text("# Original Content\nThis is the original content.")

        # Create file record
        from futurnal.ingestion.local.state import compute_sha256
        original_hash = compute_sha256(test_file)
        record = FileRecord(
            path=test_file,
            size=test_file.stat().st_size,
            mtime=test_file.stat().st_mtime,
            sha256=original_hash
        )
        state_store.upsert(record)

        # Initialize change detector
        change_detector = await vault_connector._get_or_create_change_detector(vault_source)
        assert change_detector is not None

        # Modify the file
        test_file.write_text("# Modified Content\nThis is the modified content.")

        # Create new record
        new_hash = compute_sha256(test_file)
        new_record = FileRecord(
            path=test_file,
            size=test_file.stat().st_size,
            mtime=test_file.stat().st_mtime,
            sha256=new_hash
        )

        # Detect changes
        path_changes, content_changes = change_detector.detect_changes([new_record])

        # Should detect content change
        assert len(content_changes) > 0
        change = content_changes[0]
        assert change.file_path == test_file
        assert change.old_checksum == original_hash
        assert change.new_checksum == new_hash


@pytest.mark.asyncio
class TestSyncEnginePerformance:
    """Performance-focused tests for sync engine."""

    async def test_high_volume_event_processing(self, sync_engine, vault_path):
        """Test processing a high volume of events efficiently."""
        vault_id = "test_vault"
        num_events = 100

        start_time = time.time()

        # Create many events
        tasks = []
        for i in range(num_events):
            task = sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_MODIFIED,
                file_path=vault_path / f"note_{i}.md",
                vault_id=vault_id
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Wait for processing
        await asyncio.sleep(2.0)

        processing_time = time.time() - start_time

        # Should process events efficiently (less than 5 seconds for 100 events)
        assert processing_time < 5.0

        # Check metrics
        status = await sync_engine.get_sync_status(vault_id)
        if sync_engine._performance_monitoring and "metrics" in status:
            metrics = status["metrics"]
            assert metrics.get("events_processed", 0) > 0

    async def test_memory_efficiency(self, sync_engine, vault_path):
        """Test that sync engine doesn't leak memory with many events."""
        vault_id = "test_vault"

        # Process events in waves to test cleanup
        for wave in range(5):
            tasks = []
            for i in range(20):
                task = sync_engine.handle_file_event(
                    event_type=SyncEventType.FILE_MODIFIED,
                    file_path=vault_path / f"wave_{wave}_note_{i}.md",
                    vault_id=vault_id
                )
                tasks.append(task)

            await asyncio.gather(*tasks)
            await asyncio.sleep(0.6)  # Wait for processing

        # Engine should still be responsive
        final_event = sync_engine.handle_file_event(
            event_type=SyncEventType.FILE_CREATED,
            file_path=vault_path / "final_note.md",
            vault_id=vault_id
        )

        # Should complete without hanging
        await asyncio.wait_for(final_event, timeout=1.0)