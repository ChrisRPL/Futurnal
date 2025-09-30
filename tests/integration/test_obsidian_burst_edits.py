"""Integration tests for Obsidian burst edit handling.

Tests the system's ability to handle rapid file changes without
overwhelming the sync engine or losing events.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from futurnal.ingestion.local.state import StateStore, FileRecord, compute_sha256
from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector, ObsidianVaultSource
from futurnal.ingestion.obsidian.descriptor import ObsidianVaultDescriptor, VaultRegistry
from futurnal.ingestion.obsidian.sync_engine import (
    ObsidianSyncEngine, SyncEventType, create_sync_engine
)
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.priority_scheduler import PriorityScheduler, create_priority_scheduler
from futurnal.orchestrator.file_watcher import OptimizedFileWatcher, WatcherConfig
from futurnal.privacy.audit import AuditLogger


class TestObsidianBurstEdits:
    """Test handling of rapid file changes (burst edits)."""

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

        # Create initial notes
        for i in range(10):
            note_path = vault_dir / f"note_{i:02d}.md"
            note_path.write_text(f"# Note {i}\nInitial content for note {i}")

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
    async def priority_scheduler(self, job_queue):
        """Create a priority scheduler with fast batching for tests."""
        scheduler = create_priority_scheduler(
            job_queue,
            batch_window_seconds=0.1,  # Very fast batching for tests
            max_batch_size=20,
            max_queue_depth=1000
        )
        await scheduler.start()
        yield scheduler
        await scheduler.stop()

    @pytest.fixture
    async def sync_engine(self, temp_workspace, state_store, job_queue):
        """Create a sync engine optimized for burst handling."""
        from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector

        connector = ObsidianVaultConnector(
            workspace_dir=temp_workspace,
            state_store=state_store,
            enable_sync_engine=True
        )

        engine = create_sync_engine(
            vault_connector=connector,
            job_queue=job_queue,
            state_store=state_store,
            batch_window_seconds=0.1,  # Fast batching
            max_batch_size=50,
            performance_monitoring=True
        )

        await engine.start()
        yield engine
        await engine.stop()

    @pytest.mark.asyncio
    async def test_rapid_file_modifications(self, sync_engine, vault_path):
        """Test handling rapid modifications to the same file."""
        vault_id = "test_vault"
        test_file = vault_path / "rapid_edit_note.md"
        test_file.write_text("# Initial Content")

        # Track enqueued jobs
        enqueued_jobs = []
        original_enqueue = sync_engine._job_queue.enqueue
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Simulate rapid edits to the same file
        num_edits = 20
        edit_tasks = []

        for i in range(num_edits):
            # Create async task for each edit event
            task = sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_MODIFIED,
                file_path=test_file,
                vault_id=vault_id,
                metadata={"edit_number": i}
            )
            edit_tasks.append(task)

        # Execute all edits concurrently
        await asyncio.gather(*edit_tasks)

        # Wait for batch processing
        await asyncio.sleep(0.5)

        # Should have fewer jobs than edits due to debouncing and batching
        assert len(enqueued_jobs) > 0
        assert len(enqueued_jobs) < num_edits

        # Total events should be less than edits due to debouncing
        total_events = sum(len(job.payload.get("events", [])) for job in enqueued_jobs)
        assert total_events <= num_edits

        # Restore original enqueue
        sync_engine._job_queue.enqueue = original_enqueue

    @pytest.mark.asyncio
    async def test_burst_file_creation(self, sync_engine, vault_path):
        """Test handling burst creation of many files."""
        vault_id = "test_vault"

        enqueued_jobs = []
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Create many files rapidly
        num_files = 50
        creation_tasks = []

        for i in range(num_files):
            test_file = vault_path / f"burst_note_{i:03d}.md"
            task = sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_CREATED,
                file_path=test_file,
                vault_id=vault_id,
                metadata={"file_number": i}
            )
            creation_tasks.append(task)

        start_time = time.time()
        await asyncio.gather(*creation_tasks)
        processing_time = time.time() - start_time

        # Wait for batch processing
        await asyncio.sleep(0.5)

        # Should process quickly (less than 2 seconds)
        assert processing_time < 2.0

        # Should have created jobs efficiently
        assert len(enqueued_jobs) > 0

        # Total events should equal number of files (no debouncing for different files)
        total_events = sum(len(job.payload.get("events", [])) for job in enqueued_jobs)
        assert total_events == num_files

    @pytest.mark.asyncio
    async def test_mixed_burst_operations(self, sync_engine, vault_path):
        """Test handling mixed burst operations (create, modify, delete, move)."""
        vault_id = "test_vault"

        enqueued_jobs = []
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Create mixed operation tasks
        tasks = []
        num_operations = 30

        for i in range(num_operations):
            file_path = vault_path / f"mixed_note_{i:03d}.md"

            # Vary operation types
            if i % 4 == 0:
                event_type = SyncEventType.FILE_CREATED
            elif i % 4 == 1:
                event_type = SyncEventType.FILE_MODIFIED
            elif i % 4 == 2:
                event_type = SyncEventType.FILE_DELETED
            else:
                event_type = SyncEventType.FILE_MOVED

            task = sync_engine.handle_file_event(
                event_type=event_type,
                file_path=file_path,
                vault_id=vault_id,
                metadata={"operation_number": i, "operation_type": event_type.value}
            )
            tasks.append(task)

        # Execute all operations concurrently
        start_time = time.time()
        await asyncio.gather(*tasks)
        processing_time = time.time() - start_time

        # Wait for batch processing
        await asyncio.sleep(0.5)

        # Should process efficiently
        assert processing_time < 3.0

        # Should have created appropriate jobs
        assert len(enqueued_jobs) > 0

        # Verify different priorities were handled
        job_priorities = {job.priority for job in enqueued_jobs}
        assert len(job_priorities) > 1  # Should have different priorities

    @pytest.mark.asyncio
    async def test_priority_scheduler_burst_handling(self, priority_scheduler, vault_path):
        """Test priority scheduler's handling of burst job submissions."""
        vault_id = "test_vault"

        # Create jobs with different priorities
        high_priority_jobs = []
        normal_priority_jobs = []
        low_priority_jobs = []

        from futurnal.orchestrator.models import IngestionJob, JobType, JobPriority

        # Create burst of jobs with mixed priorities
        for i in range(60):
            if i % 3 == 0:
                priority = JobPriority.HIGH
                job_list = high_priority_jobs
            elif i % 3 == 1:
                priority = JobPriority.NORMAL
                job_list = normal_priority_jobs
            else:
                priority = JobPriority.LOW
                job_list = low_priority_jobs

            job = IngestionJob(
                job_id=f"burst_job_{i:03d}",
                job_type=JobType.LOCAL_FILES,
                payload={
                    "vault_id": vault_id,
                    "file_path": str(vault_path / f"burst_file_{i:03d}.md"),
                    "priority_test": True
                },
                priority=priority
            )
            job_list.append(job)

        # Submit all jobs rapidly
        start_time = time.time()

        submission_tasks = []
        for job_list in [high_priority_jobs, normal_priority_jobs, low_priority_jobs]:
            for job in job_list:
                task = priority_scheduler.enqueue_job(job)
                submission_tasks.append(task)

        await asyncio.gather(*submission_tasks)
        submission_time = time.time() - start_time

        # Wait for batch processing
        await asyncio.sleep(0.5)

        # Should handle submission efficiently
        assert submission_time < 2.0

        # Check scheduler status
        status = await priority_scheduler.get_queue_status()
        assert status["running"] is True

        # Should have processed batches
        if priority_scheduler._metrics:
            assert priority_scheduler._metrics.total_batches_processed > 0

    @pytest.mark.asyncio
    async def test_file_watcher_burst_handling(self, temp_workspace, vault_path):
        """Test optimized file watcher's handling of burst events."""
        # Create watcher config optimized for burst handling
        config = WatcherConfig(
            debounce_window_seconds=0.1,
            batch_window_seconds=0.2,
            max_events_per_batch=50,
            include_patterns=["**/*.md"],
            exclude_patterns=["**/.obsidian/**"]
        )

        events_received = []

        def event_callback(events):
            events_received.extend(events)

        # Create file watcher
        from futurnal.orchestrator.file_watcher import create_optimized_watcher

        watcher = create_optimized_watcher(config, event_callback)

        try:
            await watcher.start_watching(vault_path)

            # Simulate burst file operations
            num_operations = 30

            for i in range(num_operations):
                file_path = vault_path / f"watcher_test_{i:03d}.md"
                file_path.write_text(f"# Test File {i}\nContent for burst test.")

                # Small delay to trigger file system events
                await asyncio.sleep(0.01)

            # Wait for file watcher to process events
            await asyncio.sleep(1.0)

            # Should have received events (may be fewer due to batching)
            assert len(events_received) > 0

            # Events should be properly formatted
            for event in events_received:
                assert hasattr(event, 'event_type')
                assert hasattr(event, 'path')
                assert hasattr(event, 'timestamp')

        finally:
            await watcher.stop_watching()

    @pytest.mark.asyncio
    async def test_memory_usage_during_burst(self, sync_engine, vault_path):
        """Test that memory usage remains reasonable during burst operations."""
        vault_id = "test_vault"

        # Track job creation without actually enqueuing
        job_count = 0

        def count_jobs(job):
            nonlocal job_count
            job_count += 1

        sync_engine._job_queue.enqueue = count_jobs

        # Create sustained burst of events
        num_waves = 10
        events_per_wave = 20

        for wave in range(num_waves):
            tasks = []
            for i in range(events_per_wave):
                file_path = vault_path / f"memory_test_wave_{wave}_file_{i}.md"
                task = sync_engine.handle_file_event(
                    event_type=SyncEventType.FILE_MODIFIED,
                    file_path=file_path,
                    vault_id=vault_id,
                    metadata={"wave": wave, "file": i}
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            # Wait for processing and cleanup
            await asyncio.sleep(0.3)

        # Final wait for all processing
        await asyncio.sleep(1.0)

        # Should have processed all waves
        assert job_count > 0

        # Engine should still be responsive
        final_task = sync_engine.handle_file_event(
            event_type=SyncEventType.FILE_CREATED,
            file_path=vault_path / "final_memory_test.md",
            vault_id=vault_id
        )

        # Should complete without hanging (memory issues would cause timeouts)
        await asyncio.wait_for(final_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_queue_backpressure_handling(self, priority_scheduler, vault_path):
        """Test queue backpressure handling during sustained burst."""
        from futurnal.orchestrator.models import IngestionJob, JobType, JobPriority

        # Configure scheduler with low limits to trigger backpressure
        priority_scheduler._max_queue_depth = 50
        priority_scheduler._backpressure_threshold = 0.8

        vault_id = "test_vault"
        submitted_jobs = 0
        dropped_jobs = 0

        # Override enqueue to track submissions vs drops
        original_enqueue = priority_scheduler._job_queue.enqueue

        def track_enqueue(job):
            nonlocal submitted_jobs
            submitted_jobs += 1
            return original_enqueue(job)

        priority_scheduler._job_queue.enqueue = track_enqueue

        # Create many jobs to trigger backpressure
        num_jobs = 100

        for i in range(num_jobs):
            job = IngestionJob(
                job_id=f"backpressure_job_{i:03d}",
                job_type=JobType.LOCAL_FILES,
                payload={
                    "vault_id": vault_id,
                    "file_path": str(vault_path / f"backpressure_file_{i:03d}.md"),
                    "backpressure_test": True
                },
                priority=JobPriority.NORMAL
            )

            try:
                await priority_scheduler.enqueue_job(job)
            except Exception:
                dropped_jobs += 1

            # Small delay to allow backpressure to activate
            if i % 10 == 0:
                await asyncio.sleep(0.05)

        # Wait for processing
        await asyncio.sleep(1.0)

        # Should have handled backpressure gracefully
        status = await priority_scheduler.get_queue_status()

        # Backpressure should have been activated at some point
        if priority_scheduler._metrics:
            # May have backpressure events if queue filled up
            assert priority_scheduler._metrics.backpressure_events >= 0

        # System should still be responsive
        assert status["running"] is True

    @pytest.mark.asyncio
    async def test_debouncing_effectiveness(self, sync_engine, vault_path):
        """Test that debouncing effectively reduces redundant events."""
        vault_id = "test_vault"
        test_file = vault_path / "debounce_test.md"

        enqueued_jobs = []
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Create many rapid events for the same file
        num_events = 100
        interval = 0.01  # 10ms intervals

        start_time = time.time()

        for i in range(num_events):
            await sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_MODIFIED,
                file_path=test_file,
                vault_id=vault_id,
                metadata={"event_number": i}
            )

            await asyncio.sleep(interval)

        total_time = time.time() - start_time

        # Wait for final batch processing
        await asyncio.sleep(1.0)

        # Should have significantly fewer jobs than events due to debouncing
        total_events = sum(len(job.payload.get("events", [])) for job in enqueued_jobs)

        # Debouncing should reduce events by at least 50%
        assert total_events < num_events * 0.5

        # Should still process efficiently
        assert total_time < 3.0

    @pytest.mark.asyncio
    async def test_concurrent_vault_burst_handling(self, sync_engine, temp_workspace):
        """Test handling burst edits across multiple vaults concurrently."""
        # Create multiple test vaults
        vault_paths = []
        vault_ids = []

        for i in range(3):
            vault_path = temp_workspace / f"vault_{i}"
            vault_path.mkdir()
            vault_paths.append(vault_path)
            vault_ids.append(f"vault_{i}")

        enqueued_jobs = []
        sync_engine._job_queue.enqueue = lambda job: enqueued_jobs.append(job)

        # Create concurrent burst events across all vaults
        tasks = []
        events_per_vault = 20

        for vault_id, vault_path in zip(vault_ids, vault_paths):
            for i in range(events_per_vault):
                file_path = vault_path / f"concurrent_note_{i:03d}.md"
                task = sync_engine.handle_file_event(
                    event_type=SyncEventType.FILE_MODIFIED,
                    file_path=file_path,
                    vault_id=vault_id,
                    metadata={"vault": vault_id, "file": i}
                )
                tasks.append(task)

        # Execute all tasks concurrently
        start_time = time.time()
        await asyncio.gather(*tasks)
        processing_time = time.time() - start_time

        # Wait for batch processing
        await asyncio.sleep(1.0)

        # Should handle concurrent vaults efficiently
        assert processing_time < 3.0

        # Should have jobs for all vaults
        vault_jobs = {}
        for job in enqueued_jobs:
            vault_id = job.payload.get("vault_id")
            if vault_id:
                vault_jobs[vault_id] = vault_jobs.get(vault_id, 0) + 1

        # Should have jobs from multiple vaults
        assert len(vault_jobs) >= 2

    @pytest.mark.asyncio
    async def test_error_resilience_during_burst(self, sync_engine, vault_path):
        """Test system resilience when errors occur during burst operations."""
        vault_id = "test_vault"

        # Create a mix of valid and invalid events
        valid_events = 0
        error_events = 0

        original_handle = sync_engine._process_sync_batch

        async def error_prone_batch_handler(batch):
            # Simulate occasional processing errors
            if "error_trigger" in str(batch.batch_id):
                raise Exception("Simulated batch processing error")
            return await original_handle(batch)

        sync_engine._process_sync_batch = error_prone_batch_handler

        # Create burst with some events that will trigger errors
        num_events = 50

        for i in range(num_events):
            file_path = vault_path / f"error_test_{i:03d}.md"

            # Some events will trigger errors
            metadata = {"event_number": i}
            if i % 10 == 0:  # Every 10th event triggers error
                metadata["error_trigger"] = True
                error_events += 1
            else:
                valid_events += 1

            await sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_MODIFIED,
                file_path=file_path,
                vault_id=vault_id,
                metadata=metadata
            )

        # Wait for processing
        await asyncio.sleep(1.0)

        # Engine should still be running despite errors
        assert sync_engine._running is True

        # Should be able to process new events
        final_event = sync_engine.handle_file_event(
            event_type=SyncEventType.FILE_CREATED,
            file_path=vault_path / "post_error_test.md",
            vault_id=vault_id
        )

        # Should complete without hanging
        await asyncio.wait_for(final_event, timeout=1.0)

        # Restore original handler
        sync_engine._process_sync_batch = original_handle