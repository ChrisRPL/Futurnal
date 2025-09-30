"""Integration tests for large Obsidian vault performance.

Tests system performance and scalability with large vaults
containing thousands of files.
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
from futurnal.ingestion.obsidian.sync_engine import create_sync_engine
from futurnal.ingestion.obsidian.change_detector import create_change_detector
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.file_watcher import create_optimized_watcher, WatcherConfig
from futurnal.privacy.audit import AuditLogger


class TestObsidianLargeVaults:
    """Test performance with large vault scenarios."""

    @pytest.fixture
    async def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir(parents=True)
            yield workspace

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

    async def create_large_vault(self, base_path: Path, num_files: int = 1000) -> Path:
        """Create a large test vault with interconnected notes."""
        vault_dir = base_path / "large_vault"
        vault_dir.mkdir()

        # Create directory structure
        subdirs = [
            "notes",
            "notes/daily",
            "notes/projects",
            "notes/research",
            "assets",
            "templates",
            "archive"
        ]

        for subdir in subdirs:
            (vault_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Distribution of files across directories
        files_per_dir = {
            "notes": int(num_files * 0.4),
            "notes/daily": int(num_files * 0.2),
            "notes/projects": int(num_files * 0.15),
            "notes/research": int(num_files * 0.1),
            "templates": int(num_files * 0.05),
            "archive": int(num_files * 0.1)
        }

        file_count = 0

        for dir_path, count in files_per_dir.items():
            dir_full_path = vault_dir / dir_path

            for i in range(count):
                file_path = dir_full_path / f"note_{file_count:05d}.md"

                # Create realistic content with links
                content = self._generate_note_content(file_count, num_files)
                file_path.write_text(content)

                file_count += 1
                if file_count >= num_files:
                    break

            if file_count >= num_files:
                break

        # Create some asset files
        assets_dir = vault_dir / "assets"
        for i in range(min(50, num_files // 20)):  # 5% assets
            asset_path = assets_dir / f"image_{i:03d}.png"
            # Create small dummy PNG files
            asset_path.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82')

        return vault_dir

    def _generate_note_content(self, file_index: int, total_files: int) -> str:
        """Generate realistic note content with links."""
        content = f"# Note {file_index:05d}\n\n"

        # Add some text content
        content += f"This is note number {file_index} in a large vault test.\n\n"

        # Add some links to other notes (create interconnected graph)
        num_links = min(5, total_files // 100)  # 1% of vault size, max 5 links
        if num_links > 0:
            content += "## Related Notes\n\n"
            for i in range(num_links):
                target_index = (file_index + i * 127) % total_files  # Pseudo-random distribution
                content += f"- [[note_{target_index:05d}]]\n"

        # Add some tags
        tags = ["#large-vault", f"#batch-{file_index // 100}", "#test"]
        content += f"\n\nTags: {' '.join(tags)}\n"

        # Add timestamp
        content += f"\nCreated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

        return content

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_vault_initial_scan(self, temp_workspace, state_store):
        """Test initial scan performance with large vault."""
        # Create large vault (scale down for CI)
        num_files = 500  # Reduced for CI performance
        vault_path = await self.create_large_vault(temp_workspace, num_files)

        # Create vault source
        vault_source = ObsidianVaultSource(
            name="large_vault_test",
            root_path=vault_path,
            vault_id="large_vault",
            vault_name="Large Vault Test",
            include=["**/*.md"],
            exclude=["**/.obsidian/**", "**/templates/**"]
        )

        # Create connector
        connector = ObsidianVaultConnector(
            workspace_dir=temp_workspace,
            state_store=state_store,
            enable_sync_engine=False  # Disable for pure scan test
        )

        # Measure initial scan performance
        start_time = time.time()

        records = connector.crawl_source(vault_source)

        scan_time = time.time() - start_time

        # Verify results
        assert len(records) > num_files * 0.8  # Should find most files (excluding templates)

        # Performance target: should scan 500 files in under 10 seconds
        assert scan_time < 10.0

        # Calculate files per second
        files_per_second = len(records) / scan_time
        assert files_per_second > 20  # Should process at least 20 files/second

        print(f"Scanned {len(records)} files in {scan_time:.2f}s ({files_per_second:.1f} files/s)")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_vault_change_detection(self, temp_workspace, state_store):
        """Test change detection performance with large vault."""
        num_files = 300  # Reduced for CI
        vault_path = await self.create_large_vault(temp_workspace, num_files)

        # Create change detector
        change_detector = create_change_detector(
            vault_id="large_vault",
            vault_root=vault_path,
            state_store=state_store
        )

        # Populate initial state
        initial_records = []
        for md_file in vault_path.rglob("*.md"):
            record = FileRecord(
                path=md_file,
                size=md_file.stat().st_size,
                mtime=md_file.stat().st_mtime,
                sha256=compute_sha256(md_file)
            )
            state_store.upsert(record)
            initial_records.append(record)

        # Modify some files
        num_modifications = 50
        modified_files = initial_records[:num_modifications]

        for i, record in enumerate(modified_files):
            # Modify file content
            original_content = record.path.read_text()
            modified_content = original_content + f"\n\nModification {i} added at {time.time()}"
            record.path.write_text(modified_content)

        # Create new records for modified files
        new_records = []
        for record in modified_files:
            new_record = FileRecord(
                path=record.path,
                size=record.path.stat().st_size,
                mtime=record.path.stat().st_mtime,
                sha256=compute_sha256(record.path)
            )
            new_records.append(new_record)

        # Measure change detection performance
        start_time = time.time()

        path_changes, content_changes = change_detector.detect_changes(new_records)

        detection_time = time.time() - start_time

        # Verify results
        assert len(content_changes) == num_modifications

        # Performance target: should detect changes in under 5 seconds
        assert detection_time < 5.0

        print(f"Detected {len(content_changes)} changes in {detection_time:.2f}s")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_vault_sync_engine_performance(self, temp_workspace, state_store, job_queue):
        """Test sync engine performance with large vault."""
        num_files = 200  # Reduced for CI
        vault_path = await self.create_large_vault(temp_workspace, num_files)

        # Create connector and sync engine
        connector = ObsidianVaultConnector(
            workspace_dir=temp_workspace,
            state_store=state_store,
            enable_sync_engine=True
        )

        sync_engine = create_sync_engine(
            vault_connector=connector,
            job_queue=job_queue,
            state_store=state_store,
            batch_window_seconds=0.5,
            max_batch_size=100,
            performance_monitoring=True
        )

        await sync_engine.start()

        try:
            # Track job creation
            enqueued_jobs = []
            original_enqueue = job_queue.enqueue
            job_queue.enqueue = lambda job: enqueued_jobs.append(job)

            # Simulate events for many files
            vault_id = "large_vault"
            start_time = time.time()

            # Create events for subset of files
            event_tasks = []
            for i in range(num_files // 4):  # Events for 25% of files
                file_path = vault_path / "notes" / f"note_{i:05d}.md"
                task = sync_engine.handle_file_event(
                    event_type=sync_engine.SyncEventType.FILE_MODIFIED,
                    file_path=file_path,
                    vault_id=vault_id,
                    metadata={"batch_test": True}
                )
                event_tasks.append(task)

            await asyncio.gather(*event_tasks)

            # Wait for batch processing
            await asyncio.sleep(2.0)

            processing_time = time.time() - start_time

            # Verify performance
            assert len(enqueued_jobs) > 0
            total_events = sum(len(job.payload.get("events", [])) for job in enqueued_jobs)
            assert total_events > 0

            # Should process efficiently
            assert processing_time < 10.0

            # Check metrics
            status = await sync_engine.get_sync_status(vault_id)
            if sync_engine._performance_monitoring and "metrics" in status:
                metrics = status["metrics"]
                assert metrics.get("events_processed", 0) > 0

            print(f"Processed {total_events} events in {processing_time:.2f}s")

            # Restore original enqueue
            job_queue.enqueue = original_enqueue

        finally:
            await sync_engine.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_vault_file_watching(self, temp_workspace):
        """Test file watching performance with large vault."""
        num_files = 150  # Reduced for CI
        vault_path = await self.create_large_vault(temp_workspace, num_files)

        # Configure file watcher for large vault
        config = WatcherConfig(
            include_patterns=["**/*.md"],
            exclude_patterns=["**/.obsidian/**", "**/templates/**"],
            enable_large_vault_mode=True,
            large_vault_threshold=100,
            large_vault_sample_rate=0.5,  # Sample 50% of events
            debounce_window_seconds=0.1,
            batch_window_seconds=0.5,
            max_events_per_batch=50
        )

        events_received = []

        def event_callback(events):
            events_received.extend(events)

        watcher = create_optimized_watcher(config, event_callback)

        try:
            # Start watching
            await watcher.start_watching(vault_path)

            # Wait for initial setup
            await asyncio.sleep(0.5)

            # Modify multiple files
            start_time = time.time()
            num_modifications = 30

            modification_files = list(vault_path.rglob("*.md"))[:num_modifications]

            for i, file_path in enumerate(modification_files):
                # Modify file to trigger file system event
                content = file_path.read_text()
                file_path.write_text(content + f"\n<!-- Modified {i} at {time.time()} -->")

                # Small delay to spread out events
                await asyncio.sleep(0.02)

            # Wait for event processing
            await asyncio.sleep(2.0)

            processing_time = time.time() - start_time

            # Verify events were received and processed efficiently
            # Note: may receive fewer events due to sampling
            assert len(events_received) > 0

            # Should process efficiently even with many files
            assert processing_time < 5.0

            # Get watcher stats
            stats = watcher.get_stats()
            assert stats["large_vault_mode"] is True

            print(f"Received {len(events_received)} events in {processing_time:.2f}s")

        finally:
            await watcher.stop_watching()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency_large_vault(self, temp_workspace, state_store, job_queue):
        """Test memory efficiency with large vault processing."""
        num_files = 200  # Reduced for CI
        vault_path = await self.create_large_vault(temp_workspace, num_files)

        # Create components
        connector = ObsidianVaultConnector(
            workspace_dir=temp_workspace,
            state_store=state_store,
            enable_sync_engine=True
        )

        sync_engine = create_sync_engine(
            vault_connector=connector,
            job_queue=job_queue,
            state_store=state_store,
            performance_monitoring=True
        )

        await sync_engine.start()

        try:
            # Process events in waves to test memory cleanup
            vault_id = "large_vault"
            waves = 5
            events_per_wave = 20

            for wave in range(waves):
                wave_start = time.time()

                # Create events for this wave
                tasks = []
                for i in range(events_per_wave):
                    file_index = wave * events_per_wave + i
                    file_path = vault_path / "notes" / f"note_{file_index:05d}.md"

                    task = sync_engine.handle_file_event(
                        event_type=sync_engine.SyncEventType.FILE_MODIFIED,
                        file_path=file_path,
                        vault_id=vault_id,
                        metadata={"wave": wave, "file": i}
                    )
                    tasks.append(task)

                await asyncio.gather(*tasks)

                # Wait for processing and cleanup
                await asyncio.sleep(1.0)

                wave_time = time.time() - wave_start
                print(f"Wave {wave + 1}/{waves} completed in {wave_time:.2f}s")

            # Final processing wait
            await asyncio.sleep(1.0)

            # System should still be responsive
            final_task = sync_engine.handle_file_event(
                event_type=sync_engine.SyncEventType.FILE_CREATED,
                file_path=vault_path / "final_memory_test.md",
                vault_id=vault_id
            )

            # Should complete without hanging (memory issues would cause timeouts)
            await asyncio.wait_for(final_task, timeout=2.0)

            # Check final status
            status = await sync_engine.get_sync_status(vault_id)
            assert status["engine_running"] is True

        finally:
            await sync_engine.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_large_vault_operations(self, temp_workspace, state_store, job_queue):
        """Test concurrent operations across multiple large vaults."""
        # Create multiple smaller vaults
        vault_configs = [
            ("vault_1", 100),
            ("vault_2", 100),
            ("vault_3", 100)
        ]

        vault_paths = {}
        for vault_name, num_files in vault_configs:
            vault_dir = temp_workspace / vault_name
            vault_path = await self.create_large_vault(vault_dir.parent, num_files)
            vault_path = vault_path.rename(vault_dir)
            vault_paths[vault_name] = vault_path

        # Create sync engine
        connector = ObsidianVaultConnector(
            workspace_dir=temp_workspace,
            state_store=state_store,
            enable_sync_engine=True
        )

        sync_engine = create_sync_engine(
            vault_connector=connector,
            job_queue=job_queue,
            state_store=state_store,
            batch_window_seconds=0.3,
            max_batch_size=50
        )

        await sync_engine.start()

        try:
            # Create concurrent events across all vaults
            start_time = time.time()

            all_tasks = []
            for vault_name, vault_path in vault_paths.items():
                # Create events for each vault
                for i in range(20):  # 20 events per vault
                    file_path = vault_path / "notes" / f"note_{i:05d}.md"
                    task = sync_engine.handle_file_event(
                        event_type=sync_engine.SyncEventType.FILE_MODIFIED,
                        file_path=file_path,
                        vault_id=vault_name,
                        metadata={"vault": vault_name, "concurrent_test": True}
                    )
                    all_tasks.append(task)

            # Execute all tasks concurrently
            await asyncio.gather(*all_tasks)

            # Wait for batch processing
            await asyncio.sleep(2.0)

            processing_time = time.time() - start_time

            # Should handle concurrent vaults efficiently
            assert processing_time < 8.0

            # Check status for all vaults
            for vault_name in vault_paths.keys():
                status = await sync_engine.get_sync_status(vault_name)
                assert status["vault_id"] == vault_name

            print(f"Processed events for {len(vault_paths)} vaults in {processing_time:.2f}s")

        finally:
            await sync_engine.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_vault_stress_test(self, temp_workspace, state_store, job_queue):
        """Stress test with sustained load on large vault."""
        num_files = 100  # Reduced for CI
        vault_path = await self.create_large_vault(temp_workspace, num_files)

        # Create sync engine with aggressive settings
        connector = ObsidianVaultConnector(
            workspace_dir=temp_workspace,
            state_store=state_store,
            enable_sync_engine=True
        )

        sync_engine = create_sync_engine(
            vault_connector=connector,
            job_queue=job_queue,
            state_store=state_store,
            batch_window_seconds=0.1,  # Very fast batching
            max_batch_size=20,
            performance_monitoring=True
        )

        await sync_engine.start()

        try:
            vault_id = "large_vault"

            # Sustained stress test
            duration_seconds = 10  # 10 second stress test
            events_per_second = 10
            total_events = duration_seconds * events_per_second

            start_time = time.time()
            events_sent = 0

            while events_sent < total_events:
                wave_start = time.time()
                wave_tasks = []

                # Send events for this second
                for _ in range(events_per_second):
                    if events_sent >= total_events:
                        break

                    file_index = events_sent % num_files
                    file_path = vault_path / "notes" / f"note_{file_index:05d}.md"

                    task = sync_engine.handle_file_event(
                        event_type=sync_engine.SyncEventType.FILE_MODIFIED,
                        file_path=file_path,
                        vault_id=vault_id,
                        metadata={"stress_test": True, "event_id": events_sent}
                    )
                    wave_tasks.append(task)
                    events_sent += 1

                # Execute wave
                await asyncio.gather(*wave_tasks)

                # Maintain timing
                wave_time = time.time() - wave_start
                if wave_time < 1.0:
                    await asyncio.sleep(1.0 - wave_time)

            total_time = time.time() - start_time

            # Wait for final processing
            await asyncio.sleep(2.0)

            # System should survive stress test
            assert sync_engine._running is True

            # Should maintain reasonable performance
            assert total_time < duration_seconds + 5  # Allow some overhead

            # Check final status
            status = await sync_engine.get_sync_status(vault_id)
            if sync_engine._performance_monitoring and "metrics" in status:
                metrics = status["metrics"]
                print(f"Stress test metrics: {metrics}")

            print(f"Stress test: {events_sent} events in {total_time:.2f}s")

        finally:
            await sync_engine.stop()


@pytest.mark.performance
class TestLargeVaultBenchmarks:
    """Benchmark tests for large vault operations."""

    @pytest.mark.asyncio
    async def test_scan_performance_benchmark(self, temp_workspace):
        """Benchmark file scanning performance."""
        # Test with different vault sizes
        vault_sizes = [100, 250, 500]

        for size in vault_sizes:
            # Create vault
            large_vault_test = TestObsidianLargeVaults()
            vault_path = await large_vault_test.create_large_vault(temp_workspace / f"bench_{size}", size)

            # Create state store
            state_store = StateStore(temp_workspace / f"bench_{size}_state.db")

            # Create vault source
            vault_source = ObsidianVaultSource(
                name=f"benchmark_vault_{size}",
                root_path=vault_path,
                vault_id=f"benchmark_{size}",
                include=["**/*.md"]
            )

            # Create connector
            connector = ObsidianVaultConnector(
                workspace_dir=temp_workspace / f"bench_{size}_workspace",
                state_store=state_store,
                enable_sync_engine=False
            )

            # Benchmark scan
            start_time = time.time()
            records = connector.crawl_source(vault_source)
            scan_time = time.time() - start_time

            files_per_second = len(records) / scan_time

            print(f"Vault size {size}: {len(records)} files scanned in {scan_time:.2f}s ({files_per_second:.1f} files/s)")

            # Performance assertions
            assert files_per_second > 10  # Minimum 10 files/second
            assert scan_time < size / 5   # Should scan faster than 5 files/second minimum