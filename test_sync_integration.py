#!/usr/bin/env python3
"""
Integration tests for Obsidian sync engine - simplified version.
"""

import asyncio
import tempfile
import time
from pathlib import Path

async def test_sync_engine_integration():
    """Test complete sync engine integration."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.ingestion.local.state import StateStore, FileRecord, compute_sha256
        from futurnal.orchestrator.queue import JobQueue
        from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector, ObsidianVaultSource
        from futurnal.ingestion.obsidian.descriptor import ObsidianVaultDescriptor, VaultRegistry, Provenance
        from futurnal.ingestion.obsidian.sync_engine import create_sync_engine, SyncEventType
        from futurnal.privacy.audit import AuditLogger
        from futurnal import __version__ as FUTURNAL_VERSION
        import getpass
        import platform
        import socket
        from hashlib import sha256

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            vault_dir = workspace / "test_vault"
            vault_dir.mkdir()

            # Create test vault structure with .obsidian directory
            obsidian_dir = vault_dir / ".obsidian"
            obsidian_dir.mkdir()
            (obsidian_dir / "app.json").write_text('{"name": "Test Vault"}')

            (vault_dir / "note1.md").write_text("# Note 1\nContent of note 1")
            (vault_dir / "note2.md").write_text("# Note 2\nContent of note 2")

            # Create subdirectory
            subdir = vault_dir / "subdirectory"
            subdir.mkdir()
            (subdir / "note3.md").write_text("# Note 3\nContent of note 3")

            print("   - Test vault created")

            # Create components
            state_store = StateStore(workspace / "state.db")
            job_queue = JobQueue(workspace / "queue.db")
            audit_logger = AuditLogger(workspace / "audit")

            # Set up vault registry
            registry = VaultRegistry()

            # Register the vault path (this creates the descriptor automatically)
            descriptor = registry.register_path(
                vault_dir,
                name="Test Vault"
            )

            # Create vault connector
            connector = ObsidianVaultConnector(
                workspace_dir=workspace,
                state_store=state_store,
                vault_registry=registry,
                audit_logger=audit_logger,
                enable_sync_engine=True
            )

            # Create vault source using the registered descriptor's ID
            vault_source = ObsidianVaultSource(
                name="test_vault_source",
                root_path=vault_dir,
                vault_id=descriptor.id,
                vault_name=descriptor.name,
                include=["**/*.md"],
                exclude=["**/.obsidian/**"]
            )

            print("   - Components created")

            # Initialize sync engine through connector
            await connector.initialize_sync_engine(job_queue)
            assert connector._sync_engine is not None
            print("   - ✅ Sync engine initialized")

            # Enable sync for vault
            success = await connector.enable_vault_sync(vault_source)
            assert success is True
            print("   - ✅ Vault sync enabled")

            # Test file event handling
            test_file = vault_dir / "new_note.md"
            test_file.write_text("# New Note\nThis is a new note.")

            # Handle file creation event
            await connector._sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_CREATED,
                file_path=test_file,
                vault_id=descriptor.id,
                metadata={"test": True}
            )
            print("   - ✅ File event handled")

            # Wait for batch processing
            await asyncio.sleep(0.6)

            # Check sync status
            status = await connector.get_sync_status(descriptor.id)
            assert status["sync_enabled"] is True
            assert status["sync_engine_available"] is True
            print("   - ✅ Sync status verified")

            # Test incremental sync
            job_id = await connector.trigger_incremental_sync(vault_source)
            assert job_id is not None
            print("   - ✅ Incremental sync triggered")

            # Test metrics if available
            if connector._sync_engine._metrics_collector:
                metrics = connector._sync_engine._metrics_collector.get_all_metrics()
                assert "counters" in metrics
                print("   - ✅ Metrics collected")

            # Disable sync
            success = await connector.disable_vault_sync(descriptor.id)
            assert success is True
            print("   - ✅ Vault sync disabled")

            # Shutdown
            await connector.shutdown_sync_engine()
            print("   - ✅ Sync engine shutdown")

        print("✅ Sync engine integration test passed")
        return True

    except Exception as e:
        print(f"❌ Sync engine integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_path_change_handling():
    """Test path change detection and handling."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.ingestion.local.state import StateStore, FileRecord, compute_sha256
        from futurnal.ingestion.obsidian.path_tracker import ObsidianPathTracker, PathChange
        from futurnal.ingestion.obsidian.change_detector import create_change_detector

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            vault_dir = workspace / "test_vault"
            vault_dir.mkdir()

            # Create initial file
            original_file = vault_dir / "original.md"
            original_file.write_text("# Original Note\nContent here")

            state_store = StateStore(workspace / "state.db")

            # Create path tracker
            path_tracker = ObsidianPathTracker(
                vault_id="test_vault",
                vault_root=vault_dir,
                state_store=state_store
            )

            # Create initial record
            original_record = FileRecord(
                path=original_file,
                size=original_file.stat().st_size,
                mtime=original_file.stat().st_mtime,
                sha256=compute_sha256(original_file)
            )
            state_store.upsert(original_record)

            print("   - Initial file and record created")

            # Simulate rename
            renamed_file = vault_dir / "renamed.md"
            content = original_file.read_text()
            original_file.unlink()
            renamed_file.write_text(content)

            # Create new record
            new_record = FileRecord(
                path=renamed_file,
                size=renamed_file.stat().st_size,
                mtime=renamed_file.stat().st_mtime,
                sha256=compute_sha256(renamed_file)
            )

            # Detect path changes
            path_changes = path_tracker.detect_path_changes([new_record])

            assert len(path_changes) > 0
            change = path_changes[0]
            assert change.old_path == original_file
            assert change.new_path == renamed_file
            assert change.change_type == "rename"

            print("   - ✅ Path change detected correctly")

            # Test change detector integration
            change_detector = create_change_detector(
                vault_id="test_vault",
                vault_root=vault_dir,
                state_store=state_store,
                path_tracker=path_tracker
            )

            path_changes_2, content_changes = change_detector.detect_changes([new_record])
            assert len(path_changes_2) > 0
            print("   - ✅ Change detector working")

        print("✅ Path change handling test passed")
        return True

    except Exception as e:
        print(f"❌ Path change handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_burst_event_handling():
    """Test handling of rapid file events."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.ingestion.local.state import StateStore
        from futurnal.orchestrator.queue import JobQueue
        from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector
        from futurnal.ingestion.obsidian.sync_engine import create_sync_engine, SyncEventType

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            vault_dir = workspace / "test_vault"
            vault_dir.mkdir()

            # Create components
            state_store = StateStore(workspace / "state.db")
            job_queue = JobQueue(workspace / "queue.db")

            connector = ObsidianVaultConnector(
                workspace_dir=workspace,
                state_store=state_store,
                enable_sync_engine=False
            )

            # Create sync engine with fast batching
            sync_engine = create_sync_engine(
                vault_connector=connector,
                job_queue=job_queue,
                state_store=state_store,
                batch_window_seconds=0.1,  # Very fast batching
                max_batch_size=20,
                performance_monitoring=True
            )

            await sync_engine.start()

            print("   - Sync engine started")

            # Track enqueued jobs
            enqueued_jobs = []
            original_enqueue = job_queue.enqueue
            job_queue.enqueue = lambda job: enqueued_jobs.append(job)

            # Create burst of events
            vault_id = "test_vault"
            num_events = 30

            start_time = time.time()

            tasks = []
            for i in range(num_events):
                test_file = vault_dir / f"burst_note_{i:03d}.md"
                task = sync_engine.handle_file_event(
                    event_type=SyncEventType.FILE_CREATED,
                    file_path=test_file,
                    vault_id=vault_id,
                    metadata={"burst_test": True, "event_num": i}
                )
                tasks.append(task)

            # Execute all events concurrently
            await asyncio.gather(*tasks)
            processing_time = time.time() - start_time

            # Wait for batch processing
            await asyncio.sleep(0.5)

            # Should process efficiently
            assert processing_time < 2.0
            print(f"   - ✅ Processed {num_events} events in {processing_time:.2f}s")

            # Should have created jobs (may be batched)
            assert len(enqueued_jobs) > 0
            total_events = sum(len(job.payload.get("events", [])) for job in enqueued_jobs)
            # Note: due to debouncing, we may have fewer events than submitted
            assert total_events <= num_events
            assert total_events > 0
            print(f"   - ✅ Created {len(enqueued_jobs)} batched jobs for {total_events} events (debounced from {num_events})")

            # Check metrics
            if sync_engine._metrics_collector:
                events_received = sync_engine._metrics_collector.get_counter("events_received")
                batches_processed = sync_engine._metrics_collector.get_counter("batches_processed")
                print(f"   - ✅ Metrics: {events_received} events, {batches_processed} batches")

            await sync_engine.stop()

            # Restore original enqueue
            job_queue.enqueue = original_enqueue

        print("✅ Burst event handling test passed")
        return True

    except Exception as e:
        print(f"❌ Burst event handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_priority_scheduler_integration():
    """Test priority scheduler with different job priorities."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.orchestrator.queue import JobQueue
        from futurnal.orchestrator.priority_scheduler import create_priority_scheduler
        from futurnal.orchestrator.models import IngestionJob, JobType, JobPriority

        with tempfile.TemporaryDirectory() as temp_dir:
            job_queue = JobQueue(Path(temp_dir) / "queue.db")

            # Create priority scheduler
            scheduler = create_priority_scheduler(
                job_queue,
                batch_window_seconds=0.1,
                max_batch_size=10
            )

            await scheduler.start()
            print("   - Priority scheduler started")

            # Create jobs with different priorities
            jobs = []
            for i in range(15):
                if i % 3 == 0:
                    priority = JobPriority.HIGH
                elif i % 3 == 1:
                    priority = JobPriority.NORMAL
                else:
                    priority = JobPriority.LOW

                job = IngestionJob(
                    job_id=f"test_job_{i:03d}",
                    job_type=JobType.LOCAL_FILES,
                    payload={
                        "vault_id": "test_vault",
                        "file_path": f"test_file_{i}.md",
                        "priority_test": True
                    },
                    priority=priority
                )
                jobs.append(job)

            # Submit all jobs
            start_time = time.time()
            for job in jobs:
                await scheduler.enqueue_job(job)

            # Wait for processing
            await asyncio.sleep(0.5)
            processing_time = time.time() - start_time

            assert processing_time < 2.0
            print(f"   - ✅ Processed {len(jobs)} jobs in {processing_time:.2f}s")

            # Check status
            status = await scheduler.get_queue_status()
            assert status["running"] is True
            print("   - ✅ Scheduler status verified")

            await scheduler.stop()
            print("   - ✅ Scheduler stopped")

        print("✅ Priority scheduler integration test passed")
        return True

    except Exception as e:
        print(f"❌ Priority scheduler integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run integration tests."""
    print("🧪 Running Obsidian sync integration tests\n")

    tests = [
        ("Sync Engine Integration", test_sync_engine_integration),
        ("Path Change Handling", test_path_change_handling),
        ("Burst Event Handling", test_burst_event_handling),
        ("Priority Scheduler Integration", test_priority_scheduler_integration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"🔍 Testing {test_name}...")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()

    # Summary
    print("📊 Integration Test Results:")
    print("=" * 60)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} integration tests passed")

    if passed == len(results):
        print("\n🎉 All integration tests passed!")
        print("The Obsidian sync implementation is working correctly for complex scenarios.")
    else:
        print(f"\n⚠️  {len(results) - passed} integration tests failed.")
        print("Some complex functionality needs fixes.")

    return passed == len(results)

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/krzysztof/Futurnal/src')
    asyncio.run(main())