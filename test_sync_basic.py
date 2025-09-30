#!/usr/bin/env python3
"""
Basic functionality test for Obsidian sync engine implementation.
"""

import asyncio
import tempfile
import time
from pathlib import Path

def test_basic_imports():
    """Test that all sync components can be imported."""
    try:
        from futurnal.ingestion.obsidian.sync_engine import (
            ObsidianSyncEngine, SyncEventType, SyncPriority, create_sync_engine
        )
        from futurnal.ingestion.obsidian.change_detector import create_change_detector
        from futurnal.orchestrator.priority_scheduler import create_priority_scheduler
        from futurnal.orchestrator.file_watcher import create_optimized_watcher
        from futurnal.ingestion.obsidian.sync_metrics import create_metrics_collector

        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_sync_engine_creation():
    """Test sync engine can be created."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.ingestion.local.state import StateStore
        from futurnal.orchestrator.queue import JobQueue
        from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector
        from futurnal.ingestion.obsidian.sync_engine import create_sync_engine

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            # Create basic components
            state_store = StateStore(workspace / "state.db")
            job_queue = JobQueue(workspace / "queue.db")

            connector = ObsidianVaultConnector(
                workspace_dir=workspace,
                state_store=state_store,
                enable_sync_engine=False  # Don't auto-initialize
            )

            # Create sync engine
            sync_engine = create_sync_engine(
                vault_connector=connector,
                job_queue=job_queue,
                state_store=state_store,
                performance_monitoring=True
            )

            print("✅ Sync engine created successfully")
            print(f"   - Running: {sync_engine._running}")
            print(f"   - Performance monitoring: {sync_engine._performance_monitoring}")
            print(f"   - Metrics collector: {sync_engine._metrics_collector is not None}")

            return True

    except Exception as e:
        print(f"❌ Sync engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sync_engine_lifecycle():
    """Test sync engine start/stop lifecycle."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.ingestion.local.state import StateStore
        from futurnal.orchestrator.queue import JobQueue
        from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector
        from futurnal.ingestion.obsidian.sync_engine import create_sync_engine

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            # Create components
            state_store = StateStore(workspace / "state.db")
            job_queue = JobQueue(workspace / "queue.db")

            connector = ObsidianVaultConnector(
                workspace_dir=workspace,
                state_store=state_store,
                enable_sync_engine=False
            )

            sync_engine = create_sync_engine(
                vault_connector=connector,
                job_queue=job_queue,
                state_store=state_store,
                batch_window_seconds=0.1,  # Fast for testing
                performance_monitoring=True
            )

            # Test lifecycle
            print("   - Starting sync engine...")
            await sync_engine.start()
            assert sync_engine._running is True
            print("   - ✅ Started successfully")

            # Test basic functionality
            from futurnal.ingestion.obsidian.sync_engine import SyncEventType
            test_path = workspace / "test.md"

            print("   - Testing event handling...")
            await sync_engine.handle_file_event(
                event_type=SyncEventType.FILE_CREATED,
                file_path=test_path,
                vault_id="test_vault"
            )
            print("   - ✅ Event handled")

            # Wait for processing
            await asyncio.sleep(0.2)

            # Test status
            print("   - Testing status reporting...")
            status = await sync_engine.get_sync_status("test_vault")
            assert "vault_id" in status
            assert "engine_running" in status
            print("   - ✅ Status retrieved")

            # Stop
            print("   - Stopping sync engine...")
            await sync_engine.stop()
            assert sync_engine._running is False
            print("   - ✅ Stopped successfully")

            print("✅ Sync engine lifecycle test passed")
            return True

    except Exception as e:
        print(f"❌ Sync engine lifecycle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_priority_scheduler():
    """Test priority scheduler creation."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.orchestrator.queue import JobQueue
        from futurnal.orchestrator.priority_scheduler import create_priority_scheduler

        with tempfile.TemporaryDirectory() as temp_dir:
            job_queue = JobQueue(Path(temp_dir) / "queue.db")

            scheduler = create_priority_scheduler(
                job_queue,
                batch_window_seconds=0.1,
                max_batch_size=10
            )

            print("✅ Priority scheduler created successfully")
            return True

    except Exception as e:
        print(f"❌ Priority scheduler creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_change_detector():
    """Test change detector creation."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.ingestion.local.state import StateStore
        from futurnal.ingestion.obsidian.change_detector import create_change_detector

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            state_store = StateStore(workspace / "state.db")

            detector = create_change_detector(
                vault_id="test_vault",
                vault_root=workspace,
                state_store=state_store
            )

            print("✅ Change detector created successfully")
            return True

    except Exception as e:
        print(f"❌ Change detector creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_collector():
    """Test metrics collector creation."""
    try:
        import sys
        sys.path.append('/Users/krzysztof/Futurnal/src')

        from futurnal.ingestion.obsidian.sync_metrics import create_metrics_collector

        collector = create_metrics_collector(
            max_history_size=1000,
            enable_detailed_timing=True
        )

        # Test basic metrics operations
        collector.increment_counter("test_counter")
        collector.set_gauge("test_gauge", 42.0)

        with collector.timer("test_timer"):
            time.sleep(0.001)  # Small delay

        collector.record_event("test_event", "test_vault")

        # Test getting metrics
        counter_val = collector.get_counter("test_counter")
        gauge_val = collector.get_gauge("test_gauge")
        timer_stats = collector.get_timer_stats("test_timer")

        assert counter_val == 1
        assert gauge_val == 42.0
        assert "count" in timer_stats

        print("✅ Metrics collector working correctly")
        return True

    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all basic tests."""
    print("🧪 Running basic functionality tests for Obsidian sync implementation\n")

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Sync Engine Creation", test_sync_engine_creation),
        ("Sync Engine Lifecycle", test_sync_engine_lifecycle),
        ("Priority Scheduler", test_priority_scheduler),
        ("Change Detector", test_change_detector),
        ("Metrics Collector", test_metrics_collector),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"🔍 Testing {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()

    # Summary
    print("📊 Test Results Summary:")
    print("=" * 50)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All basic functionality tests passed!")
        print("The Obsidian sync implementation is working correctly.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed.")
        print("The implementation needs fixes before it's ready.")

    return passed == len(results)

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/krzysztof/Futurnal/src')
    asyncio.run(main())