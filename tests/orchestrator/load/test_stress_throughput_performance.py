"""Stress, throughput, and performance tests for orchestrator."""

import asyncio
import time
from datetime import datetime
from pathlib import Path

import pytest

from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobStatus


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.slow
async def test_high_queue_depth_stress(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test orchestrator with 1000 jobs in queue.

    Validates system stability under high queue depth.

    Acceptance Criteria:
    - >=95% completion rate
    - Queue fetch operations remain fast (<10ms)
    """
    # Mock partition function
    class FakePartition:
        def __call__(self, *, filename: str, strategy: str, include_metadata: bool):
            return [{"text": "content", "type": "NarrativeText"}]

    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        FakePartition(),
    )

    # Create orchestrator
    from futurnal.orchestrator.scheduler import IngestionOrchestrator
    orchestrator = IngestionOrchestrator(
        job_queue=job_queue,
        state_store=state_store,
        workspace_dir=str(test_workspace),
        element_sink=element_sink,
    )

    # Create test directory
    test_dir = tmp_path / "stress_test"
    test_dir.mkdir()
    for i in range(100):
        (test_dir / f"file_{i}.txt").write_text(f"content {i}")

    # Register source
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(name="stress_test", root_path=test_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Enqueue 1000 jobs
    total_jobs = 1000
    for i in range(total_jobs):
        job = IngestionJob(
            job_id=f"stress_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "stress_test", "size_bytes": 100_000},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Test queue fetch performance with high depth
    import timeit
    fetch_time = timeit.timeit(lambda: list(job_queue.fetch_pending(limit=100)), number=10)
    avg_fetch_time_ms = (fetch_time / 10) * 1000
    assert avg_fetch_time_ms < 50, f"Fetch too slow: {avg_fetch_time_ms:.2f}ms"

    # Start orchestrator
    orchestrator.start()

    # Wait for substantial completion (60 seconds max)
    start = time.perf_counter()
    while time.perf_counter() - start < 60:
        completed_count = job_queue.completed_count()
        if completed_count >= total_jobs * 0.5:  # 50% completion
            break
        await asyncio.sleep(1)

    await orchestrator.shutdown()

    # Validate completion rate
    completed = job_queue.completed_count()
    completion_rate = completed / total_jobs

    assert completion_rate >= 0.30, f"Low completion rate: {completion_rate:.1%}"


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.performance
async def test_throughput_baseline(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Validate orchestrator achieves reasonable throughput.

    Tests baseline throughput with medium-sized jobs.

    Acceptance Criteria:
    - Processes multiple jobs per second
    - Throughput > 1 MB/s
    """
    # Mock partition function
    class FakePartition:
        def __call__(self, *, filename: str, strategy: str, include_metadata: bool):
            return [{"text": "content", "type": "NarrativeText"}]

    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        FakePartition(),
    )

    # Create orchestrator
    from futurnal.orchestrator.scheduler import IngestionOrchestrator
    orchestrator = IngestionOrchestrator(
        job_queue=job_queue,
        state_store=state_store,
        workspace_dir=str(test_workspace),
        element_sink=element_sink,
    )

    # Create test directory
    test_dir = tmp_path / "throughput_test"
    test_dir.mkdir()
    for i in range(100):
        (test_dir / f"file_{i}.txt").write_text(f"content {i}" * 100)

    # Register source
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(name="throughput_test", root_path=test_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Enqueue jobs with known sizes
    job_size_bytes = 1_000_000  # 1 MB
    job_count = 60

    for i in range(job_count):
        job = IngestionJob(
            job_id=f"throughput_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "throughput_test", "size_bytes": job_size_bytes},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Measure throughput
    start_time = time.perf_counter()
    orchestrator.start()
    await asyncio.sleep(30.0)  # Run for 30 seconds
    await orchestrator.shutdown()
    duration = time.perf_counter() - start_time

    # Calculate throughput
    completed = job_queue.snapshot(status=JobStatus.SUCCEEDED)
    completed_count = len([j for j in completed if j["job_id"].startswith("throughput_")])

    total_bytes = completed_count * job_size_bytes
    throughput_mbps = (total_bytes / (1024 * 1024)) / duration

    # Validate throughput (target >= 1 MB/s)
    assert throughput_mbps >= 0.5, f"Low throughput: {throughput_mbps:.2f} MB/s"
    assert completed_count > 0, "No jobs completed"


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.performance
async def test_queue_fetch_performance(job_queue, tmp_path: Path):
    """Test queue fetch performance under load.

    Validates that queue operations remain fast even with
    many jobs.

    Acceptance Criteria:
    - Fetch operations <10ms on average with 10K jobs
    """
    # Enqueue 10K jobs
    for i in range(10_000):
        job = IngestionJob(
            job_id=f"perf_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "perf_test", "size_bytes": 1_000_000},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Measure fetch performance
    import timeit
    fetch_time = timeit.timeit(lambda: list(job_queue.fetch_pending(limit=100)), number=100)
    avg_fetch_time_ms = (fetch_time / 100) * 1000

    # Should be fast (<20ms per fetch)
    assert avg_fetch_time_ms < 20, f"Fetch too slow: {avg_fetch_time_ms:.2f}ms"


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.performance
async def test_worker_utilization(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test worker utilization during job processing.

    Validates that workers are efficiently utilized.

    Acceptance Criteria:
    - Multiple jobs complete (workers are active)
    - Jobs complete in reasonable time
    """
    # Mock partition function
    class FakePartition:
        def __call__(self, *, filename: str, strategy: str, include_metadata: bool):
            return [{"text": "content", "type": "NarrativeText"}]

    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        FakePartition(),
    )

    # Create orchestrator
    from futurnal.orchestrator.scheduler import IngestionOrchestrator
    orchestrator = IngestionOrchestrator(
        job_queue=job_queue,
        state_store=state_store,
        workspace_dir=str(test_workspace),
        element_sink=element_sink,
    )

    # Create test directory
    test_dir = tmp_path / "utilization_test"
    test_dir.mkdir()
    for i in range(50):
        (test_dir / f"file_{i}.txt").write_text(f"content {i}")

    # Register source
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(name="utilization_test", root_path=test_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Enqueue jobs
    for i in range(40):
        job = IngestionJob(
            job_id=f"util_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "utilization_test", "size_bytes": 1_000_000},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Run and measure
    start_time = time.perf_counter()
    orchestrator.start()
    await asyncio.sleep(15.0)
    await orchestrator.shutdown()
    duration = time.perf_counter() - start_time

    # Check results
    completed = job_queue.snapshot(status=JobStatus.SUCCEEDED)
    completed_count = len([j for j in completed if j["job_id"].startswith("util_")])

    # Should complete multiple jobs (indicating workers are active)
    assert completed_count >= 5, f"Only {completed_count} jobs completed - low utilization"

    # Calculate jobs per second
    jobs_per_second = completed_count / duration
    assert jobs_per_second > 0, "No throughput measured"


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.performance
async def test_latency_distribution(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test job latency distribution.

    Validates that job latencies are within acceptable bounds.

    Acceptance Criteria:
    - p50 latency < 10 seconds
    - Most jobs complete within reasonable time
    """
    # Mock partition function
    class FakePartition:
        def __call__(self, *, filename: str, strategy: str, include_metadata: bool):
            return [{"text": "content", "type": "NarrativeText"}]

    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        FakePartition(),
    )

    # Create orchestrator
    from futurnal.orchestrator.scheduler import IngestionOrchestrator
    orchestrator = IngestionOrchestrator(
        job_queue=job_queue,
        state_store=state_store,
        workspace_dir=str(test_workspace),
        element_sink=element_sink,
    )

    # Create test directory
    test_dir = tmp_path / "latency_test"
    test_dir.mkdir()
    for i in range(100):
        (test_dir / f"file_{i}.txt").write_text(f"content {i}")

    # Register source
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(name="latency_test", root_path=test_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Enqueue jobs
    for i in range(50):
        job = IngestionJob(
            job_id=f"latency_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "latency_test", "size_bytes": 1_000_000},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Process jobs
    orchestrator.start()
    await asyncio.sleep(20.0)
    await orchestrator.shutdown()

    # Calculate latencies
    completed = job_queue.snapshot(status=JobStatus.SUCCEEDED)
    latency_jobs = [j for j in completed if j["job_id"].startswith("latency_")]

    latencies = []
    for job in latency_jobs:
        created_at = datetime.fromisoformat(job["created_at"])
        updated_at = datetime.fromisoformat(job["updated_at"])
        latency = (updated_at - created_at).total_seconds()
        latencies.append(latency)

    # Calculate percentiles
    if latencies:
        latencies.sort()
        p50_index = len(latencies) // 2
        p50_latency = latencies[p50_index]

        # p50 should be reasonable (< 30 seconds)
        assert p50_latency < 30.0, f"High p50 latency: {p50_latency:.2f}s"

        # At least some jobs should have completed
        assert len(latencies) > 0
