"""Tests for LoadTestRunner functionality."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from futurnal.orchestrator.load_test import ConnectorLoad, LoadTestConfig
from futurnal.orchestrator.load_test_runner import LoadTestRunner
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue, JobStatus


def test_sample_priority_single_priority():
    """Test priority sampling with single priority (100% probability)."""
    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    distribution = {JobPriority.HIGH: 1.0}

    # Sample 10 times, should always get HIGH
    for _ in range(10):
        priority = runner._sample_priority(distribution)
        assert priority == JobPriority.HIGH


def test_sample_priority_distribution():
    """Test priority sampling respects distribution."""
    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    distribution = {
        JobPriority.HIGH: 0.2,
        JobPriority.NORMAL: 0.5,
        JobPriority.LOW: 0.3,
    }

    # Sample many times and verify rough distribution
    samples = [runner._sample_priority(distribution) for _ in range(1000)]

    high_count = sum(1 for p in samples if p == JobPriority.HIGH)
    normal_count = sum(1 for p in samples if p == JobPriority.NORMAL)
    low_count = sum(1 for p in samples if p == JobPriority.LOW)

    # Allow 10% tolerance
    assert 150 < high_count < 250  # ~20% of 1000
    assert 400 < normal_count < 600  # ~50% of 1000
    assert 200 < low_count < 400  # ~30% of 1000


def test_calculate_avg_latency_empty_jobs():
    """Test average latency calculation with no jobs."""
    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    latency = runner._calculate_avg_latency([])
    assert latency == 0.0


def test_calculate_avg_latency_with_jobs():
    """Test average latency calculation with jobs."""
    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    now = datetime.utcnow()
    jobs = [
        {
            "job_id": "1",
            "created_at": now.isoformat(),
            "updated_at": (now + timedelta(seconds=5)).isoformat(),
        },
        {
            "job_id": "2",
            "created_at": now.isoformat(),
            "updated_at": (now + timedelta(seconds=10)).isoformat(),
        },
        {
            "job_id": "3",
            "created_at": now.isoformat(),
            "updated_at": (now + timedelta(seconds=15)).isoformat(),
        },
    ]

    latency = runner._calculate_avg_latency(jobs)
    # Average of 5, 10, 15 = 10 seconds
    assert abs(latency - 10.0) < 0.1


def test_calculate_throughput_empty_jobs():
    """Test throughput calculation with no jobs."""
    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    throughput = runner._calculate_throughput([], duration=60.0)
    assert throughput == 0.0


def test_calculate_throughput_with_jobs():
    """Test throughput calculation with jobs."""
    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    # 3 jobs, each processed 10 MB (10,000,000 bytes)
    jobs = [
        {"job_id": "1", "payload": {"bytes_processed": 10_000_000}},
        {"job_id": "2", "payload": {"bytes_processed": 10_000_000}},
        {"job_id": "3", "payload": {"bytes_processed": 10_000_000}},
    ]

    # Total: 30,000,000 bytes = 28.61 MiB in 60 seconds = 0.477 MiB/s
    throughput = runner._calculate_throughput(jobs, duration=60.0)
    assert abs(throughput - 0.477) < 0.01


def test_calculate_throughput_high_rate():
    """Test throughput calculation with high data rate."""
    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    # 60 jobs, each processed 5 MB (5,000,000 bytes)
    # Total: 300,000,000 bytes = 286.1 MiB in 60 seconds = 4.77 MiB/s
    jobs = [
        {"job_id": str(i), "payload": {"bytes_processed": 5_000_000}}
        for i in range(60)
    ]

    throughput = runner._calculate_throughput(jobs, duration=60.0)
    assert abs(throughput - 4.77) < 0.1


def test_detect_starvation_no_starvation():
    """Test starvation detection with fair allocation."""
    from futurnal.orchestrator.load_test import ConnectorMetrics

    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    connector_loads = [
        ConnectorLoad(
            connector_type=JobType.LOCAL_FILES,
            jobs_per_minute=60,
            priority_distribution={JobPriority.NORMAL: 1.0},
            avg_job_size_bytes=5_000_000,
            avg_job_duration_seconds=1.0,
        ),
        ConnectorLoad(
            connector_type=JobType.OBSIDIAN_VAULT,
            jobs_per_minute=60,
            priority_distribution={JobPriority.NORMAL: 1.0},
            avg_job_size_bytes=5_000_000,
            avg_job_duration_seconds=1.0,
        ),
    ]

    # Both connectors achieving expected throughput (~5 MB/s)
    connector_metrics = {
        JobType.LOCAL_FILES: ConnectorMetrics(
            connector_type=JobType.LOCAL_FILES,
            jobs_completed=60,
            bytes_processed=300_000_000,
            total_duration_seconds=60.0,
            avg_job_latency_seconds=2.0,
            throughput_mbps=4.8,  # ~96% of expected
            worker_time_seconds=60.0,
        ),
        JobType.OBSIDIAN_VAULT: ConnectorMetrics(
            connector_type=JobType.OBSIDIAN_VAULT,
            jobs_completed=60,
            bytes_processed=300_000_000,
            total_duration_seconds=60.0,
            avg_job_latency_seconds=2.0,
            throughput_mbps=4.8,  # ~96% of expected
            worker_time_seconds=60.0,
        ),
    }

    starved = runner._detect_starvation(connector_metrics, connector_loads)
    assert len(starved) == 0


def test_detect_starvation_with_starved_connector():
    """Test starvation detection with starved connector."""
    from futurnal.orchestrator.load_test import ConnectorMetrics

    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    connector_loads = [
        ConnectorLoad(
            connector_type=JobType.LOCAL_FILES,
            jobs_per_minute=60,
            priority_distribution={JobPriority.NORMAL: 1.0},
            avg_job_size_bytes=5_000_000,
            avg_job_duration_seconds=1.0,
        ),
        ConnectorLoad(
            connector_type=JobType.OBSIDIAN_VAULT,
            jobs_per_minute=60,
            priority_distribution={JobPriority.NORMAL: 1.0},
            avg_job_size_bytes=5_000_000,
            avg_job_duration_seconds=1.0,
        ),
    ]

    # LOCAL_FILES getting most resources, OBSIDIAN_VAULT starved
    connector_metrics = {
        JobType.LOCAL_FILES: ConnectorMetrics(
            connector_type=JobType.LOCAL_FILES,
            jobs_completed=58,
            bytes_processed=290_000_000,
            total_duration_seconds=60.0,
            avg_job_latency_seconds=2.0,
            throughput_mbps=4.6,
            worker_time_seconds=58.0,
        ),
        JobType.OBSIDIAN_VAULT: ConnectorMetrics(
            connector_type=JobType.OBSIDIAN_VAULT,
            jobs_completed=3,
            bytes_processed=15_000_000,
            total_duration_seconds=60.0,
            avg_job_latency_seconds=20.0,
            throughput_mbps=0.24,  # Only ~5% of expected 5 MB/s
            worker_time_seconds=3.0,
        ),
    }

    starved = runner._detect_starvation(connector_metrics, connector_loads)
    assert JobType.OBSIDIAN_VAULT in starved
    assert JobType.LOCAL_FILES not in starved


def test_calculate_job_duration():
    """Test job duration calculation."""
    queue = JobQueue(Path(":memory:"))
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    now = datetime.utcnow()
    job = {
        "job_id": "1",
        "created_at": now.isoformat(),
        "updated_at": (now + timedelta(seconds=5)).isoformat(),
    }

    duration = runner._calculate_job_duration(job)
    assert abs(duration - 5.0) < 0.1


@pytest.mark.asyncio
async def test_wait_for_completion_immediate(tmp_path: Path):
    """Test wait for completion when queue is already empty."""
    queue = JobQueue(tmp_path / "queue.db")
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    # Queue is empty, should return immediately
    start = asyncio.get_event_loop().time()
    await runner._wait_for_completion(timeout=10)
    elapsed = asyncio.get_event_loop().time() - start

    # Should complete very quickly (< 1 second)
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_wait_for_completion_with_pending_jobs(tmp_path: Path):
    """Test wait for completion with pending jobs."""
    queue = JobQueue(tmp_path / "queue.db")
    runner = LoadTestRunner(orchestrator=None, job_queue=queue)  # type: ignore

    # Add some jobs
    for i in range(5):
        job = IngestionJob(
            job_id=f"job_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "test"},
            priority=JobPriority.NORMAL,
        )
        queue.enqueue(job)

    # Start wait (should timeout since we're not processing jobs)
    start = asyncio.get_event_loop().time()
    await runner._wait_for_completion(timeout=2)
    elapsed = asyncio.get_event_loop().time() - start

    # Should wait for the timeout
    assert 1.5 < elapsed < 2.5
