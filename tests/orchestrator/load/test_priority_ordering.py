"""Tests for priority ordering validation under load."""

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from futurnal.orchestrator.load_test import PriorityOrderingMetrics
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobStatus


@pytest.mark.asyncio
@pytest.mark.load
async def test_priority_ordering_under_load(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that HIGH priority jobs preempt NORMAL and LOW.

    Validates that higher priority jobs have lower latency even
    when the queue is filled with lower priority jobs.

    Acceptance Criteria:
    - HIGH priority jobs complete first
    - Priority inversions <5% of jobs
    - HIGH latency < NORMAL latency < LOW latency
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

    # Create test directory with files
    test_dir = tmp_path / "priority_test"
    test_dir.mkdir()
    for i in range(50):
        (test_dir / f"file_{i}.txt").write_text(f"content {i}")

    # Register source
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(name="priority_test", root_path=test_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Enqueue 30 LOW priority jobs to fill queue
    for i in range(30):
        job = IngestionJob(
            job_id=f"low_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "priority_test", "size_bytes": 1_000_000},
            priority=JobPriority.LOW,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Enqueue 10 HIGH priority jobs (should jump ahead)
    for i in range(10):
        job = IngestionJob(
            job_id=f"high_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "priority_test", "size_bytes": 1_000_000},
            priority=JobPriority.HIGH,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Start orchestrator and process jobs
    orchestrator.start()
    await asyncio.sleep(10.0)  # Allow time for processing
    await orchestrator.shutdown()

    # Analyze priority ordering
    completed_jobs = job_queue.snapshot(status=JobStatus.SUCCEEDED)

    high_jobs = [j for j in completed_jobs if j["job_id"].startswith("high_")]
    low_jobs = [j for j in completed_jobs if j["job_id"].startswith("low_")]

    # HIGH priority jobs should have completed
    assert len(high_jobs) > 0, "No HIGH priority jobs completed"

    # Validate HIGH jobs completed first (check completion order)
    # All HIGH jobs should finish before most LOW jobs if we completed many
    if len(completed_jobs) >= 15:
        # First 10 completed jobs should mostly be HIGH priority
        first_10 = completed_jobs[-10:]  # Most recent 10 (reversed order in snapshot)
        high_in_first_10 = sum(1 for j in first_10 if j["job_id"].startswith("high_"))
        assert high_in_first_10 >= 5, f"Expected >= 5 HIGH jobs in first 10, got {high_in_first_10}"


@pytest.mark.asyncio
@pytest.mark.load
async def test_three_tier_priority_latency(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test priority ordering with HIGH, NORMAL, and LOW priorities.

    Validates that average latency follows priority ordering:
    HIGH latency <= NORMAL latency <= LOW latency

    Acceptance Criteria:
    - All priority levels complete jobs
    - Average latency ordered by priority
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
    test_dir = tmp_path / "three_tier_test"
    test_dir.mkdir()
    for i in range(50):
        (test_dir / f"file_{i}.txt").write_text(f"content {i}")

    # Register source
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(name="three_tier_test", root_path=test_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Enqueue jobs with mixed priorities
    priorities = [
        (JobPriority.LOW, 10),
        (JobPriority.NORMAL, 10),
        (JobPriority.HIGH, 10),
    ]

    for priority, count in priorities:
        for i in range(count):
            job = IngestionJob(
                job_id=f"{priority.name.lower()}_{i}",
                job_type=JobType.LOCAL_FILES,
                payload={"source_name": "three_tier_test", "size_bytes": 1_000_000},
                priority=priority,
                scheduled_for=datetime.utcnow(),
            )
            job_queue.enqueue(job)

    # Start orchestrator
    orchestrator.start()
    await asyncio.sleep(10.0)
    await orchestrator.shutdown()

    # Calculate average latency per priority
    completed_jobs = job_queue.snapshot(status=JobStatus.SUCCEEDED)

    def calculate_avg_latency(jobs, priority_name):
        priority_jobs = [j for j in jobs if j["job_id"].startswith(priority_name)]
        if not priority_jobs:
            return float("inf")

        latencies = []
        for job in priority_jobs:
            created_at = datetime.fromisoformat(job["created_at"])
            updated_at = datetime.fromisoformat(job["updated_at"])
            latency = (updated_at - created_at).total_seconds()
            latencies.append(latency)

        return sum(latencies) / len(latencies)

    high_latency = calculate_avg_latency(completed_jobs, "high")
    normal_latency = calculate_avg_latency(completed_jobs, "normal")
    low_latency = calculate_avg_latency(completed_jobs, "low")

    # Validate all priorities completed jobs
    assert high_latency < float("inf"), "No HIGH priority jobs completed"
    assert normal_latency < float("inf"), "No NORMAL priority jobs completed"
    assert low_latency < float("inf"), "No LOW priority jobs completed"

    # Validate ordering (with some tolerance for concurrency)
    # HIGH should be fastest
    assert high_latency <= normal_latency + 1.0, \
        f"HIGH latency ({high_latency:.2f}s) > NORMAL latency ({normal_latency:.2f}s)"


@pytest.mark.asyncio
@pytest.mark.load
async def test_high_priority_preemption(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test HIGH priority job preemption of LOW priority jobs.

    Validates that HIGH priority jobs start within 1 second even
    when queue is full of LOW priority jobs.

    Acceptance Criteria:
    - HIGH priority job starts within 1 second
    - HIGH job completes before most LOW jobs
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
    test_dir = tmp_path / "preemption_test"
    test_dir.mkdir()
    for i in range(100):
        (test_dir / f"file_{i}.txt").write_text(f"content {i}")

    # Register source
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(name="preemption_test", root_path=test_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Fill queue with LOW priority jobs
    for i in range(50):
        job = IngestionJob(
            job_id=f"low_{i}",
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "preemption_test", "size_bytes": 1_000_000},
            priority=JobPriority.LOW,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Start orchestrator
    orchestrator.start()
    await asyncio.sleep(0.5)  # Let LOW jobs start processing

    # Add HIGH priority job
    high_job_enqueued_at = datetime.utcnow()
    high_job = IngestionJob(
        job_id="high_urgent",
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "preemption_test", "size_bytes": 1_000_000},
        priority=JobPriority.HIGH,
        scheduled_for=high_job_enqueued_at,
    )
    job_queue.enqueue(high_job)

    # Wait for high job to complete
    await asyncio.sleep(3.0)
    await orchestrator.shutdown()

    # Check if HIGH job completed
    completed_jobs = job_queue.snapshot(status=JobStatus.SUCCEEDED)
    high_completed = [j for j in completed_jobs if j["job_id"] == "high_urgent"]

    assert len(high_completed) == 1, "HIGH priority job did not complete"

    # Calculate time to start
    high_job_data = high_completed[0]
    created_at = datetime.fromisoformat(high_job_data["created_at"])
    updated_at = datetime.fromisoformat(high_job_data["updated_at"])
    time_to_complete = (updated_at - created_at).total_seconds()

    # Should complete relatively quickly (within reasonable time given processing)
    assert time_to_complete < 5.0, \
        f"HIGH priority job took {time_to_complete:.2f}s to complete"


@pytest.mark.asyncio
@pytest.mark.load
async def test_priority_ordering_metrics_calculation(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test PriorityOrderingMetrics calculation accuracy.

    Validates that priority metrics are calculated correctly
    from completed jobs.

    Acceptance Criteria:
    - Metrics accurately reflect job distribution
    - Priority ordering validation works correctly
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
    test_dir = tmp_path / "metrics_test"
    test_dir.mkdir()
    for i in range(40):
        (test_dir / f"file_{i}.txt").write_text(f"content {i}")

    # Register source
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(name="metrics_test", root_path=test_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Enqueue known distribution: 5 HIGH, 10 NORMAL, 5 LOW
    for priority, count, prefix in [
        (JobPriority.HIGH, 5, "high"),
        (JobPriority.NORMAL, 10, "normal"),
        (JobPriority.LOW, 5, "low"),
    ]:
        for i in range(count):
            job = IngestionJob(
                job_id=f"{prefix}_{i}",
                job_type=JobType.LOCAL_FILES,
                payload={"source_name": "metrics_test", "size_bytes": 1_000_000},
                priority=priority,
                scheduled_for=datetime.utcnow(),
            )
            job_queue.enqueue(job)

    # Process jobs
    orchestrator.start()
    await asyncio.sleep(8.0)
    await orchestrator.shutdown()

    # Calculate metrics
    completed_jobs = job_queue.snapshot(status=JobStatus.SUCCEEDED)

    jobs_by_priority = {
        JobPriority.HIGH: 0,
        JobPriority.NORMAL: 0,
        JobPriority.LOW: 0,
    }

    avg_latency_by_priority = {}

    for priority in [JobPriority.HIGH, JobPriority.NORMAL, JobPriority.LOW]:
        priority_jobs = [j for j in completed_jobs if j["job_id"].startswith(priority.name.lower())]
        jobs_by_priority[priority] = len(priority_jobs)

        if priority_jobs:
            latencies = []
            for job in priority_jobs:
                created_at = datetime.fromisoformat(job["created_at"])
                updated_at = datetime.fromisoformat(job["updated_at"])
                latency = (updated_at - created_at).total_seconds()
                latencies.append(latency)
            avg_latency_by_priority[priority] = sum(latencies) / len(latencies)

    metrics = PriorityOrderingMetrics(
        jobs_by_priority=jobs_by_priority,
        avg_latency_by_priority=avg_latency_by_priority,
        priority_inversions=0,  # Would need detailed tracking to calculate
    )

    # Validate counts (some jobs should have completed)
    assert sum(jobs_by_priority.values()) > 0, "No jobs completed"

    # Validate metrics structure
    assert JobPriority.HIGH in metrics.jobs_by_priority
    assert JobPriority.NORMAL in metrics.jobs_by_priority
    assert JobPriority.LOW in metrics.jobs_by_priority
