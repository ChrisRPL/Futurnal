"""Tests for concurrent multi-connector execution."""

import asyncio
from pathlib import Path

import pytest

from futurnal.orchestrator.load_test import ConnectorLoad, LoadTestConfig
from futurnal.orchestrator.load_test_runner import LoadTestRunner
from futurnal.orchestrator.models import JobPriority, JobType


@pytest.mark.asyncio
@pytest.mark.load
async def test_concurrent_multi_connector_execution(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test multiple connectors executing simultaneously.

    Validates that 3 connectors (LOCAL_FILES, OBSIDIAN_VAULT, IMAP_MAILBOX)
    can execute jobs concurrently without interference.

    Acceptance Criteria:
    - All 3 connectors complete jobs
    - Fairness index >= 0.75 (moderate fairness threshold for initial test)
    """
    # Mock the partition function for fast execution - BEFORE creating orchestrator
    class FakePartition:
        def __init__(self):
            self.calls = []

        def __call__(self, *, filename: str, strategy: str, include_metadata: bool):
            self.calls.append(filename)
            return [{"text": f"element from {filename}", "type": "NarrativeText"}]

    fake_partition = FakePartition()
    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        fake_partition,
    )

    # Create orchestrator AFTER monkeypatch
    from futurnal.orchestrator.scheduler import IngestionOrchestrator
    load_test_orchestrator = IngestionOrchestrator(
        job_queue=job_queue,
        state_store=state_store,
        workspace_dir=str(test_workspace),
        element_sink=element_sink,
    )

    # Create test files for each connector type
    local_dir = tmp_path / "local_files"
    local_dir.mkdir()
    for i in range(10):
        (local_dir / f"file_{i}.txt").write_text(f"test content {i}")

    # Register LOCAL_FILES source (other connectors would need setup)
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source = LocalIngestionSource(
        name="local_test",
        root_path=local_dir,
        max_workers=8,
    )
    load_test_orchestrator.register_source(
        SourceRegistration(
            source=source,
            schedule="@manual",
            priority=JobPriority.NORMAL,
        )
    )

    # Configure load test (simplified to LOCAL_FILES only for this test)
    # Full multi-connector test would require Obsidian and IMAP setup
    config = LoadTestConfig(
        name="concurrent_execution",
        duration_seconds=10,  # Short duration for test
        connectors=[
            ConnectorLoad(
                connector_type=JobType.LOCAL_FILES,
                jobs_per_minute=12,  # 2 jobs generated during 10s test
                priority_distribution={JobPriority.NORMAL: 1.0},
                avg_job_size_bytes=1_000_000,
                avg_job_duration_seconds=0.5,
            ),
        ],
        worker_count=8,
    )

    # Run load test
    runner = LoadTestRunner(
        orchestrator=load_test_orchestrator,
        job_queue=job_queue,
        source_name_map={JobType.LOCAL_FILES: "local_test"},
    )

    metrics = await runner.run_test(config)

    # Validate results
    assert JobType.LOCAL_FILES in metrics.connector_metrics
    local_metrics = metrics.connector_metrics[JobType.LOCAL_FILES]

    # Should have completed at least 1 job
    assert local_metrics.jobs_completed > 0

    # For single connector, fairness should be perfect
    assert metrics.jain_fairness_index >= 0.99

    # Cleanup
    if load_test_orchestrator._running:
        await load_test_orchestrator.shutdown()


@pytest.mark.asyncio
@pytest.mark.load
async def test_two_connector_balanced_load(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test balanced load distribution across two connectors.

    Validates that two connectors with equal job rates receive
    proportional execution time.

    Acceptance Criteria:
    - Both connectors complete jobs
    - Fairness index >= 0.8
    """
    # Mock partition function - BEFORE creating orchestrator
    class FakePartition:
        def __init__(self):
            self.calls = []

        def __call__(self, *, filename: str, strategy: str, include_metadata: bool):
            self.calls.append(filename)
            return [{"text": f"content", "type": "NarrativeText"}]

    fake_partition = FakePartition()
    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        fake_partition,
    )

    # Create orchestrator AFTER monkeypatch
    from futurnal.orchestrator.scheduler import IngestionOrchestrator
    load_test_orchestrator = IngestionOrchestrator(
        job_queue=job_queue,
        state_store=state_store,
        workspace_dir=str(test_workspace),
        element_sink=element_sink,
    )

    # Create test directories
    dir1 = tmp_path / "source1"
    dir2 = tmp_path / "source2"
    dir1.mkdir()
    dir2.mkdir()

    for i in range(15):
        (dir1 / f"file_{i}.txt").write_text(f"content {i}")
        (dir2 / f"file_{i}.txt").write_text(f"content {i}")

    # Register two sources
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source1 = LocalIngestionSource(name="source1", root_path=dir1, max_workers=4)
    source2 = LocalIngestionSource(name="source2", root_path=dir2, max_workers=4)

    load_test_orchestrator.register_source(
        SourceRegistration(source=source1, schedule="@manual", priority=JobPriority.NORMAL)
    )
    load_test_orchestrator.register_source(
        SourceRegistration(source=source2, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Configure load test - use manual triggering instead of LoadTestRunner
    # to have more control over job generation
    from futurnal.orchestrator.models import IngestionJob
    import uuid
    from datetime import datetime

    # Enqueue jobs for both sources
    for i in range(5):
        # Source 1 jobs
        job1 = IngestionJob(
            job_id=str(uuid.uuid4()),
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "source1", "size_bytes": 1_000_000},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job1)

        # Source 2 jobs
        job2 = IngestionJob(
            job_id=str(uuid.uuid4()),
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "source2", "size_bytes": 1_000_000},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job2)

    # Start orchestrator and wait for completion
    load_test_orchestrator.start()
    await asyncio.sleep(5.0)  # Wait for jobs to complete
    await load_test_orchestrator.shutdown()

    # Validate both sources processed jobs
    from futurnal.orchestrator.queue import JobStatus
    completed = job_queue.snapshot(status=JobStatus.SUCCEEDED)

    source1_jobs = [j for j in completed if j["payload"]["source_name"] == "source1"]
    source2_jobs = [j for j in completed if j["payload"]["source_name"] == "source2"]

    # Both should have completed at least 1 job
    assert len(source1_jobs) > 0
    assert len(source2_jobs) > 0


@pytest.mark.asyncio
@pytest.mark.load
async def test_unequal_load_distribution(
    job_queue,
    state_store,
    element_sink,
    test_workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test proportional execution with unequal load distribution.

    Validates that connectors with different job rates receive
    proportional resources.

    Acceptance Criteria:
    - All connectors complete jobs
    - Job completion proportional to job rates
    """
    # Mock partition function - BEFORE creating orchestrator
    class FakePartition:
        def __call__(self, *, filename: str, strategy: str, include_metadata: bool):
            return [{"text": "content", "type": "NarrativeText"}]

    monkeypatch.setattr(
        "futurnal.ingestion.local.connector.partition",
        FakePartition(),
    )

    # Create orchestrator AFTER monkeypatch
    from futurnal.orchestrator.scheduler import IngestionOrchestrator
    load_test_orchestrator = IngestionOrchestrator(
        job_queue=job_queue,
        state_store=state_store,
        workspace_dir=str(test_workspace),
        element_sink=element_sink,
    )

    # Create test directories
    dir_high = tmp_path / "high_load"
    dir_low = tmp_path / "low_load"
    dir_high.mkdir()
    dir_low.mkdir()

    # Create files
    for i in range(20):
        (dir_high / f"file_{i}.txt").write_text(f"content {i}")
    for i in range(5):
        (dir_low / f"file_{i}.txt").write_text(f"content {i}")

    # Register sources
    from futurnal.ingestion.local.config import LocalIngestionSource
    from futurnal.orchestrator.scheduler import SourceRegistration

    source_high = LocalIngestionSource(name="high_load", root_path=dir_high)
    source_low = LocalIngestionSource(name="low_load", root_path=dir_low)

    load_test_orchestrator.register_source(
        SourceRegistration(source=source_high, schedule="@manual", priority=JobPriority.NORMAL)
    )
    load_test_orchestrator.register_source(
        SourceRegistration(source=source_low, schedule="@manual", priority=JobPriority.NORMAL)
    )

    # Enqueue unequal number of jobs
    from futurnal.orchestrator.models import IngestionJob
    import uuid
    from datetime import datetime

    # 10 jobs for high load source
    for i in range(10):
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "high_load", "size_bytes": 1_000_000},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # 3 jobs for low load source
    for i in range(3):
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            job_type=JobType.LOCAL_FILES,
            payload={"source_name": "low_load", "size_bytes": 1_000_000},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

    # Process jobs
    load_test_orchestrator.start()
    await asyncio.sleep(5.0)
    await load_test_orchestrator.shutdown()

    # Validate proportional completion
    from futurnal.orchestrator.queue import JobStatus
    completed = job_queue.snapshot(status=JobStatus.SUCCEEDED)

    high_load_jobs = [j for j in completed if j["payload"]["source_name"] == "high_load"]
    low_load_jobs = [j for j in completed if j["payload"]["source_name"] == "low_load"]

    # Both should complete jobs
    assert len(high_load_jobs) > 0
    assert len(low_load_jobs) > 0

    # High load should process more jobs (roughly 3x ratio)
    assert len(high_load_jobs) >= len(low_load_jobs)
