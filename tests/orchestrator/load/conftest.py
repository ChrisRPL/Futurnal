"""Shared fixtures and utilities for load tests."""

import asyncio
import time
from pathlib import Path
from typing import Iterator

import pytest

from futurnal.ingestion.local.config import LocalIngestionSource
from futurnal.ingestion.local.state import StateStore
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue
from futurnal.orchestrator.scheduler import IngestionOrchestrator
from futurnal.pipeline import NormalizationSink


class MemoryStateStore(StateStore):
    """In-memory state store for testing."""

    def __init__(self) -> None:
        super().__init__(Path(":memory:"))


class MockJobExecutor:
    """Mock job executor for load testing.

    Simulates job execution without actual file processing.
    """

    def __init__(self, *, execution_time_seconds: float = 0.1, fail_rate: float = 0.0):
        """Initialize mock executor.

        Args:
            execution_time_seconds: Simulated execution time per job
            fail_rate: Probability of job failure (0.0-1.0)
        """
        self.execution_time_seconds = execution_time_seconds
        self.fail_rate = fail_rate
        self.executed_jobs = []

    async def execute(self, job: IngestionJob) -> tuple[int, int]:
        """Execute a mock job.

        Args:
            job: Job to execute

        Returns:
            Tuple of (files_processed, bytes_processed)

        Raises:
            RuntimeError: If job fails (based on fail_rate)
        """
        self.executed_jobs.append(job.job_id)

        # Simulate execution time
        await asyncio.sleep(self.execution_time_seconds)

        # Simulate random failures
        if self.fail_rate > 0:
            import random
            if random.random() < self.fail_rate:
                raise RuntimeError(f"Simulated failure for job {job.job_id}")

        # Return simulated metrics
        size_bytes = job.payload.get("size_bytes", 1_000_000)
        return (1, size_bytes)


class StubPKGWriter:
    """Stub PKG writer for testing."""

    def __init__(self) -> None:
        self.documents = []

    def write_document(self, payload: dict) -> None:
        self.documents.append(payload)

    def remove_document(self, sha256: str) -> None:
        self.documents = [doc for doc in self.documents if doc["sha256"] != sha256]


class StubVectorWriter:
    """Stub vector writer for testing."""

    def __init__(self) -> None:
        self.embeddings = []

    def write_embedding(self, payload: dict) -> None:
        self.embeddings.append(payload)

    def remove_embedding(self, sha256: str) -> None:
        self.embeddings = [emb for emb in self.embeddings if emb["sha256"] != sha256]


@pytest.fixture
def test_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for load tests.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to test workspace
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def job_queue(tmp_path: Path) -> JobQueue:
    """Create a job queue for testing.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        JobQueue instance
    """
    return JobQueue(tmp_path / "queue.db")


@pytest.fixture
def state_store() -> StateStore:
    """Create a state store for testing.

    Returns:
        MemoryStateStore instance
    """
    return MemoryStateStore()


@pytest.fixture
def element_sink() -> NormalizationSink:
    """Create an element sink for testing.

    Returns:
        NormalizationSink with stub writers
    """
    pkg_writer = StubPKGWriter()
    vector_writer = StubVectorWriter()
    return NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)


@pytest.fixture
def load_test_orchestrator(
    job_queue: JobQueue,
    state_store: StateStore,
    test_workspace: Path,
    element_sink: NormalizationSink,
) -> Iterator[IngestionOrchestrator]:
    """Create an orchestrator configured for load testing.

    Args:
        job_queue: Job queue fixture
        state_store: State store fixture
        test_workspace: Test workspace fixture
        element_sink: Element sink fixture

    Yields:
        IngestionOrchestrator instance
    """
    orchestrator = IngestionOrchestrator(
        job_queue=job_queue,
        state_store=state_store,
        workspace_dir=str(test_workspace),
        element_sink=element_sink,
    )

    yield orchestrator

    # Cleanup: ensure orchestrator is stopped
    if orchestrator._running:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(orchestrator.shutdown())


@pytest.fixture
def mock_fast_executor() -> MockJobExecutor:
    """Create a fast mock executor (~100ms per job).

    Returns:
        MockJobExecutor with fast execution
    """
    return MockJobExecutor(execution_time_seconds=0.1)


@pytest.fixture
def mock_slow_executor() -> MockJobExecutor:
    """Create a slow mock executor (~1s per job).

    Returns:
        MockJobExecutor with slow execution
    """
    return MockJobExecutor(execution_time_seconds=1.0)


@pytest.fixture
def mock_failing_executor() -> MockJobExecutor:
    """Create a mock executor with 20% failure rate.

    Returns:
        MockJobExecutor with failures
    """
    return MockJobExecutor(execution_time_seconds=0.1, fail_rate=0.2)


def create_test_job(
    job_type: JobType,
    priority: JobPriority = JobPriority.NORMAL,
    size_bytes: int = 1_000_000,
) -> IngestionJob:
    """Create a test job.

    Args:
        job_type: Type of job
        priority: Job priority (default: NORMAL)
        size_bytes: Simulated size in bytes (default: 1 MB)

    Returns:
        IngestionJob instance
    """
    import uuid
    from datetime import datetime

    return IngestionJob(
        job_id=str(uuid.uuid4()),
        job_type=job_type,
        payload={
            "source_name": f"test_{job_type.value}",
            "size_bytes": size_bytes,
            "load_test": True,
        },
        priority=priority,
        scheduled_for=datetime.utcnow(),
    )


async def wait_for_queue_drain(
    queue: JobQueue,
    timeout: float = 30.0,
    check_interval: float = 0.5,
) -> bool:
    """Wait for queue to drain (no pending or running jobs).

    Args:
        queue: Job queue to monitor
        timeout: Maximum time to wait in seconds (default: 30)
        check_interval: How often to check (default: 0.5s)

    Returns:
        True if queue drained, False if timeout
    """
    start = time.perf_counter()
    end_time = start + timeout

    while time.perf_counter() < end_time:
        pending = queue.pending_count()
        running = queue.running_count()

        if pending == 0 and running == 0:
            return True

        await asyncio.sleep(check_interval)

    return False


def collect_job_metrics(
    queue: JobQueue,
    start_time: float,
    job_ids: list[str] | None = None,
) -> dict:
    """Collect metrics from completed jobs.

    Args:
        queue: Job queue to query
        start_time: Test start time (perf_counter)
        job_ids: Optional list of job IDs to filter (default: all)

    Returns:
        Dictionary with metrics:
        - total_jobs: Total jobs completed
        - total_bytes: Total bytes processed
        - avg_latency: Average job latency
        - throughput_mbps: Throughput in MB/s
    """
    from futurnal.orchestrator.queue import JobStatus

    all_jobs = queue.snapshot(status=JobStatus.SUCCEEDED)

    if job_ids:
        jobs = [j for j in all_jobs if j["job_id"] in job_ids]
    else:
        jobs = all_jobs

    if not jobs:
        return {
            "total_jobs": 0,
            "total_bytes": 0,
            "avg_latency": 0.0,
            "throughput_mbps": 0.0,
        }

    total_bytes = sum(j["payload"].get("bytes_processed", 0) for j in jobs)
    duration = time.perf_counter() - start_time

    # Calculate average latency
    from datetime import datetime
    latencies = []
    for job in jobs:
        created_at = datetime.fromisoformat(job["created_at"])
        updated_at = datetime.fromisoformat(job["updated_at"])
        latency = (updated_at - created_at).total_seconds()
        latencies.append(latency)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    throughput_mbps = (total_bytes / (1024 * 1024)) / duration if duration > 0 else 0.0

    return {
        "total_jobs": len(jobs),
        "total_bytes": total_bytes,
        "avg_latency": avg_latency,
        "throughput_mbps": throughput_mbps,
    }
