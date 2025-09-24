"""Tests for the persistent job queue."""

from datetime import datetime, timedelta
from pathlib import Path

from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.queue import JobQueue


def make_job(job_id: str, scheduled_offset: int = 0) -> IngestionJob:
    scheduled_for = datetime.utcnow() + timedelta(seconds=scheduled_offset)
    return IngestionJob(
        job_id=job_id,
        job_type=JobType.LOCAL_FILES,
        payload={"source_name": "docs"},
        priority=JobPriority.NORMAL,
        scheduled_for=scheduled_for,
    )


def test_enqueue_and_fetch(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue.db")
    job = make_job("1")
    queue.enqueue(job)

    fetched = list(queue.fetch_pending())
    assert len(fetched) == 1
    assert fetched[0].job_id == "1"


def test_scheduled_jobs_respect_future_time(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue.db")
    future_job = make_job("future", scheduled_offset=60)
    queue.enqueue(future_job)

    assert list(queue.fetch_pending()) == []


def test_status_transitions(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue.db")
    job = make_job("job")
    queue.enqueue(job)

    queue.mark_running("job")
    queue.mark_completed("job")

    assert list(queue.fetch_pending()) == []


def test_reschedule(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue.db")
    job = make_job("job")
    queue.enqueue(job)
    queue.mark_running("job")
    queue.mark_failed("job")
    queue.reschedule("job", retry_delay_seconds=60)

    pending = list(queue.fetch_pending())
    assert pending


