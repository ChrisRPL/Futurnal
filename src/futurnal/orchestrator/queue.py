"""Persistent job queue for ingestion orchestrator."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator

from .models import IngestionJob, JobPriority, JobType


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    priority INTEGER NOT NULL,
    scheduled_for TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class JobQueue:
    """SQLite-backed persistent queue for ingestion jobs."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(SCHEMA)

    def enqueue(self, job: IngestionJob) -> None:
        payload_json = json.dumps(job.payload)
        scheduled_for = job.scheduled_for.isoformat() if job.scheduled_for else None
        now = datetime.utcnow().isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO jobs(job_id, job_type, payload, priority, scheduled_for, status, attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', 0, COALESCE((SELECT created_at FROM jobs WHERE job_id = ?), ?), ?)
                """,
                (
                    job.job_id,
                    job.job_type.value,
                    payload_json,
                    job.priority.value,
                    scheduled_for,
                    job.job_id,
                    now,
                    now,
                ),
            )

    def fetch_pending(self, limit: int = 10) -> Iterator[IngestionJob]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT job_id, job_type, payload, priority, scheduled_for
            FROM jobs
            WHERE status = 'pending'
              AND (scheduled_for IS NULL OR scheduled_for <= ?)
            ORDER BY priority DESC, COALESCE(scheduled_for, datetime('now'))
            LIMIT ?
            """,
            (datetime.utcnow().isoformat(), limit),
        )
        for job_id, job_type, payload, priority, scheduled_for in cur.fetchall():
            yield IngestionJob(
                job_id=job_id,
                job_type=JobType(job_type),
                payload=json.loads(payload),
                priority=JobPriority(priority),
                scheduled_for=datetime.fromisoformat(scheduled_for) if scheduled_for else None,
            )

    def mark_running(self, job_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = ? WHERE job_id = ?",
                (datetime.utcnow().isoformat(), job_id),
            )

    def mark_completed(self, job_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE jobs SET status = 'succeeded', updated_at = ? WHERE job_id = ?",
                (datetime.utcnow().isoformat(), job_id),
            )

    def mark_failed(self, job_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE jobs SET status = 'failed', updated_at = ? WHERE job_id = ?",
                (datetime.utcnow().isoformat(), job_id),
            )


