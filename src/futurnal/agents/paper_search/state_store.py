"""Persistent state tracking for paper downloads and ingestion."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_state (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    local_path TEXT,
    download_status TEXT NOT NULL DEFAULT 'pending',
    ingestion_status TEXT NOT NULL DEFAULT 'pending',
    downloaded_at TEXT,
    ingested_at TEXT,
    file_size_bytes INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_download_status ON paper_state(download_status);
CREATE INDEX IF NOT EXISTS idx_ingestion_status ON paper_state(ingestion_status);
"""


@dataclass
class PaperRecord:
    """Describes tracked state for a paper."""

    paper_id: str
    title: str
    local_path: Optional[str] = None
    download_status: str = "pending"  # pending, downloaded, failed
    ingestion_status: str = "pending"  # pending, queued, processing, completed, failed
    downloaded_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None
    file_size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "paperId": self.paper_id,
            "title": self.title,
            "localPath": self.local_path,
            "downloadStatus": self.download_status,
            "ingestionStatus": self.ingestion_status,
            "downloadedAt": self.downloaded_at.isoformat() if self.downloaded_at else None,
            "ingestedAt": self.ingested_at.isoformat() if self.ingested_at else None,
            "fileSizeBytes": self.file_size_bytes,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class PaperStateStore:
    """SQLite-backed state store for paper download and ingestion tracking."""

    def __init__(self, path: Path) -> None:
        """Initialize the state store.

        Args:
            path: Path to the SQLite database file.
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "PaperStateStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def upsert(self, record: PaperRecord) -> None:
        """Insert or update a paper record."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO paper_state(
                    paper_id, title, local_path, download_status, ingestion_status,
                    downloaded_at, ingested_at, file_size_bytes, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title=excluded.title,
                    local_path=excluded.local_path,
                    download_status=excluded.download_status,
                    ingestion_status=excluded.ingestion_status,
                    downloaded_at=excluded.downloaded_at,
                    ingested_at=excluded.ingested_at,
                    file_size_bytes=excluded.file_size_bytes,
                    metadata=excluded.metadata
                """,
                (
                    record.paper_id,
                    record.title,
                    record.local_path,
                    record.download_status,
                    record.ingestion_status,
                    record.downloaded_at.isoformat() if record.downloaded_at else None,
                    record.ingested_at.isoformat() if record.ingested_at else None,
                    record.file_size_bytes,
                    json.dumps(record.metadata),
                ),
            )

    def get(self, paper_id: str) -> Optional[PaperRecord]:
        """Get a paper record by ID."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT paper_id, title, local_path, download_status, ingestion_status,
                   downloaded_at, ingested_at, file_size_bytes, metadata
            FROM paper_state WHERE paper_id = ?
            """,
            (paper_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def mark_downloaded(
        self,
        paper_id: str,
        title: str,
        local_path: str,
        file_size_bytes: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a paper as downloaded."""
        record = self.get(paper_id)
        if record:
            record.local_path = local_path
            record.download_status = "downloaded"
            record.downloaded_at = datetime.now()
            record.file_size_bytes = file_size_bytes
            if metadata:
                record.metadata.update(metadata)
        else:
            record = PaperRecord(
                paper_id=paper_id,
                title=title,
                local_path=local_path,
                download_status="downloaded",
                downloaded_at=datetime.now(),
                file_size_bytes=file_size_bytes,
                metadata=metadata or {},
            )
        self.upsert(record)

    def mark_download_failed(self, paper_id: str, title: str, error: Optional[str] = None) -> None:
        """Mark a paper download as failed."""
        record = self.get(paper_id)
        if record:
            record.download_status = "failed"
            if error:
                record.metadata["download_error"] = error
        else:
            record = PaperRecord(
                paper_id=paper_id,
                title=title,
                download_status="failed",
                metadata={"download_error": error} if error else {},
            )
        self.upsert(record)

    def mark_ingestion_queued(self, paper_id: str) -> None:
        """Mark a paper as queued for ingestion."""
        record = self.get(paper_id)
        if record:
            record.ingestion_status = "queued"
            self.upsert(record)

    def mark_ingestion_processing(self, paper_id: str) -> None:
        """Mark a paper as being processed."""
        record = self.get(paper_id)
        if record:
            record.ingestion_status = "processing"
            self.upsert(record)

    def mark_ingestion_completed(self, paper_id: str) -> None:
        """Mark a paper ingestion as completed."""
        record = self.get(paper_id)
        if record:
            record.ingestion_status = "completed"
            record.ingested_at = datetime.now()
            self.upsert(record)

    def mark_ingestion_failed(self, paper_id: str, error: Optional[str] = None) -> None:
        """Mark a paper ingestion as failed."""
        record = self.get(paper_id)
        if record:
            record.ingestion_status = "failed"
            if error:
                record.metadata["ingestion_error"] = error
            self.upsert(record)

    def get_by_status(
        self,
        download_status: Optional[str] = None,
        ingestion_status: Optional[str] = None,
    ) -> List[PaperRecord]:
        """Get papers by status."""
        conditions = []
        params = []

        if download_status:
            conditions.append("download_status = ?")
            params.append(download_status)
        if ingestion_status:
            conditions.append("ingestion_status = ?")
            params.append(ingestion_status)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cur = self._conn.cursor()
        cur.execute(
            f"""
            SELECT paper_id, title, local_path, download_status, ingestion_status,
                   downloaded_at, ingested_at, file_size_bytes, metadata
            FROM paper_state WHERE {where_clause}
            """,
            params,
        )
        return [self._row_to_record(row) for row in cur.fetchall()]

    def get_unprocessed_papers(self) -> List[PaperRecord]:
        """Get papers that are downloaded but not yet ingested."""
        return self.get_by_status(download_status="downloaded", ingestion_status="pending")

    def get_pending_ingestion(self) -> List[PaperRecord]:
        """Get papers that are queued for ingestion."""
        return self.get_by_status(ingestion_status="queued")

    def iter_all(self) -> Iterator[PaperRecord]:
        """Iterate over all paper records."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT paper_id, title, local_path, download_status, ingestion_status,
                   downloaded_at, ingested_at, file_size_bytes, metadata
            FROM paper_state
            """
        )
        for row in cur.fetchall():
            yield self._row_to_record(row)

    def count_by_status(self) -> Dict[str, Dict[str, int]]:
        """Get counts grouped by status."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT download_status, ingestion_status, COUNT(*)
            FROM paper_state
            GROUP BY download_status, ingestion_status
            """
        )
        result: Dict[str, Dict[str, int]] = {
            "download": {},
            "ingestion": {},
        }
        for row in cur.fetchall():
            dl_status, ing_status, count = row
            result["download"][dl_status] = result["download"].get(dl_status, 0) + count
            result["ingestion"][ing_status] = result["ingestion"].get(ing_status, 0) + count
        return result

    def remove(self, paper_id: str) -> None:
        """Remove a paper record."""
        with self._conn:
            self._conn.execute("DELETE FROM paper_state WHERE paper_id = ?", (paper_id,))

    def _row_to_record(self, row: tuple) -> PaperRecord:
        """Convert a database row to a PaperRecord."""
        return PaperRecord(
            paper_id=row[0],
            title=row[1],
            local_path=row[2],
            download_status=row[3],
            ingestion_status=row[4],
            downloaded_at=datetime.fromisoformat(row[5]) if row[5] else None,
            ingested_at=datetime.fromisoformat(row[6]) if row[6] else None,
            file_size_bytes=row[7] or 0,
            metadata=json.loads(row[8]) if row[8] else {},
        )


def get_default_state_store() -> PaperStateStore:
    """Get the default paper state store."""
    default_path = Path.home() / ".futurnal" / "papers" / "state.db"
    return PaperStateStore(default_path)
