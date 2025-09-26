"""Persistent state tracking for local ingestion."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS file_state (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha256 TEXT NOT NULL
);
"""


@dataclass
class FileRecord:
    """Describes tracked metadata for a file."""

    path: Path
    size: int
    mtime: float
    sha256: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "path": str(self.path),
                "size": self.size,
                "mtime": self.mtime,
                "sha256": self.sha256,
            }
        )


class StateStore:
    """SQLite-backed state store for file change detection."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(SCHEMA)

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def upsert(self, record: FileRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO file_state(path, size, mtime, sha256)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime=excluded.mtime, sha256=excluded.sha256
                """,
                (str(record.path), record.size, record.mtime, record.sha256),
            )

    def remove(self, path: Path) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM file_state WHERE path = ?", (str(path),))

    def fetch(self, path: Path) -> Optional[FileRecord]:
        cur = self._conn.cursor()
        cur.execute("SELECT path, size, mtime, sha256 FROM file_state WHERE path = ?", (str(path),))
        row = cur.fetchone()
        if not row:
            return None
        return FileRecord(Path(row[0]), int(row[1]), float(row[2]), row[3])

    def iter_all(self) -> Iterator[FileRecord]:
        cur = self._conn.cursor()
        for row in cur.execute("SELECT path, size, mtime, sha256 FROM file_state"):
            yield FileRecord(Path(row[0]), int(row[1]), float(row[2]), row[3])

    def bulk_remove_missing(self, existing_paths: Iterable[Path]) -> None:
        existing_set = {str(path) for path in existing_paths}
        with self._conn:
            self._conn.executemany(
                "DELETE FROM file_state WHERE path = ?",
                [(path,) for path in self._paths_not_in(existing_set)],
            )

    def _paths_not_in(self, allowed: set[str]) -> Iterator[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT path FROM file_state")
        for (path,) in cur.fetchall():
            if path not in allowed:
                yield path


def compute_sha256(path: Path, chunk_size: int = 64 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                digest.update(chunk)
    except (FileNotFoundError, PermissionError, OSError):
        raise
    return digest.hexdigest()


