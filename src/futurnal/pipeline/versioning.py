"""Document versioning and provenance tracking for PKG diffs.

This module implements content-hash based change detection, temporal metadata tracking,
and document revision support for PKG versioning. Enables the Ghost to understand
document evolution over time and detect meaningful changes for experiential learning.

Key Features:
- SHA-256 content hashing for deterministic change detection
- Parent hash tracking for version chain traversal
- Temporal metadata (created/modified/ingested timestamps)
- Thread-safe SQLite-backed persistence
- Privacy-first design (no content stored, only metadata)
- Idempotent processing (same input = same hash)

Architecture:
- DocumentVersionStore: Low-level SQLite persistence layer
- ProvenanceTracker: High-level service for version tracking
- DocumentVersionRecord: Immutable version metadata container

Integration:
- Works with NormalizedMetadata (content_hash, parent_hash fields)
- Compatible with all connectors (local files, Obsidian, IMAP, GitHub)
- Privacy-aware logging (no content exposure)
- Audit trail for version tracking events
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .models import compute_content_hash


# ---------------------------------------------------------------------------
# Version Metadata Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentVersionRecord:
    """Immutable record of a document version.

    Represents a single point in a document's version history with
    content hash, parent linkage, and temporal metadata.

    Attributes:
        source_path: Unique identifier for the document (file path, note URI, etc.)
        content_hash: SHA-256 hash of document content
        parent_hash: SHA-256 hash of previous version (None for first version)
        created_at: Document creation timestamp (if known)
        modified_at: Document modification timestamp (if known)
        ingested_at: When this version was processed by Futurnal
        version_number: Sequential version counter (1-indexed)
    """

    source_path: str
    content_hash: str
    parent_hash: Optional[str]
    created_at: Optional[datetime]
    modified_at: Optional[datetime]
    ingested_at: datetime
    version_number: int

    def __post_init__(self):
        """Validate version record invariants."""
        if self.version_number < 1:
            raise ValueError("version_number must be >= 1")
        if len(self.content_hash) != 64:
            raise ValueError("content_hash must be valid SHA-256 hex (64 chars)")
        if self.parent_hash is not None and len(self.parent_hash) != 64:
            raise ValueError("parent_hash must be valid SHA-256 hex (64 chars)")
        if not self.source_path:
            raise ValueError("source_path cannot be empty")

        # Ensure timestamps are timezone-aware (UTC)
        if self.ingested_at.tzinfo is None:
            object.__setattr__(self, "ingested_at",
                             self.ingested_at.replace(tzinfo=timezone.utc))
        if self.created_at and self.created_at.tzinfo is None:
            object.__setattr__(self, "created_at",
                             self.created_at.replace(tzinfo=timezone.utc))
        if self.modified_at and self.modified_at.tzinfo is None:
            object.__setattr__(self, "modified_at",
                             self.modified_at.replace(tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Database Schema
# ---------------------------------------------------------------------------


VERSIONING_SCHEMA = """
-- Current version tracking (one row per unique source_path)
CREATE TABLE IF NOT EXISTS document_versions (
    source_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    parent_hash TEXT,
    created_at REAL,
    modified_at REAL,
    ingested_at REAL NOT NULL,
    version_number INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
);

-- Full version history (append-only log)
CREATE TABLE IF NOT EXISTS version_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    parent_hash TEXT,
    created_at REAL,
    modified_at REAL,
    ingested_at REAL NOT NULL,
    version_number INTEGER NOT NULL,
    recorded_at REAL NOT NULL
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_version_source ON version_history(source_path);
CREATE INDEX IF NOT EXISTS idx_version_hash ON version_history(content_hash);
CREATE INDEX IF NOT EXISTS idx_version_timestamp ON version_history(ingested_at);
"""


# ---------------------------------------------------------------------------
# Document Version Store
# ---------------------------------------------------------------------------


class DocumentVersionStore:
    """SQLite-backed persistent storage for document version tracking.

    Maintains two tables:
    1. document_versions: Current state (latest version per source_path)
    2. version_history: Complete temporal log of all versions

    Thread-safe with connection-level locking and WAL mode for concurrent reads.
    All writes happen within SQLite transactions for crash recovery.

    Privacy Guarantees:
    - No document content is stored, only SHA-256 hashes
    - No sensitive metadata is logged
    - Path redaction should be applied by caller for audit logs
    """

    def __init__(self, path: Path) -> None:
        """Initialize version store with SQLite database.

        Args:
            path: Path to SQLite database file
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Initialize database with WAL mode for concurrent access
        with self._lock:
            self._conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,  # We handle threading with _lock
                timeout=30.0  # Wait up to 30s for locks
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")  # Balance durability/performance
            self._conn.executescript(VERSIONING_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        """Close database connection.

        Should be called when store is no longer needed to ensure
        clean shutdown and WAL checkpoint.
        """
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def get_current_version(self, source_path: str) -> Optional[DocumentVersionRecord]:
        """Retrieve current version record for a document.

        Args:
            source_path: Unique document identifier

        Returns:
            Current version record, or None if document has never been versioned
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT source_path, content_hash, parent_hash,
                       created_at, modified_at, ingested_at, version_number
                FROM document_versions
                WHERE source_path = ?
                """,
                (source_path,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_record(row)

    def record_version(
        self,
        *,
        source_path: str,
        content_hash: str,
        parent_hash: Optional[str],
        created_at: Optional[datetime],
        modified_at: Optional[datetime],
        ingested_at: datetime
    ) -> DocumentVersionRecord:
        """Record a new document version.

        Updates both current version table and appends to version history.
        Automatically increments version number based on existing versions.

        Args:
            source_path: Unique document identifier
            content_hash: SHA-256 hash of document content
            parent_hash: Hash of previous version (None for first version)
            created_at: Document creation timestamp (if known)
            modified_at: Document modification timestamp (if known)
            ingested_at: When this version was processed

        Returns:
            Newly created version record with assigned version_number

        Raises:
            ValueError: If content_hash or parent_hash are invalid
        """
        # Validate hash formats
        if len(content_hash) != 64:
            raise ValueError("content_hash must be valid SHA-256 hex (64 chars)")
        if parent_hash is not None and len(parent_hash) != 64:
            raise ValueError("parent_hash must be valid SHA-256 hex (64 chars)")

        # Ensure timezone-aware
        if ingested_at.tzinfo is None:
            ingested_at = ingested_at.replace(tzinfo=timezone.utc)
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if modified_at and modified_at.tzinfo is None:
            modified_at = modified_at.replace(tzinfo=timezone.utc)

        with self._lock:
            # Determine version number
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT version_number FROM document_versions WHERE source_path = ?",
                (source_path,)
            )
            row = cursor.fetchone()
            version_number = (row[0] + 1) if row else 1

            now = datetime.now(timezone.utc).timestamp()

            # Convert datetimes to timestamps
            created_ts = created_at.timestamp() if created_at else None
            modified_ts = modified_at.timestamp() if modified_at else None
            ingested_ts = ingested_at.timestamp()

            # Update current version
            cursor.execute(
                """
                INSERT INTO document_versions
                (source_path, content_hash, parent_hash, created_at, modified_at,
                 ingested_at, version_number, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    parent_hash = excluded.parent_hash,
                    modified_at = excluded.modified_at,
                    ingested_at = excluded.ingested_at,
                    version_number = excluded.version_number,
                    updated_at = excluded.updated_at
                """,
                (source_path, content_hash, parent_hash, created_ts, modified_ts,
                 ingested_ts, version_number, now)
            )

            # Append to version history
            cursor.execute(
                """
                INSERT INTO version_history
                (source_path, content_hash, parent_hash, created_at, modified_at,
                 ingested_at, version_number, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_path, content_hash, parent_hash, created_ts, modified_ts,
                 ingested_ts, version_number, now)
            )

            self._conn.commit()

            return DocumentVersionRecord(
                source_path=source_path,
                content_hash=content_hash,
                parent_hash=parent_hash,
                created_at=created_at,
                modified_at=modified_at,
                ingested_at=ingested_at,
                version_number=version_number
            )

    def get_version_history(
        self,
        source_path: str,
        limit: Optional[int] = None
    ) -> List[DocumentVersionRecord]:
        """Retrieve version history for a document.

        Returns versions in reverse chronological order (newest first).

        Args:
            source_path: Unique document identifier
            limit: Maximum number of versions to return (None for all)

        Returns:
            List of version records, newest first
        """
        with self._lock:
            cursor = self._conn.cursor()

            query = """
                SELECT source_path, content_hash, parent_hash,
                       created_at, modified_at, ingested_at, version_number
                FROM version_history
                WHERE source_path = ?
                ORDER BY version_number DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, (source_path,))

            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_version_count(self, source_path: str) -> int:
        """Get total number of versions for a document.

        Args:
            source_path: Unique document identifier

        Returns:
            Total version count (0 if never versioned)
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM version_history WHERE source_path = ?",
                (source_path,)
            )
            return cursor.fetchone()[0]

    def iter_all_documents(self) -> Iterator[DocumentVersionRecord]:
        """Iterate over all documents with version tracking.

        Yields current version record for each tracked document.

        Yields:
            DocumentVersionRecord for each document
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT source_path, content_hash, parent_hash,
                       created_at, modified_at, ingested_at, version_number
                FROM document_versions
                ORDER BY source_path
                """
            )

            for row in cursor.fetchall():
                yield self._row_to_record(row)

    def get_document_count(self) -> int:
        """Get total number of documents with version tracking.

        Returns:
            Total document count
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_versions")
            return cursor.fetchone()[0]

    def _row_to_record(self, row: tuple) -> DocumentVersionRecord:
        """Convert database row to DocumentVersionRecord.

        Args:
            row: Tuple from SQL query (source_path, content_hash, parent_hash,
                 created_at, modified_at, ingested_at, version_number)

        Returns:
            DocumentVersionRecord instance
        """
        source_path, content_hash, parent_hash, created_ts, modified_ts, ingested_ts, version_number = row

        # Convert timestamps back to datetime
        created_at = datetime.fromtimestamp(created_ts, tz=timezone.utc) if created_ts else None
        modified_at = datetime.fromtimestamp(modified_ts, tz=timezone.utc) if modified_ts else None
        ingested_at = datetime.fromtimestamp(ingested_ts, tz=timezone.utc)

        return DocumentVersionRecord(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=parent_hash,
            created_at=created_at,
            modified_at=modified_at,
            ingested_at=ingested_at,
            version_number=version_number
        )


# ---------------------------------------------------------------------------
# Provenance Tracker Service
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """High-level service for document provenance and version tracking.

    Provides simplified API for change detection and version management.
    Integrates with DocumentVersionStore for persistence and audit logging.

    Usage:
        tracker = ProvenanceTracker(version_store)

        # Detect if document changed
        has_changed, prev_hash = await tracker.detect_change(
            source_path="/vault/note.md",
            content_hash="abc123..."
        )

        # Record new version
        await tracker.record_version(
            source_path="/vault/note.md",
            content_hash="abc123...",
            parent_hash=prev_hash,
            timestamp=datetime.now(timezone.utc)
        )
    """

    def __init__(self, version_store: DocumentVersionStore) -> None:
        """Initialize provenance tracker.

        Args:
            version_store: Document version store for persistence
        """
        self.version_store = version_store

    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content.

        Delegates to compute_content_hash from models.py for consistency.

        Args:
            content: Document content to hash

        Returns:
            SHA-256 hash as lowercase hex string (64 chars)
        """
        return compute_content_hash(content)

    async def detect_change(
        self,
        *,
        source_path: str,
        content_hash: str
    ) -> Tuple[bool, Optional[str]]:
        """Detect if document has changed since last processing.

        Compares current content hash against stored version to determine
        if document is new or has been modified.

        Args:
            source_path: Unique document identifier
            content_hash: SHA-256 hash of current content

        Returns:
            Tuple of (has_changed: bool, previous_hash: Optional[str])
            - has_changed: True if document is new or modified
            - previous_hash: Hash of previous version, or None if new document
        """
        current_version = self.version_store.get_current_version(source_path)

        if current_version is None:
            # New document - never seen before
            return True, None

        # Compare hashes to detect change
        has_changed = current_version.content_hash != content_hash
        previous_hash = current_version.content_hash if has_changed else None

        return has_changed, previous_hash

    async def record_version(
        self,
        *,
        source_path: str,
        content_hash: str,
        parent_hash: Optional[str],
        timestamp: datetime,
        created_at: Optional[datetime] = None,
        modified_at: Optional[datetime] = None
    ) -> DocumentVersionRecord:
        """Record document version in version store.

        Creates version record with temporal metadata and parent linkage.

        Args:
            source_path: Unique document identifier
            content_hash: SHA-256 hash of document content
            parent_hash: Hash of previous version (from detect_change)
            timestamp: When document was ingested/processed
            created_at: Document creation timestamp (if known)
            modified_at: Document modification timestamp (if known)

        Returns:
            Newly created version record
        """
        return self.version_store.record_version(
            source_path=source_path,
            content_hash=content_hash,
            parent_hash=parent_hash,
            created_at=created_at,
            modified_at=modified_at,
            ingested_at=timestamp
        )

    def get_version_history(
        self,
        source_path: str,
        limit: Optional[int] = None
    ) -> List[DocumentVersionRecord]:
        """Retrieve version history for a document.

        Args:
            source_path: Unique document identifier
            limit: Maximum versions to return (None for all)

        Returns:
            List of version records, newest first
        """
        return self.version_store.get_version_history(source_path, limit=limit)

    def get_version_count(self, source_path: str) -> int:
        """Get total number of versions for a document.

        Args:
            source_path: Unique document identifier

        Returns:
            Total version count (0 if never versioned)
        """
        return self.version_store.get_version_count(source_path)
