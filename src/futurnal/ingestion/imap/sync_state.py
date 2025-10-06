"""IMAP sync state management and persistence.

This module implements the sync state tracking described in
``docs/phase-1/imap-connector-production-plan/06-incremental-sync-strategy.md``.
It provides persistent state storage for incremental IMAP synchronization using
UID tracking and MODSEQ support (RFC 7162).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sync state models
# ---------------------------------------------------------------------------


class ImapSyncState(BaseModel):
    """Persistent sync state for IMAP folder."""

    # Identity
    mailbox_id: str = Field(..., description="Mailbox identifier")
    folder: str = Field(..., description="IMAP folder name")

    # IMAP state
    uidvalidity: int = Field(..., description="UIDVALIDITY value from server")
    last_synced_uid: int = Field(default=0, ge=0, description="Highest UID seen")
    highest_modseq: Optional[int] = Field(
        default=None, ge=0, description="Highest MODSEQ for delta sync"
    )

    # Sync metadata
    last_sync_time: datetime = Field(..., description="Last sync timestamp")
    message_count: int = Field(default=0, ge=0, description="Total message count")
    last_exists_count: int = Field(
        default=0, ge=0, description="Last EXISTS count for deletion detection"
    )

    # Sync statistics
    total_syncs: int = Field(default=0, ge=0, description="Total sync operations")
    messages_synced: int = Field(default=0, ge=0, description="Total messages synced")
    messages_updated: int = Field(
        default=0, ge=0, description="Total messages updated"
    )
    messages_deleted: int = Field(
        default=0, ge=0, description="Total messages deleted"
    )
    sync_errors: int = Field(default=0, ge=0, description="Total sync errors")

    # Server capabilities
    supports_idle: bool = Field(default=False, description="Server supports IDLE")
    supports_modseq: bool = Field(
        default=False, description="Server supports MODSEQ/CONDSTORE"
    )
    supports_qresync: bool = Field(
        default=False, description="Server supports QRESYNC"
    )

    class Config:
        json_schema_extra = {
            "description": "Sync state for incremental IMAP synchronization"
        }


class SyncResult(BaseModel):
    """Result of a sync operation."""

    new_messages: List[int] = Field(
        default_factory=list, description="UIDs of new messages"
    )
    updated_messages: List[int] = Field(
        default_factory=list, description="UIDs of updated messages"
    )
    deleted_messages: List[int] = Field(
        default_factory=list, description="UIDs of deleted messages"
    )
    sync_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Sync duration"
    )
    errors: List[str] = Field(default_factory=list, description="Error messages")

    @property
    def has_changes(self) -> bool:
        """Check if sync found any changes."""
        return bool(
            self.new_messages or self.updated_messages or self.deleted_messages
        )

    @property
    def total_changes(self) -> int:
        """Total number of changes detected."""
        return (
            len(self.new_messages)
            + len(self.updated_messages)
            + len(self.deleted_messages)
        )


# ---------------------------------------------------------------------------
# Sync state persistence
# ---------------------------------------------------------------------------


SCHEMA = """
CREATE TABLE IF NOT EXISTS imap_sync_state (
    mailbox_id TEXT NOT NULL,
    folder TEXT NOT NULL,
    uidvalidity INTEGER NOT NULL,
    last_synced_uid INTEGER DEFAULT 0,
    highest_modseq INTEGER,
    last_sync_time TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    last_exists_count INTEGER DEFAULT 0,
    total_syncs INTEGER DEFAULT 0,
    messages_synced INTEGER DEFAULT 0,
    messages_updated INTEGER DEFAULT 0,
    messages_deleted INTEGER DEFAULT 0,
    sync_errors INTEGER DEFAULT 0,
    supports_idle BOOLEAN DEFAULT 0,
    supports_modseq BOOLEAN DEFAULT 0,
    supports_qresync BOOLEAN DEFAULT 0,
    PRIMARY KEY (mailbox_id, folder)
);

CREATE INDEX IF NOT EXISTS idx_mailbox_id ON imap_sync_state(mailbox_id);
CREATE INDEX IF NOT EXISTS idx_last_sync_time ON imap_sync_state(last_sync_time);
"""


class ImapSyncStateStore:
    """SQLite-backed state store for IMAP sync state tracking."""

    def __init__(self, path: Path) -> None:
        """Initialize state store.

        Args:
            path: Path to SQLite database file
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        """Close database connection."""
        self._conn.commit()
        self._conn.close()

    def upsert(self, state: ImapSyncState) -> None:
        """Insert or update sync state.

        Args:
            state: Sync state to persist
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO imap_sync_state(
                    mailbox_id, folder, uidvalidity, last_synced_uid, highest_modseq,
                    last_sync_time, message_count, last_exists_count,
                    total_syncs, messages_synced, messages_updated, messages_deleted, sync_errors,
                    supports_idle, supports_modseq, supports_qresync
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mailbox_id, folder) DO UPDATE SET
                    uidvalidity=excluded.uidvalidity,
                    last_synced_uid=excluded.last_synced_uid,
                    highest_modseq=excluded.highest_modseq,
                    last_sync_time=excluded.last_sync_time,
                    message_count=excluded.message_count,
                    last_exists_count=excluded.last_exists_count,
                    total_syncs=excluded.total_syncs,
                    messages_synced=excluded.messages_synced,
                    messages_updated=excluded.messages_updated,
                    messages_deleted=excluded.messages_deleted,
                    sync_errors=excluded.sync_errors,
                    supports_idle=excluded.supports_idle,
                    supports_modseq=excluded.supports_modseq,
                    supports_qresync=excluded.supports_qresync
                """,
                (
                    state.mailbox_id,
                    state.folder,
                    state.uidvalidity,
                    state.last_synced_uid,
                    state.highest_modseq,
                    state.last_sync_time.isoformat(),
                    state.message_count,
                    state.last_exists_count,
                    state.total_syncs,
                    state.messages_synced,
                    state.messages_updated,
                    state.messages_deleted,
                    state.sync_errors,
                    int(state.supports_idle),
                    int(state.supports_modseq),
                    int(state.supports_qresync),
                ),
            )

    def fetch(self, mailbox_id: str, folder: str) -> Optional[ImapSyncState]:
        """Fetch sync state for mailbox folder.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name

        Returns:
            Sync state if found, None otherwise
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT
                mailbox_id, folder, uidvalidity, last_synced_uid, highest_modseq,
                last_sync_time, message_count, last_exists_count,
                total_syncs, messages_synced, messages_updated, messages_deleted, sync_errors,
                supports_idle, supports_modseq, supports_qresync
            FROM imap_sync_state
            WHERE mailbox_id = ? AND folder = ?
            """,
            (mailbox_id, folder),
        )
        row = cur.fetchone()
        if not row:
            return None

        return ImapSyncState(
            mailbox_id=row[0],
            folder=row[1],
            uidvalidity=row[2],
            last_synced_uid=row[3],
            highest_modseq=row[4],
            last_sync_time=datetime.fromisoformat(row[5]),
            message_count=row[6],
            last_exists_count=row[7],
            total_syncs=row[8],
            messages_synced=row[9],
            messages_updated=row[10],
            messages_deleted=row[11],
            sync_errors=row[12],
            supports_idle=bool(row[13]),
            supports_modseq=bool(row[14]),
            supports_qresync=bool(row[15]),
        )

    def remove(self, mailbox_id: str, folder: str) -> None:
        """Remove sync state.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM imap_sync_state WHERE mailbox_id = ? AND folder = ?",
                (mailbox_id, folder),
            )

    def list_by_mailbox(self, mailbox_id: str) -> Iterator[ImapSyncState]:
        """List all sync states for a mailbox.

        Args:
            mailbox_id: Mailbox identifier

        Yields:
            Sync states for the mailbox
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT
                mailbox_id, folder, uidvalidity, last_synced_uid, highest_modseq,
                last_sync_time, message_count, last_exists_count,
                total_syncs, messages_synced, messages_updated, messages_deleted, sync_errors,
                supports_idle, supports_modseq, supports_qresync
            FROM imap_sync_state
            WHERE mailbox_id = ?
            ORDER BY folder
            """,
            (mailbox_id,),
        )

        for row in cur.fetchall():
            yield ImapSyncState(
                mailbox_id=row[0],
                folder=row[1],
                uidvalidity=row[2],
                last_synced_uid=row[3],
                highest_modseq=row[4],
                last_sync_time=datetime.fromisoformat(row[5]),
                message_count=row[6],
                last_exists_count=row[7],
                total_syncs=row[8],
                messages_synced=row[9],
                messages_updated=row[10],
                messages_deleted=row[11],
                sync_errors=row[12],
                supports_idle=bool(row[13]),
                supports_modseq=bool(row[14]),
                supports_qresync=bool(row[15]),
            )

    def iter_all(self) -> Iterator[ImapSyncState]:
        """Iterate all sync states.

        Yields:
            All sync states
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT
                mailbox_id, folder, uidvalidity, last_synced_uid, highest_modseq,
                last_sync_time, message_count, last_exists_count,
                total_syncs, messages_synced, messages_updated, messages_deleted, sync_errors,
                supports_idle, supports_modseq, supports_qresync
            FROM imap_sync_state
            ORDER BY mailbox_id, folder
            """
        )

        for row in cur.fetchall():
            yield ImapSyncState(
                mailbox_id=row[0],
                folder=row[1],
                uidvalidity=row[2],
                last_synced_uid=row[3],
                highest_modseq=row[4],
                last_sync_time=datetime.fromisoformat(row[5]),
                message_count=row[6],
                last_exists_count=row[7],
                total_syncs=row[8],
                messages_synced=row[9],
                messages_updated=row[10],
                messages_deleted=row[11],
                sync_errors=row[12],
                supports_idle=bool(row[13]),
                supports_modseq=bool(row[14]),
                supports_qresync=bool(row[15]),
            )


__all__ = [
    "ImapSyncState",
    "ImapSyncStateStore",
    "SyncResult",
]
