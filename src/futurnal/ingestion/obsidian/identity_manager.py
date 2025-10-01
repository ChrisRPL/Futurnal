"""Note identity management for Obsidian vaults.

This module provides persistent UUID-based note identity management to ensure
that note IDs remain stable across renames and moves, preventing duplicate
nodes in the knowledge graph.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..local.state import FileRecord, StateStore

logger = logging.getLogger(__name__)


@dataclass
class NoteIdentity:
    """Represents a persistent note identity."""
    note_id: str
    current_path: Path
    created_at: datetime
    vault_id: str

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "note_id": self.note_id,
            "current_path": str(self.current_path),
            "created_at": self.created_at.isoformat(),
            "vault_id": self.vault_id
        }


class NoteIdentityManager:
    """Manages persistent note identities using UUID-based system.

    This class replaces the path-based note ID system with persistent UUIDs
    that remain stable across file renames and moves, ensuring no duplicate
    nodes are created in the knowledge graph.
    """

    def __init__(self, vault_id: str, vault_root: Path, db_path: Optional[Path] = None):
        """Initialize the note identity manager.

        Args:
            vault_id: Unique identifier for the vault
            vault_root: Root path of the vault
            db_path: Path to SQLite database (defaults to vault/.futurnal/identities.db)
        """
        self.vault_id = vault_id
        self.vault_root = vault_root

        # Default database path within vault
        if db_path is None:
            db_path = vault_root / ".futurnal" / "identities.db"

        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_database(self) -> None:
        """Initialize the SQLite database with required schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_identities (
                    note_id TEXT PRIMARY KEY,
                    current_path TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    vault_id TEXT NOT NULL,
                    UNIQUE(vault_id, current_path)
                )
            """)

            # Create index for efficient path lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vault_path
                ON note_identities(vault_id, current_path)
            """)

            # Create index for vault-based queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vault_id
                ON note_identities(vault_id)
            """)

            conn.commit()
            logger.debug(f"Initialized note identity database at {self.db_path}")

    def get_note_id(self, file_path: Path) -> str:
        """Get or create a persistent note ID for the given file path.

        Args:
            file_path: Path to the note file

        Returns:
            Persistent UUID-based note ID
        """
        # Only handle markdown files
        if file_path.suffix.lower() not in {'.md', '.markdown'}:
            raise ValueError(f"Note identity manager only handles markdown files, got: {file_path}")

        # Make path relative to vault root
        try:
            relative_path = file_path.relative_to(self.vault_root)
        except ValueError:
            # Path is not relative to vault root, use absolute path
            relative_path = file_path

        relative_path_str = str(relative_path)

        with sqlite3.connect(self.db_path) as conn:
            # Try to find existing note ID for this path
            cursor = conn.execute("""
                SELECT note_id FROM note_identities
                WHERE vault_id = ? AND current_path = ?
            """, (self.vault_id, relative_path_str))

            result = cursor.fetchone()
            if result:
                return result[0]

            # No existing note ID, create a new one
            note_id = str(uuid.uuid4())
            created_at = datetime.utcnow()

            conn.execute("""
                INSERT INTO note_identities (note_id, current_path, created_at, vault_id)
                VALUES (?, ?, ?, ?)
            """, (note_id, relative_path_str, created_at, self.vault_id))

            conn.commit()
            logger.info(f"Created new note ID {note_id} for {relative_path}")
            return note_id

    def update_path(self, note_id: str, new_path: Path) -> bool:
        """Update the path for an existing note ID.

        Args:
            note_id: The persistent note ID
            new_path: New path for the note

        Returns:
            True if update was successful, False if note ID not found
        """
        # Make path relative to vault root
        try:
            relative_path = new_path.relative_to(self.vault_root)
        except ValueError:
            relative_path = new_path

        relative_path_str = str(relative_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE note_identities
                SET current_path = ?
                WHERE note_id = ? AND vault_id = ?
            """, (relative_path_str, note_id, self.vault_id))

            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Updated path for note ID {note_id} to {relative_path}")
                return True
            else:
                logger.warning(f"Note ID {note_id} not found for update")
                return False

    def get_path_by_note_id(self, note_id: str) -> Optional[Path]:
        """Get the current path for a note ID.

        Args:
            note_id: The persistent note ID

        Returns:
            Current path of the note, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT current_path FROM note_identities
                WHERE note_id = ? AND vault_id = ?
            """, (note_id, self.vault_id))

            result = cursor.fetchone()
            if result:
                relative_path = Path(result[0])
                return self.vault_root / relative_path
            return None

    def get_note_id_by_path(self, file_path: Path) -> Optional[str]:
        """Get existing note ID for a path without creating a new one.

        Args:
            file_path: Path to the note file

        Returns:
            Existing note ID, or None if not found
        """
        try:
            relative_path = file_path.relative_to(self.vault_root)
        except ValueError:
            relative_path = file_path

        relative_path_str = str(relative_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT note_id FROM note_identities
                WHERE vault_id = ? AND current_path = ?
            """, (self.vault_id, relative_path_str))

            result = cursor.fetchone()
            return result[0] if result else None

    def remove_note(self, note_id: str) -> bool:
        """Remove a note identity.

        Args:
            note_id: The persistent note ID to remove

        Returns:
            True if removal was successful, False if note ID not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM note_identities
                WHERE note_id = ? AND vault_id = ?
            """, (note_id, self.vault_id))

            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Removed note ID {note_id}")
                return True
            else:
                logger.warning(f"Note ID {note_id} not found for removal")
                return False

    def list_all_identities(self) -> List[NoteIdentity]:
        """List all note identities for this vault.

        Returns:
            List of all note identities
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT note_id, current_path, created_at, vault_id
                FROM note_identities
                WHERE vault_id = ?
                ORDER BY created_at
            """, (self.vault_id,))

            identities = []
            for row in cursor.fetchall():
                note_id, path_str, created_at_str, vault_id = row
                path = self.vault_root / Path(path_str)
                created_at = datetime.fromisoformat(created_at_str)

                identities.append(NoteIdentity(
                    note_id=note_id,
                    current_path=path,
                    created_at=created_at,
                    vault_id=vault_id
                ))

            return identities

    def migrate_from_path_ids(self, state_store: StateStore) -> Dict[str, str]:
        """Migrate from path-based note IDs to UUID-based system.

        This method analyzes existing file records and creates persistent UUIDs
        for any markdown files that don't already have them.

        Args:
            state_store: State store containing existing file records

        Returns:
            Dictionary mapping old path-based IDs to new UUID-based IDs
        """
        migration_map = {}

        # Get all current markdown files from state store
        markdown_files = []
        for record in state_store.iter_all():
            if record.path.suffix.lower() in {'.md', '.markdown'}:
                markdown_files.append(record)

        logger.info(f"Starting migration for {len(markdown_files)} markdown files")

        for record in markdown_files:
            # Generate old path-based ID (what was used before)
            try:
                relative_path = record.path.relative_to(self.vault_root)
                old_path_id = str(relative_path.with_suffix(''))
            except ValueError:
                old_path_id = str(record.path.with_suffix(''))

            # Get or create UUID-based ID
            uuid_id = self.get_note_id(record.path)

            # Store the mapping
            migration_map[old_path_id] = uuid_id

            logger.debug(f"Migrated {old_path_id} -> {uuid_id}")

        logger.info(f"Migration completed: {len(migration_map)} path-based IDs converted to UUIDs")
        return migration_map

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the note identity database.

        Returns:
            Dictionary with database statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM note_identities WHERE vault_id = ?
            """, (self.vault_id,))

            total_notes = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT MIN(created_at), MAX(created_at)
                FROM note_identities WHERE vault_id = ?
            """, (self.vault_id,))

            date_range = cursor.fetchone()

            return {
                "total_notes": total_notes,
                "vault_id": self.vault_id,
                "db_path": str(self.db_path),
                "earliest_note": date_range[0] if date_range[0] else None,
                "latest_note": date_range[1] if date_range[1] else None
            }