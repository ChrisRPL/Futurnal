"""Database backup, restore, and maintenance utilities.

Provides utilities for managing the orchestrator's SQLite queue database,
including backup creation, restoration, integrity checking, and cleanup.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class DatabaseError(Exception):
    """Base exception for database utility errors."""


class BackupError(DatabaseError):
    """Raised when database backup fails."""


class RestoreError(DatabaseError):
    """Raised when database restore fails."""


class IntegrityError(DatabaseError):
    """Raised when database integrity check fails."""


class DatabaseManager:
    """Manages queue database backup, restore, and maintenance operations."""

    def __init__(self, database_path: Path, backup_dir: Optional[Path] = None) -> None:
        """Initialize database manager.

        Args:
            database_path: Path to SQLite database (e.g., ~/.futurnal/queue/jobs.db)
            backup_dir: Directory for backups (default: database_path.parent / "backups")
        """
        self._db_path = database_path
        self._backup_dir = backup_dir or (database_path.parent / "backups")
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def backup(self, comment: Optional[str] = None) -> Path:
        """Create a backup of the database.

        Args:
            comment: Optional comment to include in backup filename

        Returns:
            Path to created backup file

        Raises:
            BackupError: If backup creation fails
        """
        if not self._db_path.exists():
            raise BackupError(f"Database not found: {self._db_path}")

        # Generate backup filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        comment_part = f"-{comment}" if comment else ""
        backup_filename = f"queue-{timestamp}{comment_part}.db"
        backup_path = self._backup_dir / backup_filename

        try:
            # Use SQLite's backup API for consistency
            source_conn = sqlite3.connect(self._db_path)
            backup_conn = sqlite3.connect(backup_path)

            with backup_conn:
                source_conn.backup(backup_conn)

            source_conn.close()
            backup_conn.close()

            return backup_path

        except Exception as e:
            raise BackupError(f"Failed to create backup: {e}")

    def restore(self, backup_path: Path, force: bool = False) -> None:
        """Restore database from a backup.

        Args:
            backup_path: Path to backup file
            force: If True, overwrite existing database without confirmation

        Raises:
            RestoreError: If restore fails
        """
        if not backup_path.exists():
            raise RestoreError(f"Backup file not found: {backup_path}")

        # Verify backup is a valid SQLite database
        try:
            conn = sqlite3.connect(backup_path)
            conn.execute("SELECT 1")
            conn.close()
        except sqlite3.Error as e:
            raise RestoreError(f"Invalid backup file: {e}")

        # Check if database exists
        if self._db_path.exists() and not force:
            raise RestoreError(
                f"Database already exists: {self._db_path}. Use force=True to overwrite"
            )

        try:
            # Create backup of current database if it exists
            if self._db_path.exists():
                current_backup = self._db_path.with_suffix(".db.before-restore")
                shutil.copy2(self._db_path, current_backup)

            # Restore from backup
            shutil.copy2(backup_path, self._db_path)

        except Exception as e:
            raise RestoreError(f"Failed to restore from backup: {e}")

    def check_integrity(self) -> Tuple[bool, str]:
        """Check database integrity.

        Returns:
            Tuple of (is_valid, detail_message)

        Raises:
            IntegrityError: If integrity check cannot be performed
        """
        if not self._db_path.exists():
            return False, f"Database not found: {self._db_path}"

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            # Run PRAGMA integrity_check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]

            conn.close()

            if result == "ok":
                return True, "Database integrity check passed"
            else:
                return False, f"Database integrity check failed: {result}"

        except sqlite3.Error as e:
            raise IntegrityError(f"Failed to check database integrity: {e}")

    def list_backups(self) -> List[Tuple[Path, datetime, int]]:
        """List available backup files.

        Returns:
            List of tuples: (backup_path, created_timestamp, size_bytes)
        """
        backups = []

        if not self._backup_dir.exists():
            return backups

        for backup_file in self._backup_dir.glob("queue-*.db"):
            stat = backup_file.stat()
            created = datetime.fromtimestamp(stat.st_mtime)
            size = stat.st_size

            backups.append((backup_file, created, size))

        # Sort by creation time, newest first
        backups.sort(key=lambda x: x[1], reverse=True)
        return backups

    def purge_old_backups(
        self,
        keep_count: int = 10,
        older_than_days: Optional[int] = None,
        dry_run: bool = False,
    ) -> int:
        """Remove old backup files.

        Args:
            keep_count: Keep at least this many recent backups
            older_than_days: Also remove backups older than this many days
            dry_run: If True, don't delete, just count

        Returns:
            Number of backups removed (or would be removed in dry run)
        """
        backups = self.list_backups()

        # Determine which backups to remove
        to_remove = []

        # Keep most recent backups
        if len(backups) > keep_count:
            to_remove.extend(backups[keep_count:])

        # Also remove backups older than threshold
        if older_than_days is not None:
            cutoff_date = datetime.utcnow() - __import__('datetime').timedelta(
                days=older_than_days
            )
            to_remove.extend(
                [b for b in backups if b[1] < cutoff_date and b not in to_remove]
            )

        # Remove duplicates
        to_remove = list(set(to_remove))

        # Delete files if not dry run
        if not dry_run:
            for backup_path, _, _ in to_remove:
                backup_path.unlink()

        return len(to_remove)

    def vacuum(self) -> None:
        """Vacuum the database to reclaim space and optimize.

        Raises:
            DatabaseError: If vacuum fails
        """
        if not self._db_path.exists():
            raise DatabaseError(f"Database not found: {self._db_path}")

        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("VACUUM")
            conn.close()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to vacuum database: {e}")

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dict with database stats (size, page_count, etc.)

        Raises:
            DatabaseError: If unable to get stats
        """
        if not self._db_path.exists():
            return {
                "exists": False,
                "size_bytes": 0,
                "size_mb": 0.0,
            }

        try:
            stat = self._db_path.stat()
            size_bytes = stat.st_size
            size_mb = size_bytes / (1024 * 1024)

            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            # Get page count and page size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]

            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]

            # Get table counts
            cursor.execute("SELECT COUNT(*) FROM jobs")
            job_count = cursor.fetchone()[0]

            conn.close()

            return {
                "exists": True,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "page_count": page_count,
                "page_size": page_size,
                "job_count": job_count,
            }

        except (sqlite3.Error, OSError) as e:
            raise DatabaseError(f"Failed to get database stats: {e}")
