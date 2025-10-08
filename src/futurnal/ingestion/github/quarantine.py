"""Quarantine workflow for failed GitHub file processing.

This module implements resilient error handling for GitHub repository sync operations,
providing quarantine storage, retry policies, and recovery mechanisms for failed files.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QuarantineEntry:
    """Metadata for a quarantined file."""

    quarantine_id: str
    repo_id: str
    file_path_hash: str
    error_type: str
    error_message: str
    retry_count: int
    max_retries: int
    quarantined_at: datetime
    last_retry_at: Optional[datetime]
    content_size: int
    file_metadata: Dict[str, str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "quarantine_id": self.quarantine_id,
            "repo_id": self.repo_id,
            "file_path_hash": self.file_path_hash,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "quarantined_at": self.quarantined_at.isoformat(),
            "last_retry_at": self.last_retry_at.isoformat()
            if self.last_retry_at
            else None,
            "content_size": self.content_size,
            "file_metadata": self.file_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuarantineEntry":
        """Create from dictionary."""
        return cls(
            quarantine_id=data["quarantine_id"],
            repo_id=data["repo_id"],
            file_path_hash=data["file_path_hash"],
            error_type=data["error_type"],
            error_message=data["error_message"],
            retry_count=data["retry_count"],
            max_retries=data["max_retries"],
            quarantined_at=datetime.fromisoformat(data["quarantined_at"]),
            last_retry_at=datetime.fromisoformat(data["last_retry_at"])
            if data.get("last_retry_at")
            else None,
            content_size=data["content_size"],
            file_metadata=data["file_metadata"],
        )


class GitHubQuarantineHandler:
    """Handles failed GitHub file processing with retry policies.

    This handler provides resilient error recovery for GitHub sync operations:
    - Quarantines files that fail processing
    - Implements exponential backoff retry policy
    - Stores error metadata for debugging
    - Provides cleanup and recovery mechanisms
    """

    def __init__(
        self,
        quarantine_dir: Path,
        max_retries: int = 3,
        base_backoff_seconds: int = 60,
    ):
        """Initialize quarantine handler.

        Args:
            quarantine_dir: Directory for quarantine storage
            max_retries: Maximum retry attempts (default: 3)
            base_backoff_seconds: Base backoff period for exponential retry (default: 60)
        """
        self.quarantine_dir = quarantine_dir
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds

        logger.info(
            f"Initialized GitHub quarantine handler: {quarantine_dir} "
            f"(max_retries={max_retries})"
        )

    def quarantine_file(
        self,
        repo_id: str,
        file_path: str,
        content: bytes,
        error: Exception,
        retry_count: int = 0,
        file_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Quarantine a file that failed processing.

        Args:
            repo_id: Repository identifier
            file_path: File path within repository
            content: File content bytes
            error: Exception that caused the failure
            retry_count: Current retry count
            file_metadata: Optional file metadata

        Returns:
            Quarantine ID
        """
        # Generate quarantine ID
        path_hash = sha256(file_path.encode()).hexdigest()
        quarantine_id = f"{repo_id[:8]}_{path_hash[:8]}"

        logger.warning(
            f"Quarantining file from {repo_id}: {file_path[:50]}... "
            f"(retry {retry_count}/{self.max_retries}, error: {type(error).__name__})"
        )

        # Create quarantine entry
        entry = QuarantineEntry(
            quarantine_id=quarantine_id,
            repo_id=repo_id,
            file_path_hash=path_hash,
            error_type=type(error).__name__,
            error_message=str(error),
            retry_count=retry_count,
            max_retries=self.max_retries,
            quarantined_at=datetime.now(timezone.utc),
            last_retry_at=None,
            content_size=len(content),
            file_metadata=file_metadata or {},
        )

        # Write entry metadata
        entry_path = self.quarantine_dir / f"{quarantine_id}.json"
        entry_path.write_text(json.dumps(entry.to_dict(), indent=2))

        # Write content separately
        content_path = self.quarantine_dir / f"{quarantine_id}.bin"
        content_path.write_bytes(content)

        logger.info(
            f"Quarantined: {quarantine_id} ({len(content)} bytes) "
            f"to {self.quarantine_dir}"
        )

        return quarantine_id

    def get_entry(self, quarantine_id: str) -> Optional[QuarantineEntry]:
        """Get quarantine entry by ID.

        Args:
            quarantine_id: Quarantine identifier

        Returns:
            QuarantineEntry if found, None otherwise
        """
        entry_path = self.quarantine_dir / f"{quarantine_id}.json"
        if not entry_path.exists():
            return None

        try:
            data = json.loads(entry_path.read_text())
            return QuarantineEntry.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load quarantine entry {quarantine_id}: {e}")
            return None

    def get_content(self, quarantine_id: str) -> Optional[bytes]:
        """Get quarantined file content.

        Args:
            quarantine_id: Quarantine identifier

        Returns:
            File content bytes if found, None otherwise
        """
        content_path = self.quarantine_dir / f"{quarantine_id}.bin"
        if not content_path.exists():
            return None

        try:
            return content_path.read_bytes()
        except Exception as e:
            logger.error(f"Failed to load quarantine content {quarantine_id}: {e}")
            return None

    def list_entries(
        self, *, repo_id: Optional[str] = None, retriable_only: bool = False
    ) -> List[QuarantineEntry]:
        """List quarantine entries.

        Args:
            repo_id: Optional filter by repository ID
            retriable_only: Only return entries that can be retried

        Returns:
            List of QuarantineEntry objects
        """
        entries: List[QuarantineEntry] = []

        for entry_path in sorted(self.quarantine_dir.glob("*.json")):
            try:
                data = json.loads(entry_path.read_text())
                entry = QuarantineEntry.from_dict(data)

                # Filter by repo_id if specified
                if repo_id and entry.repo_id != repo_id:
                    continue

                # Filter retriable entries
                if retriable_only and entry.retry_count >= entry.max_retries:
                    continue

                entries.append(entry)

            except Exception as e:
                logger.warning(f"Failed to load quarantine entry {entry_path}: {e}")
                continue

        return entries

    def can_retry(self, quarantine_id: str) -> bool:
        """Check if entry can be retried.

        Args:
            quarantine_id: Quarantine identifier

        Returns:
            True if retry is allowed, False otherwise
        """
        entry = self.get_entry(quarantine_id)
        if not entry:
            return False

        if entry.retry_count >= entry.max_retries:
            return False

        # Check backoff period
        if entry.last_retry_at:
            backoff = self._calculate_backoff(entry.retry_count)
            elapsed = (datetime.now(timezone.utc) - entry.last_retry_at).total_seconds()
            if elapsed < backoff:
                logger.debug(
                    f"Backoff period not elapsed for {quarantine_id}: "
                    f"{elapsed:.1f}s / {backoff:.1f}s"
                )
                return False

        return True

    def mark_retry_attempted(self, quarantine_id: str) -> None:
        """Mark that a retry was attempted.

        Args:
            quarantine_id: Quarantine identifier
        """
        entry = self.get_entry(quarantine_id)
        if not entry:
            logger.warning(f"Cannot mark retry for missing entry: {quarantine_id}")
            return

        entry.retry_count += 1
        entry.last_retry_at = datetime.now(timezone.utc)

        # Update entry file
        entry_path = self.quarantine_dir / f"{quarantine_id}.json"
        entry_path.write_text(json.dumps(entry.to_dict(), indent=2))

        logger.info(
            f"Marked retry for {quarantine_id}: "
            f"attempt {entry.retry_count}/{entry.max_retries}"
        )

    def remove_entry(self, quarantine_id: str) -> bool:
        """Remove quarantine entry (successful retry or cleanup).

        Args:
            quarantine_id: Quarantine identifier

        Returns:
            True if removed, False if not found
        """
        entry_path = self.quarantine_dir / f"{quarantine_id}.json"
        content_path = self.quarantine_dir / f"{quarantine_id}.bin"

        removed = False
        if entry_path.exists():
            entry_path.unlink()
            removed = True

        if content_path.exists():
            content_path.unlink()
            removed = True

        if removed:
            logger.info(f"Removed quarantine entry: {quarantine_id}")

        return removed

    def cleanup_exhausted(self, older_than_days: int = 30) -> int:
        """Clean up entries that have exhausted retries.

        Args:
            older_than_days: Remove entries older than this many days

        Returns:
            Number of entries removed
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (older_than_days * 86400)
        removed_count = 0

        for entry_path in self.quarantine_dir.glob("*.json"):
            try:
                data = json.loads(entry_path.read_text())
                entry = QuarantineEntry.from_dict(data)

                # Check if exhausted and old enough
                if entry.retry_count >= entry.max_retries:
                    quarantined_ts = entry.quarantined_at.timestamp()
                    if quarantined_ts < cutoff:
                        if self.remove_entry(entry.quarantine_id):
                            removed_count += 1

            except Exception as e:
                logger.warning(f"Failed to process entry {entry_path}: {e}")
                continue

        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} exhausted quarantine entries "
                f"(older than {older_than_days} days)"
            )

        return removed_count

    def get_statistics(self) -> Dict[str, int]:
        """Get quarantine statistics.

        Returns:
            Dictionary with statistics
        """
        all_entries = self.list_entries()
        retriable = [e for e in all_entries if e.retry_count < e.max_retries]
        exhausted = [e for e in all_entries if e.retry_count >= e.max_retries]

        # Group by error type
        error_types: Dict[str, int] = {}
        for entry in all_entries:
            error_types[entry.error_type] = error_types.get(entry.error_type, 0) + 1

        return {
            "total_entries": len(all_entries),
            "retriable_entries": len(retriable),
            "exhausted_entries": len(exhausted),
            "unique_repositories": len(set(e.repo_id for e in all_entries)),
            "total_bytes_quarantined": sum(e.content_size for e in all_entries),
            "error_types": error_types,
        }

    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate exponential backoff period.

        Args:
            retry_count: Current retry count

        Returns:
            Backoff period in seconds
        """
        return self.base_backoff_seconds * (2**retry_count)


__all__ = [
    "GitHubQuarantineHandler",
    "QuarantineEntry",
]
