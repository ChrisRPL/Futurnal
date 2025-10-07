"""Persistent state management for GitHub repository synchronization.

This module provides file-based state persistence for tracking sync operations,
including commit SHAs, branch states, and sync statistics. Uses atomic writes
and file locking for concurrent access safety.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from filelock import FileLock

from .sync_models import SyncState, SyncStatus


@dataclass
class SyncStateManager:
    """Manager for persistent sync state storage."""

    state_dir: Path
    _lock_timeout: int = 10

    def __init__(self, state_dir: Optional[Path] = None):
        """Initialize state manager.

        Args:
            state_dir: Directory for state files (default: ~/.futurnal/sync_state/github/)
        """
        if state_dir is None:
            state_dir = Path.home() / ".futurnal" / "sync_state" / "github"

        self.state_dir = state_dir.expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, repo_id: str) -> Path:
        """Get state file path for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            Path to state file
        """
        return self.state_dir / f"{repo_id}.json"

    def _lock_path(self, repo_id: str) -> Path:
        """Get lock file path for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            Path to lock file
        """
        return self.state_dir / f"{repo_id}.lock"

    def _history_path(self, repo_id: str) -> Path:
        """Get history directory path for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            Path to history directory
        """
        history_dir = self.state_dir / "history" / repo_id
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir

    def load(self, repo_id: str) -> Optional[SyncState]:
        """Load sync state for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            SyncState if exists, None otherwise
        """
        state_path = self._state_path(repo_id)

        if not state_path.exists():
            return None

        lock = FileLock(str(self._lock_path(repo_id)), timeout=self._lock_timeout)

        try:
            with lock:
                data = json.loads(state_path.read_text())
                return SyncState.model_validate(data)
        except Exception:
            # If state file is corrupted, return None
            return None

    def save(self, state: SyncState) -> None:
        """Save sync state for repository.

        Uses atomic write to ensure consistency.

        Args:
            state: Sync state to save
        """
        state_path = self._state_path(state.repo_id)
        lock = FileLock(str(self._lock_path(state.repo_id)), timeout=self._lock_timeout)

        with lock:
            # Save to history if this is a completed sync
            if state.status == SyncStatus.COMPLETED and state.last_sync_time:
                self._save_to_history(state)

            # Write to temporary file first
            tmp_path = state_path.with_suffix(".tmp")
            tmp_path.write_text(state.model_dump_json(indent=2))

            # Atomic replace
            os.replace(tmp_path, state_path)

    def delete(self, repo_id: str) -> bool:
        """Delete sync state for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            True if deleted, False if didn't exist
        """
        state_path = self._state_path(repo_id)
        lock_path = self._lock_path(repo_id)

        if not state_path.exists():
            return False

        lock = FileLock(str(lock_path), timeout=self._lock_timeout)

        with lock:
            state_path.unlink()

            # Clean up lock file
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except Exception:
                    pass

            return True

    def list_all(self) -> List[SyncState]:
        """List all sync states.

        Returns:
            List of all sync states
        """
        states: List[SyncState] = []

        for state_file in sorted(self.state_dir.glob("*.json")):
            try:
                data = json.loads(state_file.read_text())
                state = SyncState.model_validate(data)
                states.append(state)
            except Exception:
                # Skip corrupted state files
                continue

        return states

    def find_by_status(self, status: SyncStatus) -> List[SyncState]:
        """Find sync states by status.

        Args:
            status: Status to filter by

        Returns:
            List of matching sync states
        """
        all_states = self.list_all()
        return [state for state in all_states if state.status == status]

    def find_unhealthy(self) -> List[SyncState]:
        """Find unhealthy sync states (too many consecutive failures).

        Returns:
            List of unhealthy sync states
        """
        all_states = self.list_all()
        return [state for state in all_states if not state.is_healthy()]

    def cleanup_old_states(self, days: int = 90) -> int:
        """Clean up old sync states.

        Removes state files that haven't been updated in specified days.

        Args:
            days: Number of days to retain

        Returns:
            Number of states cleaned up
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        cleaned = 0

        for state_file in self.state_dir.glob("*.json"):
            try:
                # Check file modification time
                if state_file.stat().st_mtime < cutoff:
                    # Load state to get repo_id for proper cleanup
                    data = json.loads(state_file.read_text())
                    repo_id = data.get("repo_id")

                    if repo_id:
                        self.delete(repo_id)
                        cleaned += 1
                    else:
                        # No repo_id, just delete the file
                        state_file.unlink()
                        cleaned += 1
            except Exception:
                # Skip files we can't process
                continue

        return cleaned

    def get_or_create(
        self,
        repo_id: str,
        sync_mode: str,
        local_clone_path: Optional[Path] = None,
    ) -> SyncState:
        """Get existing state or create new one.

        Args:
            repo_id: Repository identifier
            sync_mode: Sync mode (graphql_api or git_clone)
            local_clone_path: Local clone path (for git_clone mode)

        Returns:
            SyncState instance
        """
        existing = self.load(repo_id)

        if existing:
            return existing

        # Create new state
        state = SyncState(
            repo_id=repo_id,
            sync_mode=sync_mode,
            status=SyncStatus.PENDING,
            local_clone_path=local_clone_path,
        )

        self.save(state)
        return state

    def update_commit_sha(self, repo_id: str, commit_sha: str) -> None:
        """Update last commit SHA for repository.

        Args:
            repo_id: Repository identifier
            commit_sha: Commit SHA to update
        """
        state = self.load(repo_id)

        if state:
            state.last_commit_sha = commit_sha
            self.save(state)

    def mark_sync_started(self, repo_id: str) -> None:
        """Mark sync as started.

        Args:
            repo_id: Repository identifier
        """
        state = self.load(repo_id)

        if state:
            state.mark_sync_started()
            self.save(state)

    def mark_sync_completed(
        self,
        repo_id: str,
        files_synced: int = 0,
        bytes_synced: int = 0,
        commits: int = 0,
    ) -> None:
        """Mark sync as completed.

        Args:
            repo_id: Repository identifier
            files_synced: Number of files synced
            bytes_synced: Bytes synced
            commits: Number of commits processed
        """
        state = self.load(repo_id)

        if state:
            state.mark_sync_completed(
                files_synced=files_synced,
                bytes_synced=bytes_synced,
                commits=commits,
            )
            self.save(state)

    def mark_sync_failed(self, repo_id: str, error_count: int = 1) -> None:
        """Mark sync as failed.

        Args:
            repo_id: Repository identifier
            error_count: Number of errors encountered
        """
        state = self.load(repo_id)

        if state:
            state.mark_sync_failed(error_count=error_count)
            self.save(state)

    def _save_to_history(self, state: SyncState) -> None:
        """Save sync state snapshot to history.

        Keeps last 10 successful syncs in history.

        Args:
            state: Sync state to save
        """
        history_dir = self._history_path(state.repo_id)

        # Create timestamped filename
        timestamp = state.last_sync_time or datetime.now(timezone.utc)
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        history_file = history_dir / filename

        # Save snapshot
        try:
            history_file.write_text(state.model_dump_json(indent=2))
        except Exception:
            # Don't fail if history save fails
            pass

        # Clean up old history (keep last 10)
        self._cleanup_history(history_dir, keep=10)

    def _cleanup_history(self, history_dir: Path, keep: int = 10) -> None:
        """Clean up old history files.

        Args:
            history_dir: History directory
            keep: Number of recent files to keep
        """
        try:
            history_files = sorted(history_dir.glob("*.json"), reverse=True)

            # Delete old files beyond keep limit
            for old_file in history_files[keep:]:
                try:
                    old_file.unlink()
                except Exception:
                    pass
        except Exception:
            # Don't fail if cleanup fails
            pass

    def get_history(
        self, repo_id: str, limit: int = 10
    ) -> List[SyncState]:
        """Get sync history for repository.

        Args:
            repo_id: Repository identifier
            limit: Maximum number of history entries to return

        Returns:
            List of historical sync states (most recent first)
        """
        history_dir = self._history_path(repo_id)
        history_states: List[SyncState] = []

        history_files = sorted(history_dir.glob("*.json"), reverse=True)

        for history_file in history_files[:limit]:
            try:
                data = json.loads(history_file.read_text())
                state = SyncState.model_validate(data)
                history_states.append(state)
            except Exception:
                # Skip corrupted history files
                continue

        return history_states

    def get_statistics(self, repo_id: str) -> Dict[str, int]:
        """Get aggregate statistics for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            Dictionary of statistics
        """
        state = self.load(repo_id)

        if not state:
            return {
                "total_syncs": 0,
                "successful_syncs": 0,
                "failed_syncs": 0,
                "total_files_synced": 0,
                "total_bytes_synced": 0,
                "consecutive_failures": 0,
            }

        # Get history for more accurate counts
        history = self.get_history(repo_id, limit=100)

        successful = sum(1 for h in history if h.status == SyncStatus.COMPLETED)
        failed = sum(1 for h in history if h.status == SyncStatus.FAILED)

        return {
            "total_syncs": len(history),
            "successful_syncs": successful,
            "failed_syncs": failed,
            "total_files_synced": state.total_files_synced,
            "total_bytes_synced": state.total_bytes_synced,
            "consecutive_failures": state.consecutive_failures,
        }


__all__ = [
    "SyncStateManager",
]
