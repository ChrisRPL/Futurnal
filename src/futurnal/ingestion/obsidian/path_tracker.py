"""Path tracking for Obsidian note rename/move operations.

This module handles tracking when Obsidian notes are moved or renamed,
ensuring that note relationships in the PKG are updated without losing
history or breaking relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..local.state import FileRecord, StateStore

logger = logging.getLogger(__name__)


@dataclass
class PathChange:
    """Represents a detected path change for a note."""
    vault_id: str
    old_note_id: str
    new_note_id: str
    old_path: Path
    new_path: Path
    old_checksum: str
    new_checksum: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    change_type: str = "rename"  # 'rename', 'move', 'rename_and_move'

    def is_content_change(self) -> bool:
        """Check if this is a content change vs pure path change."""
        return self.old_checksum != self.new_checksum

    def is_rename_only(self) -> bool:
        """Check if this is only a filename rename (same directory)."""
        return self.old_path.parent == self.new_path.parent

    def is_move_only(self) -> bool:
        """Check if this is only a directory move (same filename)."""
        return self.old_path.name == self.new_path.name

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "vault_id": self.vault_id,
            "old_note_id": self.old_note_id,
            "new_note_id": self.new_note_id,
            "old_path": str(self.old_path),
            "new_path": str(self.new_path),
            "old_checksum": self.old_checksum,
            "new_checksum": self.new_checksum,
            "detected_at": self.detected_at.isoformat(),
            "change_type": self.change_type,
            "is_content_change": self.is_content_change(),
            "is_rename_only": self.is_rename_only(),
            "is_move_only": self.is_move_only()
        }


class ObsidianPathTracker:
    """Tracks path changes for Obsidian notes and manages graph updates.

    This class integrates with the existing StateStore to detect when notes
    are moved or renamed, and coordinates updates to the PKG to maintain
    relationship integrity.
    """

    def __init__(
        self,
        vault_id: str,
        vault_root: Path,
        state_store: StateStore,
        *,
        similarity_threshold: float = 0.8,
        max_tracking_history: int = 1000
    ):
        self.vault_id = vault_id
        self.vault_root = Path(vault_root)
        self.state_store = state_store
        self.similarity_threshold = similarity_threshold
        self.max_tracking_history = max_tracking_history

        # Tracking state
        self.path_changes: List[PathChange] = []
        self.note_id_map: Dict[str, str] = {}  # old_note_id -> new_note_id
        self.path_map: Dict[str, str] = {}  # old_path -> new_path

    def detect_path_changes(self, current_records: List[FileRecord]) -> List[PathChange]:
        """Detect path changes by comparing current records with stored state.

        Args:
            current_records: Current file records from vault scan

        Returns:
            List of detected path changes
        """
        detected_changes = []

        # Create lookup maps for current records
        current_by_checksum: Dict[str, List[FileRecord]] = {}
        current_by_path: Dict[Path, FileRecord] = {}

        for record in current_records:
            if record.path.suffix.lower() in {'.md', '.markdown'}:
                # Group by checksum for content-based matching
                if record.sha256 not in current_by_checksum:
                    current_by_checksum[record.sha256] = []
                current_by_checksum[record.sha256].append(record)
                current_by_path[record.path] = record

        # Get all stored records for comparison
        stored_records = list(self.state_store.iter_all())
        stored_by_checksum: Dict[str, List[FileRecord]] = {}
        stored_by_path: Dict[Path, FileRecord] = {}

        for record in stored_records:
            if record.path.suffix.lower() in {'.md', '.markdown'}:
                if record.sha256 not in stored_by_checksum:
                    stored_by_checksum[record.sha256] = []
                stored_by_checksum[record.sha256].append(record)
                stored_by_path[record.path] = record

        # Detect moves/renames by matching content checksums
        for checksum, stored_records_list in stored_by_checksum.items():
            if checksum in current_by_checksum:
                current_records_list = current_by_checksum[checksum]

                # Compare stored vs current records with same checksum
                for stored_record in stored_records_list:
                    # Find if this stored record has moved
                    if stored_record.path not in current_by_path:
                        # File was moved/renamed - find the new location
                        for current_record in current_records_list:
                            if current_record.path not in stored_by_path:
                                # This is likely the new location
                                change = self._create_path_change(stored_record, current_record)
                                if change:
                                    detected_changes.append(change)
                                    logger.info(f"Detected path change: {stored_record.path} -> {current_record.path}")
                                break

        # Update tracking state
        for change in detected_changes:
            self._record_path_change(change)

        # Cleanup old history
        self._cleanup_tracking_history()

        return detected_changes

    def _create_path_change(self, old_record: FileRecord, new_record: FileRecord) -> Optional[PathChange]:
        """Create a PathChange object from old and new file records."""
        try:
            # Generate note IDs
            old_note_id = self._path_to_note_id(old_record.path)
            new_note_id = self._path_to_note_id(new_record.path)

            # Determine change type
            change_type = self._determine_change_type(old_record.path, new_record.path)

            change = PathChange(
                vault_id=self.vault_id,
                old_note_id=old_note_id,
                new_note_id=new_note_id,
                old_path=old_record.path,
                new_path=new_record.path,
                old_checksum=old_record.sha256,
                new_checksum=new_record.sha256,
                change_type=change_type
            )

            return change

        except Exception as e:
            logger.error(f"Failed to create path change: {e}")
            return None

    def _path_to_note_id(self, path: Path) -> str:
        """Convert file path to note ID."""
        try:
            relative_path = path.relative_to(self.vault_root)
            return str(relative_path.with_suffix(''))
        except ValueError:
            # Path is not relative to vault root
            return str(path.with_suffix(''))

    def _determine_change_type(self, old_path: Path, new_path: Path) -> str:
        """Determine the type of path change."""
        old_parent = old_path.parent
        new_parent = new_path.parent
        old_name = old_path.name
        new_name = new_path.name

        if old_parent != new_parent and old_name != new_name:
            return "rename_and_move"
        elif old_parent != new_parent:
            return "move"
        elif old_name != new_name:
            return "rename"
        else:
            return "unknown"

    def _record_path_change(self, change: PathChange) -> None:
        """Record a path change in tracking state."""
        self.path_changes.append(change)

        # Update mapping tables
        self.note_id_map[change.old_note_id] = change.new_note_id
        self.path_map[str(change.old_path)] = str(change.new_path)

    def _cleanup_tracking_history(self) -> None:
        """Clean up old tracking history to prevent memory bloat."""
        if len(self.path_changes) > self.max_tracking_history:
            # Keep only the most recent changes
            keep_count = int(self.max_tracking_history * 0.8)  # Keep 80% of limit
            self.path_changes = self.path_changes[-keep_count:]

            # Rebuild mapping tables from remaining changes
            self.note_id_map.clear()
            self.path_map.clear()
            for change in self.path_changes:
                self.note_id_map[change.old_note_id] = change.new_note_id
                self.path_map[str(change.old_path)] = str(change.new_path)

    def resolve_note_id(self, note_id: str) -> str:
        """Resolve a note ID to its current value, following rename chain."""
        current_id = note_id
        visited = set()

        # Follow the chain of renames
        while current_id in self.note_id_map and current_id not in visited:
            visited.add(current_id)
            current_id = self.note_id_map[current_id]

        return current_id

    def resolve_path(self, path: str) -> str:
        """Resolve a path to its current value, following move chain."""
        current_path = path
        visited = set()

        # Follow the chain of moves
        while current_path in self.path_map and current_path not in visited:
            visited.add(current_path)
            current_path = self.path_map[current_path]

        return current_path

    def get_path_changes(
        self,
        *,
        since: Optional[datetime] = None,
        change_types: Optional[List[str]] = None
    ) -> List[PathChange]:
        """Get path changes with optional filtering.

        Args:
            since: Only return changes after this timestamp
            change_types: Only return changes of these types

        Returns:
            Filtered list of path changes
        """
        changes = self.path_changes

        if since:
            changes = [c for c in changes if c.detected_at >= since]

        if change_types:
            changes = [c for c in changes if c.change_type in change_types]

        return changes

    def get_statistics(self) -> Dict[str, int]:
        """Get tracking statistics."""
        change_type_counts = {}
        content_changes = 0

        for change in self.path_changes:
            change_type_counts[change.change_type] = change_type_counts.get(change.change_type, 0) + 1
            if change.is_content_change():
                content_changes += 1

        return {
            "total_changes": len(self.path_changes),
            "note_id_mappings": len(self.note_id_map),
            "path_mappings": len(self.path_map),
            "content_changes": content_changes,
            **change_type_counts
        }

    def clear_tracking_data(self) -> None:
        """Clear all tracking data (useful for testing or reset)."""
        self.path_changes.clear()
        self.note_id_map.clear()
        self.path_map.clear()
        logger.info("Cleared all path tracking data")


def create_path_tracker(
    vault_id: str,
    vault_root: Path,
    state_store: StateStore
) -> ObsidianPathTracker:
    """Factory function to create a path tracker with sensible defaults."""
    return ObsidianPathTracker(
        vault_id=vault_id,
        vault_root=vault_root,
        state_store=state_store,
        similarity_threshold=0.8,
        max_tracking_history=1000
    )