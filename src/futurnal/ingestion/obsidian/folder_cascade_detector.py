"""Folder cascade detection for Obsidian vault synchronization.

This module provides detection and handling of folder-level operations where
renaming or moving a folder affects multiple files simultaneously. It groups
related path changes and processes them atomically to maintain consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .path_tracker import PathChange

logger = logging.getLogger(__name__)


@dataclass
class FolderCascade:
    """Represents a detected folder cascade operation."""
    cascade_id: str
    operation_type: str  # 'rename', 'move', 'rename_and_move'
    old_folder_path: Path
    new_folder_path: Path
    affected_files: List[PathChange]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, any] = field(default_factory=dict)

    def get_file_count(self) -> int:
        """Get the number of affected files."""
        return len(self.affected_files)

    def get_markdown_count(self) -> int:
        """Get the number of affected markdown files."""
        return len([
            change for change in self.affected_files
            if change.old_path.suffix.lower() in {'.md', '.markdown'}
        ])

    def is_large_cascade(self, threshold: int = 10) -> bool:
        """Check if this is a large cascade affecting many files."""
        return self.get_file_count() >= threshold

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "cascade_id": self.cascade_id,
            "operation_type": self.operation_type,
            "old_folder_path": str(self.old_folder_path),
            "new_folder_path": str(self.new_folder_path),
            "file_count": self.get_file_count(),
            "markdown_count": self.get_markdown_count(),
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
            "affected_files": [change.to_dict() for change in self.affected_files]
        }


class FolderCascadeDetector:
    """Detects folder cascade operations from individual file changes.

    This class analyzes collections of path changes to identify when they
    represent folder-level operations (rename/move of entire directories)
    rather than individual file operations.
    """

    def __init__(
        self,
        vault_root: Path,
        *,
        min_cascade_size: int = 2,
        max_path_depth_difference: int = 2,
        path_similarity_threshold: float = 0.9
    ):
        """Initialize the folder cascade detector.

        Args:
            vault_root: Root path of the vault
            min_cascade_size: Minimum number of files to consider a cascade
            max_path_depth_difference: Maximum depth difference to group paths
            path_similarity_threshold: Minimum similarity for path grouping
        """
        self.vault_root = vault_root
        self.min_cascade_size = min_cascade_size
        self.max_path_depth_difference = max_path_depth_difference
        self.path_similarity_threshold = path_similarity_threshold

    def detect_folder_cascades(self, path_changes: List[PathChange]) -> List[FolderCascade]:
        """Detect folder cascade operations from path changes.

        Args:
            path_changes: List of individual path changes

        Returns:
            List of detected folder cascades
        """
        if len(path_changes) < self.min_cascade_size:
            return []

        # Group path changes by potential common folder operations
        folder_groups = self._group_changes_by_folder_operation(path_changes)

        # Convert groups to FolderCascade objects
        cascades = []
        for group_id, (old_folder, new_folder, changes) in folder_groups.items():
            if len(changes) >= self.min_cascade_size:
                cascade = self._create_folder_cascade(group_id, old_folder, new_folder, changes)
                cascades.append(cascade)
                logger.info(f"Detected folder cascade: {old_folder} -> {new_folder} ({len(changes)} files)")

        return cascades

    def _group_changes_by_folder_operation(
        self,
        path_changes: List[PathChange]
    ) -> Dict[str, Tuple[Path, Path, List[PathChange]]]:
        """Group path changes by potential folder operations.

        Returns:
            Dictionary mapping group_id to (old_folder, new_folder, changes)
        """
        groups: Dict[str, Tuple[Path, Path, List[PathChange]]] = {}

        # Build groups based on common path transformations
        path_transformations: Dict[Tuple[Path, Path], List[PathChange]] = {}

        for change in path_changes:
            # Find the deepest common transformation
            transformation = self._find_folder_transformation(change.old_path, change.new_path)
            if transformation:
                old_folder, new_folder = transformation
                key = (old_folder, new_folder)

                if key not in path_transformations:
                    path_transformations[key] = []
                path_transformations[key].append(change)

        # Convert transformations to groups
        group_counter = 0
        for (old_folder, new_folder), changes in path_transformations.items():
            if len(changes) >= self.min_cascade_size:
                group_id = f"cascade_{group_counter:04d}"
                groups[group_id] = (old_folder, new_folder, changes)
                group_counter += 1

        return groups

    def _find_folder_transformation(
        self,
        old_path: Path,
        new_path: Path
    ) -> Optional[Tuple[Path, Path]]:
        """Find the folder-level transformation that explains the path change.

        Args:
            old_path: Original file path
            new_path: New file path

        Returns:
            Tuple of (old_folder, new_folder) if a transformation is found
        """
        try:
            # Make paths relative to vault root for comparison
            old_relative = old_path.relative_to(self.vault_root)
            new_relative = new_path.relative_to(self.vault_root)
        except ValueError:
            # Paths not relative to vault root
            old_relative = old_path
            new_relative = new_path

        old_parts = old_relative.parts
        new_parts = new_relative.parts

        # Find the point where paths diverge
        common_prefix_length = 0
        for i, (old_part, new_part) in enumerate(zip(old_parts, new_parts)):
            if old_part == new_part:
                common_prefix_length = i + 1
            else:
                break

        # If only the filename changed, not a folder operation
        if common_prefix_length == len(old_parts) - 1 and common_prefix_length == len(new_parts) - 1:
            return None

        # Extract the folder transformation
        if common_prefix_length < len(old_parts) - 1 and common_prefix_length < len(new_parts) - 1:
            # Find the immediate folder that changed
            old_folder_parts = old_parts[:common_prefix_length + 1]
            new_folder_parts = new_parts[:common_prefix_length + 1]

            old_folder = self.vault_root / Path(*old_folder_parts)
            new_folder = self.vault_root / Path(*new_folder_parts)

            return (old_folder, new_folder)

        return None

    def _create_folder_cascade(
        self,
        cascade_id: str,
        old_folder: Path,
        new_folder: Path,
        changes: List[PathChange]
    ) -> FolderCascade:
        """Create a FolderCascade object from grouped changes.

        Args:
            cascade_id: Unique identifier for the cascade
            old_folder: Original folder path
            new_folder: New folder path
            changes: List of affected file changes

        Returns:
            FolderCascade object
        """
        # Determine operation type
        operation_type = self._determine_folder_operation_type(old_folder, new_folder)

        # Collect metadata
        metadata = {
            "total_files": len(changes),
            "markdown_files": len([
                c for c in changes
                if c.old_path.suffix.lower() in {'.md', '.markdown'}
            ]),
            "change_types": list(set(c.change_type for c in changes)),
            "depth_change": len(new_folder.parts) - len(old_folder.parts)
        }

        return FolderCascade(
            cascade_id=cascade_id,
            operation_type=operation_type,
            old_folder_path=old_folder,
            new_folder_path=new_folder,
            affected_files=changes,
            metadata=metadata
        )

    def _determine_folder_operation_type(self, old_folder: Path, new_folder: Path) -> str:
        """Determine the type of folder operation.

        Args:
            old_folder: Original folder path
            new_folder: New folder path

        Returns:
            Operation type: 'rename', 'move', or 'rename_and_move'
        """
        old_parent = old_folder.parent
        new_parent = new_folder.parent
        old_name = old_folder.name
        new_name = new_folder.name

        if old_parent != new_parent and old_name != new_name:
            return "rename_and_move"
        elif old_parent != new_parent:
            return "move"
        elif old_name != new_name:
            return "rename"
        else:
            return "unknown"

    def get_cascade_statistics(self, cascades: List[FolderCascade]) -> Dict[str, any]:
        """Get statistics about detected cascades.

        Args:
            cascades: List of detected cascades

        Returns:
            Dictionary with cascade statistics
        """
        if not cascades:
            return {
                "total_cascades": 0,
                "total_affected_files": 0,
                "total_affected_markdown": 0,
                "operations_by_type": {},
                "large_cascades": 0
            }

        total_files = sum(cascade.get_file_count() for cascade in cascades)
        total_markdown = sum(cascade.get_markdown_count() for cascade in cascades)
        large_cascades = len([c for c in cascades if c.is_large_cascade()])

        operations_by_type = {}
        for cascade in cascades:
            op_type = cascade.operation_type
            operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1

        return {
            "total_cascades": len(cascades),
            "total_affected_files": total_files,
            "total_affected_markdown": total_markdown,
            "operations_by_type": operations_by_type,
            "large_cascades": large_cascades,
            "average_files_per_cascade": total_files / len(cascades) if cascades else 0
        }


def create_folder_cascade_detector(vault_root: Path) -> FolderCascadeDetector:
    """Factory function to create a folder cascade detector with sensible defaults."""
    return FolderCascadeDetector(
        vault_root=vault_root,
        min_cascade_size=2,
        max_path_depth_difference=2,
        path_similarity_threshold=0.9
    )