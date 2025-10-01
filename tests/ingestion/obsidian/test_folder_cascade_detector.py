"""Tests for FolderCascadeDetector and folder operation handling."""

import pytest
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from futurnal.ingestion.obsidian.path_tracker import PathChange
from futurnal.ingestion.obsidian.folder_cascade_detector import (
    FolderCascadeDetector,
    FolderCascade,
    create_folder_cascade_detector
)


@pytest.fixture
def temp_vault():
    """Create a temporary vault directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_root = Path(temp_dir) / "test_vault"
        vault_root.mkdir()
        yield vault_root


@pytest.fixture
def cascade_detector(temp_vault):
    """Create a FolderCascadeDetector for testing."""
    return FolderCascadeDetector(
        vault_root=temp_vault,
        min_cascade_size=2,
        max_path_depth_difference=2,
        path_similarity_threshold=0.9
    )


@pytest.fixture
def sample_folder_structure(temp_vault):
    """Create a sample folder structure for testing."""
    folders = {
        "projects": temp_vault / "projects",
        "work": temp_vault / "work",
        "projects/archive": temp_vault / "projects" / "archive",
        "work/archive": temp_vault / "work" / "archive",
    }

    files = {
        "projects/project1.md": temp_vault / "projects" / "project1.md",
        "projects/project2.md": temp_vault / "projects" / "project2.md",
        "projects/archive/old1.md": temp_vault / "projects" / "archive" / "old1.md",
        "projects/archive/old2.md": temp_vault / "projects" / "archive" / "old2.md",
        "work/project1.md": temp_vault / "work" / "project1.md",
        "work/project2.md": temp_vault / "work" / "project2.md",
        "work/archive/old1.md": temp_vault / "work" / "archive" / "old1.md",
        "work/archive/old2.md": temp_vault / "work" / "archive" / "old2.md",
    }

    # Create folders
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    # Create files
    for relative_path, file_path in files.items():
        file_path.write_text(f"# {relative_path}\n\nContent for {relative_path}")

    return {"folders": folders, "files": files}


def create_path_change(old_path: Path, new_path: Path, vault_id: str = "test_vault") -> PathChange:
    """Helper to create a PathChange object."""
    note_id = str(uuid.uuid4())
    return PathChange(
        vault_id=vault_id,
        old_note_id=note_id,
        new_note_id=note_id,  # Same ID with UUID system
        old_path=old_path,
        new_path=new_path,
        old_checksum="old_checksum",
        new_checksum="new_checksum",
        change_type="rename"
    )


class TestFolderCascadeDetector:
    """Test cases for FolderCascadeDetector."""

    def test_initialization(self, temp_vault):
        """Test that FolderCascadeDetector initializes correctly."""
        detector = FolderCascadeDetector(temp_vault)

        assert detector.vault_root == temp_vault
        assert detector.min_cascade_size == 2  # Default value
        assert detector.max_path_depth_difference == 2  # Default value
        assert detector.path_similarity_threshold == 0.9  # Default value

    def test_detect_simple_folder_rename(self, cascade_detector, sample_folder_structure):
        """Test detection of a simple folder rename cascade."""
        files = sample_folder_structure["files"]

        # Create path changes for renaming "projects" to "work"
        path_changes = [
            create_path_change(
                files["projects/project1.md"],
                files["work/project1.md"]
            ),
            create_path_change(
                files["projects/project2.md"],
                files["work/project2.md"]
            ),
        ]

        cascades = cascade_detector.detect_folder_cascades(path_changes)

        assert len(cascades) == 1
        cascade = cascades[0]

        assert cascade.operation_type in ["rename", "move", "rename_and_move"]
        assert cascade.get_file_count() == 2
        assert len(cascade.affected_files) == 2

    def test_detect_nested_folder_rename(self, cascade_detector, sample_folder_structure):
        """Test detection of nested folder rename cascade."""
        files = sample_folder_structure["files"]

        # Create path changes for renaming nested folders
        path_changes = [
            create_path_change(
                files["projects/archive/old1.md"],
                files["work/archive/old1.md"]
            ),
            create_path_change(
                files["projects/archive/old2.md"],
                files["work/archive/old2.md"]
            ),
        ]

        cascades = cascade_detector.detect_folder_cascades(path_changes)

        assert len(cascades) == 1
        cascade = cascades[0]

        assert cascade.get_file_count() == 2
        assert cascade.get_markdown_count() == 2

    def test_no_cascade_for_single_file(self, cascade_detector, sample_folder_structure):
        """Test that single file changes don't create cascades."""
        files = sample_folder_structure["files"]

        # Single file change
        path_changes = [
            create_path_change(
                files["projects/project1.md"],
                files["work/project1.md"]
            )
        ]

        cascades = cascade_detector.detect_folder_cascades(path_changes)

        assert len(cascades) == 0

    def test_no_cascade_below_minimum_size(self, cascade_detector, sample_folder_structure):
        """Test that cascades below minimum size are not detected."""
        # Set minimum cascade size to 3
        cascade_detector.min_cascade_size = 3
        files = sample_folder_structure["files"]

        # Only 2 files - below minimum
        path_changes = [
            create_path_change(
                files["projects/project1.md"],
                files["work/project1.md"]
            ),
            create_path_change(
                files["projects/project2.md"],
                files["work/project2.md"]
            ),
        ]

        cascades = cascade_detector.detect_folder_cascades(path_changes)

        assert len(cascades) == 0

    def test_multiple_cascades_detected(self, cascade_detector, temp_vault):
        """Test detection of multiple separate folder cascades."""
        # Create two separate folder operations
        path_changes = [
            # First cascade: projects -> work
            create_path_change(
                temp_vault / "projects" / "file1.md",
                temp_vault / "work" / "file1.md"
            ),
            create_path_change(
                temp_vault / "projects" / "file2.md",
                temp_vault / "work" / "file2.md"
            ),
            # Second cascade: docs -> documentation
            create_path_change(
                temp_vault / "docs" / "doc1.md",
                temp_vault / "documentation" / "doc1.md"
            ),
            create_path_change(
                temp_vault / "docs" / "doc2.md",
                temp_vault / "documentation" / "doc2.md"
            ),
        ]

        cascades = cascade_detector.detect_folder_cascades(path_changes)

        assert len(cascades) == 2

    def test_large_cascade_detection(self, cascade_detector, temp_vault):
        """Test detection and marking of large cascades."""
        # Create a large cascade with many files
        path_changes = []
        for i in range(15):  # 15 files
            path_changes.append(create_path_change(
                temp_vault / "large_folder" / f"file{i}.md",
                temp_vault / "renamed_folder" / f"file{i}.md"
            ))

        cascades = cascade_detector.detect_folder_cascades(path_changes)

        assert len(cascades) == 1
        cascade = cascades[0]

        assert cascade.get_file_count() == 15
        assert cascade.is_large_cascade(threshold=10)  # Above threshold

    def test_get_cascade_statistics(self, cascade_detector, temp_vault):
        """Test getting statistics about detected cascades."""
        # Create multiple cascades with different characteristics
        path_changes = [
            # Small rename cascade
            create_path_change(
                temp_vault / "small" / "file1.md",
                temp_vault / "small_renamed" / "file1.md"
            ),
            create_path_change(
                temp_vault / "small" / "file2.md",
                temp_vault / "small_renamed" / "file2.md"
            ),
            # Large move cascade
        ] + [
            create_path_change(
                temp_vault / "big" / f"file{i}.md",
                temp_vault / "different_location" / "big" / f"file{i}.md"
            )
            for i in range(12)  # 12 files for large cascade
        ]

        cascades = cascade_detector.detect_folder_cascades(path_changes)
        stats = cascade_detector.get_cascade_statistics(cascades)

        assert stats["total_cascades"] == 2
        assert stats["total_affected_files"] == 14  # 2 + 12
        assert stats["total_affected_markdown"] == 14  # All are .md files
        assert stats["large_cascades"] == 1  # Only the 12-file cascade is large
        assert stats["average_files_per_cascade"] == 7.0  # 14/2

    def test_empty_cascade_statistics(self, cascade_detector):
        """Test statistics with no cascades."""
        stats = cascade_detector.get_cascade_statistics([])

        assert stats["total_cascades"] == 0
        assert stats["total_affected_files"] == 0
        assert stats["total_affected_markdown"] == 0
        assert stats["large_cascades"] == 0
        assert "operations_by_type" in stats


class TestFolderCascade:
    """Test cases for FolderCascade dataclass."""

    def test_folder_cascade_creation(self, temp_vault):
        """Test creating a FolderCascade object."""
        path_changes = [
            create_path_change(
                temp_vault / "old" / "file1.md",
                temp_vault / "new" / "file1.md"
            ),
            create_path_change(
                temp_vault / "old" / "file2.md",
                temp_vault / "new" / "file2.md"
            ),
        ]

        cascade = FolderCascade(
            cascade_id="test_cascade",
            operation_type="rename",
            old_folder_path=temp_vault / "old",
            new_folder_path=temp_vault / "new",
            affected_files=path_changes
        )

        assert cascade.cascade_id == "test_cascade"
        assert cascade.operation_type == "rename"
        assert cascade.get_file_count() == 2
        assert cascade.get_markdown_count() == 2

    def test_folder_cascade_markdown_count(self, temp_vault):
        """Test counting markdown files in cascade."""
        path_changes = [
            create_path_change(
                temp_vault / "old" / "note.md",
                temp_vault / "new" / "note.md"
            ),
            PathChange(
                vault_id="test",
                old_note_id="id1",
                new_note_id="id1",
                old_path=temp_vault / "old" / "image.png",
                new_path=temp_vault / "new" / "image.png",
                old_checksum="old",
                new_checksum="new"
            ),
        ]

        cascade = FolderCascade(
            cascade_id="test_cascade",
            operation_type="rename",
            old_folder_path=temp_vault / "old",
            new_folder_path=temp_vault / "new",
            affected_files=path_changes
        )

        assert cascade.get_file_count() == 2
        assert cascade.get_markdown_count() == 1  # Only .md file

    def test_folder_cascade_to_dict(self, temp_vault):
        """Test converting FolderCascade to dictionary."""
        path_changes = [
            create_path_change(
                temp_vault / "old" / "file.md",
                temp_vault / "new" / "file.md"
            ),
        ]

        created_at = datetime.utcnow()
        cascade = FolderCascade(
            cascade_id="test_cascade",
            operation_type="rename",
            old_folder_path=temp_vault / "old",
            new_folder_path=temp_vault / "new",
            affected_files=path_changes,
            detected_at=created_at,
            metadata={"test": "value"}
        )

        cascade_dict = cascade.to_dict()

        assert cascade_dict["cascade_id"] == "test_cascade"
        assert cascade_dict["operation_type"] == "rename"
        assert cascade_dict["file_count"] == 1
        assert cascade_dict["markdown_count"] == 1
        assert cascade_dict["detected_at"] == created_at.isoformat()
        assert cascade_dict["metadata"]["test"] == "value"
        assert len(cascade_dict["affected_files"]) == 1

    def test_is_large_cascade(self, temp_vault):
        """Test large cascade detection."""
        # Create small cascade
        small_changes = [
            create_path_change(
                temp_vault / "old" / f"file{i}.md",
                temp_vault / "new" / f"file{i}.md"
            )
            for i in range(5)
        ]

        small_cascade = FolderCascade(
            cascade_id="small",
            operation_type="rename",
            old_folder_path=temp_vault / "old",
            new_folder_path=temp_vault / "new",
            affected_files=small_changes
        )

        # Create large cascade
        large_changes = [
            create_path_change(
                temp_vault / "old" / f"file{i}.md",
                temp_vault / "new" / f"file{i}.md"
            )
            for i in range(15)
        ]

        large_cascade = FolderCascade(
            cascade_id="large",
            operation_type="rename",
            old_folder_path=temp_vault / "old",
            new_folder_path=temp_vault / "new",
            affected_files=large_changes
        )

        assert not small_cascade.is_large_cascade(threshold=10)
        assert large_cascade.is_large_cascade(threshold=10)


class TestFolderCascadeDetectorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_path_changes(self, cascade_detector):
        """Test handling of empty path changes list."""
        cascades = cascade_detector.detect_folder_cascades([])

        assert len(cascades) == 0

    def test_unrelated_individual_changes(self, cascade_detector, temp_vault):
        """Test that unrelated individual changes don't form cascades."""
        path_changes = [
            create_path_change(
                temp_vault / "folder1" / "file1.md",
                temp_vault / "folder1" / "renamed1.md"  # Just rename, not move
            ),
            create_path_change(
                temp_vault / "folder2" / "file2.md",
                temp_vault / "folder3" / "file2.md"  # Different transformation
            ),
        ]

        cascades = cascade_detector.detect_folder_cascades(path_changes)

        assert len(cascades) == 0

    def test_create_folder_cascade_detector_factory(self, temp_vault):
        """Test the factory function for creating detector."""
        detector = create_folder_cascade_detector(temp_vault)

        assert isinstance(detector, FolderCascadeDetector)
        assert detector.vault_root == temp_vault
        assert detector.min_cascade_size == 2
        assert detector.max_path_depth_difference == 2
        assert detector.path_similarity_threshold == 0.9

    def test_complex_nested_folder_structure(self, cascade_detector, temp_vault):
        """Test with complex nested folder structures."""
        # Create a complex folder rename: deep/nested/folders -> reorganized/structure/folders
        base_old = temp_vault / "deep" / "nested" / "folders"
        base_new = temp_vault / "reorganized" / "structure" / "folders"

        path_changes = [
            create_path_change(
                base_old / "doc1.md",
                base_new / "doc1.md"
            ),
            create_path_change(
                base_old / "subdir" / "doc2.md",
                base_new / "subdir" / "doc2.md"
            ),
            create_path_change(
                base_old / "subdir" / "deep" / "doc3.md",
                base_new / "subdir" / "deep" / "doc3.md"
            ),
        ]

        cascades = cascade_detector.detect_folder_cascades(path_changes)

        # Should detect the common folder transformation
        assert len(cascades) >= 1