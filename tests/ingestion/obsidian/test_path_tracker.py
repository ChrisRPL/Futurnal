"""Unit tests for Obsidian path tracking functionality."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

from futurnal.ingestion.obsidian.path_tracker import (
    ObsidianPathTracker,
    PathChange,
    create_path_tracker
)
from futurnal.ingestion.local.state import FileRecord, StateStore


class TestPathChange:
    """Test PathChange data structure."""

    def test_path_change_creation(self):
        """Test basic path change creation."""
        old_path = Path("/vault/old_note.md")
        new_path = Path("/vault/new_note.md")

        change = PathChange(
            vault_id="test_vault",
            old_note_id="old_note",
            new_note_id="new_note",
            old_path=old_path,
            new_path=new_path,
            old_checksum="abc123",
            new_checksum="abc123",
            change_type="rename"
        )

        assert change.vault_id == "test_vault"
        assert change.old_note_id == "old_note"
        assert change.new_note_id == "new_note"
        assert change.old_path == old_path
        assert change.new_path == new_path
        assert change.change_type == "rename"

    def test_content_change_detection(self):
        """Test content change detection."""
        # Same checksum - no content change
        change_same = PathChange(
            vault_id="test_vault",
            old_note_id="note",
            new_note_id="note",
            old_path=Path("/vault/note.md"),
            new_path=Path("/vault/note_renamed.md"),
            old_checksum="abc123",
            new_checksum="abc123"
        )
        assert not change_same.is_content_change()

        # Different checksum - content change
        change_different = PathChange(
            vault_id="test_vault",
            old_note_id="note",
            new_note_id="note",
            old_path=Path("/vault/note.md"),
            new_path=Path("/vault/note.md"),
            old_checksum="abc123",
            new_checksum="def456"
        )
        assert change_different.is_content_change()

    def test_rename_detection(self):
        """Test rename-only detection."""
        # Same directory, different filename
        rename_change = PathChange(
            vault_id="test_vault",
            old_note_id="old_note",
            new_note_id="new_note",
            old_path=Path("/vault/folder/old_note.md"),
            new_path=Path("/vault/folder/new_note.md"),
            old_checksum="abc123",
            new_checksum="abc123"
        )
        assert rename_change.is_rename_only()
        assert not rename_change.is_move_only()

    def test_move_detection(self):
        """Test move-only detection."""
        # Same filename, different directory
        move_change = PathChange(
            vault_id="test_vault",
            old_note_id="folder1/note",
            new_note_id="folder2/note",
            old_path=Path("/vault/folder1/note.md"),
            new_path=Path("/vault/folder2/note.md"),
            old_checksum="abc123",
            new_checksum="abc123"
        )
        assert move_change.is_move_only()
        assert not move_change.is_rename_only()

    def test_path_change_serialization(self):
        """Test path change serialization to dict."""
        change = PathChange(
            vault_id="test_vault",
            old_note_id="old_note",
            new_note_id="new_note",
            old_path=Path("/vault/old_note.md"),
            new_path=Path("/vault/new_note.md"),
            old_checksum="abc123",
            new_checksum="abc123",
            change_type="rename"
        )

        data = change.to_dict()
        assert data["vault_id"] == "test_vault"
        assert data["old_note_id"] == "old_note"
        assert data["new_note_id"] == "new_note"
        assert data["old_path"] == "/vault/old_note.md"
        assert data["new_path"] == "/vault/new_note.md"
        assert data["change_type"] == "rename"
        assert data["is_content_change"] is False
        assert data["is_rename_only"] is True
        assert data["is_move_only"] is False
        assert "detected_at" in data


class TestObsidianPathTracker:
    """Test ObsidianPathTracker class."""

    @pytest.fixture
    def vault_root(self, tmp_path):
        """Create a temporary vault root."""
        vault_dir = tmp_path / "test_vault"
        vault_dir.mkdir()
        return vault_dir

    @pytest.fixture
    def state_store(self):
        """Create a mock state store."""
        return Mock(spec=StateStore)

    @pytest.fixture
    def tracker(self, vault_root, state_store):
        """Create a path tracker instance."""
        return ObsidianPathTracker(
            vault_id="test_vault",
            vault_root=vault_root,
            state_store=state_store
        )

    def test_tracker_initialization(self, tracker, vault_root, state_store):
        """Test tracker initialization."""
        assert tracker.vault_id == "test_vault"
        assert tracker.vault_root == vault_root
        assert tracker.state_store == state_store
        assert tracker.similarity_threshold == 0.8
        assert tracker.max_tracking_history == 1000
        assert len(tracker.path_changes) == 0
        assert len(tracker.note_id_map) == 0
        assert len(tracker.path_map) == 0

    def test_path_to_note_id_conversion(self, tracker, vault_root):
        """Test path to note ID conversion."""
        # Path within vault
        vault_path = vault_root / "notes" / "test_note.md"
        note_id = tracker._path_to_note_id(vault_path)
        assert note_id == "notes/test_note"

        # Path outside vault (absolute path used)
        external_path = Path("/external/note.md")
        note_id = tracker._path_to_note_id(external_path)
        assert note_id == "/external/note"

    def test_change_type_determination(self, tracker):
        """Test change type determination."""
        # Rename only
        old_path = Path("/vault/folder/old_name.md")
        new_path = Path("/vault/folder/new_name.md")
        change_type = tracker._determine_change_type(old_path, new_path)
        assert change_type == "rename"

        # Move only
        old_path = Path("/vault/folder1/note.md")
        new_path = Path("/vault/folder2/note.md")
        change_type = tracker._determine_change_type(old_path, new_path)
        assert change_type == "move"

        # Rename and move
        old_path = Path("/vault/folder1/old_name.md")
        new_path = Path("/vault/folder2/new_name.md")
        change_type = tracker._determine_change_type(old_path, new_path)
        assert change_type == "rename_and_move"

    def test_detect_path_changes_rename(self, tracker, vault_root):
        """Test detection of file renames."""
        # Setup stored records (old state)
        old_record = FileRecord(
            path=vault_root / "old_note.md",
            size=1000,
            mtime=123456789,
            sha256="abc123"
        )

        # Setup current records (new state)
        new_record = FileRecord(
            path=vault_root / "new_note.md",
            size=1000,
            mtime=123456799,
            sha256="abc123"  # Same content, different name
        )

        # Mock state store
        tracker.state_store.iter_all.return_value = [old_record]

        # Detect changes
        changes = tracker.detect_path_changes([new_record])

        assert len(changes) == 1
        change = changes[0]
        assert change.old_note_id == "old_note"
        assert change.new_note_id == "new_note"
        assert change.change_type == "rename"
        assert not change.is_content_change()

    def test_detect_path_changes_move(self, tracker, vault_root):
        """Test detection of file moves."""
        # Setup stored records (old state)
        old_record = FileRecord(
            path=vault_root / "folder1" / "note.md",
            size=1000,
            mtime=123456789,
            sha256="abc123"
        )

        # Setup current records (new state)
        new_record = FileRecord(
            path=vault_root / "folder2" / "note.md",
            size=1000,
            mtime=123456799,
            sha256="abc123"  # Same content, different location
        )

        # Mock state store
        tracker.state_store.iter_all.return_value = [old_record]

        # Detect changes
        changes = tracker.detect_path_changes([new_record])

        assert len(changes) == 1
        change = changes[0]
        assert change.old_note_id == "folder1/note"
        assert change.new_note_id == "folder2/note"
        assert change.change_type == "move"
        assert not change.is_content_change()

    def test_detect_path_changes_no_changes(self, tracker, vault_root):
        """Test when no path changes occur."""
        # Setup records (same in both states)
        record = FileRecord(
            path=vault_root / "note.md",
            size=1000,
            mtime=123456789,
            sha256="abc123"
        )

        # Mock state store
        tracker.state_store.iter_all.return_value = [record]

        # Detect changes
        changes = tracker.detect_path_changes([record])

        assert len(changes) == 0

    def test_note_id_resolution(self, tracker):
        """Test note ID resolution through rename chain."""
        # Create a chain of renames: A -> B -> C
        tracker.note_id_map = {
            "note_a": "note_b",
            "note_b": "note_c"
        }

        # Should resolve to final name
        assert tracker.resolve_note_id("note_a") == "note_c"
        assert tracker.resolve_note_id("note_b") == "note_c"
        assert tracker.resolve_note_id("note_c") == "note_c"
        assert tracker.resolve_note_id("unknown") == "unknown"

    def test_path_resolution(self, tracker):
        """Test path resolution through move chain."""
        # Create a chain of moves
        tracker.path_map = {
            "/vault/old_path.md": "/vault/intermediate_path.md",
            "/vault/intermediate_path.md": "/vault/final_path.md"
        }

        # Should resolve to final path
        assert tracker.resolve_path("/vault/old_path.md") == "/vault/final_path.md"
        assert tracker.resolve_path("/vault/intermediate_path.md") == "/vault/final_path.md"
        assert tracker.resolve_path("/vault/final_path.md") == "/vault/final_path.md"
        assert tracker.resolve_path("/vault/unknown.md") == "/vault/unknown.md"

    def test_path_resolution_cycle_prevention(self, tracker):
        """Test that path resolution prevents infinite cycles."""
        # Create a cycle: A -> B -> A
        tracker.path_map = {
            "/vault/path_a.md": "/vault/path_b.md",
            "/vault/path_b.md": "/vault/path_a.md"
        }

        # Should not get stuck in infinite loop
        result = tracker.resolve_path("/vault/path_a.md")
        assert result in ["/vault/path_a.md", "/vault/path_b.md"]

    def test_get_path_changes_filtering(self, tracker, vault_root):
        """Test filtering of path changes."""
        # Create sample path changes
        now = datetime.utcnow()
        old_change = PathChange(
            vault_id="test_vault",
            old_note_id="old1",
            new_note_id="new1",
            old_path=vault_root / "old1.md",
            new_path=vault_root / "new1.md",
            old_checksum="abc",
            new_checksum="abc",
            detected_at=now - timedelta(hours=2),
            change_type="rename"
        )

        recent_change = PathChange(
            vault_id="test_vault",
            old_note_id="old2",
            new_note_id="new2",
            old_path=vault_root / "old2.md",
            new_path=vault_root / "new2.md",
            old_checksum="def",
            new_checksum="def",
            detected_at=now,
            change_type="move"
        )

        tracker.path_changes = [old_change, recent_change]

        # Filter by time
        recent_changes = tracker.get_path_changes(since=now - timedelta(hours=1))
        assert len(recent_changes) == 1
        assert recent_changes[0] == recent_change

        # Filter by change type
        rename_changes = tracker.get_path_changes(change_types=["rename"])
        assert len(rename_changes) == 1
        assert rename_changes[0] == old_change

        # Filter by both
        filtered_changes = tracker.get_path_changes(
            since=now - timedelta(hours=3),
            change_types=["move"]
        )
        assert len(filtered_changes) == 1
        assert filtered_changes[0] == recent_change

    def test_tracking_history_cleanup(self, tracker, vault_root):
        """Test cleanup of tracking history."""
        # Set low limit for testing
        tracker.max_tracking_history = 3

        # Add more changes than limit
        for i in range(5):
            change = PathChange(
                vault_id="test_vault",
                old_note_id=f"old{i}",
                new_note_id=f"new{i}",
                old_path=vault_root / f"old{i}.md",
                new_path=vault_root / f"new{i}.md",
                old_checksum=f"checksum{i}",
                new_checksum=f"checksum{i}"
            )
            tracker._record_path_change(change)

        # Trigger cleanup
        tracker._cleanup_tracking_history()

        # Should keep only recent changes
        assert len(tracker.path_changes) <= int(tracker.max_tracking_history * 0.8)

        # Mappings should be rebuilt from remaining changes
        assert len(tracker.note_id_map) == len(tracker.path_changes)
        assert len(tracker.path_map) == len(tracker.path_changes)

    def test_get_statistics(self, tracker, vault_root):
        """Test statistics generation."""
        # Add some sample changes
        rename_change = PathChange(
            vault_id="test_vault",
            old_note_id="old1",
            new_note_id="new1",
            old_path=vault_root / "old1.md",
            new_path=vault_root / "new1.md",
            old_checksum="abc123",
            new_checksum="abc123",
            change_type="rename"
        )

        move_change = PathChange(
            vault_id="test_vault",
            old_note_id="folder1/note",
            new_note_id="folder2/note",
            old_path=vault_root / "folder1" / "note.md",
            new_path=vault_root / "folder2" / "note.md",
            old_checksum="def456",
            new_checksum="ghi789",  # Content changed
            change_type="move"
        )

        tracker._record_path_change(rename_change)
        tracker._record_path_change(move_change)

        stats = tracker.get_statistics()
        assert stats["total_changes"] == 2
        assert stats["note_id_mappings"] == 2
        assert stats["path_mappings"] == 2
        assert stats["content_changes"] == 1  # Only move_change has content change
        assert stats["rename"] == 1
        assert stats["move"] == 1

    def test_clear_tracking_data(self, tracker, vault_root):
        """Test clearing all tracking data."""
        # Add some data
        change = PathChange(
            vault_id="test_vault",
            old_note_id="old",
            new_note_id="new",
            old_path=vault_root / "old.md",
            new_path=vault_root / "new.md",
            old_checksum="abc123",
            new_checksum="abc123"
        )
        tracker._record_path_change(change)

        # Verify data exists
        assert len(tracker.path_changes) > 0
        assert len(tracker.note_id_map) > 0
        assert len(tracker.path_map) > 0

        # Clear data
        tracker.clear_tracking_data()

        # Verify data is cleared
        assert len(tracker.path_changes) == 0
        assert len(tracker.note_id_map) == 0
        assert len(tracker.path_map) == 0


class TestPathTrackerFactory:
    """Test path tracker factory function."""

    def test_create_path_tracker(self, tmp_path):
        """Test factory function for creating path tracker."""
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        state_store = Mock(spec=StateStore)

        tracker = create_path_tracker(
            vault_id="test_vault",
            vault_root=vault_root,
            state_store=state_store
        )

        assert isinstance(tracker, ObsidianPathTracker)
        assert tracker.vault_id == "test_vault"
        assert tracker.vault_root == vault_root
        assert tracker.state_store == state_store
        assert tracker.similarity_threshold == 0.8
        assert tracker.max_tracking_history == 1000


if __name__ == "__main__":
    pytest.main([__file__])