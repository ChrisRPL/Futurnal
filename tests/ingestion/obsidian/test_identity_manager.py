"""Tests for the NoteIdentityManager with UUID-based note identity system."""

import pytest
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from futurnal.ingestion.local.state import FileRecord, StateStore
from futurnal.ingestion.obsidian.identity_manager import NoteIdentityManager, NoteIdentity


@pytest.fixture
def temp_vault():
    """Create a temporary vault directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_root = Path(temp_dir) / "test_vault"
        vault_root.mkdir()
        yield vault_root


@pytest.fixture
def identity_manager(temp_vault):
    """Create a NoteIdentityManager for testing."""
    return NoteIdentityManager(
        vault_id="test_vault",
        vault_root=temp_vault
    )


@pytest.fixture
def sample_notes(temp_vault):
    """Create sample note files for testing."""
    notes = []

    # Create some test notes
    note_paths = [
        "note1.md",
        "projects/project1.md",
        "projects/archive/old_note.md",
        "daily/2024-01-01.md"
    ]

    for note_path in note_paths:
        full_path = temp_vault / note_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"# {note_path}\n\nSample content for {note_path}")
        notes.append(full_path)

    return notes


class TestNoteIdentityManager:
    """Test cases for NoteIdentityManager."""

    def test_initialization(self, temp_vault):
        """Test that NoteIdentityManager initializes correctly."""
        manager = NoteIdentityManager("test_vault", temp_vault)

        assert manager.vault_id == "test_vault"
        assert manager.vault_root == temp_vault
        assert manager.db_path.exists()

        # Check database schema
        with sqlite3.connect(manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='note_identities'
            """)
            assert cursor.fetchone() is not None

    def test_get_note_id_creates_new_uuid(self, identity_manager, sample_notes):
        """Test that get_note_id creates new UUIDs for new notes."""
        note_path = sample_notes[0]

        # First call should create a new UUID
        note_id = identity_manager.get_note_id(note_path)
        assert note_id is not None
        assert len(note_id) == 36  # UUID length

        # Second call should return the same UUID
        same_note_id = identity_manager.get_note_id(note_path)
        assert note_id == same_note_id

    def test_get_note_id_different_paths_different_uuids(self, identity_manager, sample_notes):
        """Test that different paths get different UUIDs."""
        note_id1 = identity_manager.get_note_id(sample_notes[0])
        note_id2 = identity_manager.get_note_id(sample_notes[1])

        assert note_id1 != note_id2

    def test_update_path_preserves_identity(self, identity_manager, sample_notes):
        """Test that updating a path preserves the note identity."""
        original_path = sample_notes[0]
        new_path = original_path.parent / "renamed_note.md"

        # Create initial identity
        note_id = identity_manager.get_note_id(original_path)

        # Update the path
        success = identity_manager.update_path(note_id, new_path)
        assert success

        # Check that the new path now maps to the same UUID
        retrieved_note_id = identity_manager.get_note_id_by_path(new_path)
        assert retrieved_note_id == note_id

        # Check that old path no longer maps to anything
        old_note_id = identity_manager.get_note_id_by_path(original_path)
        assert old_note_id is None

    def test_get_path_by_note_id(self, identity_manager, sample_notes):
        """Test retrieving path by note ID."""
        note_path = sample_notes[0]
        note_id = identity_manager.get_note_id(note_path)

        retrieved_path = identity_manager.get_path_by_note_id(note_id)
        assert retrieved_path == note_path

    def test_get_note_id_by_path(self, identity_manager, sample_notes):
        """Test retrieving note ID by path."""
        note_path = sample_notes[0]
        note_id = identity_manager.get_note_id(note_path)

        retrieved_note_id = identity_manager.get_note_id_by_path(note_path)
        assert retrieved_note_id == note_id

    def test_remove_note(self, identity_manager, sample_notes):
        """Test removing a note identity."""
        note_path = sample_notes[0]
        note_id = identity_manager.get_note_id(note_path)

        # Remove the note
        success = identity_manager.remove_note(note_id)
        assert success

        # Check that it's gone
        retrieved_path = identity_manager.get_path_by_note_id(note_id)
        assert retrieved_path is None

        retrieved_note_id = identity_manager.get_note_id_by_path(note_path)
        assert retrieved_note_id is None

    def test_list_all_identities(self, identity_manager, sample_notes):
        """Test listing all note identities."""
        # Create identities for all sample notes
        note_ids = []
        for note_path in sample_notes:
            note_id = identity_manager.get_note_id(note_path)
            note_ids.append(note_id)

        # List all identities
        identities = identity_manager.list_all_identities()

        assert len(identities) == len(sample_notes)
        retrieved_note_ids = [identity.note_id for identity in identities]

        for note_id in note_ids:
            assert note_id in retrieved_note_ids

    def test_migrate_from_path_ids(self, identity_manager, sample_notes, temp_vault):
        """Test migration from path-based IDs to UUID system."""
        # Create a state store with file records
        state_store = StateStore(temp_vault / ".futurnal" / "state.db")

        # Add file records for sample notes
        for note_path in sample_notes:
            record = FileRecord(
                path=note_path,
                size=len(note_path.read_text()),
                mtime=note_path.stat().st_mtime,
                sha256="dummy_checksum"
            )
            state_store.upsert(record)

        # Perform migration
        migration_map = identity_manager.migrate_from_path_ids(state_store)

        # Check that migration map is created
        assert len(migration_map) == len(sample_notes)

        # Check that all path-based IDs are mapped to UUIDs
        for old_id, new_id in migration_map.items():
            assert len(new_id) == 36  # UUID length
            assert old_id != new_id  # Should be different

        # Check that new UUIDs are actually stored
        identities = identity_manager.list_all_identities()
        stored_uuids = [identity.note_id for identity in identities]

        for new_id in migration_map.values():
            assert new_id in stored_uuids

    def test_get_stats(self, identity_manager, sample_notes):
        """Test getting statistics about the identity database."""
        # Create identities for sample notes
        for note_path in sample_notes:
            identity_manager.get_note_id(note_path)

        stats = identity_manager.get_stats()

        assert stats["total_notes"] == len(sample_notes)
        assert stats["vault_id"] == "test_vault"
        assert "db_path" in stats
        assert "earliest_note" in stats
        assert "latest_note" in stats

    def test_non_markdown_files_rejected(self, identity_manager, temp_vault):
        """Test that non-markdown files are rejected."""
        text_file = temp_vault / "readme.txt"
        text_file.write_text("Not a markdown file")

        with pytest.raises(ValueError, match="only handles markdown files"):
            identity_manager.get_note_id(text_file)

    def test_relative_path_handling(self, identity_manager, temp_vault):
        """Test that paths are properly converted to relative paths."""
        # Create a note in a subdirectory
        subdir = temp_vault / "subdir"
        subdir.mkdir()
        note_path = subdir / "note.md"
        note_path.write_text("# Test Note")

        note_id = identity_manager.get_note_id(note_path)

        # Check that the stored path is relative
        with sqlite3.connect(identity_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT current_path FROM note_identities WHERE note_id = ?
            """, (note_id,))
            stored_path = cursor.fetchone()[0]

        assert stored_path == "subdir/note.md"  # Should be relative

    def test_concurrent_access_safety(self, identity_manager, sample_notes):
        """Test that concurrent access doesn't cause issues."""
        note_path = sample_notes[0]

        # Simulate concurrent access by calling get_note_id multiple times
        note_ids = []
        for _ in range(10):
            note_id = identity_manager.get_note_id(note_path)
            note_ids.append(note_id)

        # All should return the same UUID
        assert len(set(note_ids)) == 1

    def test_update_nonexistent_note_fails(self, identity_manager, temp_vault):
        """Test that updating a non-existent note ID fails gracefully."""
        fake_note_id = "00000000-0000-0000-0000-000000000000"
        new_path = temp_vault / "nonexistent.md"

        success = identity_manager.update_path(fake_note_id, new_path)
        assert not success

    def test_remove_nonexistent_note_fails(self, identity_manager):
        """Test that removing a non-existent note ID fails gracefully."""
        fake_note_id = "00000000-0000-0000-0000-000000000000"

        success = identity_manager.remove_note(fake_note_id)
        assert not success


class TestNoteIdentity:
    """Test cases for NoteIdentity dataclass."""

    def test_note_identity_creation(self, temp_vault):
        """Test creating a NoteIdentity object."""
        identity = NoteIdentity(
            note_id="test-uuid",
            current_path=temp_vault / "test.md",
            created_at=datetime.utcnow(),
            vault_id="test_vault"
        )

        assert identity.note_id == "test-uuid"
        assert identity.current_path == temp_vault / "test.md"
        assert identity.vault_id == "test_vault"

    def test_note_identity_to_dict(self, temp_vault):
        """Test converting NoteIdentity to dictionary."""
        created_at = datetime.utcnow()
        identity = NoteIdentity(
            note_id="test-uuid",
            current_path=temp_vault / "test.md",
            created_at=created_at,
            vault_id="test_vault"
        )

        identity_dict = identity.to_dict()

        assert identity_dict["note_id"] == "test-uuid"
        assert identity_dict["current_path"] == str(temp_vault / "test.md")
        assert identity_dict["created_at"] == created_at.isoformat()
        assert identity_dict["vault_id"] == "test_vault"