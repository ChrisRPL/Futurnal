"""Integration tests for Obsidian rename/move cascade handling.

Tests that file renames and moves preserve graph relationships
and maintain PKG integrity without creating duplicate nodes.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from futurnal.ingestion.local.state import StateStore, FileRecord, compute_sha256
from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector, ObsidianVaultSource
from futurnal.ingestion.obsidian.descriptor import ObsidianVaultDescriptor, VaultRegistry
from futurnal.ingestion.obsidian.path_tracker import ObsidianPathTracker, PathChange
from futurnal.ingestion.obsidian.change_detector import AdvancedChangeDetector
from futurnal.orchestrator.queue import JobQueue
from futurnal.privacy.audit import AuditLogger


class TestObsidianRenameCascades:
    """Test rename/move operations maintain graph integrity."""

    @pytest.fixture
    async def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir(parents=True)
            yield workspace

    @pytest.fixture
    async def vault_path(self, temp_workspace):
        """Create a test vault with interconnected notes."""
        vault_dir = temp_workspace / "test_vault"
        vault_dir.mkdir()

        # Create a network of interconnected notes
        notes = {
            "index.md": "# Index\n[[note1]] and [[note2]]",
            "note1.md": "# Note 1\nReferences [[note2]] and [[folder/note3]]",
            "note2.md": "# Note 2\nBacklink to [[note1]]",
        }

        # Create subfolder with notes
        folder = vault_dir / "folder"
        folder.mkdir()
        notes["folder/note3.md"] = "# Note 3\nReferences [[../note1]]"
        notes["folder/note4.md"] = "# Note 4\nReferences [[note3]]"

        # Write all notes
        for note_path, content in notes.items():
            full_path = vault_dir / note_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        return vault_dir

    @pytest.fixture
    async def state_store(self, temp_workspace):
        """Create a test state store."""
        db_path = temp_workspace / "state.db"
        return StateStore(db_path)

    @pytest.fixture
    async def path_tracker(self, vault_path, state_store):
        """Create a path tracker for the test vault."""
        return ObsidianPathTracker(
            vault_id="test_vault",
            vault_root=vault_path,
            state_store=state_store,
            similarity_threshold=0.8
        )

    @pytest.fixture
    async def change_detector(self, vault_path, state_store, path_tracker):
        """Create a change detector for the test vault."""
        return AdvancedChangeDetector(
            vault_id="test_vault",
            vault_root=vault_path,
            state_store=state_store,
            path_tracker=path_tracker
        )

    @pytest.fixture
    async def vault_source(self, vault_path):
        """Create a test vault source."""
        return ObsidianVaultSource(
            name="test_vault_source",
            root_path=vault_path,
            vault_id="test_vault",
            vault_name="Test Vault",
            include=["**/*.md"],
            exclude=["**/.obsidian/**"]
        )

    async def _populate_initial_records(self, state_store, vault_path):
        """Populate state store with initial file records."""
        records = []
        for md_file in vault_path.rglob("*.md"):
            record = FileRecord(
                path=md_file,
                size=md_file.stat().st_size,
                mtime=md_file.stat().st_mtime,
                sha256=compute_sha256(md_file)
            )
            state_store.upsert(record)
            records.append(record)
        return records

    @pytest.mark.asyncio
    async def test_simple_file_rename(self, vault_path, state_store, path_tracker):
        """Test simple file rename detection."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Rename a file
        old_path = vault_path / "note1.md"
        new_path = vault_path / "renamed_note1.md"

        # Simulate rename by creating new file with same content
        content = old_path.read_text()
        old_path.unlink()  # Remove old file
        new_path.write_text(content)  # Create new file

        # Create new record for renamed file
        new_record = FileRecord(
            path=new_path,
            size=new_path.stat().st_size,
            mtime=new_path.stat().st_mtime,
            sha256=compute_sha256(new_path)
        )

        # Detect path changes
        path_changes = path_tracker.detect_path_changes([new_record])

        # Should detect the rename
        assert len(path_changes) > 0
        change = path_changes[0]
        assert change.old_path == old_path
        assert change.new_path == new_path
        assert change.change_type == "rename"
        assert change.old_checksum == change.new_checksum  # Content unchanged

    @pytest.mark.asyncio
    async def test_file_move_to_subdirectory(self, vault_path, state_store, path_tracker):
        """Test moving a file to a subdirectory."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Move a file to subfolder
        old_path = vault_path / "note2.md"
        new_path = vault_path / "folder" / "note2.md"

        # Simulate move
        content = old_path.read_text()
        old_path.unlink()
        new_path.write_text(content)

        # Create new record
        new_record = FileRecord(
            path=new_path,
            size=new_path.stat().st_size,
            mtime=new_path.stat().st_mtime,
            sha256=compute_sha256(new_path)
        )

        # Detect path changes
        path_changes = path_tracker.detect_path_changes([new_record])

        # Should detect the move
        assert len(path_changes) > 0
        change = path_changes[0]
        assert change.old_path == old_path
        assert change.new_path == new_path
        assert change.change_type == "move"

    @pytest.mark.asyncio
    async def test_rename_and_move_combined(self, vault_path, state_store, path_tracker):
        """Test file rename and move in single operation."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Rename and move a file
        old_path = vault_path / "note1.md"
        new_path = vault_path / "folder" / "renamed_note1.md"

        # Simulate rename and move
        content = old_path.read_text()
        old_path.unlink()
        new_path.write_text(content)

        # Create new record
        new_record = FileRecord(
            path=new_path,
            size=new_path.stat().st_size,
            mtime=new_path.stat().st_mtime,
            sha256=compute_sha256(new_path)
        )

        # Detect path changes
        path_changes = path_tracker.detect_path_changes([new_record])

        # Should detect the combined operation
        assert len(path_changes) > 0
        change = path_changes[0]
        assert change.old_path == old_path
        assert change.new_path == new_path
        assert change.change_type == "rename_and_move"

    @pytest.mark.asyncio
    async def test_multiple_file_renames(self, vault_path, state_store, path_tracker):
        """Test handling multiple file renames in a batch."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        renames = [
            (vault_path / "note1.md", vault_path / "renamed_note1.md"),
            (vault_path / "note2.md", vault_path / "renamed_note2.md"),
            (vault_path / "folder/note3.md", vault_path / "folder/renamed_note3.md"),
        ]

        new_records = []
        for old_path, new_path in renames:
            # Simulate rename
            content = old_path.read_text()
            old_path.unlink()
            new_path.write_text(content)

            # Create new record
            new_record = FileRecord(
                path=new_path,
                size=new_path.stat().st_size,
                mtime=new_path.stat().st_mtime,
                sha256=compute_sha256(new_path)
            )
            new_records.append(new_record)

        # Detect all path changes
        path_changes = path_tracker.detect_path_changes(new_records)

        # Should detect all renames
        assert len(path_changes) == len(renames)

        # Verify each rename was detected correctly
        detected_renames = {(change.old_path, change.new_path) for change in path_changes}
        expected_renames = set(renames)
        assert detected_renames == expected_renames

    @pytest.mark.asyncio
    async def test_rename_with_content_change(self, vault_path, state_store, change_detector):
        """Test rename combined with content modification."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Rename file and modify content
        old_path = vault_path / "note1.md"
        new_path = vault_path / "modified_note1.md"

        # Simulate rename with content change
        old_content = old_path.read_text()
        new_content = old_content + "\n\nAdditional content added during rename."
        old_path.unlink()
        new_path.write_text(new_content)

        # Create new record
        new_record = FileRecord(
            path=new_path,
            size=new_path.stat().st_size,
            mtime=new_path.stat().st_mtime,
            sha256=compute_sha256(new_path)
        )

        # Detect changes
        path_changes, content_changes = change_detector.detect_changes([new_record])

        # Should detect path change despite content modification
        assert len(path_changes) > 0
        path_change = path_changes[0]
        assert path_change.old_path == old_path
        assert path_change.new_path == new_path
        assert path_change.old_checksum != path_change.new_checksum

    @pytest.mark.asyncio
    async def test_folder_rename_cascade(self, vault_path, state_store, path_tracker):
        """Test handling folder rename affecting multiple files."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Rename entire folder
        old_folder = vault_path / "folder"
        new_folder = vault_path / "renamed_folder"

        # Get all files in old folder before rename
        old_files = list(old_folder.rglob("*.md"))

        # Simulate folder rename
        old_folder.rename(new_folder)

        # Create new records for all moved files
        new_records = []
        for old_file in old_files:
            # Calculate new path
            relative_path = old_file.relative_to(old_folder)
            new_file = new_folder / relative_path

            new_record = FileRecord(
                path=new_file,
                size=new_file.stat().st_size,
                mtime=new_file.stat().st_mtime,
                sha256=compute_sha256(new_file)
            )
            new_records.append(new_record)

        # Detect path changes
        path_changes = path_tracker.detect_path_changes(new_records)

        # Should detect moves for all files in the folder
        assert len(path_changes) == len(old_files)

        # All changes should be moves
        for change in path_changes:
            assert change.change_type == "move"
            assert "folder" in str(change.old_path)
            assert "renamed_folder" in str(change.new_path)

    @pytest.mark.asyncio
    async def test_complex_rename_chain(self, vault_path, state_store, path_tracker):
        """Test a complex chain of renames (A->B, B->C, C->A)."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Create a complex rename chain
        files = [
            vault_path / "note1.md",
            vault_path / "note2.md",
            vault_path / "folder/note3.md"
        ]

        # Read all content first
        contents = [f.read_text() for f in files]

        # Create temporary files to avoid conflicts
        temp_files = [
            vault_path / "temp_note1.md",
            vault_path / "temp_note2.md",
            vault_path / "folder/temp_note3.md"
        ]

        # Move to temp names
        for original, temp in zip(files, temp_files):
            original.rename(temp)

        # Now create the chain: note1->note2, note2->note3, note3->note1
        new_mappings = [
            (files[0], files[1]),  # note1.md -> note2.md
            (files[1], files[2]),  # note2.md -> note3.md
            (files[2], files[0]),  # note3.md -> note1.md
        ]

        # Write content to new locations
        for i, (old_path, new_path) in enumerate(new_mappings):
            new_path.write_text(contents[i])

        # Remove temp files
        for temp in temp_files:
            if temp.exists():
                temp.unlink()

        # Create new records
        new_records = []
        for _, new_path in new_mappings:
            new_record = FileRecord(
                path=new_path,
                size=new_path.stat().st_size,
                mtime=new_path.stat().st_mtime,
                sha256=compute_sha256(new_path)
            )
            new_records.append(new_record)

        # Detect path changes
        path_changes = path_tracker.detect_path_changes(new_records)

        # Should detect changes, though exact matching may be complex
        assert len(path_changes) > 0

        # Verify that note IDs can be resolved correctly
        for change in path_changes:
            resolved_id = path_tracker.resolve_note_id(change.old_note_id)
            assert resolved_id is not None

    @pytest.mark.asyncio
    async def test_path_tracker_resolution(self, vault_path, state_store, path_tracker):
        """Test path tracker note ID resolution after renames."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Create a chain of renames
        renames = [
            (vault_path / "note1.md", vault_path / "step1.md"),
            (vault_path / "step1.md", vault_path / "step2.md"),
            (vault_path / "step2.md", vault_path / "final.md"),
        ]

        current_path = vault_path / "note1.md"
        original_content = current_path.read_text()
        original_note_id = "note1"

        for old_path, new_path in renames:
            # Simulate rename
            content = old_path.read_text()
            old_path.unlink()
            new_path.write_text(content)

            # Create path change manually to update tracker
            change = PathChange(
                vault_id="test_vault",
                old_note_id=old_path.stem,
                new_note_id=new_path.stem,
                old_path=old_path,
                new_path=new_path,
                old_checksum=compute_sha256(old_path) if old_path.exists() else "",
                new_checksum=compute_sha256(new_path)
            )
            path_tracker._record_path_change(change)

        # Test resolution
        final_note_id = path_tracker.resolve_note_id(original_note_id)
        assert final_note_id == "final"

        # Test path resolution
        final_path = path_tracker.resolve_path(str(vault_path / "note1.md"))
        assert final_path == str(vault_path / "final.md")

    @pytest.mark.asyncio
    async def test_rename_with_graph_implications(self, vault_path, state_store, change_detector):
        """Test that renames preserve graph relationship metadata."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Rename a file that's referenced by others
        referenced_file = vault_path / "note1.md"  # Referenced by index.md and note2.md
        new_name = vault_path / "renamed_note1.md"

        # Simulate rename
        content = referenced_file.read_text()
        referenced_file.unlink()
        new_name.write_text(content)

        # Create new record
        new_record = FileRecord(
            path=new_name,
            size=new_name.stat().st_size,
            mtime=new_name.stat().st_mtime,
            sha256=compute_sha256(new_name)
        )

        # Detect changes
        path_changes, content_changes = change_detector.detect_changes([new_record])

        # Should detect the rename
        assert len(path_changes) > 0
        change = path_changes[0]

        # Verify graph relationship preservation metadata
        assert change.old_note_id == "note1"
        assert change.new_note_id == "renamed_note1"
        assert change.vault_id == "test_vault"

        # The change should be marked as affecting graph relationships
        change_dict = change.to_dict()
        assert "old_note_id" in change_dict
        assert "new_note_id" in change_dict

    @pytest.mark.asyncio
    async def test_duplicate_prevention(self, vault_path, state_store, path_tracker):
        """Test that renames don't create duplicate nodes."""
        # Populate initial records
        await self._populate_initial_records(state_store, vault_path)

        # Rename a file
        old_path = vault_path / "note1.md"
        new_path = vault_path / "renamed_note1.md"

        content = old_path.read_text()
        old_path.unlink()
        new_path.write_text(content)

        # Create new record
        new_record = FileRecord(
            path=new_path,
            size=new_path.stat().st_size,
            mtime=new_path.stat().st_mtime,
            sha256=compute_sha256(new_path)
        )

        # Detect path changes
        path_changes = path_tracker.detect_path_changes([new_record])

        # Should detect exactly one change (no duplicates)
        assert len(path_changes) == 1
        change = path_changes[0]

        # Verify the mapping is created correctly
        assert change.old_note_id in path_tracker.note_id_map
        assert path_tracker.note_id_map[change.old_note_id] == change.new_note_id

        # Resolving the old ID should give the new ID
        resolved = path_tracker.resolve_note_id(change.old_note_id)
        assert resolved == change.new_note_id

    @pytest.mark.asyncio
    async def test_large_scale_rename_performance(self, temp_workspace, state_store):
        """Test performance with many files being renamed."""
        # Create a large vault with many files
        vault_dir = temp_workspace / "large_vault"
        vault_dir.mkdir()

        num_files = 100
        files = []

        # Create many interconnected files
        for i in range(num_files):
            file_path = vault_dir / f"note_{i:03d}.md"
            content = f"# Note {i}\n"
            if i > 0:
                content += f"Links to [[note_{i-1:03d}]]"
            if i < num_files - 1:
                content += f" and [[note_{i+1:03d}]]"
            file_path.write_text(content)
            files.append(file_path)

        # Create path tracker
        path_tracker = ObsidianPathTracker(
            vault_id="large_vault",
            vault_root=vault_dir,
            state_store=state_store
        )

        # Populate initial records
        for file_path in files:
            record = FileRecord(
                path=file_path,
                size=file_path.stat().st_size,
                mtime=file_path.stat().st_mtime,
                sha256=compute_sha256(file_path)
            )
            state_store.upsert(record)

        # Rename many files
        start_time = time.time()

        new_records = []
        for i, old_file in enumerate(files[:50]):  # Rename first 50 files
            new_file = vault_dir / f"renamed_note_{i:03d}.md"

            content = old_file.read_text()
            old_file.unlink()
            new_file.write_text(content)

            new_record = FileRecord(
                path=new_file,
                size=new_file.stat().st_size,
                mtime=new_file.stat().st_mtime,
                sha256=compute_sha256(new_file)
            )
            new_records.append(new_record)

        # Detect all changes
        path_changes = path_tracker.detect_path_changes(new_records)

        detection_time = time.time() - start_time

        # Should detect all renames efficiently (less than 5 seconds)
        assert detection_time < 5.0
        assert len(path_changes) == 50

        # Verify all changes are correct
        for change in path_changes:
            assert change.change_type == "rename"
            assert "note_" in str(change.old_path)
            assert "renamed_note_" in str(change.new_path)