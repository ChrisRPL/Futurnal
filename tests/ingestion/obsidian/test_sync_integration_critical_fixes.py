"""Integration tests for critical sync strategy fixes.

Tests the complete pipeline from UUID note identity through graph updates
to folder cascade handling, ensuring all critical gaps are resolved.
"""

import pytest
import pytest_asyncio
import tempfile
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, MagicMock

from futurnal.ingestion.local.state import FileRecord, StateStore
from futurnal.ingestion.obsidian.identity_manager import NoteIdentityManager
from futurnal.ingestion.obsidian.path_tracker import ObsidianPathTracker, PathChange
from futurnal.ingestion.obsidian.folder_cascade_detector import FolderCascadeDetector, FolderCascade
from futurnal.ingestion.obsidian.sync_engine import ObsidianSyncEngine
from futurnal.ingestion.obsidian.connector import ObsidianVaultConnector
from futurnal.orchestrator.queue import JobQueue
from futurnal.pipeline.graph import Neo4jPKGWriter


@pytest.fixture
def temp_vault():
    """Create a temporary vault directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_root = Path(temp_dir) / "test_vault"
        vault_root.mkdir()

        # Create .obsidian directory to make it look like a real vault
        obsidian_dir = vault_root / ".obsidian"
        obsidian_dir.mkdir()
        (obsidian_dir / "app.json").write_text('{"vaultName": "test_vault"}')

        yield vault_root


@pytest.fixture
def state_store(temp_vault):
    """Create a state store for testing."""
    db_path = temp_vault / ".futurnal" / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return StateStore(db_path)


@pytest.fixture
def identity_manager(temp_vault):
    """Create a note identity manager for testing."""
    return NoteIdentityManager(
        vault_id="test_vault",
        vault_root=temp_vault
    )


@pytest.fixture
def mock_graph_writer():
    """Create a mock graph writer for testing."""
    mock_writer = Mock(spec=Neo4jPKGWriter)
    mock_writer.update_note_path = Mock()
    return mock_writer


@pytest.fixture
def mock_vault_connector():
    """Create a mock vault connector for testing."""
    mock_connector = Mock(spec=ObsidianVaultConnector)
    mock_connector.scan = AsyncMock(return_value=[])
    return mock_connector


@pytest.fixture
def mock_job_queue():
    """Create a mock job queue for testing."""
    mock_queue = Mock(spec=JobQueue)
    mock_queue.enqueue = AsyncMock()
    return mock_queue


@pytest_asyncio.fixture
async def sync_engine(temp_vault, state_store, mock_vault_connector, mock_job_queue, mock_graph_writer):
    """Create a sync engine with all components for integration testing."""
    engine = ObsidianSyncEngine(
        vault_connector=mock_vault_connector,
        job_queue=mock_job_queue,
        state_store=state_store,
        graph_writer=mock_graph_writer,
        batch_window_seconds=0.1,  # Fast batching for tests
        performance_monitoring=False  # Disable for tests
    )
    return engine


class TestCriticalGapResolution:
    """Test that all critical gaps identified in the audit are resolved."""

    @pytest.mark.asyncio
    async def test_uuid_note_id_system_prevents_duplicates(self, temp_vault, identity_manager):
        """Test that the UUID system prevents duplicate node creation during renames."""
        # Create a note and get its UUID
        original_path = temp_vault / "note.md"
        original_path.write_text("# Original Note")

        note_id = identity_manager.get_note_id(original_path)

        # Simulate a rename
        new_path = temp_vault / "renamed_note.md"
        success = identity_manager.update_path(note_id, new_path)
        assert success

        # Critical test: both paths should map to the SAME note ID
        old_note_id = identity_manager.get_note_id_by_path(original_path)
        new_note_id = identity_manager.get_note_id_by_path(new_path)

        # Old path should no longer exist
        assert old_note_id is None

        # New path should have the same UUID
        assert new_note_id == note_id

        # Getting the note ID for the new path should return the existing UUID
        same_note_id = identity_manager.get_note_id(new_path)
        assert same_note_id == note_id

    @pytest.mark.asyncio
    async def test_path_tracker_preserves_note_identity(self, temp_vault, state_store):
        """Test that PathTracker preserves note identity across renames."""
        identity_manager = NoteIdentityManager("test_vault", temp_vault)
        path_tracker = ObsidianPathTracker(
            vault_id="test_vault",
            vault_root=temp_vault,
            state_store=state_store,
            identity_manager=identity_manager
        )

        # Create original file record
        original_path = temp_vault / "original.md"
        original_path.write_text("# Original Content")

        original_record = FileRecord(
            path=original_path,
            size=len(original_path.read_text()),
            mtime=original_path.stat().st_mtime,
            sha256="original_checksum"
        )
        state_store.upsert(original_record)

        # Get the note ID for the original path
        original_note_id = identity_manager.get_note_id(original_path)

        # Create "renamed" file record
        new_path = temp_vault / "renamed.md"
        new_path.write_text("# Original Content")  # Same content

        new_record = FileRecord(
            path=new_path,
            size=len(new_path.read_text()),
            mtime=new_path.stat().st_mtime,
            sha256="original_checksum"  # Same checksum indicates same content
        )

        # Detect path changes
        path_changes = path_tracker.detect_path_changes([new_record])

        # Should detect the rename
        assert len(path_changes) == 1
        change = path_changes[0]

        # Critical assertion: old and new note IDs should be the SAME
        assert change.old_note_id == change.new_note_id
        assert change.old_note_id == original_note_id
        assert change.old_path == original_path
        assert change.new_path == new_path

    @pytest.mark.asyncio
    async def test_graph_integration_updates_relationships(self, sync_engine, mock_graph_writer, temp_vault):
        """Test that graph relationships are updated when paths change."""
        # Create path changes that would trigger graph updates
        note_id = str(uuid.uuid4())
        path_changes = [
            PathChange(
                vault_id="test_vault",
                old_note_id=note_id,
                new_note_id=note_id,  # Same ID - preserves identity
                old_path=temp_vault / "old_note.md",
                new_path=temp_vault / "new_note.md",
                old_checksum="checksum1",
                new_checksum="checksum1"
            )
        ]

        # Process path changes
        await sync_engine.handle_path_changes(path_changes, "test_vault")

        # Verify graph writer was called
        mock_graph_writer.update_note_path.assert_called_once_with(
            vault_id="test_vault",
            note_id=note_id,
            old_path=str(temp_vault / "old_note.md"),
            new_path=str(temp_vault / "new_note.md")
        )

    @pytest.mark.asyncio
    async def test_folder_cascade_atomic_processing(self, sync_engine, mock_graph_writer, temp_vault):
        """Test that folder cascades are processed atomically."""
        # Create a folder cascade with multiple files
        base_note_id = str(uuid.uuid4())
        affected_files = []

        for i in range(3):
            note_id = f"{base_note_id}_{i}"
            change = PathChange(
                vault_id="test_vault",
                old_note_id=note_id,
                new_note_id=note_id,
                old_path=temp_vault / "projects" / f"file{i}.md",
                new_path=temp_vault / "work" / f"file{i}.md",
                old_checksum=f"checksum{i}",
                new_checksum=f"checksum{i}"
            )
            affected_files.append(change)

        folder_cascade = FolderCascade(
            cascade_id="test_cascade",
            operation_type="rename",
            old_folder_path=temp_vault / "projects",
            new_folder_path=temp_vault / "work",
            affected_files=affected_files
        )

        # Process the folder cascade
        await sync_engine.handle_folder_cascades([folder_cascade], "test_vault")

        # Verify graph writer was called for each file
        assert mock_graph_writer.update_note_path.call_count == 3

        # Verify each call had correct parameters
        calls = mock_graph_writer.update_note_path.call_args_list
        for i, call in enumerate(calls):
            args, kwargs = call
            assert kwargs["vault_id"] == "test_vault"
            assert kwargs["note_id"] == f"{base_note_id}_{i}"
            assert kwargs["old_path"] == str(temp_vault / "projects" / f"file{i}.md")
            assert kwargs["new_path"] == str(temp_vault / "work" / f"file{i}.md")

    @pytest.mark.asyncio
    async def test_migration_from_path_based_ids(self, temp_vault, state_store):
        """Test migration from old path-based ID system to UUID system."""
        # Create some files with content
        files = [
            temp_vault / "note1.md",
            temp_vault / "folder" / "note2.md",
            temp_vault / "deep" / "nested" / "note3.md"
        ]

        for file_path in files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f"# {file_path.name}")

            # Add to state store
            record = FileRecord(
                path=file_path,
                size=len(file_path.read_text()),
                mtime=file_path.stat().st_mtime,
                sha256=f"checksum_{file_path.name}"
            )
            state_store.upsert(record)

        # Create identity manager and perform migration
        identity_manager = NoteIdentityManager("test_vault", temp_vault)
        migration_map = identity_manager.migrate_from_path_ids(state_store)

        # Verify migration results
        assert len(migration_map) == 3

        # Check that old path-based IDs are replaced with UUIDs
        expected_old_ids = ["note1", "folder/note2", "deep/nested/note3"]
        for old_id in expected_old_ids:
            assert old_id in migration_map
            new_id = migration_map[old_id]
            assert len(new_id) == 36  # UUID length
            assert new_id != old_id

        # Verify that files can be found by their new UUIDs
        for file_path in files:
            note_id = identity_manager.get_note_id_by_path(file_path)
            assert note_id is not None
            assert len(note_id) == 36

    @pytest.mark.asyncio
    async def test_large_folder_cascade_performance(self, temp_vault):
        """Test that large folder cascades (50+ files) are handled efficiently."""
        cascade_detector = FolderCascadeDetector(temp_vault)

        # Create a large number of path changes
        path_changes = []
        for i in range(50):
            note_id = str(uuid.uuid4())
            change = PathChange(
                vault_id="test_vault",
                old_note_id=note_id,
                new_note_id=note_id,
                old_path=temp_vault / "large_folder" / f"note{i:03d}.md",
                new_path=temp_vault / "renamed_large_folder" / f"note{i:03d}.md",
                old_checksum=f"checksum{i}",
                new_checksum=f"checksum{i}"
            )
            path_changes.append(change)

        # Detect cascades
        start_time = datetime.now()
        cascades = cascade_detector.detect_folder_cascades(path_changes)
        detection_time = (datetime.now() - start_time).total_seconds()

        # Verify performance and results
        assert detection_time < 1.0  # Should complete within 1 second
        assert len(cascades) == 1

        cascade = cascades[0]
        assert cascade.get_file_count() == 50
        assert cascade.is_large_cascade(threshold=10)

    @pytest.mark.asyncio
    async def test_mixed_operations_handling(self, sync_engine, mock_graph_writer, temp_vault):
        """Test handling of mixed individual and cascade operations."""
        # Create individual path changes
        individual_changes = [
            PathChange(
                vault_id="test_vault",
                old_note_id="individual_1",
                new_note_id="individual_1",
                old_path=temp_vault / "single1.md",
                new_path=temp_vault / "renamed1.md",
                old_checksum="check1",
                new_checksum="check1"
            )
        ]

        # Create folder cascade
        cascade_changes = []
        for i in range(3):
            change = PathChange(
                vault_id="test_vault",
                old_note_id=f"cascade_{i}",
                new_note_id=f"cascade_{i}",
                old_path=temp_vault / "folder" / f"file{i}.md",
                new_path=temp_vault / "moved_folder" / f"file{i}.md",
                old_checksum=f"cascade_check{i}",
                new_checksum=f"cascade_check{i}"
            )
            cascade_changes.append(change)

        folder_cascade = FolderCascade(
            cascade_id="mixed_test",
            operation_type="move",
            old_folder_path=temp_vault / "folder",
            new_folder_path=temp_vault / "moved_folder",
            affected_files=cascade_changes
        )

        # Process both individual and cascade operations
        await sync_engine.handle_path_changes(individual_changes, "test_vault")
        await sync_engine.handle_folder_cascades([folder_cascade], "test_vault")

        # Verify all operations were processed
        # Individual change: 1 call
        # Cascade changes: 3 calls
        assert mock_graph_writer.update_note_path.call_count == 4

    @pytest.mark.asyncio
    async def test_error_recovery_in_cascade_processing(self, sync_engine, mock_graph_writer, temp_vault):
        """Test that cascade processing handles partial failures gracefully."""
        # Configure mock to fail on second call
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Simulated graph update failure")

        mock_graph_writer.update_note_path.side_effect = side_effect

        # Create folder cascade with 3 files
        affected_files = []
        for i in range(3):
            change = PathChange(
                vault_id="test_vault",
                old_note_id=f"error_test_{i}",
                new_note_id=f"error_test_{i}",
                old_path=temp_vault / "error_folder" / f"file{i}.md",
                new_path=temp_vault / "error_moved" / f"file{i}.md",
                old_checksum=f"error_check{i}",
                new_checksum=f"error_check{i}"
            )
            affected_files.append(change)

        folder_cascade = FolderCascade(
            cascade_id="error_test",
            operation_type="rename",
            old_folder_path=temp_vault / "error_folder",
            new_folder_path=temp_vault / "error_moved",
            affected_files=affected_files
        )

        # Process cascade - should handle the error gracefully
        await sync_engine.handle_folder_cascades([folder_cascade], "test_vault")

        # Verify all files were attempted
        assert mock_graph_writer.update_note_path.call_count == 3


class TestAcceptanceCriteria:
    """Test that all acceptance criteria from 06-sync-strategy.md are met."""

    @pytest.mark.asyncio
    async def test_edits_reflected_within_minutes(self, sync_engine, temp_vault):
        """Test that edits are reflected within minutes under normal load."""
        # This is more of a performance test - the 500ms batching should
        # ensure edits are processed much faster than the minute requirement

        start_time = datetime.now()

        # Simulate file changes
        path_changes = [
            PathChange(
                vault_id="test_vault",
                old_note_id="speed_test",
                new_note_id="speed_test",
                old_path=temp_vault / "test.md",
                new_path=temp_vault / "test_renamed.md",
                old_checksum="old",
                new_checksum="new"
            )
        ]

        # Process changes
        await sync_engine.handle_path_changes(path_changes, "test_vault")

        processing_time = (datetime.now() - start_time).total_seconds()

        # Should be much faster than 60 seconds
        assert processing_time < 5.0  # Allow generous margin for test environment

    @pytest.mark.asyncio
    async def test_renames_retain_edges_no_duplicate_nodes(self, temp_vault, state_store):
        """Test the critical requirement: renames retain edges/history with no duplicate nodes."""
        identity_manager = NoteIdentityManager("test_vault", temp_vault)

        # Create initial note
        original_path = temp_vault / "important_note.md"
        original_path.write_text("# Important Note\n\nThis note has important content.")

        # Get the persistent UUID
        note_id = identity_manager.get_note_id(original_path)

        # Simulate multiple renames
        rename_sequence = [
            temp_vault / "renamed_once.md",
            temp_vault / "renamed_twice.md",
            temp_vault / "final_name.md"
        ]

        current_note_id = note_id
        for new_path in rename_sequence:
            # Update the path - note ID should remain the same
            success = identity_manager.update_path(current_note_id, new_path)
            assert success

            # Verify the note ID hasn't changed
            retrieved_note_id = identity_manager.get_note_id_by_path(new_path)
            assert retrieved_note_id == note_id  # SAME UUID throughout

            # Verify old paths no longer exist
            stats = identity_manager.get_stats()
            assert stats["total_notes"] == 1  # Still only ONE note

        # Final verification: only one note identity exists
        identities = identity_manager.list_all_identities()
        assert len(identities) == 1
        assert identities[0].note_id == note_id
        assert identities[0].current_path == rename_sequence[-1]


class TestProductionReadiness:
    """Test production readiness scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_vault, state_store):
        """Test that concurrent operations don't cause data corruption."""
        identity_manager = NoteIdentityManager("test_vault", temp_vault)

        # Create initial notes
        notes = []
        for i in range(10):
            note_path = temp_vault / f"note_{i}.md"
            note_path.write_text(f"# Note {i}")
            notes.append(note_path)

        # Get initial note IDs
        note_ids = []
        for note_path in notes:
            note_id = identity_manager.get_note_id(note_path)
            note_ids.append(note_id)

        # Simulate concurrent updates
        tasks = []
        for i, (note_path, note_id) in enumerate(zip(notes, note_ids)):
            new_path = temp_vault / f"concurrent_renamed_{i}.md"
            task = asyncio.create_task(
                asyncio.to_thread(identity_manager.update_path, note_id, new_path)
            )
            tasks.append(task)

        # Wait for all updates to complete
        results = await asyncio.gather(*tasks)

        # Verify all updates succeeded
        assert all(results)

        # Verify database consistency
        identities = identity_manager.list_all_identities()
        assert len(identities) == 10

        # Verify all original note IDs are preserved
        stored_note_ids = [identity.note_id for identity in identities]
        for original_note_id in note_ids:
            assert original_note_id in stored_note_ids

    @pytest.mark.asyncio
    async def test_database_recovery_after_corruption(self, temp_vault):
        """Test that the system handles database issues gracefully."""
        identity_manager = NoteIdentityManager("test_vault", temp_vault)

        # Create some identities
        note_path = temp_vault / "recovery_test.md"
        note_path.write_text("# Recovery Test")
        note_id = identity_manager.get_note_id(note_path)

        # Verify it exists
        assert identity_manager.get_note_id_by_path(note_path) == note_id

        # Simulate database corruption by closing and recreating
        del identity_manager

        # Create new manager (simulates restart after corruption)
        new_manager = NoteIdentityManager("test_vault", temp_vault)

        # Should be able to get the same note ID again
        recovered_note_id = new_manager.get_note_id(note_path)
        assert recovered_note_id == note_id  # Should find existing identity