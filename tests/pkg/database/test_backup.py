"""Tests for PKG Backup and Restore.

Tests PKGBackupManager including:
- Backup file creation and structure
- Node and relationship export
- Restore from backup
- Pre-restore backup creation
- Backup verification
- Backup listing and purging

Uses testcontainers for real Neo4j instances - no mocks.

Follows production plan testing strategy:
docs/phase-1/pkg-graph-storage-production-plan/02-database-setup.md
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from futurnal.configuration.settings import StorageSettings
from futurnal.pkg.database.backup import PKGBackupManager
from futurnal.pkg.database.config import PKGDatabaseConfig
from futurnal.pkg.database.exceptions import PKGBackupError, PKGRestoreError
from futurnal.pkg.database.manager import PKGDatabaseManager

# Import test markers from conftest
from tests.pkg.conftest import (
    requires_docker,
    requires_neo4j,
    requires_testcontainers,
)


@pytest.fixture
def storage_settings(neo4j_container) -> StorageSettings:
    """Create StorageSettings from Neo4j test container."""
    return StorageSettings(
        neo4j_uri=neo4j_container.get_connection_url(),
        neo4j_username="neo4j",
        neo4j_password=SecretStr("testpassword"),
        neo4j_encrypted=False,
        chroma_path="/tmp/chroma",
    )


@pytest.fixture
def pkg_config(tmp_path) -> PKGDatabaseConfig:
    """Create PKGDatabaseConfig with test backup path."""
    return PKGDatabaseConfig(
        max_connection_retries=2,
        retry_delay_seconds=0.1,
        backup_path=tmp_path / "backups",
        backup_retention_days=7,
        max_backups=5,
    )


@pytest.fixture
def manager(storage_settings, pkg_config) -> PKGDatabaseManager:
    """Create connected PKGDatabaseManager."""
    mgr = PKGDatabaseManager(storage_settings, pkg_config)
    mgr.connect()
    yield mgr
    mgr.disconnect()


@pytest.fixture
def backup_manager(manager, pkg_config, tmp_path) -> PKGBackupManager:
    """Create PKGBackupManager instance."""
    return PKGBackupManager(manager, pkg_config, tmp_path)


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestBackupCreation:
    """Test backup creation."""

    def test_backup_creates_file(
        self, manager, backup_manager, clean_database
    ):
        """Backup creates JSON file with expected structure."""
        manager.initialize_schema()

        backup_path = backup_manager.backup()

        assert backup_path.exists()
        assert backup_path.suffix == ".json"
        assert backup_path.name.startswith("pkg-")

        # Verify JSON structure
        data = json.loads(backup_path.read_text())
        assert "version" in data
        assert "created_at" in data
        assert "metadata" in data
        assert "nodes" in data
        assert "relationships" in data

    def test_backup_with_comment(
        self, manager, backup_manager, clean_database
    ):
        """Backup with comment includes comment in filename."""
        manager.initialize_schema()

        backup_path = backup_manager.backup(comment="test-backup")

        assert "test-backup" in backup_path.name

    def test_backup_includes_all_nodes(
        self, manager, backup_manager, clean_database
    ):
        """Backup captures all node types."""
        manager.initialize_schema()

        # Create test data
        with manager.session() as session:
            session.run(
                """
                CREATE (p:Person {id: 'p1', name: 'Test Person'})
                CREATE (o:Organization {id: 'o1', name: 'Test Org'})
                CREATE (e:Event {id: 'e1', name: 'Test Event', timestamp: datetime()})
                """
            )

        backup_path = backup_manager.backup()
        data = json.loads(backup_path.read_text())

        # Verify metadata counts
        assert data["metadata"]["node_count"] >= 3

        # Verify nodes exist
        labels = [n.get("labels", []) for n in data["nodes"]]
        all_labels = [l for sublist in labels for l in sublist]
        assert "Person" in all_labels
        assert "Organization" in all_labels
        assert "Event" in all_labels

    def test_backup_includes_all_relationships(
        self, manager, backup_manager, clean_database
    ):
        """Backup captures all relationship types."""
        manager.initialize_schema()

        # Create test data with relationships
        with manager.session() as session:
            session.run(
                """
                CREATE (p:Person {id: 'p1', name: 'Test Person'})
                CREATE (o:Organization {id: 'o1', name: 'Test Org'})
                CREATE (p)-[:WORKS_AT {since: 2020}]->(o)
                """
            )

        backup_path = backup_manager.backup()
        data = json.loads(backup_path.read_text())

        # Verify relationships
        assert data["metadata"]["relationship_count"] >= 1
        rel_types = [r.get("type") for r in data["relationships"]]
        assert "WORKS_AT" in rel_types


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestBackupVerification:
    """Test backup verification."""

    def test_verify_valid_backup(
        self, manager, backup_manager, clean_database
    ):
        """Verify returns (True, message) for valid backup."""
        manager.initialize_schema()
        backup_path = backup_manager.backup()

        is_valid, message = backup_manager.verify(backup_path)

        assert is_valid is True
        assert "valid" in message.lower()

    def test_verify_invalid_json(self, backup_manager, tmp_path):
        """Verify returns (False, message) for invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json{")

        is_valid, message = backup_manager.verify(invalid_file)

        assert is_valid is False
        assert "JSON" in message or "json" in message.lower()

    def test_verify_missing_fields(self, backup_manager, tmp_path):
        """Verify returns (False, message) for missing required fields."""
        incomplete_file = tmp_path / "incomplete.json"
        incomplete_file.write_text(json.dumps({"version": 1}))

        is_valid, message = backup_manager.verify(incomplete_file)

        assert is_valid is False
        assert "Missing" in message or "missing" in message.lower()

    def test_verify_nonexistent_file(self, backup_manager, tmp_path):
        """Verify returns (False, message) for nonexistent file."""
        is_valid, message = backup_manager.verify(tmp_path / "nonexistent.json")

        assert is_valid is False
        assert "not found" in message.lower() or "File" in message


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestRestore:
    """Test restore operations."""

    def test_restore_to_empty_database(
        self, manager, backup_manager, clean_database
    ):
        """Restore populates empty database correctly."""
        manager.initialize_schema()

        # Create and backup data
        with manager.session() as session:
            session.run(
                """
                CREATE (p:Person {id: 'restore_test', name: 'Restore Person'})
                CREATE (o:Organization {id: 'restore_org', name: 'Restore Org'})
                CREATE (p)-[:WORKS_AT]->(o)
                """
            )

        backup_path = backup_manager.backup()

        # Clear database
        with manager.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

        # Verify empty
        with manager.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            assert result.single()["count"] == 0

        # Restore
        backup_manager.restore(backup_path)

        # Verify restored
        with manager.session() as session:
            result = session.run(
                "MATCH (p:Person {id: 'restore_test'}) RETURN p.name as name"
            )
            record = result.single()
            assert record is not None
            assert record["name"] == "Restore Person"

    def test_restore_creates_pre_restore_backup(
        self, manager, backup_manager, pkg_config, clean_database
    ):
        """Restore creates backup of current state before restoring."""
        manager.initialize_schema()

        # Create initial backup
        with manager.session() as session:
            session.run("CREATE (n:Person {id: 'original', name: 'Original'})")
        original_backup = backup_manager.backup(comment="original")

        # Modify database
        with manager.session() as session:
            session.run("CREATE (n:Person {id: 'modified', name: 'Modified'})")

        # Count backups before restore
        backups_before = len(backup_manager.list_backups())

        # Restore original backup
        backup_manager.restore(original_backup)

        # Should have created pre-restore backup
        backups_after = len(backup_manager.list_backups())
        assert backups_after > backups_before

        # Verify pre-restore backup exists
        backups = backup_manager.list_backups()
        pre_restore = [b for b in backups if "before-restore" in b[0].name]
        assert len(pre_restore) > 0

    def test_restore_invalid_backup_fails(self, backup_manager, tmp_path):
        """Restore fails with clear error for invalid backup."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not json")

        with pytest.raises(PKGRestoreError):
            backup_manager.restore(invalid_file)

    def test_restore_nonexistent_file_fails(self, backup_manager, tmp_path):
        """Restore fails with clear error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            backup_manager.restore(tmp_path / "nonexistent.json")


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestBackupListing:
    """Test backup listing functionality."""

    def test_list_backups_sorted(
        self, manager, backup_manager, clean_database
    ):
        """List returns backups sorted by date, newest first."""
        manager.initialize_schema()

        # Create multiple backups
        backup1 = backup_manager.backup(comment="first")
        backup2 = backup_manager.backup(comment="second")
        backup3 = backup_manager.backup(comment="third")

        backups = backup_manager.list_backups()

        assert len(backups) >= 3
        # Newest should be first
        assert backups[0][0] == backup3
        assert backups[1][0] == backup2
        assert backups[2][0] == backup1

    def test_list_backups_returns_metadata(
        self, manager, backup_manager, clean_database
    ):
        """List returns (path, datetime, size) tuples."""
        manager.initialize_schema()
        backup_manager.backup()

        backups = backup_manager.list_backups()

        assert len(backups) >= 1
        path, created, size = backups[0]
        assert isinstance(path, Path)
        assert isinstance(created, datetime)
        assert isinstance(size, int)
        assert size > 0


@requires_neo4j
@requires_testcontainers
@requires_docker
class TestBackupPurging:
    """Test backup purging functionality."""

    def test_purge_keeps_recent(
        self, manager, backup_manager, clean_database
    ):
        """Purge removes old backups but keeps recent ones."""
        manager.initialize_schema()

        # Create multiple backups
        for i in range(7):
            backup_manager.backup(comment=f"backup{i}")

        # Purge keeping only 3
        removed = backup_manager.purge_old_backups(keep_count=3)

        # Should have removed 4
        assert removed == 4

        # Should have 3 left
        backups = backup_manager.list_backups()
        assert len(backups) == 3

    def test_purge_respects_retention_days(
        self, manager, backup_manager, pkg_config, tmp_path, clean_database
    ):
        """Purge removes backups older than retention period."""
        manager.initialize_schema()

        # Create backups with config retention of 7 days
        backup_manager.backup(comment="current")

        # With only recent backups, none should be purged
        removed = backup_manager.purge_old_backups(keep_count=0, older_than_days=7)
        assert removed == 0

    def test_purge_with_high_keep_count(
        self, manager, backup_manager, clean_database
    ):
        """Purge with high keep_count doesn't remove anything."""
        manager.initialize_schema()

        for i in range(3):
            backup_manager.backup(comment=f"backup{i}")

        # Keep more than we have
        removed = backup_manager.purge_old_backups(keep_count=100)

        assert removed == 0
        assert len(backup_manager.list_backups()) == 3
