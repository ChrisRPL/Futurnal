"""Tests for Cloud Sync Consent Scopes.

Tests validate the CloudSyncScope enum, consent status dataclass,
PKG metadata export format, and audit event structures.
"""

import pytest
from datetime import datetime
from typing import List

from src.futurnal.privacy.cloud_consent import (
    CloudSyncScope,
    CloudSyncConsentStatus,
    PKGMetadataExport,
    CloudSyncAuditEntry,
    CLOUD_SYNC_SCOPE_INFO,
    get_scope_info,
    get_default_enabled_scopes,
    get_required_scopes,
)


class TestCloudSyncScope:
    """Test CloudSyncScope enum."""

    def test_scope_values(self):
        """Test that scope enum values are correct."""
        assert CloudSyncScope.PKG_METADATA_BACKUP.value == "cloud:pkg:metadata_backup"
        assert CloudSyncScope.PKG_SETTINGS_BACKUP.value == "cloud:pkg:settings_backup"
        assert CloudSyncScope.SEARCH_HISTORY_SYNC.value == "cloud:search:history_sync"

    def test_scope_is_string_enum(self):
        """Test that scope values work as strings."""
        scope = CloudSyncScope.PKG_METADATA_BACKUP
        assert isinstance(scope.value, str)
        assert "cloud:" in scope.value
        assert scope == "cloud:pkg:metadata_backup"

    def test_all_scopes_have_info(self):
        """Test that all scopes have corresponding info entries."""
        for scope in CloudSyncScope:
            info = get_scope_info(scope)
            assert info is not None, f"Missing info for scope: {scope}"
            assert info.title, f"Missing title for scope: {scope}"
            assert info.description, f"Missing description for scope: {scope}"

    def test_scope_info_structure(self):
        """Test that scope info has all required fields."""
        for info in CLOUD_SYNC_SCOPE_INFO:
            assert hasattr(info, 'scope')
            assert hasattr(info, 'title')
            assert hasattr(info, 'description')
            assert hasattr(info, 'required')
            assert hasattr(info, 'default_enabled')
            assert isinstance(info.required, bool)
            assert isinstance(info.default_enabled, bool)


class TestCloudSyncConsentStatus:
    """Test CloudSyncConsentStatus dataclass."""

    def test_create_no_consent(self):
        """Test creating status with no consent."""
        status = CloudSyncConsentStatus(
            has_consent=False,
            granted_scopes=[],
        )
        assert status.has_consent is False
        assert status.granted_scopes == []
        assert status.granted_at is None
        assert status.is_syncing is False
        assert status.last_sync_at is None

    def test_create_with_consent(self):
        """Test creating status with consent."""
        now = datetime.utcnow()
        status = CloudSyncConsentStatus(
            has_consent=True,
            granted_scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            granted_at=now,
            is_syncing=False,
            last_sync_at=now,
        )
        assert status.has_consent is True
        assert CloudSyncScope.PKG_METADATA_BACKUP in status.granted_scopes
        assert status.granted_at == now
        assert status.is_syncing is False

    def test_multiple_scopes(self):
        """Test status with multiple scopes."""
        status = CloudSyncConsentStatus(
            has_consent=True,
            granted_scopes=[
                CloudSyncScope.PKG_METADATA_BACKUP,
                CloudSyncScope.PKG_SETTINGS_BACKUP,
            ],
        )
        assert len(status.granted_scopes) == 2

    def test_to_dict(self):
        """Test serialization to dict."""
        status = CloudSyncConsentStatus(
            has_consent=True,
            granted_scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
        )
        data = status.to_dict()
        assert data["has_consent"] is True
        assert "cloud:pkg:metadata_backup" in data["granted_scopes"]


class TestPKGMetadataExport:
    """Test PKGMetadataExport structure."""

    def test_create_export(self):
        """Test creating a metadata export."""
        now = datetime.utcnow()
        export = PKGMetadataExport(
            node_id="node_123",
            node_type="Person",
            label="John Doe",
            created_at=now,
            updated_at=now,
            source_type="obsidian",
        )
        assert export.node_id == "node_123"
        assert export.node_type == "Person"
        assert export.label == "John Doe"
        assert export.source_type == "obsidian"

    def test_export_excludes_content(self):
        """Test that export does not include content fields."""
        now = datetime.utcnow()
        export = PKGMetadataExport(
            node_id="node_123",
            node_type="Document",
            label="My Document",
            created_at=now,
            updated_at=now,
            source_type="obsidian",
        )
        # PKGMetadataExport should NOT have content field
        assert not hasattr(export, 'content')
        assert not hasattr(export, 'body')
        assert not hasattr(export, 'text')

    def test_export_to_dict(self):
        """Test serialization to dict."""
        now = datetime.utcnow()
        export = PKGMetadataExport(
            node_id="node_123",
            node_type="Person",
            label="John Doe",
            created_at=now,
            updated_at=now,
            source_type="obsidian",
        )
        data = export.to_dict()
        assert data["node_id"] == "node_123"
        assert data["node_type"] == "Person"
        assert data["label"] == "John Doe"


class TestCloudSyncAuditEntry:
    """Test CloudSyncAuditEntry structure."""

    def test_create_audit_entry(self):
        """Test creating an audit entry."""
        entry = CloudSyncAuditEntry(
            action="sync_started",
            success=True,
            nodes_affected=0,
        )
        assert entry.action == "sync_started"
        assert entry.success is True
        assert entry.nodes_affected == 0
        assert entry.timestamp is not None

    def test_audit_entry_with_error(self):
        """Test audit entry with error message."""
        entry = CloudSyncAuditEntry(
            action="sync_failed",
            success=False,
            nodes_affected=0,
            error_message="Network timeout",
        )
        assert entry.success is False
        assert entry.error_message == "Network timeout"

    def test_audit_entry_with_scope(self):
        """Test audit entry with scope."""
        entry = CloudSyncAuditEntry(
            action="consent_granted",
            success=True,
            scope=CloudSyncScope.PKG_METADATA_BACKUP,
        )
        assert entry.scope == CloudSyncScope.PKG_METADATA_BACKUP


class TestScopeHelpers:
    """Test scope helper functions."""

    def test_get_default_enabled_scopes(self):
        """Test getting default enabled scopes."""
        defaults = get_default_enabled_scopes()
        assert isinstance(defaults, list)
        # Metadata backup should be default
        assert CloudSyncScope.PKG_METADATA_BACKUP in defaults
        # Settings backup should be default
        assert CloudSyncScope.PKG_SETTINGS_BACKUP in defaults
        # Search history should NOT be default
        assert CloudSyncScope.SEARCH_HISTORY_SYNC not in defaults

    def test_get_required_scopes(self):
        """Test getting required scopes."""
        required = get_required_scopes()
        assert isinstance(required, list)
        # Metadata backup should be required
        assert CloudSyncScope.PKG_METADATA_BACKUP in required
        # Settings and search should not be required
        assert CloudSyncScope.PKG_SETTINGS_BACKUP not in required
        assert CloudSyncScope.SEARCH_HISTORY_SYNC not in required

    def test_get_scope_info_valid(self):
        """Test getting info for valid scope."""
        info = get_scope_info(CloudSyncScope.PKG_METADATA_BACKUP)
        assert info is not None
        assert info.scope == CloudSyncScope.PKG_METADATA_BACKUP
        assert info.required is True
        assert info.default_enabled is True

    def test_get_scope_info_invalid(self):
        """Test getting info for invalid scope."""
        info = get_scope_info("invalid:scope")
        assert info is None


class TestScopeInfoContent:
    """Test scope info content for UI display."""

    def test_metadata_backup_info(self):
        """Test metadata backup scope info."""
        info = get_scope_info(CloudSyncScope.PKG_METADATA_BACKUP)
        assert info.title == "Knowledge Graph Structure"
        assert "metadata" in info.description.lower() or "graph" in info.description.lower()
        assert info.required is True
        # Should have data_shared list
        if info.data_shared:
            assert len(info.data_shared) > 0
        # Should have data_not_shared list
        if info.data_not_shared:
            assert len(info.data_not_shared) > 0

    def test_settings_backup_info(self):
        """Test settings backup scope info."""
        info = get_scope_info(CloudSyncScope.PKG_SETTINGS_BACKUP)
        assert info.title == "App Settings"
        assert info.required is False
        assert info.default_enabled is True

    def test_search_history_info(self):
        """Test search history sync scope info."""
        info = get_scope_info(CloudSyncScope.SEARCH_HISTORY_SYNC)
        assert info.title == "Search History"
        assert info.required is False
        assert info.default_enabled is False
