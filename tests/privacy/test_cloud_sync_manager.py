"""Tests for CloudSyncConsentManager.

Tests validate consent management for cloud sync operations,
audit logging integration, and revocation behavior.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from futurnal.privacy.cloud_consent import (
    CloudSyncScope,
    CloudSyncConsentStatus,
)
from futurnal.privacy.cloud_sync_manager import (
    CloudSyncConsentManager,
)
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.audit import AuditLogger


class TestCloudSyncConsentManagerBasics:
    """Test basic CloudSyncConsentManager operations."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create workspace with consent manager."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)

        manager = CloudSyncConsentManager(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

        return {
            "manager": manager,
            "registry": registry,
            "audit_logger": audit_logger,
            "tmp_path": tmp_path,
        }

    def test_initial_status_no_consent(self, workspace):
        """Test that initial status shows no consent."""
        manager = workspace["manager"]
        status = manager.get_status()

        assert status.has_consent is False
        assert status.granted_scopes == []
        assert status.granted_at is None

    def test_grant_consent_single_scope(self, workspace):
        """Test granting consent for a single scope."""
        manager = workspace["manager"]

        status = manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        assert status.has_consent is True
        assert CloudSyncScope.PKG_METADATA_BACKUP in status.granted_scopes
        assert status.granted_at is not None

    def test_grant_consent_multiple_scopes(self, workspace):
        """Test granting consent for multiple scopes."""
        manager = workspace["manager"]

        status = manager.grant_consent(
            scopes=[
                CloudSyncScope.PKG_METADATA_BACKUP,
                CloudSyncScope.PKG_SETTINGS_BACKUP,
            ],
            operator="test_user@example.com",
        )

        assert status.has_consent is True
        assert len(status.granted_scopes) == 2
        assert CloudSyncScope.PKG_METADATA_BACKUP in status.granted_scopes
        assert CloudSyncScope.PKG_SETTINGS_BACKUP in status.granted_scopes

    def test_revoke_consent(self, workspace):
        """Test revoking consent."""
        manager = workspace["manager"]

        # Grant first
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Verify granted
        status = manager.get_status()
        assert status.has_consent is True

        # Revoke
        manager.revoke_consent(operator="test_user@example.com")

        # Verify revoked
        status = manager.get_status()
        assert status.has_consent is False
        assert status.granted_scopes == []

    def test_grant_auto_adds_required_scope(self, workspace):
        """Test that granting without required scope auto-adds it."""
        manager = workspace["manager"]

        # Grant with only settings scope - required scope should be auto-added
        status = manager.grant_consent(
            scopes=[CloudSyncScope.PKG_SETTINGS_BACKUP],
            operator="test_user@example.com",
        )

        # Both scopes should be granted (required was auto-added)
        assert status.has_consent is True
        assert CloudSyncScope.PKG_METADATA_BACKUP.value in status.granted_scopes
        assert CloudSyncScope.PKG_SETTINGS_BACKUP.value in status.granted_scopes


class TestCloudSyncConsentEnforcement:
    """Test consent enforcement in sync operations."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create workspace with consent manager."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)

        manager = CloudSyncConsentManager(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

        return {
            "manager": manager,
            "registry": registry,
            "audit_logger": audit_logger,
        }

    def test_require_consent_without_consent(self, workspace):
        """Test that require_consent raises without consent."""
        manager = workspace["manager"]

        with pytest.raises(ConsentRequiredError):
            manager.require_consent(CloudSyncScope.PKG_METADATA_BACKUP)

    def test_require_consent_with_consent(self, workspace):
        """Test that require_consent passes with consent."""
        manager = workspace["manager"]

        # Grant consent
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Should not raise
        manager.require_consent(CloudSyncScope.PKG_METADATA_BACKUP)

    def test_require_consent_wrong_scope(self, workspace):
        """Test that require_consent fails for non-granted scope."""
        manager = workspace["manager"]

        # Grant only metadata scope
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Should fail for settings scope
        with pytest.raises(ConsentRequiredError):
            manager.require_consent(CloudSyncScope.PKG_SETTINGS_BACKUP)

    def test_has_scope_helper(self, workspace):
        """Test the has_scope helper method."""
        manager = workspace["manager"]

        # Initially no scopes
        assert manager.has_scope(CloudSyncScope.PKG_METADATA_BACKUP) is False

        # Grant metadata scope
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Now has metadata scope but not settings
        assert manager.has_scope(CloudSyncScope.PKG_METADATA_BACKUP) is True
        assert manager.has_scope(CloudSyncScope.PKG_SETTINGS_BACKUP) is False


class TestCloudSyncAuditLogging:
    """Test audit logging integration."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create workspace with consent manager."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)

        manager = CloudSyncConsentManager(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

        return {
            "manager": manager,
            "audit_logger": audit_logger,
        }

    def test_grant_consent_logged(self, workspace):
        """Test that granting consent is logged."""
        manager = workspace["manager"]
        audit_logger = workspace["audit_logger"]

        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        events = list(audit_logger.iter_events())
        consent_events = [
            e for e in events
            if "consent" in e.get("action", "").lower() and "grant" in e.get("action", "").lower()
        ]
        assert len(consent_events) >= 1

    def test_revoke_consent_logged(self, workspace):
        """Test that revoking consent is logged."""
        manager = workspace["manager"]
        audit_logger = workspace["audit_logger"]

        # Grant first
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Revoke
        manager.revoke_consent(operator="test_user@example.com")

        events = list(audit_logger.iter_events())
        revoke_events = [
            e for e in events
            if "consent" in e.get("action", "").lower() and "revoke" in e.get("action", "").lower()
        ]
        assert len(revoke_events) >= 1

    def test_sync_started_logged(self, workspace):
        """Test that sync start is logged."""
        manager = workspace["manager"]
        audit_logger = workspace["audit_logger"]

        # Grant consent first
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Log sync started
        manager.log_sync_started()

        events = list(audit_logger.iter_events())
        sync_events = [
            e for e in events
            if "sync" in e.get("action", "").lower() and "start" in e.get("action", "").lower()
        ]
        assert len(sync_events) >= 1

    def test_sync_completed_logged(self, workspace):
        """Test that sync completion is logged."""
        manager = workspace["manager"]
        audit_logger = workspace["audit_logger"]

        # Grant consent first
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Log sync completed
        manager.log_sync_completed(nodes_synced=42, duration_ms=1500)

        events = list(audit_logger.iter_events())
        sync_events = [
            e for e in events
            if "sync" in e.get("action", "").lower() and "complet" in e.get("action", "").lower()
        ]
        assert len(sync_events) >= 1

    def test_sync_failed_logged(self, workspace):
        """Test that sync failure is logged."""
        manager = workspace["manager"]
        audit_logger = workspace["audit_logger"]

        # Grant consent first
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Log sync failed
        manager.log_sync_failed(error="Network timeout")

        events = list(audit_logger.iter_events())
        fail_events = [
            e for e in events
            if "sync" in e.get("action", "").lower() and "fail" in e.get("action", "").lower()
        ]
        assert len(fail_events) >= 1


class TestCloudSyncRevocationBehavior:
    """Test consent revocation and data deletion behavior."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create workspace with consent manager."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)

        manager = CloudSyncConsentManager(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

        return {
            "manager": manager,
            "registry": registry,
            "audit_logger": audit_logger,
        }

    def test_revoke_clears_all_scopes(self, workspace):
        """Test that revocation clears all granted scopes."""
        manager = workspace["manager"]

        # Grant multiple scopes
        manager.grant_consent(
            scopes=[
                CloudSyncScope.PKG_METADATA_BACKUP,
                CloudSyncScope.PKG_SETTINGS_BACKUP,
                CloudSyncScope.SEARCH_HISTORY_SYNC,
            ],
            operator="test_user@example.com",
        )

        # Verify all granted
        status = manager.get_status()
        assert len(status.granted_scopes) == 3

        # Revoke
        manager.revoke_consent(operator="test_user@example.com")

        # Verify all cleared
        status = manager.get_status()
        assert status.has_consent is False
        assert status.granted_scopes == []

    def test_revoke_triggers_deletion_request(self, workspace):
        """Test that revocation logs data deletion request."""
        manager = workspace["manager"]
        audit_logger = workspace["audit_logger"]

        # Grant consent
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Revoke
        manager.revoke_consent(operator="test_user@example.com")

        events = list(audit_logger.iter_events())
        deletion_events = [
            e for e in events
            if "delet" in e.get("action", "").lower()
        ]
        assert len(deletion_events) >= 1

    def test_cannot_sync_after_revoke(self, workspace):
        """Test that sync operations fail after revocation."""
        manager = workspace["manager"]

        # Grant consent
        manager.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Revoke
        manager.revoke_consent(operator="test_user@example.com")

        # Should fail to require consent
        with pytest.raises(ConsentRequiredError):
            manager.require_consent(CloudSyncScope.PKG_METADATA_BACKUP)


class TestCloudSyncConsentPersistence:
    """Test consent persistence across manager instances."""

    def test_consent_persists_across_instances(self, tmp_path):
        """Test that consent survives manager restart."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        # First instance - grant consent
        registry1 = ConsentRegistry(consent_dir)
        audit_logger1 = AuditLogger(output_dir=audit_dir)
        manager1 = CloudSyncConsentManager(
            consent_registry=registry1,
            audit_logger=audit_logger1,
        )

        manager1.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )

        # Second instance - should still have consent
        registry2 = ConsentRegistry(consent_dir)
        audit_logger2 = AuditLogger(output_dir=audit_dir)
        manager2 = CloudSyncConsentManager(
            consent_registry=registry2,
            audit_logger=audit_logger2,
        )

        status = manager2.get_status()
        assert status.has_consent is True
        assert CloudSyncScope.PKG_METADATA_BACKUP in status.granted_scopes

    def test_revocation_persists_across_instances(self, tmp_path):
        """Test that revocation survives manager restart."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        # First instance - grant then revoke
        registry1 = ConsentRegistry(consent_dir)
        audit_logger1 = AuditLogger(output_dir=audit_dir)
        manager1 = CloudSyncConsentManager(
            consent_registry=registry1,
            audit_logger=audit_logger1,
        )

        manager1.grant_consent(
            scopes=[CloudSyncScope.PKG_METADATA_BACKUP],
            operator="test_user@example.com",
        )
        manager1.revoke_consent(operator="test_user@example.com")

        # Second instance - should have no consent
        registry2 = ConsentRegistry(consent_dir)
        audit_logger2 = AuditLogger(output_dir=audit_dir)
        manager2 = CloudSyncConsentManager(
            consent_registry=registry2,
            audit_logger=audit_logger2,
        )

        status = manager2.get_status()
        assert status.has_consent is False
        assert status.granted_scopes == []
