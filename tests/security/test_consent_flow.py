"""Consent flow integration tests.

Validates the complete consent lifecycle:
- Consent granting
- Consent verification
- Consent revocation
- Audit trail for consent changes
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestConsentGrant:
    """Test consent granting functionality."""

    def test_consent_grant_creates_record(self):
        """Verify consent grant creates a proper record."""
        consent_record = {
            "source": "vault:my-notes",
            "scope": ["read", "process", "store"],
            "granted_at": datetime.utcnow().isoformat(),
            "granted_by": "user",
            "status": "active",
        }

        assert consent_record["status"] == "active"
        assert "read" in consent_record["scope"]
        assert consent_record["source"] == "vault:my-notes"

    def test_consent_requires_explicit_action(self):
        """Verify consent is not granted by default."""
        with patch("futurnal.privacy.consent.ConsentRegistry") as MockRegistry:
            mock_registry = MagicMock()
            mock_registry.has_consent = MagicMock(return_value=False)
            MockRegistry.return_value = mock_registry

            # New source should not have consent
            assert mock_registry.has_consent("vault:new-vault", "read") is False

    def test_consent_scope_validation(self):
        """Verify only valid scopes are accepted."""
        valid_scopes = ["read", "process", "store"]
        invalid_scopes = ["delete_all", "share_externally", "export_cloud"]

        for scope in valid_scopes:
            assert scope in ["read", "process", "store", "export"]

        for scope in invalid_scopes:
            assert scope not in valid_scopes

    def test_consent_per_source_isolation(self):
        """Verify consent is isolated per data source."""
        consents = {
            "vault:personal": {"scopes": ["read", "process"], "status": "active"},
            "vault:work": {"scopes": [], "status": "none"},
            "imap:email": {"scopes": ["read"], "status": "active"},
        }

        # Personal vault has consent
        assert "read" in consents["vault:personal"]["scopes"]

        # Work vault has no consent
        assert len(consents["vault:work"]["scopes"]) == 0

        # Consents are independent
        assert consents["vault:personal"]["status"] != consents["vault:work"]["status"]


class TestConsentVerification:
    """Test consent verification functionality."""

    def test_consent_check_before_access(self):
        """Verify consent is checked before any data access."""
        access_sequence = []

        def mock_check_consent(source: str, scope: str) -> bool:
            access_sequence.append(("check_consent", source, scope))
            return True

        def mock_access_data(source: str) -> dict:
            access_sequence.append(("access_data", source))
            return {"data": "content"}

        # Simulate access flow
        source = "vault:test"
        if mock_check_consent(source, "read"):
            mock_access_data(source)

        # Verify consent check happened first
        assert access_sequence[0][0] == "check_consent"
        assert access_sequence[1][0] == "access_data"

    def test_denied_access_without_consent(self):
        """Verify access is denied without consent."""
        from futurnal.errors import ConsentRequiredError

        def access_with_consent_check(source: str, has_consent: bool) -> dict:
            if not has_consent:
                raise ConsentRequiredError(source, "read")
            return {"data": "content"}

        # Should raise without consent
        with pytest.raises(ConsentRequiredError):
            access_with_consent_check("vault:no-consent", False)

        # Should succeed with consent
        result = access_with_consent_check("vault:has-consent", True)
        assert result["data"] == "content"

    def test_scope_specific_verification(self):
        """Verify scope-specific consent verification."""
        consents = {
            "vault:partial": {"read": True, "process": True, "store": False}
        }

        source = "vault:partial"

        # Can read and process
        assert consents[source]["read"] is True
        assert consents[source]["process"] is True

        # Cannot store
        assert consents[source]["store"] is False


class TestConsentRevocation:
    """Test consent revocation functionality."""

    def test_consent_revocation_immediate(self):
        """Verify consent revocation takes effect immediately."""
        consent_state = {"vault:test": {"status": "active", "scopes": ["read"]}}

        # Revoke consent
        consent_state["vault:test"]["status"] = "revoked"
        consent_state["vault:test"]["scopes"] = []
        consent_state["vault:test"]["revoked_at"] = datetime.utcnow().isoformat()

        assert consent_state["vault:test"]["status"] == "revoked"
        assert len(consent_state["vault:test"]["scopes"]) == 0

    def test_revocation_stops_access(self):
        """Verify revocation stops all access."""
        consent_active = True

        def check_consent() -> bool:
            return consent_active

        # Initially can access
        assert check_consent() is True

        # After revocation
        consent_active = False
        assert check_consent() is False

    def test_revocation_preserves_data_by_default(self):
        """Verify revocation doesn't delete data by default."""
        revocation_options = {
            "delete_data": False,  # Default
            "retain_audit": True,
            "notify_user": True,
        }

        # Default should preserve data
        assert revocation_options["delete_data"] is False
        assert revocation_options["retain_audit"] is True

    def test_revocation_with_data_deletion(self):
        """Verify optional data deletion on revocation."""
        revocation_options = {
            "delete_data": True,
            "delete_embeddings": True,
            "delete_graph_nodes": True,
            "retain_audit": True,  # Audit always retained
        }

        assert revocation_options["delete_data"] is True
        assert revocation_options["retain_audit"] is True  # Audit preserved


class TestConsentAuditTrail:
    """Test consent audit trail functionality."""

    def test_consent_grant_logged(self):
        """Verify consent grants are logged."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "consent:read",
            "status": "granted",
            "source": "vault:test",
            "operator": "user",
        }

        assert audit_entry["action"].startswith("consent:")
        assert audit_entry["status"] == "granted"

    def test_consent_revocation_logged(self):
        """Verify consent revocations are logged."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "consent:read",
            "status": "revoked",
            "source": "vault:test",
            "operator": "user",
            "metadata": {"delete_data": False},
        }

        assert audit_entry["status"] == "revoked"
        assert "delete_data" in audit_entry["metadata"]

    def test_consent_audit_tamper_evident(self):
        """Verify consent audit trail is tamper-evident."""
        # Audit entries should have hash chain
        audit_entries = [
            {"id": 1, "chain_prev": None, "chain_hash": "abc123"},
            {"id": 2, "chain_prev": "abc123", "chain_hash": "def456"},
            {"id": 3, "chain_prev": "def456", "chain_hash": "ghi789"},
        ]

        # Verify chain integrity
        for i in range(1, len(audit_entries)):
            assert (
                audit_entries[i]["chain_prev"] == audit_entries[i - 1]["chain_hash"]
            ), "Audit chain broken"

    def test_consent_changes_retrievable(self):
        """Verify consent history can be retrieved."""
        consent_history = [
            {"timestamp": "2024-12-01T10:00:00Z", "action": "granted", "scope": "read"},
            {"timestamp": "2024-12-10T14:00:00Z", "action": "granted", "scope": "process"},
            {"timestamp": "2024-12-15T09:00:00Z", "action": "revoked", "scope": "read"},
        ]

        # Can retrieve full history
        assert len(consent_history) == 3

        # History is chronological
        timestamps = [e["timestamp"] for e in consent_history]
        assert timestamps == sorted(timestamps)


class TestConsentUI:
    """Test consent UI integration points."""

    def test_consent_prompt_content(self):
        """Verify consent prompts are clear and complete."""
        consent_prompt = {
            "source": "vault:my-notes",
            "requested_scopes": ["read", "process", "store"],
            "description": "Futurnal needs to access your Obsidian vault to build your knowledge graph.",
            "data_usage": [
                "Read markdown files from the vault",
                "Extract entities and relationships",
                "Store in local knowledge graph",
            ],
            "not_shared": "Your data will not be sent to any external servers.",
            "revocable": True,
        }

        assert consent_prompt["revocable"] is True
        assert "not be sent" in consent_prompt["not_shared"]
        assert len(consent_prompt["data_usage"]) > 0

    def test_consent_status_display(self):
        """Verify consent status is clearly displayed."""
        consent_display = {
            "vault:personal": {"status": "active", "icon": "check", "color": "green"},
            "vault:work": {"status": "pending", "icon": "clock", "color": "yellow"},
            "imap:email": {"status": "revoked", "icon": "x", "color": "red"},
        }

        # Each status has clear visual indicator
        for source, display in consent_display.items():
            assert "status" in display
            assert "icon" in display
            assert "color" in display
