"""Comprehensive privacy integration tests for Obsidian connector.

Tests the complete privacy framework including vault privacy settings,
consent management, audit logging, and redaction policies.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

import pytest

from futurnal.ingestion.obsidian import (
    ObsidianVaultDescriptor,
    VaultRegistry,
    VaultPrivacySettings,
    PrivacyLevel,
    ConsentScope,
    ObsidianPrivacyPolicy,
    VaultConsentManager,
    ObsidianVaultConnector,
    ObsidianVaultSource,
)
from futurnal.privacy.audit import AuditLogger, AuditEvent
from futurnal.privacy.consent import ConsentRegistry
from futurnal.privacy.redaction import RedactionPolicy


class TestVaultPrivacySettings:
    """Test vault privacy settings and policy creation."""

    def test_default_privacy_settings(self):
        """Test default privacy settings are secure by default."""
        settings = VaultPrivacySettings()

        assert settings.privacy_level == PrivacyLevel.STANDARD
        assert ConsentScope.VAULT_SCAN in settings.required_consent_scopes
        assert settings.enable_content_redaction is True
        assert settings.enable_path_anonymization is True
        assert settings.audit_content_changes is True
        assert settings.audit_link_changes is True
        assert settings.retain_audit_days == 90

    def test_strict_privacy_settings(self):
        """Test strict privacy level configuration."""
        settings = VaultPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
            required_consent_scopes=[
                ConsentScope.VAULT_SCAN,
                ConsentScope.CONTENT_ANALYSIS,
                ConsentScope.ASSET_EXTRACTION,
                ConsentScope.CLOUD_MODELS,
            ],
            privacy_tags=["secret", "personal", "confidential"],
            tag_based_privacy_classification=True,
        )

        assert settings.privacy_level == PrivacyLevel.STRICT
        assert len(settings.required_consent_scopes) == 4
        assert settings.tag_based_privacy_classification is True
        assert "secret" in settings.privacy_tags

    def test_permissive_privacy_settings(self):
        """Test permissive privacy level configuration."""
        settings = VaultPrivacySettings(
            privacy_level=PrivacyLevel.PERMISSIVE,
            enable_path_anonymization=False,
            required_consent_scopes=[ConsentScope.VAULT_SCAN],
        )

        assert settings.privacy_level == PrivacyLevel.PERMISSIVE
        assert settings.enable_path_anonymization is False
        assert len(settings.required_consent_scopes) == 1


class TestObsidianVaultDescriptor:
    """Test vault descriptor with privacy settings."""

    def setup_method(self):
        """Set up test vault directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault_path = Path(self.temp_dir.name)

        # Create .obsidian directory to make it a valid vault
        obsidian_dir = self.vault_path / ".obsidian"
        obsidian_dir.mkdir()

        # Create some test notes
        (self.vault_path / "public_note.md").write_text("# Public Note\nPublic content")
        (self.vault_path / "private_note.md").write_text("# Private Note\nSensitive content")
        (self.vault_path / "secret_project.md").write_text("# Secret Project\nConfidential content")

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_vault_descriptor_with_privacy_settings(self):
        """Test creating vault descriptor with custom privacy settings."""
        privacy_settings = VaultPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
            required_consent_scopes=[ConsentScope.VAULT_SCAN, ConsentScope.CONTENT_ANALYSIS],
            privacy_tags=["private", "secret"],
            tag_based_privacy_classification=True,
        )

        descriptor = ObsidianVaultDescriptor.from_path(
            self.vault_path,
            name="Test Vault",
            redact_title_patterns=["secret.*", "private.*"],
            privacy_settings=privacy_settings,
        )

        assert descriptor.privacy_settings.privacy_level == PrivacyLevel.STRICT
        assert len(descriptor.privacy_settings.required_consent_scopes) == 2
        assert descriptor.privacy_settings.tag_based_privacy_classification is True
        assert "secret.*" in descriptor.redact_title_patterns

    def test_redaction_policy_creation(self):
        """Test creating redaction policy from vault settings."""
        privacy_settings = VaultPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
            privacy_tags=["private", "secret"],
            tag_based_privacy_classification=True,
        )

        descriptor = ObsidianVaultDescriptor.from_path(
            self.vault_path,
            redact_title_patterns=["secret.*"],
            privacy_settings=privacy_settings,
        )

        policy = descriptor.build_redaction_policy()

        # Test redaction of sensitive file
        secret_path = self.vault_path / "secret_project.md"
        redacted = policy.apply(secret_path)

        # Should be heavily redacted due to strict privacy level
        assert redacted.redacted != str(secret_path)
        assert len(redacted.path_hash) > 0

    def test_consent_scope_helpers(self):
        """Test consent scope helper methods."""
        privacy_settings = VaultPrivacySettings(
            required_consent_scopes=[ConsentScope.VAULT_SCAN, ConsentScope.CONTENT_ANALYSIS]
        )

        descriptor = ObsidianVaultDescriptor.from_path(
            self.vault_path,
            privacy_settings=privacy_settings,
        )

        assert descriptor.requires_consent_for_scope(ConsentScope.VAULT_SCAN)
        assert descriptor.requires_consent_for_scope(ConsentScope.CONTENT_ANALYSIS)
        assert not descriptor.requires_consent_for_scope(ConsentScope.CLOUD_MODELS)

        required_scopes = descriptor.get_required_consent_scopes()
        assert len(required_scopes) == 2
        assert ConsentScope.VAULT_SCAN.value in required_scopes
        assert ConsentScope.CONTENT_ANALYSIS.value in required_scopes


class TestObsidianPrivacyPolicy:
    """Test the comprehensive privacy policy system."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault_path = Path(self.temp_dir.name)

        # Create .obsidian directory
        obsidian_dir = self.vault_path / ".obsidian"
        obsidian_dir.mkdir()

        # Create consent registry
        self.consent_dir = tempfile.TemporaryDirectory()
        self.consent_registry = ConsentRegistry(Path(self.consent_dir.name))

    def teardown_method(self):
        """Clean up test directories."""
        self.temp_dir.cleanup()
        self.consent_dir.cleanup()

    def test_privacy_policy_creation(self):
        """Test creating privacy policy from vault descriptor."""
        privacy_settings = VaultPrivacySettings(
            privacy_level=PrivacyLevel.STANDARD,
            required_consent_scopes=[ConsentScope.VAULT_SCAN, ConsentScope.CONTENT_ANALYSIS],
        )

        descriptor = ObsidianVaultDescriptor.from_path(
            self.vault_path,
            name="Test Vault",
            privacy_settings=privacy_settings,
        )

        policy = ObsidianPrivacyPolicy.from_vault_descriptor(
            descriptor,
            consent_registry=self.consent_registry,
        )

        assert policy.vault_id == descriptor.id
        assert policy.privacy_settings.privacy_level == PrivacyLevel.STANDARD
        assert policy.consent_registry == self.consent_registry

    def test_consent_checking(self):
        """Test consent checking functionality."""
        privacy_settings = VaultPrivacySettings(
            required_consent_scopes=[ConsentScope.VAULT_SCAN, ConsentScope.CONTENT_ANALYSIS],
        )

        descriptor = ObsidianVaultDescriptor.from_path(
            self.vault_path,
            privacy_settings=privacy_settings,
        )

        policy = ObsidianPrivacyPolicy.from_vault_descriptor(
            descriptor,
            consent_registry=self.consent_registry,
        )

        # No consent granted initially
        assert not policy.check_consent(ConsentScope.VAULT_SCAN)
        assert not policy.check_consent(ConsentScope.CONTENT_ANALYSIS)

        # Grant consent for vault scan
        policy.grant_consent(ConsentScope.VAULT_SCAN, operator="test_user")
        assert policy.check_consent(ConsentScope.VAULT_SCAN)
        assert not policy.check_consent(ConsentScope.CONTENT_ANALYSIS)

    def test_vault_consent_manager(self):
        """Test the simplified consent manager interface."""
        privacy_settings = VaultPrivacySettings(
            required_consent_scopes=[
                ConsentScope.VAULT_SCAN,
                ConsentScope.CONTENT_ANALYSIS,
                ConsentScope.ASSET_EXTRACTION,
            ],
        )

        descriptor = ObsidianVaultDescriptor.from_path(
            self.vault_path,
            privacy_settings=privacy_settings,
        )

        policy = ObsidianPrivacyPolicy.from_vault_descriptor(
            descriptor,
            consent_registry=self.consent_registry,
        )

        consent_manager = VaultConsentManager(policy)

        # Grant all required consent
        records = consent_manager.grant_all_required_consent(
            operator="test_user",
            duration_hours=24,
        )

        assert len(records) == 3
        assert consent_manager.require_consent_for_scan()
        assert consent_manager.require_consent_for_content_analysis()
        assert consent_manager.require_consent_for_asset_extraction()

        # Check missing consent
        missing = consent_manager.get_missing_consent()
        assert len(missing) == 0


class TestVaultRegistryWithAudit:
    """Test vault registry with audit logging."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault_path = Path(self.temp_dir.name) / "vault"
        self.vault_path.mkdir()

        # Create .obsidian directory
        obsidian_dir = self.vault_path / ".obsidian"
        obsidian_dir.mkdir()

        # Set up audit logging
        self.audit_dir = tempfile.TemporaryDirectory()
        self.audit_logger = AuditLogger(
            output_dir=Path(self.audit_dir.name),
            filename="vault_audit.log"
        )

        # Set up registry with audit logging
        self.registry_dir = tempfile.TemporaryDirectory()
        self.vault_registry = VaultRegistry(
            registry_root=Path(self.registry_dir.name),
            audit_logger=self.audit_logger,
        )

    def teardown_method(self):
        """Clean up test directories."""
        self.temp_dir.cleanup()
        self.audit_dir.cleanup()
        self.registry_dir.cleanup()

    def test_vault_registration_audit(self):
        """Test that vault registration creates audit events."""
        descriptor = self.vault_registry.register_path(
            self.vault_path,
            name="Test Vault",
            operator="test_user",
        )

        # Check that audit log was created
        audit_file = Path(self.audit_dir.name) / "vault_audit.log"
        assert audit_file.exists()

        # Read audit log
        audit_lines = audit_file.read_text().strip().split('\n')
        assert len(audit_lines) >= 1

        # Parse audit event
        audit_event = json.loads(audit_lines[0])
        assert audit_event["action"] == "vault_registered"
        assert audit_event["status"] == "success"
        assert audit_event["operator_action"] == "test_user"
        assert audit_event["metadata"]["vault_id"] == descriptor.id

    def test_vault_update_audit(self):
        """Test that vault updates create audit events."""
        # Register initial vault
        descriptor = self.vault_registry.register_path(
            self.vault_path,
            name="Test Vault",
            operator="test_user",
        )

        # Update with new privacy settings
        new_privacy_settings = VaultPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
            required_consent_scopes=[ConsentScope.VAULT_SCAN, ConsentScope.CLOUD_MODELS],
        )

        updated_descriptor = descriptor.model_copy(update={
            "privacy_settings": new_privacy_settings,
            "name": "Updated Test Vault",
        })

        self.vault_registry.add_or_update(updated_descriptor, operator="test_user")

        # Check audit log
        audit_file = Path(self.audit_dir.name) / "vault_audit.log"
        audit_lines = audit_file.read_text().strip().split('\n')

        # Should have registration and update events
        assert len(audit_lines) >= 2

        # Check update event
        update_event = json.loads(audit_lines[1])
        assert update_event["action"] == "vault_updated"
        assert update_event["status"] == "success"

    def test_vault_privacy_change_audit(self):
        """Test that privacy setting changes are audited."""
        # Register vault with standard privacy
        descriptor = self.vault_registry.register_path(
            self.vault_path,
            name="Test Vault",
            privacy_settings=VaultPrivacySettings(privacy_level=PrivacyLevel.STANDARD),
            operator="test_user",
        )

        # Update to strict privacy
        updated_descriptor = descriptor.model_copy(update={
            "privacy_settings": VaultPrivacySettings(
                privacy_level=PrivacyLevel.STRICT,
                required_consent_scopes=[ConsentScope.VAULT_SCAN, ConsentScope.CLOUD_MODELS],
            ),
        })

        self.vault_registry.add_or_update(updated_descriptor, operator="test_user")

        # Check for privacy change audit
        audit_file = Path(self.audit_dir.name) / "vault_audit.log"
        audit_lines = audit_file.read_text().strip().split('\n')

        # Look for privacy update event
        privacy_event = None
        for line in audit_lines:
            event = json.loads(line)
            if event.get("action") == "vault_privacy_updated":
                privacy_event = event
                break

        assert privacy_event is not None
        assert "privacy_changes" in privacy_event["metadata"]
        changes = privacy_event["metadata"]["privacy_changes"]
        assert "privacy_level" in changes
        assert changes["privacy_level"]["from"] == "standard"
        assert changes["privacy_level"]["to"] == "strict"

    def test_vault_removal_audit(self):
        """Test that vault removal creates audit events."""
        # Register vault
        descriptor = self.vault_registry.register_path(
            self.vault_path,
            name="Test Vault",
            operator="test_user",
        )

        # Remove vault
        self.vault_registry.remove(descriptor.id, operator="test_user")

        # Check audit log
        audit_file = Path(self.audit_dir.name) / "vault_audit.log"
        audit_lines = audit_file.read_text().strip().split('\n')

        # Should have registration and removal events
        assert len(audit_lines) >= 2

        # Check removal event
        removal_event = json.loads(audit_lines[-1])
        assert removal_event["action"] == "vault_removed"
        assert removal_event["status"] == "success"
        assert removal_event["operator_action"] == "test_user"


class TestConnectorPrivacyIntegration:
    """Test privacy integration in the vault connector."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault_path = Path(self.temp_dir.name) / "vault"
        self.vault_path.mkdir()

        # Create .obsidian directory
        obsidian_dir = self.vault_path / ".obsidian"
        obsidian_dir.mkdir()

        # Create test notes with different privacy levels
        (self.vault_path / "public_note.md").write_text("""# Public Note
This is public content.

[[private_note]]
""")

        (self.vault_path / "private_note.md").write_text("""# Private Note
This contains #private information.
""")

        (self.vault_path / "secret_project.md").write_text("""# Secret Project
This is #confidential data.

![[diagram.png]]
""")

        # Set up workspace
        self.workspace_dir = tempfile.TemporaryDirectory()

        # Set up audit logging
        self.audit_dir = tempfile.TemporaryDirectory()
        self.audit_logger = AuditLogger(Path(self.audit_dir.name))

        # Set up consent registry
        self.consent_dir = tempfile.TemporaryDirectory()
        self.consent_registry = ConsentRegistry(Path(self.consent_dir.name))

        # Set up vault registry
        self.registry_dir = tempfile.TemporaryDirectory()
        self.vault_registry = VaultRegistry(
            Path(self.registry_dir.name),
            audit_logger=self.audit_logger,
        )

    def teardown_method(self):
        """Clean up test directories."""
        self.temp_dir.cleanup()
        self.workspace_dir.cleanup()
        self.audit_dir.cleanup()
        self.consent_dir.cleanup()
        self.registry_dir.cleanup()

    def test_connector_privacy_policy_integration(self):
        """Test that connector uses vault-specific privacy policies."""
        # Register vault with strict privacy
        privacy_settings = VaultPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
            required_consent_scopes=[ConsentScope.VAULT_SCAN, ConsentScope.CONTENT_ANALYSIS],
            privacy_tags=["private", "confidential"],
            tag_based_privacy_classification=True,
        )

        descriptor = self.vault_registry.register_path(
            self.vault_path,
            name="Test Vault",
            redact_title_patterns=["secret.*"],
            privacy_settings=privacy_settings,
        )

        # Create connector
        connector = ObsidianVaultConnector(
            workspace_dir=self.workspace_dir.name,
            state_store=Mock(),  # Mock for testing
            vault_registry=self.vault_registry,
            audit_logger=self.audit_logger,
            consent_registry=self.consent_registry,
        )

        # Create vault source
        vault_source = ObsidianVaultSource.from_vault_descriptor(descriptor)

        # Get privacy policy
        privacy_policy = connector._get_privacy_policy(vault_source)
        assert privacy_policy is not None
        assert privacy_policy.privacy_settings.privacy_level == PrivacyLevel.STRICT

        # Test redaction policy
        redaction_policy = privacy_policy.build_redaction_policy()

        secret_file = self.vault_path / "secret_project.md"
        redacted = redaction_policy.apply(secret_file)

        # Should be heavily redacted due to strict privacy and pattern matching
        assert redacted.redacted != str(secret_file)
        assert "secret" not in redacted.redacted.lower()

    def test_consent_checking_in_connector(self):
        """Test that connector properly checks consent before processing."""
        # Register vault requiring consent
        privacy_settings = VaultPrivacySettings(
            required_consent_scopes=[ConsentScope.VAULT_SCAN, ConsentScope.CONTENT_ANALYSIS],
        )

        descriptor = self.vault_registry.register_path(
            self.vault_path,
            privacy_settings=privacy_settings,
        )

        # Create connector
        connector = ObsidianVaultConnector(
            workspace_dir=self.workspace_dir.name,
            state_store=Mock(),
            vault_registry=self.vault_registry,
            consent_registry=self.consent_registry,
        )

        vault_source = ObsidianVaultSource.from_vault_descriptor(descriptor)

        # Get consent manager
        consent_manager = connector._get_consent_manager(vault_source)
        assert consent_manager is not None

        # Should require consent
        assert not consent_manager.require_consent_for_scan()
        assert not consent_manager.require_consent_for_content_analysis()

        # Grant consent
        consent_manager.grant_all_required_consent(operator="test_user")

        # Should now have consent
        assert consent_manager.require_consent_for_scan()
        assert consent_manager.require_consent_for_content_analysis()

    @patch('futurnal.ingestion.obsidian.connector.ObsidianDocumentProcessor')
    def test_link_graph_audit_logging(self, mock_processor):
        """Test that link graph events are properly audited."""
        # Setup mock processor to return link data
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance

        mock_processor_instance.process_document.return_value = [
            {
                'text': 'Test content',
                'metadata': {
                    'obsidian_links': [
                        {'target': 'private_note', 'type': 'wikilink'},
                        {'target': 'secret_project', 'type': 'wikilink'},
                    ],
                    'obsidian_tags': ['public', 'test'],
                }
            }
        ]

        # Register vault
        descriptor = self.vault_registry.register_path(self.vault_path)

        # Create connector
        connector = ObsidianVaultConnector(
            workspace_dir=self.workspace_dir.name,
            state_store=Mock(),
            vault_registry=self.vault_registry,
            audit_logger=self.audit_logger,
            consent_registry=self.consent_registry,
        )

        vault_source = ObsidianVaultSource.from_vault_descriptor(descriptor)

        # Grant consent for processing
        consent_manager = connector._get_consent_manager(vault_source)
        consent_manager.grant_all_required_consent(operator="test_user")

        # Process a document (mock file record)
        from futurnal.ingestion.local.state import FileRecord
        mock_record = FileRecord(
            path=self.vault_path / "public_note.md",
            size=100,
            mtime=1000000000,
            sha256="abc123",
        )

        # Mock state store to return our record
        mock_state_store = Mock()
        mock_state_store.iter_all.return_value = [mock_record]
        connector._state_store = mock_state_store

        # Run ingestion
        list(connector.ingest(vault_source, job_id="test_job"))

        # Check audit log for link graph events
        audit_file = Path(self.audit_dir.name) / "audit.log"
        if audit_file.exists():
            audit_content = audit_file.read_text()

            # Should contain link graph events
            assert "link_graph_" in audit_content or "document_processed" in audit_content


class TestPrivacyComplianceValidation:
    """Test compliance with privacy requirements."""

    def test_no_content_in_audit_logs(self):
        """Verify that no actual content appears in audit logs."""
        temp_dir = tempfile.TemporaryDirectory()
        audit_dir = tempfile.TemporaryDirectory()

        try:
            vault_path = Path(temp_dir.name) / "vault"
            vault_path.mkdir()
            (vault_path / ".obsidian").mkdir()

            # Create file with sensitive content
            sensitive_file = vault_path / "sensitive_data.md"
            sensitive_content = "This is highly confidential information that should never appear in logs."
            sensitive_file.write_text(sensitive_content)

            # Set up audit logger
            audit_logger = AuditLogger(Path(audit_dir.name))

            # Create a file event with sensitive content
            from futurnal.ingestion.local.state import FileRecord
            record = FileRecord(
                path=sensitive_file,
                size=len(sensitive_content),
                mtime=1000000000,
                sha256="test_hash",
            )

            # Log the event
            audit_logger.record_file_event(
                job_id="test_job",
                source="test_source",
                action="process",
                status="success",
                path=sensitive_file,
                sha256="test_hash",
            )

            # Check that audit log doesn't contain sensitive content
            audit_file = Path(audit_dir.name) / "audit.log"
            if audit_file.exists():
                audit_content = audit_file.read_text()
                assert "confidential information" not in audit_content
                assert "sensitive_data" not in audit_content  # Filename should be redacted

        finally:
            temp_dir.cleanup()
            audit_dir.cleanup()

    def test_consistent_redaction_across_events(self):
        """Verify that redaction is applied consistently across all audit events."""
        temp_dir = tempfile.TemporaryDirectory()
        audit_dir = tempfile.TemporaryDirectory()

        try:
            vault_path = Path(temp_dir.name) / "vault"
            vault_path.mkdir()
            (vault_path / ".obsidian").mkdir()

            # Create vault with strict privacy
            privacy_settings = VaultPrivacySettings(
                privacy_level=PrivacyLevel.STRICT,
                redact_title_patterns=["secret.*"],
            )

            descriptor = ObsidianVaultDescriptor.from_path(
                vault_path,
                privacy_settings=privacy_settings,
            )

            # Create redaction policy
            policy = descriptor.build_redaction_policy()

            # Test file with sensitive name
            sensitive_file = vault_path / "secret_project.md"
            sensitive_file.write_text("content")

            # Apply redaction multiple times
            redacted1 = policy.apply(sensitive_file)
            redacted2 = policy.apply(sensitive_file)

            # Should be consistent
            assert redacted1.redacted == redacted2.redacted
            assert redacted1.path_hash == redacted2.path_hash

            # Should not reveal sensitive filename
            assert "secret" not in redacted1.redacted.lower()

        finally:
            temp_dir.cleanup()
            audit_dir.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])