"""Comprehensive privacy integration tests for GitHub connector.

Tests the complete privacy framework including:
- GitHub-specific consent scopes
- Privacy-aware audit logging
- Consent management functions
- Redaction policy creation
- Secret scanning integration
- End-to-end privacy workflows
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from futurnal.ingestion.github import (
    GitHubAuditEvent,
    GitHubAuditLogger,
    GitHubConsentScope,
    GitHubRepositoryDescriptor,
    PrivacyLevel,
    RepositoryPrivacySettings,
    create_github_redaction_policy,
    register_repository_consent,
    require_consent,
    revoke_repository_consent,
)
from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.redaction import RedactionPolicy


class TestGitHubConsentScope:
    """Test GitHub-specific consent scope enumeration."""

    def test_consent_scope_values(self):
        """Test that consent scopes have correct values."""
        assert GitHubConsentScope.GITHUB_REPO_ACCESS.value == "github:repo:access"
        assert (
            GitHubConsentScope.GITHUB_CODE_ANALYSIS.value == "github:repo:code_analysis"
        )
        assert (
            GitHubConsentScope.GITHUB_ISSUE_METADATA.value
            == "github:repo:issue_metadata"
        )
        assert GitHubConsentScope.GITHUB_PR_METADATA.value == "github:repo:pr_metadata"
        assert GitHubConsentScope.GITHUB_WIKI_ACCESS.value == "github:repo:wiki_access"
        assert (
            GitHubConsentScope.GITHUB_CLOUD_MODELS.value == "github:repo:cloud_models"
        )
        assert (
            GitHubConsentScope.GITHUB_PARTICIPANT_ANALYSIS.value
            == "github:repo:participant_analysis"
        )

    def test_scope_enum_members(self):
        """Test that all expected scope members exist."""
        scopes = list(GitHubConsentScope)
        assert len(scopes) == 7
        assert GitHubConsentScope.GITHUB_REPO_ACCESS in scopes
        assert GitHubConsentScope.GITHUB_CODE_ANALYSIS in scopes


class TestGitHubAuditEvent:
    """Test GitHub audit event model."""

    def test_audit_event_creation(self):
        """Test creating a GitHub audit event."""
        event = GitHubAuditEvent(
            job_id="test_job_123",
            action="repository_sync",
            status="success",
            repo_id="repo_123",
            repo_full_name_hash="abc123def456",
            branch="main",
            files_processed=10,
            files_skipped=2,
            bytes_processed=1024,
            commits_processed=5,
        )

        assert event.job_id == "test_job_123"
        assert event.source == "github_connector"
        assert event.action == "repository_sync"
        assert event.status == "success"
        assert event.repo_id == "repo_123"
        assert event.repo_full_name_hash == "abc123def456"
        assert event.branch == "main"
        assert event.files_processed == 10
        assert event.files_skipped == 2
        assert event.bytes_processed == 1024
        assert event.commits_processed == 5

    def test_audit_event_to_base_conversion(self):
        """Test conversion to base AuditEvent."""
        event = GitHubAuditEvent(
            job_id="test_job_123",
            action="repository_sync",
            status="success",
            repo_id="repo_123",
            repo_full_name_hash="abc123def456",
            branch="main",
            files_processed=10,
            error_type="ValueError",
            error_message_sanitized="Test error",
        )

        base_event = event.to_base_audit_event()

        assert base_event.job_id == "test_job_123"
        assert base_event.source == "github_connector"
        assert base_event.action == "repository_sync"
        assert base_event.status == "success"
        assert base_event.metadata["repo_id"] == "repo_123"
        assert base_event.metadata["branch"] == "main"
        assert base_event.metadata["files_processed"] == 10
        assert base_event.metadata["error_type"] == "ValueError"
        assert base_event.metadata["error_message_sanitized"] == "Test error"


class TestGitHubAuditLogger:
    """Test GitHub-specific audit logger."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_dir = Path(self.temp_dir.name)

        self.base_logger = AuditLogger(output_dir=self.audit_dir)
        self.github_logger = GitHubAuditLogger(
            audit_logger=self.base_logger,
        )

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_log_repository_sync(self):
        """Test logging repository sync events."""
        self.github_logger.log_repository_sync(
            repo_id="repo_123",
            repo_full_name="owner/repo",
            branch="main",
            files_processed=10,
            files_skipped=2,
            bytes_processed=1024,
            commits_processed=5,
            modified_files=["file1.py", "file2.py"],
            status="success",
        )

        # Verify audit log was created
        audit_file = self.audit_dir / "audit.log"
        assert audit_file.exists()

        # Read and parse audit log
        audit_content = audit_file.read_text().strip()
        audit_event = json.loads(audit_content)

        assert audit_event["source"] == "github_connector"
        assert audit_event["action"] == "repository_sync"
        assert audit_event["status"] == "success"
        assert audit_event["metadata"]["repo_id"] == "repo_123"
        assert audit_event["metadata"]["branch"] == "main"
        assert audit_event["metadata"]["files_processed"] == 10

    def test_log_consent_check(self):
        """Test logging consent check events."""
        self.github_logger.log_consent_check(
            repo_id="repo_123",
            scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS,
            granted=True,
        )

        audit_file = self.audit_dir / "audit.log"
        assert audit_file.exists()

        audit_content = audit_file.read_text().strip()
        audit_event = json.loads(audit_content)

        assert "consent_check" in audit_event["action"]
        assert audit_event["status"] == "granted"
        assert audit_event["metadata"]["repo_id"] == "repo_123"

    def test_log_secret_detection(self):
        """Test logging secret detection events."""
        self.github_logger.log_secret_detection(
            repo_id="repo_123",
            file_path="/path/to/secret_file.txt",
            detected=True,
        )

        audit_file = self.audit_dir / "audit.log"
        assert audit_file.exists()

        audit_content = audit_file.read_text().strip()
        audit_event = json.loads(audit_content)

        assert audit_event["action"] == "secret_detection"
        assert audit_event["status"] == "detected"
        assert "file_path_hashes" in audit_event["metadata"]

    def test_log_error(self):
        """Test logging error events with sanitized messages."""
        test_error = ValueError("Test error message")

        self.github_logger.log_error(
            repo_id="repo_123",
            action="repository_sync",
            error=test_error,
        )

        audit_file = self.audit_dir / "audit.log"
        assert audit_file.exists()

        audit_content = audit_file.read_text().strip()
        audit_event = json.loads(audit_content)

        assert audit_event["action"] == "repository_sync"
        assert audit_event["status"] == "error"
        assert audit_event["metadata"]["error_type"] == "ValueError"
        assert "error_message_sanitized" in audit_event["metadata"]

    def test_path_hashing_privacy(self):
        """Test that file paths are hashed for privacy."""
        self.github_logger.log_repository_sync(
            repo_id="repo_123",
            repo_full_name="owner/secret-repo",
            branch="main",
            modified_files=["/path/to/secret/file.txt"],
        )

        audit_file = self.audit_dir / "audit.log"
        audit_content = audit_file.read_text()

        # Verify that actual paths don't appear in logs
        assert "/path/to/secret/file.txt" not in audit_content
        assert "secret-repo" not in audit_content

        # Verify that hashes are present
        audit_event = json.loads(audit_content.strip())
        assert "file_path_hashes" in audit_event["metadata"]
        assert len(audit_event["metadata"]["file_path_hashes"]) > 0


class TestConsentManagement:
    """Test consent management functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.consent_dir = Path(self.temp_dir.name) / "consent"
        self.consent_registry = ConsentRegistry(self.consent_dir)

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_register_repository_consent(self):
        """Test registering consent for repository."""
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner="octocat",
            repo="Hello-World",
            credential_id="test_cred_123",
            privacy_settings=RepositoryPrivacySettings(
                required_consent_scopes=[
                    GitHubConsentScope.GITHUB_REPO_ACCESS,
                    GitHubConsentScope.GITHUB_CODE_ANALYSIS,
                ]
            ),
        )

        records = register_repository_consent(
            consent_registry=self.consent_registry,
            repo_descriptor=descriptor,
            operator="test_user",
        )

        assert len(records) == 2
        assert all(record.granted for record in records)
        assert all(record.operator == "test_user" for record in records)

        # Verify consent is stored
        consent = self.consent_registry.get(
            source=descriptor.id,
            scope=GitHubConsentScope.GITHUB_REPO_ACCESS.value,
        )
        assert consent is not None
        assert consent.is_active()

    def test_require_consent_granted(self):
        """Test require_consent context manager when consent is granted."""
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner="octocat",
            repo="Hello-World",
            credential_id="test_cred_123",
        )

        # Grant consent
        self.consent_registry.grant(
            source=descriptor.id,
            scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS.value,
        )

        # Should not raise
        with require_consent(
            consent_registry=self.consent_registry,
            repo_id=descriptor.id,
            scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS,
        ) as consent:
            assert consent is not None
            assert consent.is_active()

    def test_require_consent_not_granted(self):
        """Test require_consent raises error when consent not granted."""
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner="octocat",
            repo="Hello-World",
            credential_id="test_cred_123",
        )

        # Should raise ConsentRequiredError
        with pytest.raises(ConsentRequiredError, match="Consent required"):
            with require_consent(
                consent_registry=self.consent_registry,
                repo_id=descriptor.id,
                scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS,
            ):
                pass

    def test_require_consent_with_audit_logging(self):
        """Test that consent checks are logged to audit."""
        temp_audit_dir = tempfile.TemporaryDirectory()
        try:
            audit_logger = AuditLogger(output_dir=Path(temp_audit_dir.name))
            github_audit_logger = GitHubAuditLogger(audit_logger=audit_logger)

            descriptor = GitHubRepositoryDescriptor.from_registration(
                owner="octocat",
                repo="Hello-World",
                credential_id="test_cred_123",
            )

            # Grant consent
            self.consent_registry.grant(
                source=descriptor.id,
                scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS.value,
            )

            # Use require_consent with audit logging
            with require_consent(
                consent_registry=self.consent_registry,
                repo_id=descriptor.id,
                scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS,
                audit_logger=github_audit_logger,
            ):
                pass

            # Check that audit log contains consent check
            audit_file = Path(temp_audit_dir.name) / "audit.log"
            assert audit_file.exists()

            audit_content = audit_file.read_text()
            assert "consent_check" in audit_content
            assert "granted" in audit_content

        finally:
            temp_audit_dir.cleanup()

    def test_revoke_repository_consent(self):
        """Test revoking repository consent."""
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner="octocat",
            repo="Hello-World",
            credential_id="test_cred_123",
        )

        # Grant consent first
        self.consent_registry.grant(
            source=descriptor.id,
            scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS.value,
        )

        # Verify consent is active
        consent = self.consent_registry.get(
            source=descriptor.id,
            scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS.value,
        )
        assert consent.is_active()

        # Revoke consent
        import asyncio

        asyncio.run(
            revoke_repository_consent(
                consent_registry=self.consent_registry,
                repo_id=descriptor.id,
                scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS,
                cleanup_data=False,
            )
        )

        # Verify consent is revoked
        consent = self.consent_registry.get(
            source=descriptor.id,
            scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS.value,
        )
        assert not consent.is_active()


class TestRedactionPolicyCreation:
    """Test GitHub redaction policy creation."""

    def test_create_standard_privacy_policy(self):
        """Test creating redaction policy for standard privacy level."""
        privacy_settings = RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.STANDARD,
        )

        policy = create_github_redaction_policy(privacy_settings)

        assert isinstance(policy, RedactionPolicy)
        assert policy.allow_plaintext is False
        assert policy.reveal_filename is True
        assert policy.reveal_extension is False

    def test_create_strict_privacy_policy(self):
        """Test creating redaction policy for strict privacy level."""
        privacy_settings = RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
        )

        policy = create_github_redaction_policy(privacy_settings)

        assert policy.allow_plaintext is False
        assert policy.reveal_filename is False
        assert policy.reveal_extension is False

    def test_create_permissive_privacy_policy(self):
        """Test creating redaction policy for permissive privacy level."""
        privacy_settings = RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.PERMISSIVE,
            enable_path_anonymization=False,
        )

        policy = create_github_redaction_policy(privacy_settings)

        assert policy.allow_plaintext is True
        assert policy.reveal_extension is True

    def test_policy_respects_path_anonymization_setting(self):
        """Test that policy respects path anonymization setting."""
        # Permissive with anonymization enabled
        privacy_settings = RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.PERMISSIVE,
            enable_path_anonymization=True,
        )

        policy = create_github_redaction_policy(privacy_settings)
        assert policy.allow_plaintext is False


class TestEndToEndPrivacyWorkflow:
    """Test complete end-to-end privacy workflows."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.consent_dir = Path(self.temp_dir.name) / "consent"
        self.audit_dir = Path(self.temp_dir.name) / "audit"

        self.consent_registry = ConsentRegistry(self.consent_dir)
        self.audit_logger = AuditLogger(output_dir=self.audit_dir)
        self.github_audit_logger = GitHubAuditLogger(
            audit_logger=self.audit_logger,
        )

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_full_repository_privacy_lifecycle(self):
        """Test complete repository privacy lifecycle."""
        # 1. Create repository descriptor with privacy settings
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner="octocat",
            repo="Hello-World",
            credential_id="test_cred_123",
            privacy_settings=RepositoryPrivacySettings(
                privacy_level=PrivacyLevel.STRICT,
                required_consent_scopes=[
                    GitHubConsentScope.GITHUB_REPO_ACCESS,
                    GitHubConsentScope.GITHUB_CODE_ANALYSIS,
                ],
            ),
        )

        # 2. Register consent
        records = register_repository_consent(
            consent_registry=self.consent_registry,
            repo_descriptor=descriptor,
            operator="test_user",
        )
        assert len(records) == 2

        # 3. Verify consent with audit logging
        with require_consent(
            consent_registry=self.consent_registry,
            repo_id=descriptor.id,
            scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS,
            audit_logger=self.github_audit_logger,
        ):
            # Simulate repository sync
            self.github_audit_logger.log_repository_sync(
                repo_id=descriptor.id,
                repo_full_name=descriptor.full_name,
                branch="main",
                files_processed=10,
            )

        # 4. Verify audit logs exist and contain no sensitive data
        audit_file = self.audit_dir / "audit.log"
        assert audit_file.exists()

        audit_content = audit_file.read_text()
        assert "Hello-World" not in audit_content  # Repo name should be hashed
        assert "consent_check" in audit_content
        assert "repository_sync" in audit_content

        # 5. Revoke consent
        import asyncio

        asyncio.run(
            revoke_repository_consent(
                consent_registry=self.consent_registry,
                repo_id=descriptor.id,
                scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS,
                cleanup_data=False,
                audit_logger=self.github_audit_logger,
            )
        )

        # 6. Verify consent is revoked
        with pytest.raises(ConsentRequiredError):
            with require_consent(
                consent_registry=self.consent_registry,
                repo_id=descriptor.id,
                scope=GitHubConsentScope.GITHUB_CODE_ANALYSIS,
            ):
                pass


class TestPrivacyComplianceValidation:
    """Test compliance with privacy requirements."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_dir = Path(self.temp_dir.name) / "audit"
        self.audit_logger = AuditLogger(output_dir=self.audit_dir)
        self.github_audit_logger = GitHubAuditLogger(
            audit_logger=self.audit_logger,
        )

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_no_source_code_in_audit_logs(self):
        """Verify that no source code appears in audit logs."""
        self.github_audit_logger.log_repository_sync(
            repo_id="repo_123",
            repo_full_name="owner/secret-repo",
            branch="main",
            files_processed=5,
            modified_files=[
                "/src/secret_module.py",
                "/config/credentials.json",
            ],
        )

        audit_file = self.audit_dir / "audit.log"
        audit_content = audit_file.read_text()

        # Verify no file paths appear
        assert "/src/secret_module.py" not in audit_content
        assert "credentials.json" not in audit_content
        assert "secret-repo" not in audit_content

    def test_no_file_paths_in_audit_logs(self):
        """Verify that file paths are always hashed."""
        self.github_audit_logger.log_secret_detection(
            repo_id="repo_123",
            file_path="/path/to/sensitive/file.txt",
            detected=True,
        )

        audit_file = self.audit_dir / "audit.log"
        audit_content = audit_file.read_text()

        # Verify path is hashed
        assert "/path/to/sensitive/file.txt" not in audit_content
        assert "sensitive" not in audit_content

    def test_consistent_hash_output(self):
        """Verify that hashing is deterministic."""
        logger1 = GitHubAuditLogger(audit_logger=self.audit_logger)
        logger2 = GitHubAuditLogger(audit_logger=self.audit_logger)

        hash1 = logger1._hash_path("/test/path/file.txt")
        hash2 = logger2._hash_path("/test/path/file.txt")

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
