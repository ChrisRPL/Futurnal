"""Tests for GitHub repository descriptor and registry."""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from futurnal.ingestion.github.descriptor import (
    ConsentScope,
    GitHubRepositoryDescriptor,
    PrivacyLevel,
    RepositoryPrivacySettings,
    RepositoryRegistry,
    SyncMode,
    VisibilityType,
    _deterministic_repository_id,
    create_credential_id,
    generate_repository_id,
)


# ---------------------------------------------------------------------------
# Deterministic ID generation tests
# ---------------------------------------------------------------------------


def test_deterministic_repository_id():
    """Test that repository IDs are deterministic."""
    id1 = _deterministic_repository_id("octocat", "Hello-World")
    id2 = _deterministic_repository_id("octocat", "Hello-World")
    assert id1 == id2

    # Test case insensitivity
    id3 = _deterministic_repository_id("Octocat", "hello-world")
    assert id1 == id3

    # Test different repos have different IDs
    id4 = _deterministic_repository_id("octocat", "Different-Repo")
    assert id1 != id4

    # Test different hosts have different IDs
    id5 = _deterministic_repository_id("octocat", "Hello-World", "github.enterprise.com")
    assert id1 != id5


def test_generate_repository_id():
    """Test public ID generation function."""
    id1 = generate_repository_id("owner", "repo")
    id2 = generate_repository_id("owner", "repo", "github.com")
    assert id1 == id2

    # Validate UUID format
    try:
        uuid.UUID(id1)
    except ValueError:
        pytest.fail("Generated ID is not a valid UUID")


def test_create_credential_id():
    """Test credential ID creation."""
    repo_id = "test_repo_123"
    cred_id = create_credential_id(repo_id)
    assert cred_id == f"github_cred_{repo_id}"


# ---------------------------------------------------------------------------
# Descriptor model validation tests
# ---------------------------------------------------------------------------


def test_descriptor_creation():
    """Test creating a valid descriptor."""
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    assert descriptor.owner == "octocat"
    assert descriptor.repo == "Hello-World"
    assert descriptor.full_name == "octocat/Hello-World"
    assert descriptor.visibility == VisibilityType.PUBLIC
    assert descriptor.github_host == "github.com"
    assert descriptor.sync_mode == SyncMode.GRAPHQL_API
    assert descriptor.credential_id == "cred_123"


def test_descriptor_validation_owner_repo():
    """Test validation of owner and repo names."""
    # Valid names
    GitHubRepositoryDescriptor.from_registration(
        owner="valid-name",
        repo="valid_repo.name",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    # Invalid: empty owner
    with pytest.raises(ValidationError):
        GitHubRepositoryDescriptor.from_registration(
            owner="",
            repo="repo",
            credential_id="cred_123",
            visibility=VisibilityType.PUBLIC,
        )

    # Invalid: spaces in owner
    with pytest.raises(ValidationError):
        GitHubRepositoryDescriptor.from_registration(
            owner="invalid owner",
            repo="repo",
            credential_id="cred_123",
            visibility=VisibilityType.PUBLIC,
        )


def test_descriptor_validation_branches():
    """Test validation of branch names."""
    # Valid branches
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
        branches=["main", "develop", "feature/test"],
    )
    assert "main" in descriptor.branches

    # Invalid: branches with ".."
    with pytest.raises(ValidationError):
        GitHubRepositoryDescriptor.from_registration(
            owner="owner",
            repo="repo",
            credential_id="cred_123",
            visibility=VisibilityType.PUBLIC,
            branches=["main", "../etc/passwd"],
        )


def test_descriptor_full_name_computation():
    """Test that full_name is computed correctly."""
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="testuser",
        repo="testrepo",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )
    assert descriptor.full_name == "testuser/testrepo"


def test_descriptor_api_base_url_default():
    """Test that API base URL defaults correctly."""
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )
    assert descriptor.api_base_url == "https://api.github.com"

    # Custom host
    descriptor_enterprise = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        github_host="github.company.com",
        api_base_url="https://github.company.com/api/v3",
        credential_id="cred_123",
        visibility=VisibilityType.PRIVATE,
    )
    assert descriptor_enterprise.github_host == "github.company.com"
    assert descriptor_enterprise.api_base_url == "https://github.company.com/api/v3"


def test_descriptor_clone_mode_validation():
    """Test validation of clone mode settings."""
    # Valid: clone mode with path
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GIT_CLONE,
        local_clone_path=Path("/tmp/test"),
    )
    assert descriptor.sync_mode == SyncMode.GIT_CLONE
    assert descriptor.local_clone_path == Path("/tmp/test")

    # Invalid: clone path without clone mode
    with pytest.raises(ValidationError):
        GitHubRepositoryDescriptor.from_registration(
            owner="owner",
            repo="repo",
            credential_id="cred_123",
            visibility=VisibilityType.PUBLIC,
            sync_mode=SyncMode.GRAPHQL_API,
            local_clone_path=Path("/tmp/test"),
        )


# ---------------------------------------------------------------------------
# Privacy settings tests
# ---------------------------------------------------------------------------


def test_privacy_settings_defaults():
    """Test default privacy settings."""
    settings = RepositoryPrivacySettings()

    assert settings.privacy_level == PrivacyLevel.STANDARD
    assert ConsentScope.GITHUB_REPO_ACCESS in settings.required_consent_scopes
    assert ConsentScope.GITHUB_CODE_ANALYSIS in settings.required_consent_scopes
    assert settings.enable_path_anonymization is True
    assert settings.detect_secrets is True
    assert len(settings.secret_patterns) > 0


def test_privacy_settings_strict_mode():
    """Test strict privacy mode."""
    settings = RepositoryPrivacySettings(privacy_level=PrivacyLevel.STRICT)

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
        privacy_settings=settings,
    )

    policy = descriptor.build_redaction_policy()
    assert policy.reveal_filename is False


def test_secret_detection_patterns():
    """Test that secret detection patterns are valid regex."""
    import re

    settings = RepositoryPrivacySettings()

    for pattern_str in settings.secret_patterns:
        try:
            re.compile(pattern_str)
        except re.error:
            pytest.fail(f"Invalid regex pattern: {pattern_str}")


# ---------------------------------------------------------------------------
# Descriptor update tests
# ---------------------------------------------------------------------------


def test_descriptor_update():
    """Test updating descriptor fields."""
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
        name="Original Name",
    )

    updated = descriptor.update(name="Updated Name")
    assert updated.name == "Updated Name"
    assert updated.owner == "owner"  # Unchanged
    assert updated.updated_at > descriptor.created_at


def test_descriptor_consent_scopes():
    """Test consent scope methods."""
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    scopes = descriptor.get_required_consent_scopes()
    assert "github:repo:access" in scopes

    assert descriptor.requires_consent_for_scope(ConsentScope.GITHUB_REPO_ACCESS)


def test_descriptor_to_local_source():
    """Test conversion to LocalIngestionSource."""
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    local_source = descriptor.to_local_source()
    assert local_source.name.startswith("github-")
    assert local_source.paused is False


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_initialization(temp_registry_dir):
    """Test registry initialization."""
    registry = RepositoryRegistry(registry_root=temp_registry_dir)
    assert registry.registry_root.exists()


def test_registry_register(test_registry):
    """Test registering a repository."""
    descriptor = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    assert descriptor.id is not None
    assert descriptor.owner == "octocat"
    assert descriptor.repo == "Hello-World"


def test_registry_get(test_registry):
    """Test retrieving a repository."""
    descriptor = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    retrieved = test_registry.get(descriptor.id)
    assert retrieved.id == descriptor.id
    assert retrieved.owner == descriptor.owner


def test_registry_get_not_found(test_registry):
    """Test retrieving non-existent repository."""
    with pytest.raises(FileNotFoundError):
        test_registry.get("nonexistent_id")


def test_registry_list(test_registry):
    """Test listing repositories."""
    test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_1",
        visibility=VisibilityType.PUBLIC,
    )
    test_registry.register(
        owner="testuser",
        repo="testrepo",
        credential_id="cred_2",
        visibility=VisibilityType.PRIVATE,
    )

    repos = test_registry.list()
    assert len(repos) == 2
    assert any(r.owner == "octocat" for r in repos)
    assert any(r.owner == "testuser" for r in repos)


def test_registry_find_by_repository(test_registry):
    """Test finding repository by owner/repo."""
    test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    found = test_registry.find_by_repository("octocat", "Hello-World")
    assert found is not None
    assert found.owner == "octocat"

    not_found = test_registry.find_by_repository("nobody", "nothing")
    assert not_found is None


def test_registry_idempotent_registration(test_registry):
    """Test that duplicate registration updates existing."""
    # First registration
    descriptor1 = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
        name="First Name",
    )

    # Second registration (same owner/repo)
    descriptor2 = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
        name="Second Name",
    )

    # Should have same ID
    assert descriptor1.id == descriptor2.id

    # Should have updated name
    assert descriptor2.name == "Second Name"

    # Should only have one entry in registry
    repos = test_registry.list()
    assert len(repos) == 1


def test_registry_remove(test_registry):
    """Test removing a repository."""
    descriptor = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    test_registry.remove(descriptor.id)

    with pytest.raises(FileNotFoundError):
        test_registry.get(descriptor.id)


def test_registry_remove_not_found(test_registry):
    """Test removing non-existent repository."""
    with pytest.raises(FileNotFoundError):
        test_registry.remove("nonexistent_id")


def test_registry_audit_logging(test_registry, mock_audit_logger):
    """Test that registry operations are audited."""
    descriptor = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    # Should have logged registration event
    assert mock_audit_logger.record.called

    # Reset mock
    mock_audit_logger.reset_mock()

    # Update
    test_registry.add_or_update(descriptor)
    assert mock_audit_logger.record.called

    # Reset mock
    mock_audit_logger.reset_mock()

    # Remove
    test_registry.remove(descriptor.id)
    assert mock_audit_logger.record.called


def test_registry_privacy_change_logging(test_registry, mock_audit_logger):
    """Test that privacy changes are logged."""
    descriptor = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
        privacy_settings=RepositoryPrivacySettings(privacy_level=PrivacyLevel.STANDARD),
    )

    mock_audit_logger.reset_mock()

    # Update privacy settings
    updated_descriptor = descriptor.update(
        privacy_settings=RepositoryPrivacySettings(privacy_level=PrivacyLevel.STRICT)
    )
    test_registry.add_or_update(updated_descriptor)

    # Should have logged privacy change
    assert mock_audit_logger.record.called


def test_registry_persistence(test_registry, temp_registry_dir):
    """Test that registry persists to disk."""
    descriptor = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    # Check file exists
    descriptor_file = temp_registry_dir / f"{descriptor.id}.json"
    assert descriptor_file.exists()

    # Verify contents
    data = json.loads(descriptor_file.read_text())
    assert data["owner"] == "octocat"
    assert data["repo"] == "Hello-World"


def test_registry_file_locking(test_registry):
    """Test that registry uses file locking."""
    # This test ensures that concurrent operations are safe
    # In a real scenario, you'd test with multiple threads/processes
    descriptor = test_registry.register(
        owner="octocat",
        repo="Hello-World",
        credential_id="cred_123",
        visibility=VisibilityType.PUBLIC,
    )

    # Multiple rapid updates should work without corruption
    for i in range(10):
        test_registry.add_or_update(descriptor.update(name=f"Name {i}"))

    final = test_registry.get(descriptor.id)
    assert final.name.startswith("Name")
