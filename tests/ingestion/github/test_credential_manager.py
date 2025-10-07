"""Tests for GitHub credential manager."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from futurnal.ingestion.github.credential_manager import (
    CredentialType,
    EnterpriseOAuthConfig,
    GitHubCredential,
    GitHubCredentialManager,
    GitHubOAuthConfig,
    OAuthToken,
    OAuthTokens,
    PersonalAccessToken,
    auto_refresh_wrapper,
    detect_token_type,
    secure_credential_context,
    validate_token_format,
)


# ---------------------------------------------------------------------------
# Token detection and validation tests
# ---------------------------------------------------------------------------


def test_detect_token_type():
    """Test token type detection from format."""
    # Classic PAT
    assert detect_token_type("ghp_" + "x" * 36) == CredentialType.PERSONAL_ACCESS_TOKEN

    # Fine-grained PAT
    assert (
        detect_token_type("github_pat_" + "x" * 71)
        == CredentialType.PERSONAL_ACCESS_TOKEN
    )

    # OAuth token (fallback)
    assert detect_token_type("gho_" + "x" * 36) == CredentialType.OAUTH_TOKEN


def test_validate_token_format():
    """Test token format validation."""
    # Valid classic PAT
    assert validate_token_format(
        "ghp_" + "x" * 36, CredentialType.PERSONAL_ACCESS_TOKEN
    )

    # Valid fine-grained PAT
    assert validate_token_format(
        "github_pat_" + "x" * 71, CredentialType.PERSONAL_ACCESS_TOKEN
    )

    # Invalid PAT (wrong length)
    assert not validate_token_format("ghp_short", CredentialType.PERSONAL_ACCESS_TOKEN)

    # Valid OAuth token (lenient)
    assert validate_token_format("x" * 40, CredentialType.OAUTH_TOKEN)

    # Invalid OAuth token (too short)
    assert not validate_token_format("short", CredentialType.OAUTH_TOKEN)


# ---------------------------------------------------------------------------
# Credential manager initialization tests
# ---------------------------------------------------------------------------


def test_credential_manager_initialization(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test credential manager initialization."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    assert manager.metadata_path.parent.exists()
    assert manager.metadata_path.exists()


def test_credential_manager_requires_keyring():
    """Test that credential manager requires keyring module."""
    with pytest.raises(RuntimeError, match="keyring module is unavailable"):
        GitHubCredentialManager(keyring_module=None)


# ---------------------------------------------------------------------------
# OAuth token storage and retrieval tests
# ---------------------------------------------------------------------------


def test_store_oauth_token(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test storing OAuth token."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    credential = manager.store_oauth_token(
        credential_id="test_cred_123",
        token="gho_test_token_12345678901234567890",
        scopes=["repo", "user"],
        github_host="github.com",
        note="Test OAuth token",
    )

    assert credential.credential_id == "test_cred_123"
    assert credential.credential_type == CredentialType.OAUTH_TOKEN
    assert credential.scopes == ["repo", "user"]
    assert mock_audit_logger.record.called


def test_retrieve_oauth_token(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test retrieving OAuth token."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store token
    manager.store_oauth_token(
        credential_id="test_cred_123",
        token="gho_test_token_12345678901234567890",
        scopes=["repo"],
    )

    # Retrieve token
    creds = manager.retrieve_credentials("test_cred_123")

    assert isinstance(creds, OAuthToken)
    assert creds.token == "gho_test_token_12345678901234567890"
    assert creds.token_type == "Bearer"
    assert creds.scopes == ["repo"]


# ---------------------------------------------------------------------------
# Personal access token storage and retrieval tests
# ---------------------------------------------------------------------------


def test_store_personal_access_token(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test storing personal access token."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    credential = manager.store_personal_access_token(
        credential_id="test_cred_pat",
        token="ghp_" + "x" * 36,
        scopes=["repo", "read:org"],
        note="Test PAT",
    )

    assert credential.credential_id == "test_cred_pat"
    assert credential.credential_type == CredentialType.PERSONAL_ACCESS_TOKEN
    assert credential.scopes == ["repo", "read:org"]


def test_retrieve_personal_access_token(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test retrieving personal access token."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    token_value = "ghp_" + "x" * 36

    # Store token
    manager.store_personal_access_token(
        credential_id="test_cred_pat", token=token_value, scopes=["repo"]
    )

    # Retrieve token
    creds = manager.retrieve_credentials("test_cred_pat")

    assert isinstance(creds, PersonalAccessToken)
    assert creds.token == token_value
    assert creds.scopes == ["repo"]


# ---------------------------------------------------------------------------
# Credential deletion tests
# ---------------------------------------------------------------------------


def test_delete_credentials(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test deleting credentials."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store credential
    manager.store_oauth_token(
        credential_id="test_cred_delete",
        token="gho_test_token_12345678901234567890",
        scopes=["repo"],
    )

    # Delete credential
    manager.delete_credentials("test_cred_delete")

    # Should raise KeyError when trying to retrieve
    with pytest.raises(KeyError):
        manager.retrieve_credentials("test_cred_delete")


# ---------------------------------------------------------------------------
# Credential listing tests
# ---------------------------------------------------------------------------


def test_list_credentials(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test listing all credentials."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store multiple credentials
    manager.store_oauth_token(
        credential_id="cred_1", token="token_1" * 10, scopes=["repo"]
    )
    manager.store_personal_access_token(
        credential_id="cred_2", token="ghp_" + "x" * 36, scopes=["public_repo"]
    )

    # List credentials
    credentials = manager.list_credentials()

    assert len(credentials) == 2
    assert any(c.credential_id == "cred_1" for c in credentials)
    assert any(c.credential_id == "cred_2" for c in credentials)


# ---------------------------------------------------------------------------
# Metadata operations tests
# ---------------------------------------------------------------------------


def test_get_credential_metadata(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test getting credential metadata without retrieving token."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    manager.store_oauth_token(
        credential_id="test_cred",
        token="token_value",
        scopes=["repo"],
        note="Test note",
    )

    # Get metadata only
    metadata = manager.get_credential_metadata("test_cred")

    assert metadata.credential_id == "test_cred"
    assert metadata.note == "Test note"
    assert metadata.scopes == ["repo"]


def test_update_scopes(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test updating credential scopes."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    manager.store_oauth_token(
        credential_id="test_cred", token="token_value", scopes=["repo"]
    )

    # Update scopes
    updated = manager.update_scopes("test_cred", ["repo", "user", "read:org"])

    assert updated.scopes == ["repo", "user", "read:org"]


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_retrieve_nonexistent_credentials(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test retrieving non-existent credentials."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    with pytest.raises(KeyError):
        manager.retrieve_credentials("nonexistent")


def test_get_nonexistent_metadata(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test getting metadata for non-existent credential."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    with pytest.raises(KeyError):
        manager.get_credential_metadata("nonexistent")


# ---------------------------------------------------------------------------
# Secure context tests
# ---------------------------------------------------------------------------


def test_secure_credential_context(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test secure credential context manager."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    manager.store_oauth_token(
        credential_id="test_cred", token="token_value", scopes=["repo"]
    )

    # Use context manager
    with secure_credential_context(manager, "test_cred") as creds:
        assert isinstance(creds, OAuthToken)
        assert creds.token == "token_value"

    # Credentials should be cleared after context


# ---------------------------------------------------------------------------
# Audit logging tests
# ---------------------------------------------------------------------------


def test_audit_logging_create(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test audit logging for credential creation."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    manager.store_oauth_token(
        credential_id="test_cred",
        token="token_value",
        scopes=["repo"],
        operator="test_user",
    )

    # Should have logged creation event
    assert mock_audit_logger.record.called
    call_args = mock_audit_logger.record.call_args[0][0]
    assert call_args.action == "credential_create"
    assert call_args.status == "success"


def test_audit_logging_access(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test audit logging for credential access."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    manager.store_oauth_token(
        credential_id="test_cred", token="token_value", scopes=["repo"]
    )

    mock_audit_logger.reset_mock()

    # Access credential
    manager.retrieve_credentials("test_cred")

    # Should have logged access event
    assert mock_audit_logger.record.called
    call_args = mock_audit_logger.record.call_args[0][0]
    assert call_args.action == "credential_access"


def test_audit_logging_delete(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test audit logging for credential deletion."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    manager.store_oauth_token(
        credential_id="test_cred", token="token_value", scopes=["repo"]
    )

    mock_audit_logger.reset_mock()

    # Delete credential
    manager.delete_credentials("test_cred", operator="test_user")

    # Should have logged deletion event
    assert mock_audit_logger.record.called
    call_args = mock_audit_logger.record.call_args[0][0]
    assert call_args.action == "credential_delete"


# ---------------------------------------------------------------------------
# Last used tracking tests
# ---------------------------------------------------------------------------


def test_last_used_tracking(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test that credentials track last used time."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    manager.store_oauth_token(
        credential_id="test_cred", token="token_value", scopes=["repo"]
    )

    # Initial state: no last_used_at
    metadata = manager.get_credential_metadata("test_cred")
    assert metadata.last_used_at is None

    # Retrieve credential
    manager.retrieve_credentials("test_cred")

    # Should now have last_used_at
    metadata = manager.get_credential_metadata("test_cred")
    assert metadata.last_used_at is not None


# ---------------------------------------------------------------------------
# Enhanced OAuth token model tests
# ---------------------------------------------------------------------------


def test_oauth_tokens_model():
    """Test OAuthTokens model with expiration."""
    tokens = OAuthTokens(
        access_token="gho_test123456789012345678901234567890",
        refresh_token="ghr_refresh123456789012345678901234",
        scopes=["repo", "user"],
        expires_in=3600,
    )

    # Should calculate expires_at from expires_in
    assert tokens.expires_at is not None
    assert tokens.token_type == "Bearer"
    assert tokens.scopes == ["repo", "user"]


def test_oauth_tokens_should_refresh():
    """Test should_refresh logic."""
    # Token expires in 10 minutes
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    tokens = OAuthTokens(
        access_token="test_token",
        expires_at=expires_at,
    )

    # Should not refresh (10 min > 5 min default buffer)
    assert not tokens.should_refresh()

    # Token expires in 2 minutes
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=2)
    tokens = OAuthTokens(
        access_token="test_token",
        expires_at=expires_at,
    )

    # Should refresh (2 min < 5 min buffer)
    assert tokens.should_refresh()

    # No expiration
    tokens = OAuthTokens(access_token="test_token")
    assert not tokens.should_refresh()


def test_oauth_tokens_serialize():
    """Test OAuthTokens serialization."""
    tokens = OAuthTokens(
        access_token="test_token",
        refresh_token="test_refresh",
        scopes=["repo"],
        username="testuser",
    )

    serialized = tokens.serialize()
    assert serialized["type"] == CredentialType.OAUTH_TOKEN
    assert serialized["access_token"] == "test_token"
    assert serialized["refresh_token"] == "test_refresh"
    assert serialized["scopes"] == ["repo"]
    assert serialized["username"] == "testuser"


# ---------------------------------------------------------------------------
# OAuth configuration tests
# ---------------------------------------------------------------------------


def test_github_oauth_config_default():
    """Test GitHubOAuthConfig with defaults."""
    config = GitHubOAuthConfig()

    # Should have github.com endpoints
    assert "github.com" in config.device_code_endpoint
    assert "github.com" in config.token_endpoint
    assert config.default_scopes == ["repo", "read:org"]


def test_github_oauth_config_get_config_for_host():
    """Test getting OAuth config for specific host."""
    config = GitHubOAuthConfig(client_id="test_client_id")

    # GitHub.com
    github_com_config = config.get_config_for_host("github.com")
    assert github_com_config["client_id"] == "test_client_id"
    assert "github.com" in github_com_config["device_code_endpoint"]

    # Enterprise Server
    enterprise_config = EnterpriseOAuthConfig(
        host="github.company.com",
        client_id="enterprise_client_id",
        api_base_url="https://github.company.com/api/v3",
    )
    config.enterprise_configs["github.company.com"] = enterprise_config

    enterprise_host_config = config.get_config_for_host("github.company.com")
    assert enterprise_host_config["client_id"] == "enterprise_client_id"
    assert "github.company.com" in enterprise_host_config["device_code_endpoint"]


def test_github_oauth_config_unknown_host():
    """Test error for unknown host."""
    config = GitHubOAuthConfig()

    with pytest.raises(ValueError, match="No OAuth configuration found"):
        config.get_config_for_host("unknown.host.com")


# ---------------------------------------------------------------------------
# Enhanced OAuth token storage tests
# ---------------------------------------------------------------------------


def test_store_oauth_tokens_with_expiration(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test storing OAuth tokens with expiration."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    tokens = OAuthTokens(
        access_token="gho_test123456789012345678901234567890",
        refresh_token="ghr_refresh123456789012345678901234",
        scopes=["repo", "user"],
        expires_at=expires_at,
        username="testuser",
    )

    credential = manager.store_oauth_tokens(
        credential_id="test_cred_oauth",
        tokens=tokens,
        note="Test OAuth with expiration",
    )

    assert credential.credential_id == "test_cred_oauth"
    assert credential.credential_type == CredentialType.OAUTH_TOKEN
    assert credential.scopes == ["repo", "user"]


def test_retrieve_oauth_tokens_with_expiration(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test retrieving OAuth tokens with expiration."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    tokens = OAuthTokens(
        access_token="gho_test123456789012345678901234567890",
        refresh_token="ghr_refresh123456789012345678901234",
        scopes=["repo"],
        expires_at=expires_at,
    )

    manager.store_oauth_tokens(credential_id="test_cred", tokens=tokens)

    # Retrieve should return OAuthTokens
    retrieved = manager.retrieve_credentials("test_cred", auto_refresh=False)

    assert isinstance(retrieved, OAuthTokens)
    assert retrieved.access_token == "gho_test123456789012345678901234567890"
    assert retrieved.refresh_token == "ghr_refresh123456789012345678901234"
    assert retrieved.expires_at is not None


# ---------------------------------------------------------------------------
# Token refresh tests
# ---------------------------------------------------------------------------


@patch("futurnal.ingestion.github.credential_manager.requests.post")
def test_refresh_oauth_token(
    mock_post, temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test OAuth token refresh."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
        oauth_config=GitHubOAuthConfig(client_id="test_client_id"),
    )

    # Store token with refresh token
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=2)
    tokens = OAuthTokens(
        access_token="old_access_token",
        refresh_token="test_refresh_token",
        expires_at=expires_at,
        scopes=["repo"],
    )
    manager.store_oauth_tokens(credential_id="test_cred", tokens=tokens)

    # Mock successful refresh response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_access_token",
        "refresh_token": "new_refresh_token",
        "token_type": "bearer",
        "scope": "repo user",
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    # Refresh token
    new_tokens = manager.refresh_oauth_token("test_cred")

    assert new_tokens.access_token == "new_access_token"
    assert new_tokens.refresh_token == "new_refresh_token"
    assert "repo" in new_tokens.scopes
    assert "user" in new_tokens.scopes
    assert mock_post.called


def test_refresh_oauth_token_no_refresh_token(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test error when no refresh token available."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store token without refresh token
    tokens = OAuthTokens(access_token="test_token", scopes=["repo"])
    manager.store_oauth_tokens(credential_id="test_cred", tokens=tokens)

    # Should raise ValueError
    with pytest.raises(ValueError, match="No refresh token available"):
        manager.refresh_oauth_token("test_cred")


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@patch("futurnal.ingestion.github.credential_manager.requests.get")
def test_fetch_username(mock_get, temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test fetching username from GitHub API."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Mock API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"login": "testuser", "id": 12345}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    username = manager.fetch_username("test_token")

    assert username == "testuser"
    assert mock_get.called


@patch("futurnal.ingestion.github.credential_manager.requests.get")
def test_detect_scopes_from_api(
    mock_get, temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test detecting scopes from GitHub API."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Mock API response with scopes
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"X-OAuth-Scopes": "repo, user, read:org"}
    mock_response.json.return_value = {"rate": {"limit": 5000}}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    scopes = manager.detect_scopes_from_api("test_token")

    assert "repo" in scopes
    assert "user" in scopes
    assert "read:org" in scopes
    assert mock_get.called


@patch("futurnal.ingestion.github.credential_manager.requests.get")
def test_validate_credentials(
    mock_get, temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test credential validation."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store credential
    manager.store_oauth_token(
        credential_id="test_cred", token="valid_token", scopes=["repo"]
    )

    # Mock successful API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"login": "testuser"}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Should return True for valid credential
    assert manager.validate_credentials("test_cred") is True


def test_validate_credentials_invalid(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test validation of invalid credentials."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Non-existent credential
    assert manager.validate_credentials("nonexistent") is False


# ---------------------------------------------------------------------------
# Auto-refresh wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_refresh_wrapper(temp_credentials_dir, mock_keyring, mock_audit_logger):
    """Test auto-refresh wrapper utility."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store OAuth tokens with expiration
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    tokens = OAuthTokens(
        access_token="test_token",
        expires_at=expires_at,
        scopes=["repo"],
    )
    manager.store_oauth_tokens(credential_id="test_cred", tokens=tokens)

    # Use wrapper
    retrieved = await auto_refresh_wrapper(manager, "test_cred")

    assert isinstance(retrieved, OAuthTokens)
    assert retrieved.access_token == "test_token"


@pytest.mark.asyncio
async def test_auto_refresh_wrapper_wrong_type(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test auto-refresh wrapper with non-OAuth credential."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store PAT (not OAuth)
    manager.store_personal_access_token(
        credential_id="test_cred", token="ghp_" + "x" * 36, scopes=["repo"]
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="only supports OAuthTokens"):
        await auto_refresh_wrapper(manager, "test_cred")


# ---------------------------------------------------------------------------
# New metadata fields tests (username, expires_at, token_prefix)
# ---------------------------------------------------------------------------


def test_oauth_tokens_username_in_metadata(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test that username is stored in GitHubCredential metadata."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store OAuth tokens with username
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    tokens = OAuthTokens(
        access_token="gho_test123",
        scopes=["repo"],
        expires_at=expires_at,
        username="octocat",
    )
    manager.store_oauth_tokens(credential_id="test_cred", tokens=tokens)

    # Retrieve metadata
    metadata = manager.get_credential_metadata("test_cred")

    assert metadata.username == "octocat"
    assert metadata.expires_at is not None
    assert metadata.expires_at == expires_at


def test_oauth_tokens_expires_at_in_metadata(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test that expires_at is stored in GitHubCredential metadata."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    # Store OAuth tokens with expiration
    expires_at = datetime.now(timezone.utc) + timedelta(hours=2)
    tokens = OAuthTokens(
        access_token="gho_test456",
        scopes=["repo", "user"],
        expires_at=expires_at,
    )
    manager.store_oauth_tokens(credential_id="test_cred_exp", tokens=tokens)

    # Retrieve metadata (without accessing keychain)
    metadata = manager.get_credential_metadata("test_cred_exp")

    assert metadata.expires_at is not None
    # Compare timestamps (allow small drift)
    assert abs((metadata.expires_at - expires_at).total_seconds()) < 1


def test_personal_access_token_prefix(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test that PersonalAccessToken includes token_prefix."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    token = "ghp_" + "x" * 36
    manager.store_personal_access_token(
        credential_id="test_pat", token=token, scopes=["repo"]
    )

    # Retrieve credentials
    creds = manager.retrieve_credentials("test_pat")

    assert isinstance(creds, PersonalAccessToken)
    assert creds.token_prefix == "ghp_xxx"
    assert creds.token == token


def test_personal_access_token_fine_grained_prefix(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test token_prefix for fine-grained PAT."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    token = "github_pat_" + "y" * 71
    manager.store_personal_access_token(
        credential_id="test_pat_fg", token=token, scopes=["repo"]
    )

    # Retrieve credentials
    creds = manager.retrieve_credentials("test_pat_fg")

    assert isinstance(creds, PersonalAccessToken)
    assert creds.token_prefix == "github_"
    assert creds.token == token


def test_metadata_serialization_with_new_fields(
    temp_credentials_dir, mock_keyring, mock_audit_logger
):
    """Test that new fields are properly serialized in metadata."""
    manager = GitHubCredentialManager(
        metadata_path=temp_credentials_dir,
        keyring_module=mock_keyring,
        audit_logger=mock_audit_logger,
    )

    expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    tokens = OAuthTokens(
        access_token="gho_serialization_test",
        scopes=["repo"],
        expires_at=expires_at,
        username="testuser",
    )
    manager.store_oauth_tokens(credential_id="test_serialize", tokens=tokens)

    # Load raw metadata from file
    raw_metadata = manager._load_metadata()
    cred_data = raw_metadata["test_serialize"]

    # Verify fields are in serialized form
    assert cred_data["username"] == "testuser"
    assert cred_data["expires_at"] is not None
    assert "T" in cred_data["expires_at"]  # ISO format timestamp
