"""Shared fixtures and mocks for GitHub connector tests."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock

import pytest

from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    PrivacyLevel,
    RepositoryPrivacySettings,
    RepositoryRegistry,
    SyncMode,
    VisibilityType,
)


# ---------------------------------------------------------------------------
# Mock keyring
# ---------------------------------------------------------------------------


class MockKeyring:
    """Mock keyring for testing credential storage."""

    def __init__(self):
        self.storage = {}

    def set_password(self, service: str, username: str, password: str):
        key = f"{service}:{username}"
        self.storage[key] = password

    def get_password(self, service: str, username: str) -> Optional[str]:
        key = f"{service}:{username}"
        return self.storage.get(key)

    def delete_password(self, service: str, username: str):
        key = f"{service}:{username}"
        if key in self.storage:
            del self.storage[key]


@pytest.fixture
def mock_keyring():
    """Provide mock keyring for testing."""
    return MockKeyring()


# ---------------------------------------------------------------------------
# Temporary directories
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_registry_dir(tmp_path):
    """Provide temporary directory for repository registry."""
    registry_dir = tmp_path / "sources" / "github"
    registry_dir.mkdir(parents=True)
    return registry_dir


@pytest.fixture
def temp_credentials_dir(tmp_path):
    """Provide temporary directory for credentials metadata."""
    creds_dir = tmp_path / "credentials"
    creds_dir.mkdir(parents=True)
    return creds_dir / "github.json"


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repo_descriptor():
    """Provide sample repository descriptor for testing."""
    return GitHubRepositoryDescriptor.from_registration(
        owner="octocat",
        repo="Hello-World",
        github_host="github.com",
        credential_id="github_cred_test123",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GRAPHQL_API,
        branches=["main", "develop"],
        privacy_settings=RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.STANDARD,
        ),
    )


@pytest.fixture
def sample_private_repo_descriptor():
    """Provide sample private repository descriptor."""
    return GitHubRepositoryDescriptor.from_registration(
        owner="myorg",
        repo="private-repo",
        github_host="github.com",
        credential_id="github_cred_private123",
        visibility=VisibilityType.PRIVATE,
        sync_mode=SyncMode.GRAPHQL_API,
        branches=["main"],
        privacy_settings=RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
        ),
    )


# ---------------------------------------------------------------------------
# Mock GitHub API
# ---------------------------------------------------------------------------


class MockGitHubAPI:
    """Mock GitHub API for testing."""

    def __init__(self):
        self.repositories = {}
        self.tokens = {}
        self.rate_limit = 5000
        self.rate_remaining = 5000
        self.call_log = []

    def add_repository(
        self,
        owner: str,
        repo: str,
        *,
        visibility: str = "public",
        default_branch: str = "main",
        archived: bool = False,
    ):
        """Add a repository to mock API."""
        self.repositories[f"{owner}/{repo}"] = {
            "owner": {"login": owner},
            "name": repo,
            "full_name": f"{owner}/{repo}",
            "description": f"Test repository {owner}/{repo}",
            "private": visibility == "private",
            "visibility": visibility,
            "default_branch": default_branch,
            "fork": False,
            "archived": archived,
            "created_at": "2020-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "pushed_at": "2023-06-01T00:00:00Z",
            "size": 1024,
            "language": "Python",
            "has_issues": True,
            "has_wiki": True,
        }

    def add_token(self, token: str, scopes: list[str]):
        """Add a token to mock API."""
        self.tokens[token] = {
            "scopes": scopes,
            "rate_limit": self.rate_limit,
            "rate_remaining": self.rate_remaining,
        }

    def get_repository(self, owner: str, repo: str, token: str) -> Dict[str, Any]:
        """Mock GET /repos/:owner/:repo."""
        self.call_log.append(("GET", f"/repos/{owner}/{repo}"))

        if token not in self.tokens:
            raise Exception("401 Unauthorized")

        full_name = f"{owner}/{repo}"
        if full_name not in self.repositories:
            raise Exception("404 Not Found")

        return self.repositories[full_name]

    def get_rate_limit(self, token: str) -> Dict[str, Any]:
        """Mock GET /rate_limit."""
        self.call_log.append(("GET", "/rate_limit"))

        if token not in self.tokens:
            raise Exception("401 Unauthorized")

        token_info = self.tokens[token]
        return {
            "rate": {
                "limit": token_info["rate_limit"],
                "remaining": token_info["rate_remaining"],
                "reset": int(datetime.now(timezone.utc).timestamp()) + 3600,
            }
        }

    def list_branches(self, owner: str, repo: str, token: str) -> list[Dict[str, Any]]:
        """Mock GET /repos/:owner/:repo/branches."""
        self.call_log.append(("GET", f"/repos/{owner}/{repo}/branches"))

        if token not in self.tokens:
            raise Exception("401 Unauthorized")

        full_name = f"{owner}/{repo}"
        if full_name not in self.repositories:
            raise Exception("404 Not Found")

        repo_data = self.repositories[full_name]
        default_branch = repo_data.get("default_branch", "main")

        return [
            {
                "name": default_branch,
                "commit": {"sha": "abc123def456"},
                "protected": True,
            },
            {
                "name": "develop",
                "commit": {"sha": "def456abc789"},
                "protected": False,
            },
        ]


@pytest.fixture
def mock_github_api():
    """Provide mock GitHub API for testing."""
    api = MockGitHubAPI()

    # Add some default repositories
    api.add_repository("octocat", "Hello-World", visibility="public")
    api.add_repository("myorg", "private-repo", visibility="private")
    api.add_repository("testuser", "archived-repo", visibility="public", archived=True)

    # Add some default tokens
    api.add_token("ghp_validtoken123456789012345678901234", ["repo", "user"])
    api.add_token("ghp_publictoken123456789012345678901", ["public_repo"])

    return api


# ---------------------------------------------------------------------------
# Mock OAuth provider
# ---------------------------------------------------------------------------


class MockOAuthProvider:
    """Mock OAuth provider for device flow testing."""

    def __init__(self):
        self.device_codes = {}
        self.access_tokens = {}

    def create_device_code(
        self, client_id: str, scopes: list[str]
    ) -> Dict[str, Any]:
        """Mock device code creation."""
        device_code = f"device_{len(self.device_codes)}"
        user_code = f"ABCD-{len(self.device_codes):04d}"

        self.device_codes[device_code] = {
            "client_id": client_id,
            "scopes": scopes,
            "user_code": user_code,
            "status": "pending",
            "access_token": None,
        }

        return {
            "device_code": device_code,
            "user_code": user_code,
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 5,
        }

    def authorize_device(self, device_code: str, access_token: str):
        """Authorize a device code."""
        if device_code in self.device_codes:
            self.device_codes[device_code]["status"] = "authorized"
            self.device_codes[device_code]["access_token"] = access_token

    def check_device_authorization(self, device_code: str) -> Dict[str, Any]:
        """Check device authorization status."""
        if device_code not in self.device_codes:
            return {"error": "expired_token"}

        code_data = self.device_codes[device_code]

        if code_data["status"] == "pending":
            return {"error": "authorization_pending"}
        elif code_data["status"] == "authorized":
            return {
                "access_token": code_data["access_token"],
                "token_type": "bearer",
                "scope": " ".join(code_data["scopes"]),
            }
        else:
            return {"error": "access_denied"}


@pytest.fixture
def mock_oauth_provider():
    """Provide mock OAuth provider for testing."""
    return MockOAuthProvider()


# ---------------------------------------------------------------------------
# Mock audit logger
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_audit_logger():
    """Provide mock audit logger for testing."""
    logger = MagicMock()
    logger.record = MagicMock()
    return logger


# ---------------------------------------------------------------------------
# Registry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_registry(temp_registry_dir, mock_audit_logger):
    """Provide test repository registry."""
    return RepositoryRegistry(
        registry_root=temp_registry_dir, audit_logger=mock_audit_logger
    )
