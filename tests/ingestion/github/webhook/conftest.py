"""Shared fixtures and mocks for webhook tests."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    PrivacyLevel,
    RepositoryPrivacySettings,
    RepositoryRegistry,
    SyncMode,
    VisibilityType,
)
from futurnal.ingestion.github.webhook.models import (
    WebhookConfig,
    WebhookEvent,
    WebhookEventType,
)


# ---------------------------------------------------------------------------
# Mock aiohttp Components
# ---------------------------------------------------------------------------


class MockRequest:
    """Mock aiohttp.web.Request for testing."""

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        body: bytes = b"",
        json_data: Optional[Dict[str, Any]] = None,
    ):
        self.headers = headers or {}
        self._body = body
        self._json_data = json_data or {}

    async def read(self) -> bytes:
        """Mock read method."""
        return self._body

    async def json(self) -> Dict[str, Any]:
        """Mock json method."""
        if self._json_data:
            return self._json_data
        return json.loads(self._body.decode("utf-8"))


class MockResponse:
    """Mock aiohttp.web.Response for testing."""

    def __init__(self, status: int = 200, text: str = "", json_data: Optional[Dict[str, Any]] = None):
        self.status = status
        self.text = text
        self._json_data = json_data

    def json(self) -> Dict[str, Any]:
        """Return JSON data."""
        return self._json_data or {}


@pytest.fixture
def mock_request():
    """Provide mock aiohttp request."""
    return MockRequest


@pytest.fixture
def mock_response():
    """Provide mock aiohttp response."""
    return MockResponse


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_webhook_config():
    """Provide sample webhook configuration."""
    return WebhookConfig(
        enabled=True,
        listen_host="127.0.0.1",
        listen_port=8765,
        public_url="https://example.com/webhook/github",
        secret="test-webhook-secret-1234567890",
        verify_signature=True,
        enabled_events=[
            WebhookEventType.PUSH,
            WebhookEventType.PULL_REQUEST,
            WebhookEventType.ISSUES,
        ],
        queue_size=1000,
        process_async=True,
    )


@pytest.fixture
def sample_push_payload():
    """Provide sample GitHub push webhook payload."""
    return {
        "ref": "refs/heads/main",
        "repository": {
            "id": 123456,
            "name": "Hello-World",
            "full_name": "octocat/Hello-World",
            "owner": {
                "name": "octocat",
                "login": "octocat",
            },
        },
        "pusher": {
            "name": "octocat",
            "email": "octocat@github.com",
        },
        "commits": [
            {
                "id": "abc123",
                "message": "Update README",
                "timestamp": "2024-01-15T10:30:00Z",
                "author": {
                    "name": "octocat",
                    "email": "octocat@github.com",
                },
            }
        ],
    }


@pytest.fixture
def sample_pr_payload():
    """Provide sample GitHub pull request webhook payload."""
    return {
        "action": "opened",
        "number": 42,
        "pull_request": {
            "number": 42,
            "state": "open",
            "title": "Fix bug in parser",
            "body": "This PR fixes a critical bug",
            "user": {
                "login": "octocat",
            },
            "head": {
                "ref": "fix-bug",
                "sha": "abc123",
            },
            "base": {
                "ref": "main",
                "sha": "def456",
            },
        },
        "repository": {
            "full_name": "octocat/Hello-World",
        },
    }


@pytest.fixture
def sample_issue_payload():
    """Provide sample GitHub issue webhook payload."""
    return {
        "action": "opened",
        "issue": {
            "number": 123,
            "state": "open",
            "title": "Bug report",
            "body": "Found a bug in the code",
            "user": {
                "login": "octocat",
            },
        },
        "repository": {
            "full_name": "octocat/Hello-World",
        },
    }


@pytest.fixture
def sample_webhook_event(sample_push_payload):
    """Provide sample webhook event."""
    return WebhookEvent(
        event_id="12345-67890-abcdef",
        event_type=WebhookEventType.PUSH,
        repository="octocat/Hello-World",
        timestamp=datetime.now(timezone.utc),
        payload=sample_push_payload,
    )


@pytest.fixture
def sample_repo_descriptor():
    """Provide sample repository descriptor."""
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


# ---------------------------------------------------------------------------
# Mock Components
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sync_engine():
    """Provide mock IncrementalSyncEngine."""
    engine = AsyncMock()
    engine.sync_repository = AsyncMock()
    return engine


@pytest.fixture
def mock_issue_normalizer():
    """Provide mock IssueNormalizer."""
    normalizer = AsyncMock()
    normalizer.normalize_issue = AsyncMock()
    return normalizer


@pytest.fixture
def mock_pr_normalizer():
    """Provide mock PullRequestNormalizer."""
    normalizer = AsyncMock()
    normalizer.normalize_pull_request = AsyncMock()
    return normalizer


@pytest.fixture
def mock_api_client_manager():
    """Provide mock GitHubAPIClientManager."""
    manager = AsyncMock()
    client = AsyncMock()
    client.rest_request = AsyncMock()
    manager.get_client = AsyncMock(return_value=client)
    return manager


@pytest.fixture
def mock_state_manager(tmp_path):
    """Provide mock SyncStateManager."""
    from futurnal.ingestion.github.sync_state_manager import SyncStateManager

    state_dir = tmp_path / "sync_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return SyncStateManager(state_dir=state_dir)


@pytest.fixture
def mock_repository_registry(tmp_path, sample_repo_descriptor):
    """Provide mock RepositoryRegistry with sample data."""
    from futurnal.privacy.audit import AuditLogger

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_logger = AuditLogger(audit_dir)

    registry = RepositoryRegistry(audit_logger=audit_logger)

    # Mock list() method to return sample descriptor
    registry.list = Mock(return_value=[sample_repo_descriptor])

    return registry


@pytest.fixture
def mock_audit_logger(tmp_path):
    """Provide mock AuditLogger."""
    from futurnal.privacy.audit import AuditLogger

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return AuditLogger(audit_dir)


# ---------------------------------------------------------------------------
# Event Loop Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


__all__ = [
    "sample_webhook_config",
    "sample_push_payload",
    "sample_pr_payload",
    "sample_issue_payload",
    "sample_webhook_event",
    "sample_repo_descriptor",
    "mock_sync_engine",
    "mock_issue_normalizer",
    "mock_pr_normalizer",
    "mock_api_client_manager",
    "mock_state_manager",
    "mock_repository_registry",
    "mock_audit_logger",
    "mock_request",
    "mock_response",
]
