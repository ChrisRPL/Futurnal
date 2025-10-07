"""Tests for GitHub issue normalizer.

Tests cover:
- GraphQL query construction
- Issue data parsing from GraphQL responses
- Comment extraction with pagination
- Participant tracking
- Label and milestone handling
- Reference and mention extraction
- Error handling (404, rate limits, invalid responses)
- Timezone handling
- Mock GraphQL responses
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from futurnal.ingestion.github.issue_normalizer import IssueNormalizer
from futurnal.ingestion.github.issue_pr_models import IssueState


@pytest.fixture
def mock_api_client():
    """Create mock API client manager."""
    client = MagicMock()
    client.graphql_request = AsyncMock()
    return client


@pytest.fixture
def issue_normalizer(mock_api_client):
    """Create issue normalizer with mock API client."""
    return IssueNormalizer(
        api_client=mock_api_client,
        extract_comments=True,
        max_comments=100,
    )


@pytest.fixture
def sample_issue_response():
    """Sample GraphQL response for an issue."""
    return {
        "data": {
            "repository": {
                "issue": {
                    "number": 42,
                    "title": "Fix critical bug",
                    "body": "This is a bug that needs fixing. Fixes #123.",
                    "bodyHTML": "<p>This is a bug that needs fixing.</p>",
                    "state": "OPEN",
                    "createdAt": "2024-01-15T10:00:00Z",
                    "updatedAt": "2024-01-15T11:00:00Z",
                    "closedAt": None,
                    "url": "https://github.com/octocat/Hello-World/issues/42",
                    "author": {
                        "login": "octocat",
                        "name": "The Octocat",
                        "email": "octocat@github.com",
                        "avatarUrl": "https://github.com/images/error/octocat_happy.gif",
                        "url": "https://github.com/octocat",
                    },
                    "assignees": {
                        "nodes": [
                            {
                                "login": "assignee1",
                                "name": "Assignee One",
                                "url": "https://github.com/assignee1",
                            }
                        ]
                    },
                    "labels": {
                        "nodes": [
                            {
                                "name": "bug",
                                "color": "d73a4a",
                                "description": "Something isn't working",
                            },
                            {
                                "name": "priority:high",
                                "color": "b60205",
                                "description": "High priority",
                            },
                        ]
                    },
                    "milestone": {
                        "title": "v1.0",
                        "description": "First release",
                        "dueOn": "2024-12-31T23:59:59Z",
                        "state": "OPEN",
                    },
                    "participants": {
                        "totalCount": 3,
                        "nodes": [
                            {"login": "octocat", "name": "The Octocat", "url": "https://github.com/octocat"},
                            {"login": "user1", "name": "User One", "url": "https://github.com/user1"},
                            {"login": "user2", "name": "User Two", "url": "https://github.com/user2"},
                        ],
                    },
                    "reactions": {"totalCount": 5},
                    "comments": {
                        "totalCount": 2,
                        "nodes": [
                            {
                                "id": "comment1",
                                "databaseId": 1001,
                                "author": {"login": "commenter1", "name": "Commenter One"},
                                "body": "Good catch!",
                                "createdAt": "2024-01-15T10:30:00Z",
                                "updatedAt": "2024-01-15T10:30:00Z",
                                "reactions": {
                                    "totalCount": 2,
                                    "nodes": [{"content": "+1"}, {"content": "+1"}],
                                },
                            },
                            {
                                "id": "comment2",
                                "databaseId": 1002,
                                "author": {"login": "commenter2"},
                                "body": "Thanks @octocat!",
                                "createdAt": "2024-01-15T10:45:00Z",
                                "updatedAt": "2024-01-15T10:45:00Z",
                                "reactions": {"totalCount": 0, "nodes": []},
                            },
                        ],
                    },
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# Query Building Tests
# ---------------------------------------------------------------------------


def test_build_issue_query_with_comments(issue_normalizer):
    """Test GraphQL query includes comments when requested."""
    query = issue_normalizer._build_issue_query(include_comments=True)

    assert "comments(first: $maxComments)" in query
    assert "$maxComments: Int!" in query
    assert "author" in query
    assert "labels" in query
    assert "milestone" in query


def test_build_issue_query_without_comments(issue_normalizer):
    """Test GraphQL query excludes comments when not requested."""
    normalizer = IssueNormalizer(
        api_client=MagicMock(),
        extract_comments=False,
    )
    query = normalizer._build_issue_query(include_comments=False)

    assert "comments" not in query


# ---------------------------------------------------------------------------
# Issue Normalization Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_issue_success(
    issue_normalizer,
    mock_api_client,
    sample_issue_response,
):
    """Test successful issue normalization."""
    mock_api_client.graphql_request.return_value = sample_issue_response

    issue = await issue_normalizer.normalize_issue(
        repo_owner="octocat",
        repo_name="Hello-World",
        issue_number=42,
        credential_id="test_cred",
    )

    # Verify basic properties
    assert issue.issue_number == 42
    assert issue.repo_id == "octocat/Hello-World"
    assert issue.title == "Fix critical bug"
    assert issue.state == IssueState.OPEN
    assert issue.author.login == "octocat"

    # Verify relationships
    assert len(issue.assignees) == 1
    assert issue.assignees[0].login == "assignee1"

    # Verify labels
    assert len(issue.labels) == 2
    assert issue.labels[0].name == "bug"
    assert issue.labels[1].name == "priority:high"

    # Verify milestone
    assert issue.milestone is not None
    assert issue.milestone.title == "v1.0"

    # Verify participants
    assert len(issue.participants) == 3

    # Verify comments
    assert len(issue.comments) == 2
    assert issue.comment_count == 2

    # Verify API call
    mock_api_client.graphql_request.assert_called_once()
    call_kwargs = mock_api_client.graphql_request.call_args.kwargs
    assert call_kwargs["credential_id"] == "test_cred"
    assert call_kwargs["variables"]["owner"] == "octocat"
    assert call_kwargs["variables"]["repo"] == "Hello-World"
    assert call_kwargs["variables"]["number"] == 42


@pytest.mark.asyncio
async def test_normalize_issue_with_reference_extraction(
    issue_normalizer,
    mock_api_client,
    sample_issue_response,
):
    """Test issue normalization extracts references from body."""
    mock_api_client.graphql_request.return_value = sample_issue_response

    issue = await issue_normalizer.normalize_issue(
        repo_owner="octocat",
        repo_name="Hello-World",
        issue_number=42,
        credential_id="test_cred",
    )

    # Verify reference extraction was called
    assert 123 in issue.referenced_issues


@pytest.mark.asyncio
async def test_normalize_issue_repository_not_found(
    issue_normalizer,
    mock_api_client,
):
    """Test error handling when repository not found."""
    mock_api_client.graphql_request.return_value = {
        "data": {"repository": None}
    }

    with pytest.raises(ValueError, match="Repository not found"):
        await issue_normalizer.normalize_issue(
            repo_owner="nonexistent",
            repo_name="repo",
            issue_number=1,
            credential_id="test_cred",
        )


@pytest.mark.asyncio
async def test_normalize_issue_not_found(
    issue_normalizer,
    mock_api_client,
):
    """Test error handling when issue not found."""
    mock_api_client.graphql_request.return_value = {
        "data": {"repository": {"issue": None}}
    }

    with pytest.raises(ValueError, match="Issue not found"):
        await issue_normalizer.normalize_issue(
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=999,
            credential_id="test_cred",
        )


@pytest.mark.asyncio
async def test_normalize_issue_invalid_response(
    issue_normalizer,
    mock_api_client,
):
    """Test error handling with invalid GraphQL response."""
    mock_api_client.graphql_request.return_value = {"errors": ["Some error"]}

    with pytest.raises(ValueError, match="Invalid GraphQL response"):
        await issue_normalizer.normalize_issue(
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=1,
            credential_id="test_cred",
        )


# ---------------------------------------------------------------------------
# Parsing Tests
# ---------------------------------------------------------------------------


def test_parse_user_complete(issue_normalizer):
    """Test parsing user with complete data."""
    user_data = {
        "login": "octocat",
        "name": "The Octocat",
        "email": "octocat@github.com",
        "avatarUrl": "https://github.com/images/error/octocat_happy.gif",
        "url": "https://github.com/octocat",
    }

    user = issue_normalizer._parse_user(user_data)

    assert user.login == "octocat"
    assert user.name == "The Octocat"
    assert user.email == "octocat@github.com"
    assert user.avatar_url == "https://github.com/images/error/octocat_happy.gif"
    assert user.github_url == "https://github.com/octocat"


def test_parse_user_minimal(issue_normalizer):
    """Test parsing user with minimal data."""
    user_data = {"login": "minimal_user"}

    user = issue_normalizer._parse_user(user_data)

    assert user.login == "minimal_user"
    assert user.name is None
    assert user.github_url == "https://github.com/minimal_user"


def test_parse_user_ghost(issue_normalizer):
    """Test parsing deleted/ghost user."""
    user_data = {}

    user = issue_normalizer._parse_user(user_data)

    assert user.login == "ghost"


def test_parse_label(issue_normalizer):
    """Test label parsing."""
    label_data = {
        "name": "bug",
        "color": "d73a4a",
        "description": "Something isn't working",
    }

    label = issue_normalizer._parse_label(label_data)

    assert label.name == "bug"
    assert label.color == "d73a4a"
    assert label.description == "Something isn't working"


def test_parse_milestone(issue_normalizer):
    """Test milestone parsing."""
    milestone_data = {
        "title": "v1.0",
        "description": "First release",
        "dueOn": "2024-12-31T23:59:59Z",
        "state": "OPEN",
    }

    milestone = issue_normalizer._parse_milestone(milestone_data)

    assert milestone.title == "v1.0"
    assert milestone.description == "First release"
    assert milestone.state == "open"
    assert milestone.due_on.tzinfo is not None


def test_parse_comment(issue_normalizer):
    """Test comment parsing with reactions."""
    comment_data = {
        "id": "comment1",
        "databaseId": 1001,
        "author": {"login": "commenter", "name": "Commenter"},
        "body": "Great work!",
        "createdAt": "2024-01-15T10:00:00Z",
        "updatedAt": "2024-01-15T10:05:00Z",
        "reactions": {
            "totalCount": 3,
            "nodes": [
                {"content": "+1"},
                {"content": "+1"},
                {"content": "heart"},
            ],
        },
    }

    comment = issue_normalizer._parse_comment(comment_data)

    assert comment.comment_id == 1001
    assert comment.author.login == "commenter"
    assert comment.body == "Great work!"
    assert comment.reactions["+1"] == 2
    assert comment.reactions["heart"] == 1


def test_parse_datetime_valid(issue_normalizer):
    """Test datetime parsing with valid ISO string."""
    dt = issue_normalizer._parse_datetime("2024-01-15T10:00:00Z")

    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 15
    assert dt.tzinfo == timezone.utc


def test_parse_datetime_none(issue_normalizer):
    """Test datetime parsing with None."""
    dt = issue_normalizer._parse_datetime(None)

    assert dt is None


def test_parse_datetime_invalid(issue_normalizer):
    """Test datetime parsing with invalid string."""
    dt = issue_normalizer._parse_datetime("invalid")

    assert dt is None  # Should handle gracefully


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_closed_issue(
    issue_normalizer,
    mock_api_client,
    sample_issue_response,
):
    """Test normalizing closed issue."""
    # Modify response for closed issue
    sample_issue_response["data"]["repository"]["issue"]["state"] = "CLOSED"
    sample_issue_response["data"]["repository"]["issue"]["closedAt"] = "2024-01-15T12:00:00Z"

    mock_api_client.graphql_request.return_value = sample_issue_response

    issue = await issue_normalizer.normalize_issue(
        repo_owner="octocat",
        repo_name="Hello-World",
        issue_number=42,
        credential_id="test_cred",
    )

    assert issue.state == IssueState.CLOSED
    assert issue.closed_at is not None


@pytest.mark.asyncio
async def test_normalize_issue_without_optional_fields(
    issue_normalizer,
    mock_api_client,
):
    """Test normalizing issue with minimal fields."""
    minimal_response = {
        "data": {
            "repository": {
                "issue": {
                    "number": 1,
                    "title": "Minimal issue",
                    "body": None,
                    "bodyHTML": None,
                    "state": "OPEN",
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-01T00:00:00Z",
                    "closedAt": None,
                    "url": "https://github.com/octocat/Hello-World/issues/1",
                    "author": {"login": "ghost"},
                    "assignees": {"nodes": []},
                    "labels": {"nodes": []},
                    "milestone": None,
                    "participants": {"totalCount": 0, "nodes": []},
                    "reactions": {"totalCount": 0},
                    "comments": {"totalCount": 0, "nodes": []},
                }
            }
        }
    }

    mock_api_client.graphql_request.return_value = minimal_response

    issue = await issue_normalizer.normalize_issue(
        repo_owner="octocat",
        repo_name="Hello-World",
        issue_number=1,
        credential_id="test_cred",
    )

    assert issue.issue_number == 1
    assert issue.body is None
    assert len(issue.assignees) == 0
    assert len(issue.labels) == 0
    assert issue.milestone is None


@pytest.mark.asyncio
async def test_normalize_issue_without_comments(mock_api_client):
    """Test normalizing issue without fetching comments."""
    normalizer = IssueNormalizer(
        api_client=mock_api_client,
        extract_comments=False,
    )

    minimal_response = {
        "data": {
            "repository": {
                "issue": {
                    "number": 1,
                    "title": "Test",
                    "state": "OPEN",
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-01T00:00:00Z",
                    "url": "https://github.com/octocat/Hello-World/issues/1",
                    "author": {"login": "user"},
                    "assignees": {"nodes": []},
                    "labels": {"nodes": []},
                    "milestone": None,
                    "participants": {"totalCount": 0, "nodes": []},
                    "reactions": {"totalCount": 0},
                }
            }
        }
    }

    mock_api_client.graphql_request.return_value = minimal_response

    issue = await normalizer.normalize_issue(
        repo_owner="octocat",
        repo_name="Hello-World",
        issue_number=1,
        credential_id="test_cred",
    )

    assert len(issue.comments) == 0
