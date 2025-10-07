"""Tests for GitHub pull request normalizer.

Tests cover:
- GraphQL query construction with review and file data
- PR data parsing from GraphQL responses
- Review status tracking (approved, changes requested)
- Merger and reviewer tracking
- File modification list extraction
- Code change metrics
- Branch information
- Error handling (404, invalid responses)
- State determination (open, closed, merged)
- Mock GraphQL responses
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from futurnal.ingestion.github.pr_normalizer import PullRequestNormalizer
from futurnal.ingestion.github.issue_pr_models import PRState, ReviewDecision


@pytest.fixture
def mock_api_client():
    """Create mock API client manager."""
    client = MagicMock()
    client.graphql_request = AsyncMock()
    return client


@pytest.fixture
def pr_normalizer(mock_api_client):
    """Create PR normalizer with mock API client."""
    return PullRequestNormalizer(
        api_client=mock_api_client,
        extract_reviews=True,
        extract_files=True,
        max_files=100,
    )


@pytest.fixture
def sample_pr_response():
    """Sample GraphQL response for a PR."""
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "number": 123,
                    "title": "Add new feature",
                    "body": "This PR adds a new feature. Closes #42.",
                    "state": "OPEN",
                    "createdAt": "2024-01-15T10:00:00Z",
                    "updatedAt": "2024-01-15T12:00:00Z",
                    "closedAt": None,
                    "mergedAt": None,
                    "url": "https://github.com/octocat/Hello-World/pull/123",
                    "author": {
                        "login": "contributor",
                        "name": "Contributor Name",
                        "email": "contributor@example.com",
                        "avatarUrl": "https://github.com/images/contributor.gif",
                        "url": "https://github.com/contributor",
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
                                "name": "feature",
                                "color": "0e8a16",
                                "description": "New feature",
                            }
                        ]
                    },
                    "milestone": {
                        "title": "v2.0",
                        "description": "Second release",
                        "dueOn": "2024-06-30T23:59:59Z",
                        "state": "OPEN",
                    },
                    "reviewDecision": "APPROVED",
                    "changedFiles": 5,
                    "additions": 150,
                    "deletions": 30,
                    "commits": {"totalCount": 3},
                    "baseRefName": "main",
                    "headRefName": "feature-branch",
                    "mergedBy": None,
                    "comments": {"totalCount": 2},
                    "reviews": {
                        "totalCount": 2,
                        "nodes": [
                            {
                                "author": {
                                    "login": "reviewer1",
                                    "name": "Reviewer One",
                                    "url": "https://github.com/reviewer1",
                                },
                                "state": "APPROVED",
                                "body": "Looks good!",
                                "createdAt": "2024-01-15T11:00:00Z",
                            },
                            {
                                "author": {
                                    "login": "reviewer2",
                                    "name": "Reviewer Two",
                                    "url": "https://github.com/reviewer2",
                                },
                                "state": "APPROVED",
                                "body": "LGTM",
                                "createdAt": "2024-01-15T11:30:00Z",
                            },
                        ],
                    },
                    "reviewRequests": {
                        "nodes": [
                            {
                                "requestedReviewer": {
                                    "login": "pending_reviewer",
                                    "name": "Pending Reviewer",
                                    "url": "https://github.com/pending_reviewer",
                                }
                            }
                        ]
                    },
                    "files": {
                        "totalCount": 5,
                        "nodes": [
                            {"path": "src/main.py", "additions": 50, "deletions": 10},
                            {"path": "src/utils.py", "additions": 30, "deletions": 5},
                            {"path": "tests/test_main.py", "additions": 40, "deletions": 10},
                            {"path": "README.md", "additions": 20, "deletions": 3},
                            {"path": "requirements.txt", "additions": 10, "deletions": 2},
                        ],
                    },
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# Query Building Tests
# ---------------------------------------------------------------------------


def test_build_pr_query_with_reviews(pr_normalizer):
    """Test GraphQL query includes reviews when requested."""
    query = pr_normalizer._build_pr_query(include_reviews=True, include_files=True)

    assert "reviews(first: 20)" in query
    assert "reviewRequests(first: 10)" in query
    assert "files(first: $maxFiles)" in query
    assert "$maxFiles: Int!" in query


def test_build_pr_query_without_reviews(pr_normalizer):
    """Test GraphQL query excludes reviews when not requested."""
    normalizer = PullRequestNormalizer(
        api_client=MagicMock(),
        extract_reviews=False,
        extract_files=False,
    )
    query = normalizer._build_pr_query(include_reviews=False, include_files=False)

    assert "reviews" not in query
    assert "files" not in query


# ---------------------------------------------------------------------------
# PR Normalization Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_pr_success(
    pr_normalizer,
    mock_api_client,
    sample_pr_response,
):
    """Test successful PR normalization."""
    mock_api_client.graphql_request.return_value = sample_pr_response

    pr = await pr_normalizer.normalize_pull_request(
        repo_owner="octocat",
        repo_name="Hello-World",
        pr_number=123,
        credential_id="test_cred",
    )

    # Verify basic properties
    assert pr.pr_number == 123
    assert pr.repo_id == "octocat/Hello-World"
    assert pr.title == "Add new feature"
    assert pr.state == PRState.OPEN
    assert pr.author.login == "contributor"

    # Verify code changes
    assert pr.changed_files == 5
    assert pr.additions == 150
    assert pr.deletions == 30
    assert pr.commits == 3
    assert pr.total_changes == 180

    # Verify branches
    assert pr.base_branch == "main"
    assert pr.head_branch == "feature-branch"

    # Verify review status
    assert pr.review_decision == ReviewDecision.APPROVED
    assert len(pr.approved_by) == 2
    assert pr.is_approved is True

    # Verify reviewers
    assert len(pr.reviewers) == 1
    assert pr.reviewers[0].login == "pending_reviewer"

    # Verify files
    assert len(pr.modified_files) == 5
    assert "src/main.py" in pr.modified_files
    assert "tests/test_main.py" in pr.modified_files


@pytest.mark.asyncio
async def test_normalize_merged_pr(
    pr_normalizer,
    mock_api_client,
    sample_pr_response,
):
    """Test normalizing merged PR."""
    # Modify response for merged PR
    sample_pr_response["data"]["repository"]["pullRequest"]["state"] = "MERGED"
    sample_pr_response["data"]["repository"]["pullRequest"]["mergedAt"] = "2024-01-15T13:00:00Z"
    sample_pr_response["data"]["repository"]["pullRequest"]["closedAt"] = "2024-01-15T13:00:00Z"
    sample_pr_response["data"]["repository"]["pullRequest"]["mergedBy"] = {
        "login": "maintainer",
        "name": "Maintainer",
        "url": "https://github.com/maintainer",
    }

    mock_api_client.graphql_request.return_value = sample_pr_response

    pr = await pr_normalizer.normalize_pull_request(
        repo_owner="octocat",
        repo_name="Hello-World",
        pr_number=123,
        credential_id="test_cred",
    )

    assert pr.state == PRState.MERGED
    assert pr.is_merged is True
    assert pr.merged_at is not None
    assert pr.merged_by is not None
    assert pr.merged_by.login == "maintainer"


@pytest.mark.asyncio
async def test_normalize_pr_with_changes_requested(
    pr_normalizer,
    mock_api_client,
    sample_pr_response,
):
    """Test normalizing PR with changes requested."""
    # Modify response for changes requested
    sample_pr_response["data"]["repository"]["pullRequest"]["reviewDecision"] = "CHANGES_REQUESTED"
    sample_pr_response["data"]["repository"]["pullRequest"]["reviews"]["nodes"] = [
        {
            "author": {"login": "reviewer1", "name": "Reviewer One", "url": "https://github.com/reviewer1"},
            "state": "CHANGES_REQUESTED",
            "body": "Please fix formatting",
            "createdAt": "2024-01-15T11:00:00Z",
        }
    ]

    mock_api_client.graphql_request.return_value = sample_pr_response

    pr = await pr_normalizer.normalize_pull_request(
        repo_owner="octocat",
        repo_name="Hello-World",
        pr_number=123,
        credential_id="test_cred",
    )

    assert pr.review_decision == ReviewDecision.CHANGES_REQUESTED
    assert len(pr.changes_requested_by) == 1
    assert pr.changes_requested_by[0].login == "reviewer1"


@pytest.mark.asyncio
async def test_normalize_pr_repository_not_found(
    pr_normalizer,
    mock_api_client,
):
    """Test error handling when repository not found."""
    mock_api_client.graphql_request.return_value = {
        "data": {"repository": None}
    }

    with pytest.raises(ValueError, match="Repository not found"):
        await pr_normalizer.normalize_pull_request(
            repo_owner="nonexistent",
            repo_name="repo",
            pr_number=1,
            credential_id="test_cred",
        )


@pytest.mark.asyncio
async def test_normalize_pr_not_found(
    pr_normalizer,
    mock_api_client,
):
    """Test error handling when PR not found."""
    mock_api_client.graphql_request.return_value = {
        "data": {"repository": {"pullRequest": None}}
    }

    with pytest.raises(ValueError, match="PR not found"):
        await pr_normalizer.normalize_pull_request(
            repo_owner="octocat",
            repo_name="Hello-World",
            pr_number=999,
            credential_id="test_cred",
        )


@pytest.mark.asyncio
async def test_normalize_pr_invalid_response(
    pr_normalizer,
    mock_api_client,
):
    """Test error handling with invalid GraphQL response."""
    mock_api_client.graphql_request.return_value = {"errors": ["Some error"]}

    with pytest.raises(ValueError, match="Invalid GraphQL response"):
        await pr_normalizer.normalize_pull_request(
            repo_owner="octocat",
            repo_name="Hello-World",
            pr_number=1,
            credential_id="test_cred",
        )


# ---------------------------------------------------------------------------
# Parsing Tests
# ---------------------------------------------------------------------------


def test_parse_user_complete(pr_normalizer):
    """Test parsing user with complete data."""
    user_data = {
        "login": "developer",
        "name": "Developer Name",
        "email": "dev@example.com",
        "avatarUrl": "https://github.com/images/dev.gif",
        "url": "https://github.com/developer",
    }

    user = pr_normalizer._parse_user(user_data)

    assert user.login == "developer"
    assert user.name == "Developer Name"
    assert user.email == "dev@example.com"
    assert user.github_url == "https://github.com/developer"


def test_parse_user_minimal(pr_normalizer):
    """Test parsing user with minimal data."""
    user_data = {"login": "minimal"}

    user = pr_normalizer._parse_user(user_data)

    assert user.login == "minimal"
    assert user.name is None
    assert user.github_url == "https://github.com/minimal"


def test_parse_datetime_valid(pr_normalizer):
    """Test datetime parsing with valid ISO string."""
    dt = pr_normalizer._parse_datetime("2024-01-15T10:00:00Z")

    assert dt is not None
    assert dt.year == 2024
    assert dt.tzinfo == timezone.utc


def test_parse_datetime_none(pr_normalizer):
    """Test datetime parsing with None."""
    dt = pr_normalizer._parse_datetime(None)

    assert dt is None


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_pr_without_optional_fields(
    pr_normalizer,
    mock_api_client,
):
    """Test normalizing PR with minimal fields."""
    minimal_response = {
        "data": {
            "repository": {
                "pullRequest": {
                    "number": 1,
                    "title": "Minimal PR",
                    "body": None,
                    "state": "OPEN",
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-01T00:00:00Z",
                    "closedAt": None,
                    "mergedAt": None,
                    "url": "https://github.com/octocat/Hello-World/pull/1",
                    "author": {"login": "ghost"},
                    "assignees": {"nodes": []},
                    "labels": {"nodes": []},
                    "milestone": None,
                    "reviewDecision": None,
                    "changedFiles": 1,
                    "additions": 10,
                    "deletions": 5,
                    "commits": {"totalCount": 1},
                    "baseRefName": "main",
                    "headRefName": "fix",
                    "mergedBy": None,
                    "comments": {"totalCount": 0},
                    "reviews": {"nodes": []},
                    "reviewRequests": {"nodes": []},
                    "files": {"nodes": []},
                }
            }
        }
    }

    mock_api_client.graphql_request.return_value = minimal_response

    pr = await pr_normalizer.normalize_pull_request(
        repo_owner="octocat",
        repo_name="Hello-World",
        pr_number=1,
        credential_id="test_cred",
    )

    assert pr.pr_number == 1
    assert pr.body is None
    assert len(pr.assignees) == 0
    assert len(pr.labels) == 0
    assert pr.milestone is None
    assert len(pr.reviewers) == 0
    assert len(pr.modified_files) == 0


@pytest.mark.asyncio
async def test_normalize_closed_unmerged_pr(
    pr_normalizer,
    mock_api_client,
    sample_pr_response,
):
    """Test normalizing closed but unmerged PR."""
    sample_pr_response["data"]["repository"]["pullRequest"]["state"] = "CLOSED"
    sample_pr_response["data"]["repository"]["pullRequest"]["closedAt"] = "2024-01-15T13:00:00Z"
    sample_pr_response["data"]["repository"]["pullRequest"]["mergedAt"] = None

    mock_api_client.graphql_request.return_value = sample_pr_response

    pr = await pr_normalizer.normalize_pull_request(
        repo_owner="octocat",
        repo_name="Hello-World",
        pr_number=123,
        credential_id="test_cred",
    )

    assert pr.state == PRState.CLOSED
    assert pr.is_merged is False
    assert pr.closed_at is not None
    assert pr.merged_at is None


@pytest.mark.asyncio
async def test_normalize_pr_without_files(mock_api_client):
    """Test normalizing PR without fetching files."""
    normalizer = PullRequestNormalizer(
        api_client=mock_api_client,
        extract_reviews=True,
        extract_files=False,
    )

    minimal_response = {
        "data": {
            "repository": {
                "pullRequest": {
                    "number": 1,
                    "title": "Test",
                    "state": "OPEN",
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-01T00:00:00Z",
                    "url": "https://github.com/octocat/Hello-World/pull/1",
                    "author": {"login": "user"},
                    "assignees": {"nodes": []},
                    "labels": {"nodes": []},
                    "milestone": None,
                    "reviewDecision": None,
                    "changedFiles": 1,
                    "additions": 10,
                    "deletions": 5,
                    "commits": {"totalCount": 1},
                    "baseRefName": "main",
                    "headRefName": "fix",
                    "mergedBy": None,
                    "comments": {"totalCount": 0},
                    "reviews": {"nodes": []},
                    "reviewRequests": {"nodes": []},
                }
            }
        }
    }

    mock_api_client.graphql_request.return_value = minimal_response

    pr = await normalizer.normalize_pull_request(
        repo_owner="octocat",
        repo_name="Hello-World",
        pr_number=1,
        credential_id="test_cred",
    )

    assert len(pr.modified_files) == 0


@pytest.mark.asyncio
async def test_normalize_pr_with_reference_extraction(
    pr_normalizer,
    mock_api_client,
    sample_pr_response,
):
    """Test PR normalization extracts references from body."""
    mock_api_client.graphql_request.return_value = sample_pr_response

    pr = await pr_normalizer.normalize_pull_request(
        repo_owner="octocat",
        repo_name="Hello-World",
        pr_number=123,
        credential_id="test_cred",
    )

    # Verify reference extraction was called
    assert 42 in pr.referenced_issues


@pytest.mark.asyncio
async def test_normalize_pr_review_states(
    pr_normalizer,
    mock_api_client,
    sample_pr_response,
):
    """Test parsing different review states."""
    # Mix of approved and changes requested
    sample_pr_response["data"]["repository"]["pullRequest"]["reviews"]["nodes"] = [
        {
            "author": {"login": "approver", "url": "https://github.com/approver"},
            "state": "APPROVED",
            "body": "LGTM",
            "createdAt": "2024-01-15T11:00:00Z",
        },
        {
            "author": {"login": "requester", "url": "https://github.com/requester"},
            "state": "CHANGES_REQUESTED",
            "body": "Please fix",
            "createdAt": "2024-01-15T11:30:00Z",
        },
    ]

    mock_api_client.graphql_request.return_value = sample_pr_response

    pr = await pr_normalizer.normalize_pull_request(
        repo_owner="octocat",
        repo_name="Hello-World",
        pr_number=123,
        credential_id="test_cred",
    )

    assert len(pr.approved_by) == 1
    assert pr.approved_by[0].login == "approver"
    assert len(pr.changes_requested_by) == 1
    assert pr.changes_requested_by[0].login == "requester"
