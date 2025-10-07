"""Tests for GitHub issue and PR data models.

Tests cover:
- IssueMetadata validation and defaults
- PullRequestMetadata validation and defaults
- GitHubUser model
- Label validation (color format)
- Milestone validation
- Comment model
- Reaction model
- Datetime timezone handling
- Reference extraction helpers
- Property helpers (is_open, is_merged, etc.)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from futurnal.ingestion.github.issue_pr_models import (
    Comment,
    GitHubUser,
    IssueMetadata,
    IssueState,
    Label,
    Milestone,
    PRState,
    PullRequestMetadata,
    Reaction,
    ReviewDecision,
)


# ---------------------------------------------------------------------------
# GitHubUser Tests
# ---------------------------------------------------------------------------


def test_github_user_basic():
    """Test basic GitHubUser creation."""
    user = GitHubUser(
        login="octocat",
        name="The Octocat",
        email="octocat@github.com",
        avatar_url="https://github.com/images/error/octocat_happy.gif",
        github_url="https://github.com/octocat",
    )

    assert user.login == "octocat"
    assert user.name == "The Octocat"
    assert user.email == "octocat@github.com"
    assert user.display_identifier == "The Octocat"


def test_github_user_auto_url():
    """Test GitHubUser auto-generates URL from login."""
    user = GitHubUser(
        login="testuser",
        github_url="",  # Will be auto-generated
    )

    assert user.github_url == "https://github.com/testuser"


def test_github_user_display_identifier_fallback():
    """Test display_identifier falls back to login when name is None."""
    user = GitHubUser(
        login="testuser",
        name=None,
        github_url="https://github.com/testuser",
    )

    assert user.display_identifier == "testuser"


# ---------------------------------------------------------------------------
# Label Tests
# ---------------------------------------------------------------------------


def test_label_valid_color():
    """Test Label with valid hex color."""
    label = Label(
        name="bug",
        color="d73a4a",
        description="Something isn't working",
    )

    assert label.name == "bug"
    assert label.color == "d73a4a"
    assert label.description == "Something isn't working"


def test_label_color_normalization():
    """Test Label normalizes color with # prefix."""
    label = Label(
        name="feature",
        color="#0e8a16",  # With # prefix
    )

    assert label.color == "0e8a16"  # Without # prefix


def test_label_invalid_color():
    """Test Label rejects invalid hex color."""
    with pytest.raises(ValueError, match="Invalid color hex"):
        Label(name="invalid", color="ZZZZZZ")


# ---------------------------------------------------------------------------
# Milestone Tests
# ---------------------------------------------------------------------------


def test_milestone_basic():
    """Test basic Milestone creation."""
    milestone = Milestone(
        title="v1.0",
        description="First release",
        due_on=datetime(2024, 12, 31, tzinfo=timezone.utc),
        state="open",
    )

    assert milestone.title == "v1.0"
    assert milestone.description == "First release"
    assert milestone.state == "open"
    assert milestone.due_on.tzinfo is not None


def test_milestone_state_validation():
    """Test Milestone validates state."""
    with pytest.raises(ValueError, match="Invalid milestone state"):
        Milestone(title="v1.0", state="invalid")


def test_milestone_timezone_aware():
    """Test Milestone ensures timezone-aware datetime."""
    milestone = Milestone(
        title="v1.0",
        due_on=datetime(2024, 12, 31),  # Naive datetime
        state="open",
    )

    assert milestone.due_on.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Reaction Tests
# ---------------------------------------------------------------------------


def test_reaction_valid():
    """Test valid Reaction creation."""
    reaction = Reaction(content="+1", count=5)

    assert reaction.content == "+1"
    assert reaction.count == 5


def test_reaction_invalid_content():
    """Test Reaction rejects invalid content."""
    with pytest.raises(ValueError, match="Invalid reaction"):
        Reaction(content="invalid_emoji", count=1)


# ---------------------------------------------------------------------------
# Comment Tests
# ---------------------------------------------------------------------------


def test_comment_basic():
    """Test basic Comment creation."""
    user = GitHubUser(login="commenter", github_url="https://github.com/commenter")
    comment = Comment(
        comment_id=123,
        author=user,
        body="Great work!",
        created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
        reactions={"+1": 3, "heart": 1},
    )

    assert comment.comment_id == 123
    assert comment.author.login == "commenter"
    assert comment.body == "Great work!"
    assert comment.total_reactions == 4


def test_comment_timezone_aware():
    """Test Comment ensures timezone-aware datetimes."""
    user = GitHubUser(login="user", github_url="https://github.com/user")
    comment = Comment(
        comment_id=1,
        author=user,
        body="Test",
        created_at=datetime(2024, 1, 1),  # Naive
        updated_at=datetime(2024, 1, 1),  # Naive
    )

    assert comment.created_at.tzinfo == timezone.utc
    assert comment.updated_at.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# IssueMetadata Tests
# ---------------------------------------------------------------------------


def test_issue_metadata_basic():
    """Test basic IssueMetadata creation."""
    author = GitHubUser(login="author", github_url="https://github.com/author")
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Fix bug",
        state=IssueState.OPEN,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        author=author,
    )

    assert issue.issue_number == 42
    assert issue.repo_id == "octocat/Hello-World"
    assert issue.title == "Fix bug"
    assert issue.is_open is True
    assert issue.has_assignees is False


def test_issue_metadata_with_relationships():
    """Test IssueMetadata with assignees and labels."""
    author = GitHubUser(login="author", github_url="https://github.com/author")
    assignee = GitHubUser(login="assignee", github_url="https://github.com/assignee")
    label = Label(name="bug", color="d73a4a")

    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Fix bug",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=author,
        assignees=[assignee],
        labels=[label],
    )

    assert issue.has_assignees is True
    assert issue.has_labels is True
    assert len(issue.assignees) == 1
    assert len(issue.labels) == 1


def test_issue_metadata_extract_references():
    """Test IssueMetadata extracts references from body."""
    author = GitHubUser(login="author", github_url="https://github.com/author")
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Fix bug",
        body="Fixes #123 and relates to #456. Thanks @octocat! See https://example.com/doc",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=author,
    )

    issue.extract_references_from_body()

    assert 123 in issue.referenced_issues
    assert 456 in issue.referenced_issues
    assert "octocat" in issue.mentioned_users
    assert "https://example.com/doc" in issue.external_links


def test_issue_metadata_invalid_repo_id():
    """Test IssueMetadata rejects invalid repo_id format."""
    author = GitHubUser(login="author", github_url="https://github.com/author")

    with pytest.raises(ValueError, match="Invalid repo_id format"):
        IssueMetadata(
            issue_number=1,
            repo_id="invalid",  # Missing /
            github_url="https://github.com/invalid/issues/1",
            title="Test",
            state=IssueState.OPEN,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            author=author,
        )


# ---------------------------------------------------------------------------
# PullRequestMetadata Tests
# ---------------------------------------------------------------------------


def test_pr_metadata_basic():
    """Test basic PullRequestMetadata creation."""
    author = GitHubUser(login="author", github_url="https://github.com/author")
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        author=author,
        base_branch="main",
        head_branch="feature-branch",
    )

    assert pr.pr_number == 123
    assert pr.title == "Add feature"
    assert pr.is_open is True
    assert pr.is_merged is False
    assert pr.base_branch == "main"


def test_pr_metadata_merged_state():
    """Test PullRequestMetadata with merged state."""
    author = GitHubUser(login="author", github_url="https://github.com/author")
    merger = GitHubUser(login="merger", github_url="https://github.com/merger")

    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.MERGED,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        merged_at=datetime.now(timezone.utc),
        author=author,
        merged_by=merger,
        base_branch="main",
        head_branch="feature",
    )

    assert pr.is_merged is True
    assert pr.merged_by.login == "merger"


def test_pr_metadata_review_status():
    """Test PullRequestMetadata review status properties."""
    author = GitHubUser(login="author", github_url="https://github.com/author")
    approver = GitHubUser(login="approver", github_url="https://github.com/approver")

    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=author,
        base_branch="main",
        head_branch="feature",
        review_decision=ReviewDecision.APPROVED,
        approved_by=[approver],
    )

    assert pr.is_approved is True
    assert pr.needs_review is False
    assert len(pr.approved_by) == 1


def test_pr_metadata_code_changes():
    """Test PullRequestMetadata code change metrics."""
    author = GitHubUser(login="author", github_url="https://github.com/author")
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=author,
        base_branch="main",
        head_branch="feature",
        changed_files=5,
        additions=150,
        deletions=30,
        commits=3,
    )

    assert pr.changed_files == 5
    assert pr.total_changes == 180
    assert pr.commits == 3


def test_pr_metadata_extract_references():
    """Test PullRequestMetadata extracts references from body."""
    author = GitHubUser(login="author", github_url="https://github.com/author")
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        body="Closes #42, fixes #56. Thanks @reviewer1!",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=author,
        base_branch="main",
        head_branch="feature",
    )

    pr.extract_references_from_body()

    assert 42 in pr.referenced_issues
    assert 56 in pr.referenced_issues
    assert "reviewer1" in pr.mentioned_users
