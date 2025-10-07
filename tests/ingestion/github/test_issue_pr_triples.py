"""Tests for GitHub issue and PR semantic triple extraction.

Tests cover:
- Issue triple generation
- PR triple generation
- User entity triples
- Label entity triples
- Milestone entity triples
- Comment entity triples
- Review relationship triples
- Reference and mention triples
- URI generation consistency
- Edge cases (missing fields, empty lists)
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
    ReviewDecision,
)
from futurnal.ingestion.github.issue_pr_triples import (
    extract_issue_triples,
    extract_pr_triples,
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _create_sample_user(login: str = "testuser") -> GitHubUser:
    """Create a sample GitHubUser for testing."""
    return GitHubUser(
        login=login,
        name=f"{login.capitalize()} Name",
        email=f"{login}@example.com",
        github_url=f"https://github.com/{login}",
    )


def _create_sample_label(name: str = "bug") -> Label:
    """Create a sample Label for testing."""
    return Label(
        name=name,
        color="d73a4a",
        description=f"{name.capitalize()} label",
    )


def _create_sample_milestone(title: str = "v1.0") -> Milestone:
    """Create a sample Milestone for testing."""
    return Milestone(
        title=title,
        description=f"{title} release",
        due_on=datetime(2024, 12, 31, tzinfo=timezone.utc),
        state="open",
    )


def _create_sample_comment(comment_id: int = 1) -> Comment:
    """Create a sample Comment for testing."""
    return Comment(
        comment_id=comment_id,
        author=_create_sample_user("commenter"),
        body="Great work!",
        created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Issue Triple Extraction Tests
# ---------------------------------------------------------------------------


def test_extract_issue_basic_triples():
    """Test extraction of basic issue triples."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Fix bug",
        state=IssueState.OPEN,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        author=_create_sample_user("author"),
    )

    triples = extract_issue_triples(issue)

    # Convert to set for easier testing
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check issue type
    assert ("github:issue/octocat/Hello-World/42", "rdf:type", "github:Issue") in triple_tuples

    # Check title
    assert ("github:issue/octocat/Hello-World/42", "dc:title", "Fix bug") in triple_tuples

    # Check state
    assert ("github:issue/octocat/Hello-World/42", "github:state", "open") in triple_tuples

    # Check number
    assert ("github:issue/octocat/Hello-World/42", "github:number", "42") in triple_tuples

    # Check repository relationship
    assert ("github:issue/octocat/Hello-World/42", "github:inRepository", "github:repo/octocat/Hello-World") in triple_tuples


def test_extract_issue_author_triples():
    """Test extraction of issue author triples."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("octocat"),
    )

    triples = extract_issue_triples(issue)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check author relationship
    assert ("github:issue/octocat/Hello-World/42", "github:createdBy", "github:user/octocat") in triple_tuples

    # Check user entity
    assert ("github:user/octocat", "rdf:type", "github:User") in triple_tuples
    assert ("github:user/octocat", "github:login", "octocat") in triple_tuples


def test_extract_issue_assignee_triples():
    """Test extraction of issue assignee triples."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        assignees=[_create_sample_user("assignee1"), _create_sample_user("assignee2")],
    )

    triples = extract_issue_triples(issue)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check assignee relationships
    assert ("github:issue/octocat/Hello-World/42", "github:assignedTo", "github:user/assignee1") in triple_tuples
    assert ("github:issue/octocat/Hello-World/42", "github:assignedTo", "github:user/assignee2") in triple_tuples


def test_extract_issue_label_triples():
    """Test extraction of issue label triples."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        labels=[_create_sample_label("bug"), _create_sample_label("urgent")],
    )

    triples = extract_issue_triples(issue)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check label relationships
    assert ("github:issue/octocat/Hello-World/42", "github:hasLabel", "github:label/octocat/Hello-World/bug") in triple_tuples
    assert ("github:issue/octocat/Hello-World/42", "github:hasLabel", "github:label/octocat/Hello-World/urgent") in triple_tuples

    # Check label entities
    assert ("github:label/octocat/Hello-World/bug", "rdf:type", "github:Label") in triple_tuples
    assert ("github:label/octocat/Hello-World/bug", "github:labelName", "bug") in triple_tuples


def test_extract_issue_milestone_triples():
    """Test extraction of issue milestone triples."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        milestone=_create_sample_milestone("v1.0"),
    )

    triples = extract_issue_triples(issue)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check milestone relationship
    assert ("github:issue/octocat/Hello-World/42", "github:inMilestone", "github:milestone/octocat/Hello-World/v1.0") in triple_tuples

    # Check milestone entity
    assert ("github:milestone/octocat/Hello-World/v1.0", "rdf:type", "github:Milestone") in triple_tuples
    assert ("github:milestone/octocat/Hello-World/v1.0", "dc:title", "v1.0") in triple_tuples


def test_extract_issue_temporal_triples():
    """Test extraction of issue temporal triples."""
    created = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    updated = datetime(2024, 1, 2, 11, 0, 0, tzinfo=timezone.utc)
    closed = datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc)

    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.CLOSED,
        created_at=created,
        updated_at=updated,
        closed_at=closed,
        author=_create_sample_user("author"),
    )

    triples = extract_issue_triples(issue)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check temporal triples
    assert ("github:issue/octocat/Hello-World/42", "dc:created", created.isoformat()) in triple_tuples
    assert ("github:issue/octocat/Hello-World/42", "dc:modified", updated.isoformat()) in triple_tuples
    assert ("github:issue/octocat/Hello-World/42", "github:closedAt", closed.isoformat()) in triple_tuples


def test_extract_issue_engagement_triples():
    """Test extraction of issue engagement metrics triples."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        comment_count=5,
        reaction_count=10,
    )

    triples = extract_issue_triples(issue)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check engagement metrics
    assert ("github:issue/octocat/Hello-World/42", "github:commentCount", "5") in triple_tuples
    assert ("github:issue/octocat/Hello-World/42", "github:reactionCount", "10") in triple_tuples


def test_extract_issue_comment_triples():
    """Test extraction of issue comment triples."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        comments=[_create_sample_comment(1), _create_sample_comment(2)],
    )

    triples = extract_issue_triples(issue)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check comment triples
    comment_uri = "github:issue/octocat/Hello-World/42/comment/1"
    assert (comment_uri, "rdf:type", "github:Comment") in triple_tuples
    assert (comment_uri, "github:commentOn", "github:issue/octocat/Hello-World/42") in triple_tuples


def test_extract_issue_reference_triples():
    """Test extraction of issue reference triples."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        referenced_issues=[10, 20],
        mentioned_users=["user1", "user2"],
    )

    triples = extract_issue_triples(issue)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check reference triples
    assert ("github:issue/octocat/Hello-World/42", "github:references", "github:issue/octocat/Hello-World/10") in triple_tuples
    assert ("github:issue/octocat/Hello-World/42", "github:references", "github:issue/octocat/Hello-World/20") in triple_tuples

    # Check mention triples
    assert ("github:issue/octocat/Hello-World/42", "github:mentions", "github:user/user1") in triple_tuples
    assert ("github:issue/octocat/Hello-World/42", "github:mentions", "github:user/user2") in triple_tuples


# ---------------------------------------------------------------------------
# Pull Request Triple Extraction Tests
# ---------------------------------------------------------------------------


def test_extract_pr_basic_triples():
    """Test extraction of basic PR triples."""
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="feature",
    )

    triples = extract_pr_triples(pr)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check PR type
    assert ("github:pr/octocat/Hello-World/123", "rdf:type", "github:PullRequest") in triple_tuples

    # Check title
    assert ("github:pr/octocat/Hello-World/123", "dc:title", "Add feature") in triple_tuples

    # Check state
    assert ("github:pr/octocat/Hello-World/123", "github:state", "open") in triple_tuples


def test_extract_pr_code_change_triples():
    """Test extraction of PR code change metrics triples."""
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="feature",
        changed_files=5,
        additions=150,
        deletions=30,
        commits=3,
    )

    triples = extract_pr_triples(pr)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check code change metrics
    assert ("github:pr/octocat/Hello-World/123", "github:changedFiles", "5") in triple_tuples
    assert ("github:pr/octocat/Hello-World/123", "github:additions", "150") in triple_tuples
    assert ("github:pr/octocat/Hello-World/123", "github:deletions", "30") in triple_tuples
    assert ("github:pr/octocat/Hello-World/123", "github:commits", "3") in triple_tuples


def test_extract_pr_branch_triples():
    """Test extraction of PR branch triples."""
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="feature-x",
    )

    triples = extract_pr_triples(pr)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check branch info
    assert ("github:pr/octocat/Hello-World/123", "github:baseBranch", "main") in triple_tuples
    assert ("github:pr/octocat/Hello-World/123", "github:headBranch", "feature-x") in triple_tuples


def test_extract_pr_review_decision_triples():
    """Test extraction of PR review decision triples."""
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="feature",
        review_decision=ReviewDecision.APPROVED,
    )

    triples = extract_pr_triples(pr)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check review decision
    assert ("github:pr/octocat/Hello-World/123", "github:reviewDecision", "APPROVED") in triple_tuples


def test_extract_pr_reviewer_triples():
    """Test extraction of PR reviewer triples."""
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="feature",
        reviewers=[_create_sample_user("reviewer1")],
    )

    triples = extract_pr_triples(pr)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check reviewer relationship
    assert ("github:pr/octocat/Hello-World/123", "github:reviewedBy", "github:user/reviewer1") in triple_tuples


def test_extract_pr_approved_by_triples():
    """Test extraction of PR approval triples."""
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="feature",
        approved_by=[_create_sample_user("approver1"), _create_sample_user("approver2")],
    )

    triples = extract_pr_triples(pr)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check approval relationships
    assert ("github:pr/octocat/Hello-World/123", "github:approvedBy", "github:user/approver1") in triple_tuples
    assert ("github:pr/octocat/Hello-World/123", "github:approvedBy", "github:user/approver2") in triple_tuples


def test_extract_pr_merged_by_triples():
    """Test extraction of PR merge triples."""
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.MERGED,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        merged_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="feature",
        merged_by=_create_sample_user("maintainer"),
    )

    triples = extract_pr_triples(pr)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check merged by relationship
    assert ("github:pr/octocat/Hello-World/123", "github:mergedBy", "github:user/maintainer") in triple_tuples


def test_extract_pr_modified_files_triples():
    """Test extraction of PR modified files triples."""
    pr = PullRequestMetadata(
        pr_number=123,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/123",
        title="Add feature",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="feature",
        modified_files=["src/main.py", "tests/test_main.py"],
    )

    triples = extract_pr_triples(pr)
    triple_tuples = {(t.subject, t.predicate, t.object) for t in triples}

    # Check file modification triples
    assert ("github:pr/octocat/Hello-World/123", "github:modifiesFile", "src/main.py") in triple_tuples
    assert ("github:pr/octocat/Hello-World/123", "github:modifiesFile", "tests/test_main.py") in triple_tuples


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


def test_extract_issue_triples_no_optional_fields():
    """Test issue triple extraction with minimal fields."""
    issue = IssueMetadata(
        issue_number=1,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/1",
        title="Minimal issue",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
    )

    triples = extract_issue_triples(issue)

    # Should have basic triples without errors
    assert len(triples) > 0
    triple_subjects = {t.subject for t in triples}
    assert "github:issue/octocat/Hello-World/1" in triple_subjects


def test_extract_pr_triples_no_optional_fields():
    """Test PR triple extraction with minimal fields."""
    pr = PullRequestMetadata(
        pr_number=1,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/pull/1",
        title="Minimal PR",
        state=PRState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
        base_branch="main",
        head_branch="fix",
    )

    triples = extract_pr_triples(pr)

    # Should have basic triples without errors
    assert len(triples) > 0
    triple_subjects = {t.subject for t in triples}
    assert "github:pr/octocat/Hello-World/1" in triple_subjects


def test_triple_source_metadata():
    """Test that triples include proper source metadata."""
    issue = IssueMetadata(
        issue_number=42,
        repo_id="octocat/Hello-World",
        github_url="https://github.com/octocat/Hello-World/issues/42",
        title="Test",
        state=IssueState.OPEN,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        author=_create_sample_user("author"),
    )

    triples = extract_issue_triples(issue)

    # Check that all triples have source metadata
    for triple in triples:
        assert triple.source_element_id is not None
        assert triple.extraction_method is not None
        assert "github" in triple.extraction_method
