"""Data models for GitHub Issues and Pull Requests.

This module defines normalized data structures for GitHub issues, pull requests,
and related entities (users, labels, milestones, comments). All models use
timezone-aware datetime handling and include validation for data integrity.

These models serve as the intermediate representation between GitHub's GraphQL
API responses and the semantic triple extraction for PKG construction.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IssueState(str, Enum):
    """Issue state enum."""

    OPEN = "open"
    CLOSED = "closed"


class PRState(str, Enum):
    """Pull request state enum."""

    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class ReviewDecision(str, Enum):
    """PR review decision enum."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Supporting Entity Models
# ---------------------------------------------------------------------------


class GitHubUser(BaseModel):
    """Normalized GitHub user entity.

    Represents a GitHub user with core identification and profile information.
    Used for authors, assignees, reviewers, and other participants.
    """

    login: str = Field(..., description="GitHub username")
    name: Optional[str] = Field(default=None, description="Display name")
    email: Optional[str] = Field(default=None, description="Email address")
    avatar_url: Optional[str] = Field(default=None, description="Avatar URL")
    github_url: str = Field(..., description="Profile URL")

    @field_validator("github_url", mode="before")
    @classmethod
    def _ensure_github_url(cls, v: Optional[str], info) -> str:
        """Ensure github_url is set, constructing from login if needed."""
        if v:
            return v
        # Construct from login if not provided
        login = info.data.get("login", "unknown")
        return f"https://github.com/{login}"

    @property
    def display_identifier(self) -> str:
        """Get best display identifier (name if available, else login)."""
        return self.name if self.name else self.login


class Label(BaseModel):
    """GitHub issue/PR label.

    Represents a label used for categorization and classification.
    """

    name: str = Field(..., description="Label name")
    color: str = Field(..., description="Label color (hex without #)")
    description: Optional[str] = Field(default=None, description="Label description")

    @field_validator("color")
    @classmethod
    def _validate_color(cls, v: str) -> str:
        """Validate color is valid hex (without #)."""
        # Remove # if present
        v = v.lstrip("#")
        # Validate hex format
        if not re.match(r"^[0-9a-fA-F]{6}$", v):
            raise ValueError(f"Invalid color hex: {v}")
        return v.lower()


class Milestone(BaseModel):
    """GitHub project milestone.

    Represents a project milestone for tracking progress toward goals.
    """

    title: str = Field(..., description="Milestone title")
    description: Optional[str] = Field(default=None, description="Milestone description")
    due_on: Optional[datetime] = Field(default=None, description="Due date")
    state: str = Field(..., description="Milestone state (open/closed)")

    @field_validator("due_on")
    @classmethod
    def _ensure_timezone(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime has timezone info."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @field_validator("state")
    @classmethod
    def _validate_state(cls, v: str) -> str:
        """Validate milestone state."""
        if v.lower() not in ("open", "closed"):
            raise ValueError(f"Invalid milestone state: {v}")
        return v.lower()


class Reaction(BaseModel):
    """Reaction to issue/PR/comment.

    Represents emoji reactions (👍, ❤️, etc.) with counts.
    """

    content: str = Field(..., description="Reaction emoji/type")
    count: int = Field(default=1, ge=1, description="Number of reactions")

    VALID_REACTIONS: ClassVar[set] = {
        "+1",
        "-1",
        "laugh",
        "hooray",
        "confused",
        "heart",
        "rocket",
        "eyes",
    }

    @field_validator("content")
    @classmethod
    def _validate_content(cls, v: str) -> str:
        """Validate reaction content."""
        if v not in cls.VALID_REACTIONS:
            raise ValueError(f"Invalid reaction: {v}. Must be one of {cls.VALID_REACTIONS}")
        return v


class Comment(BaseModel):
    """Issue or PR comment.

    Represents a comment with author, body, and engagement metrics.
    """

    comment_id: int = Field(..., description="Comment ID", ge=1)
    author: GitHubUser = Field(..., description="Comment author")
    body: str = Field(..., description="Comment body text")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    reactions: Dict[str, int] = Field(
        default_factory=dict, description="Reactions by type"
    )

    @field_validator("created_at", "updated_at")
    @classmethod
    def _ensure_timezone(cls, v: datetime) -> datetime:
        """Ensure datetime has timezone info."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def total_reactions(self) -> int:
        """Total number of reactions."""
        return sum(self.reactions.values())


# ---------------------------------------------------------------------------
# Issue Model
# ---------------------------------------------------------------------------


class IssueMetadata(BaseModel):
    """Normalized GitHub issue metadata.

    Complete issue representation including content, status, participants,
    classification, and engagement metrics.
    """

    # Identity
    issue_number: int = Field(..., description="Issue number", ge=1)
    repo_id: str = Field(..., description="Repository identifier (owner/repo)")
    github_url: str = Field(..., description="Issue URL")

    # Content
    title: str = Field(..., description="Issue title")
    body: Optional[str] = Field(default=None, description="Issue body (markdown)")
    body_html: Optional[str] = Field(default=None, description="Rendered HTML body")

    # Status
    state: IssueState = Field(..., description="Issue state")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    closed_at: Optional[datetime] = Field(default=None, description="Closure timestamp")

    # Participants
    author: GitHubUser = Field(..., description="Issue author")
    assignees: List[GitHubUser] = Field(
        default_factory=list, description="Assigned users"
    )
    participants: List[GitHubUser] = Field(
        default_factory=list, description="All commenters/participants"
    )

    # Classification
    labels: List[Label] = Field(default_factory=list, description="Issue labels")
    milestone: Optional[Milestone] = Field(default=None, description="Milestone")

    # Engagement metrics
    comment_count: int = Field(default=0, ge=0, description="Number of comments")
    reaction_count: int = Field(default=0, ge=0, description="Total reactions")
    reactions: Dict[str, int] = Field(
        default_factory=dict, description="Reactions by type"
    )

    # Links and references
    mentioned_users: List[str] = Field(
        default_factory=list, description="@mentioned usernames"
    )
    referenced_issues: List[int] = Field(
        default_factory=list, description="Referenced issue numbers"
    )
    referenced_prs: List[int] = Field(
        default_factory=list, description="Referenced PR numbers"
    )
    external_links: List[str] = Field(
        default_factory=list, description="External URLs"
    )

    # Comments (optional, populated if requested)
    comments: List[Comment] = Field(
        default_factory=list, description="Issue comments"
    )

    @field_validator("created_at", "updated_at", "closed_at")
    @classmethod
    def _ensure_timezone(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime has timezone info."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @field_validator("repo_id")
    @classmethod
    def _validate_repo_id(cls, v: str) -> str:
        """Validate repo_id format (owner/repo)."""
        if "/" not in v:
            raise ValueError(f"Invalid repo_id format: {v}. Expected 'owner/repo'")
        return v

    @property
    def is_open(self) -> bool:
        """Check if issue is open."""
        return self.state == IssueState.OPEN

    @property
    def has_assignees(self) -> bool:
        """Check if issue has assignees."""
        return len(self.assignees) > 0

    @property
    def has_labels(self) -> bool:
        """Check if issue has labels."""
        return len(self.labels) > 0

    @property
    def has_milestone(self) -> bool:
        """Check if issue is part of a milestone."""
        return self.milestone is not None

    def extract_references_from_body(self) -> None:
        """Extract issue/PR references and mentions from body text.

        Updates referenced_issues, referenced_prs, mentioned_users, and
        external_links fields by parsing the body text.
        """
        if not self.body:
            return

        # Extract issue/PR references (#123)
        issue_refs = re.findall(r"#(\d+)", self.body)
        self.referenced_issues.extend([int(ref) for ref in issue_refs])

        # Extract user mentions (@username)
        mentions = re.findall(r"@([\w-]+)", self.body)
        self.mentioned_users.extend(mentions)

        # Extract external links
        url_pattern = r"https?://[^\s)]+"
        urls = re.findall(url_pattern, self.body)
        self.external_links.extend(urls)

        # Deduplicate
        self.referenced_issues = list(set(self.referenced_issues))
        self.mentioned_users = list(set(self.mentioned_users))
        self.external_links = list(set(self.external_links))


# ---------------------------------------------------------------------------
# Pull Request Model
# ---------------------------------------------------------------------------


class PullRequestMetadata(BaseModel):
    """Normalized GitHub pull request metadata.

    Complete PR representation including issue-like properties plus
    code review status, file changes, and merge information.
    """

    # Identity
    pr_number: int = Field(..., description="PR number", ge=1)
    repo_id: str = Field(..., description="Repository identifier (owner/repo)")
    github_url: str = Field(..., description="PR URL")

    # Content
    title: str = Field(..., description="PR title")
    body: Optional[str] = Field(default=None, description="PR body (markdown)")

    # Status
    state: PRState = Field(..., description="PR state")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    closed_at: Optional[datetime] = Field(default=None, description="Closure timestamp")
    merged_at: Optional[datetime] = Field(default=None, description="Merge timestamp")

    # Participants
    author: GitHubUser = Field(..., description="PR author")
    reviewers: List[GitHubUser] = Field(
        default_factory=list, description="Requested reviewers"
    )
    assignees: List[GitHubUser] = Field(
        default_factory=list, description="Assigned users"
    )
    merged_by: Optional[GitHubUser] = Field(default=None, description="User who merged")

    # Code changes
    changed_files: int = Field(default=0, ge=0, description="Number of changed files")
    additions: int = Field(default=0, ge=0, description="Lines added")
    deletions: int = Field(default=0, ge=0, description="Lines deleted")
    commits: int = Field(default=0, ge=0, description="Number of commits")
    base_branch: str = Field(..., description="Base branch name")
    head_branch: str = Field(..., description="Head branch name")

    # Review status
    review_decision: ReviewDecision = Field(
        default=ReviewDecision.NONE, description="Review decision status"
    )
    approved_by: List[GitHubUser] = Field(
        default_factory=list, description="Users who approved"
    )
    changes_requested_by: List[GitHubUser] = Field(
        default_factory=list, description="Users who requested changes"
    )

    # Classification
    labels: List[Label] = Field(default_factory=list, description="PR labels")
    milestone: Optional[Milestone] = Field(default=None, description="Milestone")

    # Files
    modified_files: List[str] = Field(
        default_factory=list, description="List of modified file paths"
    )

    # Engagement metrics
    comment_count: int = Field(default=0, ge=0, description="Number of comments")
    review_comment_count: int = Field(
        default=0, ge=0, description="Number of review comments"
    )
    reactions: Dict[str, int] = Field(
        default_factory=dict, description="Reactions by type"
    )

    # Links and references
    mentioned_users: List[str] = Field(
        default_factory=list, description="@mentioned usernames"
    )
    referenced_issues: List[int] = Field(
        default_factory=list, description="Referenced issue numbers"
    )
    external_links: List[str] = Field(
        default_factory=list, description="External URLs"
    )

    # Comments (optional)
    comments: List[Comment] = Field(default_factory=list, description="PR comments")

    @field_validator("created_at", "updated_at", "closed_at", "merged_at")
    @classmethod
    def _ensure_timezone(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime has timezone info."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @field_validator("repo_id")
    @classmethod
    def _validate_repo_id(cls, v: str) -> str:
        """Validate repo_id format (owner/repo)."""
        if "/" not in v:
            raise ValueError(f"Invalid repo_id format: {v}. Expected 'owner/repo'")
        return v

    @property
    def is_merged(self) -> bool:
        """Check if PR is merged."""
        return self.state == PRState.MERGED

    @property
    def is_open(self) -> bool:
        """Check if PR is open."""
        return self.state == PRState.OPEN

    @property
    def is_approved(self) -> bool:
        """Check if PR is approved."""
        return self.review_decision == ReviewDecision.APPROVED

    @property
    def needs_review(self) -> bool:
        """Check if PR needs review."""
        return self.review_decision == ReviewDecision.REVIEW_REQUIRED

    @property
    def total_changes(self) -> int:
        """Total lines changed (additions + deletions)."""
        return self.additions + self.deletions

    @property
    def has_reviewers(self) -> bool:
        """Check if PR has requested reviewers."""
        return len(self.reviewers) > 0

    def extract_references_from_body(self) -> None:
        """Extract issue/PR references and mentions from body text.

        Updates referenced_issues, mentioned_users, and external_links
        fields by parsing the body text.
        """
        if not self.body:
            return

        # Extract issue/PR references (#123)
        issue_refs = re.findall(r"#(\d+)", self.body)
        self.referenced_issues.extend([int(ref) for ref in issue_refs])

        # Extract user mentions (@username)
        mentions = re.findall(r"@([\w-]+)", self.body)
        self.mentioned_users.extend(mentions)

        # Extract external links
        url_pattern = r"https?://[^\s)]+"
        urls = re.findall(url_pattern, self.body)
        self.external_links.extend(urls)

        # Deduplicate
        self.referenced_issues = list(set(self.referenced_issues))
        self.mentioned_users = list(set(self.mentioned_users))
        self.external_links = list(set(self.external_links))


__all__ = [
    "Comment",
    "GitHubUser",
    "IssueMetadata",
    "IssueState",
    "Label",
    "Milestone",
    "PRState",
    "PullRequestMetadata",
    "Reaction",
    "ReviewDecision",
]
