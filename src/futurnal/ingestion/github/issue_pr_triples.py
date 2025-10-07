"""Semantic triple extraction from GitHub issues and PRs for PKG construction.

This module extracts structured semantic relationships from GitHub issues and
pull requests to populate the Personal Knowledge Graph (PKG). Triples capture:
- Issue/PR entities and their properties (title, state, dates)
- User entities and relationships (authors, assignees, reviewers)
- Label and milestone classification
- Comment and discussion threads
- Review relationships and decisions
- Code change tracking
- Temporal relationships for experiential timeline

These triples form the foundation for the Ghost's understanding of GitHub
collaboration patterns, enabling project analysis, contribution tracking,
and decision-making insight.

Ghost→Animal Evolution:
- **Phase 1 (CURRENT)**: Metadata-based triple extraction from GraphQL
- **Phase 2 (FUTURE)**: Content-based relationship extraction from discussions
- **Phase 3 (FUTURE)**: Causal relationship inference from project evolution
"""

from __future__ import annotations

import logging
from typing import List

from ...pipeline.triples import SemanticTriple
from .issue_pr_models import (
    Comment,
    GitHubUser,
    IssueMetadata,
    Label,
    Milestone,
    PullRequestMetadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue Triple Extraction
# ---------------------------------------------------------------------------


def extract_issue_triples(issue: IssueMetadata) -> List[SemanticTriple]:
    """Extract semantic triples from issue metadata.

    Generates PKG triples representing:
    - Issue entity with properties (title, state, dates)
    - Author and participant person entities
    - Assignee relationships
    - Label classification
    - Milestone project phase
    - Comment threads
    - Temporal metadata
    - References and mentions

    Args:
        issue: Normalized issue metadata

    Returns:
        List of SemanticTriple objects for PKG storage

    Example triples generated:
        (github:issue:octocat/Hello-World:42, rdf:type, github:Issue)
        (github:issue:octocat/Hello-World:42, dc:title, "Fix bug in parser")
        (github:issue:octocat/Hello-World:42, github:createdBy, github:user:octocat)
        (github:issue:octocat/Hello-World:42, github:assignedTo, github:user:dev1)
        (github:issue:octocat/Hello-World:42, github:hasLabel, github:label:bug)
    """
    triples = []

    # Create issue URI
    issue_uri = _create_issue_uri(issue.repo_id, issue.issue_number)
    source_id = f"{issue.repo_id}:issue:{issue.issue_number}"

    # Issue type triple
    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="rdf:type",
            object="github:Issue",
            source_element_id=source_id,
            source_path=issue.github_url,
            extraction_method="github_issue_metadata",
        )
    )

    # Basic properties
    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="dc:title",
            object=issue.title,
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="github:state",
            object=issue.state.value,
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="github:number",
            object=str(issue.issue_number),
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    # Repository relationship
    repo_uri = _create_repo_uri(issue.repo_id)
    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="github:inRepository",
            object=repo_uri,
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    # Author relationship
    triples.extend(
        _extract_user_triples(
            issue.author,
            source_id,
        )
    )
    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="github:createdBy",
            object=_create_user_uri(issue.author.login),
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    # Assignee relationships
    for assignee in issue.assignees:
        triples.extend(_extract_user_triples(assignee, source_id))
        triples.append(
            SemanticTriple(
                subject=issue_uri,
                predicate="github:assignedTo",
                object=_create_user_uri(assignee.login),
                source_element_id=source_id,
                extraction_method="github_issue_metadata",
            )
        )

    # Participant relationships
    for participant in issue.participants:
        triples.extend(_extract_user_triples(participant, source_id))
        triples.append(
            SemanticTriple(
                subject=issue_uri,
                predicate="github:participatedBy",
                object=_create_user_uri(participant.login),
                source_element_id=source_id,
                extraction_method="github_issue_metadata",
            )
        )

    # Label classification
    for label in issue.labels:
        triples.extend(_extract_label_triples(label, issue.repo_id, source_id))
        label_uri = _create_label_uri(issue.repo_id, label.name)
        triples.append(
            SemanticTriple(
                subject=issue_uri,
                predicate="github:hasLabel",
                object=label_uri,
                source_element_id=source_id,
                extraction_method="github_issue_metadata",
            )
        )

    # Milestone relationship
    if issue.milestone:
        triples.extend(
            _extract_milestone_triples(issue.milestone, issue.repo_id, source_id)
        )
        milestone_uri = _create_milestone_uri(issue.repo_id, issue.milestone.title)
        triples.append(
            SemanticTriple(
                subject=issue_uri,
                predicate="github:inMilestone",
                object=milestone_uri,
                source_element_id=source_id,
                extraction_method="github_issue_metadata",
            )
        )

    # Temporal properties
    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="dc:created",
            object=issue.created_at.isoformat(),
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="dc:modified",
            object=issue.updated_at.isoformat(),
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    if issue.closed_at:
        triples.append(
            SemanticTriple(
                subject=issue_uri,
                predicate="github:closedAt",
                object=issue.closed_at.isoformat(),
                source_element_id=source_id,
                extraction_method="github_issue_metadata",
            )
        )

    # Engagement metrics
    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="github:commentCount",
            object=str(issue.comment_count),
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=issue_uri,
            predicate="github:reactionCount",
            object=str(issue.reaction_count),
            source_element_id=source_id,
            extraction_method="github_issue_metadata",
        )
    )

    # Comment triples
    for comment in issue.comments:
        triples.extend(_extract_comment_triples(comment, issue_uri, source_id))

    # Reference triples
    for ref_issue_num in issue.referenced_issues:
        ref_uri = _create_issue_uri(issue.repo_id, ref_issue_num)
        triples.append(
            SemanticTriple(
                subject=issue_uri,
                predicate="github:references",
                object=ref_uri,
                source_element_id=source_id,
                extraction_method="github_issue_metadata",
            )
        )

    # Mention triples
    for mentioned_user in issue.mentioned_users:
        user_uri = _create_user_uri(mentioned_user)
        triples.append(
            SemanticTriple(
                subject=issue_uri,
                predicate="github:mentions",
                object=user_uri,
                source_element_id=source_id,
                extraction_method="github_issue_metadata",
            )
        )

    return triples


# ---------------------------------------------------------------------------
# Pull Request Triple Extraction
# ---------------------------------------------------------------------------


def extract_pr_triples(pr: PullRequestMetadata) -> List[SemanticTriple]:
    """Extract semantic triples from pull request metadata.

    Generates PKG triples representing:
    - PR entity with properties (title, state, dates)
    - Author and reviewer relationships
    - Review decision status
    - Code change metrics
    - Merge information
    - Label and milestone classification
    - Modified files
    - Temporal metadata

    Args:
        pr: Normalized PR metadata

    Returns:
        List of SemanticTriple objects for PKG storage

    Example triples generated:
        (github:pr:octocat/Hello-World:123, rdf:type, github:PullRequest)
        (github:pr:octocat/Hello-World:123, dc:title, "Add new feature")
        (github:pr:octocat/Hello-World:123, github:createdBy, github:user:dev1)
        (github:pr:octocat/Hello-World:123, github:reviewedBy, github:user:reviewer1)
        (github:pr:octocat/Hello-World:123, github:reviewDecision, "APPROVED")
    """
    triples = []

    # Create PR URI
    pr_uri = _create_pr_uri(pr.repo_id, pr.pr_number)
    source_id = f"{pr.repo_id}:pr:{pr.pr_number}"

    # PR type triple
    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="rdf:type",
            object="github:PullRequest",
            source_element_id=source_id,
            source_path=pr.github_url,
            extraction_method="github_pr_metadata",
        )
    )

    # Basic properties
    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="dc:title",
            object=pr.title,
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:state",
            object=pr.state.value,
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:number",
            object=str(pr.pr_number),
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    # Repository relationship
    repo_uri = _create_repo_uri(pr.repo_id)
    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:inRepository",
            object=repo_uri,
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    # Author relationship
    triples.extend(_extract_user_triples(pr.author, source_id))
    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:createdBy",
            object=_create_user_uri(pr.author.login),
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    # Reviewer relationships
    for reviewer in pr.reviewers:
        triples.extend(_extract_user_triples(reviewer, source_id))
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:reviewedBy",
                object=_create_user_uri(reviewer.login),
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    # Approved by relationships
    for approver in pr.approved_by:
        triples.extend(_extract_user_triples(approver, source_id))
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:approvedBy",
                object=_create_user_uri(approver.login),
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    # Changes requested by relationships
    for requester in pr.changes_requested_by:
        triples.extend(_extract_user_triples(requester, source_id))
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:changesRequestedBy",
                object=_create_user_uri(requester.login),
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    # Merged by relationship
    if pr.merged_by:
        triples.extend(_extract_user_triples(pr.merged_by, source_id))
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:mergedBy",
                object=_create_user_uri(pr.merged_by.login),
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    # Review decision
    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:reviewDecision",
            object=pr.review_decision.value,
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    # Code change metrics
    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:changedFiles",
            object=str(pr.changed_files),
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:additions",
            object=str(pr.additions),
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:deletions",
            object=str(pr.deletions),
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:commits",
            object=str(pr.commits),
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    # Branch information
    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:baseBranch",
            object=pr.base_branch,
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="github:headBranch",
            object=pr.head_branch,
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    # Label classification
    for label in pr.labels:
        triples.extend(_extract_label_triples(label, pr.repo_id, source_id))
        label_uri = _create_label_uri(pr.repo_id, label.name)
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:hasLabel",
                object=label_uri,
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    # Milestone relationship
    if pr.milestone:
        triples.extend(_extract_milestone_triples(pr.milestone, pr.repo_id, source_id))
        milestone_uri = _create_milestone_uri(pr.repo_id, pr.milestone.title)
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:inMilestone",
                object=milestone_uri,
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    # Temporal properties
    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="dc:created",
            object=pr.created_at.isoformat(),
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    triples.append(
        SemanticTriple(
            subject=pr_uri,
            predicate="dc:modified",
            object=pr.updated_at.isoformat(),
            source_element_id=source_id,
            extraction_method="github_pr_metadata",
        )
    )

    if pr.merged_at:
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:mergedAt",
                object=pr.merged_at.isoformat(),
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    if pr.closed_at:
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:closedAt",
                object=pr.closed_at.isoformat(),
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    # Modified files
    for file_path in pr.modified_files:
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:modifiesFile",
                object=file_path,
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    # Comment triples
    for comment in pr.comments:
        triples.extend(_extract_comment_triples(comment, pr_uri, source_id))

    # Reference triples
    for ref_issue_num in pr.referenced_issues:
        ref_uri = _create_issue_uri(pr.repo_id, ref_issue_num)
        triples.append(
            SemanticTriple(
                subject=pr_uri,
                predicate="github:references",
                object=ref_uri,
                source_element_id=source_id,
                extraction_method="github_pr_metadata",
            )
        )

    return triples


# ---------------------------------------------------------------------------
# Supporting Triple Extraction Functions
# ---------------------------------------------------------------------------


def _extract_user_triples(
    user: GitHubUser,
    source_id: str,
) -> List[SemanticTriple]:
    """Extract user entity triples.

    Args:
        user: GitHub user
        source_id: Source element ID

    Returns:
        List of user triples
    """
    triples = []
    user_uri = _create_user_uri(user.login)

    # User type
    triples.append(
        SemanticTriple(
            subject=user_uri,
            predicate="rdf:type",
            object="github:User",
            source_element_id=source_id,
            extraction_method="github_user_metadata",
        )
    )

    # User login
    triples.append(
        SemanticTriple(
            subject=user_uri,
            predicate="github:login",
            object=user.login,
            source_element_id=source_id,
            extraction_method="github_user_metadata",
        )
    )

    # User name (if available)
    if user.name:
        triples.append(
            SemanticTriple(
                subject=user_uri,
                predicate="foaf:name",
                object=user.name,
                source_element_id=source_id,
                extraction_method="github_user_metadata",
            )
        )

    # Profile URL
    triples.append(
        SemanticTriple(
            subject=user_uri,
            predicate="foaf:homepage",
            object=user.github_url,
            source_element_id=source_id,
            extraction_method="github_user_metadata",
        )
    )

    return triples


def _extract_label_triples(
    label: Label,
    repo_id: str,
    source_id: str,
) -> List[SemanticTriple]:
    """Extract label entity triples.

    Args:
        label: Label
        repo_id: Repository ID
        source_id: Source element ID

    Returns:
        List of label triples
    """
    triples = []
    label_uri = _create_label_uri(repo_id, label.name)

    # Label type
    triples.append(
        SemanticTriple(
            subject=label_uri,
            predicate="rdf:type",
            object="github:Label",
            source_element_id=source_id,
            extraction_method="github_label_metadata",
        )
    )

    # Label name
    triples.append(
        SemanticTriple(
            subject=label_uri,
            predicate="github:labelName",
            object=label.name,
            source_element_id=source_id,
            extraction_method="github_label_metadata",
        )
    )

    # Label color
    triples.append(
        SemanticTriple(
            subject=label_uri,
            predicate="github:labelColor",
            object=label.color,
            source_element_id=source_id,
            extraction_method="github_label_metadata",
        )
    )

    # Label description (if available)
    if label.description:
        triples.append(
            SemanticTriple(
                subject=label_uri,
                predicate="dc:description",
                object=label.description,
                source_element_id=source_id,
                extraction_method="github_label_metadata",
            )
        )

    return triples


def _extract_milestone_triples(
    milestone: Milestone,
    repo_id: str,
    source_id: str,
) -> List[SemanticTriple]:
    """Extract milestone entity triples.

    Args:
        milestone: Milestone
        repo_id: Repository ID
        source_id: Source element ID

    Returns:
        List of milestone triples
    """
    triples = []
    milestone_uri = _create_milestone_uri(repo_id, milestone.title)

    # Milestone type
    triples.append(
        SemanticTriple(
            subject=milestone_uri,
            predicate="rdf:type",
            object="github:Milestone",
            source_element_id=source_id,
            extraction_method="github_milestone_metadata",
        )
    )

    # Milestone title
    triples.append(
        SemanticTriple(
            subject=milestone_uri,
            predicate="dc:title",
            object=milestone.title,
            source_element_id=source_id,
            extraction_method="github_milestone_metadata",
        )
    )

    # Milestone state
    triples.append(
        SemanticTriple(
            subject=milestone_uri,
            predicate="github:state",
            object=milestone.state,
            source_element_id=source_id,
            extraction_method="github_milestone_metadata",
        )
    )

    # Milestone description (if available)
    if milestone.description:
        triples.append(
            SemanticTriple(
                subject=milestone_uri,
                predicate="dc:description",
                object=milestone.description,
                source_element_id=source_id,
                extraction_method="github_milestone_metadata",
            )
        )

    # Due date (if available)
    if milestone.due_on:
        triples.append(
            SemanticTriple(
                subject=milestone_uri,
                predicate="github:dueOn",
                object=milestone.due_on.isoformat(),
                source_element_id=source_id,
                extraction_method="github_milestone_metadata",
            )
        )

    return triples


def _extract_comment_triples(
    comment: Comment,
    parent_uri: str,
    source_id: str,
) -> List[SemanticTriple]:
    """Extract comment entity triples.

    Args:
        comment: Comment
        parent_uri: Parent issue/PR URI
        source_id: Source element ID

    Returns:
        List of comment triples
    """
    triples = []
    comment_uri = f"{parent_uri}/comment/{comment.comment_id}"

    # Comment type
    triples.append(
        SemanticTriple(
            subject=comment_uri,
            predicate="rdf:type",
            object="github:Comment",
            source_element_id=source_id,
            extraction_method="github_comment_metadata",
        )
    )

    # Comment on parent
    triples.append(
        SemanticTriple(
            subject=comment_uri,
            predicate="github:commentOn",
            object=parent_uri,
            source_element_id=source_id,
            extraction_method="github_comment_metadata",
        )
    )

    # Author
    triples.extend(_extract_user_triples(comment.author, source_id))
    triples.append(
        SemanticTriple(
            subject=comment_uri,
            predicate="github:createdBy",
            object=_create_user_uri(comment.author.login),
            source_element_id=source_id,
            extraction_method="github_comment_metadata",
        )
    )

    # Created at
    triples.append(
        SemanticTriple(
            subject=comment_uri,
            predicate="dc:created",
            object=comment.created_at.isoformat(),
            source_element_id=source_id,
            extraction_method="github_comment_metadata",
        )
    )

    return triples


# ---------------------------------------------------------------------------
# URI Creation Functions
# ---------------------------------------------------------------------------


def _create_issue_uri(repo_id: str, issue_number: int) -> str:
    """Create URI for issue.

    Args:
        repo_id: Repository ID (owner/repo)
        issue_number: Issue number

    Returns:
        Issue URI
    """
    return f"github:issue/{repo_id}/{issue_number}"


def _create_pr_uri(repo_id: str, pr_number: int) -> str:
    """Create URI for pull request.

    Args:
        repo_id: Repository ID (owner/repo)
        pr_number: PR number

    Returns:
        PR URI
    """
    return f"github:pr/{repo_id}/{pr_number}"


def _create_user_uri(login: str) -> str:
    """Create URI for user.

    Args:
        login: GitHub username

    Returns:
        User URI
    """
    return f"github:user/{login}"


def _create_repo_uri(repo_id: str) -> str:
    """Create URI for repository.

    Args:
        repo_id: Repository ID (owner/repo)

    Returns:
        Repository URI
    """
    return f"github:repo/{repo_id}"


def _create_label_uri(repo_id: str, label_name: str) -> str:
    """Create URI for label.

    Args:
        repo_id: Repository ID
        label_name: Label name

    Returns:
        Label URI
    """
    clean_name = label_name.replace(" ", "_").replace("/", "_")
    return f"github:label/{repo_id}/{clean_name}"


def _create_milestone_uri(repo_id: str, milestone_title: str) -> str:
    """Create URI for milestone.

    Args:
        repo_id: Repository ID
        milestone_title: Milestone title

    Returns:
        Milestone URI
    """
    clean_title = milestone_title.replace(" ", "_").replace("/", "_")
    return f"github:milestone/{repo_id}/{clean_title}"


__all__ = [
    "extract_issue_triples",
    "extract_pr_triples",
]
