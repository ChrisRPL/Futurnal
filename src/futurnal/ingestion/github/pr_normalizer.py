"""GitHub pull request normalizer for PKG-ready metadata extraction.

This module normalizes GitHub pull requests from GraphQL API responses into
structured PullRequestMetadata objects ready for semantic triple extraction.
It handles:
- Complete PR metadata including review status
- Code change metrics (additions, deletions, file counts)
- Review decision tracking (approved, changes requested)
- Reviewer and merger tracking
- File modification lists
- Privacy-aware data handling

The normalizer extends issue-like functionality with PR-specific features
such as review workflows, merge tracking, and code change analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .api_client_manager import GitHubAPIClientManager
from .issue_pr_models import (
    Comment,
    GitHubUser,
    Label,
    Milestone,
    PRState,
    PullRequestMetadata,
    ReviewDecision,
)

logger = logging.getLogger(__name__)


class PullRequestNormalizer:
    """Normalizes GitHub pull requests into PKG-ready format.

    Fetches PR data via GraphQL and transforms it into structured
    PullRequestMetadata objects with complete review tracking, code
    change metrics, and participant relationships.

    Example:
        >>> api_client = GitHubAPIClientManager(credential_manager)
        >>> normalizer = PullRequestNormalizer(api_client=api_client)
        >>> pr = await normalizer.normalize_pull_request(
        ...     repo_owner="octocat",
        ...     repo_name="Hello-World",
        ...     pr_number=123,
        ...     credential_id="github_cred_123"
        ... )
        >>> print(f"PR: {pr.title} ({pr.state})")
        >>> print(f"Review: {pr.review_decision}")
    """

    def __init__(
        self,
        *,
        api_client: GitHubAPIClientManager,
        extract_reviews: bool = True,
        extract_files: bool = True,
        max_files: int = 100,
    ):
        """Initialize pull request normalizer.

        Args:
            api_client: GitHub API client manager for GraphQL requests
            extract_reviews: Whether to fetch review information
            extract_files: Whether to fetch modified file list
            max_files: Maximum number of files to fetch (pagination limit)
        """
        self.api_client = api_client
        self.extract_reviews = extract_reviews
        self.extract_files = extract_files
        self.max_files = max_files

    async def normalize_pull_request(
        self,
        *,
        repo_owner: str,
        repo_name: str,
        pr_number: int,
        credential_id: str,
        github_host: str = "github.com",
    ) -> PullRequestMetadata:
        """Normalize a single GitHub pull request.

        Fetches PR data via GraphQL, parses the response, and optionally
        extracts reviews and file changes to produce a complete
        PullRequestMetadata object.

        Args:
            repo_owner: Repository owner login
            repo_name: Repository name
            pr_number: Pull request number
            credential_id: Credential identifier for authentication
            github_host: GitHub hostname (default: github.com)

        Returns:
            Normalized PR metadata

        Raises:
            Exception: If GraphQL request fails or parsing errors occur
        """
        logger.info(
            f"Normalizing PR {repo_owner}/{repo_name}#{pr_number}",
            extra={
                "repo": f"{repo_owner}/{repo_name}",
                "pr_number": pr_number,
            },
        )

        # Build and execute GraphQL query
        query = self._build_pr_query(
            include_reviews=self.extract_reviews,
            include_files=self.extract_files,
        )
        variables = {
            "owner": repo_owner,
            "repo": repo_name,
            "number": pr_number,
            "maxFiles": self.max_files if self.extract_files else 0,
        }

        response = await self.api_client.graphql_request(
            credential_id=credential_id,
            query=query,
            variables=variables,
            github_host=github_host,
        )

        # Validate response structure
        if "data" not in response or "repository" not in response["data"]:
            raise ValueError(f"Invalid GraphQL response structure: {response}")

        repo_data = response["data"]["repository"]
        if repo_data is None:
            raise ValueError(f"Repository not found: {repo_owner}/{repo_name}")

        if "pullRequest" not in repo_data or repo_data["pullRequest"] is None:
            raise ValueError(f"PR not found: {repo_owner}/{repo_name}#{pr_number}")

        pr_data = repo_data["pullRequest"]

        # Parse PR data
        metadata = self._parse_pr_data(pr_data, repo_owner, repo_name)

        # Extract references from body
        metadata.extract_references_from_body()

        logger.debug(
            f"Normalized PR: {metadata.changed_files} files, "
            f"+{metadata.additions}/-{metadata.deletions}, "
            f"review: {metadata.review_decision.value}",
            extra={
                "pr_number": pr_number,
                "state": metadata.state.value,
                "author": metadata.author.login,
                "is_merged": metadata.is_merged,
            },
        )

        return metadata

    def _build_pr_query(
        self,
        include_reviews: bool = True,
        include_files: bool = True,
    ) -> str:
        """Build GraphQL query for PR data.

        Constructs an optimized GraphQL query that fetches all necessary
        PR metadata in a single request, including review status, file
        changes, and participant information.

        Args:
            include_reviews: Whether to include review data
            include_files: Whether to include file list

        Returns:
            GraphQL query string
        """
        reviews_fragment = ""
        if include_reviews:
            reviews_fragment = """
            reviews(first: 20) {
              totalCount
              nodes {
                author {
                  login
                  ... on User {
                    name
                    url
                  }
                }
                state
                body
                createdAt
              }
            }
            reviewRequests(first: 10) {
              nodes {
                requestedReviewer {
                  ... on User {
                    login
                    name
                    url
                  }
                }
              }
            }
            """

        files_fragment = ""
        if include_files:
            files_fragment = """
            files(first: $maxFiles) {
              totalCount
              nodes {
                path
                additions
                deletions
              }
            }
            """

        query = f"""
        query($owner: String!, $repo: String!, $number: Int!, $maxFiles: Int!) {{
          repository(owner: $owner, name: $repo) {{
            pullRequest(number: $number) {{
              number
              title
              body
              state
              createdAt
              updatedAt
              closedAt
              mergedAt
              url
              author {{
                login
                ... on User {{
                  name
                  email
                  avatarUrl
                  url
                }}
              }}
              assignees(first: 10) {{
                nodes {{
                  login
                  name
                  url
                }}
              }}
              labels(first: 20) {{
                nodes {{
                  name
                  color
                  description
                }}
              }}
              milestone {{
                title
                description
                dueOn
                state
              }}
              reviewDecision
              changedFiles
              additions
              deletions
              commits {{
                totalCount
              }}
              baseRefName
              headRefName
              mergedBy {{
                login
                name
                url
              }}
              comments {{
                totalCount
              }}
              {reviews_fragment}
              {files_fragment}
            }}
          }}
        }}
        """
        return query

    def _parse_pr_data(
        self,
        pr_data: Dict[str, Any],
        repo_owner: str,
        repo_name: str,
    ) -> PullRequestMetadata:
        """Parse raw GitHub PR data into normalized format.

        Transforms GraphQL response data into a structured
        PullRequestMetadata object with all fields properly typed
        and validated.

        Args:
            pr_data: Raw PR data from GraphQL response
            repo_owner: Repository owner
            repo_name: Repository name

        Returns:
            Normalized PullRequestMetadata object
        """
        # Extract author
        author_data = pr_data.get("author") or {}
        author = self._parse_user(author_data)

        # Extract assignees
        assignees_data = pr_data.get("assignees", {}).get("nodes", [])
        assignees = [self._parse_user(user) for user in assignees_data if user]

        # Extract merged_by
        merged_by = None
        merged_by_data = pr_data.get("mergedBy")
        if merged_by_data:
            merged_by = self._parse_user(merged_by_data)

        # Extract labels
        labels_data = pr_data.get("labels", {}).get("nodes", [])
        labels = [self._parse_label(label) for label in labels_data if label]

        # Extract milestone
        milestone_data = pr_data.get("milestone")
        milestone = self._parse_milestone(milestone_data) if milestone_data else None

        # Extract review decision
        review_decision_str = pr_data.get("reviewDecision") or "NONE"
        try:
            review_decision = ReviewDecision(review_decision_str)
        except ValueError:
            review_decision = ReviewDecision.NONE

        # Extract reviewers and review status
        reviewers: List[GitHubUser] = []
        approved_by: List[GitHubUser] = []
        changes_requested_by: List[GitHubUser] = []

        # Extract requested reviewers
        review_requests = pr_data.get("reviewRequests", {}).get("nodes", [])
        for request in review_requests:
            if request and "requestedReviewer" in request:
                reviewer_data = request["requestedReviewer"]
                if reviewer_data:
                    reviewers.append(self._parse_user(reviewer_data))

        # Extract review states
        reviews_data = pr_data.get("reviews", {}).get("nodes", [])
        for review in reviews_data:
            if not review or "author" not in review:
                continue

            author_data = review["author"]
            if not author_data:
                continue

            user = self._parse_user(author_data)
            state = review.get("state", "")

            if state == "APPROVED":
                approved_by.append(user)
            elif state == "CHANGES_REQUESTED":
                changes_requested_by.append(user)

        # Extract file paths
        modified_files: List[str] = []
        files_data = pr_data.get("files", {})
        if files_data:
            file_nodes = files_data.get("nodes", [])
            modified_files = [
                file["path"] for file in file_nodes if file and "path" in file
            ]

        # Determine PR state
        state = PRState.OPEN
        if pr_data.get("mergedAt"):
            state = PRState.MERGED
        elif pr_data.get("state", "").upper() == "CLOSED":
            state = PRState.CLOSED
        elif pr_data.get("state", "").upper() == "OPEN":
            state = PRState.OPEN

        # Parse timestamps
        created_at = self._parse_datetime(pr_data.get("createdAt"))
        updated_at = self._parse_datetime(pr_data.get("updatedAt"))
        closed_at = self._parse_datetime(pr_data.get("closedAt"))
        merged_at = self._parse_datetime(pr_data.get("mergedAt"))

        # Create metadata object
        return PullRequestMetadata(
            pr_number=pr_data["number"],
            repo_id=f"{repo_owner}/{repo_name}",
            github_url=pr_data.get("url", f"https://github.com/{repo_owner}/{repo_name}/pull/{pr_data['number']}"),
            title=pr_data["title"],
            body=pr_data.get("body"),
            state=state,
            created_at=created_at,
            updated_at=updated_at,
            closed_at=closed_at,
            merged_at=merged_at,
            author=author,
            reviewers=reviewers,
            assignees=assignees,
            merged_by=merged_by,
            changed_files=pr_data.get("changedFiles", 0),
            additions=pr_data.get("additions", 0),
            deletions=pr_data.get("deletions", 0),
            commits=pr_data.get("commits", {}).get("totalCount", 0),
            base_branch=pr_data.get("baseRefName", "main"),
            head_branch=pr_data.get("headRefName", "unknown"),
            review_decision=review_decision,
            approved_by=approved_by,
            changes_requested_by=changes_requested_by,
            labels=labels,
            milestone=milestone,
            modified_files=modified_files,
            comment_count=pr_data.get("comments", {}).get("totalCount", 0),
        )

    def _parse_user(self, user_data: Dict[str, Any]) -> GitHubUser:
        """Parse user data into GitHubUser object.

        Args:
            user_data: Raw user data from GraphQL

        Returns:
            GitHubUser object
        """
        login = user_data.get("login", "ghost")
        return GitHubUser(
            login=login,
            name=user_data.get("name"),
            email=user_data.get("email"),
            avatar_url=user_data.get("avatarUrl"),
            github_url=user_data.get("url", f"https://github.com/{login}"),
        )

    def _parse_label(self, label_data: Dict[str, Any]) -> Label:
        """Parse label data into Label object.

        Args:
            label_data: Raw label data from GraphQL

        Returns:
            Label object
        """
        return Label(
            name=label_data["name"],
            color=label_data["color"],
            description=label_data.get("description"),
        )

    def _parse_milestone(self, milestone_data: Dict[str, Any]) -> Milestone:
        """Parse milestone data into Milestone object.

        Args:
            milestone_data: Raw milestone data from GraphQL

        Returns:
            Milestone object
        """
        due_on = None
        if milestone_data.get("dueOn"):
            due_on = self._parse_datetime(milestone_data["dueOn"])

        return Milestone(
            title=milestone_data["title"],
            description=milestone_data.get("description"),
            due_on=due_on,
            state=milestone_data["state"].lower(),
        )

    def _parse_datetime(self, dt_string: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string with timezone awareness.

        Args:
            dt_string: ISO format datetime string

        Returns:
            Timezone-aware datetime or None
        """
        if not dt_string:
            return None

        try:
            # Parse ISO format and ensure UTC timezone
            dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse datetime: {dt_string}: {e}")
            return None


__all__ = ["PullRequestNormalizer"]
