"""GitHub issue normalizer for PKG-ready metadata extraction.

This module normalizes GitHub issues from GraphQL API responses into structured
IssueMetadata objects ready for semantic triple extraction. It handles:
- Complete issue metadata extraction via optimized GraphQL queries
- Comment thread reconstruction with pagination
- Participant and engagement tracking
- Link and reference extraction
- Privacy-aware data handling

The normalizer uses the GitHubAPIClientManager for rate-limited GraphQL requests
and produces normalized data suitable for PKG storage.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .api_client_manager import GitHubAPIClientManager
from .issue_pr_models import (
    Comment,
    GitHubUser,
    IssueMetadata,
    IssueState,
    Label,
    Milestone,
)

logger = logging.getLogger(__name__)


class IssueNormalizer:
    """Normalizes GitHub issues into PKG-ready format.

    Fetches issue data via GraphQL and transforms it into structured
    IssueMetadata objects with complete participant tracking, engagement
    metrics, and reference extraction.

    Example:
        >>> api_client = GitHubAPIClientManager(credential_manager)
        >>> normalizer = IssueNormalizer(api_client=api_client)
        >>> issue = await normalizer.normalize_issue(
        ...     repo_owner="octocat",
        ...     repo_name="Hello-World",
        ...     issue_number=42,
        ...     credential_id="github_cred_123"
        ... )
        >>> print(f"Issue: {issue.title} by {issue.author.login}")
    """

    def __init__(
        self,
        *,
        api_client: GitHubAPIClientManager,
        extract_comments: bool = True,
        max_comments: int = 100,
    ):
        """Initialize issue normalizer.

        Args:
            api_client: GitHub API client manager for GraphQL requests
            extract_comments: Whether to fetch and include comments
            max_comments: Maximum number of comments to fetch (pagination limit)
        """
        self.api_client = api_client
        self.extract_comments = extract_comments
        self.max_comments = max_comments

    async def normalize_issue(
        self,
        *,
        repo_owner: str,
        repo_name: str,
        issue_number: int,
        credential_id: str,
        github_host: str = "github.com",
    ) -> IssueMetadata:
        """Normalize a single GitHub issue.

        Fetches issue data via GraphQL, parses the response, and optionally
        extracts comments to produce a complete IssueMetadata object.

        Args:
            repo_owner: Repository owner login
            repo_name: Repository name
            issue_number: Issue number
            credential_id: Credential identifier for authentication
            github_host: GitHub hostname (default: github.com)

        Returns:
            Normalized issue metadata

        Raises:
            Exception: If GraphQL request fails or parsing errors occur
        """
        logger.info(
            f"Normalizing issue {repo_owner}/{repo_name}#{issue_number}",
            extra={
                "repo": f"{repo_owner}/{repo_name}",
                "issue_number": issue_number,
            },
        )

        # Build and execute GraphQL query
        query = self._build_issue_query(include_comments=self.extract_comments)
        variables = {
            "owner": repo_owner,
            "repo": repo_name,
            "number": issue_number,
            "maxComments": self.max_comments if self.extract_comments else 0,
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

        if "issue" not in repo_data or repo_data["issue"] is None:
            raise ValueError(
                f"Issue not found: {repo_owner}/{repo_name}#{issue_number}"
            )

        issue_data = repo_data["issue"]

        # Parse issue data
        metadata = self._parse_issue_data(issue_data, repo_owner, repo_name)

        # Extract references from body
        metadata.extract_references_from_body()

        logger.debug(
            f"Normalized issue with {len(metadata.comments)} comments, "
            f"{len(metadata.labels)} labels, {metadata.comment_count} total comments",
            extra={
                "issue_number": issue_number,
                "state": metadata.state.value,
                "author": metadata.author.login,
            },
        )

        return metadata

    def _build_issue_query(self, include_comments: bool = True) -> str:
        """Build GraphQL query for issue data.

        Constructs an optimized GraphQL query that fetches all necessary
        issue metadata in a single request, including participants, labels,
        milestone, reactions, and optionally comments.

        Args:
            include_comments: Whether to include comment data

        Returns:
            GraphQL query string
        """
        comment_fragment = ""
        if include_comments:
            comment_fragment = """
            comments(first: $maxComments) {
              totalCount
              nodes {
                id
                databaseId
                author {
                  login
                  ... on User {
                    name
                    email
                  }
                }
                body
                createdAt
                updatedAt
                reactions(first: 10) {
                  totalCount
                  nodes {
                    content
                  }
                }
              }
            }
            """

        query = f"""
        query($owner: String!, $repo: String!, $number: Int!, $maxComments: Int!) {{
          repository(owner: $owner, name: $repo) {{
            issue(number: $number) {{
              number
              title
              body
              bodyHTML
              state
              createdAt
              updatedAt
              closedAt
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
              participants(first: 50) {{
                totalCount
                nodes {{
                  login
                  name
                  url
                }}
              }}
              reactions {{
                totalCount
              }}
              {comment_fragment}
            }}
          }}
        }}
        """
        return query

    def _parse_issue_data(
        self,
        issue_data: Dict[str, Any],
        repo_owner: str,
        repo_name: str,
    ) -> IssueMetadata:
        """Parse raw GitHub issue data into normalized format.

        Transforms GraphQL response data into a structured IssueMetadata
        object with all fields properly typed and validated.

        Args:
            issue_data: Raw issue data from GraphQL response
            repo_owner: Repository owner
            repo_name: Repository name

        Returns:
            Normalized IssueMetadata object
        """
        # Extract author
        author_data = issue_data.get("author") or {}
        author = self._parse_user(author_data)

        # Extract assignees
        assignees_data = issue_data.get("assignees", {}).get("nodes", [])
        assignees = [self._parse_user(user) for user in assignees_data if user]

        # Extract participants
        participants_data = issue_data.get("participants", {}).get("nodes", [])
        participants = [self._parse_user(user) for user in participants_data if user]

        # Extract labels
        labels_data = issue_data.get("labels", {}).get("nodes", [])
        labels = [self._parse_label(label) for label in labels_data if label]

        # Extract milestone
        milestone_data = issue_data.get("milestone")
        milestone = self._parse_milestone(milestone_data) if milestone_data else None

        # Extract comments
        comments = []
        comments_data = issue_data.get("comments", {})
        if comments_data:
            comment_nodes = comments_data.get("nodes", [])
            comments = [
                self._parse_comment(comment) for comment in comment_nodes if comment
            ]

        # Parse timestamps
        created_at = self._parse_datetime(issue_data.get("createdAt"))
        updated_at = self._parse_datetime(issue_data.get("updatedAt"))
        closed_at = self._parse_datetime(issue_data.get("closedAt"))

        # Create metadata object
        return IssueMetadata(
            issue_number=issue_data["number"],
            repo_id=f"{repo_owner}/{repo_name}",
            github_url=issue_data.get("url", f"https://github.com/{repo_owner}/{repo_name}/issues/{issue_data['number']}"),
            title=issue_data["title"],
            body=issue_data.get("body"),
            body_html=issue_data.get("bodyHTML"),
            state=IssueState(issue_data["state"].lower()),
            created_at=created_at,
            updated_at=updated_at,
            closed_at=closed_at,
            author=author,
            assignees=assignees,
            participants=participants,
            labels=labels,
            milestone=milestone,
            comment_count=comments_data.get("totalCount", 0) if comments_data else 0,
            reaction_count=issue_data.get("reactions", {}).get("totalCount", 0),
            comments=comments,
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

    def _parse_comment(self, comment_data: Dict[str, Any]) -> Comment:
        """Parse comment data into Comment object.

        Args:
            comment_data: Raw comment data from GraphQL

        Returns:
            Comment object
        """
        # Parse author
        author_data = comment_data.get("author") or {}
        author = self._parse_user(author_data)

        # Parse reactions
        reactions: Dict[str, int] = {}
        reactions_data = comment_data.get("reactions", {})
        if reactions_data:
            reaction_nodes = reactions_data.get("nodes", [])
            for reaction in reaction_nodes:
                if reaction and "content" in reaction:
                    content = reaction["content"]
                    reactions[content] = reactions.get(content, 0) + 1

        # Parse timestamps
        created_at = self._parse_datetime(comment_data["createdAt"])
        updated_at = self._parse_datetime(comment_data["updatedAt"])

        return Comment(
            comment_id=comment_data.get("databaseId", 0),
            author=author,
            body=comment_data["body"],
            created_at=created_at,
            updated_at=updated_at,
            reactions=reactions,
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


__all__ = ["IssueNormalizer"]
