"""GitHub API client wrapper for repository operations.

This module provides a high-level interface to GitHub's REST API for
fetching repository metadata, validating tokens, and managing repository access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from .descriptor import VisibilityType


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RepositoryInfo(BaseModel):
    """GitHub repository metadata."""

    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    full_name: str = Field(..., description="Full repository name (owner/repo)")
    description: Optional[str] = Field(default=None, description="Repository description")
    visibility: VisibilityType = Field(..., description="Repository visibility")
    default_branch: str = Field(..., description="Default branch name")
    is_private: bool = Field(..., description="Whether repository is private")
    is_fork: bool = Field(..., description="Whether repository is a fork")
    is_archived: bool = Field(..., description="Whether repository is archived")
    created_at: datetime = Field(..., description="Repository creation date")
    updated_at: datetime = Field(..., description="Last update date")
    pushed_at: Optional[datetime] = Field(default=None, description="Last push date")
    size: int = Field(..., description="Repository size in KB")
    language: Optional[str] = Field(default=None, description="Primary language")
    has_issues: bool = Field(default=False, description="Has issues enabled")
    has_wiki: bool = Field(default=False, description="Has wiki enabled")


class BranchInfo(BaseModel):
    """GitHub branch metadata."""

    name: str = Field(..., description="Branch name")
    commit_sha: str = Field(..., description="Latest commit SHA")
    protected: bool = Field(default=False, description="Whether branch is protected")


class TokenInfo(BaseModel):
    """GitHub token metadata."""

    scopes: List[str] = Field(default_factory=list, description="Token scopes")
    rate_limit: int = Field(..., description="Rate limit")
    rate_remaining: int = Field(..., description="Remaining rate limit")
    rate_reset_at: datetime = Field(..., description="Rate limit reset time")


# ---------------------------------------------------------------------------
# GitHub API Client
# ---------------------------------------------------------------------------


@dataclass
class GitHubAPIClient:
    """High-level GitHub API client."""

    token: str
    github_host: str = "github.com"
    api_base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        """Initialize API client."""
        if self.api_base_url is None:
            if self.github_host == "github.com":
                self.api_base_url = "https://api.github.com"
            else:
                # GitHub Enterprise Server
                self.api_base_url = f"https://{self.github_host}/api/v3"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Futurnal-GitHub-Connector",
            }
        )

    def get_repository(self, owner: str, repo: str) -> RepositoryInfo:
        """Get repository metadata.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            RepositoryInfo with repository metadata

        Raises:
            RuntimeError: If repository not found or access denied
        """
        url = f"{self.api_base_url}/repos/{owner}/{repo}"
        response = self._request("GET", url)

        data = response.json()

        # Determine visibility
        visibility = VisibilityType.PUBLIC
        if data.get("private"):
            visibility = VisibilityType.PRIVATE
        if data.get("visibility") == "internal":
            visibility = VisibilityType.INTERNAL

        return RepositoryInfo(
            owner=owner,
            repo=repo,
            full_name=data["full_name"],
            description=data.get("description"),
            visibility=visibility,
            default_branch=data.get("default_branch", "main"),
            is_private=data.get("private", False),
            is_fork=data.get("fork", False),
            is_archived=data.get("archived", False),
            created_at=datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                data["updated_at"].replace("Z", "+00:00")
            ),
            pushed_at=datetime.fromisoformat(data["pushed_at"].replace("Z", "+00:00"))
            if data.get("pushed_at")
            else None,
            size=data.get("size", 0),
            language=data.get("language"),
            has_issues=data.get("has_issues", False),
            has_wiki=data.get("has_wiki", False),
        )

    def list_branches(
        self, owner: str, repo: str, *, limit: int = 100
    ) -> List[BranchInfo]:
        """List repository branches.

        Args:
            owner: Repository owner
            repo: Repository name
            limit: Maximum number of branches to fetch

        Returns:
            List of BranchInfo

        Raises:
            RuntimeError: If request fails
        """
        url = f"{self.api_base_url}/repos/{owner}/{repo}/branches"
        params = {"per_page": min(limit, 100)}

        branches = []
        while len(branches) < limit:
            response = self._request("GET", url, params=params)
            data = response.json()

            if not data:
                break

            for item in data:
                branches.append(
                    BranchInfo(
                        name=item["name"],
                        commit_sha=item["commit"]["sha"],
                        protected=item.get("protected", False),
                    )
                )

            # Check for pagination
            if "next" not in response.links:
                break
            url = response.links["next"]["url"]

        return branches[:limit]

    def validate_token(self) -> TokenInfo:
        """Validate token and get metadata.

        Returns:
            TokenInfo with token scopes and rate limits

        Raises:
            RuntimeError: If token is invalid
        """
        url = f"{self.api_base_url}/rate_limit"
        response = self._request("GET", url)

        # Parse scopes from header
        scopes = []
        scope_header = response.headers.get("X-OAuth-Scopes", "")
        if scope_header:
            scopes = [s.strip() for s in scope_header.split(",")]

        # Parse rate limit
        data = response.json()
        rate_data = data.get("rate", {})

        return TokenInfo(
            scopes=scopes,
            rate_limit=rate_data.get("limit", 5000),
            rate_remaining=rate_data.get("remaining", 0),
            rate_reset_at=datetime.fromtimestamp(rate_data.get("reset", 0)),
        )

    def test_repository_access(self, owner: str, repo: str) -> bool:
        """Test if token has access to repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            True if access granted, False otherwise
        """
        try:
            self.get_repository(owner, repo)
            return True
        except RuntimeError:
            return False

    def verify_required_scopes(
        self, required_scopes: List[str], visibility: VisibilityType
    ) -> bool:
        """Verify token has required scopes for repository visibility.

        Args:
            required_scopes: List of required scopes
            visibility: Repository visibility

        Returns:
            True if token has required scopes

        Raises:
            RuntimeError: If token validation fails
        """
        token_info = self.validate_token()
        token_scopes = set(token_info.scopes)

        # For private repositories, require 'repo' scope (classic PAT)
        # Fine-grained PATs don't use traditional scopes, so check is only for classic PATs
        if visibility == VisibilityType.PRIVATE:
            # If there are scopes and 'repo' isn't one, fail for classic PAT
            # Fine-grained PATs have empty scopes but may still have access
            if token_scopes and "repo" not in token_scopes:
                return False

        # For public repositories, no special scope is required
        # Fine-grained PATs can access public repos without traditional OAuth scopes
        # Classic PATs with no scopes can still read public repos via API

        # Check additional required scopes
        for scope in required_scopes:
            if scope not in token_scopes:
                return False

        return True

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            data: Request body

        Returns:
            Response object

        Raises:
            RuntimeError: If request fails after retries
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.timeout,
                )

                # Handle rate limiting
                if response.status_code == 403:
                    rate_remaining = response.headers.get("X-RateLimit-Remaining", "0")
                    if rate_remaining == "0":
                        reset_time = int(response.headers.get("X-RateLimit-Reset", "0"))
                        wait_time = max(0, reset_time - time.time())
                        raise RuntimeError(
                            f"Rate limit exceeded. Reset in {wait_time:.0f} seconds."
                        )

                # Handle other errors
                if response.status_code == 401:
                    raise RuntimeError(
                        "Authentication failed. Token may be invalid or expired."
                    )
                elif response.status_code == 404:
                    raise RuntimeError("Repository not found or access denied.")
                elif response.status_code >= 500:
                    # Server error, retry
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                    raise RuntimeError(
                        f"GitHub API error: {response.status_code} - {response.text}"
                    )

                response.raise_for_status()
                return response

            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue

        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def create_api_client(
    token: str,
    github_host: str = "github.com",
    api_base_url: Optional[str] = None,
) -> GitHubAPIClient:
    """Create GitHub API client (convenience function).

    Args:
        token: GitHub access token
        github_host: GitHub hostname
        api_base_url: Custom API base URL

    Returns:
        GitHubAPIClient instance
    """
    return GitHubAPIClient(
        token=token, github_host=github_host, api_base_url=api_base_url
    )


__all__ = [
    "BranchInfo",
    "GitHubAPIClient",
    "RepositoryInfo",
    "TokenInfo",
    "create_api_client",
]
