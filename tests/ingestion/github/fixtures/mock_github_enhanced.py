"""Enhanced GitHub API mock for comprehensive testing.

Provides realistic GitHub API behavior including:
- REST and GraphQL API support
- Rate limiting with throttling
- Caching with ETag support
- Circuit breaker scenarios
- Pagination for large datasets
- Network failure simulation
- OAuth device flow simulation
"""

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import pytest


# ---------------------------------------------------------------------------
# Enhanced Rate Limiting
# ---------------------------------------------------------------------------


class EnhancedRateLimitSimulator:
    """Simulates GitHub rate limiting with realistic behavior."""

    def __init__(
        self,
        limit: int = 5000,
        graphql_limit: int = 5000,
        reset_interval_seconds: int = 3600,
    ):
        self.limit = limit
        self.remaining = limit
        self.graphql_limit = graphql_limit
        self.graphql_remaining = graphql_limit
        self.reset_at = datetime.now(timezone.utc) + timedelta(seconds=reset_interval_seconds)
        self.reset_interval = reset_interval_seconds
        self.request_history: List[Tuple[datetime, str]] = []

    def consume(self, cost: int = 1, api_type: str = "rest") -> Dict[str, Any]:
        """Consume rate limit and return headers."""
        now = datetime.now(timezone.utc)

        # Reset if past reset time
        if now >= self.reset_at:
            self.remaining = self.limit
            self.graphql_remaining = self.graphql_limit
            self.reset_at = now + timedelta(seconds=self.reset_interval)

        # Consume based on API type
        if api_type == "graphql":
            self.graphql_remaining = max(0, self.graphql_remaining - cost)
            remaining = self.graphql_remaining
            limit = self.graphql_limit
        else:
            self.remaining = max(0, self.remaining - cost)
            remaining = self.remaining
            limit = self.limit

        # Record request
        self.request_history.append((now, api_type))

        # Build rate limit headers
        return {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
            "X-RateLimit-Used": str(limit - remaining),
            "X-RateLimit-Resource": "core" if api_type == "rest" else "graphql",
        }

    def is_exhausted(self, api_type: str = "rest") -> bool:
        """Check if rate limit is exhausted."""
        if api_type == "graphql":
            return self.graphql_remaining == 0
        return self.remaining == 0

    def get_reset_in_seconds(self) -> int:
        """Get seconds until rate limit reset."""
        now = datetime.now(timezone.utc)
        if now >= self.reset_at:
            return 0
        return int((self.reset_at - now).total_seconds())


# ---------------------------------------------------------------------------
# Enhanced Caching with ETag
# ---------------------------------------------------------------------------


class EnhancedCacheSimulator:
    """Simulates caching with ETag support."""

    def __init__(self):
        self.cache: Dict[str, Tuple[str, Any, datetime]] = {}  # url -> (etag, data, timestamp)
        self.hit_count = 0
        self.miss_count = 0

    def generate_etag(self, data: Any) -> str:
        """Generate ETag for data."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, url: str, if_none_match: Optional[str] = None) -> Tuple[bool, Optional[Any], Optional[str]]:
        """Get cached data.

        Returns: (cache_hit, data, etag)
        """
        if url not in self.cache:
            self.miss_count += 1
            return (False, None, None)

        etag, data, timestamp = self.cache[url]

        # Check if ETag matches
        if if_none_match and if_none_match == etag:
            self.hit_count += 1
            return (True, None, etag)  # 304 Not Modified

        self.hit_count += 1
        return (True, data, etag)

    def set(self, url: str, data: Any):
        """Cache data with ETag."""
        etag = self.generate_etag(data)
        self.cache[url] = (etag, data, datetime.now(timezone.utc))

    def invalidate(self, url: str):
        """Invalidate cache entry."""
        if url in self.cache:
            del self.cache[url]

    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


# ---------------------------------------------------------------------------
# Circuit Breaker Simulator
# ---------------------------------------------------------------------------


class CircuitBreakerSimulator:
    """Simulates circuit breaker behavior."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "closed"  # closed, open, half-open
        self.last_failure_time: Optional[datetime] = None
        self.success_count_in_half_open = 0

    def record_success(self):
        """Record successful request."""
        if self.state == "half-open":
            self.success_count_in_half_open += 1
            if self.success_count_in_half_open >= 2:
                # Close circuit after 2 successes
                self.state = "closed"
                self.failure_count = 0
                self.success_count_in_half_open = 0
        elif self.state == "closed":
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record failed request."""
        self.last_failure_time = datetime.now(timezone.utc)
        self.failure_count += 1

        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            self.state = "open"
        elif self.state == "half-open":
            # Failure in half-open returns to open
            self.state = "open"
            self.success_count_in_half_open = 0

    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "half-open"
                    self.success_count_in_half_open = 0
                    return True
            return False

        # half-open state
        return True

    def get_state(self) -> str:
        """Get current circuit state."""
        return self.state


# ---------------------------------------------------------------------------
# Enhanced Mock GitHub API
# ---------------------------------------------------------------------------


class EnhancedMockGitHubAPI:
    """Enhanced mock GitHub API with realistic behavior."""

    def __init__(self):
        self.rate_limiter = EnhancedRateLimitSimulator()
        self.cache = EnhancedCacheSimulator()
        self.circuit_breaker = CircuitBreakerSimulator()

        # Data storage
        self.repositories: Dict[str, Dict[str, Any]] = {}
        self.file_contents: Dict[str, Dict[str, Any]] = {}  # repo_id -> {path -> content}
        self.commits: Dict[str, List[Dict[str, Any]]] = {}  # repo_id -> commits
        self.branches: Dict[str, List[Dict[str, Any]]] = {}  # repo_id -> branches

        # Configuration
        self.fail_next_request = False
        self.slow_response_delay = 0.0
        self.network_failure_mode = False

        # Metrics
        self.request_count = 0
        self.graphql_query_count = 0

    def add_repository(self, owner: str, repo: str, mock_repo_data: Dict[str, Any]):
        """Add a repository to the mock API."""
        repo_id = f"{owner}/{repo}"
        self.repositories[repo_id] = mock_repo_data
        self.file_contents[repo_id] = {}
        self.commits[repo_id] = []
        self.branches[repo_id] = [
            {
                "name": mock_repo_data.get("default_branch", "main"),
                "commit": {"sha": "abc123def456"},
                "protected": True,
            }
        ]

    def add_file_content(self, owner: str, repo: str, path: str, content: str, sha: str):
        """Add file content to repository."""
        repo_id = f"{owner}/{repo}"
        if repo_id not in self.file_contents:
            self.file_contents[repo_id] = {}
        self.file_contents[repo_id][path] = {
            "content": content,
            "sha": sha,
            "size": len(content.encode()),
        }

    def _check_preconditions(self):
        """Check request preconditions (rate limit, circuit breaker)."""
        # Check circuit breaker
        if not self.circuit_breaker.should_allow_request():
            raise Exception("503 Service Unavailable - Circuit breaker open")

        # Check network failure mode
        if self.network_failure_mode:
            self.circuit_breaker.record_failure()
            raise Exception("Network error: Connection timeout")

        # Check forced failure
        if self.fail_next_request:
            self.fail_next_request = False
            self.circuit_breaker.record_failure()
            raise Exception("500 Internal Server Error")

        # Check rate limit
        if self.rate_limiter.is_exhausted():
            raise Exception("403 Forbidden - Rate limit exceeded")

        # Simulate slow response
        if self.slow_response_delay > 0:
            time.sleep(self.slow_response_delay)

    def get_repository(
        self,
        owner: str,
        repo: str,
        if_none_match: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Get repository metadata.

        Returns: (response_data, headers)
        """
        self._check_preconditions()
        self.request_count += 1

        repo_id = f"{owner}/{repo}"
        url = f"/repos/{owner}/{repo}"

        # Check cache
        cache_hit, cached_data, etag = self.cache.get(url, if_none_match)
        if cache_hit and cached_data is None:
            # 304 Not Modified
            headers = self.rate_limiter.consume(cost=1, api_type="rest")
            headers["ETag"] = etag
            self.circuit_breaker.record_success()
            return ({}, headers)

        if repo_id not in self.repositories:
            self.circuit_breaker.record_failure()
            raise Exception("404 Not Found")

        data = self.repositories[repo_id].copy()

        # Cache the response
        self.cache.set(url, data)
        _, _, new_etag = self.cache.get(url)

        headers = self.rate_limiter.consume(cost=1, api_type="rest")
        headers["ETag"] = new_etag
        headers["Cache-Control"] = "max-age=60"

        self.circuit_breaker.record_success()
        return (data, headers)

    def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str = "main",
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Get file content.

        Returns: (response_data, headers)
        """
        self._check_preconditions()
        self.request_count += 1

        repo_id = f"{owner}/{repo}"

        if repo_id not in self.file_contents or path not in self.file_contents[repo_id]:
            self.circuit_breaker.record_failure()
            raise Exception("404 Not Found")

        file_data = self.file_contents[repo_id][path].copy()

        headers = self.rate_limiter.consume(cost=1, api_type="rest")
        self.circuit_breaker.record_success()

        return (file_data, headers)

    def graphql_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Execute GraphQL query.

        Returns: (response_data, headers)
        """
        self._check_preconditions()
        self.graphql_query_count += 1

        # Calculate query cost (simplified)
        query_cost = max(1, len(query) // 100)

        # Check if it's a repository query
        if "repository(" in query:
            # Extract owner/repo from variables or query
            owner = variables.get("owner") if variables else None
            repo_name = variables.get("name") if variables else None

            if owner and repo_name:
                repo_id = f"{owner}/{repo_name}"
                if repo_id in self.repositories:
                    data = {
                        "data": {
                            "repository": self._build_graphql_repository(repo_id)
                        }
                    }
                else:
                    data = {"data": {"repository": None}}
            else:
                data = {"errors": [{"message": "Missing required variables"}]}
        else:
            # Generic response
            data = {"data": {}}

        headers = self.rate_limiter.consume(cost=query_cost, api_type="graphql")
        headers["X-RateLimit-Cost"] = str(query_cost)
        self.circuit_breaker.record_success()

        return (data, headers)

    def _build_graphql_repository(self, repo_id: str) -> Dict[str, Any]:
        """Build GraphQL repository response."""
        repo_data = self.repositories[repo_id]

        # Build file tree
        files = []
        if repo_id in self.file_contents:
            for path, file_data in self.file_contents[repo_id].items():
                files.append({
                    "path": path,
                    "oid": file_data["sha"],
                    "object": {
                        "text": file_data["content"],
                        "byteSize": file_data["size"],
                    },
                })

        return {
            "name": repo_data["name"],
            "owner": repo_data["owner"],
            "description": repo_data.get("description"),
            "defaultBranchRef": {
                "name": repo_data.get("default_branch", "main"),
                "target": {
                    "oid": "abc123def456",
                    "tree": {
                        "entries": files[:100],  # Limit for pagination
                    },
                },
            },
        }

    def list_commits(
        self,
        owner: str,
        repo: str,
        sha: Optional[str] = None,
        per_page: int = 30,
        page: int = 1,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """List commits with pagination.

        Returns: (commits, headers)
        """
        self._check_preconditions()
        self.request_count += 1

        repo_id = f"{owner}/{repo}"

        if repo_id not in self.commits:
            self.circuit_breaker.record_failure()
            raise Exception("404 Not Found")

        all_commits = self.commits[repo_id]

        # Pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        commits = all_commits[start_idx:end_idx]

        headers = self.rate_limiter.consume(cost=1, api_type="rest")

        # Add pagination headers
        total_pages = (len(all_commits) + per_page - 1) // per_page
        if page < total_pages:
            headers["Link"] = f'<https://api.github.com/repos/{owner}/{repo}/commits?page={page+1}>; rel="next"'

        self.circuit_breaker.record_success()
        return (commits, headers)

    def trigger_failure(self, count: int = 1):
        """Trigger failure for next N requests."""
        self.fail_next_request = True

    def enable_network_failure(self):
        """Enable network failure mode."""
        self.network_failure_mode = True

    def disable_network_failure(self):
        """Disable network failure mode."""
        self.network_failure_mode = False

    def set_slow_response(self, delay_seconds: float):
        """Set slow response delay."""
        self.slow_response_delay = delay_seconds

    def reset(self):
        """Reset API state."""
        self.rate_limiter = EnhancedRateLimitSimulator()
        self.cache.clear()
        self.circuit_breaker = CircuitBreakerSimulator()
        self.request_count = 0
        self.graphql_query_count = 0
        self.fail_next_request = False
        self.slow_response_delay = 0.0
        self.network_failure_mode = False


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def enhanced_mock_github_api():
    """Provide enhanced mock GitHub API for testing."""
    api = EnhancedMockGitHubAPI()

    # Add some default repositories
    api.add_repository(
        "octocat",
        "Hello-World",
        {
            "name": "Hello-World",
            "full_name": "octocat/Hello-World",
            "owner": {"login": "octocat"},
            "description": "Test repository",
            "private": False,
            "default_branch": "main",
        },
    )

    return api


@pytest.fixture
def rate_limit_exhausted_api():
    """Provide API with exhausted rate limit."""
    api = EnhancedMockGitHubAPI()
    api.rate_limiter.remaining = 0
    api.rate_limiter.graphql_remaining = 0
    return api


@pytest.fixture
def circuit_breaker_open_api():
    """Provide API with open circuit breaker."""
    api = EnhancedMockGitHubAPI()
    # Trigger enough failures to open circuit
    for _ in range(5):
        try:
            api.trigger_failure()
            api.get_repository("test", "test")
        except:
            pass
    return api
