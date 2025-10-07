"""GitHub API client with rate limiting, caching, and error handling using GitHubKit.

This module implements a robust GitHub API client manager that handles rate limiting,
request caching, exponential backoff, and error recovery. It provides intelligent
request queuing to maximize API efficiency while respecting GitHub's rate limits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from githubkit import GitHub, TokenAuthStrategy
from githubkit.exception import GitHubException
from pydantic import BaseModel, Field

from .credential_manager import (
    GitHubCredentialManager,
    OAuthToken,
    OAuthTokens,
    PersonalAccessToken,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RateLimitInfo(BaseModel):
    """Rate limit information from GitHub."""

    limit: int = Field(..., description="Total rate limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_at: datetime = Field(..., description="Rate limit reset time")
    used: int = Field(..., description="Used requests")

    @property
    def reset_in_seconds(self) -> int:
        """Seconds until rate limit resets."""
        now = datetime.now(timezone.utc)
        if self.reset_at.tzinfo is None:
            reset_at = self.reset_at.replace(tzinfo=timezone.utc)
        else:
            reset_at = self.reset_at
        return max(0, int((reset_at - now).total_seconds()))

    @property
    def usage_percentage(self) -> float:
        """Percentage of rate limit used."""
        return (self.used / self.limit) * 100 if self.limit > 0 else 0.0


class GraphQLRateLimitInfo(BaseModel):
    """GraphQL-specific rate limit info."""

    limit: int = Field(..., description="Total points limit")
    remaining: int = Field(..., description="Remaining points")
    reset_at: datetime = Field(..., description="Rate limit reset time")
    cost: int = Field(default=0, description="Cost of last query")
    node_count: int = Field(default=0, description="Nodes returned in last query")


class APIRequestMetrics(BaseModel):
    """Metrics for API request tracking."""

    request_id: str = Field(..., description="Unique request identifier")
    method: str = Field(..., description="HTTP method")
    endpoint: str = Field(..., description="API endpoint")
    api_type: str = Field(..., description="API type (rest or graphql)")
    duration_ms: float = Field(..., description="Request duration in milliseconds")
    status_code: int = Field(..., description="HTTP status code")
    rate_limit_remaining: int = Field(..., description="Rate limit remaining after request")
    cached: bool = Field(..., description="Whether response was cached")
    retries: int = Field(default=0, description="Number of retries")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CacheEntry(BaseModel):
    """Cache entry for API responses."""

    key: str = Field(..., description="Cache key hash")
    response: Dict[str, Any] = Field(..., description="Cached response data")
    etag: Optional[str] = Field(default=None, description="ETag for conditional requests")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreaker:
    """Circuit breaker for failing API endpoints."""

    failure_threshold: int = 5
    timeout_seconds: int = 60
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open

    def record_failure(self) -> None:
        """Record an API failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def record_success(self) -> None:
        """Record an API success."""
        self.failure_count = 0
        self.state = "closed"

    def can_attempt(self) -> bool:
        """Check if we can attempt a request."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout has elapsed
            if self.last_failure_time:
                now = datetime.now(timezone.utc)
                elapsed = (now - self.last_failure_time).total_seconds()
                if elapsed > self.timeout_seconds:
                    self.state = "half_open"
                    return True
            return False

        # half_open state
        return True


# ---------------------------------------------------------------------------
# Exponential Backoff
# ---------------------------------------------------------------------------


@dataclass
class ExponentialBackoff:
    """Exponential backoff with jitter for retries."""

    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    attempt: int = 0

    def next_delay(self) -> float:
        """Calculate next delay with exponential backoff."""
        delay = min(
            self.base_delay * (self.multiplier ** self.attempt),
            self.max_delay
        )

        if self.jitter:
            # ±25% jitter
            delay *= (0.75 + random.random() * 0.5)

        self.attempt += 1
        return delay

    def reset(self) -> None:
        """Reset backoff state."""
        self.attempt = 0


# ---------------------------------------------------------------------------
# GitHub API Client Manager
# ---------------------------------------------------------------------------


@dataclass
class GitHubAPIClientManager:
    """Manages GitHub API clients with rate limiting and caching."""

    credential_manager: GitHubCredentialManager
    cache_dir: Optional[Path] = None
    max_concurrent_requests: int = 80  # Leave buffer below 100
    rate_limit_threshold: float = 0.9  # Start throttling at 90%
    _semaphore: Optional[asyncio.Semaphore] = field(default=None, init=False)
    _rate_limit_info: Dict[str, RateLimitInfo] = field(default_factory=dict, init=False)
    _circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=dict, init=False)
    _metrics: List[APIRequestMetrics] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize API client manager."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".futurnal" / "cache" / "github"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize semaphore (will be created in async context)
        self._semaphore = None

    async def _ensure_semaphore(self) -> asyncio.Semaphore:
        """Ensure semaphore is initialized in async context."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._semaphore

    def get_client(
        self,
        credential_id: str,
        github_host: str = "github.com",
    ) -> GitHub:
        """Get authenticated GitHub client.

        Args:
            credential_id: Credential identifier
            github_host: GitHub hostname (default: github.com)

        Returns:
            Authenticated GitHub client

        Raises:
            ValueError: If credential type is not supported
        """
        # Retrieve credentials
        credentials = self.credential_manager.retrieve_credentials(credential_id)

        # Extract token based on credential type
        if isinstance(credentials, OAuthTokens):
            token = credentials.access_token
        elif isinstance(credentials, OAuthToken):
            token = credentials.token
        elif isinstance(credentials, PersonalAccessToken):
            token = credentials.token
        else:
            raise ValueError(f"Unknown credential type: {type(credentials)}")

        # Configure API base URL for Enterprise
        config: Dict[str, Any] = {}
        if github_host != "github.com":
            config["base_url"] = f"https://{github_host}/api/v3"

        # Create client
        return GitHub(TokenAuthStrategy(token), **config)

    async def rest_request(
        self,
        *,
        credential_id: str,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        github_host: str = "github.com",
    ) -> Dict[str, Any]:
        """Make REST API request with rate limiting and caching.

        Args:
            credential_id: Credential identifier
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /repos/owner/repo)
            params: Query parameters
            data: Request body
            use_cache: Whether to use caching (GET requests only)
            github_host: GitHub hostname

        Returns:
            Response data as dictionary

        Raises:
            Exception: If request fails after retries
        """
        semaphore = await self._ensure_semaphore()

        async with semaphore:
            # Check rate limit
            await self._check_rate_limit(credential_id, "rest")

            # Check cache for GET requests
            if use_cache and method == "GET":
                cached = self._get_from_cache(endpoint, params)
                if cached:
                    logger.debug(f"Cache hit for {endpoint}")
                    return cached

            # Get GitHub client
            client = self.get_client(credential_id, github_host)
            start_time = time.time()
            request_id = hashlib.sha256(
                f"{credential_id}:{endpoint}:{time.time()}".encode()
            ).hexdigest()[:16]

            try:
                # Make request with retry logic
                response_data = await self._execute_rest_request(
                    client, method, endpoint, params, data, use_cache
                )

                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                self._record_metrics(
                    request_id=request_id,
                    method=method,
                    endpoint=endpoint,
                    api_type="rest",
                    duration_ms=duration_ms,
                    status_code=200,
                    cached=False,
                    rate_limit_remaining=self._rate_limit_info.get(
                        f"{credential_id}:rest", RateLimitInfo(limit=5000, remaining=0, reset_at=datetime.now(timezone.utc), used=0)
                    ).remaining,
                )

                return response_data

            except Exception as e:
                self._handle_api_error(e, endpoint)
                raise

    async def _execute_rest_request(
        self,
        client: GitHub,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        use_cache: bool,
    ) -> Dict[str, Any]:
        """Execute REST request with exponential backoff.

        Args:
            client: GitHub client
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            use_cache: Whether to use ETag caching

        Returns:
            Response data
        """
        backoff = ExponentialBackoff()
        last_error = None

        # Get cached ETag for conditional request
        cached_etag = None
        if use_cache and method == "GET":
            cached_entry = self._get_cache_entry(endpoint, params)
            if cached_entry:
                cached_etag = cached_entry.etag

        for attempt in range(3):  # Max 3 attempts
            try:
                # Prepare headers
                headers = {}
                if cached_etag:
                    headers["If-None-Match"] = cached_etag

                # Make request
                response = await client.arequest(
                    method,
                    endpoint,
                    params=params,
                    json=data,
                    headers=headers if headers else None,
                )

                # Update rate limit from headers
                if hasattr(response, 'headers'):
                    self._update_rate_limit_from_headers(
                        client, response.headers
                    )

                # Handle 304 Not Modified
                if hasattr(response, 'status_code') and response.status_code == 304:
                    cached_entry = self._get_cache_entry(endpoint, params)
                    if cached_entry:
                        logger.debug(f"304 Not Modified, using cached data for {endpoint}")
                        return cached_entry.response

                # Parse response
                if hasattr(response, 'parsed_data'):
                    response_data = response.parsed_data
                    if isinstance(response_data, dict):
                        result = response_data
                    else:
                        # Convert Pydantic model to dict
                        result = response_data.model_dump() if hasattr(response_data, 'model_dump') else {}
                else:
                    result = {}

                # Cache GET responses
                if method == "GET" and use_cache:
                    etag = response.headers.get("ETag") if hasattr(response, 'headers') else None
                    self._store_in_cache(
                        endpoint, params, result, etag=etag
                    )

                # Record success in circuit breaker
                if endpoint in self._circuit_breakers:
                    self._circuit_breakers[endpoint].record_success()

                return result

            except GitHubException as e:
                last_error = e

                # Check if error is retriable (502, 503, 504)
                status_code = getattr(e, 'status_code', 0)
                if status_code in (502, 503, 504) and attempt < 2:
                    delay = backoff.next_delay()
                    logger.warning(
                        f"Server error {status_code}, retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

            except Exception as e:
                last_error = e

                # Check if this is a retriable error (502, 503, 504)
                status_code = getattr(e, 'status_code', 0)
                if status_code in (502, 503, 504) and attempt < 2:
                    delay = backoff.next_delay()
                    logger.warning(
                        f"Server error {status_code}, retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

        # If we get here, all retries failed
        raise RuntimeError(f"Request failed after retries: {last_error}")

    async def graphql_request(
        self,
        *,
        credential_id: str,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        github_host: str = "github.com",
    ) -> Dict[str, Any]:
        """Make GraphQL API request with rate limiting.

        Args:
            credential_id: Credential identifier
            query: GraphQL query string
            variables: Query variables
            github_host: GitHub hostname

        Returns:
            Response data

        Raises:
            Exception: If request fails
        """
        semaphore = await self._ensure_semaphore()

        async with semaphore:
            # Check rate limit
            await self._check_rate_limit(credential_id, "graphql")

            # Get GitHub client
            client = self.get_client(credential_id, github_host)
            start_time = time.time()
            request_id = hashlib.sha256(
                f"{credential_id}:graphql:{time.time()}".encode()
            ).hexdigest()[:16]

            try:
                # Make GraphQL request
                response = await client.async_graphql(query, variables)

                # Extract rate limit from response if available
                if isinstance(response, dict) and "data" in response:
                    if "rateLimit" in response["data"]:
                        self._update_graphql_rate_limit(
                            credential_id, response["data"]["rateLimit"]
                        )

                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                self._record_metrics(
                    request_id=request_id,
                    method="POST",
                    endpoint="/graphql",
                    api_type="graphql",
                    duration_ms=duration_ms,
                    status_code=200,
                    cached=False,
                    rate_limit_remaining=self._rate_limit_info.get(
                        f"{credential_id}:graphql", RateLimitInfo(limit=5000, remaining=0, reset_at=datetime.now(timezone.utc), used=0)
                    ).remaining,
                )

                return response

            except Exception as e:
                self._handle_api_error(e, "/graphql")
                raise

    async def _check_rate_limit(
        self,
        credential_id: str,
        api_type: str,
    ) -> None:
        """Check if we should throttle requests.

        Args:
            credential_id: Credential identifier
            api_type: API type (rest or graphql)
        """
        rate_limit = self._rate_limit_info.get(f"{credential_id}:{api_type}")

        if not rate_limit:
            return  # No rate limit info yet, proceed

        # If we're close to limit, wait
        if rate_limit.usage_percentage > self.rate_limit_threshold * 100:
            wait_time = min(rate_limit.reset_in_seconds, 60)  # Max 1 minute wait
            logger.warning(
                f"Approaching rate limit ({rate_limit.usage_percentage:.1f}%), "
                f"waiting {wait_time}s"
            )
            await asyncio.sleep(wait_time)

    def _update_rate_limit_from_headers(
        self,
        client: GitHub,
        headers: Dict[str, str],
    ) -> None:
        """Update rate limit info from REST API response headers.

        Args:
            client: GitHub client (to extract credential ID)
            headers: Response headers
        """
        if "X-RateLimit-Limit" in headers:
            # Extract credential ID from client (use hash for privacy)
            credential_id = hashlib.sha256(
                str(id(client)).encode()
            ).hexdigest()[:16]

            self._rate_limit_info[f"{credential_id}:rest"] = RateLimitInfo(
                limit=int(headers["X-RateLimit-Limit"]),
                remaining=int(headers["X-RateLimit-Remaining"]),
                reset_at=datetime.fromtimestamp(
                    int(headers["X-RateLimit-Reset"]),
                    tz=timezone.utc
                ),
                used=int(headers["X-RateLimit-Limit"]) -
                     int(headers["X-RateLimit-Remaining"]),
            )

    def _update_graphql_rate_limit(
        self,
        credential_id: str,
        rate_limit_data: Dict[str, Any],
    ) -> None:
        """Update rate limit info from GraphQL response.

        Args:
            credential_id: Credential identifier
            rate_limit_data: Rate limit data from GraphQL response
        """
        reset_at_str = rate_limit_data.get("resetAt", "")
        if reset_at_str:
            # Parse ISO format timestamp
            reset_at = datetime.fromisoformat(
                reset_at_str.replace("Z", "+00:00")
            )
        else:
            reset_at = datetime.now(timezone.utc) + timedelta(hours=1)

        self._rate_limit_info[f"{credential_id}:graphql"] = RateLimitInfo(
            limit=rate_limit_data.get("limit", 5000),
            remaining=rate_limit_data.get("remaining", 0),
            reset_at=reset_at,
            used=rate_limit_data.get("used", 0),
        )

    def _get_from_cache(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve response from cache.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Cached response data or None
        """
        cache_entry = self._get_cache_entry(endpoint, params)
        if not cache_entry:
            return None

        # Check expiration
        now = datetime.now(timezone.utc)
        if now > cache_entry.expires_at:
            # Cache expired, delete it
            cache_path = self._get_cache_path(endpoint, params)
            if cache_path.exists():
                cache_path.unlink()
            return None

        return cache_entry.response

    def _get_cache_entry(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[CacheEntry]:
        """Get cache entry without expiration check.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            CacheEntry or None
        """
        cache_path = self._get_cache_path(endpoint, params)

        if not cache_path.exists():
            return None

        try:
            entry = CacheEntry.model_validate_json(cache_path.read_text())
            return entry
        except Exception as e:
            logger.warning(f"Failed to load cache entry: {e}")
            return None

    def _store_in_cache(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]],
        response: Dict[str, Any],
        etag: Optional[str] = None,
        ttl_seconds: int = 300,  # 5 minutes default
    ) -> None:
        """Store response in cache.

        Args:
            endpoint: API endpoint
            params: Query parameters
            response: Response data
            etag: ETag header value
            ttl_seconds: Time to live in seconds
        """
        cache_key = self._compute_cache_key(endpoint, params)
        cache_path = self._get_cache_path(endpoint, params)

        entry = CacheEntry(
            key=cache_key,
            response=response,
            etag=etag,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            created_at=datetime.now(timezone.utc),
        )

        try:
            cache_path.write_text(entry.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Failed to store cache entry: {e}")

    def _get_cache_path(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Get cache file path.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Path to cache file
        """
        cache_key = self._compute_cache_key(endpoint, params)
        return self.cache_dir / f"{cache_key}.json"

    def _compute_cache_key(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compute cache key from endpoint and parameters.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Cache key hash
        """
        key_parts = [endpoint]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        key_string = "".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _handle_api_error(
        self,
        error: Exception,
        endpoint: str,
    ) -> None:
        """Handle API errors with circuit breaker pattern.

        Args:
            error: Exception that occurred
            endpoint: API endpoint
        """
        # Update circuit breaker
        if endpoint not in self._circuit_breakers:
            self._circuit_breakers[endpoint] = CircuitBreaker()

        self._circuit_breakers[endpoint].record_failure()

        # Log error (without sensitive data)
        logger.error(
            f"GitHub API error: {type(error).__name__}",
            extra={"endpoint": endpoint}
        )

    def _record_metrics(
        self,
        *,
        request_id: str,
        method: str,
        endpoint: str,
        api_type: str,
        duration_ms: float,
        status_code: int,
        cached: bool,
        rate_limit_remaining: int = 0,
        retries: int = 0,
    ) -> None:
        """Record request metrics.

        Args:
            request_id: Unique request ID
            method: HTTP method
            endpoint: API endpoint
            api_type: API type (rest or graphql)
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code
            cached: Whether response was cached
            rate_limit_remaining: Remaining rate limit
            retries: Number of retries
        """
        metric = APIRequestMetrics(
            request_id=request_id,
            method=method,
            endpoint=endpoint,
            api_type=api_type,
            duration_ms=duration_ms,
            status_code=status_code,
            rate_limit_remaining=rate_limit_remaining,
            cached=cached,
            retries=retries,
        )
        self._metrics.append(metric)

        # Keep only last 1000 metrics
        if len(self._metrics) > 1000:
            self._metrics = self._metrics[-1000:]

    def get_metrics(self) -> List[APIRequestMetrics]:
        """Get collected request metrics.

        Returns:
            List of request metrics
        """
        return self._metrics.copy()

    def get_rate_limit_status(self, credential_id: str) -> Dict[str, Optional[RateLimitInfo]]:
        """Get current rate limit status for a credential.

        Args:
            credential_id: Credential identifier

        Returns:
            Dictionary with 'rest' and 'graphql' rate limit info
        """
        return {
            "rest": self._rate_limit_info.get(f"{credential_id}:rest"),
            "graphql": self._rate_limit_info.get(f"{credential_id}:graphql"),
        }


__all__ = [
    "APIRequestMetrics",
    "CacheEntry",
    "CircuitBreaker",
    "ExponentialBackoff",
    "GitHubAPIClientManager",
    "GraphQLRateLimitInfo",
    "RateLimitInfo",
]
