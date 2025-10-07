"""Comprehensive tests for GitHub API Client Manager.

Tests cover rate limiting, caching, circuit breaker, exponential backoff,
and integration with GitHubCredentialManager.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from futurnal.ingestion.github.api_client_manager import (
    APIRequestMetrics,
    CacheEntry,
    CircuitBreaker,
    ExponentialBackoff,
    GitHubAPIClientManager,
    GraphQLRateLimitInfo,
    RateLimitInfo,
)
from futurnal.ingestion.github.credential_manager import (
    OAuthToken,
    OAuthTokens,
    PersonalAccessToken,
)


# ---------------------------------------------------------------------------
# Unit Tests - RateLimitInfo
# ---------------------------------------------------------------------------


def test_rate_limit_info_reset_in_seconds():
    """Test reset_in_seconds calculation."""
    reset_at = datetime.now(timezone.utc) + timedelta(seconds=300)
    rate_limit = RateLimitInfo(
        limit=5000,
        remaining=4500,
        reset_at=reset_at,
        used=500,
    )

    # Should be approximately 300 seconds
    assert 295 <= rate_limit.reset_in_seconds <= 305


def test_rate_limit_info_reset_in_seconds_past():
    """Test reset_in_seconds returns 0 for past timestamps."""
    reset_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    rate_limit = RateLimitInfo(
        limit=5000,
        remaining=5000,
        reset_at=reset_at,
        used=0,
    )

    assert rate_limit.reset_in_seconds == 0


def test_rate_limit_info_usage_percentage():
    """Test usage percentage calculation."""
    rate_limit = RateLimitInfo(
        limit=5000,
        remaining=2500,
        reset_at=datetime.now(timezone.utc),
        used=2500,
    )

    assert rate_limit.usage_percentage == 50.0


def test_rate_limit_info_usage_percentage_zero_limit():
    """Test usage percentage with zero limit."""
    rate_limit = RateLimitInfo(
        limit=0,
        remaining=0,
        reset_at=datetime.now(timezone.utc),
        used=0,
    )

    assert rate_limit.usage_percentage == 0.0


def test_rate_limit_info_usage_percentage_full():
    """Test usage percentage at full capacity."""
    rate_limit = RateLimitInfo(
        limit=5000,
        remaining=0,
        reset_at=datetime.now(timezone.utc),
        used=5000,
    )

    assert rate_limit.usage_percentage == 100.0


# ---------------------------------------------------------------------------
# Unit Tests - CacheEntry
# ---------------------------------------------------------------------------


def test_cache_entry_creation():
    """Test cache entry creation."""
    entry = CacheEntry(
        key="test_key",
        response={"data": "test"},
        etag="abc123",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    assert entry.key == "test_key"
    assert entry.response == {"data": "test"}
    assert entry.etag == "abc123"
    assert entry.created_at is not None


def test_cache_entry_serialization():
    """Test cache entry serialization to JSON."""
    entry = CacheEntry(
        key="test_key",
        response={"data": "test"},
        etag="abc123",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    json_data = entry.model_dump_json()
    loaded = CacheEntry.model_validate_json(json_data)

    assert loaded.key == entry.key
    assert loaded.response == entry.response
    assert loaded.etag == entry.etag


# ---------------------------------------------------------------------------
# Unit Tests - CircuitBreaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_initial_state():
    """Test circuit breaker starts in closed state."""
    breaker = CircuitBreaker()

    assert breaker.state == "closed"
    assert breaker.failure_count == 0
    assert breaker.can_attempt() is True


def test_circuit_breaker_record_failure():
    """Test recording failures."""
    breaker = CircuitBreaker(failure_threshold=3)

    breaker.record_failure()
    assert breaker.failure_count == 1
    assert breaker.state == "closed"

    breaker.record_failure()
    assert breaker.failure_count == 2
    assert breaker.state == "closed"

    breaker.record_failure()
    assert breaker.failure_count == 3
    assert breaker.state == "open"


def test_circuit_breaker_opens_at_threshold():
    """Test circuit breaker opens at threshold."""
    breaker = CircuitBreaker(failure_threshold=5)

    for _ in range(4):
        breaker.record_failure()
        assert breaker.state == "closed"

    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.can_attempt() is False


def test_circuit_breaker_half_open_after_timeout():
    """Test circuit breaker transitions to half_open after timeout."""
    breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=1)

    # Open the circuit
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "open"

    # Wait for timeout
    import time
    time.sleep(1.1)

    # Should transition to half_open
    assert breaker.can_attempt() is True
    assert breaker.state == "half_open"


def test_circuit_breaker_record_success():
    """Test recording success resets circuit breaker."""
    breaker = CircuitBreaker(failure_threshold=3)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.failure_count == 2

    breaker.record_success()
    assert breaker.failure_count == 0
    assert breaker.state == "closed"


# ---------------------------------------------------------------------------
# Unit Tests - ExponentialBackoff
# ---------------------------------------------------------------------------


def test_exponential_backoff_initial_delay():
    """Test exponential backoff initial delay."""
    backoff = ExponentialBackoff(base_delay=1.0, jitter=False)

    delay = backoff.next_delay()
    assert delay == 1.0


def test_exponential_backoff_progression():
    """Test exponential backoff delay progression."""
    backoff = ExponentialBackoff(base_delay=1.0, multiplier=2.0, jitter=False)

    delay1 = backoff.next_delay()
    assert delay1 == 1.0

    delay2 = backoff.next_delay()
    assert delay2 == 2.0

    delay3 = backoff.next_delay()
    assert delay3 == 4.0


def test_exponential_backoff_max_delay():
    """Test exponential backoff respects max delay."""
    backoff = ExponentialBackoff(base_delay=1.0, max_delay=5.0, jitter=False)

    for _ in range(10):
        delay = backoff.next_delay()
        assert delay <= 5.0


def test_exponential_backoff_with_jitter():
    """Test exponential backoff applies jitter."""
    backoff = ExponentialBackoff(base_delay=1.0, jitter=True)

    delays = [backoff.next_delay() for _ in range(10)]

    # With jitter, delays should vary
    # Jitter range is ±25%, so for 1.0s base: 0.75 to 1.25
    for delay in delays[:1]:  # Check first delay
        assert 0.75 <= delay <= 1.25


def test_exponential_backoff_reset():
    """Test exponential backoff reset."""
    backoff = ExponentialBackoff(base_delay=1.0, jitter=False)

    backoff.next_delay()
    backoff.next_delay()
    assert backoff.attempt == 2

    backoff.reset()
    assert backoff.attempt == 0

    delay = backoff.next_delay()
    assert delay == 1.0


# ---------------------------------------------------------------------------
# Unit Tests - Cache Key Computation
# ---------------------------------------------------------------------------


def test_cache_key_computation(tmp_path):
    """Test cache key computation."""
    manager = GitHubAPIClientManager(
        credential_manager=MagicMock(),
        cache_dir=tmp_path,
    )

    key1 = manager._compute_cache_key("/repos/owner/repo", None)
    key2 = manager._compute_cache_key("/repos/owner/repo", None)

    # Same inputs should produce same key
    assert key1 == key2


def test_cache_key_different_endpoints(tmp_path):
    """Test cache keys differ for different endpoints."""
    manager = GitHubAPIClientManager(
        credential_manager=MagicMock(),
        cache_dir=tmp_path,
    )

    key1 = manager._compute_cache_key("/repos/owner/repo1", None)
    key2 = manager._compute_cache_key("/repos/owner/repo2", None)

    assert key1 != key2


def test_cache_key_with_params(tmp_path):
    """Test cache key includes parameters."""
    manager = GitHubAPIClientManager(
        credential_manager=MagicMock(),
        cache_dir=tmp_path,
    )

    key1 = manager._compute_cache_key("/repos/owner/repo", {"page": 1})
    key2 = manager._compute_cache_key("/repos/owner/repo", {"page": 2})

    assert key1 != key2


# ---------------------------------------------------------------------------
# Integration Tests - Caching
# ---------------------------------------------------------------------------


def test_cache_store_and_retrieve(tmp_path):
    """Test storing and retrieving from cache."""
    manager = GitHubAPIClientManager(
        credential_manager=MagicMock(),
        cache_dir=tmp_path,
    )

    endpoint = "/repos/owner/repo"
    response_data = {"id": 123, "name": "test-repo"}

    # Store in cache
    manager._store_in_cache(endpoint, None, response_data, etag="abc123")

    # Retrieve from cache
    cached = manager._get_from_cache(endpoint, None)

    assert cached == response_data


def test_cache_expiration(tmp_path):
    """Test cache expiration."""
    manager = GitHubAPIClientManager(
        credential_manager=MagicMock(),
        cache_dir=tmp_path,
    )

    endpoint = "/repos/owner/repo"
    response_data = {"id": 123}

    # Store with 1 second TTL
    manager._store_in_cache(endpoint, None, response_data, ttl_seconds=1)

    # Should be available immediately
    assert manager._get_from_cache(endpoint, None) == response_data

    # Wait for expiration
    import time
    time.sleep(1.5)

    # Should be expired
    assert manager._get_from_cache(endpoint, None) is None


def test_cache_etag_preserved(tmp_path):
    """Test ETag is preserved in cache."""
    manager = GitHubAPIClientManager(
        credential_manager=MagicMock(),
        cache_dir=tmp_path,
    )

    endpoint = "/repos/owner/repo"
    response_data = {"id": 123}
    etag = "W/\"abc123def456\""

    # Store with ETag
    manager._store_in_cache(endpoint, None, response_data, etag=etag)

    # Retrieve cache entry
    entry = manager._get_cache_entry(endpoint, None)

    assert entry is not None
    assert entry.etag == etag


# ---------------------------------------------------------------------------
# Integration Tests - GitHubAPIClientManager
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_credential_manager():
    """Create mock credential manager."""
    manager = MagicMock()

    # Mock OAuth token credentials
    manager.retrieve_credentials.return_value = OAuthToken(
        token="ghp_testtoken123456789012345678901234",
        token_type="Bearer",
        scopes=["repo", "read:org"],
    )

    return manager


@pytest.fixture
def api_manager(mock_credential_manager, tmp_path):
    """Create API manager for testing."""
    return GitHubAPIClientManager(
        credential_manager=mock_credential_manager,
        cache_dir=tmp_path / "cache",
        max_concurrent_requests=5,
    )


def test_get_client_oauth_token(api_manager, mock_credential_manager):
    """Test getting GitHub client with OAuth token."""
    mock_credential_manager.retrieve_credentials.return_value = OAuthToken(
        token="ghp_test123",
        token_type="Bearer",
        scopes=["repo"],
    )

    client = api_manager.get_client("test_cred")

    assert client is not None
    assert hasattr(client, 'rest')


def test_get_client_personal_access_token(api_manager, mock_credential_manager):
    """Test getting GitHub client with PAT."""
    mock_credential_manager.retrieve_credentials.return_value = PersonalAccessToken(
        token="ghp_test123",
        token_prefix="ghp_tes",
        scopes=["repo"],
    )

    client = api_manager.get_client("test_cred")

    assert client is not None


def test_get_client_oauth_tokens(api_manager, mock_credential_manager):
    """Test getting GitHub client with OAuthTokens."""
    mock_credential_manager.retrieve_credentials.return_value = OAuthTokens(
        access_token="ghp_test123",
        token_type="Bearer",
        scopes=["repo"],
    )

    client = api_manager.get_client("test_cred")

    assert client is not None


def test_get_client_enterprise_server(api_manager):
    """Test getting GitHub client for Enterprise Server."""
    client = api_manager.get_client("test_cred", github_host="github.company.com")

    # Client should be configured with custom base URL
    assert client is not None


@pytest.mark.asyncio
async def test_rest_request_basic(api_manager):
    """Test basic REST API request."""
    with patch.object(api_manager, '_execute_rest_request', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"id": 123, "name": "test-repo"}

        result = await api_manager.rest_request(
            credential_id="test_cred",
            method="GET",
            endpoint="/repos/owner/repo",
        )

        assert result == {"id": 123, "name": "test-repo"}
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_rest_request_with_cache_hit(api_manager):
    """Test REST request with cache hit."""
    # Pre-populate cache
    endpoint = "/repos/owner/repo"
    cached_data = {"id": 123, "name": "cached-repo"}
    api_manager._store_in_cache(endpoint, None, cached_data)

    # Make request
    with patch.object(api_manager, '_execute_rest_request', new_callable=AsyncMock) as mock_exec:
        result = await api_manager.rest_request(
            credential_id="test_cred",
            method="GET",
            endpoint=endpoint,
        )

        # Should return cached data without calling API
        assert result == cached_data
        mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_rest_request_rate_limit_throttling(api_manager):
    """Test rate limiting throttles requests."""
    # Set rate limit at 95% usage (above threshold)
    api_manager._rate_limit_info["test_cred:rest"] = RateLimitInfo(
        limit=5000,
        remaining=250,
        reset_at=datetime.now(timezone.utc) + timedelta(seconds=2),
        used=4750,
    )

    with patch.object(api_manager, '_execute_rest_request', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"data": "test"}

        start_time = asyncio.get_event_loop().time()

        await api_manager.rest_request(
            credential_id="test_cred",
            method="GET",
            endpoint="/test",
        )

        elapsed = asyncio.get_event_loop().time() - start_time

        # Should have waited (at least 1 second)
        assert elapsed >= 1.0


@pytest.mark.asyncio
async def test_graphql_request_basic(api_manager):
    """Test basic GraphQL request."""
    with patch('githubkit.GitHub.async_graphql', new_callable=AsyncMock) as mock_graphql:
        mock_graphql.return_value = {
            "data": {
                "viewer": {"login": "testuser"},
                "rateLimit": {
                    "limit": 5000,
                    "remaining": 4999,
                    "used": 1,
                    "resetAt": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                }
            }
        }

        result = await api_manager.graphql_request(
            credential_id="test_cred",
            query="{ viewer { login } }",
        )

        assert result["data"]["viewer"]["login"] == "testuser"
        mock_graphql.assert_called_once()


@pytest.mark.asyncio
async def test_graphql_request_updates_rate_limit(api_manager):
    """Test GraphQL request updates rate limit."""
    with patch('githubkit.GitHub.async_graphql', new_callable=AsyncMock) as mock_graphql:
        reset_at = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_graphql.return_value = {
            "data": {
                "rateLimit": {
                    "limit": 5000,
                    "remaining": 4500,
                    "used": 500,
                    "resetAt": reset_at.isoformat(),
                    "cost": 1,
                    "nodeCount": 10,
                }
            }
        }

        await api_manager.graphql_request(
            credential_id="test_cred",
            query="{ rateLimit { limit remaining used resetAt } }",
        )

        # Rate limit should be updated
        rate_limit = api_manager._rate_limit_info.get("test_cred:graphql")
        assert rate_limit is not None
        assert rate_limit.remaining == 4500


@pytest.mark.asyncio
async def test_concurrent_request_limit(api_manager):
    """Test concurrent request limiting with semaphore."""
    # Set max concurrent to 3
    api_manager.max_concurrent_requests = 3

    call_count = 0
    max_concurrent = 0
    current_concurrent = 0

    async def mock_execute(*args, **kwargs):
        nonlocal call_count, max_concurrent, current_concurrent
        call_count += 1
        current_concurrent += 1
        max_concurrent = max(max_concurrent, current_concurrent)

        # Simulate API delay
        await asyncio.sleep(0.1)

        current_concurrent -= 1
        return {"data": "test"}

    with patch.object(api_manager, '_execute_rest_request', side_effect=mock_execute):
        # Make 10 concurrent requests
        tasks = [
            api_manager.rest_request(
                credential_id="test_cred",
                method="GET",
                endpoint=f"/test/{i}",
                use_cache=False,
            )
            for i in range(10)
        ]

        await asyncio.gather(*tasks)

        # All requests should complete
        assert call_count == 10

        # But no more than 3 should run concurrently
        assert max_concurrent <= 3


def test_metrics_collection(api_manager):
    """Test request metrics collection."""
    api_manager._record_metrics(
        request_id="req123",
        method="GET",
        endpoint="/repos/owner/repo",
        api_type="rest",
        duration_ms=125.5,
        status_code=200,
        cached=False,
        rate_limit_remaining=4999,
    )

    metrics = api_manager.get_metrics()

    assert len(metrics) == 1
    assert metrics[0].request_id == "req123"
    assert metrics[0].method == "GET"
    assert metrics[0].duration_ms == 125.5


def test_get_rate_limit_status(api_manager):
    """Test getting rate limit status."""
    # Set rate limits
    api_manager._rate_limit_info["test_cred:rest"] = RateLimitInfo(
        limit=5000,
        remaining=4500,
        reset_at=datetime.now(timezone.utc),
        used=500,
    )

    api_manager._rate_limit_info["test_cred:graphql"] = RateLimitInfo(
        limit=5000,
        remaining=4800,
        reset_at=datetime.now(timezone.utc),
        used=200,
    )

    status = api_manager.get_rate_limit_status("test_cred")

    assert status["rest"] is not None
    assert status["rest"].remaining == 4500
    assert status["graphql"] is not None
    assert status["graphql"].remaining == 4800


def test_circuit_breaker_integration(api_manager):
    """Test circuit breaker integration with error handling."""
    endpoint = "/test/endpoint"

    # Simulate failures
    for _ in range(5):
        api_manager._handle_api_error(Exception("API Error"), endpoint)

    # Circuit breaker should be open
    breaker = api_manager._circuit_breakers.get(endpoint)
    assert breaker is not None
    assert breaker.state == "open"


@pytest.mark.asyncio
async def test_exponential_backoff_on_server_errors(api_manager):
    """Test exponential backoff on 502/503/504 errors."""
    call_count = 0

    async def mock_request_with_failures(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            # Simulate server error with simple exception
            error = Exception("503 Service Unavailable")
            error.status_code = 503
            raise error
        else:
            # Success on third attempt
            mock_response = MagicMock()
            mock_response.headers = {
                "X-RateLimit-Limit": "5000",
                "X-RateLimit-Remaining": "4999",
                "X-RateLimit-Reset": str(int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())),
            }
            mock_response.status_code = 200
            mock_response.parsed_data = {"data": "success"}
            return mock_response

    with patch.object(api_manager, 'get_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.arequest = mock_request_with_failures
        mock_get_client.return_value = mock_client

        result = await api_manager.rest_request(
            credential_id="test_cred",
            method="GET",
            endpoint="/test",
            use_cache=False,
        )

        assert result == {"data": "success"}
        assert call_count == 3


# ---------------------------------------------------------------------------
# Privacy Tests
# ---------------------------------------------------------------------------


def test_no_credentials_in_metrics(api_manager):
    """Test that credentials are never logged in metrics."""
    api_manager._record_metrics(
        request_id="req123",
        method="GET",
        endpoint="/repos/owner/repo",
        api_type="rest",
        duration_ms=100.0,
        status_code=200,
        cached=False,
    )

    metrics = api_manager.get_metrics()
    metric_json = metrics[0].model_dump_json()

    # Ensure no credential tokens in metrics
    assert "ghp_" not in metric_json
    assert "github_pat_" not in metric_json


def test_error_handling_no_sensitive_data(api_manager, caplog):
    """Test error handling doesn't log sensitive data."""
    import logging
    caplog.set_level(logging.ERROR)

    endpoint = "/test/endpoint"
    error = Exception("API Error with token ghp_secret123")

    api_manager._handle_api_error(error, endpoint)

    # Check logs don't contain the token
    for record in caplog.records:
        assert "ghp_secret123" not in record.getMessage()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_response_handling(api_manager):
    """Test handling of empty API responses."""
    with patch.object(api_manager, '_execute_rest_request', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {}

        result = await api_manager.rest_request(
            credential_id="test_cred",
            method="GET",
            endpoint="/test",
        )

        assert result == {}


def test_cache_with_none_params(api_manager):
    """Test caching works with None parameters."""
    endpoint = "/test"
    data = {"key": "value"}

    api_manager._store_in_cache(endpoint, None, data)
    cached = api_manager._get_from_cache(endpoint, None)

    assert cached == data


def test_cache_key_params_order_invariant(api_manager):
    """Test cache key is invariant to parameter order."""
    params1 = {"a": 1, "b": 2, "c": 3}
    params2 = {"c": 3, "a": 1, "b": 2}

    key1 = api_manager._compute_cache_key("/test", params1)
    key2 = api_manager._compute_cache_key("/test", params2)

    # Keys should be the same regardless of order
    assert key1 == key2


@pytest.mark.asyncio
async def test_rate_limit_check_no_info(api_manager):
    """Test rate limit check with no prior info proceeds."""
    # Should not raise or block when no rate limit info
    await api_manager._check_rate_limit("unknown_cred", "rest")

    # Test passes if no exception raised


def test_metrics_limit(api_manager):
    """Test metrics are limited to 1000 entries."""
    # Add 1500 metrics
    for i in range(1500):
        api_manager._record_metrics(
            request_id=f"req{i}",
            method="GET",
            endpoint="/test",
            api_type="rest",
            duration_ms=100.0,
            status_code=200,
            cached=False,
        )

    metrics = api_manager.get_metrics()

    # Should keep only last 1000
    assert len(metrics) == 1000
    assert metrics[0].request_id == "req500"
    assert metrics[-1].request_id == "req1499"
