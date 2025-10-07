Summary: Implement GitHub API client with rate limiting, caching, and error handling using GitHubKit.

# 03 · API Client Manager

## Purpose
Provide a robust GitHub API client manager that handles rate limiting, request caching, exponential backoff, and error recovery. Implements intelligent request queuing to maximize API efficiency while respecting GitHub's rate limits (5000 req/hr REST, 5000 points/hr GraphQL).

## Scope
- GitHubKit library integration for REST and GraphQL APIs
- Adaptive rate limiting with request queuing
- Conditional requests with ETag caching
- Exponential backoff and retry logic
- Circuit breaker pattern for failing endpoints
- Request deduplication
- Concurrent request management (max 100 concurrent)
- Error classification and handling
- GitHub Enterprise Server support

## Requirements Alignment
- **Rate limit compliance**: Never exceed GitHub's rate limits
- **Efficient API usage**: Minimize unnecessary requests through caching
- **Resilience**: Graceful degradation during API failures
- **Privacy**: No sensitive data in API request logs
- **Performance**: Sub-second responses for cached data

## API Rate Limits (2025)

### REST API
- **Authenticated**: 5000 requests/hour per user
- **Unauthenticated**: 60 requests/hour per IP
- **Concurrent**: Max 100 concurrent requests (shared with GraphQL)
- **Reset**: Hourly window, resets at fixed time
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### GraphQL API
- **Point-based**: 5000 points/hour per user
- **Query cost**: Varies by query complexity (1-5000+ points)
- **Concurrent**: Max 100 concurrent requests (shared with REST)
- **Node limit**: Max 500,000 nodes per query
- **Rate limit query**: Available via introspection

### Secondary Rate Limits
- **Search API**: 30 requests/minute (authenticated)
- **Content Creation**: 80 requests/hour per user
- **Abuse detection**: Automatic throttling for suspicious patterns

## Data Model

### RateLimitInfo
```python
class RateLimitInfo(BaseModel):
    """Rate limit information from GitHub."""
    limit: int
    remaining: int
    reset_at: datetime
    used: int

    @property
    def reset_in_seconds(self) -> int:
        """Seconds until rate limit resets."""
        return max(0, int((self.reset_at - datetime.utcnow()).total_seconds()))

    @property
    def usage_percentage(self) -> float:
        """Percentage of rate limit used."""
        return (self.used / self.limit) * 100 if self.limit > 0 else 0.0

class GraphQLRateLimitInfo(BaseModel):
    """GraphQL-specific rate limit info."""
    limit: int
    remaining: int
    reset_at: datetime
    cost: int  # Cost of last query
    node_count: int  # Nodes returned in last query
```

### APIRequestMetrics
```python
class APIRequestMetrics(BaseModel):
    """Metrics for API request tracking."""
    request_id: str
    method: str  # GET, POST, etc.
    endpoint: str  # /repos/owner/repo
    api_type: str  # "rest" or "graphql"
    duration_ms: float
    status_code: int
    rate_limit_remaining: int
    cached: bool
    retries: int
    timestamp: datetime
```

### CacheEntry
```python
class CacheEntry(BaseModel):
    """Cache entry for API responses."""
    key: str  # Hash of request parameters
    response: Dict[str, Any]
    etag: Optional[str] = None
    expires_at: datetime
    created_at: datetime
```

## Component Design

### GitHubAPIClientManager
```python
class GitHubAPIClientManager:
    """Manages GitHub API clients with rate limiting and caching."""

    def __init__(
        self,
        *,
        credential_manager: GitHubCredentialManager,
        cache_dir: Optional[Path] = None,
        max_concurrent_requests: int = 80,  # Leave buffer below 100
        rate_limit_threshold: float = 0.9,  # Start throttling at 90%
    ):
        self.credential_manager = credential_manager
        self.cache_dir = cache_dir or Path.home() / ".futurnal" / "cache" / "github"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_threshold = rate_limit_threshold
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._rate_limit_info: Dict[str, RateLimitInfo] = {}

        # Request queue
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Metrics
        self._metrics: List[APIRequestMetrics] = []

    def get_client(
        self,
        credential_id: str,
        github_host: str = "github.com",
    ) -> GitHub:
        """Get authenticated GitHub client."""
        from githubkit import GitHub, TokenAuthStrategy

        # Retrieve credentials
        credentials = self.credential_manager.retrieve_credentials(credential_id)

        # Extract token
        if isinstance(credentials, OAuthTokens):
            token = credentials.access_token
        elif isinstance(credentials, PersonalAccessToken):
            token = credentials.token
        else:
            raise ValueError(f"Unknown credential type: {type(credentials)}")

        # Configure API base URL for Enterprise
        config = {}
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
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Make REST API request with rate limiting and caching."""
        async with self._semaphore:
            # Check rate limit
            await self._check_rate_limit(credential_id, "rest")

            # Check cache
            if use_cache and method == "GET":
                cached = self._get_from_cache(endpoint, params)
                if cached:
                    return cached

            # Make request
            client = self.get_client(credential_id)
            start_time = time.time()

            try:
                response = await self._execute_rest_request(
                    client, method, endpoint, params, data
                )

                # Update rate limit info
                self._update_rate_limit_from_headers(
                    credential_id, response.headers
                )

                # Cache if GET request
                if method == "GET" and response.status_code == 200:
                    self._store_in_cache(
                        endpoint, params, response.json(),
                        etag=response.headers.get("ETag")
                    )

                # Record metrics
                self._record_metrics(
                    method, endpoint, "rest",
                    duration_ms=(time.time() - start_time) * 1000,
                    status_code=response.status_code,
                    cached=False,
                )

                return response.json()

            except Exception as e:
                self._handle_api_error(e, endpoint)
                raise

    async def graphql_request(
        self,
        *,
        credential_id: str,
        query: str,
        variables: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make GraphQL API request with rate limiting."""
        async with self._semaphore:
            # Check rate limit
            await self._check_rate_limit(credential_id, "graphql")

            # Make request
            client = self.get_client(credential_id)
            start_time = time.time()

            try:
                response = await client.async_graphql(query, variables)

                # Extract rate limit from response
                if "rateLimit" in response.get("data", {}):
                    self._update_graphql_rate_limit(
                        credential_id, response["data"]["rateLimit"]
                    )

                # Record metrics
                self._record_metrics(
                    "POST", "/graphql", "graphql",
                    duration_ms=(time.time() - start_time) * 1000,
                    status_code=200,
                    cached=False,
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
        """Check if we should throttle requests."""
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
        credential_id: str,
        headers: Dict[str, str],
    ) -> None:
        """Update rate limit info from REST API response headers."""
        if "X-RateLimit-Limit" in headers:
            self._rate_limit_info[f"{credential_id}:rest"] = RateLimitInfo(
                limit=int(headers["X-RateLimit-Limit"]),
                remaining=int(headers["X-RateLimit-Remaining"]),
                reset_at=datetime.fromtimestamp(
                    int(headers["X-RateLimit-Reset"])
                ),
                used=int(headers["X-RateLimit-Limit"]) -
                     int(headers["X-RateLimit-Remaining"]),
            )

    def _update_graphql_rate_limit(
        self,
        credential_id: str,
        rate_limit_data: Dict,
    ) -> None:
        """Update rate limit info from GraphQL response."""
        self._rate_limit_info[f"{credential_id}:graphql"] = RateLimitInfo(
            limit=rate_limit_data["limit"],
            remaining=rate_limit_data["remaining"],
            reset_at=datetime.fromisoformat(
                rate_limit_data["resetAt"].replace("Z", "+00:00")
            ),
            used=rate_limit_data["used"],
        )

    def _get_from_cache(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Retrieve response from cache."""
        cache_key = self._compute_cache_key(endpoint, params)
        cache_path = self.cache_dir / f"{cache_key}.json"

        if not cache_path.exists():
            return None

        entry = CacheEntry.model_validate_json(cache_path.read_text())

        # Check expiration
        if datetime.utcnow() > entry.expires_at:
            cache_path.unlink()
            return None

        return entry.response

    def _store_in_cache(
        self,
        endpoint: str,
        params: Optional[Dict],
        response: Dict,
        etag: Optional[str] = None,
        ttl_seconds: int = 300,  # 5 minutes default
    ) -> None:
        """Store response in cache."""
        cache_key = self._compute_cache_key(endpoint, params)
        cache_path = self.cache_dir / f"{cache_key}.json"

        entry = CacheEntry(
            key=cache_key,
            response=response,
            etag=etag,
            expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds),
            created_at=datetime.utcnow(),
        )

        cache_path.write_text(entry.model_dump_json(indent=2))

    def _compute_cache_key(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> str:
        """Compute cache key from endpoint and parameters."""
        import hashlib
        key_parts = [endpoint]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        return hashlib.sha256("".join(key_parts).encode()).hexdigest()[:16]

    def _handle_api_error(
        self,
        error: Exception,
        endpoint: str,
    ) -> None:
        """Handle API errors with circuit breaker pattern."""
        # Update circuit breaker
        breaker = self._circuit_breakers.get(endpoint)
        if breaker:
            breaker.record_failure()

        # Log error (without sensitive data)
        logger.error(
            f"GitHub API error: {type(error).__name__}",
            extra={"endpoint": endpoint}
        )
```

### CircuitBreaker
```python
class CircuitBreaker:
    """Circuit breaker for failing API endpoints."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open

    def record_failure(self) -> None:
        """Record an API failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

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
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed > self.timeout_seconds:
                    self.state = "half_open"
                    return True
            return False

        # half_open state
        return True
```

### Exponential Backoff
```python
class ExponentialBackoff:
    """Exponential backoff with jitter for retries."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt = 0

    def next_delay(self) -> float:
        """Calculate next delay with exponential backoff."""
        delay = min(
            self.base_delay * (self.multiplier ** self.attempt),
            self.max_delay
        )

        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # ±25% jitter

        self.attempt += 1
        return delay

    def reset(self) -> None:
        """Reset backoff state."""
        self.attempt = 0
```

## Acceptance Criteria

- ✅ REST API requests respect rate limits (never exceed 5000/hour)
- ✅ GraphQL API requests respect point-based rate limits
- ✅ Concurrent requests never exceed 100 (shared across REST + GraphQL)
- ✅ GET requests are cached with ETag support
- ✅ Rate limit info extracted from response headers
- ✅ Automatic throttling at 90% rate limit usage
- ✅ Exponential backoff on transient failures (502, 503, 504)
- ✅ Circuit breaker prevents cascading failures
- ✅ Request metrics collected for monitoring
- ✅ GitHub Enterprise Server support with custom API base URLs
- ✅ Credentials never logged in request/response data

## Test Plan

### Unit Tests
- Rate limit calculation and throttling logic
- Cache key computation and collision resistance
- Circuit breaker state transitions
- Exponential backoff delay calculation
- Request deduplication
- Metrics recording

### Integration Tests
- REST API requests with real GitHub API (rate-limited)
- GraphQL API requests with query cost tracking
- Cache hit/miss behavior with ETags
- Rate limit header parsing
- Concurrent request management (80-100 requests)
- Circuit breaker activation on repeated failures
- Backoff and retry on 502/503 errors

### Load Tests
- Sustained requests approaching rate limit
- Burst traffic handling
- Cache effectiveness under load
- Concurrent request limits

### Security Tests
- Credentials not exposed in logs
- Request/response data privacy
- Cache entries don't leak sensitive data

## Implementation Notes

### GraphQL Rate Limit Query
```graphql
query RateLimitQuery {
  rateLimit {
    limit
    cost
    remaining
    resetAt
    nodeCount
  }
}
```

### Conditional Requests with ETag
```python
headers = {}
if cached_etag:
    headers["If-None-Match"] = cached_etag

response = client.get(endpoint, headers=headers)

if response.status_code == 304:  # Not Modified
    # Use cached response
    return cached_response
```

### Request Priority Queue
```python
class RequestPriority(Enum):
    HIGH = 1      # User-initiated requests
    NORMAL = 2    # Scheduled sync operations
    LOW = 3       # Background metadata refresh

# Process high-priority requests first
```

## Open Questions

- Should we implement request batching for GraphQL?
- How to handle rate limit resets across time zones?
- Should we support request prioritization?
- How to share rate limit info across multiple connector instances?
- Should we implement predictive rate limiting based on sync schedules?
- How to handle abuse detection rate limits (secondary limits)?

## Dependencies
- GitHubKit (`pip install githubkit`)
- AsyncIO for concurrent request management
- Python hashlib for cache key generation
- GitHubCredentialManager for token retrieval


