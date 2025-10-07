Summary: Comprehensive testing strategy for GitHub connector production readiness.

# 11 · Quality Gates & Testing

## Purpose
Define comprehensive testing strategy covering unit, integration, security, performance, and load testing to ensure production-ready quality for the GitHub Repository Connector. Establish quality gates that must pass before deployment.

## Scope
- Unit test coverage (>90%)
- Integration tests with real GitHub API
- Security testing (credentials, secrets, HTTPS)
- Performance benchmarks
- Load testing (rate limits, concurrent repos)
- Provider-specific tests (GitHub.com vs Enterprise)
- End-to-end scenario testing
- Regression test suite

## Requirements Alignment
- **<0.5% failure rate**: Reliable operation across diverse repositories
- **Security**: No credential leakage, proper secret detection
- **Performance**: API rate limit compliance, efficient sync
- **Reliability**: Handle edge cases (force pushes, large repos, API failures)

## Test Categories

### 1. Unit Tests (90%+ Coverage)

#### Repository Descriptor
```python
def test_descriptor_validation():
    """Test descriptor schema validation."""
    # Valid descriptor
    # Invalid owner format
    # Missing required fields
    # Privacy settings validation

def test_deterministic_id_generation():
    """Test ID generation consistency."""
    # Same owner/repo → same ID
    # Different hosts → different IDs
    # Case insensitivity

def test_branch_pattern_matching():
    """Test branch selection logic."""
    # Exact matches
    # Glob patterns
    # Exclusions
```

#### OAuth Authentication
```python
def test_oauth_device_flow():
    """Test OAuth device flow state machine."""
    # Initiate flow
    # Poll for token
    # Timeout handling
    # Denial handling

def test_token_storage_retrieval():
    """Test keychain operations."""
    # Store OAuth tokens
    # Retrieve tokens
    # Token expiration detection
    # Secure deletion

def test_token_refresh():
    """Test automatic token refresh."""
    # Detect expiration
    # Refresh before expiry
    # Update stored tokens
```

#### API Client Manager
```python
def test_rate_limit_tracking():
    """Test rate limit detection and throttling."""
    # Extract from headers
    # Throttle at threshold
    # Wait for reset

def test_caching():
    """Test request caching."""
    # Cache GET requests
    # ETag support
    # Cache expiration
    # Cache hit/miss

def test_circuit_breaker():
    """Test circuit breaker pattern."""
    # Record failures
    # Open circuit after threshold
    # Half-open retry
    # Close on success
```

#### Content Classifier
```python
def test_extension_classification():
    """Test file type classification."""
    # Code files (.py, .js, .go)
    # Documentation (.md, .rst)
    # Configuration (.json, .yaml)
    # Assets (.jpg, .png)

def test_language_detection():
    """Test programming language detection."""
    # By extension
    # By shebang
    # By content patterns

def test_secret_detection():
    """Test secret pattern matching."""
    # API keys
    # GitHub tokens
    # Private keys
    # Passwords
```

#### Incremental Sync Engine
```python
def test_commit_sha_tracking():
    """Test commit-based state tracking."""
    # Initial sync
    # Incremental sync
    # No changes (idempotent)

def test_force_push_detection():
    """Test force push detection."""
    # Normal history
    # Force push scenario
    # Rebase detection
```

### 2. Integration Tests

#### End-to-End Repository Sync
```python
@pytest.mark.integration
async def test_graphql_api_sync():
    """Test full sync via GraphQL API."""
    # Create test repository
    # Register in connector
    # Perform full sync
    # Verify files ingested
    # Verify PKG entities created

@pytest.mark.integration
async def test_git_clone_sync():
    """Test full sync via git clone."""
    # Clone test repository
    # Verify files on disk
    # Perform incremental update
    # Verify delta processing
```

#### OAuth Flow (Manual)
```python
@pytest.mark.manual
@pytest.mark.integration
def test_github_oauth_flow():
    """Test real GitHub OAuth (manual verification)."""
    # Initiate device flow
    # Display user code
    # Manual browser authorization
    # Verify token receipt
    # Verify token stored
```

#### API Operations
```python
@pytest.mark.integration
async def test_repository_metadata_fetch():
    """Test fetching repository metadata."""
    # Get repository info
    # Get branch list
    # Get commit history
    # Verify data accuracy

@pytest.mark.integration
async def test_issue_pr_normalization():
    """Test issue/PR metadata extraction."""
    # Fetch issue
    # Parse metadata
    # Extract triples
    # Verify relationships
```

#### Webhook Integration
```python
@pytest.mark.integration
async def test_webhook_server():
    """Test webhook receiver."""
    # Start webhook server
    # Send test payload
    # Verify signature
    # Verify event routing

@pytest.mark.integration
async def test_webhook_event_processing():
    """Test webhook event handling."""
    # Push event → sync triggered
    # PR event → metadata updated
    # Issue event → metadata updated
```

### 3. Security Tests

#### Credential Security
```python
@pytest.mark.security
def test_no_credentials_in_logs():
    """Verify credentials never logged."""
    # Capture all log output
    # Perform operations with credentials
    # Assert no tokens/passwords in logs

@pytest.mark.security
def test_no_credentials_in_exceptions():
    """Verify credentials not in error messages."""
    # Trigger errors with auth
    # Verify exception messages sanitized

@pytest.mark.security
def test_keychain_deletion():
    """Verify credential deletion."""
    # Store credential
    # Delete credential
    # Verify removal from keychain
```

#### Secret Detection
```python
@pytest.mark.security
def test_secret_pattern_detection():
    """Test secret detection patterns."""
    # Sample files with secrets
    # Verify detection
    # Verify exclusion from processing

@pytest.mark.security
def test_file_exclusion_patterns():
    """Test sensitive file exclusion."""
    # .env files
    # credentials.json
    # *.pem files
    # Verify exclusion
```

#### HTTPS Enforcement
```python
@pytest.mark.security
def test_https_only():
    """Verify all API calls use HTTPS."""
    # Capture network requests
    # Verify HTTPS URLs only
    # No plaintext HTTP
```

### 4. Performance Tests

#### Sync Performance
```python
@pytest.mark.performance
async def test_small_repository_sync_time():
    """Test sync time for small repo (<100 files)."""
    # Target: <10 seconds

@pytest.mark.performance
async def test_medium_repository_sync_time():
    """Test sync time for medium repo (100-1000 files)."""
    # Target: <60 seconds

@pytest.mark.performance
async def test_incremental_sync_speed():
    """Test incremental sync performance."""
    # Target: <5 seconds for 10 changed files
```

#### API Efficiency
```python
@pytest.mark.performance
async def test_api_request_count():
    """Verify API usage efficiency."""
    # Full sync of test repo
    # Count API requests
    # Verify reasonable usage
    # Target: <100 requests for medium repo

@pytest.mark.performance
async def test_graphql_batching():
    """Test GraphQL query efficiency."""
    # Batch file fetches
    # Verify cost calculation
    # Optimize query complexity
```

#### Memory Usage
```python
@pytest.mark.performance
async def test_memory_footprint():
    """Test memory usage during sync."""
    # Track memory during large repo sync
    # Target: <500MB peak memory
```

### 5. Load Tests

#### Rate Limit Compliance
```python
@pytest.mark.load
async def test_rate_limit_compliance():
    """Test rate limit handling under load."""
    # Sustained requests approaching limit
    # Verify throttling
    # Verify no limit exceeded

@pytest.mark.load
async def test_concurrent_repository_sync():
    """Test multiple repos syncing concurrently."""
    # Sync 10 repos simultaneously
    # Verify no race conditions
    # Verify rate limits respected
```

#### Large Repository Handling
```python
@pytest.mark.load
async def test_large_repository():
    """Test handling of large repository (>10k files)."""
    # Clone or fetch large repo
    # Verify performance acceptable
    # Verify memory usage reasonable

@pytest.mark.load
async def test_high_commit_frequency():
    """Test handling of high-frequency commits."""
    # Simulate 100 commits in 1 hour
    # Verify all processed
    # Verify no backlog
```

### 6. Provider-Specific Tests

#### GitHub.com
```python
@pytest.mark.provider
async def test_github_com_oauth():
    """Test OAuth with GitHub.com."""
    # Full OAuth flow
    # Token refresh
    # API access

@pytest.mark.provider
async def test_github_com_api_limits():
    """Test GitHub.com rate limits."""
    # Verify 5000 req/hour
    # Verify concurrent limits
```

#### GitHub Enterprise Server
```python
@pytest.mark.provider
async def test_github_enterprise():
    """Test GitHub Enterprise Server."""
    # Custom API base URL
    # Custom OAuth endpoints
    # API compatibility
```

## Quality Gates

### Pre-Commit Gates
- ✅ All unit tests pass
- ✅ Code coverage >90%
- ✅ Linting passes (ruff, mypy)
- ✅ Security scans pass (bandit)

### Pre-Merge Gates
- ✅ All unit tests pass
- ✅ Integration tests pass
- ✅ Security tests pass
- ✅ No credential leakage
- ✅ Code review approved

### Pre-Release Gates
- ✅ All tests pass (unit, integration, security)
- ✅ Performance benchmarks meet targets
- ✅ Load tests pass
- ✅ Manual OAuth flow verified
- ✅ Documentation complete
- ✅ Migration guide ready

## Test Fixtures

### Test Repositories
```python
@pytest.fixture
def small_test_repo():
    """Small repository for testing (10 files)."""
    return {
        "owner": "octocat",
        "repo": "Hello-World",
        "branches": ["main"],
        "files": 10,
    }

@pytest.fixture
def medium_test_repo():
    """Medium repository for testing (500 files)."""
    return {
        "owner": "test-org",
        "repo": "test-repo-medium",
        "branches": ["main", "develop"],
        "files": 500,
    }
```

### Mock Services
```python
@pytest.fixture
def mock_github_api():
    """Mock GitHub API responses."""
    # Return fixture data for API calls
    pass

@pytest.fixture
def mock_oauth_provider():
    """Mock OAuth provider."""
    # Simulate OAuth device flow
    pass
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: GitHub Connector Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest tests/github/unit/ --cov

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/github/integration/
        env:
          GITHUB_TOKEN: ${{ secrets.TEST_GITHUB_TOKEN }}

  security-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security tests
        run: |
          pytest tests/github/security/
          bandit -r src/futurnal/ingestion/github/
```

## Acceptance Criteria

- ✅ Unit test coverage >90%
- ✅ All integration tests pass
- ✅ Security tests pass (no credential leakage)
- ✅ Performance targets met
- ✅ Rate limit compliance verified
- ✅ Large repository handling verified
- ✅ Provider-specific tests pass
- ✅ CI/CD pipeline green

## Test Execution

### Local Testing
```bash
# Run all tests
pytest tests/github/

# Run specific category
pytest tests/github/unit/
pytest tests/github/integration/ -m integration
pytest tests/github/security/ -m security
pytest tests/github/performance/ -m performance

# Run with coverage
pytest tests/github/ --cov=futurnal.ingestion.github --cov-report=html
```

### Manual Testing Checklist
- [ ] OAuth device flow with real GitHub
- [ ] OAuth token refresh
- [ ] Repository sync (small, medium, large)
- [ ] Incremental sync
- [ ] Force push handling
- [ ] Webhook configuration
- [ ] Webhook event processing
- [ ] Issue/PR metadata extraction
- [ ] Secret detection
- [ ] Consent enforcement

## Regression Test Suite

### Critical Path Tests
```python
@pytest.mark.regression
class TestCriticalPath:
    """Tests for critical user journeys."""

    async def test_repository_registration_and_sync(self):
        """Test full repository registration and sync."""
        # Register repository
        # Grant consent
        # Perform full sync
        # Verify data in PKG

    async def test_incremental_sync_after_push(self):
        """Test incremental sync workflow."""
        # Initial sync
        # Simulate push
        # Incremental sync
        # Verify only new data processed
```

## Implementation Notes

### Test Data Management
- Use fixture repositories on GitHub
- Mock external dependencies
- Isolate test data from production

### Performance Baselines
- Small repo (<100 files): <10s full sync
- Medium repo (<1000 files): <60s full sync
- Incremental sync: <5s for 10 files
- API requests: <100 for medium repo

## Open Questions

- Should we support canary testing (gradual rollout)?
- How to test with real user repositories safely?
- Should we implement chaos testing (random failures)?
- How to benchmark against other ingestion connectors?

## Dependencies
- pytest (`pip install pytest pytest-asyncio pytest-cov`)
- pytest-mock for mocking
- pytest-benchmark for performance tests
- bandit for security scanning
- Test fixtures and mock data


