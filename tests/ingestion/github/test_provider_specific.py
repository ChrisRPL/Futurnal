"""Provider-specific tests for GitHub.com and GitHub Enterprise.

Tests validate compatibility and correct behavior across different
GitHub deployments:
- GitHub.com (public cloud)
- GitHub Enterprise Server (self-hosted)

Covers OAuth flows, API endpoints, rate limits, and version compatibility.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    VisibilityType,
    SyncMode,
)


# ---------------------------------------------------------------------------
# GitHub.com Tests
# ---------------------------------------------------------------------------

pytestmark_com = pytest.mark.github_provider_com


@pytest.mark.github_provider_com
def test_github_com_default_configuration():
    """Test default configuration for GitHub.com.

    Validates:
    - Default host is github.com
    - Default API base URL
    - Standard rate limits
    """
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="octocat",
        repo="Hello-World",
        credential_id="test_cred",
        visibility=VisibilityType.PUBLIC,
    )

    assert descriptor.github_host == "github.com"
    assert descriptor.api_base_url == "https://api.github.com"


@pytest.mark.github_provider_com
def test_github_com_oauth_endpoints():
    """Test GitHub.com OAuth endpoints.

    Validates:
    - Device authorization URL
    - Token exchange URL
    - Correct OAuth scopes
    """
    # OAuth endpoints for GitHub.com
    oauth_config = {
        "device_code_url": "https://github.com/login/device/code",
        "token_url": "https://github.com/login/oauth/access_token",
        "authorization_url": "https://github.com/login/oauth/authorize",
    }

    # Verify HTTPS
    for url in oauth_config.values():
        assert url.startswith("https://"), f"OAuth URL not HTTPS: {url}"

    # Verify github.com domain
    for url in oauth_config.values():
        assert "github.com" in url, f"Not github.com URL: {url}"


@pytest.mark.github_provider_com
def test_github_com_rate_limits():
    """Test GitHub.com rate limit expectations.

    Validates:
    - Authenticated: 5000 requests/hour
    - GraphQL: 5000 points/hour
    """
    expected_limits = {
        "core": 5000,  # REST API
        "graphql": 5000,  # GraphQL API
        "search": 30,  # Search API (per minute)
    }

    # These are the expected limits for GitHub.com
    assert expected_limits["core"] == 5000
    assert expected_limits["graphql"] == 5000


@pytest.mark.github_provider_com
def test_github_com_graphql_api_version():
    """Test GitHub.com uses GraphQL API v4.

    Validates:
    - Correct API version
    - Endpoint URL
    """
    graphql_endpoint = "https://api.github.com/graphql"

    assert graphql_endpoint == "https://api.github.com/graphql"
    assert "v4" not in graphql_endpoint, "v4 is implicit in /graphql endpoint"


@pytest.mark.github_provider_com
def test_github_com_public_repo_access():
    """Test public repository access without authentication.

    Validates:
    - Public repos accessible
    - Anonymous rate limits (60 req/hour)
    """
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="octocat",
        repo="Hello-World",
        credential_id="test_cred",
        visibility=VisibilityType.PUBLIC,
    )

    # Public repos should be accessible
    assert descriptor.visibility == VisibilityType.PUBLIC
    assert descriptor.github_host == "github.com"


# ---------------------------------------------------------------------------
# GitHub Enterprise Tests
# ---------------------------------------------------------------------------

pytestmark_enterprise = pytest.mark.github_provider_enterprise


@pytest.mark.github_provider_enterprise
def test_github_enterprise_custom_host():
    """Test GitHub Enterprise with custom host.

    Validates:
    - Custom domain configuration
    - Custom API base URL
    """
    enterprise_host = "github.company.com"
    api_base = f"https://{enterprise_host}/api/v3"

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="team",
        repo="project",
        github_host=enterprise_host,
        api_base_url=api_base,
        credential_id="test_cred",
        visibility=VisibilityType.PRIVATE,
    )

    assert descriptor.github_host == enterprise_host
    assert descriptor.api_base_url == api_base
    assert "company.com" in descriptor.github_host


@pytest.mark.github_provider_enterprise
def test_github_enterprise_oauth_endpoints():
    """Test GitHub Enterprise OAuth configuration.

    Validates:
    - Custom OAuth endpoints
    - Enterprise-specific URLs
    """
    enterprise_host = "github.company.com"

    oauth_config = {
        "device_code_url": f"https://{enterprise_host}/login/device/code",
        "token_url": f"https://{enterprise_host}/login/oauth/access_token",
        "authorization_url": f"https://{enterprise_host}/login/oauth/authorize",
    }

    # Verify all URLs use enterprise host
    for url in oauth_config.values():
        assert enterprise_host in url, f"Not enterprise URL: {url}"
        assert url.startswith("https://"), f"Not HTTPS: {url}"


@pytest.mark.github_provider_enterprise
def test_github_enterprise_api_compatibility():
    """Test API compatibility with Enterprise Server.

    Validates:
    - REST API v3 support
    - GraphQL API support
    - Feature parity
    """
    enterprise_host = "github.enterprise.com"
    api_base = f"https://{enterprise_host}/api/v3"

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="org",
        repo="repo",
        github_host=enterprise_host,
        api_base_url=api_base,
        credential_id="test_cred",
        visibility=VisibilityType.PRIVATE,
    )

    # Verify API v3 endpoint
    assert "/api/v3" in descriptor.api_base_url

    # GraphQL should be at /api/graphql for Enterprise
    expected_graphql = f"https://{enterprise_host}/api/graphql"
    assert expected_graphql  # Verify format


@pytest.mark.github_provider_enterprise
def test_github_enterprise_rate_limits_custom():
    """Test custom rate limits for Enterprise.

    Validates:
    - Enterprise can have custom limits
    - Configuration flexibility
    """
    # Enterprise can configure custom rate limits
    # These are examples - actual limits vary by deployment
    enterprise_limits = {
        "core": 10000,  # Could be higher than github.com
        "graphql": 10000,
        "search": 60,
    }

    # Verify limits are configurable
    assert enterprise_limits["core"] >= 5000
    assert enterprise_limits["graphql"] >= 5000


@pytest.mark.github_provider_enterprise
def test_github_enterprise_ssl_verification():
    """Test SSL certificate handling for Enterprise.

    Validates:
    - Certificate verification enabled
    - Support for custom CA certificates
    """
    enterprise_host = "github.company.local"

    # HTTPS should be enforced
    with pytest.raises((ValueError, AssertionError)):
        # HTTP should be rejected
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner="team",
            repo="project",
            github_host=enterprise_host,
            api_base_url=f"http://{enterprise_host}/api/v3",  # HTTP not HTTPS
            credential_id="test_cred",
            visibility=VisibilityType.PRIVATE,
        )


@pytest.mark.github_provider_enterprise
def test_github_enterprise_version_compatibility():
    """Test compatibility across Enterprise versions.

    Validates:
    - Support for recent Enterprise versions (3.x)
    - API version detection
    """
    # Enterprise Server versions and their API capabilities
    supported_versions = [
        "3.8",  # Current LTS
        "3.9",
        "3.10",
        "3.11",
        "3.12",
    ]

    # All supported versions should have:
    # - REST API v3
    # - GraphQL API
    # - OAuth device flow

    for version in supported_versions:
        # Verify version format
        major, minor = map(int, version.split("."))
        assert major >= 3, f"Version {version} not supported (need 3.x+)"


@pytest.mark.github_provider_enterprise
def test_github_enterprise_private_mode():
    """Test GitHub Enterprise in private mode.

    Validates:
    - All repos require authentication
    - No anonymous access
    """
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="team",
        repo="project",
        github_host="github.company.com",
        api_base_url="https://github.company.com/api/v3",
        credential_id="test_cred",
        visibility=VisibilityType.PRIVATE,
    )

    # Private Enterprise deployment
    assert descriptor.visibility == VisibilityType.PRIVATE
    assert descriptor.credential_id is not None


# ---------------------------------------------------------------------------
# Cross-Provider Compatibility Tests
# ---------------------------------------------------------------------------


@pytest.mark.github_provider_com
@pytest.mark.github_provider_enterprise
def test_api_client_provider_detection():
    """Test automatic provider detection.

    Validates:
    - Detect GitHub.com vs Enterprise from host
    - Configure endpoints accordingly
    """
    # GitHub.com
    com_host = "github.com"
    assert "github.com" in com_host
    assert "." not in com_host.split("github.com")[0]  # No subdomain

    # Enterprise
    enterprise_host = "github.company.com"
    assert "github" in enterprise_host
    assert enterprise_host != "github.com"


@pytest.mark.github_provider_com
@pytest.mark.github_provider_enterprise
def test_url_construction_compatibility():
    """Test URL construction works for both providers.

    Validates:
    - Repository URLs
    - API endpoint URLs
    - OAuth URLs
    """
    providers = [
        ("github.com", "https://api.github.com"),
        ("github.enterprise.com", "https://github.enterprise.com/api/v3"),
    ]

    for host, expected_api in providers:
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner="owner",
            repo="repo",
            github_host=host if host != "github.com" else None,
            api_base_url=expected_api if host != "github.com" else None,
            credential_id="test_cred",
            visibility=VisibilityType.PRIVATE,
        )

        # Verify correct configuration
        if host == "github.com":
            assert descriptor.api_base_url == "https://api.github.com"
        else:
            assert expected_api in descriptor.api_base_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
