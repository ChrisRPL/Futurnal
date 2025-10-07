"""GitHub repository connector for Futurnal.

This package implements the GitHub connector as specified in Phase 1 (Archivist),
providing secure OAuth and PAT-based authentication, repository metadata management,
and privacy-first credential storage.
"""

from .api_client import (
    BranchInfo,
    GitHubAPIClient,
    RepositoryInfo,
    TokenInfo,
    create_api_client,
)
from .api_client_manager import (
    APIRequestMetrics,
    CacheEntry,
    CircuitBreaker,
    ExponentialBackoff,
    GitHubAPIClientManager,
    GraphQLRateLimitInfo,
    RateLimitInfo,
)
from .credential_manager import (
    CredentialType,
    EnterpriseOAuthConfig,
    GitHubCredential,
    GitHubCredentialManager,
    GitHubOAuthConfig,
    OAuthToken,
    OAuthTokens,
    PersonalAccessToken,
    auto_refresh_wrapper,
    detect_token_type,
    secure_credential_context,
    validate_token_format,
)
from .descriptor import (
    ConsentScope,
    GitHubRepositoryDescriptor,
    PrivacyLevel,
    Provenance,
    RepositoryPrivacySettings,
    RepositoryRegistry,
    SyncMode,
    VisibilityType,
    create_credential_id,
    generate_repository_id,
)
from .oauth_flow import (
    DeviceCodeResponse,
    DeviceFlowResult,
    DeviceFlowStatus,
    GitHubOAuthDeviceFlow,
    start_github_oauth_flow,
)

__all__ = [
    # Descriptor and registry
    "ConsentScope",
    "GitHubRepositoryDescriptor",
    "PrivacyLevel",
    "Provenance",
    "RepositoryPrivacySettings",
    "RepositoryRegistry",
    "SyncMode",
    "VisibilityType",
    "create_credential_id",
    "generate_repository_id",
    # Credential management
    "CredentialType",
    "EnterpriseOAuthConfig",
    "GitHubCredential",
    "GitHubCredentialManager",
    "GitHubOAuthConfig",
    "OAuthToken",
    "OAuthTokens",
    "PersonalAccessToken",
    "auto_refresh_wrapper",
    "detect_token_type",
    "secure_credential_context",
    "validate_token_format",
    # OAuth flow
    "DeviceCodeResponse",
    "DeviceFlowResult",
    "DeviceFlowStatus",
    "GitHubOAuthDeviceFlow",
    "start_github_oauth_flow",
    # API client
    "BranchInfo",
    "GitHubAPIClient",
    "RepositoryInfo",
    "TokenInfo",
    "create_api_client",
    # API client manager
    "APIRequestMetrics",
    "CacheEntry",
    "CircuitBreaker",
    "ExponentialBackoff",
    "GitHubAPIClientManager",
    "GraphQLRateLimitInfo",
    "RateLimitInfo",
]
