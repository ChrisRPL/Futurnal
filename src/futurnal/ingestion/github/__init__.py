"""GitHub repository connector for Futurnal.

This package implements the GitHub connector as specified in Phase 1 (Archivist),
providing secure OAuth and PAT-based authentication, repository metadata management,
privacy-first credential storage, and dual-mode repository synchronization.
"""

from .api_client import (
    BranchInfo,
    GitHubAPIClient,
    RepositoryInfo,
    TokenInfo,
    create_api_client,
)
from .connector import ElementSink, GitHubRepositoryConnector
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
from .git_clone_sync import GitCloneRepositorySync
from .graphql_sync import GraphQLRepositorySync
from .oauth_flow import (
    DeviceCodeResponse,
    DeviceFlowResult,
    DeviceFlowStatus,
    GitHubOAuthDeviceFlow,
    start_github_oauth_flow,
)
from .sync_models import (
    BranchSyncState,
    DiskSpaceEstimate,
    FileContent,
    FileEntry,
    SyncResult,
    SyncState,
    SyncStatus,
    SyncStrategy,
)
from .sync_orchestrator import GitHubSyncOrchestrator
from .sync_state_manager import SyncStateManager
from .sync_utils import (
    PatternMatcher,
    check_disk_space_sufficient,
    cleanup_clone_directory,
    ensure_clone_directory,
    estimate_git_clone_size,
    format_bytes,
    format_progress,
    format_sync_statistics,
    get_available_disk_space,
    get_available_disk_space_gb,
    get_default_clone_base_dir,
    get_git_binary_path,
    parse_git_remote_url,
    truncate_sha,
    validate_git_installed,
)
from .classifier_models import (
    FileCategory,
    FileClassification,
    ProgrammingLanguage,
)
from .file_classifier import FileClassifier
from .language_detector import detect_language
from .secret_detector import SecretDetector

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
    # Sync models
    "BranchSyncState",
    "DiskSpaceEstimate",
    "FileContent",
    "FileEntry",
    "SyncResult",
    "SyncState",
    "SyncStatus",
    "SyncStrategy",
    # Sync implementations
    "GitCloneRepositorySync",
    "GitHubSyncOrchestrator",
    "GraphQLRepositorySync",
    "SyncStateManager",
    # Connector
    "ElementSink",
    "GitHubRepositoryConnector",
    # Sync utilities
    "PatternMatcher",
    "check_disk_space_sufficient",
    "cleanup_clone_directory",
    "ensure_clone_directory",
    "estimate_git_clone_size",
    "format_bytes",
    "format_progress",
    "format_sync_statistics",
    "get_available_disk_space",
    "get_available_disk_space_gb",
    "get_default_clone_base_dir",
    "get_git_binary_path",
    "parse_git_remote_url",
    "truncate_sha",
    "validate_git_installed",
    # File classification
    "FileCategory",
    "FileClassification",
    "ProgrammingLanguage",
    "FileClassifier",
    "detect_language",
    "SecretDetector",
]
