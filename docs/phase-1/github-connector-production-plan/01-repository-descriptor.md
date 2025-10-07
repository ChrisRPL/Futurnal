Summary: Specify repository metadata model, storage, and registration flow for GitHub repositories.

# 01 · Repository Descriptor

## Purpose
Define the GitHub-specific source descriptor used by the ingestion orchestrator: repository UID, owner/name, visibility (public/private), branch selection, credential reference, and privacy flags. Ensures the connector can register, authenticate, and monitor repositories safely with explicit AI learning consent.

## Scope
- Descriptor schema with required/optional fields
- Persistent storage location and format with encryption metadata
- Validation rules and conflict handling
- Integration points with `src/futurnal/ingestion/github/*` and orchestrator scheduler
- CLI registration flow with OAuth/PAT modes
- GitHub Enterprise Server support

## Requirements Alignment
- Privacy by default; all processing is local unless explicitly escalated
- Explicit consent for AI experiential learning from repository content
- OAuth-first for GitHub.com; Personal Access Token fallback
- Feed Unstructured.io parsing and emit semantic triples into PKG and vectors
- Keep graph and vector indices synchronized during updates

## Data Model

### GitHubRepositoryDescriptor
```python
class SyncMode(str, Enum):
    GRAPHQL_API = "graphql_api"  # Lightweight, online-only
    GIT_CLONE = "git_clone"      # Full fidelity, offline-capable

class VisibilityType(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"  # GitHub Enterprise only

class GitHubRepositoryDescriptor(BaseModel):
    # Identity
    id: str  # Deterministic ULID/UUIDv7 derived from owner/repo
    name: Optional[str]  # Human label (e.g., "My Awesome Project")
    icon: Optional[str]  # Optional emoji or icon

    # Repository Identity
    owner: str  # Repository owner (user or organization)
    repo: str   # Repository name
    full_name: str  # owner/repo (computed)
    visibility: VisibilityType

    # GitHub Instance
    github_host: str = "github.com"  # Default to GitHub.com
    api_base_url: Optional[str] = None  # For GitHub Enterprise

    # Authentication
    credential_id: str  # Reference to keychain credential

    # Sync Configuration
    sync_mode: SyncMode = SyncMode.GRAPHQL_API

    # Branch Selection
    branches: List[str] = Field(
        default_factory=lambda: ["main", "master"],
        description="Branch whitelist for sync"
    )
    branch_patterns: List[str] = Field(
        default_factory=list,
        description="Glob patterns for branch selection"
    )
    exclude_branches: List[str] = Field(
        default_factory=lambda: ["gh-pages", "dependabot/*"],
        description="Branches to skip"
    )

    # File Selection
    include_paths: List[str] = Field(
        default_factory=list,
        description="Path patterns to include (empty = all)"
    )
    exclude_paths: List[str] = Field(
        default_factory=lambda: [
            ".git/",
            "node_modules/",
            "__pycache__/",
            "*.pyc",
            ".env*",
            "secrets.*",
            "credentials.*",
        ],
        description="Path patterns to exclude"
    )

    # Content Scope
    sync_issues: bool = True
    sync_pull_requests: bool = True
    sync_wiki: bool = True
    sync_releases: bool = True

    # Temporal Scope
    sync_from_date: Optional[datetime] = None  # Only sync commits after this date
    max_commit_age_days: Optional[int] = None  # Only sync recent commits

    # Git Clone Mode Settings (only used when sync_mode=GIT_CLONE)
    clone_depth: Optional[int] = None  # Shallow clone depth (None = full history)
    sparse_checkout: bool = False  # Use sparse checkout for large repos
    local_clone_path: Optional[Path] = None  # Where to store cloned repo

    # Privacy & Consent
    privacy_settings: RepositoryPrivacySettings

    # Provenance
    created_at: datetime
    updated_at: datetime
    provenance: Provenance  # OS user, machine hash, tool version
```

### RepositoryPrivacySettings
```python
class RepositoryPrivacySettings(BaseModel):
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD

    # Consent scopes
    required_consent_scopes: List[ConsentScope] = Field(
        default_factory=lambda: [
            ConsentScope.GITHUB_REPO_ACCESS,
            ConsentScope.GITHUB_CODE_ANALYSIS,
        ]
    )

    # Redaction
    enable_path_anonymization: bool = True  # Redact paths in logs
    enable_author_anonymization: bool = False  # Redact commit authors
    redact_file_patterns: List[str] = Field(
        default_factory=lambda: [
            "*secret*",
            "*password*",
            "*token*",
            ".env*",
            "credentials.*",
        ]
    )  # Files to completely exclude

    # Content filtering
    exclude_extensions: List[str] = Field(
        default_factory=lambda: [
            ".exe", ".dll", ".so", ".dylib",  # Binaries
            ".jpg", ".png", ".gif", ".mp4",   # Media
            ".zip", ".tar", ".gz",            # Archives
        ]
    )
    max_file_size_mb: int = 10  # Skip files larger than this

    # Sensitive content detection
    detect_secrets: bool = True  # Use secret detection patterns
    secret_patterns: List[str] = Field(
        default_factory=lambda: [
            r"(?i)(api[_-]?key|apikey)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})",
            r"(?i)(password|passwd|pwd)[\s]*[=:]+[\s]*['\"]?([^\s'\"]{8,})",
            r"(?i)(token)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})",
            r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
        ]
    )

    # Audit
    audit_sync_events: bool = True
    audit_content_changes: bool = False  # Expensive, checksum only
    retain_audit_days: int = 90
```

### ConsentScope Extensions
```python
class ConsentScope(str, Enum):
    # Existing scopes...

    # GitHub-specific scopes
    GITHUB_REPO_ACCESS = "github:repo:access"
    GITHUB_CODE_ANALYSIS = "github:repo:code_analysis"
    GITHUB_ISSUE_METADATA = "github:repo:issue_metadata"
    GITHUB_PR_METADATA = "github:repo:pr_metadata"
    GITHUB_WIKI_ACCESS = "github:repo:wiki_access"
    GITHUB_CLOUD_MODELS = "github:repo:cloud_models"
```

## Storage

### File Location
- Stored in workspace registry under `~/.futurnal/sources/github/<repo_id>.json`
- Credentials stored separately in OS keychain (never in descriptor files)
- Transaction-safe updates guarded by file locks
- Backed up by `workspace/backup.py` routines (excluding credentials)

### Local Clone Storage (Git Clone Mode)
- Cloned repositories stored under `~/.futurnal/data/github/clones/<repo_id>/`
- Git LFS objects stored separately to manage disk usage
- Shallow clones preferred for large repositories
- Automatic cleanup of stale clones

### Encryption Metadata
Descriptor stores metadata for credential retrieval but never the credentials themselves:
```json
{
  "id": "repo_01HQWX...",
  "owner": "octocat",
  "repo": "Hello-World",
  "full_name": "octocat/Hello-World",
  "credential_id": "github_cred_01HQWX...",
  "github_host": "github.com",
  "sync_mode": "graphql_api",
  ...
}
```

Actual tokens stored via Python keyring:
```python
import keyring
keyring.set_password("futurnal.github", credential_id, token_json)
```

## CLI Registration Flow

### OAuth Device Flow (Recommended for CLI)
```bash
futurnal sources github add \
  --repo octocat/Hello-World \
  --name "Hello World Example" \
  --auth oauth
```

Steps:
1. Parse owner/repo from input
2. Launch OAuth Device Flow (displays user code)
3. User authorizes via browser at github.com/login/device
4. Exchange device code for access token
5. Store token in OS keychain with refresh token
6. Fetch repository metadata via API
7. Detect default branch and visibility
8. Prompt for sync mode (GraphQL API vs Git Clone)
9. Prompt for branch selection (default: main/master)
10. Create descriptor with credential reference
11. Persist descriptor JSON

### Personal Access Token Flow (Fallback)
```bash
futurnal sources github add \
  --repo octocat/Hello-World \
  --name "Hello World Example" \
  --auth token \
  --token ghp_xxxxxxxxxxxxx
```

Steps:
1. Parse owner/repo and validate token format
2. Validate token has required scopes (repo, read:org)
3. Store token in OS keychain
4. Fetch repository metadata
5. Continue with same flow as OAuth

### GitHub Enterprise Server
```bash
futurnal sources github add \
  --repo myorg/myrepo \
  --name "Enterprise Repo" \
  --host github.company.com \
  --api-base https://github.company.com/api/v3 \
  --auth token \
  --token ghp_xxxxxxxxxxxxx
```

### Management Commands
```bash
futurnal sources github list
futurnal sources github inspect <repo_id>
futurnal sources github update <repo_id> --branches "main,develop"
futurnal sources github test-connection <repo_id>
futurnal sources github refresh-oauth <repo_id>
futurnal sources github remove <repo_id> [--delete-clone] [--delete-credentials]
```

## Validation Rules

### Required Validations
- Repository name must match GitHub format (owner/repo)
- Owner and repo must exist and be accessible
- Token must have required scopes (repo for private, public_repo for public)
- Reject duplicates: same owner/repo → same repository id
- Branch names must be valid Git ref names
- File patterns must be valid glob patterns

### Security Validations
- Enforce HTTPS for all API connections
- Validate OAuth provider URLs match known GitHub instances
- Check credential expiration for OAuth tokens
- Warn on tokens with excessive scopes
- Detect and warn on insecure token storage attempts

### GitHub Instance Detection
**GitHub.com**:
- Host: `github.com`
- API: `https://api.github.com`
- OAuth: Standard GitHub OAuth endpoints

**GitHub Enterprise Server**:
- Custom host
- API: `https://<host>/api/v3`
- OAuth: Custom endpoints on enterprise instance

### Sync Mode Validation
**GraphQL API Mode**:
- Requires network connectivity
- Lower disk usage
- Supports selective file access
- Cannot operate offline

**Git Clone Mode**:
- Requires disk space (validate available space)
- Can operate offline after initial clone
- Full git history available
- Higher disk usage

## Acceptance Criteria

- ✅ Creating, reading, and listing descriptors works via CLI and programmatic API
- ✅ OAuth Device Flow completes successfully for GitHub.com
- ✅ Personal Access Token flow works for both GitHub.com and Enterprise
- ✅ Credentials never appear in descriptor files or logs
- ✅ Repository metadata fetched and validated before persisting
- ✅ Branch selection respects repository's actual branches
- ✅ Duplicate registration is idempotent (updates existing descriptor)
- ✅ Privacy settings correctly filter out sensitive file patterns
- ✅ Secret detection patterns prevent credential ingestion
- ✅ Backups include descriptor files but exclude credentials and clones
- ✅ Audit events logged for registration/update/removal actions
- ✅ GitHub Enterprise Server support with custom API base URLs

## Test Plan

### Unit Tests
- Schema validation for all descriptor fields
- ID determinism (same owner/repo → same ID)
- Privacy pattern matching (file exclusions, secret detection)
- Branch pattern expansion (glob matching)
- Validation rule enforcement
- Sync mode configuration validation

### Integration Tests
- CLI add/list/inspect/update/remove commands
- OAuth Device Flow with mock OAuth provider
- Personal Access Token flow with token validation
- Repository metadata fetching via API
- Credential storage/retrieval via keyring
- Branch enumeration from repository
- Duplicate registration handling
- GitHub Enterprise Server support

### Security Tests
- Credentials never logged or persisted in files
- Secret detection prevents credential ingestion
- OAuth token encryption at rest
- Secure credential deletion on removal
- File pattern exclusions work correctly
- Path redaction in audit logs

## Implementation Notes

### Deterministic ID Generation
```python
def _deterministic_repository_id(owner: str, repo: str, host: str = "github.com") -> str:
    """Generate deterministic repository ID from owner/repo/host."""
    normalized = f"{owner.lower()}/{repo.lower()}@{host.lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"github:{normalized}"))
```

### Credential Reference Pattern
```python
def _create_credential_id(repo_id: str) -> str:
    """Create credential ID for keychain storage."""
    return f"github_cred_{repo_id}"
```

### GitHub Instance Detection
```python
GITHUB_INSTANCES = {
    "github.com": {
        "api_base": "https://api.github.com",
        "oauth_url": "https://github.com/login/oauth",
        "type": "cloud"
    },
    # Enterprise instances detected dynamically
}
```

### Full Name Construction
```python
@property
def full_name(self) -> str:
    """Compute full repository name."""
    return f"{self.owner}/{self.repo}"
```

## Open Questions

- Should we auto-detect organization repositories and offer bulk registration?
- How to handle repository transfers (owner changes)?
- Should we support submodules and how deep?
- How to handle repository forks (duplicate content)?
- Should we cache repository metadata or fetch fresh on each sync?
- How to handle archived repositories?
- Support for GitHub Gists as separate source type?

## Dependencies
- Python keyring library for OS credential storage
- OAuth2 client library for GitHub OAuth flows (authlib or requests-oauthlib)
- GitHubKit for API interactions and token management
- GitPython for git clone mode operations
- ConsentRegistry for privacy approval workflow
- AuditLogger for registration event tracking


