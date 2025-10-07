Summary: Implement secure credential storage and GitHub OAuth management using OS-native keychains.

# 02 · OAuth Authentication Manager

## Purpose
Provide secure, OS-native storage for GitHub credentials (OAuth tokens, Personal Access Tokens) with automatic token refresh, memory clearing after use, and audit-compliant credential lifecycle management. Ensures the Ghost never exposes credentials in logs, files, or process memory longer than necessary.

## Scope
- OS keychain integration via Python keyring library
- GitHub OAuth Device Flow implementation (CLI-friendly, no browser required)
- OAuth Web Flow support (optional, browser-based)
- Personal Access Token secure storage
- Automatic token refresh before expiration
- Credential lifecycle management (create, retrieve, update, delete)
- Memory-safe credential handling (clear after use)
- Audit logging for credential operations (without exposing credentials)
- GitHub Enterprise Server OAuth support

## Requirements Alignment
- **Privacy-first**: Credentials never logged, never in files, cleared from memory after use
- **OAuth-first**: Prefer OAuth over PAT for better security and automatic refresh
- **Automatic refresh**: Tokens refreshed transparently before expiration
- **Audit compliance**: Log credential operations without exposing values
- **Secure deletion**: Completely remove credentials on repository removal

## Data Model

### CredentialStorage (Abstract)
```python
class CredentialType(str, Enum):
    OAUTH_TOKEN = "oauth_token"
    PERSONAL_ACCESS_TOKEN = "personal_access_token"

class GitHubCredential(BaseModel):
    credential_id: str
    credential_type: CredentialType
    github_host: str  # "github.com" or enterprise host
    username: Optional[str] = None  # Fetched from API after authentication
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None  # For OAuth tokens
    scopes: List[str] = Field(default_factory=list)  # Token scopes

class OAuthTokens(BaseModel):
    """Never persisted in plain files - only in OS keychain."""
    access_token: str
    refresh_token: Optional[str] = None  # May not be available for all flows
    token_type: str = "Bearer"
    expires_in: Optional[int] = None  # seconds (None = no expiration)
    expires_at: Optional[datetime] = None  # Calculated from expires_in
    scope: List[str] = Field(default_factory=list)

class PersonalAccessToken(BaseModel):
    """Never persisted in plain files - only in OS keychain."""
    token: str
    token_prefix: str  # First 7 chars for identification (ghp_xxxx)
    scopes: List[str] = Field(default_factory=list)  # Detected from API
```

### Keychain Storage Format
```python
# Stored in OS keychain under service name "futurnal.github"
# Account name is the credential_id
keyring.set_password(
    "futurnal.github",
    credential_id,
    json.dumps({
        "type": "oauth_token",
        "access_token": "gho_xxxxxxxxxxxxx",
        "refresh_token": "ghr_xxxxxxxxxxxxx",  # If available
        "expires_at": "2025-10-15T12:00:00Z",
        "scopes": ["repo", "read:org"],
        ...
    })
)
```

## Component Design

### GitHubCredentialManager
```python
class GitHubCredentialManager:
    """Manages GitHub credentials with OS keychain integration."""

    def __init__(
        self,
        *,
        audit_logger: Optional[AuditLogger] = None,
        oauth_config: Optional[GitHubOAuthConfig] = None,
    ):
        self._audit = audit_logger
        self._oauth = oauth_config or GitHubOAuthConfig()
        self._service_name = "futurnal.github"

    def store_oauth_tokens(
        self,
        *,
        credential_id: str,
        github_host: str,
        tokens: OAuthTokens,
        operator: Optional[str] = None,
    ) -> GitHubCredential:
        """Store OAuth tokens in OS keychain."""
        # Store tokens
        # Fetch username from API
        # Log audit event (without token values)
        # Return credential metadata

    def store_personal_access_token(
        self,
        *,
        credential_id: str,
        github_host: str,
        token: str,
        operator: Optional[str] = None,
    ) -> GitHubCredential:
        """Store Personal Access Token in OS keychain."""
        # Validate token format
        # Detect scopes from API
        # Store token
        # Clear token from memory
        # Log audit event
        # Return credential metadata

    def retrieve_credentials(
        self,
        credential_id: str,
    ) -> Union[OAuthTokens, PersonalAccessToken]:
        """Retrieve credentials from OS keychain.

        For OAuth: Automatically refreshes if expired.
        For PAT: Returns token (caller must clear).
        """
        # Retrieve from keychain
        # If OAuth and expired, refresh automatically
        # Log audit event (credential access)
        # Return credentials

    def refresh_oauth_token(
        self,
        credential_id: str,
    ) -> OAuthTokens:
        """Manually refresh OAuth token."""
        # Retrieve current tokens
        # Call GitHub token refresh endpoint
        # Update keychain with new tokens
        # Log audit event
        # Return new tokens

    def delete_credentials(
        self,
        credential_id: str,
        *,
        operator: Optional[str] = None,
    ) -> None:
        """Securely delete credentials from OS keychain."""
        # Delete from keychain
        # Log audit event
        # Verify deletion

    def list_credentials(self) -> List[GitHubCredential]:
        """List credential metadata (not values)."""
        # Scan keychain for futurnal.github entries
        # Return metadata only

    def validate_credentials(
        self,
        credential_id: str,
    ) -> bool:
        """Test if credentials are valid by making API call."""
        # Retrieve credentials
        # Make test API call (GET /user)
        # Return validity status
```

### GitHubOAuthConfig
```python
class GitHubOAuthConfig(BaseModel):
    """OAuth configuration for GitHub instances."""

    # GitHub.com OAuth (default)
    client_id: str = Field(
        default_factory=lambda: os.getenv("FUTURNAL_GITHUB_CLIENT_ID", ""),
        description="OAuth App Client ID"
    )
    client_secret: Optional[str] = Field(
        default_factory=lambda: os.getenv("FUTURNAL_GITHUB_CLIENT_SECRET"),
        description="OAuth App Client Secret (optional for device flow)"
    )

    # GitHub Enterprise support
    enterprise_configs: Dict[str, "EnterpriseOAuthConfig"] = Field(
        default_factory=dict,
        description="OAuth configs for enterprise instances"
    )

    # OAuth endpoints (GitHub.com defaults)
    authorization_endpoint: str = "https://github.com/login/oauth/authorize"
    token_endpoint: str = "https://github.com/login/oauth/access_token"
    device_code_endpoint: str = "https://github.com/login/device/code"

    # Scopes
    default_scopes: List[str] = Field(
        default_factory=lambda: ["repo", "read:org"],
        description="Default OAuth scopes"
    )

class EnterpriseOAuthConfig(BaseModel):
    """OAuth configuration for GitHub Enterprise Server."""
    host: str
    client_id: str
    client_secret: Optional[str] = None
    api_base_url: str  # e.g., https://github.company.com/api/v3
```

### OAuth Device Flow Implementation
```python
class GitHubOAuthDeviceFlow:
    """Implements GitHub OAuth Device Flow for CLI authentication."""

    def __init__(self, config: GitHubOAuthConfig):
        self.config = config

    def initiate_flow(
        self,
        *,
        scopes: Optional[List[str]] = None,
        github_host: str = "github.com",
    ) -> DeviceFlowResponse:
        """Initiate OAuth Device Flow."""
        # POST to device_code_endpoint
        # Return device_code, user_code, verification_uri

    def display_user_code(self, response: DeviceFlowResponse) -> None:
        """Display user code for verification."""
        print(f"""
        Please visit: {response.verification_uri}
        And enter code: {response.user_code}

        Waiting for authorization...
        """)

    def poll_for_token(
        self,
        device_code: str,
        interval: int = 5,
        timeout: int = 300,
    ) -> OAuthTokens:
        """Poll GitHub for token after user authorization."""
        # Poll token_endpoint with device_code
        # Respect interval to avoid rate limiting
        # Return tokens when authorized
        # Raise exception on timeout or denial

class DeviceFlowResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int  # Polling interval in seconds
```

### Memory Safety Utilities
```python
def clear_sensitive_string(s: str) -> None:
    """Attempt to clear string from memory (best effort)."""
    # Python doesn't allow true memory clearing due to immutable strings
    # But we can at least dereference and suggest GC
    del s
    import gc
    gc.collect()

@contextmanager
def secure_credential_context(
    credential_manager: GitHubCredentialManager,
    credential_id: str
):
    """Context manager for secure credential handling."""
    credentials = None
    try:
        credentials = credential_manager.retrieve_credentials(credential_id)
        yield credentials
    finally:
        if credentials:
            # Clear from memory
            if isinstance(credentials, OAuthTokens):
                clear_sensitive_string(credentials.access_token)
                if credentials.refresh_token:
                    clear_sensitive_string(credentials.refresh_token)
            elif isinstance(credentials, PersonalAccessToken):
                clear_sensitive_string(credentials.token)
            del credentials
            gc.collect()
```

## OAuth Device Flow CLI Integration

### Complete Flow Example
```bash
$ futurnal sources github add --repo octocat/Hello-World --auth oauth

Initiating GitHub OAuth Device Flow...

Please visit: https://github.com/login/device
And enter code: ABCD-1234

Waiting for authorization... (timeout in 5 minutes)

✓ Authorization successful!
✓ Tokens stored securely in keychain
✓ Repository registered: octocat/Hello-World
```

### Implementation
```python
def cli_oauth_device_flow(
    credential_manager: GitHubCredentialManager,
    github_host: str = "github.com",
) -> str:
    """Execute OAuth Device Flow from CLI."""
    flow = GitHubOAuthDeviceFlow(credential_manager._oauth)

    # Initiate flow
    response = flow.initiate_flow(github_host=github_host)

    # Display user code
    flow.display_user_code(response)

    # Poll for token
    try:
        tokens = flow.poll_for_token(
            device_code=response.device_code,
            interval=response.interval,
            timeout=response.expires_in,
        )
    except TimeoutError:
        raise RuntimeError("OAuth authorization timed out")
    except PermissionError:
        raise RuntimeError("OAuth authorization was denied")

    # Store tokens
    credential_id = f"github_cred_{uuid.uuid4().hex[:12]}"
    credential = credential_manager.store_oauth_tokens(
        credential_id=credential_id,
        github_host=github_host,
        tokens=tokens,
    )

    return credential_id
```

## Token Refresh Logic

### Automatic Refresh
```python
def should_refresh_token(tokens: OAuthTokens, buffer_seconds: int = 300) -> bool:
    """Check if token should be refreshed (5-minute buffer before expiration)."""
    if not tokens.expires_at:
        return False  # No expiration, no refresh needed
    return datetime.utcnow() + timedelta(seconds=buffer_seconds) >= tokens.expires_at

async def auto_refresh_wrapper(
    credential_manager: GitHubCredentialManager,
    credential_id: str,
) -> OAuthTokens:
    """Automatically refresh token if needed."""
    tokens = credential_manager.retrieve_credentials(credential_id)

    if isinstance(tokens, OAuthTokens) and should_refresh_token(tokens):
        logger.info("Auto-refreshing OAuth token", extra={"credential_id": credential_id})
        tokens = credential_manager.refresh_oauth_token(credential_id)

    return tokens
```

### Manual Refresh via CLI
```bash
$ futurnal sources github refresh-oauth <repo_id>

Refreshing OAuth token for repository...
✓ Token refreshed successfully
```

## Acceptance Criteria

- ✅ OAuth Device Flow completes successfully for GitHub.com
- ✅ OAuth tokens stored securely in OS keychain (macOS Keychain)
- ✅ Personal Access Tokens stored securely in OS keychain
- ✅ Credentials never appear in logs or audit events
- ✅ Credentials never persisted in plain files
- ✅ OAuth tokens automatically refreshed before expiration
- ✅ Manual token refresh works via CLI command
- ✅ Credentials cleared from memory after use (best effort)
- ✅ Token validation via GitHub API works
- ✅ Scope detection for PAT works correctly
- ✅ Credential deletion completely removes from keychain
- ✅ Audit events logged for all credential operations (without values)
- ✅ GitHub Enterprise Server OAuth support with custom endpoints

## Test Plan

### Unit Tests
- Keychain storage/retrieval (mock keyring)
- Token expiration detection
- OAuth Device Flow state machine
- Memory clearing utilities (verify GC behavior)
- Credential ID generation
- Scope parsing and validation

### Integration Tests
- End-to-end OAuth Device Flow with mock provider
- Token refresh with mock token endpoint
- Personal Access Token storage/retrieval
- Token validation via API
- Credential deletion verification
- Audit event generation
- GitHub Enterprise Server OAuth

### Security Tests
- Credentials never logged (inspect all log output)
- Credentials never in exception messages
- Memory cleared after context manager exit
- Keychain deletion leaves no residue
- OAuth state parameter prevents CSRF
- Token validation prevents invalid credentials

### Provider-Specific Tests
- GitHub.com OAuth Device Flow (manual test)
- Personal Access Token with various scopes (manual test)
- Token refresh with real GitHub (manual test)
- Enterprise Server OAuth (manual test with test instance)

## Implementation Notes

### Python Keyring Backend Selection
```python
import keyring

# Prefer most secure backend
if sys.platform == "darwin":
    # macOS Keychain
    keyring.set_keyring(keyring.backends.macOS.Keyring())
elif sys.platform == "win32":
    # Windows Credential Manager
    keyring.set_keyring(keyring.backends.Windows.WinVaultKeyring())
else:
    # Linux Secret Service
    keyring.set_keyring(keyring.backends.SecretService.Keyring())
```

### Token Expiration Handling
```python
def calculate_expiration(expires_in: Optional[int]) -> Optional[datetime]:
    """Calculate absolute expiration time from relative expires_in."""
    if expires_in is None:
        return None  # No expiration
    return datetime.utcnow() + timedelta(seconds=expires_in)
```

### Audit Event Structure
```python
AuditEvent(
    job_id=f"credential_{operation}_{credential_id}",
    source="github_credential_manager",
    action=f"credential_{operation}",  # e.g., credential_create, credential_refresh
    status="success",
    timestamp=datetime.utcnow(),
    metadata={
        "credential_id": credential_id,
        "credential_type": credential_type.value,
        "github_host": github_host,
        "username_hash": sha256(username.encode()).hexdigest()[:16] if username else None,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "scopes": scopes,
    },
    operator_action=operator,
)
```

## Open Questions

- Should we support GitHub App installation tokens?
- How to handle organization-level OAuth Apps vs personal?
- Should we support fine-grained personal access tokens (beta)?
- How to handle multiple credentials for the same repository (team collaboration)?
- Should we support credential sharing across multiple repositories?
- How to migrate credentials between machines (export/import)?
- Should we cache tokens in memory for repeated API calls (security vs performance)?

## Dependencies
- Python keyring library (`pip install keyring`)
- GitHubKit for OAuth and API interactions (`pip install githubkit`)
- AuditLogger for credential operation logging
- Environment variables for OAuth client IDs/secrets (optional for device flow)


