Summary: Implement secure credential storage and OAuth token management using OS-native keychains.

# 02 · Credential Manager

## Purpose
Provide secure, OS-native storage for IMAP credentials (OAuth tokens, app passwords) with automatic token refresh, memory clearing after use, and audit-compliant credential lifecycle management. Ensures the Ghost never exposes credentials in logs, files, or process memory longer than necessary.

## Scope
- OS keychain integration via Python keyring library
- OAuth2 token storage with refresh token handling
- App password secure storage
- Automatic token refresh before expiration
- Credential lifecycle management (create, retrieve, update, delete)
- Memory-safe credential handling (clear after use)
- Audit logging for credential operations (without exposing credentials)

## Requirements Alignment
- **Privacy-first**: Credentials never logged, never in files, cleared from memory after use
- **OAuth2-first**: Prefer OAuth2 over app passwords for supported providers
- **Automatic refresh**: Tokens refreshed transparently before expiration
- **Audit compliance**: Log credential operations without exposing values
- **Secure deletion**: Completely remove credentials on mailbox removal

## Data Model

### CredentialStorage (Abstract)
```python
class CredentialType(str, Enum):
    OAUTH2 = "oauth2"
    APP_PASSWORD = "app_password"

class ImapCredential(BaseModel):
    credential_id: str
    credential_type: CredentialType
    email_address: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None  # For OAuth2 tokens

class OAuth2Tokens(BaseModel):
    """Never persisted in plain files - only in OS keychain."""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds
    expires_at: datetime  # Calculated from expires_in
    scope: List[str] = Field(default_factory=list)

class AppPassword(BaseModel):
    """Never persisted in plain files - only in OS keychain."""
    password: str
```

### Keychain Storage Format
```python
# Stored in OS keychain under service name "futurnal.imap"
# Account name is the credential_id
keyring.set_password(
    "futurnal.imap",
    credential_id,
    json.dumps({
        "type": "oauth2",
        "access_token": "ya29.a0...",
        "refresh_token": "1//...",
        "expires_at": "2024-01-15T12:00:00Z",
        ...
    })
)
```

## Component Design

### CredentialManager
```python
class CredentialManager:
    """Manages IMAP credentials with OS keychain integration."""

    def __init__(
        self,
        *,
        audit_logger: Optional[AuditLogger] = None,
        oauth_provider_registry: Optional[OAuthProviderRegistry] = None,
    ):
        self._audit = audit_logger
        self._providers = oauth_provider_registry or OAuthProviderRegistry()
        self._service_name = "futurnal.imap"

    def store_oauth_tokens(
        self,
        *,
        credential_id: str,
        email_address: str,
        tokens: OAuth2Tokens,
        operator: Optional[str] = None,
    ) -> ImapCredential:
        """Store OAuth2 tokens in OS keychain."""
        # Store tokens
        # Log audit event (without token values)
        # Return credential metadata

    def store_app_password(
        self,
        *,
        credential_id: str,
        email_address: str,
        password: str,
        operator: Optional[str] = None,
    ) -> ImapCredential:
        """Store app password in OS keychain."""
        # Store password
        # Clear password from memory
        # Log audit event
        # Return credential metadata

    def retrieve_credentials(
        self,
        credential_id: str,
    ) -> Union[OAuth2Tokens, AppPassword]:
        """Retrieve credentials from OS keychain.

        For OAuth2: Automatically refreshes if expired.
        For app password: Returns password (caller must clear).
        """
        # Retrieve from keychain
        # If OAuth2 and expired, refresh automatically
        # Log audit event (credential access)
        # Return credentials

    def refresh_oauth_token(
        self,
        credential_id: str,
    ) -> OAuth2Tokens:
        """Manually refresh OAuth2 token."""
        # Retrieve current tokens
        # Call provider refresh endpoint
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

    def list_credentials(self) -> List[ImapCredential]:
        """List credential metadata (not values)."""
        # Scan keychain for futurnal.imap entries
        # Return metadata only
```

### OAuthProviderRegistry
```python
class OAuthProvider(BaseModel):
    name: str
    authorization_endpoint: str
    token_endpoint: str
    scopes: List[str]
    client_id: str  # From environment or config
    client_secret: Optional[str] = None  # For confidential clients

class OAuthProviderRegistry:
    """Registry of OAuth2 providers for IMAP."""

    GMAIL = OAuthProvider(
        name="gmail",
        authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        scopes=[
            "https://mail.google.com/",  # Full IMAP access
        ],
        client_id=os.getenv("FUTURNAL_GMAIL_CLIENT_ID"),
    )

    OFFICE365 = OAuthProvider(
        name="office365",
        authorization_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        scopes=[
            "https://outlook.office.com/IMAP.AccessAsUser.All",
            "offline_access",
        ],
        client_id=os.getenv("FUTURNAL_OFFICE365_CLIENT_ID"),
    )

    def get_provider(self, name: str) -> OAuthProvider:
        """Get provider configuration by name."""

    def exchange_code_for_tokens(
        self,
        provider: OAuthProvider,
        authorization_code: str,
        redirect_uri: str,
    ) -> OAuth2Tokens:
        """Exchange authorization code for tokens."""

    def refresh_access_token(
        self,
        provider: OAuthProvider,
        refresh_token: str,
    ) -> OAuth2Tokens:
        """Refresh access token using refresh token."""
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
def secure_credential_context(credential_manager: CredentialManager, credential_id: str):
    """Context manager for secure credential handling."""
    credentials = None
    try:
        credentials = credential_manager.retrieve_credentials(credential_id)
        yield credentials
    finally:
        if credentials:
            # Clear from memory
            if isinstance(credentials, OAuth2Tokens):
                clear_sensitive_string(credentials.access_token)
                clear_sensitive_string(credentials.refresh_token)
            elif isinstance(credentials, AppPassword):
                clear_sensitive_string(credentials.password)
            del credentials
            gc.collect()
```

## OAuth2 Flow Implementation

### Authorization Flow
```python
def initiate_oauth_flow(
    provider: OAuthProvider,
    state: str,
    redirect_uri: str = "http://localhost:8080/callback",
) -> str:
    """Generate OAuth2 authorization URL."""
    params = {
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(provider.scopes),
        "state": state,
        "access_type": "offline",  # Request refresh token
        "prompt": "consent",  # Force consent screen for refresh token
    }
    return f"{provider.authorization_endpoint}?{urlencode(params)}"
```

### Local Callback Server
```python
class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Handle OAuth2 callback redirect."""

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            self.server.authorization_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authorization successful!</h1><p>You can close this window.</p>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h1>Authorization failed</h1>")

def run_oauth_callback_server(port: int = 8080, timeout: int = 120) -> Optional[str]:
    """Run local server to capture OAuth2 callback."""
    server = http.server.HTTPServer(("localhost", port), OAuthCallbackHandler)
    server.timeout = timeout
    server.authorization_code = None

    while server.authorization_code is None:
        server.handle_request()

    return server.authorization_code
```

### Token Refresh Logic
```python
def should_refresh_token(tokens: OAuth2Tokens, buffer_seconds: int = 300) -> bool:
    """Check if token should be refreshed (5-minute buffer before expiration)."""
    if not tokens.expires_at:
        return False
    return datetime.utcnow() + timedelta(seconds=buffer_seconds) >= tokens.expires_at

async def auto_refresh_wrapper(
    credential_manager: CredentialManager,
    credential_id: str,
) -> OAuth2Tokens:
    """Automatically refresh token if needed."""
    tokens = credential_manager.retrieve_credentials(credential_id)

    if isinstance(tokens, OAuth2Tokens) and should_refresh_token(tokens):
        logger.info("Auto-refreshing OAuth2 token", extra={"credential_id": credential_id})
        tokens = credential_manager.refresh_oauth_token(credential_id)

    return tokens
```

## Acceptance Criteria

- ✅ OAuth2 tokens stored securely in OS keychain (macOS Keychain)
- ✅ App passwords stored securely in OS keychain
- ✅ Credentials never appear in logs or audit events
- ✅ Credentials never persisted in plain files
- ✅ OAuth2 tokens automatically refreshed before expiration
- ✅ Manual token refresh works via CLI command
- ✅ Credentials cleared from memory after use (best effort)
- ✅ OAuth2 authorization flow completes successfully for Gmail
- ✅ OAuth2 authorization flow completes successfully for Office 365
- ✅ App password flow works for generic IMAP servers
- ✅ Credential deletion completely removes from keychain
- ✅ Audit events logged for all credential operations (without values)
- ✅ Connection failures due to expired credentials trigger auto-refresh

## Test Plan

### Unit Tests
- Keychain storage/retrieval (mock keyring)
- Token expiration detection
- OAuth2 provider configuration
- Memory clearing utilities (verify GC behavior)
- Credential ID generation

### Integration Tests
- End-to-end OAuth2 flow with mock provider
- Token refresh with mock token endpoint
- App password storage/retrieval
- Credential deletion verification
- Audit event generation

### Security Tests
- Credentials never logged (inspect all log output)
- Credentials never in exception messages
- Memory cleared after context manager exit
- Keychain deletion leaves no residue
- OAuth2 PKCE support (future enhancement)

### Provider-Specific Tests
- Gmail OAuth2 flow with real endpoints (manual test)
- Office 365 OAuth2 flow with real endpoints (manual test)
- Token refresh with real providers (manual test)

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
def calculate_expiration(expires_in: int) -> datetime:
    """Calculate absolute expiration time from relative expires_in."""
    return datetime.utcnow() + timedelta(seconds=expires_in)
```

### Audit Event Structure
```python
AuditEvent(
    job_id=f"credential_{operation}_{credential_id}",
    source="imap_credential_manager",
    action=f"credential_{operation}",  # e.g., credential_create, credential_refresh
    status="success",
    timestamp=datetime.utcnow(),
    metadata={
        "credential_id": credential_id,
        "credential_type": credential_type.value,
        "email_address_hash": sha256(email_address.encode()).hexdigest()[:16],
        "expires_at": expires_at.isoformat() if expires_at else None,
    },
    operator_action=operator,
)
```

## Open Questions

- Should we support OAuth2 PKCE (Proof Key for Code Exchange) for enhanced security?
- How to handle multiple OAuth2 providers for the same email (e.g., Gmail via Google Workspace)?
- Should we cache decrypted credentials in memory for a short period to reduce keychain access?
- How to handle credential expiration during long-running sync operations?
- Should we support hardware security modules (HSM) for enterprise deployments?
- How to migrate credentials between machines (export/import)?

## Dependencies
- Python keyring library (`pip install keyring`)
- OAuth2 client library (`pip install authlib` or `requests-oauthlib`)
- AuditLogger for credential operation logging
- Environment variables for OAuth2 client IDs/secrets


