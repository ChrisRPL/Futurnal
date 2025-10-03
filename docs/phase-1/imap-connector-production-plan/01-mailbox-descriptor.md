Summary: Specify mailbox metadata model, storage, and registration flow for IMAP mailboxes.

# 01 · Mailbox Descriptor

## Purpose
Define the IMAP-specific source descriptor used by the ingestion orchestrator: mailbox UID, server configuration, folder selection, credential reference, and privacy flags. Ensures the connector can register, authenticate, and monitor mailboxes safely with explicit AI learning consent.

## Scope
- Descriptor schema with required/optional fields
- Persistent storage location and format with encryption metadata
- Validation rules and conflict handling
- Integration points with `src/futurnal/ingestion/imap/*` and orchestrator scheduler
- CLI registration flow with OAuth/app-password modes

## Requirements Alignment
- Privacy by default; all processing is local unless explicitly escalated
- Explicit consent for AI experiential learning from email communications
- OAuth2-first for Gmail/Office 365; app-password fallback
- Feed Unstructured.io parsing and emit semantic triples into PKG and vectors
- Keep graph and vector indices synchronized during updates

## Data Model

### ImapMailboxDescriptor
```python
class AuthMode(str, Enum):
    OAUTH2 = "oauth2"
    APP_PASSWORD = "app_password"

class ImapMailboxDescriptor(BaseModel):
    # Identity
    id: str  # Deterministic ULID/UUIDv7 derived from email + server
    name: Optional[str]  # Human label (e.g., "Work Email")
    icon: Optional[str]  # Optional emoji or icon

    # Server Configuration
    imap_host: str  # e.g., imap.gmail.com
    imap_port: int = 993  # Default IMAPS port
    email_address: str  # User's email address

    # Authentication
    auth_mode: AuthMode
    credential_id: str  # Reference to keychain credential

    # Folder Selection
    folders: List[str] = Field(default_factory=lambda: ["INBOX"])
    folder_patterns: List[str] = Field(default_factory=list)  # glob patterns
    exclude_folders: List[str] = Field(default_factory=lambda: ["[Gmail]/Trash", "[Gmail]/Spam"])

    # Temporal Scope
    sync_from_date: Optional[datetime] = None  # Only sync emails after this date
    max_message_age_days: Optional[int] = None  # Only sync recent messages

    # Privacy & Consent
    privacy_settings: MailboxPrivacySettings

    # Provenance
    created_at: datetime
    updated_at: datetime
    provenance: Provenance  # OS user, machine hash, tool version
```

### MailboxPrivacySettings
```python
class MailboxPrivacySettings(BaseModel):
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD

    # Consent scopes
    required_consent_scopes: List[ConsentScope] = Field(
        default_factory=lambda: [
            ConsentScope.MAILBOX_ACCESS,
            ConsentScope.EMAIL_CONTENT_ANALYSIS,
        ]
    )

    # Redaction
    enable_sender_anonymization: bool = True
    enable_recipient_anonymization: bool = True
    enable_subject_redaction: bool = False  # Only for audit logs
    redact_email_patterns: List[str] = Field(default_factory=list)  # Regex patterns

    # Content filtering
    exclude_email_patterns: List[str] = Field(default_factory=list)  # Skip these emails
    privacy_subject_keywords: List[str] = Field(
        default_factory=lambda: ["private", "confidential", "nda"]
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

    # IMAP-specific scopes
    MAILBOX_ACCESS = "imap:mailbox:access"
    EMAIL_CONTENT_ANALYSIS = "imap:email:content_analysis"
    ATTACHMENT_EXTRACTION = "imap:email:attachment_extraction"
    THREAD_RECONSTRUCTION = "imap:email:thread_reconstruction"
    PARTICIPANT_ANALYSIS = "imap:email:participant_analysis"
    CLOUD_MODELS = "imap:email:cloud_models"
```

## Storage

### File Location
- Stored in workspace registry under `~/.futurnal/sources/imap/<mailbox_id>.json`
- Credentials stored separately in OS keychain (never in descriptor files)
- Transaction-safe updates guarded by file locks
- Backed up by `workspace/backup.py` routines (excluding credentials)

### Encryption Metadata
Descriptor stores metadata for credential retrieval but never the credentials themselves:
```json
{
  "id": "mailbox_01HQWX...",
  "credential_id": "imap_cred_01HQWX...",
  "auth_mode": "oauth2",
  "imap_host": "imap.gmail.com",
  ...
}
```

Actual tokens/passwords stored via Python keyring:
```python
import keyring
keyring.set_password("futurnal.imap", credential_id, token_json)
```

## CLI Registration Flow

### OAuth2 Flow (Recommended)
```bash
futurnal sources imap add \
  --email user@gmail.com \
  --name "Personal Gmail" \
  --provider gmail \
  --auth oauth2
```

Steps:
1. Validate email format and provider
2. Launch OAuth2 consent flow (browser-based)
3. Exchange authorization code for tokens
4. Store tokens in OS keychain with refresh token
5. Test IMAP connection
6. Prompt for folder selection (default: INBOX)
7. Create descriptor with credential reference
8. Persist descriptor JSON

### App Password Flow (Fallback)
```bash
futurnal sources imap add \
  --email user@example.com \
  --name "Work Email" \
  --host imap.example.com \
  --auth app-password
```

Steps:
1. Prompt for app-specific password (secure input)
2. Validate password format
3. Store password in OS keychain
4. Test IMAP connection
5. Prompt for folder selection
6. Create descriptor with credential reference
7. Persist descriptor JSON

### Management Commands
```bash
futurnal sources imap list
futurnal sources imap inspect <mailbox_id>
futurnal sources imap update <mailbox_id> --folders "INBOX,Sent"
futurnal sources imap test-connection <mailbox_id>
futurnal sources imap refresh-oauth <mailbox_id>
futurnal sources imap remove <mailbox_id> [--delete-credentials]
```

## Validation Rules

### Required Validations
- Email address must be valid format (RFC 5322)
- IMAP host must be resolvable
- Port must be valid (1-65535, typically 993 for IMAPS)
- Credentials must be testable (attempt connection)
- Reject duplicates: same email + host → same mailbox id
- Warn on non-TLS connections (port 143)
- Validate folder names exist on server

### Security Validations
- Enforce TLS for all connections (reject port 143 unless explicit override)
- Validate OAuth provider URLs match known providers
- Check credential expiration for OAuth tokens
- Warn on insecure authentication methods

### Provider-Specific Validations
**Gmail**:
- Host must be `imap.gmail.com`
- OAuth2 required (app passwords deprecated)
- Warn about Gmail-specific folder names (`[Gmail]/...`)

**Office 365**:
- Host typically `outlook.office365.com`
- OAuth2 required
- Validate tenant-specific settings

**Generic IMAP**:
- Allow custom host/port
- App password supported
- Validate server capabilities (IDLE, MODSEQ support)

## Acceptance Criteria

- ✅ Creating, reading, and listing descriptors works via CLI and programmatic API
- ✅ OAuth2 flow completes successfully for Gmail and Office 365
- ✅ App password flow works for generic IMAP servers
- ✅ Credentials never appear in descriptor files or logs
- ✅ Connection testing validates credentials before persisting
- ✅ Folder selection respects server-side folder structure
- ✅ Duplicate registration is idempotent (updates existing descriptor)
- ✅ Privacy settings correctly filter out sensitive patterns
- ✅ Backups include descriptor files but exclude credentials
- ✅ Audit events logged for registration/update/removal actions

## Test Plan

### Unit Tests
- Schema validation for all descriptor fields
- ID determinism (same email + host → same ID)
- Privacy pattern matching (subject keywords, email patterns)
- Folder pattern expansion (glob matching)
- Validation rule enforcement

### Integration Tests
- CLI add/list/inspect/update/remove commands
- OAuth2 flow with mock OAuth provider
- App password flow with secure input
- Connection testing with mock IMAP server
- Credential storage/retrieval via keyring
- Folder enumeration from server
- Duplicate registration handling

### Security Tests
- Credentials never logged or persisted in files
- TLS enforcement (reject port 143 connections)
- OAuth token encryption at rest
- Secure credential deletion on removal

## Implementation Notes

### Deterministic ID Generation
```python
def _deterministic_mailbox_id(email: str, imap_host: str) -> str:
    """Generate deterministic mailbox ID from email + host."""
    normalized = f"{email.lower()}@{imap_host.lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"imap:{normalized}"))
```

### Credential Reference Pattern
```python
def _create_credential_id(mailbox_id: str) -> str:
    """Create credential ID for keychain storage."""
    return f"imap_cred_{mailbox_id}"
```

### Provider Detection
```python
KNOWN_PROVIDERS = {
    "gmail": {"host": "imap.gmail.com", "port": 993, "oauth_required": True},
    "office365": {"host": "outlook.office365.com", "port": 993, "oauth_required": True},
    "yahoo": {"host": "imap.mail.yahoo.com", "port": 993, "oauth_required": False},
}
```

## Open Questions

- Should we auto-detect provider from email domain (e.g., `@gmail.com` → Gmail settings)?
- How to handle multi-folder sync priority (process INBOX first)?
- Should we support IMAP over STARTTLS (port 143 + STARTTLS) or only IMAPS (port 993)?
- How to handle server capability detection (IDLE, MODSEQ, QRESYNC)?
- Should we cache folder list or fetch fresh on each sync?
- How to handle mailbox rename/migration (preserve history)?

## Dependencies
- Python keyring library for OS credential storage
- OAuth2 client library for Gmail/Office 365 flows
- IMAPClient for server connection testing
- ConsentRegistry for privacy approval workflow
- AuditLogger for registration event tracking


