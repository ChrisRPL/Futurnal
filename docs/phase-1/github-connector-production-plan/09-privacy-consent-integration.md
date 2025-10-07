Summary: Integrate repository-level consent management and privacy-aware audit logging.

# 09 · Privacy & Consent Integration

## Purpose
Implement comprehensive privacy controls for GitHub repository ingestion, including explicit consent scopes, file pattern exclusions, secret detection, and privacy-aware audit logging. Ensure no source code or sensitive data appears in logs or telemetry.

## Scope
- Repository-level consent scopes
- File pattern exclusion (secrets, credentials)
- Secret detection in configuration files
- Privacy-aware audit logging (no code in logs)
- Path redaction for sensitive repositories
- Participant anonymization (optional)
- Data retention policies
- Consent revocation and data cleanup

## Requirements Alignment
- **Privacy-first**: All data processing requires explicit consent
- **Local processing**: Default to on-device, no cloud escalation without consent
- **Audit transparency**: All operations logged without exposing content
- **User control**: Granular consent scopes, easy revocation
- **Security**: Detect and exclude files with secrets

## Consent Scopes

### GitHub-Specific Scopes
```python
class GitHubConsentScope(str, Enum):
    """Consent scopes for GitHub repository ingestion."""

    # Basic access
    GITHUB_REPO_ACCESS = "github:repo:access"
    """Permission to read repository metadata and structure"""

    # Code analysis
    GITHUB_CODE_ANALYSIS = "github:repo:code_analysis"
    """Permission to analyze source code and generate embeddings"""

    # Metadata
    GITHUB_ISSUE_METADATA = "github:repo:issue_metadata"
    """Permission to ingest issue discussions"""

    GITHUB_PR_METADATA = "github:repo:pr_metadata"
    """Permission to ingest pull request reviews"""

    GITHUB_WIKI_ACCESS = "github:repo:wiki_access"
    """Permission to access repository wiki"""

    # Advanced processing
    GITHUB_CLOUD_MODELS = "github:repo:cloud_models"
    """Permission to use cloud models for analysis (optional escalation)"""

    GITHUB_PARTICIPANT_ANALYSIS = "github:repo:participant_analysis"
    """Permission to analyze contributor patterns and relationships"""
```

### Consent Registration
```python
def register_repository_consent(
    consent_registry: ConsentRegistry,
    repo_descriptor: GitHubRepositoryDescriptor,
    operator: Optional[str] = None,
) -> List[ConsentRecord]:
    """Register consent for repository ingestion."""

    required_scopes = repo_descriptor.privacy_settings.required_consent_scopes

    records = []
    for scope in required_scopes:
        record = consent_registry.grant(
            source=repo_descriptor.id,
            scope=scope.value,
            operator=operator,
            duration_hours=None,  # No expiration
        )
        records.append(record)

    return records
```

## Data Model

### RepositoryPrivacySettings
```python
class RepositoryPrivacySettings(BaseModel):
    """Privacy configuration for repository ingestion."""

    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD

    # Consent scopes
    required_consent_scopes: List[GitHubConsentScope] = Field(
        default_factory=lambda: [
            GitHubConsentScope.GITHUB_REPO_ACCESS,
            GitHubConsentScope.GITHUB_CODE_ANALYSIS,
        ]
    )

    # File exclusions
    enable_path_anonymization: bool = Field(
        default=True,
        description="Redact file paths in logs and audit trails"
    )
    enable_author_anonymization: bool = Field(
        default=False,
        description="Redact commit/issue authors in logs"
    )

    redact_file_patterns: List[str] = Field(
        default_factory=lambda: [
            "*secret*",
            "*password*",
            "*token*",
            ".env*",
            "credentials.*",
            "*.key",
            "*.pem",
        ],
        description="Files matching these patterns are excluded completely"
    )

    # Secret detection
    detect_secrets: bool = Field(
        default=True,
        description="Scan files for secret patterns"
    )

    secret_patterns: List[str] = Field(
        default_factory=lambda: [
            r"(?i)(api[_-]?key|apikey)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})",
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub PAT
            r"gho_[a-zA-Z0-9]{36}",  # GitHub OAuth
            r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
        ],
        description="Regex patterns for secret detection"
    )

    # Content filtering
    exclude_extensions: List[str] = Field(
        default_factory=lambda: [
            ".exe", ".dll", ".so", ".dylib",  # Binaries
            ".jpg", ".png", ".gif", ".mp4",   # Media
            ".zip", ".tar", ".gz",            # Archives
        ]
    )

    max_file_size_mb: int = Field(
        default=10,
        description="Skip files larger than this"
    )

    # Audit settings
    audit_sync_events: bool = True
    audit_content_changes: bool = False  # Only checksums, not content
    retain_audit_days: int = 90
```

## Privacy-Aware Audit Logging

### Audit Events
```python
class GitHubAuditEvent(BaseModel):
    """Audit event for GitHub operations."""

    job_id: str
    source: str  # "github_connector"
    action: str  # "repository_sync", "issue_fetch", etc.
    status: str  # "success", "failure", "skipped"
    timestamp: datetime

    # Repository context (safe to log)
    repo_id: str
    repo_full_name_hash: str  # SHA256 hash of owner/repo
    branch: Optional[str] = None

    # Aggregated statistics (no content)
    files_processed: int = 0
    files_skipped: int = 0
    bytes_processed: int = 0
    commits_processed: int = 0

    # File path hashes (no actual paths)
    file_path_hashes: List[str] = Field(default_factory=list)

    # Error information (sanitized)
    error_type: Optional[str] = None
    error_message_sanitized: Optional[str] = None

    # Operator
    operator_action: Optional[str] = None
```

### Audit Logger Integration
```python
class GitHubAuditLogger:
    """Privacy-aware audit logger for GitHub connector."""

    def __init__(
        self,
        audit_logger: AuditLogger,
        redaction_policy: RedactionPolicy,
    ):
        self.audit_logger = audit_logger
        self.redaction_policy = redaction_policy

    def log_repository_sync(
        self,
        repo_id: str,
        repo_full_name: str,
        branch: str,
        result: SyncResult,
        status: str = "success",
    ):
        """Log repository sync event."""

        event = GitHubAuditEvent(
            job_id=f"github_sync_{repo_id}_{int(time.time())}",
            source="github_connector",
            action="repository_sync",
            status=status,
            timestamp=datetime.utcnow(),
            repo_id=repo_id,
            repo_full_name_hash=self._hash_repo_name(repo_full_name),
            branch=branch,
            files_processed=result.files_synced,
            commits_processed=result.commits_processed,
            bytes_processed=result.bytes_synced,
            file_path_hashes=[
                self._hash_path(p) for p in result.modified_files[:100]
            ],  # Limit to 100 for log size
        )

        self.audit_logger.log_event(event)

    def log_consent_check(
        self,
        repo_id: str,
        scope: GitHubConsentScope,
        granted: bool,
    ):
        """Log consent check result."""

        event = GitHubAuditEvent(
            job_id=f"consent_check_{repo_id}",
            source="github_connector",
            action=f"consent_check_{scope.value}",
            status="granted" if granted else "denied",
            timestamp=datetime.utcnow(),
            repo_id=repo_id,
            repo_full_name_hash="",  # Not needed for consent check
        )

        self.audit_logger.log_event(event)

    def _hash_repo_name(self, repo_full_name: str) -> str:
        """Hash repository name for privacy."""
        return sha256(repo_full_name.encode()).hexdigest()[:16]

    def _hash_path(self, file_path: str) -> str:
        """Hash file path for privacy."""
        return sha256(file_path.encode()).hexdigest()[:16]
```

## Consent-Guarded Operations

### Consent Wrapper
```python
@contextmanager
def require_consent(
    consent_registry: ConsentRegistry,
    repo_id: str,
    scope: GitHubConsentScope,
    audit_logger: Optional[GitHubAuditLogger] = None,
):
    """Context manager to enforce consent for operations."""

    # Check consent
    consent = consent_registry.get(source=repo_id, scope=scope.value)

    if not consent or not consent.is_active():
        if audit_logger:
            audit_logger.log_consent_check(repo_id, scope, granted=False)

        raise ConsentRequiredError(
            f"Consent required for {scope.value} on repository {repo_id}"
        )

    # Log consent granted
    if audit_logger:
        audit_logger.log_consent_check(repo_id, scope, granted=True)

    try:
        yield consent
    except Exception as e:
        # Log failure (sanitized)
        logger.error(f"Operation failed: {type(e).__name__}")
        raise

# Usage
async def sync_repository(descriptor: GitHubRepositoryDescriptor):
    """Sync repository with consent checking."""

    with require_consent(
        consent_registry,
        descriptor.id,
        GitHubConsentScope.GITHUB_REPO_ACCESS,
        audit_logger,
    ):
        # Perform sync operations
        result = await sync_engine.sync_repository(descriptor)
        return result
```

## Secret Detection

### Secret Scanner
```python
class SecretScanner:
    """Scans files for secrets and sensitive data."""

    def __init__(self, patterns: List[str]):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def scan_file(self, file_path: str, content: str) -> bool:
        """Check if file contains secrets."""

        for pattern in self.patterns:
            if pattern.search(content):
                logger.warning(
                    f"Secret pattern detected in file (path hash: "
                    f"{sha256(file_path.encode()).hexdigest()[:16]})"
                )
                return True

        return False

    def should_exclude_file(
        self,
        file_path: str,
        content: Optional[str],
        privacy_settings: RepositoryPrivacySettings,
    ) -> bool:
        """Determine if file should be excluded due to secrets."""

        if not privacy_settings.detect_secrets:
            return False

        if content and self.scan_file(file_path, content):
            return True

        return False
```

## Path Redaction

### Redaction Policy
```python
def create_github_redaction_policy(
    privacy_settings: RepositoryPrivacySettings,
) -> RedactionPolicy:
    """Create redaction policy for GitHub repositories."""

    return RedactionPolicy(
        redact_paths=privacy_settings.enable_path_anonymization,
        redact_patterns=privacy_settings.redact_file_patterns,
        hash_algorithm="sha256",
        hash_length=16,
    )
```

## Consent Revocation

### Data Cleanup on Revocation
```python
async def revoke_repository_consent(
    consent_registry: ConsentRegistry,
    repo_id: str,
    scope: GitHubConsentScope,
    cleanup_data: bool = True,
):
    """Revoke consent and optionally cleanup data."""

    # Revoke consent
    consent_registry.revoke(
        source=repo_id,
        scope=scope.value,
    )

    # Cleanup data if requested
    if cleanup_data:
        if scope == GitHubConsentScope.GITHUB_CODE_ANALYSIS:
            # Remove code embeddings from vector store
            await cleanup_code_embeddings(repo_id)

        if scope == GitHubConsentScope.GITHUB_ISSUE_METADATA:
            # Remove issue data from PKG
            await cleanup_issue_data(repo_id)

        # ... more cleanup logic
```

## Acceptance Criteria

- ✅ Repository registration requires explicit consent
- ✅ Consent checked before all operations
- ✅ ConsentRequiredError raised when consent missing
- ✅ File patterns correctly exclude sensitive files
- ✅ Secret detection flags credential patterns
- ✅ Audit logs contain no source code or file paths
- ✅ Path redaction works with hash-based anonymization
- ✅ Consent revocation triggers data cleanup
- ✅ Participant anonymization optional
- ✅ Audit log retention policies enforced

## Test Plan

### Unit Tests
- Consent scope enumeration
- Secret pattern matching
- Path redaction logic
- Audit event sanitization

### Integration Tests
- End-to-end consent flow
- Consent revocation and cleanup
- Secret detection on sample files
- Audit log generation
- Path redaction in logs

### Security Tests
- Ensure no code in audit logs
- Verify secret patterns catch test secrets
- Confirm path redaction prevents path leakage
- Test consent enforcement (reject without consent)

## Implementation Notes

### Consent CLI Commands
```bash
# Grant consent
futurnal sources github consent grant <repo_id> \
  --scope github:repo:code_analysis

# Revoke consent
futurnal sources github consent revoke <repo_id> \
  --scope github:repo:code_analysis \
  --cleanup-data

# List consents
futurnal sources github consent list <repo_id>
```

## Open Questions

- Should we support time-limited consent (expires after N days)?
- How to handle partial consent (some scopes but not all)?
- Should we implement consent inheritance for organization repositories?
- How to audit consent changes over time?

## Dependencies
- ConsentRegistry from privacy framework
- AuditLogger from privacy framework
- RedactionPolicy from privacy framework
- re (stdlib) for pattern matching


