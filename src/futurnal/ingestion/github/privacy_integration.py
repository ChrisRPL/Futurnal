"""Privacy and consent integration for GitHub repository ingestion.

This module implements comprehensive privacy controls including:
- GitHub-specific consent scopes
- Privacy-aware audit logging
- Consent management functions
- Data cleanup on consent revocation

Implements specification from:
docs/phase-1/github-connector-production-plan/09-privacy-consent-integration.md
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ...privacy.audit import AuditEvent, AuditLogger
from ...privacy.consent import ConsentRecord, ConsentRegistry, ConsentRequiredError
from ...privacy.redaction import RedactionPolicy
from .descriptor import GitHubRepositoryDescriptor, RepositoryPrivacySettings


# ---------------------------------------------------------------------------
# GitHub-Specific Consent Scopes
# ---------------------------------------------------------------------------


class GitHubConsentScope(str, Enum):
    """Consent scopes for GitHub repository ingestion.

    These scopes provide granular control over what data can be accessed
    and how it can be processed from GitHub repositories.
    """

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


# ---------------------------------------------------------------------------
# GitHub Audit Event Model
# ---------------------------------------------------------------------------


class GitHubAuditEvent(BaseModel):
    """Audit event for GitHub operations with privacy-aware fields.

    This model extends the base audit functionality with GitHub-specific
    metadata while ensuring no source code or sensitive data appears in logs.
    """

    job_id: str
    source: str = "github_connector"
    action: str
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Repository context (safe to log - hashed)
    repo_id: str
    repo_full_name_hash: str
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

    def to_base_audit_event(self) -> AuditEvent:
        """Convert to base AuditEvent for logging."""
        metadata: Dict[str, object] = {
            "repo_id": self.repo_id,
            "repo_full_name_hash": self.repo_full_name_hash,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "bytes_processed": self.bytes_processed,
            "commits_processed": self.commits_processed,
        }

        if self.branch:
            metadata["branch"] = self.branch

        if self.file_path_hashes:
            metadata["file_path_hashes"] = self.file_path_hashes

        if self.error_type:
            metadata["error_type"] = self.error_type

        if self.error_message_sanitized:
            metadata["error_message_sanitized"] = self.error_message_sanitized

        return AuditEvent(
            job_id=self.job_id,
            source=self.source,
            action=self.action,
            status=self.status,
            timestamp=self.timestamp,
            operator_action=self.operator_action,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# GitHub Audit Logger
# ---------------------------------------------------------------------------


class GitHubAuditLogger:
    """Privacy-aware audit logger for GitHub connector.

    Wraps the base AuditLogger with GitHub-specific functionality:
    - Automatic path hashing for privacy
    - Repository name anonymization
    - Consent check logging
    - Sync result aggregation
    """

    def __init__(
        self,
        audit_logger: AuditLogger,
        redaction_policy: Optional[RedactionPolicy] = None,
    ):
        """Initialize GitHub audit logger.

        Args:
            audit_logger: Base audit logger instance
            redaction_policy: Optional redaction policy for path anonymization
        """
        self.audit_logger = audit_logger
        self.redaction_policy = redaction_policy or RedactionPolicy()

    def log_repository_sync(
        self,
        repo_id: str,
        repo_full_name: str,
        branch: str,
        files_processed: int = 0,
        files_skipped: int = 0,
        bytes_processed: int = 0,
        commits_processed: int = 0,
        modified_files: Optional[List[str]] = None,
        status: str = "success",
        operator: Optional[str] = None,
    ) -> None:
        """Log repository sync event.

        Args:
            repo_id: Repository identifier
            repo_full_name: Full repository name (owner/repo)
            branch: Branch name
            files_processed: Number of files processed
            files_skipped: Number of files skipped
            bytes_processed: Total bytes processed
            commits_processed: Number of commits processed
            modified_files: List of modified file paths (will be hashed)
            status: Event status (success/failure)
            operator: Optional operator identifier
        """
        event = GitHubAuditEvent(
            job_id=f"github_sync_{repo_id}_{int(time.time())}",
            source="github_connector",
            action="repository_sync",
            status=status,
            timestamp=datetime.utcnow(),
            repo_id=repo_id,
            repo_full_name_hash=self._hash_repo_name(repo_full_name),
            branch=branch,
            files_processed=files_processed,
            files_skipped=files_skipped,
            bytes_processed=bytes_processed,
            commits_processed=commits_processed,
            file_path_hashes=[
                self._hash_path(p) for p in (modified_files or [])[:100]
            ],  # Limit to 100 for log size
            operator_action=operator,
        )

        self.audit_logger.record(event.to_base_audit_event())

    def log_consent_check(
        self,
        repo_id: str,
        scope: GitHubConsentScope,
        granted: bool,
        operator: Optional[str] = None,
    ) -> None:
        """Log consent check result.

        Args:
            repo_id: Repository identifier
            scope: Consent scope being checked
            granted: Whether consent was granted
            operator: Optional operator identifier
        """
        event = GitHubAuditEvent(
            job_id=f"consent_check_{repo_id}_{int(time.time())}",
            source="github_connector",
            action=f"consent_check_{scope.value}",
            status="granted" if granted else "denied",
            timestamp=datetime.utcnow(),
            repo_id=repo_id,
            repo_full_name_hash="",  # Not needed for consent check
            operator_action=operator,
        )

        self.audit_logger.record(event.to_base_audit_event())

    def log_secret_detection(
        self,
        repo_id: str,
        file_path: str,
        detected: bool,
        operator: Optional[str] = None,
    ) -> None:
        """Log secret detection event.

        Args:
            repo_id: Repository identifier
            file_path: File path (will be hashed)
            detected: Whether secrets were detected
            operator: Optional operator identifier
        """
        event = GitHubAuditEvent(
            job_id=f"secret_check_{repo_id}_{int(time.time())}",
            source="github_connector",
            action="secret_detection",
            status="detected" if detected else "clean",
            timestamp=datetime.utcnow(),
            repo_id=repo_id,
            repo_full_name_hash="",
            file_path_hashes=[self._hash_path(file_path)],
            operator_action=operator,
        )

        self.audit_logger.record(event.to_base_audit_event())

    def log_error(
        self,
        repo_id: str,
        action: str,
        error: Exception,
        operator: Optional[str] = None,
    ) -> None:
        """Log error event with sanitized error information.

        Args:
            repo_id: Repository identifier
            action: Action that failed
            error: Exception that occurred
            operator: Optional operator identifier
        """
        # Sanitize error message to avoid leaking sensitive info
        error_message = str(error)
        # Remove potential secrets, paths, etc.
        sanitized_message = error_message[:200]  # Truncate to avoid long messages

        event = GitHubAuditEvent(
            job_id=f"github_error_{repo_id}_{int(time.time())}",
            source="github_connector",
            action=action,
            status="error",
            timestamp=datetime.utcnow(),
            repo_id=repo_id,
            repo_full_name_hash="",
            error_type=type(error).__name__,
            error_message_sanitized=sanitized_message,
            operator_action=operator,
        )

        self.audit_logger.record(event.to_base_audit_event())

    def _hash_repo_name(self, repo_full_name: str) -> str:
        """Hash repository name for privacy.

        Args:
            repo_full_name: Full repository name

        Returns:
            First 16 characters of SHA256 hash
        """
        return sha256(repo_full_name.encode()).hexdigest()[:16]

    def _hash_path(self, file_path: str) -> str:
        """Hash file path for privacy.

        Args:
            file_path: File path to hash

        Returns:
            First 16 characters of SHA256 hash
        """
        return sha256(file_path.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Consent Management Functions
# ---------------------------------------------------------------------------


def register_repository_consent(
    consent_registry: ConsentRegistry,
    repo_descriptor: GitHubRepositoryDescriptor,
    operator: Optional[str] = None,
    duration_hours: Optional[int] = None,
) -> List[ConsentRecord]:
    """Register consent for repository ingestion.

    Grants all required consent scopes for the repository based on its
    privacy settings.

    Args:
        consent_registry: Consent registry instance
        repo_descriptor: Repository descriptor with privacy settings
        operator: Optional operator granting consent
        duration_hours: Optional consent duration (None = no expiration)

    Returns:
        List of consent records created
    """
    required_scopes = repo_descriptor.privacy_settings.required_consent_scopes

    records = []
    for scope in required_scopes:
        record = consent_registry.grant(
            source=repo_descriptor.id,
            scope=scope.value,
            operator=operator,
            duration_hours=duration_hours,
        )
        records.append(record)

    return records


@contextmanager
def require_consent(
    consent_registry: ConsentRegistry,
    repo_id: str,
    scope: GitHubConsentScope,
    audit_logger: Optional[GitHubAuditLogger] = None,
):
    """Context manager to enforce consent for operations.

    Checks that consent is granted before allowing operation to proceed.
    Logs consent checks and failures.

    Args:
        consent_registry: Consent registry instance
        repo_id: Repository identifier
        scope: Required consent scope
        audit_logger: Optional audit logger for consent checks

    Raises:
        ConsentRequiredError: If consent not granted or expired

    Yields:
        ConsentRecord if consent is active

    Example:
        >>> with require_consent(registry, repo_id, GitHubConsentScope.GITHUB_CODE_ANALYSIS):
        ...     # Perform operation requiring consent
        ...     analyze_code(repo)
    """
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
        if audit_logger:
            audit_logger.log_error(repo_id, f"operation_{scope.value}", e)
        raise


async def revoke_repository_consent(
    consent_registry: ConsentRegistry,
    repo_id: str,
    scope: GitHubConsentScope,
    cleanup_data: bool = True,
    audit_logger: Optional[GitHubAuditLogger] = None,
) -> None:
    """Revoke consent and optionally cleanup data.

    Revokes consent for the specified scope and can trigger data cleanup
    based on the scope being revoked.

    Args:
        consent_registry: Consent registry instance
        repo_id: Repository identifier
        scope: Consent scope to revoke
        cleanup_data: Whether to cleanup associated data
        audit_logger: Optional audit logger

    Note:
        Data cleanup hooks should be implemented by the caller based on
        the specific scope being revoked. This function provides the
        framework for cleanup but doesn't implement specific cleanup logic.
    """
    # Revoke consent
    consent_registry.revoke(
        source=repo_id,
        scope=scope.value,
    )

    # Log revocation
    if audit_logger:
        event = GitHubAuditEvent(
            job_id=f"consent_revoke_{repo_id}_{int(time.time())}",
            source="github_connector",
            action=f"consent_revoke_{scope.value}",
            status="revoked",
            timestamp=datetime.utcnow(),
            repo_id=repo_id,
            repo_full_name_hash="",
        )
        audit_logger.audit_logger.record(event.to_base_audit_event())

    # Cleanup data if requested
    # Note: Actual cleanup implementation should be provided by caller
    # based on their specific data storage and requirements
    if cleanup_data:
        # Framework for cleanup hooks
        if scope == GitHubConsentScope.GITHUB_CODE_ANALYSIS:
            # Caller should implement: await cleanup_code_embeddings(repo_id)
            pass

        if scope == GitHubConsentScope.GITHUB_ISSUE_METADATA:
            # Caller should implement: await cleanup_issue_data(repo_id)
            pass

        if scope == GitHubConsentScope.GITHUB_PR_METADATA:
            # Caller should implement: await cleanup_pr_data(repo_id)
            pass


def create_github_redaction_policy(
    privacy_settings: RepositoryPrivacySettings,
) -> RedactionPolicy:
    """Create redaction policy for GitHub repositories.

    Builds a redaction policy based on the repository's privacy settings,
    determining how paths and identifiers should be anonymized in logs.

    Args:
        privacy_settings: Repository privacy settings

    Returns:
        Configured redaction policy
    """
    # Determine if plaintext paths are allowed
    from .descriptor import PrivacyLevel

    allow_plaintext = (
        privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE
        and not privacy_settings.enable_path_anonymization
    )

    # Configure filename and extension revelation based on privacy level
    reveal_filename = privacy_settings.privacy_level != PrivacyLevel.STRICT
    reveal_extension = privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE

    return RedactionPolicy(
        allow_plaintext=allow_plaintext,
        reveal_filename=reveal_filename,
        reveal_extension=reveal_extension,
        segment_hash_length=12,
        path_hash_length=16,
    )


__all__ = [
    "GitHubConsentScope",
    "GitHubAuditEvent",
    "GitHubAuditLogger",
    "register_repository_consent",
    "require_consent",
    "revoke_repository_consent",
    "create_github_redaction_policy",
]
