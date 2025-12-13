"""GitHub-specific audit event types and logging helpers.

This module defines GitHub-specific audit event types and provides helper
functions for logging common GitHub operations with privacy-compliant
redaction.

Event Types:
- Repository events (registered, unregistered, synced)
- Authentication events (token validated, token expired)
- API events (rate limited, quota used)
- Sync events (started, completed, failed)
- Content processing (file fetched, commit processed)
- Privacy events (consent granted/revoked)

Integration:
- Uses existing AuditLogger infrastructure
- Applies path redaction for privacy
- Ensures no code content in audit logs
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...privacy.audit import AuditLogger, AuditEvent


class GitHubAuditEvents:
    """Audit event types for GitHub operations.

    Constants for standardized GitHub audit event types, ensuring
    consistent event naming across the GitHub connector.
    """

    # Repository events
    REPO_REGISTERED = "github_repo_registered"
    REPO_UNREGISTERED = "github_repo_unregistered"
    REPO_SYNCED = "github_repo_synced"
    REPO_SYNC_FAILED = "github_repo_sync_failed"

    # Authentication events
    TOKEN_VALIDATED = "github_token_validated"
    TOKEN_EXPIRED = "github_token_expired"
    TOKEN_REFRESHED = "github_token_refreshed"
    AUTH_FAILED = "github_auth_failed"

    # API events
    API_REQUEST = "github_api_request"
    API_RATE_LIMITED = "github_api_rate_limited"
    API_QUOTA_WARNING = "github_api_quota_warning"

    # Sync events
    SYNC_STARTED = "github_sync_started"
    SYNC_COMPLETED = "github_sync_completed"
    SYNC_FAILED = "github_sync_failed"
    SYNC_INCREMENTAL = "github_sync_incremental"

    # Content processing
    FILE_FETCHED = "github_file_fetched"
    COMMIT_PROCESSED = "github_commit_processed"
    BRANCH_SCANNED = "github_branch_scanned"
    PR_PROCESSED = "github_pr_processed"
    ISSUE_PROCESSED = "github_issue_processed"

    # Privacy events
    CONSENT_GRANTED = "github_consent_granted"
    CONSENT_REVOKED = "github_consent_revoked"
    CONSENT_CHECK_FAILED = "github_consent_check_failed"


def _redact_repo_path(owner: str, repo: str, file_path: Optional[str] = None) -> str:
    """Create redacted repository path.

    Args:
        owner: Repository owner
        repo: Repository name
        file_path: Optional file path within repo

    Returns:
        Redacted path string (owner/repo or owner/repo/path)
    """
    base = f"{owner}/{repo}"
    if file_path:
        # Only include first 2 path components for privacy
        parts = file_path.split("/")
        if len(parts) > 2:
            return f"{base}/{parts[0]}/{parts[1]}/..."
        return f"{base}/{file_path}"
    return base


def log_repo_event(
    audit_logger: "AuditLogger",
    *,
    repo_id: str,
    owner: str,
    repo: str,
    action: str,
    status: str = "success",
    branch: Optional[str] = None,
    file_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Log repository registration/sync event.

    Args:
        audit_logger: Audit logger instance
        repo_id: Repository identifier
        owner: Repository owner
        repo: Repository name
        action: Event action (registered, unregistered, synced)
        status: Event status (success, failed)
        branch: Optional branch name
        file_count: Optional count of files in repo
        error: Optional error message
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "repo_id": repo_id,
        "repo_path": _redact_repo_path(owner, repo),
    }

    if branch:
        metadata["branch"] = branch

    if file_count is not None:
        metadata["file_count"] = file_count

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"github_repo_{repo_id}_{int(datetime.utcnow().timestamp())}",
            source="github_connector",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_auth_event(
    audit_logger: "AuditLogger",
    *,
    action: str,
    status: str = "success",
    scopes: Optional[List[str]] = None,
    expires_in_hours: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Log authentication event.

    Args:
        audit_logger: Audit logger instance
        action: Auth action (validated, expired, refreshed, failed)
        status: Event status
        scopes: OAuth scopes (if available)
        expires_in_hours: Token expiration time
        error: Optional error message

    Privacy Guarantee:
        - Token value NEVER logged
        - Only token metadata logged
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {}

    if scopes:
        metadata["scopes"] = scopes

    if expires_in_hours is not None:
        metadata["expires_in_hours"] = expires_in_hours

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"github_auth_{int(datetime.utcnow().timestamp())}",
            source="github_auth_manager",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_api_event(
    audit_logger: "AuditLogger",
    *,
    endpoint: str,
    method: str = "GET",
    status_code: Optional[int] = None,
    rate_limit_remaining: Optional[int] = None,
    rate_limit_reset: Optional[datetime] = None,
    response_time_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log GitHub API request event.

    Args:
        audit_logger: Audit logger instance
        endpoint: API endpoint path
        method: HTTP method
        status_code: Response status code
        rate_limit_remaining: Remaining API calls
        rate_limit_reset: Rate limit reset time
        response_time_ms: Response time in milliseconds
        error: Optional error message
    """
    from ...privacy.audit import AuditEvent

    # Determine if this is a rate limit event
    is_rate_limited = status_code == 429 or (
        rate_limit_remaining is not None and rate_limit_remaining < 10
    )

    action = (
        GitHubAuditEvents.API_RATE_LIMITED
        if is_rate_limited
        else GitHubAuditEvents.API_REQUEST
    )

    status = "success"
    if status_code and status_code >= 400:
        status = "failed"
    elif is_rate_limited:
        status = "rate_limited"

    metadata: Dict[str, object] = {
        "endpoint": endpoint,
        "method": method,
    }

    if status_code:
        metadata["status_code"] = status_code

    if rate_limit_remaining is not None:
        metadata["rate_limit_remaining"] = rate_limit_remaining

    if rate_limit_reset:
        metadata["rate_limit_reset"] = rate_limit_reset.isoformat()

    if response_time_ms is not None:
        metadata["response_time_ms"] = round(response_time_ms, 2)

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"github_api_{int(datetime.utcnow().timestamp())}",
            source="github_api_client",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_sync_event(
    audit_logger: "AuditLogger",
    *,
    repo_id: str,
    owner: str,
    repo: str,
    action: str,
    status: str = "success",
    commits_processed: int = 0,
    files_processed: int = 0,
    prs_processed: int = 0,
    issues_processed: int = 0,
    duration_seconds: Optional[float] = None,
    is_incremental: bool = False,
    since_commit: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Log sync event with statistics.

    Args:
        audit_logger: Audit logger instance
        repo_id: Repository identifier
        owner: Repository owner
        repo: Repository name
        action: Sync action (started, completed, failed)
        status: Event status
        commits_processed: Count of processed commits
        files_processed: Count of processed files
        prs_processed: Count of processed pull requests
        issues_processed: Count of processed issues
        duration_seconds: Sync duration in seconds
        is_incremental: Whether this was an incremental sync
        since_commit: Starting commit for incremental sync
        error: Optional error message
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "repo_id": repo_id,
        "repo_path": _redact_repo_path(owner, repo),
        "commits_processed": commits_processed,
        "files_processed": files_processed,
        "prs_processed": prs_processed,
        "issues_processed": issues_processed,
        "is_incremental": is_incremental,
    }

    if duration_seconds is not None:
        metadata["duration_seconds"] = round(duration_seconds, 2)

    if since_commit:
        # Only log first 7 chars of commit SHA
        metadata["since_commit"] = since_commit[:7]

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"github_sync_{repo_id}_{int(datetime.utcnow().timestamp())}",
            source="github_sync_engine",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_content_event(
    audit_logger: "AuditLogger",
    *,
    repo_id: str,
    owner: str,
    repo: str,
    content_type: str,
    content_id: str,
    action: str,
    status: str = "success",
    file_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Log content processing event.

    Args:
        audit_logger: Audit logger instance
        repo_id: Repository identifier
        owner: Repository owner
        repo: Repository name
        content_type: Type of content (file, commit, pr, issue)
        content_id: Content identifier (SHA, PR number, etc.)
        action: Processing action
        status: Event status
        file_path: Optional file path
        file_size_bytes: Optional file size
        error: Optional error message

    Privacy Guarantee:
        - File content NEVER logged
        - Paths redacted for privacy
        - Only metadata logged
    """
    from ...privacy.audit import AuditEvent

    metadata: Dict[str, object] = {
        "repo_id": repo_id,
        "repo_path": _redact_repo_path(owner, repo, file_path),
        "content_type": content_type,
        "content_id": content_id[:7] if len(content_id) > 7 else content_id,
    }

    if file_size_bytes is not None:
        metadata["file_size_bytes"] = file_size_bytes

    if error:
        metadata["error"] = error

    audit_logger.record(
        AuditEvent(
            job_id=f"github_content_{repo_id}_{content_id[:7] if len(content_id) > 7 else content_id}",
            source="github_content_processor",
            action=action,
            status=status,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
    )


def log_consent_check_failed(
    audit_logger: "AuditLogger",
    *,
    repo_id: str,
    scope: str,
    operation: str,
) -> None:
    """Log consent check failure.

    Args:
        audit_logger: Audit logger instance
        repo_id: Repository identifier
        scope: Required consent scope
        operation: Operation that was blocked
    """
    from ...privacy.audit import AuditEvent

    audit_logger.record(
        AuditEvent(
            job_id=f"github_consent_fail_{repo_id}_{int(datetime.utcnow().timestamp())}",
            source="github_consent_manager",
            action=GitHubAuditEvents.CONSENT_CHECK_FAILED,
            status="blocked",
            timestamp=datetime.utcnow(),
            metadata={
                "repo_id": repo_id,
                "scope": scope,
                "operation": operation,
            },
        )
    )


__all__ = [
    "GitHubAuditEvents",
    "log_repo_event",
    "log_auth_event",
    "log_api_event",
    "log_sync_event",
    "log_content_event",
    "log_consent_check_failed",
]
