"""Descriptor and registry for GitHub repositories.

This module implements the GitHub Repository Descriptor as specified in
``docs/phase-1/github-connector-production-plan/01-repository-descriptor.md``.
It provides persistent metadata models, privacy-aware audit logging, and
deterministic identifiers while maintaining compatibility with the existing
Futurnal architecture.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import re
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from filelock import FileLock
from pydantic import BaseModel, Field, field_validator, model_validator

from futurnal import __version__ as FUTURNAL_VERSION
from ..local.config import LocalIngestionSource
from ...privacy.audit import AuditEvent
from ...privacy.redaction import RedactionPolicy, redact_path


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------


class SyncMode(str, Enum):
    """Synchronization modes for GitHub repositories."""

    GRAPHQL_API = "graphql_api"  # Lightweight, online-only
    GIT_CLONE = "git_clone"  # Full fidelity, offline-capable


class VisibilityType(str, Enum):
    """Repository visibility types."""

    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"  # GitHub Enterprise only


class PrivacyLevel(str, Enum):
    """Privacy levels for repository processing."""

    STRICT = "strict"  # Maximum privacy, minimal data exposure
    STANDARD = "standard"  # Balanced privacy with functionality
    PERMISSIVE = "permissive"  # Reduced privacy for enhanced features


class ConsentScope(str, Enum):
    """GitHub-specific consent scopes for repository operations."""

    GITHUB_REPO_ACCESS = "github:repo:access"
    GITHUB_CODE_ANALYSIS = "github:repo:code_analysis"
    GITHUB_ISSUE_METADATA = "github:repo:issue_metadata"
    GITHUB_PR_METADATA = "github:repo:pr_metadata"
    GITHUB_WIKI_ACCESS = "github:repo:wiki_access"
    GITHUB_CLOUD_MODELS = "github:repo:cloud_models"


# Default secret detection patterns from specification
DEFAULT_SECRET_PATTERNS = [
    r"(?i)(api[_-]?key|apikey)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})",
    r"(?i)(password|passwd|pwd)[\s]*[=:]+[\s]*['\"]?([^\s'\"]{8,})",
    r"(?i)(token)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})",
    r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
]

# Default file exclusion patterns
DEFAULT_EXCLUDE_PATHS = [
    ".git/",
    "node_modules/",
    "__pycache__/",
    "*.pyc",
    ".env*",
    "secrets.*",
    "credentials.*",
]

# Default extension exclusions
DEFAULT_EXCLUDE_EXTENSIONS = [
    ".exe",
    ".dll",
    ".so",
    ".dylib",  # Binaries
    ".jpg",
    ".png",
    ".gif",
    ".mp4",  # Media
    ".zip",
    ".tar",
    ".gz",  # Archives
]

# Default branch exclusions
DEFAULT_EXCLUDE_BRANCHES = ["gh-pages", "dependabot/*"]


# ---------------------------------------------------------------------------
# Privacy settings model
# ---------------------------------------------------------------------------


class RepositoryPrivacySettings(BaseModel):
    """Privacy configuration for a GitHub repository."""

    privacy_level: PrivacyLevel = Field(
        default=PrivacyLevel.STANDARD, description="Overall privacy posture"
    )

    # Consent scopes
    required_consent_scopes: List[ConsentScope] = Field(
        default_factory=lambda: [
            ConsentScope.GITHUB_REPO_ACCESS,
            ConsentScope.GITHUB_CODE_ANALYSIS,
        ],
        description="Consent scopes required for repository operations",
    )

    # Redaction
    enable_path_anonymization: bool = Field(
        default=True, description="Redact paths in logs and audit trails"
    )
    enable_author_anonymization: bool = Field(
        default=False, description="Redact commit author information"
    )
    redact_file_patterns: List[str] = Field(
        default_factory=lambda: [
            "*secret*",
            "*password*",
            "*token*",
            ".env*",
            "credentials.*",
        ],
        description="File patterns to completely exclude",
    )

    # Content filtering
    exclude_extensions: List[str] = Field(
        default_factory=lambda: DEFAULT_EXCLUDE_EXTENSIONS.copy(),
        description="File extensions to skip",
    )
    max_file_size_mb: int = Field(
        default=10, ge=1, le=100, description="Maximum file size in MB"
    )

    # Sensitive content detection
    detect_secrets: bool = Field(
        default=True, description="Enable secret detection patterns"
    )
    secret_patterns: List[str] = Field(
        default_factory=lambda: DEFAULT_SECRET_PATTERNS.copy(),
        description="Regular expressions for secret detection",
    )

    # Audit
    audit_sync_events: bool = Field(
        default=True, description="Audit sync activity events"
    )
    audit_content_changes: bool = Field(
        default=False, description="Audit content-level changes (checksum only)"
    )
    retain_audit_days: int = Field(
        default=90, ge=1, le=365, description="Audit log retention period"
    )


# ---------------------------------------------------------------------------
# Provenance model
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Provenance metadata for descriptor creation."""

    os_user: str
    machine_id_hash: str
    tool_version: str


# ---------------------------------------------------------------------------
# Main descriptor model
# ---------------------------------------------------------------------------


class GitHubRepositoryDescriptor(BaseModel):
    """Persistent descriptor for a GitHub repository."""

    # Identity
    id: str = Field(..., description="Deterministic repository identifier")
    name: Optional[str] = Field(
        default=None, description="Human-readable label"
    )
    icon: Optional[str] = Field(default=None, description="Optional emoji or icon")

    # Repository Identity
    owner: str = Field(..., description="Repository owner (user or organization)")
    repo: str = Field(..., description="Repository name")
    full_name: str = Field(..., description="owner/repo (computed)")
    visibility: VisibilityType = Field(
        ..., description="Repository visibility level"
    )

    # GitHub Instance
    github_host: str = Field(
        default="github.com", description="GitHub hostname"
    )
    api_base_url: Optional[str] = Field(
        default=None, description="Custom API base URL for GitHub Enterprise"
    )

    # Authentication
    credential_id: str = Field(
        ..., description="Reference to keychain credential"
    )

    # Sync Configuration
    sync_mode: SyncMode = Field(
        default=SyncMode.GRAPHQL_API, description="Synchronization mode"
    )

    # Branch Selection
    branches: List[str] = Field(
        default_factory=lambda: ["main", "master"],
        description="Branch whitelist for sync",
    )
    branch_patterns: List[str] = Field(
        default_factory=list, description="Glob patterns for branch selection"
    )
    exclude_branches: List[str] = Field(
        default_factory=lambda: DEFAULT_EXCLUDE_BRANCHES.copy(),
        description="Branches to skip",
    )

    # File Selection
    include_paths: List[str] = Field(
        default_factory=list, description="Path patterns to include (empty = all)"
    )
    exclude_paths: List[str] = Field(
        default_factory=lambda: DEFAULT_EXCLUDE_PATHS.copy(),
        description="Path patterns to exclude",
    )

    # Content Scope
    sync_issues: bool = Field(default=True, description="Sync issues")
    sync_pull_requests: bool = Field(default=True, description="Sync pull requests")
    sync_wiki: bool = Field(default=True, description="Sync wiki")
    sync_releases: bool = Field(default=True, description="Sync releases")

    # Temporal Scope
    sync_from_date: Optional[datetime] = Field(
        default=None, description="Only sync commits after this date"
    )
    max_commit_age_days: Optional[int] = Field(
        default=None, ge=1, description="Only sync recent commits"
    )

    # Git Clone Mode Settings (only used when sync_mode=GIT_CLONE)
    clone_depth: Optional[int] = Field(
        default=None, ge=1, description="Shallow clone depth (None = full history)"
    )
    sparse_checkout: bool = Field(
        default=False, description="Use sparse checkout for large repos"
    )
    local_clone_path: Optional[Path] = Field(
        default=None, description="Where to store cloned repo"
    )

    # Privacy & Consent
    privacy_settings: RepositoryPrivacySettings = Field(
        default_factory=RepositoryPrivacySettings,
        description="Privacy configuration",
    )

    # Provenance
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    provenance: Provenance = Field(..., description="Creation metadata")

    @field_validator("owner", "repo")
    @classmethod
    def _validate_name_component(cls, value: str) -> str:  # type: ignore[override]
        """Validate owner and repo name components."""
        if not value or not value.strip():
            raise ValueError("Owner and repo names must be non-empty")
        if " " in value:
            raise ValueError("Owner and repo names cannot contain spaces")
        # GitHub allows alphanumeric, hyphens, underscores, and dots
        if not re.match(r"^[a-zA-Z0-9._-]+$", value):
            raise ValueError(
                "Owner and repo names must contain only alphanumeric, dots, hyphens, and underscores"
            )
        return value

    @field_validator("branches", "exclude_branches")
    @classmethod
    def _validate_branches(cls, value: List[str]) -> List[str]:  # type: ignore[override]
        """Validate branch names are valid Git refs."""
        validated = []
        for branch in value:
            branch = branch.strip()
            if not branch:
                continue
            # Basic Git ref validation (simplified)
            if branch.startswith("/") or branch.endswith("/"):
                raise ValueError(f"Invalid branch name: {branch}")
            if ".." in branch or "@{" in branch:
                raise ValueError(f"Invalid branch name: {branch}")
            validated.append(branch)
        return validated

    @model_validator(mode="after")
    def _compute_full_name(self) -> "GitHubRepositoryDescriptor":
        """Compute full_name from owner and repo."""
        if not self.full_name or self.full_name != f"{self.owner}/{self.repo}":
            object.__setattr__(self, "full_name", f"{self.owner}/{self.repo}")
        return self

    @model_validator(mode="after")
    def _set_api_base_url(self) -> "GitHubRepositoryDescriptor":
        """Set default API base URL for GitHub.com."""
        if self.github_host == "github.com" and not self.api_base_url:
            object.__setattr__(self, "api_base_url", "https://api.github.com")
        return self

    @model_validator(mode="after")
    def _validate_clone_path(self) -> "GitHubRepositoryDescriptor":
        """Validate clone path if specified."""
        if self.local_clone_path and self.sync_mode != SyncMode.GIT_CLONE:
            raise ValueError(
                "local_clone_path can only be set when sync_mode=GIT_CLONE"
            )
        return self

    @classmethod
    def from_registration(
        cls,
        *,
        owner: str,
        repo: str,
        github_host: str = "github.com",
        api_base_url: Optional[str] = None,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        credential_id: str,
        visibility: VisibilityType = VisibilityType.PUBLIC,
        sync_mode: SyncMode = SyncMode.GRAPHQL_API,
        branches: Optional[Iterable[str]] = None,
        branch_patterns: Optional[Iterable[str]] = None,
        exclude_branches: Optional[Iterable[str]] = None,
        include_paths: Optional[Iterable[str]] = None,
        exclude_paths: Optional[Iterable[str]] = None,
        sync_issues: bool = True,
        sync_pull_requests: bool = True,
        sync_wiki: bool = True,
        sync_releases: bool = True,
        sync_from_date: Optional[datetime] = None,
        max_commit_age_days: Optional[int] = None,
        clone_depth: Optional[int] = None,
        sparse_checkout: bool = False,
        local_clone_path: Optional[Path] = None,
        privacy_settings: Optional[RepositoryPrivacySettings] = None,
    ) -> "GitHubRepositoryDescriptor":
        """Create descriptor from registration parameters."""
        repo_id = _deterministic_repository_id(owner, repo, github_host)
        full_name = f"{owner}/{repo}"

        branches_list = list(branches) if branches else ["main", "master"]
        exclude_branches_list = (
            list(exclude_branches) if exclude_branches else DEFAULT_EXCLUDE_BRANCHES.copy()
        )
        exclude_paths_list = (
            list(exclude_paths) if exclude_paths else DEFAULT_EXCLUDE_PATHS.copy()
        )

        return cls(
            id=repo_id,
            name=name,
            icon=icon,
            owner=owner,
            repo=repo,
            full_name=full_name,
            visibility=visibility,
            github_host=github_host,
            api_base_url=api_base_url,
            credential_id=credential_id,
            sync_mode=sync_mode,
            branches=branches_list,
            branch_patterns=list(branch_patterns or []),
            exclude_branches=exclude_branches_list,
            include_paths=list(include_paths or []),
            exclude_paths=exclude_paths_list,
            sync_issues=sync_issues,
            sync_pull_requests=sync_pull_requests,
            sync_wiki=sync_wiki,
            sync_releases=sync_releases,
            sync_from_date=sync_from_date,
            max_commit_age_days=max_commit_age_days,
            clone_depth=clone_depth,
            sparse_checkout=sparse_checkout,
            local_clone_path=local_clone_path,
            privacy_settings=privacy_settings or RepositoryPrivacySettings(),
            provenance=Provenance(
                os_user=getpass.getuser(),
                machine_id_hash=_machine_id_hash(),
                tool_version=FUTURNAL_VERSION,
            ),
        )

    def update(self, **changes: Any) -> "GitHubRepositoryDescriptor":
        """Create updated descriptor with changes."""
        payload = self.model_dump()
        payload.update(changes)
        payload["updated_at"] = datetime.utcnow()
        return GitHubRepositoryDescriptor.model_validate(payload)

    def to_local_source(
        self,
        *,
        max_workers: Optional[int] = None,
        max_files_per_batch: Optional[int] = None,
        scan_interval_seconds: Optional[float] = None,
        watcher_debounce_seconds: Optional[float] = None,
        schedule: str = "@manual",
        priority: str = "normal",
    ) -> LocalIngestionSource:
        """Convert to LocalIngestionSource for orchestrator integration."""
        source_name = self.name or f"github-{self.id[:8]}"

        # Determine privacy settings
        privacy_settings = self.privacy_settings
        allow_plaintext = (
            privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE
            and not privacy_settings.enable_path_anonymization
        )
        require_consent = len(privacy_settings.required_consent_scopes) > 1

        # Build external processing scope
        external_scopes = [
            scope.value
            for scope in privacy_settings.required_consent_scopes
            if scope != ConsentScope.GITHUB_REPO_ACCESS
        ]
        external_scope = (
            ",".join(external_scopes)
            if external_scopes
            else "github.external_processing"
        )

        # Determine root path based on sync mode
        if self.sync_mode == SyncMode.GIT_CLONE and self.local_clone_path:
            root_path = self.local_clone_path
        else:
            # For GraphQL API mode, use workspace directory
            root_path = (
                Path.home() / ".futurnal" / "workspace" / "github" / self.id
            )
            root_path.mkdir(parents=True, exist_ok=True)

        return LocalIngestionSource(
            name=source_name,
            root_path=root_path,
            include=self.include_paths,
            exclude=self.exclude_paths,
            follow_symlinks=False,
            ignore_file=None,
            max_workers=max_workers,
            max_files_per_batch=max_files_per_batch,
            scan_interval_seconds=scan_interval_seconds,
            watcher_debounce_seconds=watcher_debounce_seconds,
            allow_plaintext_paths=allow_plaintext,
            require_external_processing_consent=require_consent,
            external_processing_scope=external_scope,
            schedule=schedule,
            priority=priority,
            paused=False,
        )

    def build_redaction_policy(
        self, *, allow_plaintext: Optional[bool] = None
    ) -> RedactionPolicy:
        """Build redaction policy respecting privacy settings."""
        privacy_settings = self.privacy_settings
        if allow_plaintext is None:
            allow_plaintext = (
                privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE
                and not privacy_settings.enable_path_anonymization
            )

        return RedactionPolicy(
            allow_plaintext=allow_plaintext,
            reveal_filename=privacy_settings.privacy_level != PrivacyLevel.STRICT,
            reveal_extension=privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE,
        )

    def get_required_consent_scopes(self) -> List[str]:
        """Get list of consent scope strings required for this repository."""
        return [
            scope.value for scope in self.privacy_settings.required_consent_scopes
        ]

    def requires_consent_for_scope(self, scope: ConsentScope) -> bool:
        """Check if specific consent scope is required."""
        return scope in self.privacy_settings.required_consent_scopes

    def get_audit_retention_days(self) -> int:
        """Get audit log retention period."""
        return self.privacy_settings.retain_audit_days

    def compile_secret_patterns(self) -> List[re.Pattern]:
        """Compile secret detection patterns for performance."""
        return [
            re.compile(pattern)
            for pattern in self.privacy_settings.secret_patterns
        ]


# ---------------------------------------------------------------------------
# Registry implementation
# ---------------------------------------------------------------------------


@dataclass
class RepositoryRegistry:
    """File-based registry for GitHub repository descriptors."""

    registry_root: Path
    audit_logger: Optional[Any] = None

    def __init__(
        self,
        registry_root: Optional[Path] = None,
        audit_logger: Optional[Any] = None,
    ) -> None:
        default_root = Path.home() / ".futurnal" / "sources" / "github"
        self.registry_root = (registry_root or default_root).expanduser()
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.audit_logger = audit_logger

    def _descriptor_path(self, repo_id: str) -> Path:
        return self.registry_root / f"{repo_id}.json"

    def _lock_path(self, repo_id: str) -> Path:
        return self.registry_root / f"{repo_id}.json.lock"

    def register(
        self,
        *,
        owner: str,
        repo: str,
        github_host: str = "github.com",
        api_base_url: Optional[str] = None,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        credential_id: str,
        visibility: VisibilityType = VisibilityType.PUBLIC,
        sync_mode: SyncMode = SyncMode.GRAPHQL_API,
        branches: Optional[Iterable[str]] = None,
        branch_patterns: Optional[Iterable[str]] = None,
        exclude_branches: Optional[Iterable[str]] = None,
        include_paths: Optional[Iterable[str]] = None,
        exclude_paths: Optional[Iterable[str]] = None,
        sync_issues: bool = True,
        sync_pull_requests: bool = True,
        sync_wiki: bool = True,
        sync_releases: bool = True,
        sync_from_date: Optional[datetime] = None,
        max_commit_age_days: Optional[int] = None,
        clone_depth: Optional[int] = None,
        sparse_checkout: bool = False,
        local_clone_path: Optional[Path] = None,
        privacy_settings: Optional[RepositoryPrivacySettings] = None,
        operator: Optional[str] = None,
    ) -> GitHubRepositoryDescriptor:
        """Register a new repository or update existing."""
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner=owner,
            repo=repo,
            github_host=github_host,
            api_base_url=api_base_url,
            name=name,
            icon=icon,
            credential_id=credential_id,
            visibility=visibility,
            sync_mode=sync_mode,
            branches=branches,
            branch_patterns=branch_patterns,
            exclude_branches=exclude_branches,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            sync_issues=sync_issues,
            sync_pull_requests=sync_pull_requests,
            sync_wiki=sync_wiki,
            sync_releases=sync_releases,
            sync_from_date=sync_from_date,
            max_commit_age_days=max_commit_age_days,
            clone_depth=clone_depth,
            sparse_checkout=sparse_checkout,
            local_clone_path=local_clone_path,
            privacy_settings=privacy_settings,
        )
        return self.add_or_update(descriptor, operator=operator)

    def add_or_update(
        self,
        descriptor: GitHubRepositoryDescriptor,
        *,
        operator: Optional[str] = None,
    ) -> GitHubRepositoryDescriptor:
        """Add or update descriptor with idempotent behavior."""
        path = self._descriptor_path(descriptor.id)
        lock = FileLock(str(self._lock_path(descriptor.id)))

        with lock:
            now = datetime.utcnow()
            is_update = path.exists()
            previous_settings = None

            if is_update:
                try:
                    existing = self.get(descriptor.id)
                    previous_settings = existing.privacy_settings

                    # Preserve created_at and provenance; update mutable fields
                    updated = existing.model_copy(
                        update={
                            "name": descriptor.name or existing.name,
                            "icon": descriptor.icon or existing.icon,
                            "visibility": descriptor.visibility,
                            "credential_id": descriptor.credential_id,
                            "sync_mode": descriptor.sync_mode,
                            "branches": descriptor.branches or existing.branches,
                            "branch_patterns": descriptor.branch_patterns
                            or existing.branch_patterns,
                            "exclude_branches": descriptor.exclude_branches
                            or existing.exclude_branches,
                            "include_paths": descriptor.include_paths
                            or existing.include_paths,
                            "exclude_paths": descriptor.exclude_paths
                            or existing.exclude_paths,
                            "sync_issues": descriptor.sync_issues,
                            "sync_pull_requests": descriptor.sync_pull_requests,
                            "sync_wiki": descriptor.sync_wiki,
                            "sync_releases": descriptor.sync_releases,
                            "sync_from_date": descriptor.sync_from_date,
                            "max_commit_age_days": descriptor.max_commit_age_days,
                            "clone_depth": descriptor.clone_depth,
                            "sparse_checkout": descriptor.sparse_checkout,
                            "local_clone_path": descriptor.local_clone_path,
                            "privacy_settings": descriptor.privacy_settings
                            or existing.privacy_settings,
                            "updated_at": now,
                        }
                    )
                    self._write(path, updated)

                    # Log update event
                    self._log_repo_event("updated", "success", updated, operator=operator)

                    # Log privacy changes if any
                    self._log_privacy_change(updated, previous_settings, operator=operator)

                    return updated

                except Exception as e:
                    # If corrupt, overwrite with fresh descriptor
                    descriptor = descriptor.update(created_at=now, updated_at=now)
                    self._log_repo_event(
                        "update_failed",
                        "error",
                        descriptor,
                        metadata={"error": str(e)},
                        operator=operator,
                    )

            # New repository registration
            descriptor = descriptor.update(created_at=now, updated_at=now)
            self._write(path, descriptor)

            # Log registration event
            if not is_update:
                self._log_repo_event("registered", "success", descriptor, operator=operator)

            return descriptor

    def get(self, repo_id: str) -> GitHubRepositoryDescriptor:
        """Get descriptor by repository ID."""
        path = self._descriptor_path(repo_id)
        if not path.exists():
            raise FileNotFoundError(f"Repository {repo_id} not found")
        data = json.loads(path.read_text())
        return GitHubRepositoryDescriptor.model_validate(data)

    def list(self) -> List[GitHubRepositoryDescriptor]:
        """List all registered repositories."""
        items: List[GitHubRepositoryDescriptor] = []
        for file in sorted(self.registry_root.glob("*.json")):
            try:
                data = json.loads(file.read_text())
                items.append(GitHubRepositoryDescriptor.model_validate(data))
            except Exception:
                # Skip malformed entries
                continue
        return items

    def find_by_repository(
        self, owner: str, repo: str, github_host: str = "github.com"
    ) -> Optional[GitHubRepositoryDescriptor]:
        """Find descriptor by owner/repo/host."""
        repo_id = _deterministic_repository_id(owner, repo, github_host)
        try:
            return self.get(repo_id)
        except FileNotFoundError:
            return None

    def remove(
        self, repo_id: str, *, operator: Optional[str] = None
    ) -> None:
        """Remove repository from registry."""
        lock = FileLock(str(self._lock_path(repo_id)))
        with lock:
            path = self._descriptor_path(repo_id)
            if not path.exists():
                raise FileNotFoundError(f"Repository {repo_id} not found")

            # Get descriptor for audit logging before removal
            try:
                descriptor = self.get(repo_id)
                path.unlink()

                # Log removal event
                self._log_repo_event("removed", "success", descriptor, operator=operator)

            except Exception as e:
                # Log failed removal
                try:
                    descriptor = self.get(repo_id)
                    self._log_repo_event(
                        "remove_failed",
                        "error",
                        descriptor,
                        metadata={"error": str(e)},
                        operator=operator,
                    )
                except:
                    # Can't get descriptor, log basic error
                    if self.audit_logger:
                        event = AuditEvent(
                            job_id=f"github_registry_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                            source="github_repository_registry",
                            action="repo_remove_failed",
                            status="error",
                            timestamp=datetime.utcnow(),
                            metadata={"repo_id": repo_id, "error": str(e)},
                            operator_action=operator,
                        )
                        self.audit_logger.record(event)
                raise

    def _write(self, path: Path, descriptor: GitHubRepositoryDescriptor) -> None:
        """Write descriptor to file atomically."""
        payload = json.dumps(descriptor.model_dump(mode="json"), indent=2)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)

    def _log_repo_event(
        self,
        action: str,
        status: str,
        descriptor: GitHubRepositoryDescriptor,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        operator: Optional[str] = None,
    ) -> None:
        """Log repository lifecycle events."""
        if self.audit_logger is None:
            return

        try:
            # Build redaction policy
            policy = descriptor.build_redaction_policy(allow_plaintext=False)

            event_metadata = {
                "repo_id": descriptor.id,
                "owner": descriptor.owner,
                "repo": descriptor.repo,
                "full_name": descriptor.full_name,
                "github_host": descriptor.github_host,
                "visibility": descriptor.visibility.value,
                "sync_mode": descriptor.sync_mode.value,
                "privacy_level": descriptor.privacy_settings.privacy_level.value,
                "required_consent_scopes": [
                    scope.value
                    for scope in descriptor.privacy_settings.required_consent_scopes
                ],
                "created_at": descriptor.created_at.isoformat(),
                "updated_at": descriptor.updated_at.isoformat(),
                "tool_version": descriptor.provenance.tool_version,
            }

            if metadata:
                event_metadata.update(metadata)

            # Redact repository identifier for privacy
            redacted_identifier = policy.apply(descriptor.full_name)

            event = AuditEvent(
                job_id=f"github_registry_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                source="github_repository_registry",
                action=f"repo_{action}",
                status=status,
                timestamp=datetime.utcnow(),
                redacted_path=redacted_identifier.redacted,
                path_hash=redacted_identifier.path_hash,
                operator_action=operator,
                metadata=event_metadata,
            )

            self.audit_logger.record(event)

        except Exception as e:
            # Don't fail registry operations due to audit logging issues
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to log repository audit event: {e}")

    def _log_privacy_change(
        self,
        descriptor: GitHubRepositoryDescriptor,
        previous_settings: Optional[RepositoryPrivacySettings],
        *,
        operator: Optional[str] = None,
    ) -> None:
        """Log privacy settings changes."""
        if self.audit_logger is None or previous_settings is None:
            return

        current_settings = descriptor.privacy_settings

        # Detect changes
        changes = {}
        if previous_settings.privacy_level != current_settings.privacy_level:
            changes["privacy_level"] = {
                "from": previous_settings.privacy_level.value,
                "to": current_settings.privacy_level.value,
            }

        if set(previous_settings.required_consent_scopes) != set(
            current_settings.required_consent_scopes
        ):
            changes["consent_scopes"] = {
                "from": [
                    scope.value for scope in previous_settings.required_consent_scopes
                ],
                "to": [
                    scope.value for scope in current_settings.required_consent_scopes
                ],
            }

        if (
            previous_settings.enable_path_anonymization
            != current_settings.enable_path_anonymization
        ):
            changes["path_anonymization"] = {
                "from": previous_settings.enable_path_anonymization,
                "to": current_settings.enable_path_anonymization,
            }

        if (
            previous_settings.enable_author_anonymization
            != current_settings.enable_author_anonymization
        ):
            changes["author_anonymization"] = {
                "from": previous_settings.enable_author_anonymization,
                "to": current_settings.enable_author_anonymization,
            }

        if changes:
            self._log_repo_event(
                "privacy_updated",
                "success",
                descriptor,
                metadata={"privacy_changes": changes},
                operator=operator,
            )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _deterministic_repository_id(
    owner: str, repo: str, host: str = "github.com"
) -> str:
    """Generate deterministic repository ID from owner/repo/host."""
    normalized = f"{owner.lower()}/{repo.lower()}@{host.lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"github:{normalized}"))


def _machine_id_hash() -> str:
    """Generate machine fingerprint hash."""
    try:
        node = uuid.getnode()
        host = socket.gethostname()
        payload = f"{node}:{host}:{platform.system()}:{platform.machine()}".encode()
        return sha256(payload).hexdigest()
    except Exception:
        return "unknown"


def create_credential_id(repo_id: str) -> str:
    """Create credential ID for keychain storage."""
    return f"github_cred_{repo_id}"


def generate_repository_id(
    owner: str, repo: str, github_host: str = "github.com"
) -> str:
    """Generate deterministic repository ID."""
    return _deterministic_repository_id(owner, repo, github_host)


__all__ = [
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
]
