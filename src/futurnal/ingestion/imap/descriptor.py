"""Descriptor and registry for IMAP mailboxes.

This module extends the descriptor/registry pattern established for Obsidian
vaults to the IMAP connector. It provides a persistent metadata model,
privacy-aware audit logging, and deterministic identifiers while respecting
the IMAP production plan in ``docs/phase-1/imap-connector-production-plan``.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from filelock import FileLock
from pydantic import BaseModel, Field, field_validator, model_validator

from futurnal import __version__ as FUTURNAL_VERSION
from ..local.config import LocalIngestionSource
from ...privacy.audit import AuditEvent
from ...privacy.redaction import RedactionPolicy, redact_path

if TYPE_CHECKING:
    from ...privacy.redaction import RedactedPath


# ---------------------------------------------------------------------------
# Enums and privacy settings
# ---------------------------------------------------------------------------


class PrivacyLevel(str, Enum):
    """Privacy levels for mailbox processing."""

    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"


class ConsentScope(str, Enum):
    """IMAP-specific consent scopes derived from production plan."""

    MAILBOX_ACCESS = "imap:mailbox:access"
    EMAIL_CONTENT_ANALYSIS = "imap:email:content_analysis"
    ATTACHMENT_EXTRACTION = "imap:email:attachment_extraction"
    THREAD_RECONSTRUCTION = "imap:email:thread_reconstruction"
    PARTICIPANT_ANALYSIS = "imap:email:participant_analysis"
    CLOUD_MODELS = "imap:email:cloud_models"


class AuthMode(str, Enum):
    """Supported authentication methods."""

    OAUTH2 = "oauth2"
    APP_PASSWORD = "app_password"


class MailboxPrivacySettings(BaseModel):
    """Privacy configuration for IMAP mailboxes."""

    privacy_level: PrivacyLevel = Field(
        default=PrivacyLevel.STANDARD,
        description="Overall privacy posture for this mailbox",
    )
    required_consent_scopes: List[ConsentScope] = Field(
        default_factory=lambda: [
            ConsentScope.MAILBOX_ACCESS,
            ConsentScope.EMAIL_CONTENT_ANALYSIS,
        ],
        description="Consent scopes required for operations",
    )
    enable_sender_anonymization: bool = Field(
        default=True,
        description="Redact sender addresses in audit outputs",
    )
    enable_recipient_anonymization: bool = Field(
        default=True,
        description="Redact recipient addresses in audit outputs",
    )
    enable_subject_redaction: bool = Field(
        default=False,
        description="Redact subject lines in audit outputs",
    )
    redact_email_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns of emails to exclude from ingestion",
    )
    exclude_email_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns of messages to skip during sync",
    )
    privacy_subject_keywords: List[str] = Field(
        default_factory=lambda: ["private", "confidential", "nda"],
        description="Keywords triggering subject redaction",
    )
    audit_sync_events: bool = Field(
        default=True,
        description="Emit audit events for sync activity",
    )
    audit_content_changes: bool = Field(
        default=False,
        description="Audit content-level changes (checksum only)",
    )
    retain_audit_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Retention period for audit logs",
    )


# ---------------------------------------------------------------------------
# Descriptor model
# ---------------------------------------------------------------------------


class ImapMailboxDescriptor(BaseModel):
    """Persistent descriptor for an IMAP mailbox."""

    id: str = Field(..., description="Deterministic mailbox identifier")
    name: Optional[str] = Field(default=None, description="Human-readable label")
    icon: Optional[str] = Field(default=None, description="Optional emoji/icon")
    imap_host: str = Field(..., description="IMAP hostname")
    imap_port: int = Field(
        default=993,
        ge=1,
        le=65535,
        description="IMAP port (TLS enforced)",
    )
    email_address: str = Field(..., description="Mailbox email address")
    auth_mode: AuthMode = Field(..., description="Authentication mode")
    credential_id: str = Field(..., description="Keychain credential identifier")
    folders: List[str] = Field(
        default_factory=lambda: ["INBOX"],
        description="Folder whitelist",
    )
    folder_patterns: List[str] = Field(
        default_factory=list,
        description="Glob-style patterns for folder selection",
    )
    exclude_folders: List[str] = Field(
        default_factory=lambda: ["[Gmail]/Trash", "[Gmail]/Spam"],
        description="Folders to skip",
    )
    sync_from_date: Optional[datetime] = Field(
        default=None,
        description="Only ingest messages after this timestamp",
    )
    max_message_age_days: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional rolling window limit",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Provider hint (gmail, office365, generic)",
    )
    privacy_settings: MailboxPrivacySettings = Field(
        default_factory=MailboxPrivacySettings,
        description="Privacy settings",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_user: str = Field(..., description="OS user who registered")
    provenance_machine_hash: str = Field(
        ..., description="Machine fingerprint for audit"
    )
    provenance_tool_version: str = Field(
        ..., description="Futurnal version at registration"
    )
    network_warning: Optional[str] = Field(
        default=None,
        description="Optional warning about provider requirements",
    )

    @field_validator("imap_host")
    @classmethod
    def _validate_host(cls, value: str) -> str:  # type: ignore[override]
        if not value or " " in value:
            raise ValueError("imap_host must be a valid hostname")
        return value

    @field_validator("imap_port")
    @classmethod
    def _validate_port(cls, value: int) -> int:  # type: ignore[override]
        if value == 143:
            raise ValueError("Plain IMAP (port 143) is unsupported; use 993")
        if not (1 <= value <= 65535):
            raise ValueError("imap_port must be between 1 and 65535")
        return value

    @field_validator("email_address")
    @classmethod
    def _validate_email(cls, value: str) -> str:  # type: ignore[override]
        if "@" not in value or value.count("@") != 1:
            raise ValueError("Invalid email address")
        return value

    @field_validator("folders", "exclude_folders")
    @classmethod
    def _normalize_folders(
        cls, value: List[str]
    ) -> List[str]:  # type: ignore[override]
        return sorted({folder.strip() for folder in value if folder.strip()})

    @model_validator(mode="after")
    def _validate_folder_overlap(self) -> "ImapMailboxDescriptor":
        include_set = set(self.folders)
        exclude_set = set(self.exclude_folders)
        overlap = include_set.intersection(exclude_set)
        if overlap:
            raise ValueError(f"Folders cannot be both included and excluded: {overlap}")
        return self

    @classmethod
    def from_registration(
        cls,
        *,
        email_address: str,
        imap_host: str,
        imap_port: int = 993,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        auth_mode: AuthMode,
        credential_id: str,
        folders: Optional[Iterable[str]] = None,
        folder_patterns: Optional[Iterable[str]] = None,
        exclude_folders: Optional[Iterable[str]] = None,
        sync_from_date: Optional[datetime] = None,
        max_message_age_days: Optional[int] = None,
        provider: Optional[str] = None,
        privacy_settings: Optional[MailboxPrivacySettings] = None,
    ) -> "ImapMailboxDescriptor":
        folders_list = list(folders or ["INBOX"])
        exclude_list = list(exclude_folders or ["[Gmail]/Trash", "[Gmail]/Spam"])
        return cls(
            id=_deterministic_mailbox_id(email_address, imap_host),
            name=name,
            icon=icon,
            imap_host=imap_host,
            imap_port=imap_port,
            email_address=email_address,
            auth_mode=auth_mode,
            credential_id=credential_id,
            folders=folders_list,
            folder_patterns=list(folder_patterns or []),
            exclude_folders=exclude_list,
            sync_from_date=sync_from_date,
            max_message_age_days=max_message_age_days,
            provider=provider,
            privacy_settings=privacy_settings or MailboxPrivacySettings(),
            provenance_user=getpass.getuser(),
            provenance_machine_hash=_machine_id_hash(),
            provenance_tool_version=FUTURNAL_VERSION,
            network_warning=_detect_provider_warning(imap_host),
        )

    def update(self, **changes: Any) -> "ImapMailboxDescriptor":
        payload = self.model_dump()
        payload.update(changes)
        payload["updated_at"] = datetime.utcnow()
        return ImapMailboxDescriptor.model_validate(payload)

    def to_local_source(
        self,
        *,
        workspace_root: Optional[Path] = None,
        max_workers: Optional[int] = None,
        max_files_per_batch: Optional[int] = None,
        schedule: str = "@manual",
        priority: str = "normal",
        watcher_debounce_seconds: Optional[float] = None,
    ) -> LocalIngestionSource:
        """Build ``LocalIngestionSource`` representation for orchestrator."""

        source_name = self.name or f"imap-{self.id[:8]}"
        allow_plaintext_paths = (
            self.privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE
            and not self.privacy_settings.enable_subject_redaction
        )
        external_scopes = [
            scope.value
            for scope in self.privacy_settings.required_consent_scopes
            if scope != ConsentScope.MAILBOX_ACCESS
        ]
        external_scope = ",".join(external_scopes) if external_scopes else "imap.external_processing"
        base_workspace = (workspace_root or Path.home() / ".futurnal" / "workspace" / "imap").expanduser()
        mailbox_root = base_workspace / self.id
        mailbox_root.mkdir(parents=True, exist_ok=True)
        return LocalIngestionSource(
            name=source_name,
            root_path=mailbox_root,
            include=[],
            exclude=[],
            follow_symlinks=False,
            ignore_file=None,
            max_workers=max_workers,
            max_files_per_batch=max_files_per_batch,
            scan_interval_seconds=None,
            watcher_debounce_seconds=watcher_debounce_seconds,
            allow_plaintext_paths=allow_plaintext_paths,
            require_external_processing_consent=bool(external_scopes),
            external_processing_scope=external_scope,
            schedule=schedule,
            priority=priority,
            paused=False,
        )

    def build_redaction_policy(self) -> RedactionPolicy:
        """Create a redaction policy respecting mailbox privacy settings."""

        allow_plaintext = (
            self.privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE
            and not self.privacy_settings.enable_subject_redaction
        )

        class MailboxRedactionPolicy(RedactionPolicy):
            def __init__(
                self,
                descriptor: ImapMailboxDescriptor,
                *,
                allow_plaintext: bool,
            ) -> None:
                super().__init__(
                    allow_plaintext=allow_plaintext,
                    reveal_filename=descriptor.privacy_settings.privacy_level
                    != PrivacyLevel.STRICT,
                    reveal_extension=False,
                )
                self._descriptor = descriptor

            def apply(self, path: Path | str) -> "RedactedPath":
                redacted = super().apply(path)
                if self._descriptor.privacy_settings.enable_subject_redaction:
                    redacted = redact_path(
                        "subject://redacted",
                        policy=RedactionPolicy(
                            allow_plaintext=False,
                            reveal_filename=False,
                            reveal_extension=False,
                        ),
                    )
                return redacted

        return MailboxRedactionPolicy(self, allow_plaintext=allow_plaintext)

    def get_required_consent_scopes(self) -> List[str]:
        return [scope.value for scope in self.privacy_settings.required_consent_scopes]

    def requires_consent_for_scope(self, scope: ConsentScope) -> bool:
        return scope in self.privacy_settings.required_consent_scopes

    def get_audit_retention_days(self) -> int:
        return self.privacy_settings.retain_audit_days


# ---------------------------------------------------------------------------
# Registry implementation
# ---------------------------------------------------------------------------


@dataclass
class MailboxRegistry:
    """File-based registry for IMAP mailbox descriptors."""

    registry_root: Path
    audit_logger: Optional[Any] = None

    def __init__(
        self,
        registry_root: Optional[Path] = None,
        audit_logger: Optional[Any] = None,
    ) -> None:
        default_root = Path.home() / ".futurnal" / "sources" / "imap"
        self.registry_root = (registry_root or default_root).expanduser()
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.audit_logger = audit_logger

    def _descriptor_path(self, mailbox_id: str) -> Path:
        return self.registry_root / f"{mailbox_id}.json"

    def _lock_path(self, mailbox_id: str) -> Path:
        return self.registry_root / f"{mailbox_id}.json.lock"

    def register(
        self,
        *,
        email_address: str,
        imap_host: str,
        imap_port: int = 993,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        auth_mode: AuthMode,
        credential_id: str,
        folders: Optional[Iterable[str]] = None,
        folder_patterns: Optional[Iterable[str]] = None,
        exclude_folders: Optional[Iterable[str]] = None,
        sync_from_date: Optional[datetime] = None,
        max_message_age_days: Optional[int] = None,
        provider: Optional[str] = None,
        privacy_settings: Optional[MailboxPrivacySettings] = None,
        operator: Optional[str] = None,
    ) -> ImapMailboxDescriptor:
        descriptor = ImapMailboxDescriptor.from_registration(
            email_address=email_address,
            imap_host=imap_host,
            imap_port=imap_port,
            name=name,
            icon=icon,
            auth_mode=auth_mode,
            credential_id=credential_id,
            folders=folders,
            folder_patterns=folder_patterns,
            exclude_folders=exclude_folders,
            sync_from_date=sync_from_date,
            max_message_age_days=max_message_age_days,
            provider=provider,
            privacy_settings=privacy_settings,
        )
        return self.add_or_update(descriptor, operator=operator)

    def add_or_update(
        self,
        descriptor: ImapMailboxDescriptor,
        *,
        operator: Optional[str] = None,
    ) -> ImapMailboxDescriptor:
        path = self._descriptor_path(descriptor.id)
        lock = FileLock(str(self._lock_path(descriptor.id)))
        with lock:
            now = datetime.utcnow()
            descriptor = descriptor.update(updated_at=now)
            created_at = descriptor.created_at
            is_update = path.exists()
            if is_update:
                try:
                    existing = self.get(descriptor.id)
                    created_at = existing.created_at
                except FileNotFoundError:
                    pass
            descriptor = descriptor.update(created_at=created_at)
            self._write(path, descriptor)
            self._log_mailbox_event(
                "updated" if is_update else "registered",
                "success",
                descriptor,
                operator=operator,
            )
            return descriptor

    def list(self) -> List[ImapMailboxDescriptor]:
        items: List[ImapMailboxDescriptor] = []
        for file in sorted(self.registry_root.glob("*.json")):
            try:
                payload = json.loads(file.read_text())
                items.append(ImapMailboxDescriptor.model_validate(payload))
            except Exception:
                continue
        return items

    def get(self, mailbox_id: str) -> ImapMailboxDescriptor:
        path = self._descriptor_path(mailbox_id)
        if not path.exists():
            raise FileNotFoundError(f"Mailbox {mailbox_id} not found")
        payload = json.loads(path.read_text())
        return ImapMailboxDescriptor.model_validate(payload)

    def remove(
        self,
        mailbox_id: str,
        *,
        operator: Optional[str] = None,
    ) -> None:
        lock = FileLock(str(self._lock_path(mailbox_id)))
        with lock:
            path = self._descriptor_path(mailbox_id)
            if not path.exists():
                raise FileNotFoundError(mailbox_id)
            descriptor = self.get(mailbox_id)
            path.unlink()
            self._log_mailbox_event(
                "removed", "success", descriptor, operator=operator
            )

    def _write(self, path: Path, descriptor: ImapMailboxDescriptor) -> None:
        payload = json.dumps(descriptor.model_dump(mode="json"), indent=2)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)

    def _log_mailbox_event(
        self,
        action: str,
        status: str,
        descriptor: ImapMailboxDescriptor,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        operator: Optional[str] = None,
    ) -> None:
        if self.audit_logger is None:
            return
        redacted_path = redact_path(descriptor.email_address)
        event_metadata: Dict[str, Any] = {
            "mailbox_id": descriptor.id,
            "email_hash": sha256(descriptor.email_address.encode()).hexdigest()[:16],
            "auth_mode": descriptor.auth_mode.value,
            "provider": descriptor.provider,
            "consent_scopes": descriptor.get_required_consent_scopes(),
            "privacy_level": descriptor.privacy_settings.privacy_level.value,
            "created_at": descriptor.created_at.isoformat(),
            "updated_at": descriptor.updated_at.isoformat(),
            "tool_version": descriptor.provenance_tool_version,
        }
        if metadata:
            event_metadata.update(metadata)
        event = AuditEvent(
            job_id=f"imap_registry_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            source="imap_mailbox_registry",
            action=f"mailbox_{action}",
            status=status,
            timestamp=datetime.utcnow(),
            redacted_path=redacted_path.redacted,
            path_hash=redacted_path.path_hash,
            operator_action=operator,
            metadata=event_metadata,
        )
        try:
            self.audit_logger.record(event)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _deterministic_mailbox_id(email_address: str, imap_host: str) -> str:
    normalized = f"{email_address.lower()}@{imap_host.lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"imap:{normalized}"))


def _machine_id_hash() -> str:
    try:
        node = uuid.getnode()
        host = socket.gethostname()
        payload = f"{node}:{host}:{platform.system()}:{platform.machine()}".encode()
        return sha256(payload).hexdigest()
    except Exception:
        return "unknown"


def _detect_provider_warning(imap_host: str) -> Optional[str]:
    provider_warnings = {
        "imap.gmail.com": "Gmail requires OAuth2; app passwords deprecated",
        "outlook.office365.com": "Office 365 requires OAuth2 with modern auth",
    }
    return provider_warnings.get(imap_host)


def generate_mailbox_id(email_address: str, imap_host: str) -> str:
    """Generate deterministic mailbox ID from email and host."""

    return _deterministic_mailbox_id(email_address, imap_host)


def create_credential_id(mailbox_id: str) -> str:
    """Create credential ID for keychain storage."""

    return f"imap_cred_{mailbox_id}"


__all__ = [
    "AuthMode",
    "ConsentScope",
    "ImapMailboxDescriptor",
    "MailboxPrivacySettings",
    "MailboxRegistry",
    "PrivacyLevel",
    "create_credential_id",
    "generate_mailbox_id",
]

