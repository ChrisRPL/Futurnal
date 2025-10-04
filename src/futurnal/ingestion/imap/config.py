"""Configuration model for IMAP mailbox ingestion.

This model wraps the descriptor for use by the ingestion orchestrator and
testing harnesses. It mirrors the structure used by local sources while
providing IMAP-specific validation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .descriptor import AuthMode, MailboxPrivacySettings, PrivacyLevel


class ImapMailboxConfig(BaseModel):
    """Runtime configuration for an IMAP mailbox connector."""

    mailbox_id: str = Field(..., description="Identifier from the registry")
    descriptor_path: Path = Field(..., description="Path to descriptor JSON")
    state_path: Path = Field(..., description="Path to sync state DB")
    telemetry_dir: Path = Field(..., description="Directory for telemetry")
    workspace_dir: Path = Field(..., description="Workspace root for IMAP")
    max_parallel_folders: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Maximum number of folders processed concurrently",
    )
    idle_timeout_seconds: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Idle renewal timeout",
    )
    reconnection_backoff_seconds: List[int] = Field(
        default_factory=lambda: [5, 15, 45, 120],
        description="Backoff schedule for connection retries",
    )
    sync_batch_size: int = Field(
        default=250,
        ge=1,
        le=2000,
        description="Messages per sync batch",
    )
    max_concurrent_attachments: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Concurrent attachment normalization workers",
    )
    credential_refresh_buffer_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Seconds before expiry to refresh OAuth tokens",
    )
    auth_mode: AuthMode = Field(..., description="Authentication mode in use")
    privacy_settings: MailboxPrivacySettings = Field(
        default_factory=MailboxPrivacySettings,
        description="Privacy settings snapshot at runtime",
    )
    last_synced_at: Optional[datetime] = Field(
        default=None, description="Timestamp of last successful sync"
    )

    @field_validator("descriptor_path", "state_path", "telemetry_dir", "workspace_dir")
    @classmethod
    def _ensure_absolute(cls, value: Path) -> Path:  # type: ignore[override]
        return value.expanduser().resolve()

    @field_validator("telemetry_dir", "workspace_dir")
    @classmethod
    def _ensure_directory(cls, value: Path) -> Path:  # type: ignore[override]
        value.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("privacy_settings")
    @classmethod
    def _validate_privacy(cls, value: MailboxPrivacySettings) -> MailboxPrivacySettings:  # type: ignore[override]
        if (
            value.privacy_level == PrivacyLevel.PERMISSIVE
            and value.enable_sender_anonymization
            and value.enable_recipient_anonymization
        ):
            raise ValueError(
                "Permissive privacy level should not enable anonymization flags; disable or choose STANDARD"
            )
        return value


__all__ = ["ImapMailboxConfig"]


