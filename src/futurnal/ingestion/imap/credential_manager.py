"""Secure credential storage and OAuth lifecycle management for IMAP connector.

This module implements the ``CredentialManager`` described in
``docs/phase-1/imap-connector-production-plan/02-credential-manager.md``.
It provides OS keychain-backed storage for IMAP credentials, automatic
OAuth token refresh, and privacy-aware audit logging aligned with the
system architecture requirements.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import requests
from filelock import FileLock
from pydantic import BaseModel, Field, field_validator, model_validator

try:  # pragma: no cover - mirror configuration.settings fallback behaviour
    import keyring  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - only when optional dep missing
    keyring = None  # type: ignore

from ...privacy.audit import AuditEvent, AuditLogger


# ---------------------------------------------------------------------------
# Models and enums
# ---------------------------------------------------------------------------


class CredentialType(str, Enum):
    """Credential categories supported by the credential manager."""

    OAUTH2 = "oauth2"
    APP_PASSWORD = "app_password"


class OAuth2Tokens(BaseModel):
    """In-memory representation of OAuth2 token payloads."""

    access_token: str
    refresh_token: str
    token_type: str = Field(default="Bearer")
    expires_in: int = Field(default=0, ge=0)
    expires_at: Optional[datetime] = None
    scope: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ensure_expiry(self) -> "OAuth2Tokens":
        if self.expires_at is None and self.expires_in:
            self.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=self.expires_in
            )
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            self.expires_at = self.expires_at.replace(tzinfo=timezone.utc)
        return self

    def should_refresh(self, buffer_seconds: int) -> bool:
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) + timedelta(
            seconds=buffer_seconds
        ) >= self.expires_at

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": CredentialType.OAUTH2,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scope": self.scope,
        }


class AppPassword(BaseModel):
    """Wrapper for app-specific passwords stored in the keychain."""

    password: str


class ImapCredential(BaseModel):
    """Metadata describing an IMAP credential entry."""

    credential_id: str
    credential_type: CredentialType
    email_address: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    provider: Optional[str] = None

    @field_validator("created_at", "last_used_at", "expires_at")
    @classmethod
    def _ensure_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:  # type: ignore[override]
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "credential_id": self.credential_id,
            "credential_type": self.credential_type.value,
            "email_address": self.email_address,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat()
            if self.last_used_at
            else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "provider": self.provider,
        }

    @classmethod
    def from_metadata(cls, payload: Dict[str, Any]) -> "ImapCredential":
        return cls(
            credential_id=payload["credential_id"],
            credential_type=CredentialType(payload["credential_type"]),
            email_address=payload["email_address"],
            created_at=_parse_timestamp(payload.get("created_at"))
            or datetime.now(timezone.utc),
            last_used_at=_parse_timestamp(payload.get("last_used_at")),
            expires_at=_parse_timestamp(payload.get("expires_at")),
            provider=payload.get("provider"),
        )

    def with_updates(self, **changes: Any) -> "ImapCredential":
        return self.model_copy(update=changes)


class OAuthProvider(BaseModel):
    """Configuration for an OAuth2 provider supporting IMAP XOAUTH2."""

    name: str
    authorization_endpoint: str
    token_endpoint: str
    scopes: List[str]
    client_id: str
    client_secret: Optional[str] = None


class OAuthProviderRegistry:
    """Registry of supported OAuth2 providers for IMAP connectors."""

    def __init__(self, providers: Optional[Iterable[OAuthProvider]] = None) -> None:
        default_providers = list(providers or _default_providers())
        self._providers = {provider.name: provider for provider in default_providers}

    def get_provider(self, name: str) -> OAuthProvider:
        try:
            return self._providers[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown OAuth provider: {name}") from exc

    def register(self, provider: OAuthProvider) -> None:
        self._providers[provider.name] = provider

    def exchange_code_for_tokens(
        self,
        *,
        provider: OAuthProvider,
        authorization_code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None,
    ) -> OAuth2Tokens:
        data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "client_id": provider.client_id,
            "redirect_uri": redirect_uri,
        }
        if provider.client_secret:
            data["client_secret"] = provider.client_secret
        if code_verifier:
            data["code_verifier"] = code_verifier
        response = self._post(provider.token_endpoint, data=data)
        return _tokens_from_response(response)

    def refresh_access_token(
        self,
        *,
        provider: OAuthProvider,
        refresh_token: str,
    ) -> OAuth2Tokens:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": provider.client_id,
        }
        if provider.client_secret:
            data["client_secret"] = provider.client_secret
        response = self._post(provider.token_endpoint, data=data)
        return _tokens_from_response(response)

    def _post(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(url, data=data, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(
                f"OAuth provider returned {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()


# ---------------------------------------------------------------------------
# Credential manager implementation
# ---------------------------------------------------------------------------


@dataclass
class CredentialManager:
    """Manages IMAP credentials with OS keychain integration."""

    audit_logger: Optional[AuditLogger] = None
    oauth_provider_registry: Optional[OAuthProviderRegistry] = None
    keyring_module: Any = keyring
    service_name: str = "futurnal.imap"
    metadata_path: Path = Path.home() / ".futurnal" / "credentials" / "imap.json"
    refresh_buffer_seconds: int = 300
    _now: Any = datetime.now

    def __post_init__(self) -> None:
        if self.keyring_module is None:
            raise RuntimeError(
                "keyring module is unavailable; install 'keyring' to manage credentials"
            )
        self.metadata_path = self.metadata_path.expanduser().resolve()
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.metadata_path.with_suffix(".lock")))
        if not self.metadata_path.exists():
            self._write_metadata({})
        if self.oauth_provider_registry is None:
            self.oauth_provider_registry = OAuthProviderRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_oauth_tokens(
        self,
        *,
        credential_id: str,
        email_address: str,
        tokens: OAuth2Tokens,
        provider: str,
        operator: Optional[str] = None,
    ) -> ImapCredential:
        existing = self._load_metadata().get(credential_id)
        created_at = (
            _parse_timestamp(existing.get("created_at"))
            if existing
            else self._utcnow()
        )
        last_used_at = _parse_timestamp(existing.get("last_used_at")) if existing else None
        expires_at = tokens.expires_at

        secret_payload = tokens.serialize()
        secret_payload["email_address"] = email_address
        secret_payload["provider"] = provider
        self._write_secret(credential_id, secret_payload)

        credential = ImapCredential(
            credential_id=credential_id,
            credential_type=CredentialType.OAUTH2,
            email_address=email_address,
            created_at=created_at,
            last_used_at=last_used_at,
            expires_at=expires_at,
            provider=provider,
        )
        self._upsert_metadata(credential)
        self._emit_audit_event(
            action="credential_update" if existing else "credential_create",
            status="success",
            credential=credential,
            operator=operator,
        )
        return credential

    def store_app_password(
        self,
        *,
        credential_id: str,
        email_address: str,
        password: str,
        operator: Optional[str] = None,
    ) -> ImapCredential:
        existing = self._load_metadata().get(credential_id)
        created_at = (
            _parse_timestamp(existing.get("created_at"))
            if existing
            else self._utcnow()
        )
        last_used_at = _parse_timestamp(existing.get("last_used_at")) if existing else None

        secret_payload = {
            "type": CredentialType.APP_PASSWORD,
            "password": password,
            "email_address": email_address,
        }
        self._write_secret(credential_id, secret_payload)

        credential = ImapCredential(
            credential_id=credential_id,
            credential_type=CredentialType.APP_PASSWORD,
            email_address=email_address,
            created_at=created_at,
            last_used_at=last_used_at,
            expires_at=None,
        )
        self._upsert_metadata(credential)
        clear_sensitive_string(password)
        self._emit_audit_event(
            action="credential_update" if existing else "credential_create",
            status="success",
            credential=credential,
            operator=operator,
        )
        return credential

    def retrieve_credentials(
        self,
        credential_id: str,
    ) -> Union[OAuth2Tokens, AppPassword]:
        secret = self._read_secret(credential_id)
        if secret is None:
            raise KeyError(f"Credential {credential_id} not found")
        credential_type = secret.get("type")
        metadata = self._load_metadata().get(credential_id)
        if not metadata:
            raise KeyError(f"Metadata for credential {credential_id} missing")

        credential = ImapCredential.from_metadata(metadata)

        if credential_type == CredentialType.OAUTH2:
            tokens = OAuth2Tokens.model_validate(secret)
            if tokens.should_refresh(self.refresh_buffer_seconds):
                tokens = self.refresh_oauth_token(credential_id)
            self._mark_used(credential_id, tokens.expires_at)
            updated = credential.with_updates(
                last_used_at=self._utcnow(), expires_at=tokens.expires_at
            )
            self._emit_audit_event(
                action="credential_access",
                status="success",
                credential=updated,
            )
            return tokens

        if credential_type == CredentialType.APP_PASSWORD:
            app_password = AppPassword(password=secret["password"])
            self._mark_used(credential_id, None)
            updated = credential.with_updates(last_used_at=self._utcnow())
            self._emit_audit_event(
                action="credential_access",
                status="success",
                credential=updated,
            )
            return app_password

        raise ValueError(f"Unsupported credential type: {credential_type}")

    def refresh_oauth_token(
        self,
        credential_id: str,
        *,
        operator: Optional[str] = None,
    ) -> OAuth2Tokens:
        secret = self._read_secret(credential_id)
        if secret is None:
            raise KeyError(f"Credential {credential_id} not found")
        if secret.get("type") != CredentialType.OAUTH2:
            raise ValueError("Only OAuth2 credentials can be refreshed")

        metadata = self._load_metadata().get(credential_id)
        if not metadata:
            raise KeyError(f"Metadata for credential {credential_id} missing")

        provider_name = secret.get("provider") or metadata.get("provider")
        if not provider_name:
            raise ValueError("OAuth provider name not stored with credential")
        registry = self.oauth_provider_registry or OAuthProviderRegistry()
        provider = registry.get_provider(provider_name)

        tokens = registry.refresh_access_token(
            provider=provider, refresh_token=secret["refresh_token"]
        )
        email_address = secret.get("email_address") or metadata.get("email_address")
        if not email_address:
            raise ValueError("Email address missing from credential metadata")

        self.store_oauth_tokens(
            credential_id=credential_id,
            email_address=email_address,
            tokens=tokens,
            provider=provider_name,
            operator=operator,
        )
        return tokens

    def delete_credentials(
        self,
        credential_id: str,
        *,
        operator: Optional[str] = None,
    ) -> None:
        metadata = self._load_metadata()
        credential_meta = metadata.pop(credential_id, None)
        try:
            self.keyring_module.delete_password(self.service_name, credential_id)
        except Exception:  # pragma: no cover - backend specific quirks
            pass
        self._write_metadata(metadata)
        if credential_meta:
            credential = ImapCredential.from_metadata(credential_meta)
            self._emit_audit_event(
                action="credential_delete",
                status="success",
                credential=credential,
                operator=operator,
            )

    def list_credentials(self) -> List[ImapCredential]:
        metadata = self._load_metadata()
        credentials = [ImapCredential.from_metadata(item) for item in metadata.values()]
        return sorted(credentials, key=lambda item: item.created_at)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_secret(self, credential_id: str, payload: Dict[str, Any]) -> None:
        json_payload = json.dumps(payload)
        self.keyring_module.set_password(self.service_name, credential_id, json_payload)

    def _read_secret(self, credential_id: str) -> Optional[Dict[str, Any]]:
        raw = self.keyring_module.get_password(self.service_name, credential_id)
        if raw is None:
            return None
        return json.loads(raw)

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            try:
                data = json.loads(self.metadata_path.read_text())
            except json.JSONDecodeError:
                data = {}
        return data

    def _write_metadata(self, metadata: Dict[str, Dict[str, Any]]) -> None:
        with self._lock:
            tmp = self.metadata_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(metadata, indent=2))
            os.replace(tmp, self.metadata_path)

    def _upsert_metadata(self, credential: ImapCredential) -> None:
        metadata = self._load_metadata()
        metadata[credential.credential_id] = credential.to_metadata()
        self._write_metadata(metadata)

    def _mark_used(self, credential_id: str, expires_at: Optional[datetime]) -> None:
        metadata = self._load_metadata()
        entry = metadata.get(credential_id)
        if not entry:
            return
        entry["last_used_at"] = self._utcnow().isoformat()
        entry["expires_at"] = expires_at.isoformat() if expires_at else None
        metadata[credential_id] = entry
        self._write_metadata(metadata)

    def _emit_audit_event(
        self,
        *,
        action: str,
        status: str,
        credential: ImapCredential,
        operator: Optional[str] = None,
    ) -> None:
        if self.audit_logger is None:
            return
        email_hash = sha256(credential.email_address.encode("utf-8")).hexdigest()[:16]
        metadata = {
            "credential_id": credential.credential_id,
            "credential_type": credential.credential_type,
            "email_hash": email_hash,
            "provider": credential.provider,
            "expires_at": credential.expires_at.isoformat()
            if credential.expires_at
            else None,
        }
        event = AuditEvent(
            job_id=f"credential_{action}_{credential.credential_id}",
            source="imap_credential_manager",
            action=action,
            status=status,
            timestamp=self._utcnow(),
            operator_action=operator,
            metadata=metadata,
        )
        try:
            self.audit_logger.record(event)
        except Exception:  # pragma: no cover - audit failure should not block
            pass

    def _utcnow(self) -> datetime:
        now = self._now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _tokens_from_response(payload: Dict[str, Any]) -> OAuth2Tokens:
    scope_value = payload.get("scope") or payload.get("scopes") or []
    if isinstance(scope_value, str):
        scope = scope_value.split()
    else:
        scope = list(scope_value)
    expires_at: Optional[datetime] = None
    raw_expiry = payload.get("expires_at") or payload.get("expiry")
    if isinstance(raw_expiry, str):
        expires_at = _parse_timestamp(raw_expiry)
    tokens = OAuth2Tokens(
        access_token=payload.get("access_token"),
        refresh_token=payload.get("refresh_token"),
        token_type=payload.get("token_type", "Bearer"),
        expires_in=payload.get("expires_in", 0),
        expires_at=expires_at,
        scope=scope,
    )
    return tokens


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def clear_sensitive_string(value: str) -> None:
    """Best-effort clearing of sensitive string references."""

    try:
        del value  # type: ignore # noqa: F821
    finally:
        import gc

        gc.collect()


@contextmanager
def secure_credential_context(
    credential_manager: CredentialManager, credential_id: str
):
    """Context manager ensuring credentials are cleared from memory."""

    credentials = None
    try:
        credentials = credential_manager.retrieve_credentials(credential_id)
        yield credentials
    finally:
        if credentials is not None:
            if isinstance(credentials, OAuth2Tokens):
                clear_sensitive_string(credentials.access_token)
                clear_sensitive_string(credentials.refresh_token)
            elif isinstance(credentials, AppPassword):
                clear_sensitive_string(credentials.password)
        del credentials  # type: ignore # noqa: F821
        import gc

        gc.collect()


async def auto_refresh_wrapper(
    credential_manager: CredentialManager,
    credential_id: str,
) -> OAuth2Tokens:
    """Retrieve credentials, refreshing OAuth tokens when necessary."""

    credentials = credential_manager.retrieve_credentials(credential_id)
    if not isinstance(credentials, OAuth2Tokens):
        raise ValueError("auto_refresh_wrapper only supports OAuth2 credentials")
    if credentials.should_refresh(credential_manager.refresh_buffer_seconds):
        credentials = credential_manager.refresh_oauth_token(credential_id)
    return credentials


def _default_providers() -> Iterable[OAuthProvider]:
    from os import getenv

    return [
        OAuthProvider(
            name="gmail",
            authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            scopes=["https://mail.google.com/"],
            client_id=getenv("FUTURNAL_GMAIL_CLIENT_ID", ""),
            client_secret=getenv("FUTURNAL_GMAIL_CLIENT_SECRET"),
        ),
        OAuthProvider(
            name="office365",
            authorization_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            token_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/token",
            scopes=[
                "https://outlook.office.com/IMAP.AccessAsUser.All",
                "offline_access",
            ],
            client_id=getenv("FUTURNAL_OFFICE365_CLIENT_ID", ""),
            client_secret=getenv("FUTURNAL_OFFICE365_CLIENT_SECRET"),
        ),
    ]


__all__ = [
    "AppPassword",
    "CredentialManager",
    "CredentialType",
    "ImapCredential",
    "OAuth2Tokens",
    "OAuthProvider",
    "OAuthProviderRegistry",
    "auto_refresh_wrapper",
    "clear_sensitive_string",
    "secure_credential_context",
]


