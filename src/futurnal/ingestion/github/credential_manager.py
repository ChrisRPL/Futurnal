"""Secure credential storage and management for GitHub connector.

This module implements credential management for GitHub repositories including
OAuth tokens and Personal Access Tokens (PATs), with OS keychain-backed storage
and privacy-aware audit logging.
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
from typing import Any, Dict, List, Optional, Union

import requests
from filelock import FileLock
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    import keyring  # type: ignore
except ModuleNotFoundError:
    keyring = None  # type: ignore

from ...privacy.audit import AuditEvent, AuditLogger


# ---------------------------------------------------------------------------
# Models and enums
# ---------------------------------------------------------------------------


class CredentialType(str, Enum):
    """Credential types supported for GitHub authentication."""

    OAUTH_TOKEN = "oauth_token"  # OAuth token from Device Flow or OAuth App
    PERSONAL_ACCESS_TOKEN = "personal_access_token"  # Classic or fine-grained PAT


class GitHubCredential(BaseModel):
    """Metadata for a GitHub credential entry."""

    credential_id: str = Field(..., description="Unique credential identifier")
    credential_type: CredentialType = Field(..., description="Type of credential")
    github_host: str = Field(default="github.com", description="GitHub hostname")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used_at: Optional[datetime] = Field(
        default=None, description="Last access timestamp"
    )
    scopes: List[str] = Field(
        default_factory=list, description="OAuth scopes granted"
    )
    note: Optional[str] = Field(
        default=None, description="User-provided description"
    )

    @field_validator("created_at", "last_used_at")
    @classmethod
    def _ensure_timezone(
        cls, value: Optional[datetime]
    ) -> Optional[datetime]:  # type: ignore[override]
        """Ensure datetime has timezone info."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for storage."""
        return {
            "credential_id": self.credential_id,
            "credential_type": self.credential_type.value,
            "github_host": self.github_host,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat()
            if self.last_used_at
            else None,
            "scopes": self.scopes,
            "note": self.note,
        }

    @classmethod
    def from_metadata(cls, payload: Dict[str, Any]) -> "GitHubCredential":
        """Create from metadata dict."""
        return cls(
            credential_id=payload["credential_id"],
            credential_type=CredentialType(payload["credential_type"]),
            github_host=payload.get("github_host", "github.com"),
            created_at=_parse_timestamp(payload.get("created_at"))
            or datetime.now(timezone.utc),
            last_used_at=_parse_timestamp(payload.get("last_used_at")),
            scopes=payload.get("scopes", []),
            note=payload.get("note"),
        )

    def with_updates(self, **changes: Any) -> "GitHubCredential":
        """Create updated copy with changes."""
        return self.model_copy(update=changes)


class OAuthToken(BaseModel):
    """OAuth token credential."""

    token: str = Field(..., description="OAuth access token")
    token_type: str = Field(default="Bearer", description="Token type")
    scopes: List[str] = Field(default_factory=list, description="Granted scopes")


class PersonalAccessToken(BaseModel):
    """Personal Access Token (classic or fine-grained)."""

    token: str = Field(..., description="Personal access token")
    scopes: List[str] = Field(default_factory=list, description="Token scopes")


class OAuthTokens(BaseModel):
    """OAuth tokens with expiration and refresh support.

    This enhanced model supports token expiration tracking and automatic
    refresh for GitHub OAuth tokens. Note that GitHub's Device Flow typically
    doesn't provide refresh tokens or expiration times.
    """

    access_token: str = Field(..., description="OAuth access token")
    refresh_token: Optional[str] = Field(
        default=None, description="Refresh token (rarely provided by GitHub)"
    )
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: Optional[int] = Field(
        default=None, ge=0, description="Expiration time in seconds"
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Absolute expiration timestamp"
    )
    scopes: List[str] = Field(default_factory=list, description="Granted scopes")
    username: Optional[str] = Field(
        default=None, description="GitHub username (fetched from API)"
    )

    @field_validator("expires_at")
    @classmethod
    def _ensure_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        """Ensure expires_at has timezone info."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def _calculate_expiration(self) -> "OAuthTokens":
        """Calculate expires_at from expires_in if not provided."""
        if self.expires_at is None and self.expires_in is not None:
            self.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=self.expires_in
            )
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            self.expires_at = self.expires_at.replace(tzinfo=timezone.utc)
        return self

    def should_refresh(self, buffer_seconds: int = 300) -> bool:
        """Check if token should be refreshed.

        Args:
            buffer_seconds: Refresh buffer time (default: 5 minutes)

        Returns:
            True if token should be refreshed, False otherwise
        """
        if not self.expires_at:
            return False  # No expiration, no need to refresh
        return datetime.now(timezone.utc) + timedelta(
            seconds=buffer_seconds
        ) >= self.expires_at

    def serialize(self) -> Dict[str, Any]:
        """Serialize tokens for keychain storage."""
        return {
            "type": CredentialType.OAUTH_TOKEN,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": self.scopes,
            "username": self.username,
        }


class EnterpriseOAuthConfig(BaseModel):
    """OAuth configuration for GitHub Enterprise Server."""

    host: str = Field(..., description="Enterprise Server hostname")
    client_id: str = Field(..., description="OAuth App Client ID")
    client_secret: Optional[str] = Field(
        default=None, description="OAuth App Client Secret (optional for device flow)"
    )
    api_base_url: str = Field(..., description="API base URL (e.g., https://github.company.com/api/v3)")
    device_code_endpoint: Optional[str] = Field(
        default=None, description="Custom device code endpoint"
    )
    token_endpoint: Optional[str] = Field(
        default=None, description="Custom token endpoint"
    )
    scopes: List[str] = Field(
        default_factory=lambda: ["repo", "read:org"],
        description="Default OAuth scopes"
    )


class GitHubOAuthConfig(BaseModel):
    """OAuth configuration for GitHub instances.

    Centralizes OAuth configuration for github.com and Enterprise Server instances.
    """

    # GitHub.com OAuth (default)
    client_id: str = Field(
        default_factory=lambda: os.getenv("FUTURNAL_GITHUB_CLIENT_ID", ""),
        description="OAuth App Client ID for github.com"
    )
    client_secret: Optional[str] = Field(
        default_factory=lambda: os.getenv("FUTURNAL_GITHUB_CLIENT_SECRET"),
        description="OAuth App Client Secret (optional for device flow)"
    )

    # OAuth endpoints (GitHub.com defaults)
    device_code_endpoint: str = Field(
        default="https://github.com/login/device/code",
        description="Device code request endpoint"
    )
    token_endpoint: str = Field(
        default="https://github.com/login/oauth/access_token",
        description="Token exchange endpoint"
    )
    authorization_endpoint: str = Field(
        default="https://github.com/login/oauth/authorize",
        description="Web authorization endpoint (optional)"
    )

    # Scopes
    default_scopes: List[str] = Field(
        default_factory=lambda: ["repo", "read:org"],
        description="Default OAuth scopes to request"
    )

    # GitHub Enterprise support
    enterprise_configs: Dict[str, EnterpriseOAuthConfig] = Field(
        default_factory=dict,
        description="OAuth configs for enterprise instances (keyed by host)"
    )

    def get_config_for_host(self, github_host: str) -> Dict[str, Any]:
        """Get OAuth configuration for a specific GitHub host.

        Args:
            github_host: GitHub hostname (e.g., 'github.com' or 'github.company.com')

        Returns:
            Dict with client_id, endpoints, and scopes for the host
        """
        if github_host == "github.com":
            return {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "device_code_endpoint": self.device_code_endpoint,
                "token_endpoint": self.token_endpoint,
                "authorization_endpoint": self.authorization_endpoint,
                "scopes": self.default_scopes,
            }

        # Enterprise Server
        if github_host in self.enterprise_configs:
            enterprise = self.enterprise_configs[github_host]
            return {
                "client_id": enterprise.client_id,
                "client_secret": enterprise.client_secret,
                "device_code_endpoint": enterprise.device_code_endpoint
                or f"https://{github_host}/login/device/code",
                "token_endpoint": enterprise.token_endpoint
                or f"https://{github_host}/login/oauth/access_token",
                "authorization_endpoint": f"https://{github_host}/login/oauth/authorize",
                "scopes": enterprise.scopes,
            }

        raise ValueError(f"No OAuth configuration found for host: {github_host}")


# ---------------------------------------------------------------------------
# Credential manager implementation
# ---------------------------------------------------------------------------


@dataclass
class GitHubCredentialManager:
    """Manages GitHub credentials with OS keychain integration."""

    audit_logger: Optional[AuditLogger] = None
    oauth_config: Optional[GitHubOAuthConfig] = None
    keyring_module: Any = keyring
    service_name: str = "futurnal.github"
    metadata_path: Path = Path.home() / ".futurnal" / "credentials" / "github.json"
    refresh_buffer_seconds: int = 300  # 5 minutes
    _now: Any = datetime.now

    def __post_init__(self) -> None:
        """Initialize credential manager."""
        if self.keyring_module is None:
            raise RuntimeError(
                "keyring module is unavailable; install 'keyring' to manage credentials"
            )

        self.metadata_path = self.metadata_path.expanduser().resolve()
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.metadata_path.with_suffix(".lock")))

        if not self.metadata_path.exists():
            self._write_metadata({})

        if self.oauth_config is None:
            self.oauth_config = GitHubOAuthConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_oauth_token(
        self,
        *,
        credential_id: str,
        token: str,
        scopes: List[str],
        github_host: str = "github.com",
        note: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> GitHubCredential:
        """Store OAuth token in keychain."""
        existing = self._load_metadata().get(credential_id)
        created_at = (
            _parse_timestamp(existing.get("created_at"))
            if existing
            else self._utcnow()
        )

        # Store token in keychain
        secret_payload = {
            "type": CredentialType.OAUTH_TOKEN,
            "token": token,
            "scopes": scopes,
            "github_host": github_host,
        }
        self._write_secret(credential_id, secret_payload)

        # Create metadata
        credential = GitHubCredential(
            credential_id=credential_id,
            credential_type=CredentialType.OAUTH_TOKEN,
            github_host=github_host,
            created_at=created_at,
            last_used_at=existing.get("last_used_at") if existing else None,
            scopes=scopes,
            note=note,
        )

        self._upsert_metadata(credential)
        self._emit_audit_event(
            action="credential_update" if existing else "credential_create",
            status="success",
            credential=credential,
            operator=operator,
        )

        return credential

    def store_personal_access_token(
        self,
        *,
        credential_id: str,
        token: str,
        scopes: List[str],
        github_host: str = "github.com",
        note: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> GitHubCredential:
        """Store Personal Access Token in keychain."""
        existing = self._load_metadata().get(credential_id)
        created_at = (
            _parse_timestamp(existing.get("created_at"))
            if existing
            else self._utcnow()
        )

        # Store token in keychain
        secret_payload = {
            "type": CredentialType.PERSONAL_ACCESS_TOKEN,
            "token": token,
            "scopes": scopes,
            "github_host": github_host,
        }
        self._write_secret(credential_id, secret_payload)

        # Create metadata
        credential = GitHubCredential(
            credential_id=credential_id,
            credential_type=CredentialType.PERSONAL_ACCESS_TOKEN,
            github_host=github_host,
            created_at=created_at,
            last_used_at=existing.get("last_used_at") if existing else None,
            scopes=scopes,
            note=note,
        )

        self._upsert_metadata(credential)
        clear_sensitive_string(token)
        self._emit_audit_event(
            action="credential_update" if existing else "credential_create",
            status="success",
            credential=credential,
            operator=operator,
        )

        return credential

    def retrieve_credentials(
        self, credential_id: str, *, auto_refresh: bool = True
    ) -> Union[OAuthToken, PersonalAccessToken, OAuthTokens]:
        """Retrieve credentials from keychain with optional auto-refresh.

        Args:
            credential_id: Credential identifier
            auto_refresh: Whether to automatically refresh expiring OAuth tokens

        Returns:
            OAuthToken, PersonalAccessToken, or OAuthTokens (if expiration data present)

        Raises:
            KeyError: If credential not found
        """
        secret = self._read_secret(credential_id)
        if secret is None:
            raise KeyError(f"Credential {credential_id} not found")

        metadata = self._load_metadata().get(credential_id)
        if not metadata:
            raise KeyError(f"Metadata for credential {credential_id} missing")

        credential = GitHubCredential.from_metadata(metadata)
        credential_type = secret.get("type")

        # Handle OAuth tokens with potential auto-refresh
        if credential_type == CredentialType.OAUTH_TOKEN:
            # Check if this is an enhanced OAuthTokens (has expiration data)
            has_expiration = secret.get("expires_at") is not None

            if has_expiration:
                # Build OAuthTokens object
                tokens = OAuthTokens(
                    access_token=secret.get("access_token", secret.get("token", "")),
                    refresh_token=secret.get("refresh_token"),
                    token_type=secret.get("token_type", "Bearer"),
                    expires_in=secret.get("expires_in"),
                    expires_at=_parse_timestamp(secret.get("expires_at")),
                    scopes=secret.get("scopes", []),
                    username=secret.get("username"),
                )

                # Auto-refresh if needed and possible
                if auto_refresh and tokens.should_refresh(self.refresh_buffer_seconds):
                    if tokens.refresh_token:
                        try:
                            tokens = self.refresh_oauth_token(credential_id)
                        except Exception:
                            # If refresh fails, continue with current token
                            pass

                # Mark as used
                self._mark_used(credential_id)

                # Emit audit event
                updated = credential.with_updates(last_used_at=self._utcnow())
                self._emit_audit_event(
                    action="credential_access",
                    status="success",
                    credential=updated,
                )

                return tokens
            else:
                # Simple OAuth token (backward compatibility)
                self._mark_used(credential_id)
                updated = credential.with_updates(last_used_at=self._utcnow())
                self._emit_audit_event(
                    action="credential_access",
                    status="success",
                    credential=updated,
                )
                return OAuthToken(
                    token=secret.get("access_token", secret.get("token", "")),
                    token_type="Bearer",
                    scopes=secret.get("scopes", []),
                )

        elif credential_type == CredentialType.PERSONAL_ACCESS_TOKEN:
            self._mark_used(credential_id)
            updated = credential.with_updates(last_used_at=self._utcnow())
            self._emit_audit_event(
                action="credential_access",
                status="success",
                credential=updated,
            )
            return PersonalAccessToken(
                token=secret["token"], scopes=secret.get("scopes", [])
            )
        else:
            raise ValueError(f"Unsupported credential type: {credential_type}")

    def delete_credentials(
        self, credential_id: str, *, operator: Optional[str] = None
    ) -> None:
        """Delete credentials from keychain."""
        metadata = self._load_metadata()
        credential_meta = metadata.pop(credential_id, None)

        try:
            self.keyring_module.delete_password(self.service_name, credential_id)
        except Exception:  # pragma: no cover
            pass

        self._write_metadata(metadata)

        if credential_meta:
            credential = GitHubCredential.from_metadata(credential_meta)
            self._emit_audit_event(
                action="credential_delete",
                status="success",
                credential=credential,
                operator=operator,
            )

    def list_credentials(self) -> List[GitHubCredential]:
        """List all stored credentials."""
        metadata = self._load_metadata()
        credentials = [
            GitHubCredential.from_metadata(item) for item in metadata.values()
        ]
        return sorted(credentials, key=lambda item: item.created_at)

    def get_credential_metadata(self, credential_id: str) -> GitHubCredential:
        """Get credential metadata without retrieving the token."""
        metadata = self._load_metadata().get(credential_id)
        if not metadata:
            raise KeyError(f"Credential {credential_id} not found")
        return GitHubCredential.from_metadata(metadata)

    def update_scopes(
        self,
        credential_id: str,
        scopes: List[str],
        *,
        operator: Optional[str] = None,
    ) -> GitHubCredential:
        """Update scopes for a credential after validation."""
        metadata = self._load_metadata().get(credential_id)
        if not metadata:
            raise KeyError(f"Credential {credential_id} not found")

        credential = GitHubCredential.from_metadata(metadata)
        updated = credential.with_updates(scopes=scopes)
        self._upsert_metadata(updated)

        self._emit_audit_event(
            action="credential_scopes_updated",
            status="success",
            credential=updated,
            operator=operator,
        )

        return updated

    def store_oauth_tokens(
        self,
        *,
        credential_id: str,
        tokens: OAuthTokens,
        github_host: str = "github.com",
        note: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> GitHubCredential:
        """Store OAuth tokens with expiration support.

        This enhanced method stores OAuthTokens with optional expiration
        and refresh token support.

        Args:
            credential_id: Unique credential identifier
            tokens: OAuthTokens with access_token, optional refresh_token, expires_at
            github_host: GitHub hostname
            note: Optional user note
            operator: Optional operator name for audit

        Returns:
            GitHubCredential metadata
        """
        existing = self._load_metadata().get(credential_id)
        created_at = (
            _parse_timestamp(existing.get("created_at"))
            if existing
            else self._utcnow()
        )

        # Store tokens in keychain
        secret_payload = tokens.serialize()
        secret_payload["github_host"] = github_host
        self._write_secret(credential_id, secret_payload)

        # Create metadata
        credential = GitHubCredential(
            credential_id=credential_id,
            credential_type=CredentialType.OAUTH_TOKEN,
            github_host=github_host,
            created_at=created_at,
            last_used_at=existing.get("last_used_at") if existing else None,
            scopes=tokens.scopes,
            note=note,
        )

        self._upsert_metadata(credential)
        self._emit_audit_event(
            action="credential_update" if existing else "credential_create",
            status="success",
            credential=credential,
            operator=operator,
        )

        return credential

    def refresh_oauth_token(
        self,
        credential_id: str,
        *,
        operator: Optional[str] = None,
    ) -> OAuthTokens:
        """Refresh OAuth token using refresh token.

        Note: GitHub's Device Flow typically doesn't provide refresh tokens.
        This method will raise an error if no refresh token is available.

        Args:
            credential_id: Credential identifier
            operator: Optional operator name for audit

        Returns:
            Refreshed OAuthTokens

        Raises:
            KeyError: If credential not found
            ValueError: If credential is not OAuth or has no refresh token
            RuntimeError: If token refresh API call fails
        """
        secret = self._read_secret(credential_id)
        if secret is None:
            raise KeyError(f"Credential {credential_id} not found")
        if secret.get("type") != CredentialType.OAUTH_TOKEN:
            raise ValueError("Only OAuth credentials can be refreshed")

        refresh_token = secret.get("refresh_token")
        if not refresh_token:
            raise ValueError(
                "No refresh token available. GitHub Device Flow typically doesn't provide refresh tokens."
            )

        metadata = self._load_metadata().get(credential_id)
        if not metadata:
            raise KeyError(f"Metadata for credential {credential_id} missing")

        github_host = secret.get("github_host") or metadata.get("github_host", "github.com")

        # Get OAuth configuration for this host
        if self.oauth_config is None:
            raise RuntimeError("OAuth configuration not initialized")

        config = self.oauth_config.get_config_for_host(github_host)

        # Call token refresh endpoint
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": config["client_id"],
        }
        if config.get("client_secret"):
            data["client_secret"] = config["client_secret"]

        try:
            response = requests.post(
                config["token_endpoint"],
                data=data,
                headers={"Accept": "application/json"},
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Token refresh failed: {e}")

        # Parse new tokens
        new_tokens = OAuthTokens(
            access_token=result.get("access_token", secret["access_token"]),
            refresh_token=result.get("refresh_token", refresh_token),
            token_type=result.get("token_type", "Bearer"),
            expires_in=result.get("expires_in"),
            scopes=result.get("scope", "").split() if result.get("scope") else secret.get("scopes", []),
            username=secret.get("username"),
        )

        # Calculate expires_at if expires_in provided
        if new_tokens.expires_in:
            new_tokens.expires_at = self._utcnow() + timedelta(seconds=new_tokens.expires_in)

        # Store refreshed tokens
        self.store_oauth_tokens(
            credential_id=credential_id,
            tokens=new_tokens,
            github_host=github_host,
            note=metadata.get("note"),
            operator=operator,
        )

        self._emit_audit_event(
            action="credential_refresh",
            status="success",
            credential=GitHubCredential.from_metadata(metadata),
            operator=operator,
        )

        return new_tokens

    def fetch_username(self, token: str, github_host: str = "github.com") -> str:
        """Fetch GitHub username using token.

        Args:
            token: GitHub access token
            github_host: GitHub hostname

        Returns:
            GitHub username

        Raises:
            RuntimeError: If API call fails
        """
        # Determine API base URL
        if github_host == "github.com":
            api_base_url = "https://api.github.com"
        else:
            api_base_url = f"https://{github_host}/api/v3"

        try:
            response = requests.get(
                f"{api_base_url}/user",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("login", "unknown")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch username: {e}")

    def detect_scopes_from_api(
        self, token: str, github_host: str = "github.com"
    ) -> List[str]:
        """Detect token scopes via GitHub API.

        Useful for Personal Access Tokens where scopes aren't provided during creation.

        Args:
            token: GitHub access token
            github_host: GitHub hostname

        Returns:
            List of detected scopes

        Raises:
            RuntimeError: If API call fails
        """
        # Determine API base URL
        if github_host == "github.com":
            api_base_url = "https://api.github.com"
        else:
            api_base_url = f"https://{github_host}/api/v3"

        try:
            response = requests.get(
                f"{api_base_url}/rate_limit",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=15,
            )
            response.raise_for_status()

            # Parse scopes from header
            scope_header = response.headers.get("X-OAuth-Scopes", "")
            if scope_header:
                return [s.strip() for s in scope_header.split(",")]
            return []
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to detect scopes: {e}")

    def validate_credentials(self, credential_id: str) -> bool:
        """Validate credentials by making a test API call.

        Args:
            credential_id: Credential identifier

        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            creds = self.retrieve_credentials(credential_id)
            metadata = self.get_credential_metadata(credential_id)

            # Make test API call to verify token
            token = creds.token if hasattr(creds, "token") else creds.access_token
            self.fetch_username(token, metadata.github_host)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_secret(self, credential_id: str, payload: Dict[str, Any]) -> None:
        """Write secret to keychain."""
        json_payload = json.dumps(payload)
        self.keyring_module.set_password(
            self.service_name, credential_id, json_payload
        )

    def _read_secret(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Read secret from keychain."""
        raw = self.keyring_module.get_password(self.service_name, credential_id)
        if raw is None:
            return None
        return json.loads(raw)

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load credential metadata from file."""
        with self._lock:
            try:
                data = json.loads(self.metadata_path.read_text())
            except json.JSONDecodeError:
                data = {}
        return data

    def _write_metadata(self, metadata: Dict[str, Dict[str, Any]]) -> None:
        """Write credential metadata to file."""
        with self._lock:
            tmp = self.metadata_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(metadata, indent=2))
            os.replace(tmp, self.metadata_path)

    def _upsert_metadata(self, credential: GitHubCredential) -> None:
        """Insert or update credential metadata."""
        metadata = self._load_metadata()
        metadata[credential.credential_id] = credential.to_metadata()
        self._write_metadata(metadata)

    def _mark_used(self, credential_id: str) -> None:
        """Mark credential as recently used."""
        metadata = self._load_metadata()
        entry = metadata.get(credential_id)
        if not entry:
            return
        entry["last_used_at"] = self._utcnow().isoformat()
        metadata[credential_id] = entry
        self._write_metadata(metadata)

    def _emit_audit_event(
        self,
        *,
        action: str,
        status: str,
        credential: GitHubCredential,
        operator: Optional[str] = None,
    ) -> None:
        """Emit audit event for credential operations."""
        if self.audit_logger is None:
            return

        credential_hash = sha256(credential.credential_id.encode("utf-8")).hexdigest()[
            :16
        ]
        metadata = {
            "credential_id_hash": credential_hash,
            "credential_type": credential.credential_type.value,
            "github_host": credential.github_host,
            "scopes": credential.scopes,
            "note": credential.note,
        }

        event = AuditEvent(
            job_id=f"github_credential_{action}_{credential.credential_id}",
            source="github_credential_manager",
            action=action,
            status=status,
            timestamp=self._utcnow(),
            operator_action=operator,
            metadata=metadata,
        )

        try:
            self.audit_logger.record(event)
        except Exception:  # pragma: no cover
            pass

    def _utcnow(self) -> datetime:
        """Get current UTC time with timezone."""
        now = self._now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string."""
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
    credential_manager: GitHubCredentialManager, credential_id: str
):
    """Context manager ensuring credentials are cleared from memory."""
    credentials = None
    try:
        credentials = credential_manager.retrieve_credentials(credential_id)
        yield credentials
    finally:
        if credentials is not None:
            if isinstance(credentials, OAuthToken):
                clear_sensitive_string(credentials.token)
            elif isinstance(credentials, PersonalAccessToken):
                clear_sensitive_string(credentials.token)
            elif isinstance(credentials, OAuthTokens):
                clear_sensitive_string(credentials.access_token)
                if credentials.refresh_token:
                    clear_sensitive_string(credentials.refresh_token)
        del credentials  # type: ignore # noqa: F821
        import gc

        gc.collect()


async def auto_refresh_wrapper(
    credential_manager: GitHubCredentialManager,
    credential_id: str,
) -> OAuthTokens:
    """Retrieve credentials, refreshing OAuth tokens when necessary.

    This wrapper automatically refreshes tokens if they are expiring soon.
    Use this for long-running processes that need up-to-date tokens.

    Args:
        credential_manager: Credential manager instance
        credential_id: Credential identifier

    Returns:
        OAuthTokens (refreshed if needed)

    Raises:
        ValueError: If credential is not OAuth with expiration support
    """
    credentials = credential_manager.retrieve_credentials(
        credential_id, auto_refresh=True
    )
    if not isinstance(credentials, OAuthTokens):
        raise ValueError(
            "auto_refresh_wrapper only supports OAuthTokens with expiration tracking"
        )
    return credentials


def detect_token_type(token: str) -> CredentialType:
    """Detect token type from format."""
    token = token.strip()

    # Classic PAT format: ghp_* (40 chars total)
    if token.startswith("ghp_") and len(token) == 40:
        return CredentialType.PERSONAL_ACCESS_TOKEN

    # Fine-grained PAT format: github_pat_* (82+ chars)
    if token.startswith("github_pat_"):
        return CredentialType.PERSONAL_ACCESS_TOKEN

    # OAuth tokens are typically longer and don't follow PAT patterns
    # If it's not a PAT format, assume OAuth token
    return CredentialType.OAUTH_TOKEN


def validate_token_format(token: str, token_type: CredentialType) -> bool:
    """Validate token format."""
    token = token.strip()

    if token_type == CredentialType.PERSONAL_ACCESS_TOKEN:
        # Check for valid PAT formats
        if token.startswith("ghp_") and len(token) == 40:
            return True
        if token.startswith("github_pat_") and len(token) >= 82:
            return True
        return False

    # OAuth tokens have less strict format requirements
    # Just check it's a reasonable length alphanumeric string
    if len(token) < 20:
        return False

    return True


__all__ = [
    "CredentialType",
    "EnterpriseOAuthConfig",
    "GitHubCredential",
    "GitHubCredentialManager",
    "GitHubOAuthConfig",
    "OAuthToken",
    "OAuthTokens",
    "PersonalAccessToken",
    "auto_refresh_wrapper",
    "clear_sensitive_string",
    "detect_token_type",
    "secure_credential_context",
    "validate_token_format",
]
