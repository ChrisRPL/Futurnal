"""Typed settings management for Futurnal local workspace.

This module wraps user configuration in Pydantic models so CLI commands and
services can rely on validated settings. It also provides a keyring-backed
secret store for persisted credentials, ensuring privacy requirements remain
front and center.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - import optional dependency
    import keyring
except ModuleNotFoundError:  # pragma: no cover - fallback when keyring is unavailable
    class _FallbackKeyringErrors:
        class PasswordDeleteError(Exception):
            """Raised when deleting a missing secret."""

    class _FallbackKeyring:
        """File-backed keyring-compatible interface."""

        errors = _FallbackKeyringErrors

        def __init__(self) -> None:
            self._path = Path.home() / ".futurnal" / "secrets.json"
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if not self._path.exists():
                self._path.write_text(json.dumps({}), encoding="utf-8")
                try:
                    os.chmod(self._path, 0o600)
                except PermissionError:
                    pass

        def _load(self) -> Dict[str, Dict[str, str]]:
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}

        def _save(self, data: Dict[str, Dict[str, str]]) -> None:
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        def set_password(self, service_name: str, key: str, value: str) -> None:
            data = self._load()
            service = data.setdefault(service_name, {})
            service[key] = value
            self._save(data)

        def get_password(self, service_name: str, key: str) -> Optional[str]:
            data = self._load()
            return data.get(service_name, {}).get(key)

        def delete_password(self, service_name: str, key: str) -> None:
            data = self._load()
            service = data.get(service_name, {})
            if key not in service:
                raise self.errors.PasswordDeleteError(f"Secret {service_name}:{key} not found")
            del service[key]
            if not service:
                data.pop(service_name, None)
            self._save(data)

    keyring = _FallbackKeyring()  # type: ignore[assignment]

from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator


DEFAULT_CONFIG_PATH = Path.home() / ".futurnal" / "config.json"
DEFAULT_SECRETS_SERVICE = "futurnal"


class StorageSettings(BaseModel):
    """Configuration for storage backends."""

    neo4j_uri: str = Field(..., description="Neo4j bolt URI")
    neo4j_username: str = Field(..., description="Neo4j username")
    neo4j_password: SecretStr = Field(..., description="Neo4j password")
    neo4j_encrypted: bool = Field(True, description="Enable driver encryption")
    chroma_path: Path = Field(..., description="Chroma persistence directory")
    chroma_auth_token: Optional[SecretStr] = Field(
        default=None, description="Optional auth token for remote Chroma"
    )

    @field_validator("neo4j_uri")
    def _validate_neo4j_uri(cls, value: str) -> str:
        if not value.startswith("bolt://") and not value.startswith("neo4j://"):
            raise ValueError("neo4j_uri must start with bolt:// or neo4j://")
        return value


class SecuritySettings(BaseModel):
    """Security toggles for the local workspace."""

    enable_storage_encryption: bool = Field(True, description="Enable workspace encryption features")
    telemetry_retention_days: int = Field(30, ge=1, le=365)
    audit_retention_days: int = Field(90, ge=30, le=730)
    backup_retention_days: int = Field(30, ge=1, le=365)


class WorkspaceSettings(BaseModel):
    """Top-level workspace configuration."""

    storage: StorageSettings
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    workspace_path: Path = Field(default=Path.home() / ".futurnal" / "workspace")


class Settings(BaseModel):
    """Root configuration state."""

    workspace: WorkspaceSettings


@dataclass
class SecretStore:
    """Keyring abstraction for storing credentials."""

    service_name: str = DEFAULT_SECRETS_SERVICE
    keyring_module: Any = field(default=keyring)

    def set_secret(self, key: str, value: str) -> None:
        if self.keyring_module is None:
            raise RuntimeError("Keyring module not configured")
        self.keyring_module.set_password(self.service_name, key, value)

    def get_secret(self, key: str) -> Optional[str]:
        if self.keyring_module is None:
            return None
        return self.keyring_module.get_password(self.service_name, key)

    def delete_secret(self, key: str) -> None:
        if self.keyring_module is None:
            return
        try:
            self.keyring_module.delete_password(self.service_name, key)
        except AttributeError:  # pragma: no cover - missing delete API
            return
        except Exception as exc:  # pragma: no cover - backend specific
            errors = getattr(self.keyring_module, "errors", None)
            password_error = getattr(errors, "PasswordDeleteError", None)
            if password_error and isinstance(exc, password_error):
                return
            raise


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load settings from disk or raise if invalid."""

    if not path.exists():
        raise FileNotFoundError(f"Settings file not found at {path}")
    payload = json.loads(path.read_text())
    try:
        return Settings.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - exercised via tests
        raise ValueError(f"Invalid configuration: {exc}") from exc


def save_settings(settings: Settings, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Persist settings to disk with masked secrets."""

    payload = settings.model_dump(mode="json")
    payload = _mask_secret_fields(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bootstrap_settings(
    *,
    path: Path = DEFAULT_CONFIG_PATH,
    secret_store: SecretStore | None = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Settings:
    """Create or load settings respecting environment overrides."""

    secret_store = secret_store or SecretStore()
    overrides = overrides or {}

    if path.exists():
        settings = load_settings(path)
    else:
        settings = _default_settings()
        save_settings(settings, path)

    merged = settings.model_dump(mode="python")
    merged = _apply_overrides(merged, overrides)
    merged = _apply_env_overrides(merged)

    resolved = Settings.model_validate(merged)
    _ensure_directories(resolved)
    _hydrate_secrets(resolved, secret_store)
    save_settings(resolved, path)
    return resolved


def _default_settings() -> Settings:
    chroma_dir = Path.home() / ".futurnal" / "workspace" / "vector"
    data = {
        "workspace": {
            "workspace_path": str(Path.home() / ".futurnal" / "workspace"),
            "storage": {
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_username": "neo4j",
                "neo4j_password": "NEO4J_PASSWORD_PLACEHOLDER",
                "neo4j_encrypted": False,
                "chroma_path": str(chroma_dir),
            },
        }
    }
    return Settings.model_validate(data)


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    for key, value in overrides.items():
        merged[key] = value
    return merged


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    storage = data.setdefault("workspace", {}).setdefault("storage", {})
    _set_env_override(storage, "neo4j_uri", "FUTURNAL_NEO4J_URI")
    _set_env_override(storage, "neo4j_username", "FUTURNAL_NEO4J_USERNAME")
    _set_env_override(storage, "neo4j_password", "FUTURNAL_NEO4J_PASSWORD")
    _set_env_override(storage, "neo4j_encrypted", "FUTURNAL_NEO4J_ENCRYPTED", cast_bool=True)
    _set_env_override(storage, "chroma_path", "FUTURNAL_CHROMA_PATH")
    _set_env_override(storage, "chroma_auth_token", "FUTURNAL_CHROMA_AUTH_TOKEN")

    security = data["workspace"].setdefault("security", {})
    _set_env_override(security, "telemetry_retention_days", "FUTURNAL_TELEMETRY_RETENTION", cast_int=True)
    _set_env_override(security, "audit_retention_days", "FUTURNAL_AUDIT_RETENTION", cast_int=True)
    _set_env_override(security, "backup_retention_days", "FUTURNAL_BACKUP_RETENTION", cast_int=True)
    return data


def _set_env_override(
    mapping: Dict[str, Any],
    key: str,
    env_name: str,
    *,
    cast_bool: bool = False,
    cast_int: bool = False,
) -> None:
    raw = os.getenv(env_name)
    if raw is None:
        return
    if cast_bool:
        mapping[key] = raw.lower() in {"1", "true", "yes"}
    elif cast_int:
        mapping[key] = int(raw)
    else:
        mapping[key] = raw


def _ensure_directories(settings: Settings) -> None:
    settings.workspace.workspace_path.mkdir(parents=True, exist_ok=True)
    settings.workspace.storage.chroma_path.mkdir(parents=True, exist_ok=True)


def _hydrate_secrets(settings: Settings, secret_store: SecretStore) -> None:
    storage = settings.workspace.storage
    neo4j_key = _secret_key("neo4j", storage.neo4j_username)
    password = secret_store.get_secret(neo4j_key)
    if not password or password == "NEO4J_PASSWORD_PLACEHOLDER":
        secret_store.set_secret(neo4j_key, storage.neo4j_password.get_secret_value())
    else:
        storage.neo4j_password = SecretStr(password)

    if storage.chroma_auth_token:
        chroma_key = _secret_key("chroma", storage.neo4j_username)
        existing = secret_store.get_secret(chroma_key)
        if not existing or existing == storage.chroma_auth_token.get_secret_value():
            secret_store.set_secret(chroma_key, storage.chroma_auth_token.get_secret_value())
        else:
            storage.chroma_auth_token = SecretStr(existing)


def rotate_secret(
    *,
    secret_store: SecretStore,
    backend: str,
    identifier: str,
    new_value: str,
) -> None:
    """Rotate a stored secret."""

    secret_key = _secret_key(backend, identifier)
    secret_store.set_secret(secret_key, new_value)


def _secret_key(backend: str, identifier: str) -> str:
    return f"{backend}:{identifier}"


def _mask_secret_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    storage = payload.get("workspace", {}).get("storage", {})
    if "neo4j_password" in storage:
        storage["neo4j_password"] = "***"
    if storage.get("chroma_auth_token"):
        storage["chroma_auth_token"] = "***"
    return payload

