"""Tests for the IMAP credential manager implementation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pytest

from futurnal.ingestion.imap.credential_manager import (
    AppPassword,
    CredentialManager,
    CredentialType,
    ImapCredential,
    OAuth2Tokens,
    OAuthProvider,
    OAuthProviderRegistry,
    auto_refresh_wrapper,
    clear_sensitive_string,
    secure_credential_context,
)


class _InMemoryKeyring:
    """Simplified in-memory keyring for tests."""

    def __init__(self) -> None:
        self.values: Dict[str, Dict[str, str]] = {}

    def set_password(self, service: str, key: str, value: str) -> None:
        self.values.setdefault(service, {})[key] = value

    def get_password(self, service: str, key: str) -> Optional[str]:
        return self.values.get(service, {}).get(key)

    def delete_password(self, service: str, key: str) -> None:
        self.values.get(service, {}).pop(key, None)


class _StubAuditLogger:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)


class _DummyRegistry(OAuthProviderRegistry):
    def __init__(self) -> None:
        super().__init__(providers=[])
        provider = OAuthProvider(
            name="dummy",
            authorization_endpoint="https://example.com/auth",
            token_endpoint="https://example.com/token",
            scopes=["imap"],
            client_id="client-id",
            client_secret="secret",
        )
        self.register(provider)
        self._mock_tokens: Dict[str, Dict[str, Any]] = {}

    def set_mock_response(self, token_endpoint: str, response: Dict[str, Any]) -> None:
        self._mock_tokens[token_endpoint] = response

    def _post(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        try:
            return self._mock_tokens[url]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected POST to {url}") from exc


@pytest.fixture()
def keyring() -> _InMemoryKeyring:
    return _InMemoryKeyring()


def _manager(tmp_path, keyring) -> CredentialManager:
    registry = _DummyRegistry()
    registry.set_mock_response(
        "https://example.com/token",
        {
            "access_token": "refreshed",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": "imap",
        },
    )
    return CredentialManager(
        audit_logger=_StubAuditLogger(),
        oauth_provider_registry=registry,
        keyring_module=keyring,
        metadata_path=tmp_path / "imap.json",
        refresh_buffer_seconds=60,
        _now=lambda tz=None: datetime.now(timezone.utc),
    )


def test_store_and_retrieve_oauth_tokens(tmp_path, keyring) -> None:
    manager = _manager(tmp_path, keyring)
    tokens = OAuth2Tokens(
        access_token="token",
        refresh_token="refresh",
        expires_in=120,
        scope=["imap"],
    )

    credential = manager.store_oauth_tokens(
        credential_id="cred", email_address="user@example.com", tokens=tokens, provider="dummy"
    )

    assert credential.credential_type == CredentialType.OAUTH2
    assert json.loads(keyring.get_password(manager.service_name, "cred"))["access_token"] == "token"

    retrieved = manager.retrieve_credentials("cred")
    assert isinstance(retrieved, OAuth2Tokens)
    assert retrieved.access_token == "token"


def test_auto_refresh_on_retrieve(tmp_path, keyring) -> None:
    manager = _manager(tmp_path, keyring)
    soon_to_expire = OAuth2Tokens(
        access_token="old",
        refresh_token="refresh-token",
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
        scope=["imap"],
    )
    manager.store_oauth_tokens(
        credential_id="cred",
        email_address="user@example.com",
        tokens=soon_to_expire,
        provider="dummy",
    )

    refreshed = manager.retrieve_credentials("cred")
    assert isinstance(refreshed, OAuth2Tokens)
    assert refreshed.access_token == "refreshed"


def test_store_and_retrieve_app_password(tmp_path, keyring) -> None:
    manager = _manager(tmp_path, keyring)
    credential = manager.store_app_password(
        credential_id="cred",
        email_address="user@example.com",
        password="secret",
    )
    assert credential.credential_type == CredentialType.APP_PASSWORD

    retrieved = manager.retrieve_credentials("cred")
    assert isinstance(retrieved, AppPassword)
    assert retrieved.password == "secret"


def test_delete_credentials_removes_entries(tmp_path, keyring) -> None:
    manager = _manager(tmp_path, keyring)
    manager.store_app_password(
        credential_id="cred",
        email_address="user@example.com",
        password="secret",
    )
    manager.delete_credentials("cred")
    assert keyring.get_password(manager.service_name, "cred") is None
    assert manager.list_credentials() == []


def test_list_credentials_returns_metadata(tmp_path, keyring) -> None:
    manager = _manager(tmp_path, keyring)
    manager.store_app_password(
        credential_id="cred",
        email_address="user@example.com",
        password="secret",
    )
    items = manager.list_credentials()
    assert len(items) == 1
    assert isinstance(items[0], ImapCredential)


@pytest.mark.asyncio()
async def test_auto_refresh_wrapper_refreshes(tmp_path, keyring) -> None:
    manager = _manager(tmp_path, keyring)
    soon_to_expire = OAuth2Tokens(
        access_token="old",
        refresh_token="refresh-token",
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
        scope=["imap"],
    )
    manager.store_oauth_tokens(
        credential_id="cred",
        email_address="user@example.com",
        tokens=soon_to_expire,
        provider="dummy",
    )
    refreshed = await auto_refresh_wrapper(manager, "cred")
    assert refreshed.access_token == "refreshed"


def test_secure_context_clears_strings(tmp_path, keyring) -> None:
    manager = _manager(tmp_path, keyring)
    manager.store_app_password(
        credential_id="cred",
        email_address="user@example.com",
        password="secret",
    )
    with secure_credential_context(manager, "cred") as credentials:
        assert isinstance(credentials, AppPassword)
        assert credentials.password == "secret"
    assert keyring.get_password(manager.service_name, "cred") is not None


def test_clear_sensitive_string_no_error() -> None:
    clear_sensitive_string("value")


