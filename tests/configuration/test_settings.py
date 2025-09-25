"""Tests for Futurnal configuration settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from futurnal.configuration.settings import (
    Settings,
    SecretStore,
    bootstrap_settings,
    load_settings,
    rotate_secret,
    save_settings,
)


class InMemorySecretStore(SecretStore):
    def __init__(self) -> None:
        super().__init__(service_name="test", keyring_module=None)
        self.storage: Dict[str, str] = {}

    def set_secret(self, key: str, value: str) -> None:  # type: ignore[override]
        self.storage[key] = value

    def get_secret(self, key: str) -> str | None:  # type: ignore[override]
        return self.storage.get(key)

    def delete_secret(self, key: str) -> None:  # type: ignore[override]
        self.storage.pop(key, None)


def test_bootstrap_creates_default_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    secret_store = InMemorySecretStore()
    settings = bootstrap_settings(path=config_path, secret_store=secret_store)

    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["workspace"]["storage"]["neo4j_uri"] == "bolt://localhost:7687"
    assert secret_store.get_secret("neo4j:neo4j") == settings.workspace.storage.neo4j_password.get_secret_value()


def test_load_settings_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    settings = Settings.model_validate(
        {
            "workspace": {
                "workspace_path": str(tmp_path / "workspace"),
                "storage": {
                    "neo4j_uri": "bolt://localhost:7687",
                    "neo4j_username": "neo4j",
                    "neo4j_password": "secret",
                    "chroma_path": str(tmp_path / "workspace" / "vector"),
                },
            }
        }
    )
    save_settings(settings, config_path)

    loaded = load_settings(config_path)
    assert loaded.workspace.storage.neo4j_uri == "bolt://localhost:7687"
    assert loaded.workspace.storage.neo4j_password.get_secret_value() == "***"


def test_rotate_secret_updates_store() -> None:
    store = InMemorySecretStore()
    rotate_secret(secret_store=store, backend="neo4j", identifier="primary", new_value="updated")
    assert store.get_secret("neo4j:primary") == "updated"

