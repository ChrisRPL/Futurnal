"""CLI tests for Obsidian vault management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from futurnal.cli.local_sources import app


runner = CliRunner()


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    return vault


def test_obsidian_add_list_show_remove(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # add
    add_res = runner.invoke(
        app,
        [
            "obsidian",
            "add",
            "--path",
            str(vault),
            "--name",
            "Notes",
            "--registry-path",
            str(registry),
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    vault_id = payload["id"]
    assert (registry / f"{vault_id}.json").exists()

    # list
    list_res = runner.invoke(app, ["obsidian", "list", "--json", "--registry-path", str(registry)])
    assert list_res.exit_code == 0
    items = json.loads(list_res.output)
    assert items and items[0]["id"] == vault_id

    # show
    show_res = runner.invoke(app, ["obsidian", "show", vault_id, "--registry-path", str(registry)])
    assert show_res.exit_code == 0
    shown = json.loads(show_res.output)
    assert shown["id"] == vault_id

    # remove
    rm_res = runner.invoke(app, ["obsidian", "remove", vault_id, "--registry-path", str(registry)])
    assert rm_res.exit_code == 0
    assert not list(registry.glob("*.json"))


def test_obsidian_to_local_source_conversion(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # Add vault
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Test Vault",
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    vault_id = payload["id"]

    # Convert to local source
    convert_res = runner.invoke(
        app,
        [
            "obsidian", "to-local-source", vault_id,
            "--registry-path", str(registry),
            "--max-workers", "2",
            "--schedule", "0 */12 * * *",
            "--priority", "high",
        ],
    )
    assert convert_res.exit_code == 0
    local_source = json.loads(convert_res.output)
    assert local_source["name"] == "Test Vault"
    assert local_source["max_workers"] == 2
    assert local_source["schedule"] == "0 */12 * * *"
    assert local_source["priority"] == "high"
    assert local_source["require_external_processing_consent"] is True


def test_obsidian_network_warning_display(tmp_path: Path, monkeypatch) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"
    
    # Mock network mount detection to always return a warning
    def mock_detect_network_mount(path):
        return "Test warning: network mount detected"
    
    monkeypatch.setattr(
        "futurnal.ingestion.obsidian.descriptor._detect_network_mount",
        mock_detect_network_mount
    )

    # Add vault - should show warning
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    assert "WARNING: Test warning: network mount detected" in add_res.stderr


