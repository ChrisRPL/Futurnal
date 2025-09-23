"""Tests for local source CLI utilities."""

from pathlib import Path
from typing import Dict

import json
import typer
from typer.testing import CliRunner

from futurnal.cli.local_sources import app, DEFAULT_CONFIG_PATH

runner = CliRunner()


def read_config(path: Path) -> Dict:
    return json.loads(path.read_text())


def test_register_and_list(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "sources.json"
    root = tmp_path / "vault"
    root.mkdir()

    result = runner.invoke(
        app,
        [
            "register",
            "--name",
            "docs",
            "--root",
            str(root),
            "--include",
            "**/*.md",
            "--config-path",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    config = read_config(config_path)
    assert config["sources"][0]["name"] == "docs"

    list_result = runner.invoke(app, ["list", "--config-path", str(config_path)])
    assert "docs" in list_result.output


def test_remove_source(tmp_path: Path) -> None:
    config_path = tmp_path / "sources.json"
    root = tmp_path / "data"
    root.mkdir()

    runner.invoke(
        app,
        [
            "register",
            "--name",
            "data",
            "--root",
            str(root),
            "--config-path",
            str(config_path),
        ],
    )
    result = runner.invoke(app, ["remove", "data", "--config-path", str(config_path)])
    assert result.exit_code == 0
    config = read_config(config_path)
    assert config["sources"] == []


