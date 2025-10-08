"""CLI commands for managing Futurnal settings and health checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from futurnal.configuration.settings import (
    DEFAULT_CONFIG_PATH,
    Settings,
    SecretStore,
    bootstrap_settings,
    load_settings,
    rotate_secret,
    save_settings,
)
from futurnal.orchestrator.health import collect_health_report


config_app = typer.Typer(help="Manage Futurnal configuration")
health_app = typer.Typer(help="Run health diagnostics")


@config_app.command("init")
def init_config(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config file"),
    neo4j_uri: Optional[str] = typer.Option(None, help="Override Neo4j URI"),
    neo4j_username: Optional[str] = typer.Option(None, help="Override Neo4j username"),
    neo4j_password: Optional[str] = typer.Option(None, help="Override Neo4j password"),
    chroma_path: Optional[Path] = typer.Option(None, help="Override Chroma directory"),
) -> None:
    """Initialize the Futurnal settings file."""

    overrides = {}
    if neo4j_uri:
        overrides.setdefault("workspace", {}).setdefault("storage", {})["neo4j_uri"] = neo4j_uri
    if neo4j_username:
        overrides.setdefault("workspace", {}).setdefault("storage", {})[
            "neo4j_username"
        ] = neo4j_username
    if neo4j_password:
        overrides.setdefault("workspace", {}).setdefault("storage", {})[
            "neo4j_password"
        ] = neo4j_password
    if chroma_path:
        overrides.setdefault("workspace", {}).setdefault("storage", {})[
            "chroma_path"
        ] = str(chroma_path)

    settings = bootstrap_settings(path=config_path, overrides=overrides)
    typer.echo(f"Configuration initialized at {config_path}")
    typer.echo(_summarize_settings(settings))


@config_app.command("show")
def show_config(config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config")) -> None:
    """Display effective configuration with secrets masked."""

    settings = load_settings(config_path)
    typer.echo(_summarize_settings(settings))


@config_app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key, e.g. workspace.storage.neo4j_uri"),
    value: str = typer.Argument(..., help="New value"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config"),
) -> None:
    """Update a configuration value."""

    settings = load_settings(config_path)
    payload = settings.model_dump(mode="python")
    _assign(payload, key.split("."), value)
    updated = Settings.model_validate(payload)
    save_settings(updated, config_path)
    typer.echo(f"Updated {key}")


@config_app.command("validate")
def validate_config(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to config file"),
) -> None:
    """Validate configuration file for correctness."""

    try:
        settings = load_settings(config_path)
        typer.echo(f"✅ Configuration valid at {config_path}")
        typer.echo(f"   Workspace: {settings.workspace.workspace_path}")
        typer.echo(f"   Neo4j URI: {settings.workspace.storage.neo4j_uri}")
        typer.echo(f"   Chroma path: {settings.workspace.storage.chroma_path}")
    except Exception as e:
        typer.echo(f"❌ Configuration invalid: {e}", err=True)
        raise typer.Exit(code=1)


@config_app.command("rotate-secret")
def rotate_secret_command(
    backend: str = typer.Argument(..., help="Backend identifier, e.g. neo4j"),
    identifier: str = typer.Argument(..., help="Account or resource identifier"),
    value: str = typer.Argument(..., help="New secret value"),
    service: str = typer.Option("futurnal", help="Keychain service name"),
) -> None:
    """Rotate a stored secret in the system keychain."""

    store = SecretStore(service_name=service)
    rotate_secret(secret_store=store, backend=backend, identifier=identifier, new_value=value)
    typer.echo(f"Secret for {backend}:{identifier} rotated")


@health_app.command("run")
def run_health(
    workspace_path: Optional[Path] = typer.Option(None, help="Workspace directory"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Config file"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output"),
) -> None:
    """Run health diagnostics for Futurnal services."""

    settings = bootstrap_settings(path=config_path)
    report = collect_health_report(settings=settings, workspace_path=workspace_path)
    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo("Futurnal Health Report")
        for entry in report["checks"]:
            status_icon = "✅" if entry["status"] == "ok" else "⚠️"
            typer.echo(f"- {status_icon} {entry['name']}: {entry['detail']}")


@health_app.command("check")
def check_subsystem(
    subsystem: str = typer.Argument(..., help="Subsystem to check (neo4j, chroma, queue)"),
    workspace_path: Optional[Path] = typer.Option(None, help="Workspace directory"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Config file"),
) -> None:
    """Check health of a specific subsystem."""

    settings = bootstrap_settings(path=config_path)
    report = collect_health_report(settings=settings, workspace_path=workspace_path)

    # Map subsystem names to check names
    subsystem_map = {
        "neo4j": "neo4j_connection",
        "chroma": "chroma_connection",
        "queue": "queue_database",
    }

    check_name = subsystem_map.get(subsystem.lower())
    if not check_name:
        typer.echo(f"❌ Unknown subsystem: {subsystem}", err=True)
        typer.echo(f"Available subsystems: {', '.join(subsystem_map.keys())}")
        raise typer.Exit(code=1)

    # Find the specific check
    for entry in report["checks"]:
        if entry["name"] == check_name:
            status_icon = "✅" if entry["status"] == "ok" else "⚠️"
            typer.echo(f"{status_icon} {subsystem.upper()}: {entry['detail']}")
            if entry["status"] != "ok":
                raise typer.Exit(code=1)
            return

    typer.echo(f"⚠️ No health check found for {subsystem}", err=True)
    raise typer.Exit(code=1)


def _summarize_settings(settings: Settings) -> str:
    data = settings.model_dump(mode="json")
    return json.dumps(data, indent=2)


def _assign(payload: dict, keys: list[str], value: str) -> None:
    current = payload
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value

