"""CLI utilities for managing local ingestion sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..ingestion.local.config import LocalIngestionSource, load_config_from_dict

app = typer.Typer(help="Manage Futurnal local data sources")

DEFAULT_CONFIG_PATH = Path.home() / ".futurnal" / "sources.json"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {"sources": []}
    return json.loads(path.read_text())


def _save_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


@app.command("register")
def register_local_source(
    name: str = typer.Option(..., help="Unique name for the source"),
    root: Path = typer.Option(..., exists=True, file_okay=False, help="Root directory to ingest"),
    include: Optional[str] = typer.Option(None, help="Comma-separated glob include patterns"),
    exclude: Optional[str] = typer.Option(None, help="Comma-separated glob exclude patterns"),
    ignore_file: Optional[Path] = typer.Option(None, help="Custom ignore file path"),
    follow_symlinks: bool = typer.Option(False, help="Follow symlinks when scanning"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Register or update a local directory source."""

    config = _load_config(config_path)
    include_list = include.split(",") if include else []
    exclude_list = exclude.split(",") if exclude else []

    source_dict = {
        "name": name,
        "root_path": str(root),
        "include": include_list,
        "exclude": exclude_list,
        "follow_symlinks": follow_symlinks,
        "ignore_file": str(ignore_file) if ignore_file else None,
    }

    # Validate source configuration
    LocalIngestionSource(**source_dict)

    sources = [src for src in config.get("sources", []) if src.get("name") != name]
    sources.append(source_dict)
    _save_config(config_path, {"sources": sources})
    typer.echo(f"Registered source '{name}' at {root}")


@app.command("list")
def list_sources(config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config")) -> None:
    """List configured local sources."""

    config = _load_config(config_path)
    load_config_from_dict(config)  # Validate schema
    for source in config.get("sources", []):
        typer.echo(f"- {source['name']}: {source['root_path']}")


@app.command("remove")
def remove_source(
    name: str = typer.Argument(..., help="Name of the source to remove"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to sources config"),
) -> None:
    """Remove a configured local source."""

    config = _load_config(config_path)
    sources = [src for src in config.get("sources", []) if src.get("name") != name]
    if len(sources) == len(config.get("sources", [])):
        typer.echo(f"Source '{name}' not found")
        raise typer.Exit(code=1)
    _save_config(config_path, {"sources": sources})
    typer.echo(f"Removed source '{name}'")


if __name__ == "__main__":
    app()


