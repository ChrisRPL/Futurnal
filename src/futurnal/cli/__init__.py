"""Command line entry points for Futurnal utilities."""

from typer import Typer

from .local_sources import app as sources_app
from ..configuration.cli import config_app, health_app


cli = Typer(help="Futurnal command line tools")
cli.add_typer(sources_app, name="sources")
cli.add_typer(config_app, name="config")
cli.add_typer(health_app, name="health")

__all__ = ["cli", "sources_app", "config_app", "health_app"]


