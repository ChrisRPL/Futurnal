"""Command line entry points for Futurnal utilities."""

from typer import Typer

from .local_sources import app as sources_app
from ..configuration.cli import config_app, health_app
from ..ingestion.imap.cli import imap_app
from .orchestrator import orchestrator_app


cli = Typer(help="Futurnal command line tools")
cli.add_typer(sources_app, name="sources")
cli.add_typer(config_app, name="config")
cli.add_typer(health_app, name="health")
cli.add_typer(imap_app, name="imap")
cli.add_typer(orchestrator_app, name="orchestrator")

__all__ = ["cli", "sources_app", "config_app", "health_app", "imap_app", "orchestrator_app"]


