"""Command line entry points for Futurnal utilities."""

from typer import Typer

from .local_sources import app as sources_app
from ..configuration.cli import config_app, health_app
from ..ingestion.imap.cli import imap_app
from ..ingestion.github.cli import app as github_app
from .orchestrator import orchestrator_app
from .search import search_app
from .privacy import privacy_app
from .cloud_sync import cloud_sync_app
from .chat import chat_app


cli = Typer(help="Futurnal command line tools")
cli.add_typer(sources_app, name="sources")
cli.add_typer(config_app, name="config")
cli.add_typer(health_app, name="health")
cli.add_typer(imap_app, name="imap")
cli.add_typer(github_app, name="github")
cli.add_typer(orchestrator_app, name="orchestrator")
cli.add_typer(search_app, name="search")
cli.add_typer(privacy_app, name="privacy")
cli.add_typer(cloud_sync_app, name="cloud-sync")
cli.add_typer(chat_app, name="chat")

__all__ = ["cli", "sources_app", "config_app", "health_app", "imap_app", "github_app", "orchestrator_app", "search_app", "privacy_app", "cloud_sync_app", "chat_app"]


