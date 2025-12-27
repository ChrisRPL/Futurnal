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
from .multimodal import multimodal_app
from .causal import causal_app
from .activity import activity_app
from .schema import schema_app
from .learning import learning_app
from .insights import insights_app
from .papers import papers_app
from .infrastructure import infrastructure_app
from .research import research_app
from .notifications import notifications_app
from .agents import agents_app


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
cli.add_typer(multimodal_app, name="multimodal")
cli.add_typer(causal_app, name="causal")
cli.add_typer(activity_app, name="activity")
cli.add_typer(schema_app, name="schema")
cli.add_typer(learning_app, name="learning")
cli.add_typer(insights_app, name="insights")
cli.add_typer(papers_app, name="papers")
cli.add_typer(infrastructure_app, name="infrastructure")
cli.add_typer(research_app, name="research")
cli.add_typer(notifications_app, name="notifications")
cli.add_typer(agents_app, name="agents")

__all__ = ["cli", "sources_app", "config_app", "health_app", "imap_app", "github_app", "orchestrator_app", "search_app", "privacy_app", "cloud_sync_app", "chat_app", "multimodal_app", "causal_app", "activity_app", "schema_app", "learning_app", "insights_app", "papers_app", "infrastructure_app", "research_app", "notifications_app", "agents_app"]


