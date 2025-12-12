"""Search CLI commands.

Provides search functionality for the Futurnal desktop app.

Commands:
    futurnal search query "your query" --top-k 10 --json
    futurnal search history --limit 50 --json
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from futurnal.search.api import create_hybrid_search_api

logger = logging.getLogger(__name__)

search_app = typer.Typer(help="Search commands for knowledge graph")


def _get_history_file() -> Path:
    """Get path to search history file."""
    return Path.home() / ".futurnal" / "search_history.json"


def _load_history() -> list:
    """Load search history from disk."""
    history_file = _get_history_file()
    if history_file.exists():
        try:
            return json.loads(history_file.read_text())
        except Exception:
            return []
    return []


def _save_history_item(query: str, result_count: int) -> None:
    """Save a search query to history."""
    history = _load_history()
    history.insert(0, {
        "id": f"search_{len(history) + 1}",
        "query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "result_count": result_count,
    })
    # Keep last 100 queries
    history = history[:100]

    history_file = _get_history_file()
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.write_text(json.dumps(history, indent=2))


@search_app.command("query")
def search_query(
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    entity_types: Optional[str] = typer.Option(
        None, "--entity-types", "-e", help="Filter by entity types (comma-separated)"
    ),
) -> None:
    """Execute a search query against the knowledge graph."""
    try:
        # Create search API
        api = create_hybrid_search_api()

        # Parse filters
        filters = {}
        if entity_types:
            filters["entity_types"] = [t.strip() for t in entity_types.split(",")]

        # Execute search
        results = asyncio.run(api.search(query, top_k=top_k, filters=filters or None))

        # Save to history
        _save_history_item(query, len(results))

        # Build response
        response = {
            "results": results,
            "total": len(results),
            "query_id": f"q_{hash(query) % 100000}",
            "intent": {
                "primary": api.last_strategy or "exploratory",
                "temporal": None,
                "causal": api.last_strategy == "causal",
            },
            "execution_time_ms": 0,  # Will be populated by actual timing
        }

        if output_json:
            print(json.dumps(response))
        else:
            # Human-readable output
            typer.echo(f"\nSearch: '{query}'")
            typer.echo(f"Found {len(results)} results:\n")

            for i, result in enumerate(results, 1):
                entity_type = result.get("entity_type", "Unknown")
                score = result.get("score", 0)
                content = result.get("content", "")[:100]
                label = result.get("metadata", {}).get("label", result.get("id", ""))

                typer.echo(f"{i}. [{entity_type}] {label}")
                typer.echo(f"   Score: {score:.2f}")
                typer.echo(f"   {content}...")
                typer.echo()

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e), "results": [], "total": 0}))
            sys.exit(1)
        else:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)


@search_app.command("history")
def search_history(
    limit: int = typer.Option(50, "--limit", "-l", help="Number of history items"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Get search history."""
    try:
        history = _load_history()[:limit]

        if output_json:
            print(json.dumps(history))
        else:
            typer.echo(f"\nRecent searches ({len(history)}):\n")
            for item in history:
                typer.echo(f"  [{item['timestamp'][:10]}] {item['query']} ({item['result_count']} results)")

    except Exception as e:
        if output_json:
            print(json.dumps([]))
        else:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)


@search_app.command("clear-history")
def clear_history() -> None:
    """Clear search history."""
    history_file = _get_history_file()
    if history_file.exists():
        history_file.unlink()
        typer.echo("Search history cleared.")
    else:
        typer.echo("No history to clear.")
