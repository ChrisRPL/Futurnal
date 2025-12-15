"""Search CLI commands.

Provides search functionality for the Futurnal desktop app.

Commands:
    futurnal search query "your query" --top-k 10 --json
    futurnal search answer "query" --model llama3.1:8b-instruct-q4_0 --json
    futurnal search history --limit 50 --json

Step 02: LLM Answer Generation
Research Foundation:
- CausalRAG (ACL 2025): Causal-aware generation
- LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach
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


@search_app.command("answer")
def search_with_answer(
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    no_answer: bool = typer.Option(False, "--no-answer", help="Skip answer generation"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model for answer generation"
    ),
    entity_types: Optional[str] = typer.Option(
        None, "--entity-types", "-e", help="Filter by entity types (comma-separated)"
    ),
) -> None:
    """Execute search with LLM-generated answer synthesis.

    Research Foundation:
    - CausalRAG (ACL 2025): Causal-aware generation
    - LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach

    Available models (per LLM_MODEL_REGISTRY.md):
    - phi3:mini (4GB, fast)
    - llama3.1:8b-instruct-q4_0 (8GB, balanced - default)
    - bielik:4.5b-v3-instruct (5GB, Polish)
    - kimi-k2:thinking (16GB, reasoning)
    - qwen2.5:32b-instruct-q4_0 (16GB, quality)

    Examples:
        futurnal search answer "What is Python?" --json
        futurnal search answer "What happened yesterday?" --model phi3:mini
        futurnal search answer "Query" --no-answer  # Skip generation
    """
    try:
        # Create search API
        api = create_hybrid_search_api()

        # Parse filters
        filters = {}
        if entity_types:
            filters["entity_types"] = [t.strip() for t in entity_types.split(",")]

        # Execute search with answer
        response = asyncio.run(
            api.search_with_answer(
                query,
                top_k=top_k,
                filters=filters or None,
                generate_answer=not no_answer,
                model=model,
            )
        )

        # Save to history
        _save_history_item(query, len(response.get("results", [])))

        # Add query_id and total for compatibility
        response["query_id"] = f"q_{hash(query) % 100000}"
        response["total"] = len(response.get("results", []))

        if output_json:
            print(json.dumps(response))
        else:
            # Human-readable output
            typer.echo(f"\nSearch: '{query}'")

            # Show answer if generated
            if response.get("answer"):
                typer.echo("\n=== Answer ===")
                typer.echo(response["answer"])

                if response.get("sources"):
                    typer.echo("\n=== Sources ===")
                    for src in response["sources"]:
                        typer.echo(f"  - {src}")

                if response.get("model"):
                    typer.echo(f"\n(Generated by {response['model']})")

            # Show results
            results = response.get("results", [])
            typer.echo(f"\n=== Results ({len(results)}) ===")

            for i, result in enumerate(results[:5], 1):  # Show top 5
                entity_type = result.get("entity_type", "Unknown")
                score = result.get("score", 0)
                label = result.get("metadata", {}).get("label", result.get("id", ""))

                typer.echo(f"{i}. [{entity_type}] {label} (score: {score:.2f})")

            if len(results) > 5:
                typer.echo(f"\n... and {len(results) - 5} more results")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e), "results": [], "total": 0, "answer": None}))
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


@search_app.command("index")
def index_documents(
    workspace_path: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    neo4j_uri: str = typer.Option("bolt://localhost:7687", "--neo4j-uri", help="Neo4j connection URI"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Index all parsed documents to ChromaDB and Neo4j for GraphRAG search.

    This command indexes documents from the workspace to enable hybrid search:
    - ChromaDB: Vector embeddings for semantic similarity search
    - Neo4j: Graph nodes and relationships for graph traversal

    Run this after syncing data sources to make content searchable via GraphRAG.

    Examples:
        futurnal search index                    # Index with default settings
        futurnal search index --json             # Output as JSON
        futurnal search index --neo4j-uri bolt://myhost:7687
    """
    from pathlib import Path
    from futurnal.pipeline.kg_indexer import KnowledgeGraphIndexer

    workspace = Path(workspace_path) if workspace_path else Path.home() / ".futurnal" / "workspace"

    if not output_json:
        typer.echo(f"Indexing documents from {workspace}...")
        typer.echo("  Connecting to ChromaDB and Neo4j...")

    try:
        indexer = KnowledgeGraphIndexer(
            workspace_dir=workspace,
            neo4j_uri=neo4j_uri,
        )

        stats = indexer.index_all_parsed()

        if output_json:
            print(json.dumps(stats))
        else:
            typer.echo(f"\nIndexing complete:")
            typer.echo(f"  Documents found: {stats.get('documents_found', 0)}")
            typer.echo(f"  ChromaDB indexed: {stats.get('chroma_indexed', 0)}")
            typer.echo(f"  Neo4j indexed: {stats.get('neo4j_indexed', 0)}")
            typer.echo(f"  Entities indexed: {stats.get('entities_indexed', 0)}")
            if stats.get('errors', 0) > 0:
                typer.echo(f"  Errors: {stats.get('errors', 0)}")

        indexer.close()

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e), "documents_found": 0}))
            sys.exit(1)
        else:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)


@search_app.command("status")
def index_status(
    workspace_path: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    neo4j_uri: str = typer.Option("bolt://localhost:7687", "--neo4j-uri", help="Neo4j connection URI"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Check the status of the knowledge graph indexes.

    Shows the number of documents indexed in ChromaDB and Neo4j.
    """
    from pathlib import Path
    from futurnal.pipeline.kg_indexer import KnowledgeGraphIndexer

    workspace = Path(workspace_path) if workspace_path else Path.home() / ".futurnal" / "workspace"

    try:
        indexer = KnowledgeGraphIndexer(
            workspace_dir=workspace,
            neo4j_uri=neo4j_uri,
        )

        stats = indexer.get_index_stats()

        if output_json:
            print(json.dumps(stats))
        else:
            typer.echo("Knowledge Graph Index Status:")
            typer.echo(f"\nChromaDB:")
            typer.echo(f"  Connected: {stats['chromadb']['connected']}")
            typer.echo(f"  Documents: {stats['chromadb']['document_count']}")
            typer.echo(f"\nNeo4j:")
            typer.echo(f"  Connected: {stats['neo4j']['connected']}")
            typer.echo(f"  Nodes: {stats['neo4j']['node_count']}")
            typer.echo(f"  Relationships: {stats['neo4j']['relationship_count']}")

        indexer.close()

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)
        else:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
