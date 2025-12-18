"""Papers CLI commands.

Phase D: Academic Paper Agent

Commands for searching, downloading, and managing academic papers.

Usage:
    futurnal papers search "causal inference" --limit 10
    futurnal papers download <paper_id> --pdf-url <url>
    futurnal papers recommend <paper_id>
    futurnal papers get <paper_id>
    futurnal papers ingest --all
    futurnal papers status <paper_id>
"""

import asyncio
import json
import os
import sys
from typing import List, Optional

from typer import Argument, Option, Typer

from ..agents.paper_search.agent import PaperSearchAgent
from ..agents.paper_search.models import PaperMetadata
from ..agents.paper_search.agentic_search import AgenticPaperSearchAgent
from ..agents.paper_search.multi_provider import MultiProviderSearch, Provider


def _get_email() -> Optional[str]:
    """Get email for API polite pools (OpenAlex, CrossRef)."""
    return os.environ.get("FUTURNAL_EMAIL")


def _get_semantic_scholar_api_key() -> Optional[str]:
    """Get Semantic Scholar API key from environment.

    Checks for SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY environment variables.
    Get a free API key at: https://www.semanticscholar.org/product/api
    """
    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")

papers_app = Typer(help="Academic paper search and management")


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def _convert_keys_to_camel_case(data):
    """Recursively convert dictionary keys from snake_case to camelCase."""
    if isinstance(data, dict):
        return {_to_camel_case(k): _convert_keys_to_camel_case(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_keys_to_camel_case(item) for item in data]
    else:
        return data


def _get_agent() -> PaperSearchAgent:
    """Get paper search agent instance."""
    api_key = _get_semantic_scholar_api_key()
    return PaperSearchAgent(semantic_scholar_key=api_key)


def _run_async(coro):
    """Run async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


@papers_app.command("search")
def search_papers(
    query: str,
    limit: int = Option(10, "--limit", "-l", help="Maximum number of results"),
    year_from: Optional[int] = Option(None, "--year-from", help="Filter papers from this year"),
    year_to: Optional[int] = Option(None, "--year-to", help="Filter papers until this year"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search for academic papers.

    Searches Semantic Scholar for papers matching the query.
    Returns metadata including title, authors, year, citations, and PDF links.

    Examples:
        futurnal papers search "graph neural networks"
        futurnal papers search "causal inference" --limit 5 --year-from 2020
    """
    try:
        agent = _get_agent()
        results = _run_async(
            agent.search(
                query=query,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
            )
        )

        if output_json:
            output = {
                "success": True,
                "query": results.query,
                "total": results.total_results,
                "papers": [
                    {
                        "paperId": p.paper_id,
                        "title": p.title,
                        "authors": [{"name": a.name, "authorId": a.author_id} for a in p.authors],
                        "year": p.year,
                        "abstractText": p.abstract,
                        "venue": p.venue,
                        "citationCount": p.citation_count,
                        "pdfUrl": p.pdf_url,
                        "semanticScholarUrl": p.source_url,
                        "doi": p.doi,
                        "arxivId": p.arxiv_id,
                        "fieldsOfStudy": p.fields_of_study,
                    }
                    for p in results.papers
                ],
                "searchTimeMs": results.search_time_ms,
            }
            print(json.dumps(output))
        else:
            print(f"\nFound {results.total_results} papers for '{query}':\n")
            for i, paper in enumerate(results.papers, 1):
                pdf_marker = "📄" if paper.pdf_url else "  "
                print(f"{i}. {pdf_marker} {paper.title}")
                print(f"   {paper.short_authors} ({paper.year or 'N/A'}) | Citations: {paper.citation_count}")
                if paper.venue:
                    print(f"   Venue: {paper.venue}")
                print(f"   ID: {paper.paper_id}")
                print()

    except Exception as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


@papers_app.command("agentic-search")
def agentic_search_papers(
    query: str,
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Intelligent agentic paper search with query understanding.

    This command uses an LLM-powered agent that:
    1. Analyzes your query to understand intent
    2. Tries multiple search strategies (synonyms, expansions)
    3. Scores each paper's relevance to your specific need
    4. Synthesizes results and suggests follow-up searches

    Examples:
        futurnal papers agentic-search "zero shot object detection SOTA"
        futurnal papers agentic-search "transformer architecture survey"

    Uses multiple providers (OpenAlex, arXiv) by default for reliable results.
    Set FUTURNAL_EMAIL env var for faster API responses (polite pools).
    """
    try:
        # Use multi-provider search (OpenAlex + arXiv by default)
        email = _get_email()
        s2_api_key = _get_semantic_scholar_api_key()

        # Build list of providers
        providers = [Provider.OPENALEX, Provider.ARXIV]
        if s2_api_key:
            providers.append(Provider.SEMANTIC_SCHOLAR)

        search_client = MultiProviderSearch(
            providers=providers,
            email=email,
            semantic_scholar_key=s2_api_key,
        )
        agent = AgenticPaperSearchAgent(search_client=search_client)
        result = _run_async(agent.search(query))

        if output_json:
            output = {
                "success": True,
                "query": result.query,
                "totalEvaluated": result.total_papers_evaluated,
                "papers": [
                    {
                        "paperId": sp.paper.paper_id,
                        "title": sp.paper.title,
                        "authors": [{"name": a.name, "authorId": a.author_id} for a in sp.paper.authors],
                        "year": sp.paper.year,
                        "abstractText": sp.paper.abstract,
                        "venue": sp.paper.venue,
                        "citationCount": sp.paper.citation_count,
                        "pdfUrl": sp.paper.pdf_url,
                        "sourceUrl": sp.paper.source_url,
                        "relevanceScore": sp.relevance_score,
                        "rationale": sp.rationale,
                    }
                    for sp in result.papers
                ],
                "synthesis": result.synthesis,
                "suggestions": result.suggestions,
                "strategiesTried": [
                    {"query": s.query, "type": s.strategy_type, "rationale": s.rationale}
                    for s in result.strategies_tried
                ],
                "searchTimeMs": result.search_time_ms,
            }
            print(json.dumps(output))
        else:
            print(f"\n{'='*60}")
            print(f"AGENTIC PAPER SEARCH: '{query}'")
            print(f"{'='*60}\n")

            # Show strategies tried
            print(f"Strategies tried: {len(result.strategies_tried)}")
            for s in result.strategies_tried[:3]:
                print(f"  • {s.strategy_type}: {s.query}")

            print(f"\nPapers evaluated: {result.total_papers_evaluated}")
            print(f"Relevant papers found: {len(result.papers)}\n")

            # Show synthesis
            print(f"SYNTHESIS:\n{result.synthesis}\n")

            # Show papers
            if result.papers:
                print("RELEVANT PAPERS:")
                print("-" * 40)
                for i, sp in enumerate(result.papers[:10], 1):
                    relevance_bar = "█" * int(sp.relevance_score * 10) + "░" * (10 - int(sp.relevance_score * 10))
                    pdf_marker = "📄" if sp.paper.pdf_url else "  "
                    print(f"\n{i}. {pdf_marker} {sp.paper.title}")
                    print(f"   Relevance: [{relevance_bar}] {sp.relevance_score:.0%}")
                    print(f"   {sp.paper.short_authors} ({sp.paper.year or 'N/A'}) | Citations: {sp.paper.citation_count}")
                    if sp.rationale:
                        print(f"   Why: {sp.rationale}")
                    print(f"   ID: {sp.paper.paper_id}")

            # Show suggestions
            if result.suggestions:
                print(f"\nSUGGESTIONS:")
                for suggestion in result.suggestions:
                    print(f"  → {suggestion}")

            print(f"\n({result.search_time_ms}ms)")

    except Exception as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


@papers_app.command("download")
def download_paper(
    paper_id: str,
    pdf_url: str = Option(..., "--pdf-url", help="Direct URL to PDF"),
    title: Optional[str] = Option(None, "--title", help="Paper title for filename"),
    year: Optional[int] = Option(None, "--year", help="Publication year"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Download a paper PDF.

    Downloads a paper PDF to ~/.futurnal/papers/{year}/{title}.pdf

    Examples:
        futurnal papers download abc123 --pdf-url "https://example.com/paper.pdf"
        futurnal papers download abc123 --pdf-url "..." --title "My Paper" --year 2024
    """
    try:
        agent = _get_agent()

        # Create minimal paper metadata for download
        paper = PaperMetadata(
            paper_id=paper_id,
            title=title or paper_id,
            year=year,
            pdf_url=pdf_url,
        )

        downloaded = _run_async(agent.download_paper(paper))

        if downloaded:
            # Save to state store for later ingestion
            from ..agents.paper_search.state_store import get_default_state_store
            state_store = get_default_state_store()
            state_store.mark_downloaded(
                paper_id=paper_id,
                title=title or paper_id,
                local_path=downloaded.local_path,
                file_size_bytes=downloaded.file_size_bytes,
            )
            state_store.close()

            if output_json:
                output = {
                    "success": True,
                    "downloaded": {
                        "paperId": paper_id,
                        "title": title or paper_id,
                        "localPath": downloaded.local_path,
                        "fileSizeBytes": downloaded.file_size_bytes,
                    },
                }
                print(json.dumps(output))
            else:
                print(f"Downloaded: {downloaded.local_path}")
                print(f"Size: {downloaded.file_size_bytes / 1024:.1f} KB")
        else:
            if output_json:
                print(json.dumps({"success": False, "error": "Download failed", "paperId": paper_id}))
            else:
                print("Download failed", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e), "paperId": paper_id}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


@papers_app.command("recommend")
def get_recommendations(
    paper_id: str,
    limit: int = Option(5, "--limit", "-l", help="Number of recommendations"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get paper recommendations based on a paper.

    Returns papers similar to or citing the given paper.

    Examples:
        futurnal papers recommend abc123
        futurnal papers recommend abc123 --limit 10
    """
    try:
        agent = _get_agent()
        papers = _run_async(agent.get_recommendations(paper_id, limit=limit))

        if output_json:
            output = {
                "success": True,
                "sourcePaperId": paper_id,
                "recommendations": [
                    {
                        "paperId": p.paper_id,
                        "title": p.title,
                        "authors": [{"name": a.name, "authorId": a.author_id} for a in p.authors],
                        "year": p.year,
                        "abstractText": p.abstract,
                        "venue": p.venue,
                        "citationCount": p.citation_count,
                        "pdfUrl": p.pdf_url,
                        "semanticScholarUrl": p.source_url,
                        "doi": p.doi,
                        "arxivId": p.arxiv_id,
                        "fieldsOfStudy": p.fields_of_study,
                    }
                    for p in papers
                ],
            }
            print(json.dumps(output))
        else:
            if not papers:
                print(f"No recommendations found for paper {paper_id}")
                return

            print(f"\nRecommendations for paper {paper_id}:\n")
            for i, paper in enumerate(papers, 1):
                pdf_marker = "📄" if paper.pdf_url else "  "
                print(f"{i}. {pdf_marker} {paper.title}")
                print(f"   {paper.short_authors} ({paper.year or 'N/A'}) | Citations: {paper.citation_count}")
                print(f"   ID: {paper.paper_id}")
                print()

    except Exception as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e), "sourcePaperId": paper_id}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


@papers_app.command("get")
def get_paper_details(
    paper_id: str,
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get detailed metadata for a paper.

    Retrieves full paper metadata from Semantic Scholar.

    Examples:
        futurnal papers get abc123
        futurnal papers get "DOI:10.1234/example"
        futurnal papers get "ARXIV:2301.00001"
    """
    try:
        agent = _get_agent()
        paper = _run_async(agent.get_paper_details(paper_id))

        if paper:
            if output_json:
                output = {
                    "paperId": paper.paper_id,
                    "title": paper.title,
                    "authors": [{"name": a.name, "authorId": a.author_id} for a in paper.authors],
                    "year": paper.year,
                    "abstractText": paper.abstract,
                    "venue": paper.venue,
                    "citationCount": paper.citation_count,
                    "pdfUrl": paper.pdf_url,
                    "semanticScholarUrl": paper.source_url,
                    "doi": paper.doi,
                    "arxivId": paper.arxiv_id,
                    "fieldsOfStudy": paper.fields_of_study,
                }
                print(json.dumps(output))
            else:
                print(f"\n{paper.title}")
                print("=" * len(paper.title))
                print(f"\nAuthors: {', '.join(paper.author_names)}")
                print(f"Year: {paper.year or 'N/A'}")
                print(f"Venue: {paper.venue or 'N/A'}")
                print(f"Citations: {paper.citation_count}")
                print(f"References: {paper.reference_count}")

                if paper.fields_of_study:
                    print(f"Fields: {', '.join(paper.fields_of_study)}")

                if paper.doi:
                    print(f"DOI: {paper.doi}")
                if paper.arxiv_id:
                    print(f"arXiv: {paper.arxiv_id}")

                if paper.pdf_url:
                    print(f"PDF: {paper.pdf_url}")

                if paper.abstract:
                    print(f"\nAbstract:\n{paper.abstract}")
        else:
            if output_json:
                print(json.dumps({"success": False, "error": "Paper not found", "paperId": paper_id}))
            else:
                print(f"Paper not found: {paper_id}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e), "paperId": paper_id}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


@papers_app.command("ingest")
def ingest_papers(
    paper_ids: Optional[List[str]] = Argument(None, help="Specific paper IDs to ingest"),
    all_pending: bool = Option(False, "--all", "-a", help="Ingest all downloaded, unprocessed papers"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Add downloaded papers to the knowledge graph.

    Queues papers for ingestion into the PKG via the orchestrator.
    Papers are processed through the papers connector pipeline.

    Examples:
        futurnal papers ingest --all                    # Ingest all pending papers
        futurnal papers ingest abc123 def456           # Ingest specific papers
    """
    try:
        from ..agents.paper_search.state_store import get_default_state_store
        from ..agents.paper_search.models import DownloadedPaper

        agent = _get_agent()
        state_store = get_default_state_store()

        # Get papers to ingest
        papers_to_ingest = []

        if paper_ids:
            # Get specific papers from state store
            for pid in paper_ids:
                record = state_store.get(pid)
                if record and record.local_path:
                    papers_to_ingest.append(
                        DownloadedPaper(
                            metadata=PaperMetadata(
                                paper_id=record.paper_id,
                                title=record.title,
                            ),
                            local_path=record.local_path,
                            file_size_bytes=record.file_size_bytes,
                        )
                    )
                else:
                    if output_json:
                        pass  # Continue to process others
                    else:
                        print(f"Warning: Paper {pid} not found or not downloaded", file=sys.stderr)
        elif all_pending:
            # Get all unprocessed papers
            unprocessed = state_store.get_unprocessed_papers()
            for record in unprocessed:
                if record.local_path:
                    papers_to_ingest.append(
                        DownloadedPaper(
                            metadata=PaperMetadata(
                                paper_id=record.paper_id,
                                title=record.title,
                            ),
                            local_path=record.local_path,
                            file_size_bytes=record.file_size_bytes,
                        )
                    )
        else:
            if output_json:
                print(json.dumps({"success": False, "error": "Specify paper IDs or use --all flag"}))
            else:
                print("Specify paper IDs or use --all flag", file=sys.stderr)
            state_store.close()
            sys.exit(1)

        state_store.close()

        if not papers_to_ingest:
            if output_json:
                print(json.dumps({"success": True, "queued": 0, "papers": []}))
            else:
                print("No papers to ingest")
            return

        # Trigger ingestion
        queued_count = _run_async(agent.trigger_ingestion(papers_to_ingest))

        if output_json:
            output = {
                "success": True,
                "queued": queued_count,
                "papers": [
                    {
                        "paperId": p.metadata.paper_id,
                        "title": p.metadata.title,
                        "status": "queued",
                    }
                    for p in papers_to_ingest
                ],
            }
            print(json.dumps(output))
        else:
            print(f"Queued {queued_count} papers for ingestion into the knowledge graph")
            for p in papers_to_ingest:
                print(f"  • {p.metadata.title}")

    except Exception as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


@papers_app.command("status")
def paper_status(
    paper_id: Optional[str] = Argument(None, help="Paper ID to check status"),
    all_papers: bool = Option(False, "--all", "-a", help="Show status of all papers"),
    output_json: bool = Option(False, "--json", help="Output as JSON"),
) -> None:
    """Check download and ingestion status of papers.

    Shows the current status of papers in the system:
    - Download status: pending, downloaded, failed
    - Ingestion status: pending, queued, processing, completed, failed

    Examples:
        futurnal papers status abc123      # Check specific paper
        futurnal papers status --all       # Show all papers
    """
    try:
        from ..agents.paper_search.state_store import get_default_state_store

        state_store = get_default_state_store()

        if paper_id:
            # Get specific paper status
            record = state_store.get(paper_id)
            state_store.close()

            if record:
                if output_json:
                    output = {
                        "success": True,
                        "paper": _convert_keys_to_camel_case(record.to_dict()),
                    }
                    print(json.dumps(output))
                else:
                    print(f"\nPaper: {record.title}")
                    print(f"  ID: {record.paper_id}")
                    print(f"  Download Status: {record.download_status}")
                    print(f"  Ingestion Status: {record.ingestion_status}")
                    if record.local_path:
                        print(f"  Local Path: {record.local_path}")
                    if record.downloaded_at:
                        print(f"  Downloaded: {record.downloaded_at.isoformat()}")
                    if record.ingested_at:
                        print(f"  Ingested: {record.ingested_at.isoformat()}")
            else:
                if output_json:
                    print(json.dumps({"success": False, "error": "Paper not found", "paperId": paper_id}))
                else:
                    print(f"Paper not found: {paper_id}", file=sys.stderr)
                sys.exit(1)

        elif all_papers:
            # Get all papers with counts
            counts = state_store.count_by_status()
            papers = list(state_store.iter_all())
            state_store.close()

            if output_json:
                output = {
                    "success": True,
                    "total": len(papers),
                    "counts": _convert_keys_to_camel_case(counts),
                    "papers": [_convert_keys_to_camel_case(p.to_dict()) for p in papers],
                }
                print(json.dumps(output))
            else:
                print(f"\nPaper Status Summary:")
                print(f"  Total papers: {len(papers)}")
                print(f"\n  Download Status:")
                for status, count in counts.get("download", {}).items():
                    print(f"    {status}: {count}")
                print(f"\n  Ingestion Status:")
                for status, count in counts.get("ingestion", {}).items():
                    print(f"    {status}: {count}")

                if papers:
                    print(f"\n  Recent Papers:")
                    for p in papers[:10]:
                        dl_icon = "✓" if p.download_status == "downloaded" else "○"
                        ing_icon = "✓" if p.ingestion_status == "completed" else ("⏳" if p.ingestion_status == "queued" else "○")
                        print(f"    [{dl_icon}] [{ing_icon}] {p.title[:50]}...")
        else:
            state_store.close()
            if output_json:
                print(json.dumps({"success": False, "error": "Specify paper ID or use --all flag"}))
            else:
                print("Specify paper ID or use --all flag", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
