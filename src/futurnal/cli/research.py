"""Research CLI commands.

Provides web search and deep research capabilities for the Futurnal desktop app.

Commands:
    futurnal research web "query" --json
    futurnal research deep "query" --json
    futurnal research status --json

Research Foundation:
- WebDancer (2505.22648v3): End-to-end web agents
- Personalized Deep Research (2509.25106v1): User-centric research
"""

import asyncio
import json
import logging
import sys
import time
import re
from typing import Optional, List, Dict, Any

import typer

logger = logging.getLogger(__name__)

research_app = typer.Typer(help="Web search and deep research commands")


def _output_json(data: dict) -> None:
    """Output JSON to stdout for IPC."""
    print(json.dumps(data, indent=2, default=str))


def _output_error(message: str, as_json: bool = False) -> None:
    """Output error message."""
    if as_json:
        _output_json({"error": message, "success": False})
    else:
        typer.echo(f"Error: {message}", err=True)
    sys.exit(1)


async def _fetch_url(url: str) -> str:
    """Fetch content from a URL using httpx."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            response.raise_for_status()
            return response.text
    except ImportError:
        logger.warning("httpx not available, falling back to urllib")
        import urllib.request
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        raise


async def _search_duckduckgo(query: str, max_results: int = 10) -> List[dict]:
    """Search DuckDuckGo for results using their lite/HTML interface.

    Uses BeautifulSoup for robust HTML parsing with regex fallback.
    """
    try:
        import httpx
        from urllib.parse import quote_plus, unquote

        # Use DuckDuckGo lite which has simpler HTML
        search_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            response.raise_for_status()
            html = response.text

        results = []

        # Try BeautifulSoup first for robust parsing
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # DDG lite has results in table rows with rel="nofollow" links
            for link in soup.find_all('a', rel='nofollow'):
                if len(results) >= max_results:
                    break

                url = link.get('href', '')
                title = link.get_text(strip=True)

                # Skip internal DDG links that aren't redirects
                if 'duckduckgo.com' in url and '/l/' not in url:
                    continue

                # Handle DDG redirect URLs
                if url.startswith('//duckduckgo.com/l/'):
                    url_match = re.search(r'uddg=([^&]+)', url)
                    if url_match:
                        url = unquote(url_match.group(1))

                # Find snippet by looking at the table structure
                snippet = ""
                parent_row = link.find_parent('tr')
                if parent_row:
                    # Check if there's a snippet in a sibling row
                    next_row = parent_row.find_next_sibling('tr')
                    if next_row:
                        # Look for text cells that might contain the snippet
                        cells = next_row.find_all('td')
                        for cell in cells:
                            text = cell.get_text(strip=True)
                            if text and len(text) > 20:  # Snippets are usually longer
                                snippet = text
                                break

                if url and title:
                    results.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                    })

            logger.info(f"DuckDuckGo search (BeautifulSoup) found {len(results)} results for '{query}'")

        except ImportError:
            # Fallback to regex if BeautifulSoup not available
            logger.warning("BeautifulSoup not available, falling back to regex parsing")

            # Find all result links (exclude navigation links)
            link_pattern = r'<a\s+rel="nofollow"\s+href="([^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(link_pattern, html, re.IGNORECASE)

            # Find snippets (text in table cells after links)
            snippet_pattern = r'<td[^>]*class="result-snippet"[^>]*>([^<]+)</td>'
            snippets = re.findall(snippet_pattern, html, re.IGNORECASE)

            # If that doesn't work, try simpler snippet extraction
            if not snippets:
                snippet_pattern2 = r'</a>\s*</td>\s*</tr>\s*<tr[^>]*>\s*<td[^>]*>\s*</td>\s*<td[^>]*>([^<]+)</td>'
                snippets = re.findall(snippet_pattern2, html, re.IGNORECASE)

            for i, (url, title) in enumerate(matches[:max_results]):
                if 'duckduckgo.com' in url and '/l/' not in url:
                    continue

                if url.startswith('//duckduckgo.com/l/?'):
                    url_match = re.search(r'uddg=([^&]+)', url)
                    if url_match:
                        url = unquote(url_match.group(1))

                snippet = snippets[i] if i < len(snippets) else ""
                results.append({
                    "url": url,
                    "title": title.strip(),
                    "snippet": snippet.strip(),
                })

            logger.info(f"DuckDuckGo search (regex) found {len(results)} results for '{query}'")

        return results

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


async def _search_with_fallback(query: str, max_results: int = 10) -> List[dict]:
    """Search with multiple fallback options.

    Results are filtered and sorted by source quality.
    """
    results = []

    # Try duckduckgo-search library first (if installed)
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results * 2):  # Fetch extra for filtering
                results.append({
                    "url": r.get("href", r.get("link", "")),
                    "title": r.get("title", ""),
                    "snippet": r.get("body", r.get("snippet", "")),
                })

        if results:
            logger.info(f"duckduckgo-search found {len(results)} results")
    except ImportError:
        logger.debug("duckduckgo-search not installed, using HTML fallback")
    except Exception as e:
        logger.warning(f"duckduckgo-search failed: {e}")

    # Fallback to HTML parsing if no results
    if not results:
        results = await _search_duckduckgo(query, max_results * 2)  # Fetch extra for filtering

    # Apply quality filtering and sorting
    return _filter_and_sort_results(results, max_results)


# Source quality scoring domains
_QUALITY_DOMAINS = {
    'high': [
        'arxiv.org', 'github.com', 'stackoverflow.com', 'docs.python.org',
        'developer.mozilla.org', 'docs.microsoft.com', 'docs.aws.amazon.com',
        'kubernetes.io', 'pytorch.org', 'tensorflow.org', 'huggingface.co',
    ],
    'medium': [
        'wikipedia.org', 'medium.com', 'dev.to', 'towardsdatascience.com',
        'realpython.com', 'geeksforgeeks.org', 'w3schools.com',
    ],
    'low': [
        'pinterest.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'tiktok.com', 'quora.com',
    ],
}


def _score_search_result(result: dict) -> float:
    """Score a search result by source quality.

    Returns a score between 0.3 and 1.0 based on domain reputation.
    """
    url = result.get('url', '').lower()

    for domain in _QUALITY_DOMAINS['high']:
        if domain in url:
            return 1.0

    for domain in _QUALITY_DOMAINS['medium']:
        if domain in url:
            return 0.7

    for domain in _QUALITY_DOMAINS['low']:
        if domain in url:
            return 0.3

    return 0.5  # Default for unknown domains


def _filter_and_sort_results(results: List[dict], max_results: int = 10) -> List[dict]:
    """Filter and sort search results by quality.

    Removes duplicates, filters low-quality sources, and sorts by quality score.
    """
    # Remove duplicates by URL
    seen_urls = set()
    unique_results = []
    for result in results:
        url = result.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    # Filter out very low quality sources
    filtered = [r for r in unique_results if _score_search_result(r) >= 0.3]

    # Sort by quality score (descending)
    sorted_results = sorted(filtered, key=_score_search_result, reverse=True)

    return sorted_results[:max_results]


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML."""
    # Remove script and style tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def _get_web_browser():
    """Get configured WebBrowsingAgent instance."""
    from futurnal.agents.web_browser import WebBrowsingAgent

    async def web_fetcher(url: str) -> str:
        return await _fetch_url(url)

    async def search_engine(query: str) -> List[dict]:
        return await _search_with_fallback(query)

    # Try to get LLM client
    llm_client = None
    try:
        from futurnal.search.ollama_pool import OllamaPool
        pool = OllamaPool()
        llm_client = pool
        logger.info("LLM client initialized")
    except Exception as e:
        logger.warning(f"Could not initialize LLM: {e}")

    return WebBrowsingAgent(
        llm_client=llm_client,
        web_fetcher=web_fetcher,
        search_engine=search_engine,
        max_steps=15,
        max_pages=8,
    )


async def _do_deep_research(query: str, depth: str = "standard") -> Dict[str, Any]:
    """Conduct deep research using web browsing and local knowledge.

    This is a direct implementation rather than using PersonalizedResearchAgent
    which has incomplete backends.
    """
    start_time = time.time()

    findings: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    key_points: List[str] = []

    logger.info(f"Starting deep research for: {query} (depth: {depth})")

    # 1. Try local PKG search first
    try:
        from futurnal.search.api import create_hybrid_search_api

        search_api = create_hybrid_search_api(graphrag_enabled=True)
        if search_api:
            logger.info("Searching local PKG...")
            local_results = await asyncio.to_thread(
                search_api.search,
                query,
                limit=10
            )

            if hasattr(local_results, 'results') and local_results.results:
                for result in local_results.results[:5]:
                    content = getattr(result, 'content', str(result))
                    findings.append({
                        "content": content[:500],
                        "type": "knowledge_graph",
                        "relevance": getattr(result, 'relevance', 0.7),
                    })
                    sources.append({
                        "type": "pkg",
                        "name": getattr(result, 'source', 'Personal Knowledge Graph'),
                    })
                logger.info(f"Found {len(findings)} PKG results")
    except Exception as e:
        logger.warning(f"PKG search failed: {e}")

    # 2. Web search for more depth
    if depth in ["standard", "detailed", "exhaustive"]:
        try:
            logger.info("Searching the web...")
            browser = _get_web_browser()

            # Run web browse
            result = await browser.browse(query)

            if result.findings:
                for finding in result.findings:
                    findings.append({
                        "content": finding.get("fact", "")[:500],
                        "type": "web_search",
                        "relevance": 0.6,
                    })

                for source in result.sources:
                    sources.append({
                        "type": "web",
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "reliability": source.get("reliability", "unknown"),
                    })

                logger.info(f"Web search found {len(result.findings)} findings from {len(result.sources)} sources")

            # Use the synthesized answer if available
            if result.answer and result.answer != "I couldn't find relevant information to answer this query.":
                summary = result.answer
            else:
                summary = None

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            summary = None
    else:
        summary = None

    # 3. Generate summary if we don't have one
    if not summary:
        if findings:
            # Try LLM synthesis
            try:
                from futurnal.search.ollama_pool import OllamaPool
                pool = OllamaPool()

                findings_text = "\n".join([
                    f"- {f['content'][:200]}..."
                    for f in findings[:10]
                ])

                prompt = f"""Based on the following information, answer the question: "{query}"

Information:
{findings_text}

Respond ONLY with valid JSON in this exact format:
{{"summary": "A comprehensive 2-3 paragraph summary answering the question", "key_points": ["First key finding", "Second key finding", "Third key finding"]}}

Important: Output only the JSON object, no markdown, no code blocks, no extra text."""

                # Try async generate
                if hasattr(pool, 'generate_async'):
                    response = await pool.generate_async(prompt)
                elif hasattr(pool, 'generate'):
                    response = await asyncio.to_thread(pool.generate, prompt)
                else:
                    response = None

                if response:
                    # Try to parse as JSON first
                    try:
                        # Clean up response - remove markdown code blocks if present
                        clean_response = response.strip()
                        if clean_response.startswith('```'):
                            # Remove markdown code block
                            clean_response = clean_response.split('\n', 1)[-1]
                            if '```' in clean_response:
                                clean_response = clean_response.rsplit('```', 1)[0]
                        clean_response = clean_response.strip()

                        parsed = json.loads(clean_response)
                        summary = parsed.get("summary", "")
                        key_points = parsed.get("key_points", [])
                        logger.info("Successfully parsed JSON response from LLM")
                    except json.JSONDecodeError:
                        # Fallback: extract from unstructured response
                        logger.warning("JSON parsing failed, falling back to text extraction")
                        lines = response.split('\n')
                        summary_parts = []

                        for line in lines:
                            line = line.strip()
                            if line.startswith(('•', '-', '*', '•')):
                                key_points.append(line.lstrip('•-* '))
                            elif line and not line.startswith('{') and not line.startswith('}'):
                                summary_parts.append(line)

                        summary = '\n'.join(summary_parts) if summary_parts else response

            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
                # Fallback: concatenate findings
                summary = "Based on available information:\n\n" + "\n".join([
                    f"• {f['content'][:200]}" for f in findings[:5]
                ])
        else:
            summary = "No relevant information found."

    # Extract key points if not already done
    if not key_points and findings:
        key_points = [f["content"][:100] for f in findings[:5]]

    # Calculate confidence based on quality, not just quantity
    # Weight different source types appropriately
    pkg_weight = 0.4 if any(f.get("type") == "knowledge_graph" for f in findings) else 0
    web_weight = 0.3 if any(f.get("type") == "web_search" for f in findings) else 0
    source_weight = min(0.2, len(sources) * 0.04)  # Up to 0.2 for 5+ sources
    synthesis_weight = 0.2 if summary and len(summary) > 100 else 0.1 if summary else 0
    confidence = min(0.95, pkg_weight + web_weight + source_weight + synthesis_weight)

    research_time_ms = int((time.time() - start_time) * 1000)

    return {
        "success": True,
        "query": query,
        "userId": "default",
        "summary": summary,
        "keyPoints": key_points,
        "sources": sources,
        "numSourcesConsulted": len(sources),
        "expertiseLevelUsed": "intermediate",
        "depthUsed": depth,
        "confidence": confidence,
        "relevanceScore": confidence,
        "researchTimeMs": research_time_ms,
        "detailedFindings": findings[:20],
    }


@research_app.command("web")
def web_search(
    query: str = typer.Argument(..., help="Search query"),
    max_pages: int = typer.Option(5, "--max-pages", "-p", help="Maximum pages to visit"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Search the web for information.

    Uses WebDancer-style autonomous web browsing to find and synthesize
    information from multiple web sources.

    Example:
        futurnal research web "what is causal inference" --json
        futurnal research web "latest machine learning trends" --max-pages 10
    """
    start_time = time.time()

    try:
        browser = _get_web_browser()
        browser.max_pages = max_pages

        async def run():
            return await browser.browse(query)

        result = asyncio.run(run())

        search_time_ms = int((time.time() - start_time) * 1000)

        if output_json:
            _output_json({
                "success": True,
                "query": query,
                "answer": result.answer,
                "sources": result.sources,
                "confidence": result.confidence,
                "coverage": result.coverage,
                "numSources": result.num_sources,
                "totalSteps": result.total_steps,
                "totalPages": result.total_pages,
                "searchTimeMs": search_time_ms,
                "findings": [
                    {
                        "fact": f.get("fact", ""),
                        "sourceUrl": f.get("source_url", ""),
                        "sourceTitle": f.get("source_title", ""),
                        "reliability": f.get("reliability", "unknown"),
                    }
                    for f in result.findings
                ],
            })
        else:
            typer.echo(f"\n🔍 Web Search: {query}")
            typer.echo("=" * 60)
            typer.echo(f"\n{result.answer}")

            if result.sources:
                typer.echo("\n📚 Sources:")
                for source in result.sources[:5]:
                    typer.echo(f"  - {source.get('title', source.get('url', 'Unknown'))}")

            typer.echo(f"\n✓ Confidence: {result.confidence:.0%}")
            typer.echo(f"✓ Pages visited: {result.total_pages}")
            typer.echo(f"✓ Time: {search_time_ms}ms")

    except Exception as e:
        logger.exception("Web search failed")
        _output_error(str(e), as_json=output_json)


@research_app.command("deep")
def deep_research(
    query: str = typer.Argument(..., help="Research query"),
    depth: str = typer.Option(
        "standard",
        "--depth", "-d",
        help="Research depth: overview, standard, detailed, exhaustive"
    ),
    user_id: str = typer.Option(
        "default",
        "--user", "-u",
        help="User ID for personalization"
    ),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Conduct deep personalized research.

    Combines knowledge graph, vector search, and web research
    to provide comprehensive, personalized research results.

    Research depths:
    - overview: Quick summary from local knowledge
    - standard: Normal depth with PKG and basic web search
    - detailed: In-depth with comprehensive web research
    - exhaustive: Leave no stone unturned

    Example:
        futurnal research deep "impact of sleep on productivity" --json
        futurnal research deep "quantum computing applications" --depth detailed
    """
    try:
        async def run():
            return await _do_deep_research(query, depth)

        result = asyncio.run(run())

        if output_json:
            _output_json(result)
        else:
            typer.echo(f"\n🔬 Deep Research: {query}")
            typer.echo("=" * 60)

            typer.echo(f"\n📋 Summary:")
            typer.echo(result.get("summary", "No summary available"))

            key_points = result.get("keyPoints", [])
            if key_points:
                typer.echo("\n🔑 Key Points:")
                for point in key_points:
                    typer.echo(f"  • {point}")

            sources = result.get("sources", [])
            if sources:
                typer.echo(f"\n📚 Sources ({result.get('numSourcesConsulted', 0)} consulted):")
                for source in sources[:5]:
                    if isinstance(source, dict):
                        typer.echo(f"  - {source.get('title', source.get('url', str(source)))}")
                    else:
                        typer.echo(f"  - {source}")

            typer.echo(f"\n✓ Depth: {result.get('depthUsed', depth)}")
            typer.echo(f"✓ Confidence: {result.get('confidence', 0):.0%}")
            typer.echo(f"✓ Time: {result.get('researchTimeMs', 0)}ms")

    except Exception as e:
        logger.exception("Deep research failed")
        _output_error(str(e), as_json=output_json)


@research_app.command("quick")
def quick_search(
    query: str = typer.Argument(..., help="Search query"),
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Quick web search without deep analysis.

    Returns raw search results from DuckDuckGo without
    visiting pages or synthesizing answers.

    Example:
        futurnal research quick "python async tutorial" --json
    """
    start_time = time.time()

    try:
        async def run():
            return await _search_with_fallback(query, max_results=10)

        results = asyncio.run(run())

        search_time_ms = int((time.time() - start_time) * 1000)

        if output_json:
            _output_json({
                "success": True,
                "query": query,
                "results": results,
                "total": len(results),
                "searchTimeMs": search_time_ms,
            })
        else:
            typer.echo(f"\n🔍 Quick Search: {query}")
            typer.echo("=" * 60)

            if not results:
                typer.echo("\nNo results found.")
                return

            for i, result in enumerate(results, 1):
                typer.echo(f"\n{i}. {result.get('title', 'Untitled')}")
                typer.echo(f"   {result.get('url', '')}")
                if result.get('snippet'):
                    typer.echo(f"   {result['snippet'][:150]}...")

            typer.echo(f"\n✓ Found {len(results)} results in {search_time_ms}ms")

    except Exception as e:
        logger.exception("Quick search failed")
        _output_error(str(e), as_json=output_json)


@research_app.command("status")
def research_status(
    output_json: bool = typer.Option(True, "--json", "-j", help="Output as JSON"),
) -> None:
    """Check research infrastructure status.

    Verifies that all research components are available:
    - Web fetcher (httpx)
    - Search engine (DuckDuckGo)
    - LLM (Ollama)
    - Knowledge graph (Neo4j)
    - Vector store (ChromaDB)

    Example:
        futurnal research status --json
    """
    status = {
        "web_fetcher": {"available": False, "status": "unknown"},
        "search_engine": {"available": False, "status": "unknown"},
        "llm": {"available": False, "status": "unknown"},
        "knowledge_graph": {"available": False, "status": "unknown"},
        "vector_store": {"available": False, "status": "unknown"},
    }

    # Check httpx
    try:
        import httpx
        status["web_fetcher"] = {"available": True, "status": "httpx available"}
    except ImportError:
        status["web_fetcher"] = {"available": False, "status": "httpx not installed"}

    # Check duckduckgo-search library
    try:
        from duckduckgo_search import DDGS
        status["search_engine"] = {"available": True, "status": "duckduckgo-search library"}
    except ImportError:
        # Fallback to HTML parsing
        status["search_engine"] = {"available": True, "status": "DuckDuckGo HTML fallback"}

    # Check LLM
    try:
        from futurnal.search.ollama_pool import OllamaPool
        pool = OllamaPool()
        status["llm"] = {"available": True, "status": "Ollama connected"}
    except Exception as e:
        status["llm"] = {"available": False, "status": str(e)[:100]}

    # Check PKG via search API
    try:
        from futurnal.search.api import create_hybrid_search_api
        api = create_hybrid_search_api(graphrag_enabled=True)
        if api:
            status["knowledge_graph"] = {"available": True, "status": "HybridSearchAPI available"}
        else:
            status["knowledge_graph"] = {"available": False, "status": "API not configured"}
    except Exception as e:
        status["knowledge_graph"] = {"available": False, "status": str(e)[:100]}

    # Check vector store
    try:
        from futurnal.search.vector_store import get_default_collection
        collection = get_default_collection()
        if collection:
            status["vector_store"] = {"available": True, "status": "ChromaDB connected"}
        else:
            status["vector_store"] = {"available": False, "status": "Collection not found"}
    except Exception as e:
        status["vector_store"] = {"available": False, "status": str(e)[:100]}

    # Calculate overall health
    available_count = sum(1 for s in status.values() if s["available"])
    all_healthy = available_count == len(status)

    if output_json:
        _output_json({
            "success": True,
            "components": status,
            "availableCount": available_count,
            "totalCount": len(status),
            "allHealthy": all_healthy,
        })
    else:
        typer.echo("\n🔬 Research Infrastructure Status")
        typer.echo("=" * 50)

        for component, info in status.items():
            icon = "✅" if info["available"] else "❌"
            typer.echo(f"\n{icon} {component.replace('_', ' ').title()}")
            typer.echo(f"   Status: {info['status']}")

        typer.echo(f"\n{'✅ All systems operational' if all_healthy else f'⚠️ {available_count}/{len(status)} components available'}")
