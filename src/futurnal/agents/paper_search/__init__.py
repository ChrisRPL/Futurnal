"""Academic Paper Search Agent.

Phase D: Paper Agent

Provides agent capabilities for searching, downloading, and processing
academic papers into the knowledge graph.

Key Components:
- MultiProviderSearch: Aggregated search across multiple providers
- OpenAlexClient: API client for OpenAlex (free, 240M+ works)
- CrossRefClient: API client for CrossRef (free, DOI-based)
- ArxivClient: API client for arXiv (free, preprints)
- SemanticScholarClient: API client for Semantic Scholar (rate-limited)
- PaperSearchAgent: Main agent orchestrating search flow

Usage:
    from futurnal.agents.paper_search import MultiProviderSearch

    search = MultiProviderSearch()
    results = await search.search("causal inference personal knowledge")
"""

from .models import PaperMetadata, SearchResult
from .semantic_scholar import SemanticScholarClient
from .openalex import OpenAlexClient
from .crossref import CrossRefClient
from .arxiv_client import ArxivClient
from .multi_provider import MultiProviderSearch, Provider
from .agent import PaperSearchAgent

__all__ = [
    "PaperMetadata",
    "SearchResult",
    "SemanticScholarClient",
    "OpenAlexClient",
    "CrossRefClient",
    "ArxivClient",
    "MultiProviderSearch",
    "Provider",
    "PaperSearchAgent",
]
