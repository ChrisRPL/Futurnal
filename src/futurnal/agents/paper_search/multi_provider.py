"""Multi-provider paper search aggregator.

Combines results from multiple academic paper APIs:
- OpenAlex (free, open, 240M+ works)
- CrossRef (free, DOI-based, 160M+ works)
- arXiv (free, preprints, 2M+ papers)
- Semantic Scholar (optional, rate-limited without API key)

Results are deduplicated by DOI/arXiv ID and ranked by citation count.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .models import PaperMetadata, SearchResult
from .openalex import OpenAlexClient
from .crossref import CrossRefClient
from .arxiv_client import ArxivClient
from .semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Available paper search providers."""
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"


@dataclass
class ProviderResult:
    """Result from a single provider."""
    provider: Provider
    result: SearchResult
    success: bool
    error: Optional[str] = None


class MultiProviderSearch:
    """Aggregates paper search results from multiple providers.

    Searches multiple APIs in parallel, deduplicates results,
    and returns a unified result set ranked by relevance/citations.

    Example:
        search = MultiProviderSearch(
            providers=[Provider.OPENALEX, Provider.ARXIV],
            email="user@example.com"
        )
        result = await search.search("transformer attention", limit=20)
    """

    # Default provider priority for ranking
    PROVIDER_PRIORITY = {
        Provider.OPENALEX: 1,      # Best coverage + metadata
        Provider.SEMANTIC_SCHOLAR: 2,  # Good if API key available
        Provider.CROSSREF: 3,      # Good for DOI lookup
        Provider.ARXIV: 4,         # Preprints, less metadata
    }

    def __init__(
        self,
        providers: Optional[List[Provider]] = None,
        email: Optional[str] = None,
        semantic_scholar_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize multi-provider search.

        Args:
            providers: List of providers to use. Default: OpenAlex + arXiv
            email: Email for polite pools (OpenAlex, CrossRef)
            semantic_scholar_key: API key for Semantic Scholar
            timeout: Request timeout per provider
        """
        self.email = email or os.environ.get("FUTURNAL_EMAIL")
        self.semantic_scholar_key = semantic_scholar_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.timeout = timeout

        # Default providers: OpenAlex + arXiv (both free, no rate limit issues)
        if providers is None:
            providers = [Provider.OPENALEX, Provider.ARXIV]
            # Add Semantic Scholar only if API key available
            if self.semantic_scholar_key:
                providers.append(Provider.SEMANTIC_SCHOLAR)

        self.providers = providers
        self._clients: Dict[Provider, Any] = {}

        logger.info(f"MultiProviderSearch initialized with providers: {[p.value for p in providers]}")

    def _get_client(self, provider: Provider):
        """Get or create client for a provider."""
        if provider not in self._clients:
            if provider == Provider.OPENALEX:
                self._clients[provider] = OpenAlexClient(email=self.email, timeout=self.timeout)
            elif provider == Provider.CROSSREF:
                self._clients[provider] = CrossRefClient(email=self.email, timeout=self.timeout)
            elif provider == Provider.ARXIV:
                self._clients[provider] = ArxivClient(timeout=self.timeout)
            elif provider == Provider.SEMANTIC_SCHOLAR:
                self._clients[provider] = SemanticScholarClient(
                    api_key=self.semantic_scholar_key,
                    timeout=self.timeout,
                )
        return self._clients[provider]

    async def search(
        self,
        query: str,
        limit: int = 20,
        year_range: Optional[tuple[int, int]] = None,
        providers: Optional[List[Provider]] = None,
    ) -> SearchResult:
        """Search across multiple providers.

        Args:
            query: Search query
            limit: Total number of results desired
            year_range: Optional (start_year, end_year) filter
            providers: Override default providers for this search

        Returns:
            Aggregated SearchResult with deduplicated papers
        """
        start_time = time.time()
        active_providers = providers or self.providers

        # Request more results per provider to account for deduplication
        per_provider_limit = min(limit * 2, 50)

        # Search all providers in parallel
        tasks = []
        for provider in active_providers:
            task = self._search_provider(provider, query, per_provider_limit, year_range)
            tasks.append(task)

        provider_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect and deduplicate results
        all_papers: Dict[str, PaperMetadata] = {}
        total_from_providers = 0

        for i, result in enumerate(provider_results):
            provider = active_providers[i]

            if isinstance(result, Exception):
                logger.warning(f"Provider {provider.value} failed: {result}")
                continue

            if isinstance(result, ProviderResult):
                if result.success:
                    total_from_providers += len(result.result.papers)
                    for paper in result.result.papers:
                        key = self._get_dedup_key(paper)
                        if key not in all_papers:
                            all_papers[key] = paper
                        else:
                            # Merge metadata from multiple sources
                            all_papers[key] = self._merge_papers(all_papers[key], paper)
                else:
                    logger.warning(f"Provider {provider.value} error: {result.error}")

        # Sort by citation count (descending) then by title
        sorted_papers = sorted(
            all_papers.values(),
            key=lambda p: (-(p.citation_count or 0), p.title.lower()),
        )

        # Limit results
        final_papers = sorted_papers[:limit]

        search_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"Multi-provider search '{query[:30]}...' returned {len(final_papers)} papers "
            f"(from {total_from_providers} total, {len(active_providers)} providers, {search_time}ms)"
        )

        return SearchResult(
            query=query,
            papers=final_papers,
            total_results=len(all_papers),
            offset=0,
            limit=limit,
            search_time_ms=search_time,
            source="multi_provider",
        )

    async def _search_provider(
        self,
        provider: Provider,
        query: str,
        limit: int,
        year_range: Optional[tuple[int, int]],
    ) -> ProviderResult:
        """Search a single provider."""
        try:
            client = self._get_client(provider)
            result = await client.search(
                query=query,
                limit=limit,
                year_range=year_range,
            )
            return ProviderResult(
                provider=provider,
                result=result,
                success=True,
            )
        except Exception as e:
            logger.error(f"Error searching {provider.value}: {e}")
            return ProviderResult(
                provider=provider,
                result=SearchResult(
                    query=query,
                    papers=[],
                    total_results=0,
                    offset=0,
                    limit=limit,
                    search_time_ms=0,
                    source=provider.value,
                ),
                success=False,
                error=str(e),
            )

    def _get_dedup_key(self, paper: PaperMetadata) -> str:
        """Get a unique key for deduplication.

        Priority: DOI > arXiv ID > normalized title
        """
        if paper.doi:
            return f"doi:{paper.doi.lower()}"
        if paper.arxiv_id:
            # Normalize arXiv ID (remove version)
            arxiv_id = paper.arxiv_id.split("v")[0]
            return f"arxiv:{arxiv_id}"
        # Fallback to normalized title
        title_key = paper.title.lower().strip()
        # Remove common prefixes/punctuation
        title_key = "".join(c for c in title_key if c.isalnum() or c.isspace())
        return f"title:{title_key[:100]}"

    def _merge_papers(self, existing: PaperMetadata, new: PaperMetadata) -> PaperMetadata:
        """Merge metadata from two paper records (same paper, different sources).

        Prefers data from higher-priority providers.
        """
        # Keep the one with more complete metadata
        existing_score = self._metadata_completeness(existing)
        new_score = self._metadata_completeness(new)

        if new_score > existing_score:
            base, supplement = new, existing
        else:
            base, supplement = existing, new

        # Fill in missing fields from supplement
        return PaperMetadata(
            paper_id=base.paper_id,
            title=base.title or supplement.title,
            abstract=base.abstract or supplement.abstract,
            authors=base.authors if base.authors else supplement.authors,
            year=base.year or supplement.year,
            venue=base.venue or supplement.venue,
            citation_count=max(base.citation_count or 0, supplement.citation_count or 0),
            reference_count=max(base.reference_count or 0, supplement.reference_count or 0),
            fields_of_study=base.fields_of_study or supplement.fields_of_study,
            pdf_url=base.pdf_url or supplement.pdf_url,
            source_url=base.source_url or supplement.source_url,
            source=f"{base.source}+{supplement.source}",
            doi=base.doi or supplement.doi,
            arxiv_id=base.arxiv_id or supplement.arxiv_id,
        )

    def _metadata_completeness(self, paper: PaperMetadata) -> int:
        """Score metadata completeness (higher = more complete)."""
        score = 0
        if paper.title:
            score += 1
        if paper.abstract and len(paper.abstract) > 50:
            score += 2
        if paper.authors:
            score += 1
        if paper.year:
            score += 1
        if paper.venue:
            score += 1
        if paper.citation_count and paper.citation_count > 0:
            score += 2
        if paper.pdf_url:
            score += 2
        if paper.doi:
            score += 1
        if paper.arxiv_id:
            score += 1
        if paper.fields_of_study:
            score += 1
        return score


def get_default_multi_provider_search() -> MultiProviderSearch:
    """Get default multi-provider search with sensible defaults.

    Uses OpenAlex + arXiv by default.
    Adds Semantic Scholar if API key is available.
    """
    return MultiProviderSearch()
