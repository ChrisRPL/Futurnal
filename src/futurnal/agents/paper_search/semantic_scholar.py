"""Semantic Scholar API Client.

Phase D: Paper Agent

Provides API access to Semantic Scholar for paper search and metadata.

API Documentation: https://api.semanticscholar.org/api-docs/

Rate Limits (free tier):
- 100 requests per 5 minutes
- Respect rate limits with backoff

Usage:
    client = SemanticScholarClient()
    results = await client.search("causal inference", limit=10)
    paper = await client.get_paper("DOI:10.1234/example")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .models import PaperAuthor, PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Simple rate limiter for API calls."""

    max_requests: int = 100
    window_seconds: int = 300  # 5 minutes
    requests: List[float] = None

    def __post_init__(self):
        self.requests = []

    def can_make_request(self) -> bool:
        """Check if we can make a request."""
        now = time.time()
        # Remove old requests outside window
        self.requests = [r for r in self.requests if now - r < self.window_seconds]
        return len(self.requests) < self.max_requests

    def record_request(self):
        """Record that a request was made."""
        self.requests.append(time.time())

    async def wait_if_needed(self):
        """Wait if rate limited."""
        while not self.can_make_request():
            wait_time = self.window_seconds - (time.time() - self.requests[0]) + 1
            logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)


class SemanticScholarClient:
    """Client for Semantic Scholar API.

    Provides paper search and metadata retrieval with rate limiting.

    Example:
        client = SemanticScholarClient()

        # Search for papers
        results = await client.search("graph neural networks", limit=5)
        for paper in results.papers:
            print(f"{paper.title} ({paper.year}) - {paper.citation_count} citations")

        # Get specific paper
        paper = await client.get_paper("DOI:10.1145/3292500.3330961")
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    # Fields to request from API
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "venue",
        "citationCount",
        "referenceCount",
        "fieldsOfStudy",
        "authors",
        "openAccessPdf",
        "externalIds",
        "url",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize Semantic Scholar client.

        Args:
            api_key: Optional API key for higher rate limits
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limiter = RateLimiter()

        # Build headers
        self.headers = {
            "Accept": "application/json",
        }
        if api_key:
            self.headers["x-api-key"] = api_key

        logger.info("SemanticScholarClient initialized")

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        year_range: Optional[tuple[int, int]] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> SearchResult:
        """Search for papers.

        Args:
            query: Search query
            limit: Number of results (max 100)
            offset: Pagination offset
            year_range: Optional (start_year, end_year) tuple
            fields_of_study: Optional list of fields to filter by

        Returns:
            SearchResult with matching papers
        """
        start_time = time.time()

        # Build query parameters
        params = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": ",".join(self.PAPER_FIELDS),
        }

        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        # Make request
        await self.rate_limiter.wait_if_needed()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    headers=self.headers,
                )
                self.rate_limiter.record_request()

                if response.status_code == 429:
                    logger.warning("Rate limited by Semantic Scholar")
                    # Wait and retry once
                    await asyncio.sleep(60)
                    response = await client.get(
                        f"{self.BASE_URL}/paper/search",
                        params=params,
                        headers=self.headers,
                    )

                response.raise_for_status()
                data = response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Semantic Scholar API error: {e}")
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                offset=offset,
                limit=limit,
                search_time_ms=int((time.time() - start_time) * 1000),
                source="semantic_scholar",
            )

        except Exception as e:
            logger.error(f"Semantic Scholar request failed: {e}")
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                offset=offset,
                limit=limit,
                search_time_ms=int((time.time() - start_time) * 1000),
                source="semantic_scholar",
            )

        # Parse results
        papers = []
        for item in data.get("data", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)

        search_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"Semantic Scholar search '{query[:30]}...' returned {len(papers)} papers "
            f"({search_time}ms)"
        )

        return SearchResult(
            query=query,
            papers=papers,
            total_results=data.get("total", len(papers)),
            offset=offset,
            limit=limit,
            search_time_ms=search_time,
            source="semantic_scholar",
        )

    async def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get paper metadata by ID.

        Args:
            paper_id: Paper ID (Semantic Scholar ID, DOI, or arXiv ID)
                      Use "DOI:..." or "ARXIV:..." prefix for external IDs

        Returns:
            PaperMetadata or None if not found
        """
        await self.rate_limiter.wait_if_needed()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/paper/{paper_id}",
                    params={"fields": ",".join(self.PAPER_FIELDS)},
                    headers=self.headers,
                )
                self.rate_limiter.record_request()
                response.raise_for_status()
                data = response.json()

            return self._parse_paper(data)

        except Exception as e:
            logger.error(f"Failed to get paper {paper_id}: {e}")
            return None

    async def get_recommendations(
        self,
        paper_id: str,
        limit: int = 10,
    ) -> List[PaperMetadata]:
        """Get recommended papers based on a paper.

        Args:
            paper_id: Paper ID to get recommendations for
            limit: Number of recommendations

        Returns:
            List of recommended papers
        """
        await self.rate_limiter.wait_if_needed()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/recommendations/v1/papers/forpaper/{paper_id}",
                    params={
                        "limit": limit,
                        "fields": ",".join(self.PAPER_FIELDS),
                    },
                    headers=self.headers,
                )
                self.rate_limiter.record_request()
                response.raise_for_status()
                data = response.json()

            papers = []
            for item in data.get("recommendedPapers", []):
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)

            return papers

        except Exception as e:
            logger.error(f"Failed to get recommendations for {paper_id}: {e}")
            return []

    def _parse_paper(self, data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Parse paper data from API response."""
        try:
            # Extract external IDs
            external_ids = data.get("externalIds", {}) or {}
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")

            # Extract authors
            authors = []
            for author_data in data.get("authors", []) or []:
                author = PaperAuthor(
                    name=author_data.get("name", "Unknown"),
                    author_id=author_data.get("authorId"),
                )
                authors.append(author)

            # Extract PDF URL
            pdf_info = data.get("openAccessPdf", {}) or {}
            pdf_url = pdf_info.get("url")

            return PaperMetadata(
                paper_id=data.get("paperId", ""),
                title=data.get("title", "Untitled"),
                abstract=data.get("abstract", "") or "",
                authors=authors,
                year=data.get("year"),
                venue=data.get("venue"),
                citation_count=data.get("citationCount", 0) or 0,
                reference_count=data.get("referenceCount", 0) or 0,
                fields_of_study=data.get("fieldsOfStudy", []) or [],
                pdf_url=pdf_url,
                source_url=data.get("url"),
                source="semantic_scholar",
                doi=doi,
                arxiv_id=arxiv_id,
            )

        except Exception as e:
            logger.warning(f"Failed to parse paper data: {e}")
            return None
