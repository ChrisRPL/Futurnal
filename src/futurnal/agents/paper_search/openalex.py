"""OpenAlex API Client.

OpenAlex is a free, open catalog of the world's scholarly papers.
- 240M+ works indexed
- No API key required
- 100,000 requests/day limit
- Polite pool available with email

API Documentation: https://docs.openalex.org/
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from .models import PaperAuthor, PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class OpenAlexClient:
    """Client for OpenAlex API.

    OpenAlex provides free, open access to scholarly metadata.
    No API key required, but providing an email enables the polite pool
    for faster responses.

    Example:
        client = OpenAlexClient(email="user@example.com")
        results = await client.search("machine learning", limit=10)
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        email: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize OpenAlex client.

        Args:
            email: Optional email for polite pool (faster responses)
            timeout: Request timeout in seconds
        """
        self.email = email
        self.timeout = timeout
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "Futurnal/1.0 (https://github.com/futurnal; mailto:contact@futurnal.ai)",
        }
        logger.info("OpenAlexClient initialized")

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        year_range: Optional[tuple[int, int]] = None,
    ) -> SearchResult:
        """Search for papers.

        Args:
            query: Search query
            limit: Number of results (max 200 per page)
            offset: Pagination offset
            year_range: Optional (start_year, end_year) tuple

        Returns:
            SearchResult with matching papers
        """
        start_time = time.time()

        # Build query parameters
        params: Dict[str, Any] = {
            "search": query,
            "per_page": min(limit, 200),
            "page": (offset // limit) + 1 if limit > 0 else 1,
        }

        # Add email for polite pool
        if self.email:
            params["mailto"] = self.email

        # Build filter for year range
        filters = []
        if year_range:
            filters.append(f"publication_year:{year_range[0]}-{year_range[1]}")

        if filters:
            params["filter"] = ",".join(filters)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/works",
                    params=params,
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAlex API error: {e}")
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                offset=offset,
                limit=limit,
                search_time_ms=int((time.time() - start_time) * 1000),
                source="openalex",
            )

        except Exception as e:
            logger.error(f"OpenAlex request failed: {e}")
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                offset=offset,
                limit=limit,
                search_time_ms=int((time.time() - start_time) * 1000),
                source="openalex",
            )

        # Parse results
        papers = []
        for item in data.get("results", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)

        search_time = int((time.time() - start_time) * 1000)
        total_results = data.get("meta", {}).get("count", len(papers))

        logger.info(
            f"OpenAlex search '{query[:30]}...' returned {len(papers)} papers "
            f"({search_time}ms)"
        )

        return SearchResult(
            query=query,
            papers=papers,
            total_results=total_results,
            offset=offset,
            limit=limit,
            search_time_ms=search_time,
            source="openalex",
        )

    async def get_paper(self, openalex_id: str) -> Optional[PaperMetadata]:
        """Get paper metadata by OpenAlex ID.

        Args:
            openalex_id: OpenAlex work ID (e.g., "W2741809807")

        Returns:
            PaperMetadata or None if not found
        """
        try:
            params = {}
            if self.email:
                params["mailto"] = self.email

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/works/{openalex_id}",
                    params=params,
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()

            return self._parse_paper(data)

        except Exception as e:
            logger.error(f"Failed to get paper {openalex_id}: {e}")
            return None

    def _parse_paper(self, data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Parse paper data from API response."""
        try:
            # Extract OpenAlex ID
            openalex_id = data.get("id", "").replace("https://openalex.org/", "")

            # Extract authors
            authors = []
            for authorship in data.get("authorships", []) or []:
                author_data = authorship.get("author", {})
                if author_data:
                    author_id = author_data.get("id", "").replace("https://openalex.org/", "")
                    author = PaperAuthor(
                        name=author_data.get("display_name", "Unknown"),
                        author_id=author_id if author_id else None,
                    )
                    authors.append(author)

            # Extract abstract from inverted index
            abstract = ""
            abstract_inverted = data.get("abstract_inverted_index")
            if abstract_inverted:
                # Reconstruct abstract from inverted index
                words = []
                for word, positions in abstract_inverted.items():
                    for pos in positions:
                        words.append((pos, word))
                words.sort(key=lambda x: x[0])
                abstract = " ".join(word for _, word in words)

            # Extract PDF URL
            pdf_url = None
            best_oa = data.get("best_oa_location") or {}
            pdf_url = best_oa.get("pdf_url")
            if not pdf_url:
                # Try primary location
                primary = data.get("primary_location") or {}
                pdf_url = primary.get("pdf_url")

            # Extract DOI
            doi = data.get("doi", "")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")

            # Extract venue/source
            venue = None
            primary_location = data.get("primary_location") or {}
            source = primary_location.get("source") or {}
            venue = source.get("display_name")

            # Extract fields of study (concepts in OpenAlex)
            fields = []
            for concept in data.get("concepts", []) or []:
                if concept.get("level", 0) <= 1:  # Only top-level concepts
                    fields.append(concept.get("display_name", ""))

            return PaperMetadata(
                paper_id=openalex_id,
                title=data.get("title", "Untitled") or "Untitled",
                abstract=abstract,
                authors=authors,
                year=data.get("publication_year"),
                venue=venue,
                citation_count=data.get("cited_by_count", 0) or 0,
                reference_count=data.get("referenced_works_count", 0) or 0,
                fields_of_study=fields[:5],  # Limit to top 5
                pdf_url=pdf_url,
                source_url=data.get("id"),
                source="openalex",
                doi=doi if doi else None,
                arxiv_id=None,  # OpenAlex doesn't directly expose arXiv ID
            )

        except Exception as e:
            logger.warning(f"Failed to parse OpenAlex paper data: {e}")
            return None
