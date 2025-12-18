"""CrossRef API Client.

CrossRef is a DOI registration agency with metadata for 160M+ scholarly works.
- No API key required
- Include email for polite pool (recommended)
- Rich metadata including funding, licenses, references

API Documentation: https://www.crossref.org/documentation/retrieve-metadata/rest-api/
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from .models import PaperAuthor, PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class CrossRefClient:
    """Client for CrossRef API.

    CrossRef provides free access to scholarly metadata registered with DOIs.
    No API key required, but providing an email enables the polite pool.

    Example:
        client = CrossRefClient(email="user@example.com")
        results = await client.search("neural networks", limit=10)
    """

    BASE_URL = "https://api.crossref.org"

    def __init__(
        self,
        email: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize CrossRef client.

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
        logger.info("CrossRefClient initialized")

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
            limit: Number of results (max 1000)
            offset: Pagination offset
            year_range: Optional (start_year, end_year) tuple

        Returns:
            SearchResult with matching papers
        """
        start_time = time.time()

        # Build query parameters
        params: Dict[str, Any] = {
            "query": query,
            "rows": min(limit, 1000),
            "offset": offset,
        }

        # Add email for polite pool
        if self.email:
            params["mailto"] = self.email

        # Add year filter
        filters = []
        if year_range:
            filters.append(f"from-pub-date:{year_range[0]}")
            filters.append(f"until-pub-date:{year_range[1]}")

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
            logger.error(f"CrossRef API error: {e}")
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                offset=offset,
                limit=limit,
                search_time_ms=int((time.time() - start_time) * 1000),
                source="crossref",
            )

        except Exception as e:
            logger.error(f"CrossRef request failed: {e}")
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                offset=offset,
                limit=limit,
                search_time_ms=int((time.time() - start_time) * 1000),
                source="crossref",
            )

        # Parse results
        papers = []
        message = data.get("message", {})
        for item in message.get("items", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)

        search_time = int((time.time() - start_time) * 1000)
        total_results = message.get("total-results", len(papers))

        logger.info(
            f"CrossRef search '{query[:30]}...' returned {len(papers)} papers "
            f"({search_time}ms)"
        )

        return SearchResult(
            query=query,
            papers=papers,
            total_results=total_results,
            offset=offset,
            limit=limit,
            search_time_ms=search_time,
            source="crossref",
        )

    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """Get paper metadata by DOI.

        Args:
            doi: Digital Object Identifier (e.g., "10.1038/nature12373")

        Returns:
            PaperMetadata or None if not found
        """
        try:
            params = {}
            if self.email:
                params["mailto"] = self.email

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.BASE_URL}/works/{doi}",
                    params=params,
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()

            return self._parse_paper(data.get("message", {}))

        except Exception as e:
            logger.error(f"Failed to get paper by DOI {doi}: {e}")
            return None

    def _parse_paper(self, data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Parse paper data from API response."""
        try:
            # Extract DOI
            doi = data.get("DOI", "")

            # Extract title
            titles = data.get("title", [])
            title = titles[0] if titles else "Untitled"

            # Extract authors
            authors = []
            for author_data in data.get("author", []) or []:
                given = author_data.get("given", "")
                family = author_data.get("family", "")
                name = f"{given} {family}".strip() or "Unknown"

                # CrossRef uses ORCID as author ID
                orcid = author_data.get("ORCID", "")
                author_id = orcid.replace("http://orcid.org/", "").replace("https://orcid.org/", "") if orcid else None

                author = PaperAuthor(
                    name=name,
                    author_id=author_id,
                )
                authors.append(author)

            # Extract year from published date
            year = None
            published = data.get("published") or data.get("created") or {}
            date_parts = published.get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]

            # Extract abstract
            abstract = data.get("abstract", "") or ""
            # Clean HTML tags from abstract
            if abstract:
                import re
                abstract = re.sub(r'<[^>]+>', '', abstract)

            # Extract venue
            venue = None
            container_titles = data.get("container-title", [])
            if container_titles:
                venue = container_titles[0]

            # Extract PDF URL (from links)
            pdf_url = None
            for link in data.get("link", []) or []:
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL")
                    break

            # Extract fields of study (subjects in CrossRef)
            fields = data.get("subject", []) or []

            # Extract citation count
            citation_count = data.get("is-referenced-by-count", 0) or 0

            # Extract reference count
            reference_count = data.get("references-count", 0) or 0

            return PaperMetadata(
                paper_id=f"crossref:{doi}" if doi else f"crossref:{hash(title)}",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                venue=venue,
                citation_count=citation_count,
                reference_count=reference_count,
                fields_of_study=fields[:5],  # Limit to top 5
                pdf_url=pdf_url,
                source_url=f"https://doi.org/{doi}" if doi else None,
                source="crossref",
                doi=doi if doi else None,
                arxiv_id=None,
            )

        except Exception as e:
            logger.warning(f"Failed to parse CrossRef paper data: {e}")
            return None
