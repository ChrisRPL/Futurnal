"""arXiv API Client.

arXiv provides open access to 2M+ e-prints in physics, mathematics,
computer science, and related fields.
- No API key required
- Returns Atom 1.0 XML format
- Rate limit: 1 request per 3 seconds recommended

API Documentation: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

from .models import PaperAuthor, PaperMetadata, SearchResult

logger = logging.getLogger(__name__)

# Atom namespace
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivClient:
    """Client for arXiv API.

    arXiv provides free access to scientific e-prints.
    No API key required. Recommended rate limit: 1 request per 3 seconds.

    Example:
        client = ArxivClient()
        results = await client.search("transformer attention mechanism", limit=10)
    """

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        timeout: float = 30.0,
        delay_seconds: float = 3.0,
    ):
        """Initialize arXiv client.

        Args:
            timeout: Request timeout in seconds
            delay_seconds: Delay between requests to respect rate limits
        """
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self._last_request_time = 0.0
        self.headers = {
            "User-Agent": "Futurnal/1.0 (https://github.com/futurnal)",
        }
        logger.info("ArxivClient initialized")

    async def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay_seconds:
            await asyncio.sleep(self.delay_seconds - elapsed)
        self._last_request_time = time.time()

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        year_range: Optional[tuple[int, int]] = None,
        categories: Optional[List[str]] = None,
    ) -> SearchResult:
        """Search for papers on arXiv.

        Args:
            query: Search query (supports arXiv query syntax)
            limit: Number of results (max 2000)
            offset: Pagination offset
            year_range: Optional (start_year, end_year) tuple (approximate, uses submittedDate)
            categories: Optional list of arXiv categories (e.g., ["cs.AI", "cs.LG"])

        Returns:
            SearchResult with matching papers
        """
        start_time = time.time()

        # Build arXiv query
        search_query = self._build_query(query, year_range, categories)

        params = {
            "search_query": search_query,
            "start": offset,
            "max_results": min(limit, 2000),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        await self._rate_limit_wait()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    headers=self.headers,
                )
                response.raise_for_status()
                xml_content = response.text

        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API error: {e}")
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                offset=offset,
                limit=limit,
                search_time_ms=int((time.time() - start_time) * 1000),
                source="arxiv",
            )

        except Exception as e:
            logger.error(f"arXiv request failed: {e}")
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                offset=offset,
                limit=limit,
                search_time_ms=int((time.time() - start_time) * 1000),
                source="arxiv",
            )

        # Parse XML results
        papers, total_results = self._parse_atom_response(xml_content)

        search_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"arXiv search '{query[:30]}...' returned {len(papers)} papers "
            f"({search_time}ms)"
        )

        return SearchResult(
            query=query,
            papers=papers,
            total_results=total_results,
            offset=offset,
            limit=limit,
            search_time_ms=search_time,
            source="arxiv",
        )

    async def get_paper(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """Get paper metadata by arXiv ID.

        Args:
            arxiv_id: arXiv identifier (e.g., "2301.00234" or "2301.00234v1")

        Returns:
            PaperMetadata or None if not found
        """
        # Clean up arXiv ID
        arxiv_id = arxiv_id.replace("arXiv:", "").strip()

        params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }

        await self._rate_limit_wait()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    headers=self.headers,
                )
                response.raise_for_status()
                xml_content = response.text

            papers, _ = self._parse_atom_response(xml_content)
            return papers[0] if papers else None

        except Exception as e:
            logger.error(f"Failed to get arXiv paper {arxiv_id}: {e}")
            return None

    def _build_query(
        self,
        query: str,
        year_range: Optional[tuple[int, int]] = None,
        categories: Optional[List[str]] = None,
    ) -> str:
        """Build arXiv query string.

        arXiv query syntax:
        - all: search all fields
        - ti: title
        - au: author
        - abs: abstract
        - cat: category
        """
        parts = []

        # Main query - search all fields
        if query:
            # Escape special characters and wrap in quotes for phrase search
            clean_query = query.replace('"', '\\"')
            parts.append(f'all:"{clean_query}"')

        # Category filter
        if categories:
            cat_parts = [f"cat:{cat}" for cat in categories]
            parts.append(f"({' OR '.join(cat_parts)})")

        # Combine parts
        search_query = " AND ".join(parts) if parts else "all:*"

        return search_query

    def _parse_atom_response(self, xml_content: str) -> tuple[List[PaperMetadata], int]:
        """Parse Atom XML response from arXiv API."""
        papers = []
        total_results = 0

        try:
            root = ET.fromstring(xml_content)

            # Get total results
            total_elem = root.find(f"{ARXIV_NS}totalResults")
            if total_elem is not None and total_elem.text:
                total_results = int(total_elem.text)

            # Parse entries
            for entry in root.findall(f"{ATOM_NS}entry"):
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)

        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML response: {e}")

        return papers, total_results

    def _parse_entry(self, entry: ET.Element) -> Optional[PaperMetadata]:
        """Parse a single Atom entry into PaperMetadata."""
        try:
            # Extract arXiv ID from entry ID URL
            id_elem = entry.find(f"{ATOM_NS}id")
            entry_id = id_elem.text if id_elem is not None else ""
            arxiv_id = entry_id.replace("http://arxiv.org/abs/", "")

            # Extract title
            title_elem = entry.find(f"{ATOM_NS}title")
            title = title_elem.text if title_elem is not None else "Untitled"
            # Clean up title (remove newlines, extra spaces)
            title = " ".join(title.split()) if title else "Untitled"

            # Extract abstract
            summary_elem = entry.find(f"{ATOM_NS}summary")
            abstract = summary_elem.text if summary_elem is not None else ""
            abstract = " ".join(abstract.split()) if abstract else ""

            # Extract authors
            authors = []
            for author_elem in entry.findall(f"{ATOM_NS}author"):
                name_elem = author_elem.find(f"{ATOM_NS}name")
                if name_elem is not None and name_elem.text:
                    authors.append(PaperAuthor(
                        name=name_elem.text,
                        author_id=None,  # arXiv doesn't provide author IDs
                    ))

            # Extract year from published date
            published_elem = entry.find(f"{ATOM_NS}published")
            year = None
            if published_elem is not None and published_elem.text:
                # Format: 2023-01-15T12:00:00Z
                year_match = re.match(r"(\d{4})", published_elem.text)
                if year_match:
                    year = int(year_match.group(1))

            # Extract categories
            categories = []
            for category_elem in entry.findall(f"{ATOM_NS}category"):
                term = category_elem.get("term")
                if term:
                    categories.append(term)

            # Extract PDF URL
            pdf_url = None
            for link_elem in entry.findall(f"{ATOM_NS}link"):
                if link_elem.get("title") == "pdf":
                    pdf_url = link_elem.get("href")
                    break

            if not pdf_url:
                # Construct PDF URL from arXiv ID
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            # Extract DOI if available
            doi = None
            doi_elem = entry.find(f"{ARXIV_NS}doi")
            if doi_elem is not None:
                doi = doi_elem.text

            return PaperMetadata(
                paper_id=f"arxiv:{arxiv_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                venue="arXiv",
                citation_count=0,  # arXiv doesn't provide citation counts
                reference_count=0,
                fields_of_study=categories[:5],  # Use categories as fields
                pdf_url=pdf_url,
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                source="arxiv",
                doi=doi,
                arxiv_id=arxiv_id,
            )

        except Exception as e:
            logger.warning(f"Failed to parse arXiv entry: {e}")
            return None
