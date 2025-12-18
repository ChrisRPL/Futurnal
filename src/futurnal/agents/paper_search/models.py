"""Data models for Paper Search Agent.

Phase D: Paper Agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PaperAuthor:
    """Represents a paper author."""

    name: str
    author_id: Optional[str] = None
    affiliations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "author_id": self.author_id,
            "affiliations": self.affiliations,
        }


@dataclass
class PaperMetadata:
    """Metadata for an academic paper.

    Attributes:
        paper_id: Unique identifier (DOI, arXiv ID, or Semantic Scholar ID)
        title: Paper title
        abstract: Paper abstract
        authors: List of authors
        year: Publication year
        venue: Publication venue (journal/conference)
        citation_count: Number of citations
        reference_count: Number of references
        fields_of_study: Research areas
        pdf_url: Direct link to PDF if available
        source_url: Link to paper page
        source: Where this metadata came from ('semantic_scholar', 'arxiv')
        doi: Digital Object Identifier
        arxiv_id: arXiv identifier if applicable
    """

    paper_id: str
    title: str
    abstract: str = ""
    authors: List[PaperAuthor] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: int = 0
    reference_count: int = 0
    fields_of_study: List[str] = field(default_factory=list)
    pdf_url: Optional[str] = None
    source_url: Optional[str] = None
    source: str = "semantic_scholar"
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [a.to_dict() for a in self.authors],
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "fields_of_study": self.fields_of_study,
            "pdf_url": self.pdf_url,
            "source_url": self.source_url,
            "source": self.source,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
        }

    @property
    def author_names(self) -> List[str]:
        """Get list of author names."""
        return [a.name for a in self.authors]

    @property
    def short_authors(self) -> str:
        """Get abbreviated author string (First et al. or First & Second)."""
        if not self.authors:
            return "Unknown"
        if len(self.authors) == 1:
            return self.authors[0].name.split()[-1]
        if len(self.authors) == 2:
            return f"{self.authors[0].name.split()[-1]} & {self.authors[1].name.split()[-1]}"
        return f"{self.authors[0].name.split()[-1]} et al."


@dataclass
class SearchResult:
    """Result from a paper search query.

    Attributes:
        query: The original search query
        papers: List of matching papers
        total_results: Total number of results available
        offset: Current offset in result set
        limit: Number of results requested
        search_time_ms: Time taken for search in milliseconds
        source: Search source ('semantic_scholar', 'arxiv', 'combined')
    """

    query: str
    papers: List[PaperMetadata] = field(default_factory=list)
    total_results: int = 0
    offset: int = 0
    limit: int = 10
    search_time_ms: int = 0
    source: str = "semantic_scholar"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "papers": [p.to_dict() for p in self.papers],
            "total_results": self.total_results,
            "offset": self.offset,
            "limit": self.limit,
            "search_time_ms": self.search_time_ms,
            "source": self.source,
        }


@dataclass
class DownloadedPaper:
    """Represents a downloaded paper.

    Attributes:
        metadata: Paper metadata
        local_path: Path to downloaded PDF
        downloaded_at: When the paper was downloaded
        file_size_bytes: Size of downloaded file
        processed: Whether the paper has been processed by pipeline
    """

    metadata: PaperMetadata
    local_path: str
    downloaded_at: datetime = field(default_factory=datetime.utcnow)
    file_size_bytes: int = 0
    processed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "local_path": self.local_path,
            "downloaded_at": self.downloaded_at.isoformat(),
            "file_size_bytes": self.file_size_bytes,
            "processed": self.processed,
        }
