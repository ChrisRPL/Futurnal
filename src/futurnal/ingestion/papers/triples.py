"""Semantic triple extraction from academic paper metadata for PKG construction.

This module extracts structured semantic relationships from academic paper metadata
to populate the Personal Knowledge Graph (PKG). Paper triples capture:
- Paper entities and their properties (title, DOI, year)
- Author entities and their affiliations
- Venue and publication relationships
- Research field categorization
- Citation relationships

These triples enable knowledge discovery across academic literature, helping users
understand research landscapes and discover connections between papers and concepts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from futurnal.pipeline.triples import SemanticTriple

if TYPE_CHECKING:
    from .connector import PaperMetadata

logger = logging.getLogger(__name__)


def _create_paper_uri(paper_id: str) -> str:
    """Create a URI for a paper entity."""
    return f"paper:{paper_id}"


def _create_author_uri(author: Dict[str, Any]) -> str:
    """Create a URI for an author entity."""
    if author.get("author_id") or author.get("authorId"):
        author_id = author.get("author_id") or author.get("authorId")
        return f"person:{author_id}"
    # Fall back to name-based URI
    name = author.get("name", "unknown").lower().replace(" ", "_")
    return f"person:{name}"


def _create_venue_uri(venue: str) -> str:
    """Create a URI for a venue entity."""
    venue_slug = venue.lower().replace(" ", "_").replace(".", "")[:50]
    return f"venue:{venue_slug}"


def _create_field_uri(field: str) -> str:
    """Create a URI for a field of study entity."""
    field_slug = field.lower().replace(" ", "_")
    return f"field:{field_slug}"


class PaperTripleExtractor:
    """Extracts semantic triples from academic paper metadata.

    Generates PKG triples representing:
    - Paper entity with properties (title, year, DOI, abstract)
    - Author entities with names
    - Authorship relationships (Paper -> Author)
    - Publication venue relationships (Paper -> Venue)
    - Field of study relationships (Paper -> Field)
    - External ID relationships (DOI, arXiv)
    """

    def extract(self, metadata: "PaperMetadata") -> List[Dict[str, Any]]:
        """Extract semantic triples from paper metadata.

        Args:
            metadata: Paper metadata object.

        Returns:
            List of triple dictionaries for PKG storage.
        """
        triples = []

        paper_uri = _create_paper_uri(metadata.paper_id)
        source_element_id = metadata.paper_id

        # Paper type triple
        triples.append(
            SemanticTriple(
                subject=paper_uri,
                predicate="rdf:type",
                object="futurnal:AcademicPaper",
                source_element_id=source_element_id,
                extraction_method="paper_metadata",
            ).to_dict()
        )

        # Title triple
        if metadata.title:
            triples.append(
                SemanticTriple(
                    subject=paper_uri,
                    predicate="paper:title",
                    object=metadata.title,
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

        # Year triple
        if metadata.year:
            triples.append(
                SemanticTriple(
                    subject=paper_uri,
                    predicate="paper:year",
                    object=str(metadata.year),
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

        # DOI triple
        if metadata.doi:
            triples.append(
                SemanticTriple(
                    subject=paper_uri,
                    predicate="paper:doi",
                    object=metadata.doi,
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

        # arXiv ID triple
        if metadata.arxiv_id:
            triples.append(
                SemanticTriple(
                    subject=paper_uri,
                    predicate="paper:arxivId",
                    object=metadata.arxiv_id,
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

        # Citation count triple
        if metadata.citation_count and metadata.citation_count > 0:
            triples.append(
                SemanticTriple(
                    subject=paper_uri,
                    predicate="paper:citationCount",
                    object=str(metadata.citation_count),
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

        # Abstract triple (truncated for graph storage, full text in vector store)
        if metadata.abstract:
            abstract_preview = metadata.abstract[:500] + "..." if len(metadata.abstract) > 500 else metadata.abstract
            triples.append(
                SemanticTriple(
                    subject=paper_uri,
                    predicate="paper:abstractPreview",
                    object=abstract_preview,
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

        # Author triples
        triples.extend(self._extract_author_triples(paper_uri, metadata.authors, source_element_id))

        # Venue triple
        if metadata.venue:
            triples.extend(self._extract_venue_triples(paper_uri, metadata.venue, source_element_id))

        # Field of study triples
        if metadata.fields_of_study:
            triples.extend(
                self._extract_field_triples(paper_uri, metadata.fields_of_study, source_element_id)
            )

        logger.debug(
            f"Extracted {len(triples)} triples from paper {metadata.paper_id}",
            extra={
                "paper_id": metadata.paper_id,
                "triple_count": len(triples),
                "author_count": len(metadata.authors),
            },
        )

        return triples

    def _extract_author_triples(
        self,
        paper_uri: str,
        authors: List[Dict[str, Any]],
        source_element_id: str,
    ) -> List[Dict[str, Any]]:
        """Extract author-related triples."""
        triples = []

        for idx, author in enumerate(authors):
            author_uri = _create_author_uri(author)
            author_name = author.get("name", "Unknown Author")

            # Author type triple
            triples.append(
                SemanticTriple(
                    subject=author_uri,
                    predicate="rdf:type",
                    object="futurnal:Person",
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

            # Author name triple
            triples.append(
                SemanticTriple(
                    subject=author_uri,
                    predicate="person:name",
                    object=author_name,
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

            # Paper -> Author relationship
            triples.append(
                SemanticTriple(
                    subject=paper_uri,
                    predicate="paper:hasAuthor",
                    object=author_uri,
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

            # Author position (first author, etc.)
            if idx == 0:
                triples.append(
                    SemanticTriple(
                        subject=paper_uri,
                        predicate="paper:firstAuthor",
                        object=author_uri,
                        source_element_id=source_element_id,
                        extraction_method="paper_metadata",
                    ).to_dict()
                )

            # Affiliations if available
            affiliations = author.get("affiliations", [])
            for affiliation in affiliations:
                if affiliation:
                    triples.append(
                        SemanticTriple(
                            subject=author_uri,
                            predicate="person:affiliation",
                            object=affiliation,
                            source_element_id=source_element_id,
                            extraction_method="paper_metadata",
                        ).to_dict()
                    )

        return triples

    def _extract_venue_triples(
        self,
        paper_uri: str,
        venue: str,
        source_element_id: str,
    ) -> List[Dict[str, Any]]:
        """Extract venue-related triples."""
        triples = []
        venue_uri = _create_venue_uri(venue)

        # Venue type triple
        triples.append(
            SemanticTriple(
                subject=venue_uri,
                predicate="rdf:type",
                object="futurnal:Venue",
                source_element_id=source_element_id,
                extraction_method="paper_metadata",
            ).to_dict()
        )

        # Venue name triple
        triples.append(
            SemanticTriple(
                subject=venue_uri,
                predicate="venue:name",
                object=venue,
                source_element_id=source_element_id,
                extraction_method="paper_metadata",
            ).to_dict()
        )

        # Paper -> Venue relationship
        triples.append(
            SemanticTriple(
                subject=paper_uri,
                predicate="paper:publishedIn",
                object=venue_uri,
                source_element_id=source_element_id,
                extraction_method="paper_metadata",
            ).to_dict()
        )

        return triples

    def _extract_field_triples(
        self,
        paper_uri: str,
        fields: List[str],
        source_element_id: str,
    ) -> List[Dict[str, Any]]:
        """Extract field of study triples."""
        triples = []

        for field in fields:
            if not field:
                continue

            field_uri = _create_field_uri(field)

            # Field type triple
            triples.append(
                SemanticTriple(
                    subject=field_uri,
                    predicate="rdf:type",
                    object="futurnal:FieldOfStudy",
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

            # Field name triple
            triples.append(
                SemanticTriple(
                    subject=field_uri,
                    predicate="field:name",
                    object=field,
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

            # Paper -> Field relationship
            triples.append(
                SemanticTriple(
                    subject=paper_uri,
                    predicate="paper:studiesField",
                    object=field_uri,
                    source_element_id=source_element_id,
                    extraction_method="paper_metadata",
                ).to_dict()
            )

        return triples


def extract_paper_triples(metadata: "PaperMetadata") -> List[Dict[str, Any]]:
    """Convenience function to extract triples from paper metadata.

    Args:
        metadata: Paper metadata object.

    Returns:
        List of triple dictionaries for PKG storage.
    """
    extractor = PaperTripleExtractor()
    return extractor.extract(metadata)
