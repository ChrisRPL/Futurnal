"""Tests for paper semantic triple extraction.

Tests cover:
- Paper entity triple generation
- Author relationship triples
- Venue relationship triples
- Field of study triples
- Metadata preservation
- Edge cases (missing fields, empty lists)
"""

from __future__ import annotations

import pytest

from futurnal.ingestion.papers.connector import PaperMetadata
from futurnal.ingestion.papers.triples import PaperTripleExtractor


@pytest.fixture
def extractor():
    """Create a PaperTripleExtractor instance."""
    return PaperTripleExtractor()


def _create_test_paper_metadata(**overrides) -> PaperMetadata:
    """Create a test PaperMetadata with default values."""
    defaults = {
        "paper_id": "test123",
        "title": "Test Paper Title",
        "authors": [
            {"name": "John Doe", "authorId": "auth1"},
            {"name": "Jane Smith", "authorId": "auth2"},
        ],
        "year": 2024,
        "venue": "Conference on Machine Learning",
        "doi": "10.1234/test.2024",
        "arxiv_id": "2401.12345",
        "abstract": "This is a test abstract.",
        "citation_count": 42,
        "fields_of_study": ["Machine Learning", "Natural Language Processing"],
        "pdf_url": "https://example.com/paper.pdf",
    }
    defaults.update(overrides)
    return PaperMetadata(**defaults)


def test_extract_basic_paper_triples(extractor):
    """Test extraction of basic paper entity triples."""
    metadata = _create_test_paper_metadata()
    triples = extractor.extract(metadata)

    # Convert to dict for easier testing
    triples_dict = {(t["subject"], t["predicate"], t["object"]) for t in triples}

    # Paper type triple (uses paper:id format and AcademicPaper type)
    assert (
        "paper:test123",
        "rdf:type",
        "futurnal:AcademicPaper",
    ) in triples_dict

    # Title triple
    assert (
        "paper:test123",
        "paper:title",
        "Test Paper Title",
    ) in triples_dict

    # Year triple
    assert (
        "paper:test123",
        "paper:year",
        "2024",
    ) in triples_dict


def test_extract_author_triples(extractor):
    """Test extraction of author relationship triples."""
    metadata = _create_test_paper_metadata(
        authors=[
            {"name": "Alice Johnson", "authorId": "alice123"},
            {"name": "Bob Williams"},  # No author ID
        ]
    )
    triples = extractor.extract(metadata)

    triples_dict = {(t["subject"], t["predicate"], t["object"]) for t in triples}

    # Paper-author relationships (uses person:id format)
    assert (
        "paper:test123",
        "paper:hasAuthor",
        "person:alice123",
    ) in triples_dict

    # Person type triples
    assert (
        "person:alice123",
        "rdf:type",
        "futurnal:Person",
    ) in triples_dict

    # Person name triples
    assert (
        "person:alice123",
        "person:name",
        "Alice Johnson",
    ) in triples_dict


def test_extract_venue_triple(extractor):
    """Test extraction of venue relationship triple."""
    metadata = _create_test_paper_metadata(venue="NeurIPS 2024")
    triples = extractor.extract(metadata)

    triples_dict = {(t["subject"], t["predicate"], t["object"]) for t in triples}

    # Venue triple (uses venue URI)
    venue_uri = "venue:neurips_2024"
    assert (
        "paper:test123",
        "paper:publishedIn",
        venue_uri,
    ) in triples_dict

    # Venue name triple
    assert (
        venue_uri,
        "venue:name",
        "NeurIPS 2024",
    ) in triples_dict


def test_extract_field_of_study_triples(extractor):
    """Test extraction of field of study triples."""
    metadata = _create_test_paper_metadata(
        fields_of_study=["Computer Vision", "Deep Learning", "Transformers"]
    )
    triples = extractor.extract(metadata)

    triples_dict = {(t["subject"], t["predicate"], t["object"]) for t in triples}

    # Field of study triples (uses field URI)
    assert (
        "paper:test123",
        "paper:studiesField",
        "field:computer_vision",
    ) in triples_dict

    assert (
        "paper:test123",
        "paper:studiesField",
        "field:deep_learning",
    ) in triples_dict

    assert (
        "paper:test123",
        "paper:studiesField",
        "field:transformers",
    ) in triples_dict

    # Field name triples
    assert (
        "field:computer_vision",
        "field:name",
        "Computer Vision",
    ) in triples_dict


def test_extract_doi_triple(extractor):
    """Test extraction of DOI triple."""
    metadata = _create_test_paper_metadata(doi="10.1234/example.2024")
    triples = extractor.extract(metadata)

    triples_dict = {(t["subject"], t["predicate"], t["object"]) for t in triples}

    # DOI triple
    assert (
        "paper:test123",
        "paper:doi",
        "10.1234/example.2024",
    ) in triples_dict


def test_extract_arxiv_triple(extractor):
    """Test extraction of arXiv ID triple."""
    metadata = _create_test_paper_metadata(arxiv_id="2401.99999")
    triples = extractor.extract(metadata)

    triples_dict = {(t["subject"], t["predicate"], t["object"]) for t in triples}

    # arXiv ID triple
    assert (
        "paper:test123",
        "paper:arxivId",
        "2401.99999",
    ) in triples_dict


def test_extract_citation_count_triple(extractor):
    """Test extraction of citation count triple."""
    metadata = _create_test_paper_metadata(citation_count=100)
    triples = extractor.extract(metadata)

    triples_dict = {(t["subject"], t["predicate"], t["object"]) for t in triples}

    # Citation count triple
    assert (
        "paper:test123",
        "paper:citationCount",
        "100",
    ) in triples_dict


def test_no_venue_triple_when_missing(extractor):
    """Test no venue triple when venue is None."""
    metadata = _create_test_paper_metadata(venue=None)
    triples = extractor.extract(metadata)

    venue_triples = [t for t in triples if t["predicate"] == "paper:publishedIn"]
    assert len(venue_triples) == 0


def test_no_field_of_study_triples_when_empty(extractor):
    """Test no field of study triples when list is empty."""
    metadata = _create_test_paper_metadata(fields_of_study=[])
    triples = extractor.extract(metadata)

    field_triples = [t for t in triples if t["predicate"] == "paper:studiesField"]
    assert len(field_triples) == 0


def test_no_doi_triple_when_missing(extractor):
    """Test no DOI triple when DOI is None."""
    metadata = _create_test_paper_metadata(doi=None)
    triples = extractor.extract(metadata)

    doi_triples = [t for t in triples if t["predicate"] == "paper:doi"]
    assert len(doi_triples) == 0


def test_author_without_id_generates_name_uri(extractor):
    """Test that authors without IDs get name-based URIs."""
    metadata = _create_test_paper_metadata(
        authors=[
            {"name": "Anonymous Author"},  # No authorId
        ]
    )
    triples = extractor.extract(metadata)

    # Find the author relationship
    author_triples = [t for t in triples if t["predicate"] == "paper:hasAuthor"]
    assert len(author_triples) == 1

    # Author URI should be based on name
    author_uri = author_triples[0]["object"]
    assert author_uri == "person:anonymous_author"


def test_empty_authors_list(extractor):
    """Test handling of empty authors list."""
    metadata = _create_test_paper_metadata(authors=[])
    triples = extractor.extract(metadata)

    # Should not have author triples
    author_triples = [t for t in triples if t["predicate"] == "paper:hasAuthor"]
    assert len(author_triples) == 0


def test_paper_uri_creation(extractor):
    """Test paper URI is created correctly."""
    metadata = _create_test_paper_metadata(paper_id="complex-id-with-chars")
    triples = extractor.extract(metadata)

    # Find the paper type triple
    type_triples = [t for t in triples if t["predicate"] == "rdf:type" and t["object"] == "futurnal:AcademicPaper"]
    assert len(type_triples) == 1

    # Paper URI should use paper: prefix
    paper_triple = type_triples[0]
    assert paper_triple["subject"] == "paper:complex-id-with-chars"


def test_full_metadata_extraction(extractor):
    """Test extraction with complete metadata."""
    metadata = _create_test_paper_metadata(
        paper_id="full-test",
        title="A Comprehensive Study on Knowledge Graphs",
        authors=[
            {"name": "Dr. First Author", "authorId": "first1"},
            {"name": "Prof. Second Author", "authorId": "second2"},
            {"name": "Graduate Student", "authorId": "student3"},
        ],
        year=2024,
        venue="ICLR 2024",
        doi="10.5555/knowledge.2024",
        arxiv_id="2401.00001",
        abstract="A comprehensive study...",
        citation_count=150,
        fields_of_study=["Knowledge Graphs", "Semantic Web", "AI"],
    )
    triples = extractor.extract(metadata)

    # Should have many triples
    assert len(triples) > 15

    # Verify categories
    paper_triples = [t for t in triples if t["subject"].startswith("paper:")]
    person_triples = [t for t in triples if t["subject"].startswith("person:")]

    assert len(paper_triples) > 10
    assert len(person_triples) >= 3  # At least one per author


def test_triple_structure(extractor):
    """Test that triples have correct structure."""
    metadata = _create_test_paper_metadata()
    triples = extractor.extract(metadata)

    # All triples should have required fields
    for triple in triples:
        assert "subject" in triple
        assert "predicate" in triple
        assert "object" in triple
        assert triple["subject"]  # Not empty
        assert triple["predicate"]  # Not empty
        assert triple["object"]  # Not empty


def test_triples_are_serializable(extractor):
    """Test that triples can be serialized to JSON."""
    import json

    metadata = _create_test_paper_metadata()
    triples = extractor.extract(metadata)

    # All triples should be JSON-serializable
    for triple in triples:
        json_str = json.dumps(triple)
        assert json_str  # Non-empty JSON string
        parsed = json.loads(json_str)
        assert parsed == triple


def test_multiple_papers_extraction(extractor):
    """Test extraction for multiple papers produces unique URIs."""
    metadata1 = _create_test_paper_metadata(paper_id="paper1", title="Paper One")
    metadata2 = _create_test_paper_metadata(paper_id="paper2", title="Paper Two")

    triples1 = extractor.extract(metadata1)
    triples2 = extractor.extract(metadata2)

    # Get paper URIs
    paper1_uri = next(
        t["subject"] for t in triples1
        if t["predicate"] == "rdf:type" and t["object"] == "futurnal:AcademicPaper"
    )
    paper2_uri = next(
        t["subject"] for t in triples2
        if t["predicate"] == "rdf:type" and t["object"] == "futurnal:AcademicPaper"
    )

    assert paper1_uri != paper2_uri
    assert paper1_uri == "paper:paper1"
    assert paper2_uri == "paper:paper2"


def test_first_author_triple(extractor):
    """Test that first author is marked specially."""
    metadata = _create_test_paper_metadata(
        authors=[
            {"name": "First Author", "authorId": "first1"},
            {"name": "Second Author", "authorId": "second2"},
        ]
    )
    triples = extractor.extract(metadata)

    # Should have first author triple
    first_author_triples = [t for t in triples if t["predicate"] == "paper:firstAuthor"]
    assert len(first_author_triples) == 1
    assert first_author_triples[0]["object"] == "person:first1"


def test_abstract_preview_triple(extractor):
    """Test that abstract is included as preview."""
    metadata = _create_test_paper_metadata(abstract="This is the abstract content.")
    triples = extractor.extract(metadata)

    abstract_triples = [t for t in triples if t["predicate"] == "paper:abstractPreview"]
    assert len(abstract_triples) == 1
    assert abstract_triples[0]["object"] == "This is the abstract content."
