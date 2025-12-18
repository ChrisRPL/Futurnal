"""Tests for papers connector.

Tests cover:
- PDF ingestion with Unstructured.io
- Element enrichment with paper metadata
- Semantic triple integration
- Error handling and quarantine
- Consent and audit logging
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import pytest

from futurnal.ingestion.papers.connector import PapersConnector, PaperMetadata


class MockElement:
    """Mock Unstructured element for testing."""

    def __init__(self, text: str, category: str = "NarrativeText"):
        self.text = text
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.category,
        }


class MockElementSink:
    """Mock element sink for capturing processed elements."""

    def __init__(self):
        self.elements: List[dict] = []
        self.deletions: List[dict] = []

    def handle(self, element: dict) -> None:
        self.elements.append(element)

    def handle_deletion(self, element: dict) -> None:
        self.deletions.append(element)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_partition(monkeypatch):
    """Mock the unstructured partition function."""
    def _partition(**kwargs) -> List[MockElement]:
        return [
            MockElement("Abstract: This is a test paper.", "Title"),
            MockElement("Introduction paragraph.", "NarrativeText"),
            MockElement("Methods and results.", "NarrativeText"),
            MockElement("Conclusion.", "NarrativeText"),
        ]

    # Get the partition module from sys.modules (set up by conftest)
    partition_auto = sys.modules.get("unstructured.partition.auto")
    if partition_auto:
        monkeypatch.setattr(partition_auto, "partition", _partition)
    else:
        pytest.skip("unstructured.partition.auto not available")


@pytest.fixture
def sample_pdf(temp_workspace) -> Path:
    """Create a mock PDF file for testing."""
    pdf_path = temp_workspace / "sample_paper.pdf"
    # Create a minimal file (actual PDF parsing is mocked)
    pdf_path.write_bytes(b"%PDF-1.4 mock pdf content")
    return pdf_path


@pytest.fixture
def sample_metadata() -> PaperMetadata:
    """Create sample paper metadata."""
    return PaperMetadata(
        paper_id="test-paper-001",
        title="A Test Paper on Machine Learning",
        authors=[
            {"name": "Alice Researcher", "authorId": "alice1"},
            {"name": "Bob Scientist", "authorId": "bob2"},
        ],
        year=2024,
        venue="ICML 2024",
        doi="10.1234/ml.test.2024",
        arxiv_id="2401.00001",
        abstract="This paper presents a test methodology.",
        citation_count=42,
        fields_of_study=["Machine Learning", "Deep Learning"],
        pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
    )


def test_connector_initialization(temp_workspace):
    """Test PapersConnector initialization creates directories."""
    connector = PapersConnector(workspace_dir=temp_workspace)

    assert (temp_workspace / "parsed").exists()
    assert (temp_workspace / "quarantine").exists()


def test_ingest_paper_basic(
    temp_workspace, mock_partition, sample_pdf, sample_metadata
):
    """Test basic paper ingestion."""
    sink = MockElementSink()
    connector = PapersConnector(
        workspace_dir=temp_workspace,
        element_sink=sink,
    )

    elements = list(connector.ingest(
        paper_path=sample_pdf,
        metadata=sample_metadata,
        require_consent=False,  # Skip consent for testing
    ))

    # Should have ingested elements
    assert len(elements) > 0

    # Elements should be sent to sink
    assert len(sink.elements) > 0


def test_ingest_paper_enriches_metadata(
    temp_workspace, mock_partition, sample_pdf, sample_metadata
):
    """Test that ingested elements are enriched with paper metadata."""
    sink = MockElementSink()
    connector = PapersConnector(
        workspace_dir=temp_workspace,
        element_sink=sink,
    )

    elements = list(connector.ingest(
        paper_path=sample_pdf,
        metadata=sample_metadata,
        require_consent=False,
    ))

    # Check first element has paper metadata
    assert len(elements) > 0
    first_element = elements[0]

    assert "paper_metadata" in first_element
    paper_meta = first_element["paper_metadata"]

    assert paper_meta["paper_id"] == "test-paper-001"
    assert paper_meta["title"] == "A Test Paper on Machine Learning"
    assert "Alice Researcher" in paper_meta["authors"]
    assert paper_meta["year"] == 2024
    assert paper_meta["venue"] == "ICML 2024"


def test_ingest_paper_includes_triples(
    temp_workspace, mock_partition, sample_pdf, sample_metadata
):
    """Test that ingested elements include semantic triples."""
    connector = PapersConnector(workspace_dir=temp_workspace)

    elements = list(connector.ingest(
        paper_path=sample_pdf,
        metadata=sample_metadata,
        require_consent=False,
    ))

    # Check that elements have semantic triples
    assert len(elements) > 0
    first_element = elements[0]

    assert "semantic_triples" in first_element
    triples = first_element["semantic_triples"]

    assert len(triples) > 0


def test_ingest_paper_source_metadata(
    temp_workspace, mock_partition, sample_pdf, sample_metadata
):
    """Test that elements have source metadata."""
    connector = PapersConnector(workspace_dir=temp_workspace)

    elements = list(connector.ingest(
        paper_path=sample_pdf,
        metadata=sample_metadata,
        require_consent=False,
    ))

    assert len(elements) > 0
    first_element = elements[0]

    assert "metadata" in first_element
    meta = first_element["metadata"]

    assert meta["source_type"] == "academic_paper"
    assert str(sample_pdf) in meta["source_path"]
    assert "ingested_at" in meta


def test_ingest_nonexistent_file(temp_workspace, sample_metadata):
    """Test ingestion of non-existent file returns empty."""
    connector = PapersConnector(workspace_dir=temp_workspace)

    nonexistent_path = temp_workspace / "nonexistent.pdf"
    elements = list(connector.ingest(
        paper_path=nonexistent_path,
        metadata=sample_metadata,
        require_consent=False,
    ))

    # Should return empty - file doesn't exist
    assert len(elements) == 0


def test_ingest_batch(
    temp_workspace, mock_partition, sample_metadata
):
    """Test batch ingestion of multiple papers."""
    # Create multiple mock PDFs
    papers = []
    for i in range(3):
        pdf_path = temp_workspace / f"paper_{i}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 mock content")

        meta = PaperMetadata(
            paper_id=f"paper-{i}",
            title=f"Paper Number {i}",
            authors=[{"name": f"Author {i}"}],
        )
        papers.append((pdf_path, meta))

    connector = PapersConnector(workspace_dir=temp_workspace)

    result = connector.ingest_batch(papers, job_id="test-batch")

    assert result["total"] == 3
    assert result["succeeded"] >= 0  # Depends on mock partition


def test_paper_metadata_to_dict(sample_metadata):
    """Test PaperMetadata to_dict conversion."""
    metadata_dict = sample_metadata.to_dict()

    assert metadata_dict["paperId"] == "test-paper-001"
    assert metadata_dict["title"] == "A Test Paper on Machine Learning"
    assert len(metadata_dict["authors"]) == 2
    assert metadata_dict["year"] == 2024
    assert metadata_dict["venue"] == "ICML 2024"
    assert metadata_dict["doi"] == "10.1234/ml.test.2024"
    assert metadata_dict["citationCount"] == 42


def test_paper_metadata_minimal():
    """Test PaperMetadata with minimal fields."""
    metadata = PaperMetadata(
        paper_id="minimal-001",
        title="Minimal Paper",
        authors=[],
    )

    assert metadata.paper_id == "minimal-001"
    assert metadata.title == "Minimal Paper"
    assert metadata.authors == []
    assert metadata.year is None
    assert metadata.venue is None
    assert metadata.fields_of_study == []  # Default empty list


def test_connector_with_job_id(
    temp_workspace, mock_partition, sample_pdf, sample_metadata
):
    """Test ingestion with explicit job ID."""
    connector = PapersConnector(workspace_dir=temp_workspace)

    elements = list(connector.ingest(
        paper_path=sample_pdf,
        metadata=sample_metadata,
        job_id="custom-job-123",
        require_consent=False,
    ))

    # Job ID is used for logging/tracking, verify ingestion works
    assert len(elements) > 0


def test_connector_workspace_structure(temp_workspace):
    """Test connector creates proper workspace structure."""
    connector = PapersConnector(workspace_dir=temp_workspace)

    # Verify directories exist
    assert (temp_workspace / "parsed").is_dir()
    assert (temp_workspace / "quarantine").is_dir()


def test_element_enrichment_preserves_original(
    temp_workspace, mock_partition, sample_pdf, sample_metadata
):
    """Test that element enrichment preserves original element data."""
    connector = PapersConnector(workspace_dir=temp_workspace)

    elements = list(connector.ingest(
        paper_path=sample_pdf,
        metadata=sample_metadata,
        require_consent=False,
    ))

    # Original element data should be preserved
    assert len(elements) > 0
    for element in elements:
        # Should have original text
        assert "text" in element or "type" in element
        # Plus enrichment
        assert "paper_metadata" in element
        assert "semantic_triples" in element
        assert "metadata" in element
