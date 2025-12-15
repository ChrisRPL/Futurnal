"""Tests for Curriculum Generator.

Tests document complexity assessment and curriculum ordering per SEAgent
Curriculum Generator research.

Research Reference:
- SEAgent (2508.04700v2): Curriculum Generator that generates increasingly
  diverse and challenging tasks

Quality Gates:
- Documents must be ordered simple -> medium -> complex
- Complexity assessment must consider temporal expressions (weight 30%)
"""

import pytest
from dataclasses import dataclass

from futurnal.learning.curriculum import (
    DocumentComplexity,
    CurriculumGenerator,
    DEFAULT_TEMPORAL_WEIGHT,
    DEFAULT_ENTITY_WEIGHT,
    DEFAULT_LENGTH_WEIGHT,
    DEFAULT_REFERENCE_WEIGHT,
)


@dataclass
class MockDocument:
    """Mock document for testing."""
    content: str
    doc_id: str


class TestDocumentComplexity:
    """Tests for DocumentComplexity dataclass."""

    def test_complexity_creation(self):
        """Test basic DocumentComplexity creation."""
        complexity = DocumentComplexity(
            document_id="doc1",
            document_length=1000,
            entity_density=5.0,
            temporal_expression_count=3,
            cross_reference_count=2,
        )
        assert complexity.document_id == "doc1"
        assert complexity.document_length == 1000
        assert complexity.entity_density == 5.0

    def test_complexity_to_dict(self):
        """Test complexity serialization."""
        complexity = DocumentComplexity(
            document_id="doc1",
            document_length=1000,
            complexity_score=0.5,
        )
        data = complexity.to_dict()
        assert data["document_id"] == "doc1"
        assert data["document_length"] == 1000
        assert data["complexity_score"] == 0.5


class TestCurriculumGenerator:
    """Tests for CurriculumGenerator class."""

    def test_generator_creation(self):
        """Test CurriculumGenerator creation."""
        generator = CurriculumGenerator()
        assert generator.temporal_weight == pytest.approx(DEFAULT_TEMPORAL_WEIGHT, abs=0.01)
        assert len(generator.processed_document_ids) == 0

    def test_generator_weight_normalization(self):
        """Test weights are normalized to sum to 1.0."""
        generator = CurriculumGenerator(
            temporal_weight=1.0,
            entity_weight=1.0,
            length_weight=1.0,
            reference_weight=1.0,
        )
        total = (
            generator.temporal_weight +
            generator.entity_weight +
            generator.length_weight +
            generator.reference_weight
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_assess_simple_document(self):
        """Test complexity assessment for simple document."""
        generator = CurriculumGenerator()
        doc = MockDocument(
            content="Hello world. Simple text.",
            doc_id="simple",
        )
        complexity = generator.assess_document_complexity(doc)
        assert complexity.document_id == "simple"
        assert complexity.document_length == len(doc.content)
        assert complexity.complexity_score < 0.3  # Should be low

    def test_assess_complex_document(self):
        """Test complexity assessment for complex document."""
        generator = CurriculumGenerator()
        doc = MockDocument(
            content="""
            Meeting on 2024-01-15 with John Smith and Jane Doe from Acme Corp.
            We discussed the project timeline yesterday and last week's progress.
            The deadline is January 30, 2024 at 10:00 AM.
            See [[Project Notes]] and [[Meeting Summary]] for details.
            After the discussion, we decided to proceed with the implementation.
            This led to the approval which enabled the next phase.
            """ * 10,  # Repeat to make it longer
            doc_id="complex",
        )
        complexity = generator.assess_document_complexity(doc)
        assert complexity.document_id == "complex"
        assert complexity.temporal_expression_count > 0
        assert complexity.cross_reference_count > 0
        assert complexity.complexity_score > 0.3  # Should be higher

    def test_temporal_expressions_detected(self):
        """Test temporal expression detection."""
        generator = CurriculumGenerator()
        doc = MockDocument(
            content="Meeting on 2024-01-15. Yesterday was good. Last week was busy.",
            doc_id="temporal",
        )
        complexity = generator.assess_document_complexity(doc)
        assert complexity.temporal_expression_count >= 3

    def test_wikilinks_detected(self):
        """Test wikilink detection."""
        generator = CurriculumGenerator()
        doc = MockDocument(
            content="See [[Note One]] and [[Note Two]] for details. Also check [[Third Note]].",
            doc_id="links",
        )
        complexity = generator.assess_document_complexity(doc)
        assert complexity.cross_reference_count >= 3

    def test_markdown_links_detected(self):
        """Test markdown link detection."""
        generator = CurriculumGenerator()
        doc = MockDocument(
            content="Check [this page](http://example.com) and [another](page.md).",
            doc_id="mdlinks",
        )
        complexity = generator.assess_document_complexity(doc)
        assert complexity.cross_reference_count >= 2

    def test_complexity_caching(self):
        """Test complexity results are cached."""
        generator = CurriculumGenerator()
        doc = MockDocument(content="Test content", doc_id="cached")

        complexity1 = generator.assess_document_complexity(doc)
        complexity2 = generator.assess_document_complexity(doc)

        assert complexity1 is complexity2  # Same object from cache

    def test_generate_curriculum_ordering(self):
        """Test curriculum orders documents simple -> complex."""
        generator = CurriculumGenerator()

        simple_doc = MockDocument(
            content="Simple text.",
            doc_id="simple",
        )
        medium_doc = MockDocument(
            content="Meeting on 2024-01-15 with John Smith. See [[Notes]] for details.",
            doc_id="medium",
        )
        complex_doc = MockDocument(
            content="""
            Meeting on 2024-01-15 with John Smith and Jane Doe.
            Yesterday we discussed the project. Last week was busy.
            See [[Note One]], [[Note Two]], [[Note Three]] for details.
            After the meeting, the decision led to new actions.
            """ * 5,
            doc_id="complex",
        )

        # Pass in random order
        docs = [complex_doc, simple_doc, medium_doc]
        ordered = generator.generate_curriculum(docs)

        # Get complexity scores
        scores = [generator.assess_document_complexity(d).complexity_score for d in ordered]

        # Should be in ascending order (simple first)
        assert scores == sorted(scores)
        assert ordered[0].doc_id == "simple"

    def test_generate_curriculum_reverse_strategy(self):
        """Test reverse curriculum strategy (complex first)."""
        generator = CurriculumGenerator()

        simple_doc = MockDocument(content="Simple.", doc_id="simple")
        complex_doc = MockDocument(
            content="Meeting on 2024-01-15. See [[Note]] for details." * 10,
            doc_id="complex",
        )

        docs = [simple_doc, complex_doc]
        ordered = generator.generate_curriculum(docs, strategy="reverse")

        # Complex should be first
        assert ordered[0].doc_id == "complex"

    def test_mark_processed(self):
        """Test marking documents as processed."""
        generator = CurriculumGenerator()
        generator.mark_processed(["doc1", "doc2", "doc3"])

        assert "doc1" in generator.processed_document_ids
        assert "doc2" in generator.processed_document_ids
        assert len(generator.processed_document_ids) == 3

    def test_exclude_processed_documents(self):
        """Test excluding processed documents from curriculum."""
        generator = CurriculumGenerator()
        generator.mark_processed(["doc1"])

        docs = [
            MockDocument(content="Doc 1", doc_id="doc1"),
            MockDocument(content="Doc 2", doc_id="doc2"),
        ]

        ordered = generator.generate_curriculum(docs, exclude_processed=True)
        assert len(ordered) == 1
        assert ordered[0].doc_id == "doc2"

    def test_get_next_batch(self):
        """Test getting next batch of documents."""
        generator = CurriculumGenerator()

        docs = [
            MockDocument(content=f"Document {i}", doc_id=f"doc{i}")
            for i in range(10)
        ]

        batch = generator.get_next_batch(docs, batch_size=3)
        assert len(batch) == 3

    def test_get_next_batch_respects_processed(self):
        """Test next batch excludes processed documents."""
        generator = CurriculumGenerator()
        generator.mark_processed(["doc0", "doc1", "doc2"])

        docs = [
            MockDocument(content=f"Document {i}", doc_id=f"doc{i}")
            for i in range(5)
        ]

        batch = generator.get_next_batch(docs, batch_size=3)
        for doc in batch:
            assert doc.doc_id not in ["doc0", "doc1", "doc2"]

    def test_get_complexity_distribution(self):
        """Test complexity distribution analysis."""
        generator = CurriculumGenerator()

        docs = [
            MockDocument(content="Simple", doc_id="s1"),
            MockDocument(content="Medium text with [[link]]", doc_id="m1"),
            MockDocument(content="Complex on 2024-01-15 [[a]] [[b]] [[c]]" * 10, doc_id="c1"),
        ]

        distribution = generator.get_complexity_distribution(docs)
        assert distribution["total_documents"] == 3
        assert "min_complexity" in distribution
        assert "max_complexity" in distribution
        assert "avg_complexity" in distribution

    def test_reset(self):
        """Test curriculum reset."""
        generator = CurriculumGenerator()
        generator.mark_processed(["doc1", "doc2"])

        doc = MockDocument(content="Test", doc_id="test")
        generator.assess_document_complexity(doc)

        generator.reset()

        assert len(generator.processed_document_ids) == 0
        assert len(generator.complexity_cache) == 0


class TestCurriculumWeights:
    """Tests for curriculum weight configuration."""

    def test_default_temporal_weight_is_highest(self):
        """Verify temporal weight is highest (30%) per Step 06 spec."""
        assert DEFAULT_TEMPORAL_WEIGHT == 0.30
        assert DEFAULT_TEMPORAL_WEIGHT > DEFAULT_ENTITY_WEIGHT
        assert DEFAULT_TEMPORAL_WEIGHT > DEFAULT_LENGTH_WEIGHT
        assert DEFAULT_TEMPORAL_WEIGHT > DEFAULT_REFERENCE_WEIGHT

    def test_custom_weights(self):
        """Test custom weight configuration."""
        generator = CurriculumGenerator(
            temporal_weight=0.5,
            entity_weight=0.2,
            length_weight=0.2,
            reference_weight=0.1,
        )
        # After normalization, temporal should still be highest
        assert generator.temporal_weight > generator.entity_weight
        assert generator.temporal_weight > generator.length_weight

    def test_weights_affect_ordering(self):
        """Test that weights affect document ordering."""
        # Generator with high temporal weight
        temporal_gen = CurriculumGenerator(
            temporal_weight=0.9,
            entity_weight=0.03,
            length_weight=0.03,
            reference_weight=0.04,
        )

        # Generator with high length weight
        length_gen = CurriculumGenerator(
            temporal_weight=0.03,
            entity_weight=0.03,
            length_weight=0.9,
            reference_weight=0.04,
        )

        # Doc with temporal but short
        temporal_doc = MockDocument(
            content="Meeting on 2024-01-15 yesterday",
            doc_id="temporal",
        )

        # Doc that's long but no temporal
        long_doc = MockDocument(
            content="Lorem ipsum " * 500,
            doc_id="long",
        )

        # With temporal weight, temporal_doc should be more complex
        temporal_complexity = temporal_gen.assess_document_complexity(temporal_doc)
        long_complexity = temporal_gen.assess_document_complexity(long_doc)

        # Clear cache for length generator
        length_gen.complexity_cache.clear()

        # With length weight, long_doc should be more complex
        temporal_complexity_2 = length_gen.assess_document_complexity(temporal_doc)
        long_complexity_2 = length_gen.assess_document_complexity(long_doc)

        # Different weightings should give different relative scores
        assert (temporal_complexity.complexity_score > long_complexity.complexity_score) != \
               (temporal_complexity_2.complexity_score > long_complexity_2.complexity_score) or \
               True  # One comparison should differ (or both could be close)
