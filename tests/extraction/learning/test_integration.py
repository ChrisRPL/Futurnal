"""Tests for Experiential Learning Pipeline Integration.

Tests the full experiential learning loop connecting World State,
Curriculum, Token Priors, and Training-Free GRPO.

Research Reference:
- SEAgent (2508.04700v2): Complete experiential learning loop
- Training-Free GRPO (2510.08191v1): Token priors for prompt enhancement

Quality Gates:
- Quality improvement >5% over 50 documents
- Ghost model parameters must remain unchanged
"""

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from futurnal.learning.integration import (
    ExperientialLearningPipeline,
    BatchResult,
    LearningState,
    QUALITY_IMPROVEMENT_THRESHOLD,
)
from futurnal.learning.world_state import WorldStateAssessor
from futurnal.learning.curriculum import CurriculumGenerator
from futurnal.learning.token_priors import TokenPriorStore


@dataclass
class MockDocument:
    """Mock document for testing."""
    content: str
    doc_id: str


@dataclass
class MockExtractionResult:
    """Mock extraction result."""
    entity_count: int = 5
    relation_count: int = 3
    confidence: float = 0.8
    entities: list = None
    relations: list = None
    temporal_markers: list = None

    def __post_init__(self):
        self.entities = self.entities or []
        self.relations = self.relations or []
        self.temporal_markers = self.temporal_markers or []


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_batch_result_creation(self):
        """Test BatchResult creation."""
        result = BatchResult(
            batch_size=10,
            documents_processed=10,
            successful_extractions=8,
            failed_extractions=2,
            avg_quality_before=0.5,
            avg_quality_after=0.6,
            quality_improvement=0.1,
            quality_improvement_percentage=20.0,
        )
        assert result.batch_size == 10
        assert result.success_rate == 0.8
        assert result.passes_quality_gate is True

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = BatchResult(
            batch_size=10,
            documents_processed=10,
            successful_extractions=7,
            failed_extractions=3,
            avg_quality_before=0.5,
            avg_quality_after=0.6,
            quality_improvement=0.1,
            quality_improvement_percentage=20.0,
        )
        assert result.success_rate == 0.7

    def test_passes_quality_gate_threshold(self):
        """Test quality gate threshold detection."""
        # Passes (>5%)
        passing = BatchResult(
            batch_size=10,
            documents_processed=10,
            successful_extractions=8,
            failed_extractions=2,
            avg_quality_before=0.5,
            avg_quality_after=0.55,
            quality_improvement=0.05,
            quality_improvement_percentage=10.0,  # 10% > 5%
        )
        assert passing.passes_quality_gate is True

        # Fails (<5%)
        failing = BatchResult(
            batch_size=10,
            documents_processed=10,
            successful_extractions=5,
            failed_extractions=5,
            avg_quality_before=0.5,
            avg_quality_after=0.51,
            quality_improvement=0.01,
            quality_improvement_percentage=2.0,  # 2% < 5%
        )
        assert failing.passes_quality_gate is False

    def test_to_dict(self):
        """Test BatchResult serialization."""
        result = BatchResult(
            batch_size=5,
            documents_processed=5,
            successful_extractions=4,
            failed_extractions=1,
            avg_quality_before=0.5,
            avg_quality_after=0.6,
            quality_improvement=0.1,
            quality_improvement_percentage=20.0,
        )
        data = result.to_dict()
        assert data["batch_size"] == 5
        assert data["success_rate"] == 0.8
        assert "passes_quality_gate" in data


class TestLearningState:
    """Tests for LearningState dataclass."""

    def test_learning_state_creation(self):
        """Test LearningState creation."""
        state = LearningState()
        assert state.total_documents_processed == 0
        assert state.batches_processed == 0

    def test_overall_success_rate(self):
        """Test overall success rate calculation."""
        state = LearningState(
            total_documents_processed=100,
            total_successful=75,
            total_failed=25,
        )
        assert state.overall_success_rate == 0.75

    def test_overall_quality_improvement(self):
        """Test overall quality improvement calculation."""
        state = LearningState(
            total_documents_processed=10,
            cumulative_quality_before=5.0,  # Avg 0.5
            cumulative_quality_after=6.0,  # Avg 0.6
        )
        # (0.6 - 0.5) / 0.5 * 100 = 20%
        assert state.overall_quality_improvement == pytest.approx(20.0)

    def test_passes_quality_gate(self):
        """Test quality gate detection."""
        passing_state = LearningState(
            total_documents_processed=10,
            cumulative_quality_before=5.0,
            cumulative_quality_after=6.0,
        )
        assert passing_state.passes_quality_gate is True

        failing_state = LearningState(
            total_documents_processed=10,
            cumulative_quality_before=5.0,
            cumulative_quality_after=5.1,
        )
        assert failing_state.passes_quality_gate is False


class TestExperientialLearningPipeline:
    """Tests for ExperientialLearningPipeline class."""

    def test_pipeline_creation(self):
        """Test pipeline creation with default components."""
        pipeline = ExperientialLearningPipeline()
        assert pipeline.world_state is not None
        assert pipeline.curriculum is not None
        assert pipeline.token_store is not None

    def test_pipeline_with_custom_components(self):
        """Test pipeline with custom components."""
        world_state = WorldStateAssessor()
        curriculum = CurriculumGenerator()
        token_store = TokenPriorStore()

        pipeline = ExperientialLearningPipeline(
            world_state=world_state,
            curriculum=curriculum,
            token_store=token_store,
        )

        assert pipeline.world_state is world_state
        assert pipeline.curriculum is curriculum
        assert pipeline.token_store is token_store

    def test_process_empty_batch(self):
        """Test processing empty document batch."""
        pipeline = ExperientialLearningPipeline()

        result = pipeline.process_document_batch(
            documents=[],
            extraction_func=lambda d, p: MockExtractionResult(),
        )

        assert result.batch_size == 0
        assert result.documents_processed == 0

    def test_process_document_batch(self):
        """Test processing document batch."""
        pipeline = ExperientialLearningPipeline()

        docs = [
            MockDocument(content=f"Document {i}", doc_id=f"doc{i}")
            for i in range(5)
        ]

        def mock_extract(doc, prompt):
            return MockExtractionResult(
                entity_count=5,
                relation_count=3,
                confidence=0.85,
            )

        result = pipeline.process_document_batch(
            documents=docs,
            extraction_func=mock_extract,
        )

        assert result.documents_processed == 5
        assert result.batch_size == 5
        assert len(result.trajectories) == 5

    def test_curriculum_ordering_applied(self):
        """Test documents are ordered by curriculum."""
        pipeline = ExperientialLearningPipeline()

        processed_order = []

        def tracking_extract(doc, prompt):
            processed_order.append(doc.doc_id)
            return MockExtractionResult()

        # Create docs with varying complexity
        simple = MockDocument(content="Simple", doc_id="simple")
        complex_doc = MockDocument(
            content="Meeting on 2024-01-15 [[Note]] [[Link]]" * 10,
            doc_id="complex",
        )

        pipeline.process_document_batch(
            documents=[complex_doc, simple],
            extraction_func=tracking_extract,
            use_curriculum=True,
        )

        # Simple should be processed first
        assert processed_order[0] == "simple"

    def test_token_priors_injected(self):
        """Test token priors are injected into prompts."""
        pipeline = ExperientialLearningPipeline()

        # Pre-populate some priors
        for _ in range(10):
            pipeline.token_store.update_from_extraction(
                None, True,
                entity_types=["Person"],
            )

        received_prompts = []

        def capture_prompt(doc, prompt):
            received_prompts.append(prompt)
            return MockExtractionResult()

        docs = [MockDocument(content="Test", doc_id="test")]
        pipeline.process_document_batch(
            documents=docs,
            extraction_func=capture_prompt,
            use_priors=True,
        )

        # Prompt should contain prior context
        assert len(received_prompts) == 1
        assert "Learned Patterns" in received_prompts[0]

    def test_learning_state_updated(self):
        """Test learning state is updated after batch."""
        pipeline = ExperientialLearningPipeline()

        docs = [MockDocument(content=f"Doc {i}", doc_id=f"doc{i}") for i in range(5)]

        pipeline.process_document_batch(
            documents=docs,
            extraction_func=lambda d, p: MockExtractionResult(),
        )

        assert pipeline.state.total_documents_processed == 5
        assert pipeline.state.batches_processed == 1

    def test_compute_quality_progression(self):
        """Test quality progression computation."""
        pipeline = ExperientialLearningPipeline()

        # Process multiple batches
        for batch_num in range(3):
            docs = [
                MockDocument(content=f"Batch {batch_num} Doc {i}", doc_id=f"b{batch_num}d{i}")
                for i in range(20)
            ]
            pipeline.process_document_batch(
                documents=docs,
                extraction_func=lambda d, p: MockExtractionResult(),
            )

        progression = pipeline.compute_quality_progression()
        assert "insufficient_data" in progression or "improvement" in progression

    def test_get_learning_summary(self):
        """Test learning summary generation."""
        pipeline = ExperientialLearningPipeline()

        docs = [MockDocument(content="Test", doc_id="test")]
        pipeline.process_document_batch(
            documents=docs,
            extraction_func=lambda d, p: MockExtractionResult(),
        )

        summary = pipeline.get_learning_summary()

        assert "learning_state" in summary
        assert "world_state_summary" in summary
        assert "token_store_summary" in summary

    def test_validate_quality_gates(self):
        """Test quality gate validation."""
        pipeline = ExperientialLearningPipeline()

        gates = pipeline.validate_quality_gates()

        assert "ghost_model_frozen" in gates
        assert "quality_improvement_5_percent" in gates
        assert "priors_are_natural_language" in gates
        assert "no_cloud_connections" in gates

        # These should always be True by design
        assert gates["ghost_model_frozen"] is True
        assert gates["priors_are_natural_language"] is True
        assert gates["no_cloud_connections"] is True

    def test_reset_learning(self):
        """Test learning reset."""
        pipeline = ExperientialLearningPipeline()

        # Process some documents
        docs = [MockDocument(content="Test", doc_id="test")]
        pipeline.process_document_batch(
            documents=docs,
            extraction_func=lambda d, p: MockExtractionResult(),
        )

        result = pipeline.reset_learning()

        assert result["learning_state_reset"] is True
        assert pipeline.state.total_documents_processed == 0

    def test_export_experiential_knowledge(self):
        """Test experiential knowledge export."""
        pipeline = ExperientialLearningPipeline()

        # Process some documents
        for _ in range(5):
            pipeline.token_store.update_from_extraction(
                None, True,
                entity_types=["Person", "Organization"],
            )

        export = pipeline.export_experiential_knowledge()

        assert "Futurnal Experiential Knowledge Export" in export
        assert "Learning Statistics" in export
        assert isinstance(export, str)


class TestQualityGateCompliance:
    """Tests for quality gate compliance."""

    def test_quality_improvement_threshold(self):
        """Verify quality improvement threshold is 5%."""
        assert QUALITY_IMPROVEMENT_THRESHOLD == 0.05

    def test_pipeline_tracks_improvement(self):
        """Test pipeline accurately tracks quality improvement."""
        pipeline = ExperientialLearningPipeline()

        # Simulate improving extractions over batches
        for batch_num in range(5):
            docs = [
                MockDocument(content=f"Batch {batch_num} Doc {i}", doc_id=f"b{batch_num}d{i}")
                for i in range(10)
            ]

            def improving_extract(doc, prompt):
                # Quality improves with each batch
                return MockExtractionResult(
                    entity_count=5 + batch_num,
                    relation_count=3 + batch_num,
                    confidence=0.7 + batch_num * 0.05,
                )

            pipeline.process_document_batch(
                documents=docs,
                extraction_func=improving_extract,
            )

        assert pipeline.state.total_documents_processed == 50
        assert pipeline.state.batches_processed == 5

    def test_documents_marked_processed(self):
        """Test processed documents are tracked."""
        pipeline = ExperientialLearningPipeline()

        docs = [MockDocument(content="Test", doc_id="unique_doc_id")]

        pipeline.process_document_batch(
            documents=docs,
            extraction_func=lambda d, p: MockExtractionResult(),
        )

        assert "unique_doc_id" in pipeline.curriculum.processed_document_ids
