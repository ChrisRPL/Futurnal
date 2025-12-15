"""Tests for Token Prior Store.

Tests experiential knowledge storage as natural language token priors
per Training-Free GRPO research.

Research Reference:
- Training-Free GRPO (2510.08191v1): Experiential knowledge as token priors

CRITICAL Quality Gates:
- All priors must be stored as TEXT (natural language), NOT numerical weights
- Ghost model must remain frozen (no parameter updates)
"""

import pytest
from unittest.mock import MagicMock

from futurnal.learning.token_priors import (
    EntityTypePrior,
    RelationTypePrior,
    TemporalPatternPrior,
    TokenPriorStore,
)


class TestEntityTypePrior:
    """Tests for EntityTypePrior dataclass."""

    def test_prior_creation(self):
        """Test basic EntityTypePrior creation."""
        prior = EntityTypePrior(
            entity_type="Person",
            frequency=10,
            confidence=0.8,
            context_pattern="Person entities appear as proper nouns",
        )
        assert prior.entity_type == "Person"
        assert prior.frequency == 10
        assert prior.confidence == 0.8

    def test_update_success(self):
        """Test success updates prior correctly."""
        prior = EntityTypePrior(
            entity_type="Person",
            success_count=5,
            failure_count=5,
        )
        prior.update_success()
        assert prior.success_count == 6
        assert prior.frequency == 1
        assert prior.confidence == pytest.approx(6 / 11)

    def test_update_failure(self):
        """Test failure updates prior correctly."""
        prior = EntityTypePrior(
            entity_type="Person",
            success_count=5,
            failure_count=5,
        )
        prior.update_failure()
        assert prior.failure_count == 6
        assert prior.confidence == pytest.approx(5 / 11)

    def test_to_natural_language(self):
        """Test prior converts to natural language."""
        prior = EntityTypePrior(
            entity_type="Person",
            frequency=10,
            confidence=0.85,
            context_pattern="Appears as proper nouns",
            examples=["John Smith", "Jane Doe"],
        )
        nl = prior.to_natural_language()
        assert "Person" in nl
        assert "85%" in nl
        assert "Appears as proper nouns" in nl
        assert isinstance(nl, str)

    def test_prior_is_natural_language_not_numerical(self):
        """CRITICAL: Verify prior stores natural language, not numbers."""
        prior = EntityTypePrior(
            entity_type="Person",
            context_pattern="Person entities appear frequently in personal notes",
        )
        # context_pattern must be a string (natural language)
        assert isinstance(prior.context_pattern, str)
        assert len(prior.context_pattern) > 10  # Must be meaningful text

    def test_to_dict(self):
        """Test prior serialization."""
        prior = EntityTypePrior(entity_type="Person", frequency=5)
        data = prior.to_dict()
        assert data["entity_type"] == "Person"
        assert data["frequency"] == 5
        assert "context_pattern" in data


class TestRelationTypePrior:
    """Tests for RelationTypePrior dataclass."""

    def test_prior_creation(self):
        """Test basic RelationTypePrior creation."""
        prior = RelationTypePrior(
            relation_type="works_at",
            subject_types=["Person"],
            object_types=["Organization"],
            context_pattern="Connects person to employer",
        )
        assert prior.relation_type == "works_at"
        assert "Person" in prior.subject_types
        assert "Organization" in prior.object_types

    def test_to_natural_language(self):
        """Test prior converts to natural language."""
        prior = RelationTypePrior(
            relation_type="works_at",
            frequency=5,
            confidence=0.9,
            subject_types=["Person"],
            object_types=["Organization"],
            context_pattern="Employment relationship",
            examples=["John works at Acme"],
        )
        nl = prior.to_natural_language()
        assert "works_at" in nl
        assert "Person" in nl
        assert "Organization" in nl
        assert isinstance(nl, str)


class TestTemporalPatternPrior:
    """Tests for TemporalPatternPrior dataclass."""

    def test_prior_creation(self):
        """Test basic TemporalPatternPrior creation."""
        prior = TemporalPatternPrior(
            pattern_type="explicit_date",
            extraction_guidance="Look for YYYY-MM-DD format",
        )
        assert prior.pattern_type == "explicit_date"

    def test_to_natural_language(self):
        """Test prior converts to natural language."""
        prior = TemporalPatternPrior(
            pattern_type="explicit_date",
            frequency=20,
            confidence=0.95,
            extraction_guidance="Look for dates in YYYY-MM-DD format",
            examples=["2024-01-15", "2024-02-20"],
        )
        nl = prior.to_natural_language()
        assert "explicit_date" in nl
        assert "YYYY-MM-DD" in nl
        assert isinstance(nl, str)


class TestTokenPriorStore:
    """Tests for TokenPriorStore class."""

    def test_store_creation(self):
        """Test TokenPriorStore creation."""
        store = TokenPriorStore()
        assert store.capacity == 100
        assert len(store.entity_priors) == 0
        assert len(store.relation_priors) == 0
        assert len(store.temporal_priors) == 0

    def test_update_from_extraction_creates_priors(self):
        """Test updating store creates new priors."""
        store = TokenPriorStore()
        store.update_from_extraction(
            extraction_result=None,
            success=True,
            entity_types=["Person", "Organization"],
            relation_types=["works_at"],
            temporal_patterns=["explicit_date"],
        )

        assert "Person" in store.entity_priors
        assert "Organization" in store.entity_priors
        assert "works_at" in store.relation_priors
        assert "explicit_date" in store.temporal_priors

    def test_update_success_increases_confidence(self):
        """Test successful updates increase confidence."""
        store = TokenPriorStore()

        # Initial update
        store.update_from_extraction(
            extraction_result=None,
            success=True,
            entity_types=["Person"],
        )
        initial_confidence = store.entity_priors["Person"].confidence

        # Multiple successful updates
        for _ in range(5):
            store.update_from_extraction(
                extraction_result=None,
                success=True,
                entity_types=["Person"],
            )

        assert store.entity_priors["Person"].confidence >= initial_confidence

    def test_update_failure_decreases_confidence(self):
        """Test failed updates decrease confidence."""
        store = TokenPriorStore()

        # Start with success
        for _ in range(5):
            store.update_from_extraction(
                extraction_result=None,
                success=True,
                entity_types=["Person"],
            )
        high_confidence = store.entity_priors["Person"].confidence

        # Now failures
        for _ in range(5):
            store.update_from_extraction(
                extraction_result=None,
                success=False,
                entity_types=["Person"],
            )

        assert store.entity_priors["Person"].confidence < high_confidence

    def test_generate_prompt_context(self):
        """Test prompt context generation."""
        store = TokenPriorStore()
        store.update_from_extraction(
            extraction_result=None,
            success=True,
            entity_types=["Person", "Organization"],
            relation_types=["works_at"],
            temporal_patterns=["explicit_date"],
        )

        # Increase confidence with more successes
        for _ in range(5):
            store.update_from_extraction(
                extraction_result=None,
                success=True,
                entity_types=["Person"],
            )

        context = store.generate_prompt_context()

        assert "Learned Patterns" in context
        assert "Entity Types" in context
        assert "Person" in context
        assert isinstance(context, str)

    def test_prompt_context_is_natural_language(self):
        """CRITICAL: Verify prompt context is natural language, not numerical."""
        store = TokenPriorStore()

        for _ in range(10):
            store.update_from_extraction(
                extraction_result=None,
                success=True,
                entity_types=["Person"],
                temporal_patterns=["causal_sequence"],
            )

        context = store.generate_prompt_context()

        # Must be readable text
        assert isinstance(context, str)
        assert len(context) > 50  # Meaningful content
        # Should contain explanatory text, not just numbers
        assert any(word in context.lower() for word in ["pattern", "entity", "extraction", "learned"])

    def test_prune_low_confidence_priors(self):
        """Test pruning removes low confidence priors."""
        store = TokenPriorStore(min_confidence=0.6)

        # Add high confidence prior
        for _ in range(10):
            store.update_from_extraction(
                extraction_result=None,
                success=True,
                entity_types=["HighConf"],
            )

        # Add low confidence prior (lots of failures)
        for _ in range(10):
            store.update_from_extraction(
                extraction_result=None,
                success=False,
                entity_types=["LowConf"],
            )

        removed = store.prune_low_confidence_priors()

        assert "HighConf" in store.entity_priors
        assert "LowConf" not in store.entity_priors
        assert removed >= 1

    def test_add_example(self):
        """Test adding examples to priors."""
        store = TokenPriorStore()
        store.update_from_extraction(
            extraction_result=None,
            success=True,
            entity_types=["Person"],
        )

        success = store.add_example("entity", "Person", "John Smith")
        assert success is True
        assert "John Smith" in store.entity_priors["Person"].examples

    def test_update_context_pattern(self):
        """Test updating context pattern."""
        store = TokenPriorStore()
        store.update_from_extraction(
            extraction_result=None,
            success=True,
            entity_types=["Person"],
        )

        success = store.update_context_pattern(
            "entity",
            "Person",
            "Person entities appear as capitalized names",
        )
        assert success is True
        assert "capitalized" in store.entity_priors["Person"].context_pattern

    def test_export_as_natural_language(self):
        """Test full export to natural language."""
        store = TokenPriorStore()

        for _ in range(5):
            store.update_from_extraction(
                extraction_result=None,
                success=True,
                entity_types=["Person", "Organization"],
                relation_types=["works_at"],
                temporal_patterns=["explicit_date"],
            )

        export = store.export_as_natural_language()

        assert "Experiential Knowledge Export" in export
        assert "Entity Type Patterns" in export
        assert "Relationship Patterns" in export
        assert "Temporal Patterns" in export
        assert isinstance(export, str)

    def test_get_summary(self):
        """Test summary generation."""
        store = TokenPriorStore()
        store.update_from_extraction(
            extraction_result=None,
            success=True,
            entity_types=["Person"],
        )

        summary = store.get_summary()
        assert summary["entity_prior_count"] == 1
        assert summary["total_updates"] == 1

    def test_clear(self):
        """Test clearing all priors."""
        store = TokenPriorStore()
        store.update_from_extraction(
            extraction_result=None,
            success=True,
            entity_types=["Person"],
        )

        store.clear()

        assert len(store.entity_priors) == 0
        assert store.total_updates == 0

    def test_capacity_pruning(self):
        """Test priors are pruned when over capacity."""
        store = TokenPriorStore(capacity=3)

        # Add more than capacity
        for i in range(5):
            store.update_from_extraction(
                extraction_result=None,
                success=True,
                entity_types=[f"Type{i}"],
            )

        assert len(store.entity_priors) <= 3


class TestTokenPriorStoreIntegration:
    """Integration tests for TokenPriorStore with mock results."""

    def test_extract_from_mock_result(self):
        """Test extracting priors from mock extraction result."""
        store = TokenPriorStore()

        # Create mock result with entities
        mock_result = MagicMock()
        mock_result.entities = [
            MagicMock(type="Person"),
            MagicMock(type="Organization"),
        ]
        mock_result.relations = []
        mock_result.temporal_markers = []

        store.update_from_extraction(mock_result, success=True)

        assert "Person" in store.entity_priors
        assert "Organization" in store.entity_priors


class TestAllPriorsAreNaturalLanguage:
    """CRITICAL tests ensuring all priors are natural language strings."""

    def test_entity_prior_context_is_string(self):
        """Entity prior context_pattern must be string."""
        store = TokenPriorStore()
        store.update_from_extraction(None, True, entity_types=["Person"])

        prior = store.entity_priors["Person"]
        assert isinstance(prior.context_pattern, str)
        assert not isinstance(prior.context_pattern, (int, float, bytes))

    def test_relation_prior_context_is_string(self):
        """Relation prior context_pattern must be string."""
        store = TokenPriorStore()
        store.update_from_extraction(None, True, relation_types=["works_at"])

        prior = store.relation_priors["works_at"]
        assert isinstance(prior.context_pattern, str)

    def test_temporal_prior_guidance_is_string(self):
        """Temporal prior extraction_guidance must be string."""
        store = TokenPriorStore()
        store.update_from_extraction(None, True, temporal_patterns=["explicit_date"])

        prior = store.temporal_priors["explicit_date"]
        assert isinstance(prior.extraction_guidance, str)
        assert len(prior.extraction_guidance) > 0

    def test_generated_context_contains_no_raw_numbers(self):
        """Generated prompt context should be readable, not raw numbers."""
        store = TokenPriorStore()

        for _ in range(10):
            store.update_from_extraction(
                None, True,
                entity_types=["Person"],
                relation_types=["works_at"],
                temporal_patterns=["explicit_date"],
            )

        context = store.generate_prompt_context()

        # Context should have words, not just numbers
        words = context.split()
        word_count = sum(1 for w in words if w.isalpha())
        number_count = sum(1 for w in words if w.replace(".", "").isdigit())

        assert word_count > number_count  # More words than numbers
