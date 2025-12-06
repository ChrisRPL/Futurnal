"""Tests for Query Understanding Templates.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/04-query-routing-orchestration.md

Success Metrics:
- Template selection works for all intent types
- Template evolution via textual gradients
- Success rate tracking
- Composition with extraction templates
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from futurnal.search.hybrid.types import QueryIntent
from futurnal.search.hybrid.routing.templates import (
    QueryTemplate,
    QueryTemplateDatabase,
    SEED_TEMPLATES,
)


class TestQueryTemplate:
    """Tests for QueryTemplate model."""

    def test_template_creation(self):
        """Template creates with required fields."""
        template = QueryTemplate(
            template_id="test_template_v1",
            name="Test Template",
            intent_type=QueryIntent.TEMPORAL,
            pattern="Test pattern instructions",
        )

        assert template.template_id == "test_template_v1"
        assert template.name == "Test Template"
        assert template.intent_type == QueryIntent.TEMPORAL
        assert template.pattern == "Test pattern instructions"
        assert template.version == 1
        assert template.success_rate == 0.0

    def test_template_with_all_fields(self):
        """Template accepts all optional fields."""
        created = datetime(2024, 1, 1, 12, 0, 0)
        updated = datetime(2024, 1, 15, 12, 0, 0)

        template = QueryTemplate(
            template_id="advanced_v3",
            name="Advanced Template",
            intent_type=QueryIntent.CAUSAL,
            pattern="Advanced causal analysis pattern",
            version=3,
            success_rate=0.85,
            created_at=created,
            last_updated=updated,
        )

        assert template.version == 3
        assert template.success_rate == 0.85
        assert template.created_at == created
        assert template.last_updated == updated

    def test_template_success_rate_bounds(self):
        """Success rate is bounded 0-1."""
        # Valid values
        template_low = QueryTemplate(
            template_id="t1",
            name="Low",
            intent_type=QueryIntent.LOOKUP,
            pattern="p",
            success_rate=0.0,
        )
        template_high = QueryTemplate(
            template_id="t2",
            name="High",
            intent_type=QueryIntent.LOOKUP,
            pattern="p",
            success_rate=1.0,
        )

        assert template_low.success_rate == 0.0
        assert template_high.success_rate == 1.0


class TestSeedTemplates:
    """Tests for seed template definitions."""

    def test_all_intents_have_seed_templates(self):
        """Seed templates exist for all query intents."""
        for intent in QueryIntent:
            assert intent in SEED_TEMPLATES, f"Missing seed template for {intent}"

    def test_temporal_template_content(self):
        """Temporal template has expected content."""
        template = SEED_TEMPLATES[QueryIntent.TEMPORAL]

        assert template.intent_type == QueryIntent.TEMPORAL
        assert "time references" in template.pattern.lower()
        assert "temporal" in template.pattern.lower()

    def test_causal_template_content(self):
        """Causal template has expected content."""
        template = SEED_TEMPLATES[QueryIntent.CAUSAL]

        assert template.intent_type == QueryIntent.CAUSAL
        assert "cause" in template.pattern.lower()
        assert "effect" in template.pattern.lower()

    def test_lookup_template_content(self):
        """Lookup template has expected content."""
        template = SEED_TEMPLATES[QueryIntent.LOOKUP]

        assert template.intent_type == QueryIntent.LOOKUP
        assert "entity" in template.pattern.lower()

    def test_exploratory_template_content(self):
        """Exploratory template has expected content."""
        template = SEED_TEMPLATES[QueryIntent.EXPLORATORY]

        assert template.intent_type == QueryIntent.EXPLORATORY
        assert "exploration" in template.pattern.lower() or "exploratory" in template.pattern.lower()


class TestQueryTemplateDatabase:
    """Tests for QueryTemplateDatabase."""

    def test_initialization(self):
        """Database initializes with seed templates."""
        db = QueryTemplateDatabase()

        assert len(db.templates) == len(QueryIntent)
        for intent in QueryIntent:
            assert intent in db.templates

    def test_initialization_with_parent_db(self):
        """Database initializes with parent template database."""
        parent_db = MagicMock()
        db = QueryTemplateDatabase(parent_template_db=parent_db)

        assert db.parent_db is parent_db

    def test_select_template_temporal(self):
        """Selects temporal template for temporal intent."""
        db = QueryTemplateDatabase()

        template = db.select_template(QueryIntent.TEMPORAL)

        assert template.intent_type == QueryIntent.TEMPORAL
        assert "temporal" in template.template_id.lower()

    def test_select_template_causal(self):
        """Selects causal template for causal intent."""
        db = QueryTemplateDatabase()

        template = db.select_template(QueryIntent.CAUSAL)

        assert template.intent_type == QueryIntent.CAUSAL
        assert "causal" in template.template_id.lower()

    def test_select_template_lookup(self):
        """Selects lookup template for lookup intent."""
        db = QueryTemplateDatabase()

        template = db.select_template(QueryIntent.LOOKUP)

        assert template.intent_type == QueryIntent.LOOKUP
        assert "lookup" in template.template_id.lower()

    def test_select_template_exploratory(self):
        """Selects exploratory template for exploratory intent."""
        db = QueryTemplateDatabase()

        template = db.select_template(QueryIntent.EXPLORATORY)

        assert template.intent_type == QueryIntent.EXPLORATORY
        assert "exploratory" in template.template_id.lower()

    def test_get_template_by_id(self):
        """Gets template by ID."""
        db = QueryTemplateDatabase()

        template = db.get_template("temporal_query_v1")

        assert template is not None
        assert template.template_id == "temporal_query_v1"

    def test_get_template_not_found(self):
        """Returns None for unknown template ID."""
        db = QueryTemplateDatabase()

        template = db.get_template("nonexistent_template")

        assert template is None


class TestTemplateEvolution:
    """Tests for template evolution via textual gradients."""

    def test_update_template(self):
        """Template updates with new pattern."""
        db = QueryTemplateDatabase()
        original = db.select_template(QueryIntent.TEMPORAL)
        original_version = original.version

        db.update_template(
            intent=QueryIntent.TEMPORAL,
            new_pattern="Updated temporal analysis pattern",
            feedback="Improved handling of relative dates",
        )

        updated = db.select_template(QueryIntent.TEMPORAL)
        assert updated.version == original_version + 1
        assert updated.pattern == "Updated temporal analysis pattern"
        assert updated.success_rate == 0.0  # Reset for new version

    def test_update_template_preserves_name(self):
        """Update preserves template name."""
        db = QueryTemplateDatabase()
        original_name = db.select_template(QueryIntent.CAUSAL).name

        db.update_template(
            intent=QueryIntent.CAUSAL,
            new_pattern="New causal pattern",
            feedback="Test feedback",
        )

        updated = db.select_template(QueryIntent.CAUSAL)
        assert updated.name == original_name

    def test_update_nonexistent_intent(self):
        """Update handles nonexistent intent gracefully."""
        db = QueryTemplateDatabase()
        # Clear templates to test edge case
        db.templates.clear()

        # Should not raise
        db.update_template(
            intent=QueryIntent.TEMPORAL,
            new_pattern="Pattern",
            feedback="Feedback",
        )

    def test_record_success(self):
        """Success recording updates success rate."""
        db = QueryTemplateDatabase()
        original_rate = db.select_template(QueryIntent.LOOKUP).success_rate

        db.record_success(QueryIntent.LOOKUP, success=True)

        updated_rate = db.select_template(QueryIntent.LOOKUP).success_rate
        assert updated_rate > original_rate

    def test_record_failure(self):
        """Failure recording doesn't increase success rate from zero."""
        db = QueryTemplateDatabase()

        db.record_success(QueryIntent.LOOKUP, success=False)

        rate = db.select_template(QueryIntent.LOOKUP).success_rate
        assert rate == 0.0

    def test_record_success_exponential_moving_average(self):
        """Success rate uses EMA calculation."""
        db = QueryTemplateDatabase()

        # Record multiple successes
        for _ in range(10):
            db.record_success(QueryIntent.TEMPORAL, success=True)

        rate = db.select_template(QueryIntent.TEMPORAL).success_rate
        # With alpha=0.1, after 10 successes rate should be significant but < 1.0
        assert 0.5 < rate < 1.0


class TestTemplateManagement:
    """Tests for template management operations."""

    def test_get_all_templates(self):
        """Gets all templates."""
        db = QueryTemplateDatabase()

        templates = db.get_all_templates()

        assert len(templates) == len(QueryIntent)
        assert QueryIntent.TEMPORAL in templates
        assert QueryIntent.CAUSAL in templates

    def test_get_all_templates_returns_copy(self):
        """get_all_templates returns a copy."""
        db = QueryTemplateDatabase()

        templates = db.get_all_templates()
        templates.clear()

        assert len(db.templates) == len(QueryIntent)

    def test_get_template_stats(self):
        """Gets statistics for all templates."""
        db = QueryTemplateDatabase()

        stats = db.get_template_stats()

        assert len(stats) == len(QueryIntent)
        for template_id, stat in stats.items():
            assert "intent" in stat
            assert "version" in stat
            assert "success_rate" in stat
            assert "created_at" in stat
            assert "last_updated" in stat

    def test_reset_to_seed(self):
        """Reset restores seed templates."""
        db = QueryTemplateDatabase()

        # Modify a template
        db.update_template(
            intent=QueryIntent.TEMPORAL,
            new_pattern="Modified pattern",
            feedback="Test",
        )

        modified = db.select_template(QueryIntent.TEMPORAL)
        assert modified.version > 1

        # Reset
        db.reset_to_seed()

        reset = db.select_template(QueryIntent.TEMPORAL)
        assert reset.version == 1


class TestTemplateComposition:
    """Tests for template composition with extraction templates."""

    def test_compose_without_parent_db(self):
        """Composition without parent returns query template only."""
        db = QueryTemplateDatabase()

        composed = db.compose_with_extraction_template(
            intent=QueryIntent.TEMPORAL,
            extraction_template_name="entity_extraction",
        )

        # Should just be the query template pattern
        template = db.select_template(QueryIntent.TEMPORAL)
        assert composed == template.pattern

    def test_compose_with_parent_db(self):
        """Composition with parent includes extraction template."""
        parent_db = MagicMock()
        extraction_template = MagicMock()
        extraction_template.pattern = "Extract entities from text"
        parent_db.select_template.return_value = extraction_template

        db = QueryTemplateDatabase(parent_template_db=parent_db)

        composed = db.compose_with_extraction_template(
            intent=QueryIntent.LOOKUP,
            extraction_template_name="entity_extraction",
        )

        assert "Extract entities from text" in composed
        assert "# Extraction Pattern" in composed

    def test_compose_with_parent_db_exception(self):
        """Composition handles parent db exceptions gracefully."""
        parent_db = MagicMock()
        parent_db.select_template.side_effect = Exception("Template not found")

        db = QueryTemplateDatabase(parent_template_db=parent_db)

        # Should not raise
        composed = db.compose_with_extraction_template(
            intent=QueryIntent.CAUSAL,
            extraction_template_name="nonexistent",
        )

        # Should return just the query template
        template = db.select_template(QueryIntent.CAUSAL)
        assert composed == template.pattern
