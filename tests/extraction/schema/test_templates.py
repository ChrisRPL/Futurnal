"""
Tests for Thought Template System (Module 04)

Comprehensive test suite validating:
- Template database with 10+ seed templates
- Template selection and composition
- Textual gradient refinement (KEEP/FIX/DISCARD)
- Template evolution and versioning
- Integration with experiential learning and schema evolution
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from futurnal.extraction.schema.models import (
    TemplateStats,
    TextualGradient,
    ThoughtTemplate,
)
from futurnal.extraction.schema.templates import (
    TemplateDatabase,
    TemplateRefinementEngine,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM client for testing."""
    llm = Mock()
    llm.extract.return_value = {"entities": ["test_entity"]}
    llm.introspect.return_value = json.dumps({
        "decision": "KEEP",
        "feedback": "Template works well with consistent results",
        "proposed_change": None,
        "confidence": 0.9
    })
    return llm


@pytest.fixture
def temp_storage(tmp_path):
    """Temporary storage path for templates."""
    return tmp_path / "templates"


@pytest.fixture
def template_db(temp_storage):
    """Template database with seed templates."""
    return TemplateDatabase(storage_path=temp_storage)


@pytest.fixture
def refinement_engine(mock_llm, template_db):
    """Template refinement engine."""
    return TemplateRefinementEngine(mock_llm, template_db)


@pytest.fixture
def mock_extraction_results():
    """Mock extraction results for testing."""
    results = []
    for i in range(5):
        result = Mock()
        result.success = True
        result.confidence = 0.8 + (i * 0.02)
        result.content = f"Extraction result {i+1}"
        results.append(result)
    return results


def create_test_template() -> ThoughtTemplate:
    """Create a test template for testing."""
    return ThoughtTemplate(
        template_id="test_template_v1",
        name="Test Template",
        description="Template for testing",
        pattern="# Test Pattern\n\n1. Step one\n2. Step two",
        version=1
    )


# ============================================================================
# Test Class 1: Template Database
# ============================================================================

class TestTemplateDatabase:
    """Test template storage, selection, and composition."""

    def test_initialization(self, template_db):
        """Test database initialization."""
        assert isinstance(template_db, TemplateDatabase)
        assert isinstance(template_db.templates, dict)
        assert template_db.storage_path.exists()

    @pytest.mark.production_readiness
    def test_seed_templates_count(self, template_db):
        """Gate 1: ≥10 seed templates (PRODUCTION GATE)."""
        assert len(template_db.templates) >= 10, \
            f"Expected ≥10 seed templates, got {len(template_db.templates)}"

    def test_seed_template_structure(self, template_db):
        """Verify seed templates have required fields."""
        for template_id, template in template_db.templates.items():
            assert isinstance(template, ThoughtTemplate)
            assert template.template_id == template_id
            assert len(template.name) > 0
            assert len(template.description) > 0
            assert len(template.pattern) > 0
            assert template.version >= 1
            assert isinstance(template.performance_stats, TemplateStats)

    def test_specific_seed_templates_exist(self, template_db):
        """Verify specific required seed templates exist."""
        required_templates = [
            "entity_recognition_v1",
            "relationship_extraction_v1",
            "temporal_reasoning_v1",
            "event_detection_v1",
            "causal_inference_v1",
            "person_entity_v1",
            "organization_entity_v1",
            "concept_entity_v1",
            "works_at_relationship_v1",
            "created_relationship_v1",
        ]

        for template_id in required_templates:
            assert template_id in template_db.templates, \
                f"Required template '{template_id}' not found"

    @pytest.mark.production_readiness
    def test_template_selection_exact_match(self, template_db):
        """Gate 2: Template selection works - exact match (PRODUCTION GATE)."""
        template = template_db.select_template("entity_recognition_v1")
        assert template is not None
        assert template.template_id == "entity_recognition_v1"
        assert "Entity" in template.name

    def test_template_selection_keyword_match(self, template_db):
        """Test template selection by keyword."""
        # Search for temporal-related template
        template = template_db.select_template("temporal reasoning")
        assert template is not None
        assert "temporal" in template.name.lower() or "temporal" in template.description.lower()

    def test_template_selection_performance_based(self, template_db):
        """Test performance-based selection among keyword matches."""
        # Create two templates with same keyword, different performance
        template1 = ThoughtTemplate(
            template_id="test_perf_1",
            name="Test Performance A",
            description="Test template for performance testing",
            pattern="Pattern A",
            version=1,
            performance_stats=TemplateStats(
                usage_count=10,
                success_count=8,
                average_confidence=0.8
            )
        )
        template2 = ThoughtTemplate(
            template_id="test_perf_2",
            name="Test Performance B",
            description="Test template for performance testing",
            pattern="Pattern B",
            version=1,
            performance_stats=TemplateStats(
                usage_count=10,
                success_count=9,
                average_confidence=0.9
            )
        )

        template_db.templates["test_perf_1"] = template1
        template_db.templates["test_perf_2"] = template2

        # Should select template2 (higher success rate and confidence)
        selected = template_db.select_template("performance testing")
        assert selected.template_id == "test_perf_2"

    def test_template_selection_no_match(self, template_db):
        """Test template selection returns None for no match."""
        template = template_db.select_template("nonexistent_xyz_123")
        assert template is None

    @pytest.mark.production_readiness
    def test_template_composition_sequential(self, template_db):
        """Gate 3: Template composition works - sequential (PRODUCTION GATE)."""
        composed = template_db.compose_templates(
            ["entity_recognition_v1", "temporal_reasoning_v1"],
            strategy="sequential"
        )

        assert "Multi-Step Reasoning" in composed
        assert "Step 1" in composed
        assert "Step 2" in composed
        assert "Entity Recognition" in composed
        assert "Temporal" in composed

    def test_template_composition_parallel(self, template_db):
        """Test parallel template composition."""
        composed = template_db.compose_templates(
            ["entity_recognition_v1", "relationship_extraction_v1"],
            strategy="parallel"
        )

        assert "Parallel Analysis" in composed
        assert "Analysis 1" in composed
        assert "Analysis 2" in composed

    def test_template_composition_conditional(self, template_db):
        """Test conditional template composition."""
        composed = template_db.compose_templates(
            ["event_detection_v1", "causal_inference_v1"],
            strategy="conditional"
        )

        assert "Conditional Reasoning" in composed
        assert "If condition 1 met" in composed
        assert "If condition 2 met" in composed

    def test_template_composition_invalid_template(self, template_db):
        """Test composition fails for invalid template ID."""
        with pytest.raises(ValueError, match="not found"):
            template_db.compose_templates(
                ["entity_recognition_v1", "nonexistent_template"],
                strategy="sequential"
            )

    def test_template_composition_invalid_strategy(self, template_db):
        """Test composition fails for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown composition strategy"):
            template_db.compose_templates(
                ["entity_recognition_v1"],
                strategy="invalid_strategy"
            )

    def test_add_template(self, template_db):
        """Test adding new template to database."""
        new_template = create_test_template()
        template_db.add_template(new_template)

        assert "test_template_v1" in template_db.templates
        assert template_db.templates["test_template_v1"] == new_template

    def test_remove_template(self, template_db):
        """Test removing template from database."""
        # Add then remove
        new_template = create_test_template()
        template_db.add_template(new_template)
        assert "test_template_v1" in template_db.templates

        template_db.remove_template("test_template_v1")
        assert "test_template_v1" not in template_db.templates

    def test_persistence_save_and_load(self, temp_storage):
        """Test template persistence to/from disk."""
        # Create database and add template
        db1 = TemplateDatabase(storage_path=temp_storage)
        test_template = create_test_template()
        db1.add_template(test_template)

        # Verify file exists
        template_file = temp_storage / "test_template_v1.json"
        assert template_file.exists()

        # Load in new database instance
        db2 = TemplateDatabase(storage_path=temp_storage)
        assert "test_template_v1" in db2.templates
        assert db2.templates["test_template_v1"].name == "Test Template"


# ============================================================================
# Test Class 2: Textual Gradients
# ============================================================================

class TestTextualGradients:
    """Test textual gradient generation and application."""

    @pytest.mark.production_readiness
    def test_analyze_template_performance(
        self,
        refinement_engine,
        mock_extraction_results
    ):
        """Gate 4: Textual gradients refine templates (PRODUCTION GATE)."""
        template = create_test_template()
        gradient = refinement_engine.analyze_template_performance(
            template,
            mock_extraction_results
        )

        assert isinstance(gradient, TextualGradient)
        assert gradient.decision in ["KEEP", "FIX", "DISCARD"]
        assert len(gradient.feedback) > 0
        assert 0.0 <= gradient.confidence <= 1.0

    def test_build_introspection_prompt(self, refinement_engine, mock_extraction_results):
        """Test introspection prompt construction."""
        template = create_test_template()
        template.performance_stats.usage_count = 50
        template.performance_stats.success_count = 42
        template.performance_stats.average_confidence = 0.84

        prompt = refinement_engine._build_introspection_prompt(
            template,
            mock_extraction_results
        )

        assert "Test Template" in prompt
        assert "v1" in prompt
        assert "Test Pattern" in prompt
        assert "Usage count: 50" in prompt
        assert "84.0%" in prompt  # success rate
        assert "KEEP|FIX|DISCARD" in prompt
        assert "JSON" in prompt

    def test_parse_textual_gradient_keep(self, refinement_engine):
        """Test parsing KEEP decision."""
        response = json.dumps({
            "decision": "KEEP",
            "feedback": "Template is working well",
            "proposed_change": None,
            "confidence": 0.9
        })

        gradient = refinement_engine._parse_textual_gradient("template_id", response)

        assert gradient.decision == "KEEP"
        assert gradient.feedback == "Template is working well"
        assert gradient.proposed_change is None
        assert gradient.confidence == 0.9

    def test_parse_textual_gradient_fix(self, refinement_engine):
        """Test parsing FIX decision with proposed change."""
        response = json.dumps({
            "decision": "FIX",
            "feedback": "Template needs improvement in step 2",
            "proposed_change": "# Improved Pattern\n\n1. Better step one\n2. Improved step two",
            "confidence": 0.85
        })

        gradient = refinement_engine._parse_textual_gradient("template_id", response)

        assert gradient.decision == "FIX"
        assert "improvement" in gradient.feedback
        assert gradient.proposed_change is not None
        assert "Improved" in gradient.proposed_change

    def test_parse_textual_gradient_discard(self, refinement_engine):
        """Test parsing DISCARD decision."""
        response = json.dumps({
            "decision": "DISCARD",
            "feedback": "Template is fundamentally flawed",
            "proposed_change": None,
            "confidence": 0.95
        })

        gradient = refinement_engine._parse_textual_gradient("template_id", response)

        assert gradient.decision == "DISCARD"
        assert "flawed" in gradient.feedback

    def test_parse_textual_gradient_invalid_json(self, refinement_engine):
        """Test parsing fails for invalid JSON."""
        with pytest.raises(ValueError, match="Invalid textual gradient"):
            refinement_engine._parse_textual_gradient("id", "not valid json")

    def test_apply_textual_gradient_keep(self, refinement_engine):
        """Test applying KEEP decision."""
        template = create_test_template()
        gradient = TextualGradient(
            template_id=template.template_id,
            decision="KEEP",
            feedback="Working well",
            confidence=0.9
        )

        result = refinement_engine.apply_textual_gradient(template, gradient)

        assert result is not None
        assert result.version == 1  # Version unchanged
        assert result.template_id == template.template_id

    @pytest.mark.production_readiness
    def test_apply_textual_gradient_fix(self, refinement_engine):
        """Gate 5: Template evolution demonstrable - FIX creates new version (PRODUCTION GATE)."""
        template = create_test_template()
        gradient = TextualGradient(
            template_id=template.template_id,
            decision="FIX",
            feedback="Needs improvement",
            proposed_change="# Improved Pattern\n\n1. Enhanced step",
            confidence=0.85
        )

        result = refinement_engine.apply_textual_gradient(template, gradient)

        assert result is not None
        assert result.version == 2  # Version incremented
        assert result.version > template.version
        assert result.pattern == "# Improved Pattern\n\n1. Enhanced step"
        assert result.parent_version == template.template_id
        assert "v2" in result.template_id

    def test_apply_textual_gradient_discard(self, refinement_engine):
        """Test applying DISCARD decision."""
        template = create_test_template()
        gradient = TextualGradient(
            template_id=template.template_id,
            decision="DISCARD",
            feedback="Fundamentally flawed",
            confidence=0.95
        )

        result = refinement_engine.apply_textual_gradient(template, gradient)

        assert result is None  # Template discarded

    def test_evolve_templates_batch(self, refinement_engine, mock_extraction_results):
        """Test batch template evolution."""
        extraction_results = {
            "entity_recognition_v1": mock_extraction_results,
            "temporal_reasoning_v1": mock_extraction_results
        }

        gradients = refinement_engine.evolve_templates(extraction_results)

        assert len(gradients) == 2
        assert "entity_recognition_v1" in gradients
        assert "temporal_reasoning_v1" in gradients


# ============================================================================
# Test Class 3: Template Evolution
# ============================================================================

class TestTemplateEvolution:
    """Test template evolution over time."""

    def test_template_version_increments(self):
        """Test version tracking across refinements."""
        template_v1 = ThoughtTemplate(
            template_id="test_v1",
            name="Test",
            description="Test",
            pattern="Pattern v1",
            version=1
        )

        # Simulate FIX
        template_v2 = ThoughtTemplate(
            template_id="test_v2",
            name="Test",
            description="Test",
            pattern="Pattern v2",
            version=2,
            parent_version="test_v1"
        )

        assert template_v2.version > template_v1.version
        assert template_v2.parent_version == template_v1.template_id

    def test_performance_stats_update(self):
        """Test usage stats tracking."""
        stats = TemplateStats(
            usage_count=10,
            success_count=8,
            failure_count=2,
            average_confidence=0.85
        )

        assert stats.usage_count == 10
        assert stats.success_rate() == 0.8
        assert stats.average_confidence == 0.85

    def test_performance_stats_success_rate_zero_usage(self):
        """Test success rate calculation with zero usage."""
        stats = TemplateStats()
        assert stats.success_rate() == 0.0

    def test_evolution_history_tracking(self):
        """Test evolution chain tracking via parent_version."""
        v1 = ThoughtTemplate(
            template_id="template_v1",
            name="Template",
            description="Test",
            pattern="v1",
            version=1,
            parent_version=None
        )

        v2 = ThoughtTemplate(
            template_id="template_v2",
            name="Template",
            description="Test",
            pattern="v2",
            version=2,
            parent_version="template_v1"
        )

        v3 = ThoughtTemplate(
            template_id="template_v3",
            name="Template",
            description="Test",
            pattern="v3",
            version=3,
            parent_version="template_v2"
        )

        # Can trace evolution chain
        assert v3.parent_version == v2.template_id
        assert v2.parent_version == v1.template_id
        assert v1.parent_version is None

    def test_template_evolution_multiple_refinements(self, mock_llm):
        """Test template improves over multiple refinement cycles."""
        # Mock LLM to return FIX decisions
        mock_llm.introspect.return_value = json.dumps({
            "decision": "FIX",
            "feedback": "Needs improvement",
            "proposed_change": "# Improved Pattern",
            "confidence": 0.85
        })

        db = TemplateDatabase()
        engine = TemplateRefinementEngine(mock_llm, db)

        template = create_test_template()
        initial_version = template.version

        # Simulate 3 refinement cycles
        for _ in range(3):
            results = [Mock(success=True, confidence=0.8, content="test")]
            gradient = engine.analyze_template_performance(template, results)
            refined = engine.apply_textual_gradient(template, gradient)

            if refined and refined.version > template.version:
                template = refined

        # Should have evolved
        assert template.version > initial_version


# ============================================================================
# Test Class 4: Integration
# ============================================================================

class TestTemplateIntegration:
    """Test integration with other modules."""

    def test_template_composable_with_field(self, template_db):
        """Test composable_with field indicates compatible templates."""
        temporal_template = template_db.templates["temporal_reasoning_v1"]
        assert "event_detection" in temporal_template.composable_with

    def test_template_metadata_fields(self):
        """Test template has all required metadata fields."""
        template = create_test_template()

        assert hasattr(template, "created_at")
        assert hasattr(template, "last_updated")
        assert hasattr(template, "performance_stats")
        assert hasattr(template, "parent_version")
        assert hasattr(template, "composable_with")

    def test_template_serialization(self):
        """Test template can be serialized to JSON."""
        template = create_test_template()
        data = template.model_dump()

        assert data["template_id"] == "test_template_v1"
        assert data["name"] == "Test Template"
        assert data["version"] == 1
        assert "performance_stats" in data

    def test_template_stats_serialization(self):
        """Test TemplateStats can be serialized."""
        stats = TemplateStats(
            usage_count=10,
            success_count=8,
            average_confidence=0.85
        )

        data = stats.model_dump()
        assert data["usage_count"] == 10
        assert data["success_count"] == 8
        assert data["average_confidence"] == 0.85

    def test_textual_gradient_serialization(self):
        """Test TextualGradient can be serialized."""
        gradient = TextualGradient(
            template_id="test_id",
            decision="KEEP",
            feedback="Works well",
            confidence=0.9
        )

        data = gradient.model_dump()
        assert data["decision"] == "KEEP"
        assert data["feedback"] == "Works well"
        assert data["confidence"] == 0.9


# ============================================================================
# Production Readiness Summary
# ============================================================================

@pytest.mark.production_readiness
class TestProductionReadiness:
    """
    Comprehensive production readiness validation.

    All 5 gates from production plan must pass.
    """

    def test_comprehensive_production_readiness_validation(
        self,
        template_db,
        refinement_engine,
        mock_extraction_results
    ):
        """Validate all 5 production readiness gates."""
        gates = {}

        # Gate 1: ≥10 seed templates
        gates["gate_1_seed_templates"] = len(template_db.templates) >= 10

        # Gate 2: Template selection works
        template = template_db.select_template("temporal_reasoning_v1")
        gates["gate_2_selection"] = template is not None

        # Gate 3: Template composition works
        composed = template_db.compose_templates(
            ["entity_recognition_v1", "temporal_reasoning_v1"],
            strategy="sequential"
        )
        gates["gate_3_composition"] = "Step 1" in composed and "Step 2" in composed

        # Gate 4: Textual gradients work
        test_template = create_test_template()
        gradient = refinement_engine.analyze_template_performance(
            test_template,
            mock_extraction_results
        )
        gates["gate_4_gradients"] = gradient.decision in ["KEEP", "FIX", "DISCARD"]

        # Gate 5: Template evolution works
        fix_gradient = TextualGradient(
            template_id=test_template.template_id,
            decision="FIX",
            feedback="Test",
            proposed_change="# New",
            confidence=0.85
        )
        evolved = refinement_engine.apply_textual_gradient(test_template, fix_gradient)
        gates["gate_5_evolution"] = evolved.version > test_template.version

        # All gates must pass
        assert all(gates.values()), f"Failed gates: {[k for k, v in gates.items() if not v]}"

        # Report
        print("\n=== Production Readiness Gates ===")
        for gate, passed in gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} - {gate}")
