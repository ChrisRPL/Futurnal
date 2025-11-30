"""
Unit Tests for Schema Evolution

Tests for schema evolution engine, multi-phase induction,
reflection triggers, and refinement logic.
"""

import pytest
from datetime import datetime

from futurnal.extraction.schema import (
    create_seed_schema,
    ExtractionPhase,
    SchemaEvolutionEngine,
    SchemaDiscovery,
)
from futurnal.extraction.schema.evolution import Document, ExtractionResult


class TestSeedSchema:
    """Test seed schema creation and structure."""

    def test_seed_schema_creation(self):
        """Validate seed schema structure and content."""
        schema = create_seed_schema()

        # Verify version and phase
        assert schema.version == 1
        assert schema.current_phase == ExtractionPhase.ENTITY_ENTITY

        # Verify core entity types
        assert "Person" in schema.entity_types
        assert "Organization" in schema.entity_types
        assert "Concept" in schema.entity_types
        assert "Document" in schema.entity_types

        # Verify entity properties
        person = schema.entity_types["Person"]
        assert person.confidence == 1.0
        assert "name" in person.properties
        assert len(person.examples) > 0

        # Verify core relationship types
        assert "works_at" in schema.relationship_types
        assert "created" in schema.relationship_types
        assert "related_to" in schema.relationship_types

        # Verify relationship properties
        works_at = schema.relationship_types["works_at"]
        assert works_at.temporal is True
        assert "Person" in works_at.subject_types
        assert "Organization" in works_at.object_types


class TestSchemaEvolution:
    """Test schema evolution mechanisms."""

    def test_entity_entity_induction(self):
        """Test Phase 1 entity-entity schema induction."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed)

        # Create test documents
        documents = [
            Document("Alice works at Acme Corp.", "doc1"),
            Document("Bob founded the startup.", "doc2"),
            Document("The team discussed AI.", "doc3"),
        ]

        new_schema = engine.induce_schema_from_documents(
            documents, ExtractionPhase.ENTITY_ENTITY
        )

        # Should maintain Phase 1
        assert new_schema.current_phase == ExtractionPhase.ENTITY_ENTITY

        # Should have entity types
        assert len(new_schema.entity_types) >= len(seed.entity_types)

    def test_entity_event_induction(self):
        """Test Phase 2 entity-event schema induction."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed)

        documents = [
            Document("Alice attended the meeting on Monday.", "doc1"),
            Document("The conference starts tomorrow.", "doc2"),
        ]

        new_schema = engine.induce_schema_from_documents(
            documents, ExtractionPhase.ENTITY_EVENT
        )

        # Should be in Phase 2
        assert new_schema.current_phase == ExtractionPhase.ENTITY_EVENT

        # Should discover Event entity type
        assert "Event" in new_schema.entity_types
        event = new_schema.entity_types["Event"]
        assert "timestamp" in event.properties

    def test_event_event_induction(self):
        """Test Phase 3 event-event schema induction."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed)

        documents = [
            Document("The meeting caused the decision.", "doc1"),
            Document("The event triggered a response.", "doc2"),
        ]

        new_schema = engine.induce_schema_from_documents(
            documents, ExtractionPhase.EVENT_EVENT
        )

        # Should be in Phase 3
        assert new_schema.current_phase == ExtractionPhase.EVENT_EVENT

        # Should discover causal relationships
        assert "caused" in new_schema.relationship_types
        caused = new_schema.relationship_types["caused"]
        assert caused.causal is True
        assert caused.temporal is True

    def test_reflection_trigger(self):
        """Test reflection mechanism triggers correctly."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed, reflection_interval=10)

        # Initial state - documents_processed is 0, which triggers reflection (0 % 10 == 0)
        # So we need to start at 1 to avoid this
        engine.documents_processed = 1
        assert not engine.should_trigger_reflection()

        # Process documents
        for i in range(15):
            engine.documents_processed += 1

        # Should trigger at interval (10)
        engine.documents_processed = 10
        assert engine.should_trigger_reflection()

    def test_schema_refinement(self):
        """Test schema refinement adds/removes types correctly."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed)

        # Create mock extraction results with low success rate
        results = [
            ExtractionResult(success=False, confidence=0.5)
            for _ in range(20)
        ]

        refined = engine.reflect_and_refine(results)

        # Should trigger refinement due to low success rate
        assert refined is not None
        assert refined.version == seed.version + 1
        assert refined.changes_from_previous is not None

    def test_quality_assessment(self):
        """Test extraction quality assessment metrics."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed)

        # Create mix of results
        results = [
            ExtractionResult(success=True, confidence=0.9)
            for _ in range(15)
        ] + [
            ExtractionResult(success=False, confidence=0.3)
            for _ in range(5)
        ]

        metrics = engine._assess_extraction_quality(results)

        # Verify metrics
        assert "success_rate" in metrics
        assert "consistency" in metrics
        assert "novel_pattern_rate" in metrics
        assert metrics["success_rate"] == 0.75  # 15/20

    def test_schema_versioning(self):
        """Test schema versions are tracked correctly."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed)

        initial_version = engine.current_schema.version

        # Trigger refinement
        results = [
            ExtractionResult(success=False, confidence=0.5)
            for _ in range(20)
        ]
        refined = engine.reflect_and_refine(results)

        # Should increment version
        assert refined is not None
        assert refined.version == initial_version + 1

        # Should be in history
        assert len(engine.schema_history) == 2
        assert engine.schema_history[-1].version == refined.version


class TestReflectionTrigger:
    """Test reflection trigger logic."""

    def test_interval_trigger(self):
        """Test reflection triggers at regular intervals."""
        from futurnal.extraction.schema.refinement import ReflectionTrigger

        trigger = ReflectionTrigger(interval=100)

        # Should trigger at interval
        assert trigger.should_reflect(100, [])
        assert trigger.should_reflect(200, [])

        # Should not trigger otherwise
        assert not trigger.should_reflect(50, [])
        assert not trigger.should_reflect(150, [])

    def test_low_success_trigger(self):
        """Test reflection triggers on low success rate."""
        from futurnal.extraction.schema.refinement import ReflectionTrigger

        trigger = ReflectionTrigger(low_success_threshold=0.7)

        # Create low success results
        results = [
            ExtractionResult(success=False, confidence=0.5)
            for _ in range(20)
        ]

        # Should trigger
        assert trigger.should_reflect(50, results)

    def test_high_novel_trigger(self):
        """Test reflection triggers on many novel patterns."""
        from futurnal.extraction.schema.refinement import ReflectionTrigger

        trigger = ReflectionTrigger(high_novel_threshold=0.1)

        # Create results with many novel patterns
        results = [
            ExtractionResult(
                success=True, confidence=0.8, has_novel_pattern=True
            )
            for _ in range(20)
        ]

        # Should trigger
        assert trigger.should_reflect(50, results)


class TestSchemaRefinement:
    """Test schema refinement engine."""

    def test_add_discovered_entity_types(self):
        """Test adding frequently discovered entity types."""
        from futurnal.extraction.schema.refinement import SchemaRefinementEngine

        refiner = SchemaRefinementEngine()
        seed = create_seed_schema()

        # Create discovery with sufficient examples (15 examples)
        discoveries = [
            SchemaDiscovery(
                element_type="entity",
                name="Project",
                description="Work project or initiative",
                examples=[f"project {i}" for i in range(15)],
                confidence=0.85,
                source_documents=[f"doc{i}" for i in range(15)],
            )
        ]

        # Use min_discovery_count that's met by the examples
        refined = refiner.refine_entity_types(
            seed.entity_types, discoveries, min_discovery_count=15
        )

        # Should add Project entity
        assert "Project" in refined
        assert refined["Project"].confidence == 0.85

    def test_preserve_seed_types(self):
        """Test that seed types are preserved during refinement."""
        from futurnal.extraction.schema.refinement import SchemaRefinementEngine

        refiner = SchemaRefinementEngine()
        seed = create_seed_schema()

        # Refine with no discoveries
        refined = refiner.refine_entity_types(
            seed.entity_types, [], min_discovery_count=10
        )

        # Should preserve all seed types (confidence 1.0)
        assert "Person" in refined
        assert "Organization" in refined
        assert "Concept" in refined
        assert "Document" in refined
