"""
Integration Tests for Schema Evolution

End-to-end tests for multi-phase progression and schema versioning.
"""

import pytest

from futurnal.extraction.schema import (
    create_seed_schema,
    ExtractionPhase,
    SchemaEvolutionEngine,
)
from futurnal.extraction.schema.evolution import Document, ExtractionResult


class TestMultiPhaseProgression:
    """Test progression through all extraction phases."""

    def test_phase_progression(self):
        """Test progression through all 3 phases."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed)

        # Create diverse document set
        documents = [
            Document(f"Sample document {i}", f"doc{i}")
            for i in range(100)
        ]

        # Phase 1: Entity-Entity
        phase1_schema = engine.induce_schema_from_documents(
            documents[:40], ExtractionPhase.ENTITY_ENTITY
        )
        assert phase1_schema.current_phase == ExtractionPhase.ENTITY_ENTITY
        assert len(phase1_schema.entity_types) >= 4  # Seed types

        # Phase 2: Entity-Event
        phase2_schema = engine.induce_schema_from_documents(
            documents[40:70], ExtractionPhase.ENTITY_EVENT
        )
        assert phase2_schema.current_phase == ExtractionPhase.ENTITY_EVENT
        assert "Event" in phase2_schema.entity_types

        # Phase 3: Event-Event
        phase3_schema = engine.induce_schema_from_documents(
            documents[70:], ExtractionPhase.EVENT_EVENT
        )
        assert phase3_schema.current_phase == ExtractionPhase.EVENT_EVENT
        # Should have causal relationships
        assert any(
            r.causal for r in phase3_schema.relationship_types.values()
        )

    def test_schema_versioning_across_phases(self):
        """Test schema versions tracked correctly across phases."""
        seed = create_seed_schema()
        engine = SchemaEvolutionEngine(seed)

        initial_version = engine.current_schema.version

        # Trigger multiple refinements
        for _ in range(3):
            results = [
                ExtractionResult(success=False, confidence=0.5)
                for _ in range(20)
            ]
            refined = engine.reflect_and_refine(results)

        # Should have multiple versions in history
        assert len(engine.schema_history) > 1

        # Latest version should be incremented
        assert engine.current_schema.version > initial_version


class TestSchemaEvolutionIntegration:
    """Integration tests for complete workflow."""

    def test_end_to_end_evolution(self):
        """Test complete evolution workflow."""
        # Create seed schema
        schema = create_seed_schema()
        assert schema.version == 1

        # Initialize engine
        engine = SchemaEvolutionEngine(schema, reflection_interval=10)

        # Process documents and trigger reflection
        for i in range(15):
            engine.documents_processed += 1

        # Should trigger at interval
        engine.documents_processed = 10
        assert engine.should_trigger_reflection()

        # Simulate extraction results
        results = [
            ExtractionResult(success=True, confidence=0.9)
            for _ in range(20)
        ]

        # Assess quality
        metrics = engine._assess_extraction_quality(results)
        assert metrics["success_rate"] == 1.0

    def test_discovery_integration(self):
        """Test integration with discovery engine."""
        from futurnal.extraction.schema.discovery import SchemaDiscoveryEngine

        seed = create_seed_schema()
        evolution_engine = SchemaEvolutionEngine(seed)
        discovery_engine = SchemaDiscoveryEngine()

        # Create test documents
        documents = [
            Document("Alice works on project X", "doc1"),
            Document("Bob manages the initiative", "doc2"),
        ]

        # Discover patterns
        discoveries = discovery_engine.discover_entity_patterns(documents)

        # Currently returns empty list (placeholder)
        assert isinstance(discoveries, list)
