"""
Autonomous Schema Evolution Module

Implements multi-phase schema induction and reflection mechanisms
for dynamic entity/relationship discovery.

Based on AutoSchemaKG approach with three extraction phases:
1. Entity-Entity relationships
2. Entity-Event relationships  
3. Event-Event relationships (causal candidates)
"""

from futurnal.extraction.schema.models import (
    ExtractionPhase,
    SchemaElement,
    EntityType,
    RelationshipType,
    SchemaVersion,
    SchemaDiscovery,
)
from futurnal.extraction.schema.seed import create_seed_schema
from futurnal.extraction.schema.evolution import SchemaEvolutionEngine
from futurnal.extraction.schema.discovery import SchemaDiscoveryEngine
from futurnal.extraction.schema.refinement import (
    ReflectionTrigger,
    SchemaRefinementEngine,
)

__all__ = [
    "ExtractionPhase",
    "SchemaElement",
    "EntityType",
    "RelationshipType",
    "SchemaVersion",
    "SchemaDiscovery",
    "create_seed_schema",
    "SchemaEvolutionEngine",
    "SchemaDiscoveryEngine",
    "ReflectionTrigger",
    "SchemaRefinementEngine",
]
