"""
Seed Schema Generation

Creates the initial schema with core entity and relationship types
for Phase 1 (Entity-Entity) extraction.
"""

from datetime import datetime

from futurnal.extraction.schema.models import (
    EntityType,
    ExtractionPhase,
    RelationshipType,
    SchemaVersion,
)


def create_seed_schema() -> SchemaVersion:
    """
    Create initial seed schema for Phase 1 Entity-Entity extraction.
    
    Returns:
        SchemaVersion: Initial schema with core entity and relationship types
        
    Example:
        >>> schema = create_seed_schema()
        >>> assert "Person" in schema.entity_types
        >>> assert schema.version == 1
    """
    # Core entity types
    entities = {
        "Person": EntityType(
            name="Person",
            description="Individual human being",
            examples=["John Smith", "Dr. Jane Doe", "the author"],
            confidence=1.0,
            properties={"name": "string", "aliases": "list[string]"},
            aliases=["individual", "human", "author"],
            last_seen=datetime.utcnow(),
        ),
        "Organization": EntityType(
            name="Organization",
            description="Company, institution, or group",
            examples=["Acme Corp", "MIT", "the research team"],
            confidence=1.0,
            properties={"name": "string", "type": "string"},
            aliases=["company", "institution", "group"],
            last_seen=datetime.utcnow(),
        ),
        "Concept": EntityType(
            name="Concept",
            description="Abstract idea or topic",
            examples=["machine learning", "productivity", "causality"],
            confidence=1.0,
            properties={"name": "string", "category": "string"},
            aliases=["idea", "topic", "subject"],
            last_seen=datetime.utcnow(),
        ),
        "Document": EntityType(
            name="Document",
            description="Written work or reference",
            examples=["research paper", "the report", "documentation"],
            confidence=1.0,
            properties={"title": "string", "type": "string"},
            aliases=["paper", "article", "file"],
            last_seen=datetime.utcnow(),
        ),
    }

    # Core relationship types (Phase 1: Entity-Entity)
    relationships = {
        "works_at": RelationshipType(
            name="works_at",
            description="Person employed by Organization",
            examples=["John works at Acme Corp"],
            confidence=1.0,
            subject_types=["Person"],
            object_types=["Organization"],
            temporal=True,  # Can have valid_from/valid_to
            properties={"role": "string", "start_date": "datetime"},
            last_seen=datetime.utcnow(),
        ),
        "created": RelationshipType(
            name="created",
            description="Entity created another entity",
            examples=["Alice created the framework"],
            confidence=1.0,
            subject_types=["Person", "Organization"],
            object_types=["Concept", "Document"],
            temporal=True,
            properties={"creation_date": "datetime"},
            last_seen=datetime.utcnow(),
        ),
        "related_to": RelationshipType(
            name="related_to",
            description="Generic relationship between concepts",
            examples=["AI related to machine learning"],
            confidence=1.0,
            subject_types=["Concept"],
            object_types=["Concept"],
            temporal=False,
            properties={"relationship_type": "string"},
            last_seen=datetime.utcnow(),
        ),
    }

    return SchemaVersion(
        version=1,
        created_at=datetime.utcnow(),
        entity_types=entities,
        relationship_types=relationships,
        current_phase=ExtractionPhase.ENTITY_ENTITY,
        quality_metrics={},
    )
