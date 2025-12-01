"""
Schema Evolution Engine

Core logic for autonomous schema evolution via multi-phase induction
and reflection mechanisms.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from futurnal.extraction.schema.models import (
    EntityType,
    ExtractionPhase,
    RelationshipType,
    SchemaVersion,
)

if TYPE_CHECKING:
    from futurnal.extraction.schema.models import SchemaDiscovery
    from futurnal.extraction.schema.templates import TemplateDatabase


class ExtractionResult:
    """
    Result of an extraction operation.
    
    Used for quality assessment and reflection triggering.
    """
    
    def __init__(
        self,
        success: bool,
        confidence: float,
        has_novel_pattern: bool = False,
    ):
        self.success = success
        self.confidence = confidence
        self.has_novel_pattern = has_novel_pattern


class Document:
    """
    Placeholder for document type.
    
    In production, this would be the actual document model
    from the ingestion pipeline.
    """
    
    def __init__(self, content: str, doc_id: str):
        self.content = content
        self.doc_id = doc_id


class SchemaEvolutionEngine:
    """
    Autonomous schema evolution via multi-phase induction.
    
    Based on AutoSchemaKG approach:
    1. Phase 1: Entity-Entity relationships (seed schema)
    2. Phase 2: Entity-Event relationships (temporal grounding)
    3. Phase 3: Event-Event relationships (causal candidates)
    """

    def __init__(
        self,
        seed_schema: SchemaVersion,
        template_database: Optional[TemplateDatabase] = None,
        reflection_interval: int = 100,
        min_confidence_threshold: float = 0.7,
    ):
        """
        Initialize schema evolution engine.

        Args:
            seed_schema: Initial schema to evolve from
            template_database: Optional template database for guided extraction
            reflection_interval: Number of documents between reflections
            min_confidence_threshold: Minimum confidence for accepting discoveries
        """
        self.current_schema = seed_schema
        self.schema_history: List[SchemaVersion] = [seed_schema]
        self.template_database = template_database
        self.reflection_interval = reflection_interval
        self.min_confidence_threshold = min_confidence_threshold
        self.documents_processed = 0
        self.pending_discoveries: List[SchemaDiscovery] = []

    def should_trigger_reflection(self) -> bool:
        """
        Check if reflection should be triggered.
        
        Returns:
            bool: True if reflection interval reached
        """
        return self.documents_processed % self.reflection_interval == 0

    def induce_schema_from_documents(
        self,
        documents: List[Document],
        current_phase: ExtractionPhase,
    ) -> SchemaVersion:
        """
        Multi-phase schema induction from documents.
        
        AutoSchemaKG approach:
        - Phase 1: Extract entity-entity patterns
        - Phase 2: Extract entity-event patterns (temporal)
        - Phase 3: Extract event-event patterns (causal)
        
        Args:
            documents: Documents to analyze for schema patterns
            current_phase: Current extraction phase
            
        Returns:
            SchemaVersion: Induced schema for the specified phase
        """
        if current_phase == ExtractionPhase.ENTITY_ENTITY:
            return self._induce_entity_entity_schema(documents)
        elif current_phase == ExtractionPhase.ENTITY_EVENT:
            return self._induce_entity_event_schema(documents)
        elif current_phase == ExtractionPhase.EVENT_EVENT:
            return self._induce_event_event_schema(documents)
        else:
            raise ValueError(f"Unknown extraction phase: {current_phase}")

    def _induce_entity_entity_schema(
        self,
        documents: List[Document],
    ) -> SchemaVersion:
        """
        Phase 1: Discover entity types and entity-entity relationships.
        
        Examples:
        - Entities: Person, Organization, Concept, Location
        - Relationships: works_at, founded, located_in, related_to
        
        Args:
            documents: Documents to analyze
            
        Returns:
            SchemaVersion: Schema with discovered entity-entity patterns
        """
        # In a full implementation, this would use the SchemaDiscoveryEngine
        # For now, we return the current schema as a baseline
        discovered_entities = self.current_schema.entity_types.copy()
        discovered_relationships = self.current_schema.relationship_types.copy()

        return self._create_schema_version(
            phase=ExtractionPhase.ENTITY_ENTITY,
            entities=discovered_entities,
            relationships=discovered_relationships,
        )

    def _induce_entity_event_schema(
        self,
        documents: List[Document],
    ) -> SchemaVersion:
        """
        Phase 2: Discover event types and entity-event relationships.
        
        Examples:
        - Events: Meeting, Decision, Publication, Communication
        - Relationships: attended, made_decision, published, communicated_with
        
        Key: Events have temporal grounding (timestamp, duration)
        
        Args:
            documents: Documents to analyze
            
        Returns:
            SchemaVersion: Schema with discovered entity-event patterns
        """
        # Start with current schema
        discovered_entities = self.current_schema.entity_types.copy()
        discovered_relationships = self.current_schema.relationship_types.copy()
        
        # Add Event entity type if not present
        if "Event" not in discovered_entities:
            discovered_entities["Event"] = EntityType(
                name="Event",
                description="Temporal occurrence or happening",
                examples=["meeting on Monday", "the conference", "project kickoff"],
                confidence=0.9,
                properties={
                    "name": "string",
                    "event_type": "string",
                    "timestamp": "datetime",
                    "duration": "timedelta",
                },
                aliases=["occurrence", "happening", "incident"],
                last_seen=datetime.utcnow(),
            )

        return self._create_schema_version(
            phase=ExtractionPhase.ENTITY_EVENT,
            entities=discovered_entities,
            relationships=discovered_relationships,
        )

    def _induce_event_event_schema(
        self,
        documents: List[Document],
    ) -> SchemaVersion:
        """
        Phase 3: Discover event-event relationships (causal candidates).
        
        Examples:
        - Relationships: caused, enabled, triggered, led_to
        
        Key: These are causal candidates for Phase 3 validation
        
        Args:
            documents: Documents to analyze
            
        Returns:
            SchemaVersion: Schema with discovered event-event patterns
        """
        # Start with current schema
        discovered_entities = self.current_schema.entity_types.copy()
        discovered_relationships = self.current_schema.relationship_types.copy()
        
        # Add causal relationship type if not present
        if "caused" not in discovered_relationships:
            discovered_relationships["caused"] = RelationshipType(
                name="caused",
                description="Event caused another Event (causal candidate)",
                examples=["The meeting caused the decision"],
                confidence=0.8,
                subject_types=["Event"],
                object_types=["Event"],
                temporal=True,
                causal=True,
                properties={
                    "causal_confidence": "float",
                    "temporal_gap": "timedelta",
                    "is_validated": "bool",
                },
                last_seen=datetime.utcnow(),
            )

        return self._create_schema_version(
            phase=ExtractionPhase.EVENT_EVENT,
            entities=discovered_entities,
            relationships=discovered_relationships,
        )

    def _create_schema_version(
        self,
        phase: ExtractionPhase,
        entities: Dict[str, EntityType],
        relationships: Dict[str, RelationshipType],
    ) -> SchemaVersion:
        """
        Create a new schema version.
        
        Args:
            phase: Current extraction phase
            entities: Entity types
            relationships: Relationship types
            
        Returns:
            SchemaVersion: New schema version
        """
        return SchemaVersion(
            version=self.current_schema.version,
            created_at=datetime.utcnow(),
            entity_types=entities,
            relationship_types=relationships,
            current_phase=phase,
            quality_metrics={},
        )

    def reflect_and_refine(
        self,
        extraction_results: List[ExtractionResult],
    ) -> Optional[SchemaVersion]:
        """
        Reflection mechanism: assess quality and refine schema.
        
        Triggers:
        - Every N documents (reflection_interval)
        - Low extraction success rate
        - High rejection rate (low confidence)
        - Consistent pattern of new entity/relationship types
        
        Args:
            extraction_results: Recent extraction results for quality assessment
            
        Returns:
            Optional[SchemaVersion]: Refined schema if refinement triggered, else None
        """
        quality_metrics = self._assess_extraction_quality(extraction_results)

        if quality_metrics.get("should_refine", False):
            refined_schema = self._refine_schema(
                self.current_schema, quality_metrics
            )
            self._version_schema(refined_schema)
            return refined_schema

        return None

    def _assess_extraction_quality(
        self, results: List[ExtractionResult]
    ) -> Dict[str, float]:
        """
        Assess extraction quality metrics.
        
        Metrics:
        - Coverage: % of documents producing extractions
        - Consistency: % of extractions with high confidence
        - Success rate: % of extractions not quarantined
        - Novel patterns: count of unrecognized patterns
        
        Args:
            results: Extraction results to assess
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        if not results:
            return {
                "coverage": 0.0,
                "consistency": 0.0,
                "success_rate": 0.0,
                "novel_pattern_rate": 0.0,
                "should_refine": False,
            }

        total = len(results)
        successful = sum(1 for r in results if r.success)
        high_confidence = sum(1 for r in results if r.confidence > 0.8)
        novel_patterns = sum(1 for r in results if r.has_novel_pattern)

        success_rate = successful / total
        novel_rate = novel_patterns / total

        return {
            "coverage": success_rate,
            "consistency": high_confidence / total,
            "success_rate": success_rate,
            "novel_pattern_rate": novel_rate,
            "should_refine": (
                success_rate < 0.7 or  # Low success rate
                novel_rate > 0.1  # Many novel patterns
            ),
        }

    def _refine_schema(
        self, schema: SchemaVersion, metrics: Dict[str, float]
    ) -> SchemaVersion:
        """
        Refine schema based on quality metrics.
        
        Refinements:
        - Add frequently seen novel entity/relationship types
        - Remove rarely seen types (low discovery_count)
        - Adjust confidence thresholds
        - Update type hierarchies
        
        Args:
            schema: Current schema
            metrics: Quality metrics
            
        Returns:
            SchemaVersion: Refined schema
        """
        # For now, create a new version with same content
        # In full implementation, would apply actual refinements
        return SchemaVersion(
            version=schema.version + 1,
            created_at=datetime.utcnow(),
            entity_types=schema.entity_types.copy(),
            relationship_types=schema.relationship_types.copy(),
            current_phase=schema.current_phase,
            quality_metrics=metrics,
            changes_from_previous=self._generate_changelog(schema),
        )

    def _version_schema(self, schema: SchemaVersion) -> None:
        """
        Version and store a schema in history.
        
        Args:
            schema: Schema to version
        """
        self.current_schema = schema
        self.schema_history.append(schema)

    def _generate_changelog(self, previous_schema: SchemaVersion) -> str:
        """
        Generate changelog describing schema changes.
        
        Args:
            previous_schema: Previous schema version
            
        Returns:
            str: Changelog description
        """
        return f"Schema refined from v{previous_schema.version}"
