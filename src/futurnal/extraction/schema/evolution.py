"""
Schema Evolution Engine

Core logic for autonomous schema evolution via multi-phase induction
and reflection mechanisms.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from futurnal.extraction.schema.models import (
    EntityType,
    ExtractionPhase,
    RelationshipType,
    SchemaVersion,
)

logger = logging.getLogger(__name__)

# Quality gate threshold per AutoSchemaKG research
SEMANTIC_ALIGNMENT_THRESHOLD = 0.90

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

        Per AutoSchemaKG Phase 1:
        - Discover entity types (Person, Organization, Concept, Location, etc.)
        - Discover entity-entity relationships (works_at, founded, related_to, etc.)
        - Build initial graph structure without temporal grounding

        Args:
            documents: Documents to analyze

        Returns:
            SchemaVersion: Schema with discovered entity-entity patterns
        """
        from futurnal.extraction.schema.discovery import SchemaDiscoveryEngine

        # Initialize discovery engine with optional LLM from template database
        llm = self.template_database if self.template_database else None
        discovery_engine = SchemaDiscoveryEngine(llm=llm)

        # Start with current schema (preserves seed types)
        discovered_entities = self.current_schema.entity_types.copy()
        discovered_relationships = self.current_schema.relationship_types.copy()

        # Discover entity patterns from documents
        entity_discoveries = discovery_engine.discover_entity_patterns(documents)

        # Merge discovered entities with current schema
        for discovery in entity_discoveries:
            if discovery.confidence >= self.min_confidence_threshold:
                entity_name = discovery.name

                if entity_name not in discovered_entities:
                    # Add new entity type
                    discovered_entities[entity_name] = EntityType(
                        name=entity_name,
                        description=discovery.description,
                        examples=discovery.examples,
                        confidence=discovery.confidence,
                        properties={},
                        aliases=[],
                        discovery_count=len(discovery.examples),
                        last_seen=datetime.utcnow(),
                    )
                else:
                    # Update existing type
                    existing = discovered_entities[entity_name]
                    existing.discovery_count += len(discovery.examples)
                    existing.last_seen = datetime.utcnow()
                    # Merge examples (deduplicated)
                    existing.examples = list(
                        set(existing.examples + discovery.examples)
                    )[:10]
                    # Update confidence if discovery has higher confidence
                    if discovery.confidence > existing.confidence and existing.confidence < 1.0:
                        existing.confidence = discovery.confidence

        # Discover relationship patterns between known entities
        relationship_discoveries = discovery_engine.discover_relationship_patterns(
            documents, discovered_entities
        )

        # Merge discovered relationships with current schema
        for discovery in relationship_discoveries:
            if discovery.confidence >= self.min_confidence_threshold:
                rel_name = discovery.name

                if rel_name not in discovered_relationships:
                    # Parse subject/object types from description
                    # Description format: "Relationship 'verb' between [types] and [types]"
                    subject_types = ["Entity"]  # Default
                    object_types = ["Entity"]

                    # Try to extract types from source_documents metadata
                    # (stored as type hints in discovery process)
                    if discovery.source_documents:
                        # First few entries might contain type hints
                        pass  # Would parse from structured discovery

                    discovered_relationships[rel_name] = RelationshipType(
                        name=rel_name,
                        description=discovery.description,
                        examples=discovery.examples,
                        confidence=discovery.confidence,
                        subject_types=subject_types,
                        object_types=object_types,
                        temporal=False,  # Phase 1 relationships are not temporal
                        causal=False,
                        properties={},
                        discovery_count=len(discovery.examples),
                        last_seen=datetime.utcnow(),
                    )
                else:
                    # Update existing relationship
                    existing = discovered_relationships[rel_name]
                    existing.discovery_count += len(discovery.examples)
                    existing.last_seen = datetime.utcnow()

        self.documents_processed += len(documents)

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

        Per AutoSchemaKG Phase 2 + Step 04 Integration:
        - Events MUST have temporal grounding (timestamp, duration)
        - Discover event patterns correlated with temporal markers
        - Add entity-event relationships (attended, organized, participated)
        - Uses TemporalExtractionResult from Step 04

        Args:
            documents: Documents to analyze

        Returns:
            SchemaVersion: Schema with discovered entity-event patterns
        """
        from futurnal.extraction.schema.discovery import SchemaDiscoveryEngine

        # Initialize discovery engine
        llm = self.template_database if self.template_database else None
        discovery_engine = SchemaDiscoveryEngine(llm=llm)

        # Start with current schema
        discovered_entities = self.current_schema.entity_types.copy()
        discovered_relationships = self.current_schema.relationship_types.copy()

        # Ensure Event base type exists with temporal properties (CRITICAL for Step 04 integration)
        if "Event" not in discovered_entities:
            discovered_entities["Event"] = EntityType(
                name="Event",
                description="Temporal occurrence with timestamp grounding (Step 04 integration)",
                examples=["meeting on Monday", "the conference", "project kickoff"],
                confidence=0.95,
                properties={
                    "name": "string",
                    "event_type": "string",
                    "timestamp": "datetime",  # REQUIRED - from Step 04
                    "duration": "timedelta",
                    "participants": "list[string]",
                    "location": "string",
                },
                aliases=["occurrence", "happening", "incident", "meeting", "activity"],
                last_seen=datetime.utcnow(),
            )

        # Discover event patterns from documents
        event_patterns = self._discover_event_patterns(documents, discovery_engine)

        # Add discovered event subtypes
        for pattern in event_patterns:
            event_type_name = pattern.get("event_type", "GenericEvent")

            if event_type_name not in discovered_entities:
                discovered_entities[event_type_name] = EntityType(
                    name=event_type_name,
                    description=f"Event subtype: {event_type_name}",
                    examples=pattern.get("examples", []),
                    confidence=pattern.get("confidence", 0.8),
                    properties={
                        "timestamp": "datetime",  # Inherited from Event
                        "duration": "timedelta",
                    },
                    parent_type="Event",  # Hierarchical relationship
                    aliases=[],
                    discovery_count=len(pattern.get("examples", [])),
                    last_seen=datetime.utcnow(),
                )

        # Define entity-event relationship types
        entity_event_relationships = [
            (
                "attended",
                "Person participated in Event",
                ["Person"],
                ["Event"],
                {"event_role": "string", "participation_type": "string"},
            ),
            (
                "organized",
                "Person or Organization organized Event",
                ["Person", "Organization"],
                ["Event"],
                {"organizer_role": "string"},
            ),
            (
                "mentioned_in",
                "Entity was mentioned during Event",
                ["Concept", "Document", "Person", "Organization"],
                ["Event"],
                {"mention_context": "string"},
            ),
            (
                "occurred_at",
                "Event occurred at Location",
                ["Event"],
                ["Location"],
                {"venue_type": "string"},
            ),
            (
                "resulted_in",
                "Event resulted in Document or Concept",
                ["Event"],
                ["Document", "Concept"],
                {"outcome_type": "string"},
            ),
        ]

        # Add entity-event relationships
        for rel_name, description, subject_types, object_types, properties in entity_event_relationships:
            if rel_name not in discovered_relationships:
                discovered_relationships[rel_name] = RelationshipType(
                    name=rel_name,
                    description=description,
                    examples=[],
                    confidence=0.85,
                    subject_types=subject_types,
                    object_types=object_types,
                    temporal=True,  # Entity-Event relationships are ALWAYS temporal
                    causal=False,  # Causal relationships are Phase 3
                    properties=properties,
                    last_seen=datetime.utcnow(),
                )

        self.documents_processed += len(documents)

        return self._create_schema_version(
            phase=ExtractionPhase.ENTITY_EVENT,
            entities=discovered_entities,
            relationships=discovered_relationships,
        )

    def _discover_event_patterns(
        self,
        documents: List[Document],
        discovery_engine: "SchemaDiscoveryEngine",
    ) -> List[Dict]:
        """
        Discover event types from documents using temporal markers.

        Integration with Step 04:
        - Uses spaCy for event-indicating verbs/nouns
        - Correlates with temporal markers (DATE, TIME entities)

        Args:
            documents: Documents to analyze
            discovery_engine: Discovery engine instance

        Returns:
            List of event pattern dictionaries
        """
        from collections import defaultdict

        event_patterns = []
        event_indicators: Dict[str, List[Dict]] = defaultdict(list)

        # Event-indicating words/patterns
        event_keywords = {
            "meeting", "conference", "call", "discussion", "review",
            "launch", "release", "announcement", "presentation",
            "decision", "agreement", "deadline", "milestone",
            "start", "end", "begin", "complete", "finish",
            "submit", "publish", "deliver", "deploy",
        }

        for doc in documents:
            try:
                spacy_doc = discovery_engine.nlp(doc.content)

                for token in spacy_doc:
                    # Look for event-indicating words
                    if token.lemma_.lower() in event_keywords:
                        # Check for temporal context in sentence
                        has_temporal = any(
                            ent.label_ in ("DATE", "TIME", "EVENT")
                            for ent in token.sent.ents
                        )

                        if has_temporal or token.ent_type_ == "EVENT":
                            event_type = token.lemma_.title()
                            event_indicators[event_type].append(
                                {
                                    "text": token.text,
                                    "context": token.sent.text[:200],
                                    "doc_id": doc.doc_id,
                                    "has_temporal": has_temporal,
                                }
                            )

                # Also check for EVENT named entities
                for ent in spacy_doc.ents:
                    if ent.label_ == "EVENT":
                        event_type = "NamedEvent"
                        event_indicators[event_type].append(
                            {
                                "text": ent.text,
                                "context": ent.sent.text[:200] if hasattr(ent, "sent") else "",
                                "doc_id": doc.doc_id,
                                "has_temporal": True,
                            }
                        )

            except Exception as e:
                continue

        # Convert to event patterns (filter by frequency)
        for event_type, instances in event_indicators.items():
            if len(instances) >= 3:  # Minimum frequency for event type
                event_patterns.append(
                    {
                        "event_type": event_type,
                        "examples": [i["text"] for i in instances[:5]],
                        "confidence": min(len(instances) / 10, 0.95),
                        "has_temporal_grounding": any(i["has_temporal"] for i in instances),
                    }
                )

        return event_patterns

    def _induce_event_event_schema(
        self,
        documents: List[Document],
    ) -> SchemaVersion:
        """
        Phase 3: Discover event-event relationships (causal candidates).

        Per AutoSchemaKG Phase 3:
        - Event-Event relationships (caused, enabled, triggered, prevented, led_to)
        - All marked as `causal=True` for Phase 3 Bradford-Hill validation
        - CRITICAL: Temporal precedence required (cause.timestamp < effect.timestamp)
        - Discovers causal language patterns from documents

        Args:
            documents: Documents to analyze

        Returns:
            SchemaVersion: Schema with discovered event-event patterns
        """
        from futurnal.extraction.schema.discovery import SchemaDiscoveryEngine

        # Initialize discovery engine
        llm = self.template_database if self.template_database else None
        discovery_engine = SchemaDiscoveryEngine(llm=llm)

        # Start with current schema
        discovered_entities = self.current_schema.entity_types.copy()
        discovered_relationships = self.current_schema.relationship_types.copy()

        # Define causal relationship types (candidates for Phase 3 Bradford-Hill validation)
        causal_relationships = [
            RelationshipType(
                name="caused",
                description="Event directly caused another Event (causal candidate for Bradford-Hill)",
                examples=["The bug caused the outage", "The meeting caused the decision"],
                confidence=0.80,
                subject_types=["Event"],
                object_types=["Event"],
                temporal=True,
                causal=True,
                properties={
                    "causal_confidence": "float",
                    "temporal_gap": "timedelta",
                    "is_validated": "bool",  # For Phase 3 Bradford-Hill
                    "evidence_strength": "float",
                    "mechanism": "string",
                },
                last_seen=datetime.utcnow(),
            ),
            RelationshipType(
                name="enabled",
                description="Event enabled/allowed another Event to occur",
                examples=["The approval enabled the project", "The funding enabled the research"],
                confidence=0.75,
                subject_types=["Event"],
                object_types=["Event"],
                temporal=True,
                causal=True,
                properties={
                    "causal_confidence": "float",
                    "enabling_factor": "string",
                    "necessity": "string",  # "necessary", "sufficient", "contributory"
                },
                last_seen=datetime.utcnow(),
            ),
            RelationshipType(
                name="triggered",
                description="Event directly triggered another Event (immediate causation)",
                examples=["The alert triggered the response", "The commit triggered the build"],
                confidence=0.80,
                subject_types=["Event"],
                object_types=["Event"],
                temporal=True,
                causal=True,
                properties={
                    "trigger_type": "string",
                    "immediacy": "string",  # "immediate", "delayed"
                    "causal_confidence": "float",
                },
                last_seen=datetime.utcnow(),
            ),
            RelationshipType(
                name="prevented",
                description="Event prevented another Event from occurring (counterfactual)",
                examples=["The fix prevented the crash", "The warning prevented the error"],
                confidence=0.70,
                subject_types=["Event"],
                object_types=["Event"],
                temporal=True,
                causal=True,
                properties={
                    "prevention_type": "string",
                    "counterfactual_confidence": "float",
                    "mechanism": "string",
                },
                last_seen=datetime.utcnow(),
            ),
            RelationshipType(
                name="led_to",
                description="Event led to another Event (weaker causal claim, indirect)",
                examples=["The discussion led to the agreement", "The research led to discovery"],
                confidence=0.70,
                subject_types=["Event"],
                object_types=["Event"],
                temporal=True,
                causal=True,
                properties={
                    "pathway": "string",
                    "intermediate_events": "list[string]",
                    "causal_confidence": "float",
                },
                last_seen=datetime.utcnow(),
            ),
            RelationshipType(
                name="contributed_to",
                description="Event contributed to another Event (partial causation)",
                examples=["The delay contributed to the failure", "His work contributed to success"],
                confidence=0.65,
                subject_types=["Event"],
                object_types=["Event"],
                temporal=True,
                causal=True,
                properties={
                    "contribution_weight": "float",
                    "other_factors": "list[string]",
                    "causal_confidence": "float",
                },
                last_seen=datetime.utcnow(),
            ),
        ]

        # Add causal relationships to schema
        for rel in causal_relationships:
            if rel.name not in discovered_relationships:
                discovered_relationships[rel.name] = rel

        # Discover additional causal patterns from documents
        causal_patterns = self._discover_causal_patterns(documents, discovery_engine)

        # Add discovered causal relationship types
        for pattern in causal_patterns:
            rel_name = pattern["name"]
            if rel_name not in discovered_relationships:
                discovered_relationships[rel_name] = RelationshipType(
                    name=rel_name,
                    description=pattern.get("description", f"Causal relationship: {rel_name}"),
                    examples=pattern.get("examples", []),
                    confidence=pattern.get("confidence", 0.7),
                    subject_types=["Event"],
                    object_types=["Event"],
                    temporal=True,
                    causal=True,
                    properties={
                        "is_validated": "bool",
                        "causal_confidence": "float",
                    },
                    discovery_count=len(pattern.get("examples", [])),
                    last_seen=datetime.utcnow(),
                )

        self.documents_processed += len(documents)

        return self._create_schema_version(
            phase=ExtractionPhase.EVENT_EVENT,
            entities=discovered_entities,
            relationships=discovered_relationships,
        )

    def _discover_causal_patterns(
        self,
        documents: List[Document],
        discovery_engine: "SchemaDiscoveryEngine",
    ) -> List[Dict]:
        """
        Discover causal relationship patterns from documents.

        Looks for causal language indicators:
        - "caused", "led to", "resulted in", "because"
        - "enabled", "allowed", "made possible"
        - "prevented", "blocked", "stopped"
        - "triggered", "initiated", "started"

        Args:
            documents: Documents to analyze
            discovery_engine: Discovery engine instance

        Returns:
            List of causal pattern dictionaries
        """
        from collections import defaultdict

        # Causal language indicators mapped to relationship types
        causal_indicators = {
            "caused": ["caused", "cause", "causing", "because of"],
            "led_to": ["led to", "lead to", "leads to", "resulted in", "result in"],
            "enabled": ["enabled", "enable", "allowed", "allow", "made possible"],
            "prevented": ["prevented", "prevent", "blocked", "block", "stopped", "stop"],
            "triggered": ["triggered", "trigger", "initiated", "initiate", "started", "start"],
            "contributed_to": ["contributed to", "contribute to", "helped cause"],
        }

        causal_patterns = []
        pattern_counts: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "examples": [], "contexts": []}
        )

        for doc in documents:
            content_lower = doc.content.lower()

            for rel_name, indicators in causal_indicators.items():
                for indicator in indicators:
                    if indicator in content_lower:
                        pattern_counts[rel_name]["count"] += 1

                        # Extract example sentence
                        idx = content_lower.find(indicator)
                        # Find sentence boundaries
                        start = max(0, content_lower.rfind(".", 0, idx) + 1)
                        end = content_lower.find(".", idx)
                        if end == -1:
                            end = min(len(doc.content), idx + 100)

                        example = doc.content[start:end].strip()
                        if len(pattern_counts[rel_name]["examples"]) < 5:
                            pattern_counts[rel_name]["examples"].append(example[:200])

        # Convert to causal patterns (filter by frequency)
        for rel_name, data in pattern_counts.items():
            if data["count"] >= 2:  # Minimum frequency for causal pattern
                causal_patterns.append(
                    {
                        "name": rel_name,
                        "description": f"Causal relationship: {rel_name} (discovered)",
                        "examples": data["examples"],
                        "confidence": min(data["count"] / 10, 0.85),
                    }
                )

        return causal_patterns

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

    def _generate_changelog(
        self,
        previous_schema: SchemaVersion,
        new_entities: Optional[Dict[str, EntityType]] = None,
        new_relationships: Optional[Dict[str, RelationshipType]] = None,
    ) -> str:
        """
        Generate detailed changelog describing schema changes.

        Tracks:
        - Added entity types
        - Removed entity types
        - Added relationship types
        - Removed relationship types
        - Phase transitions

        Args:
            previous_schema: Previous schema version
            new_entities: New entity types (optional, uses current_schema if not provided)
            new_relationships: New relationship types (optional)

        Returns:
            str: Detailed changelog description
        """
        changes = []

        # Compare entity types
        if new_entities is not None:
            prev_entity_names = set(previous_schema.entity_types.keys())
            new_entity_names = set(new_entities.keys())

            added_entities = new_entity_names - prev_entity_names
            removed_entities = prev_entity_names - new_entity_names

            if added_entities:
                changes.append(f"Added entity types: {', '.join(sorted(added_entities))}")
            if removed_entities:
                changes.append(f"Removed entity types: {', '.join(sorted(removed_entities))}")
        else:
            # Compare with current schema
            prev_entity_names = set(previous_schema.entity_types.keys())
            curr_entity_names = set(self.current_schema.entity_types.keys())

            added_entities = curr_entity_names - prev_entity_names
            removed_entities = prev_entity_names - curr_entity_names

            if added_entities:
                changes.append(f"Added entity types: {', '.join(sorted(added_entities))}")
            if removed_entities:
                changes.append(f"Removed entity types: {', '.join(sorted(removed_entities))}")

        # Compare relationship types
        if new_relationships is not None:
            prev_rel_names = set(previous_schema.relationship_types.keys())
            new_rel_names = set(new_relationships.keys())

            added_rels = new_rel_names - prev_rel_names
            removed_rels = prev_rel_names - new_rel_names

            if added_rels:
                changes.append(f"Added relationship types: {', '.join(sorted(added_rels))}")
            if removed_rels:
                changes.append(f"Removed relationship types: {', '.join(sorted(removed_rels))}")
        else:
            # Compare with current schema
            prev_rel_names = set(previous_schema.relationship_types.keys())
            curr_rel_names = set(self.current_schema.relationship_types.keys())

            added_rels = curr_rel_names - prev_rel_names
            removed_rels = prev_rel_names - curr_rel_names

            if added_rels:
                changes.append(f"Added relationship types: {', '.join(sorted(added_rels))}")
            if removed_rels:
                changes.append(f"Removed relationship types: {', '.join(sorted(removed_rels))}")

        # Track phase transition
        if self.current_schema.current_phase != previous_schema.current_phase:
            changes.append(
                f"Phase transition: {previous_schema.current_phase.value} → "
                f"{self.current_schema.current_phase.value}"
            )

        # Track document count
        changes.append(f"Documents processed: {self.documents_processed}")

        if not changes:
            return f"Schema refined from v{previous_schema.version} (no structural changes)"

        return f"Schema v{previous_schema.version + 1}: " + "; ".join(changes)

    def compute_semantic_alignment(
        self,
        discovered_schema: SchemaVersion,
        reference_schema: SchemaVersion,
    ) -> Dict[str, float]:
        """
        Compute semantic alignment between discovered and reference schemas.

        Per AutoSchemaKG quality gate: >90% semantic alignment required.

        Metrics computed:
        1. Entity type coverage: % of reference entities matched
        2. Entity type precision: % of discovered entities that are valid
        3. Relationship coverage: % of reference relationships matched
        4. Relationship precision: % of discovered relationships that are valid
        5. Property alignment: % of properties correctly typed
        6. Overall semantic alignment: weighted average

        Args:
            discovered_schema: Schema discovered from documents
            reference_schema: Ground truth/manual schema for comparison

        Returns:
            Dict with alignment metrics including 'semantic_alignment' score
        """
        metrics: Dict[str, float] = {}

        # Entity type analysis
        ref_entities = set(reference_schema.entity_types.keys())
        disc_entities = set(discovered_schema.entity_types.keys())

        if ref_entities:
            # Coverage: how many reference entities were found
            matched_entities = ref_entities & disc_entities
            metrics["entity_coverage"] = len(matched_entities) / len(ref_entities)

            # Also check for semantic matches (aliases, similar names)
            semantic_matches = self._compute_semantic_entity_matches(
                discovered_schema.entity_types,
                reference_schema.entity_types,
            )
            metrics["entity_semantic_matches"] = semantic_matches
        else:
            metrics["entity_coverage"] = 1.0
            metrics["entity_semantic_matches"] = 1.0

        if disc_entities:
            # Precision: how many discovered entities are valid
            valid_disc = disc_entities & ref_entities
            # Also count semantically similar as valid
            metrics["entity_precision"] = max(
                len(valid_disc) / len(disc_entities),
                metrics.get("entity_semantic_matches", 0.0),
            )
        else:
            metrics["entity_precision"] = 1.0

        # Relationship type analysis
        ref_rels = set(reference_schema.relationship_types.keys())
        disc_rels = set(discovered_schema.relationship_types.keys())

        if ref_rels:
            matched_rels = ref_rels & disc_rels
            metrics["relationship_coverage"] = len(matched_rels) / len(ref_rels)

            # Semantic matches for relationships
            semantic_rel_matches = self._compute_semantic_relationship_matches(
                discovered_schema.relationship_types,
                reference_schema.relationship_types,
            )
            metrics["relationship_semantic_matches"] = semantic_rel_matches
        else:
            metrics["relationship_coverage"] = 1.0
            metrics["relationship_semantic_matches"] = 1.0

        if disc_rels:
            valid_disc_rels = disc_rels & ref_rels
            metrics["relationship_precision"] = max(
                len(valid_disc_rels) / len(disc_rels),
                metrics.get("relationship_semantic_matches", 0.0),
            )
        else:
            metrics["relationship_precision"] = 1.0

        # Property alignment (for matched entities)
        property_scores = []
        for entity_name in ref_entities & disc_entities:
            ref_props = set(reference_schema.entity_types[entity_name].properties.keys())
            disc_props = set(discovered_schema.entity_types[entity_name].properties.keys())
            if ref_props:
                prop_coverage = len(ref_props & disc_props) / len(ref_props)
                property_scores.append(prop_coverage)
        metrics["property_alignment"] = (
            sum(property_scores) / len(property_scores) if property_scores else 1.0
        )

        # Temporal/causal correctness for relationships
        temporal_correct = 0
        causal_correct = 0
        temporal_total = 0
        causal_total = 0

        for rel_name in ref_rels & disc_rels:
            ref_rel = reference_schema.relationship_types[rel_name]
            disc_rel = discovered_schema.relationship_types[rel_name]

            if ref_rel.temporal:
                temporal_total += 1
                if disc_rel.temporal:
                    temporal_correct += 1

            if ref_rel.causal:
                causal_total += 1
                if disc_rel.causal:
                    causal_correct += 1

        metrics["temporal_correctness"] = (
            temporal_correct / temporal_total if temporal_total > 0 else 1.0
        )
        metrics["causal_correctness"] = (
            causal_correct / causal_total if causal_total > 0 else 1.0
        )

        # Overall semantic alignment (weighted average per AutoSchemaKG)
        # Weights: entity coverage (30%), relationship coverage (30%),
        #          semantic matches (20%), property alignment (10%),
        #          temporal/causal (10%)
        semantic_alignment = (
            0.30 * metrics["entity_coverage"]
            + 0.30 * metrics["relationship_coverage"]
            + 0.10 * metrics["entity_semantic_matches"]
            + 0.10 * metrics["relationship_semantic_matches"]
            + 0.10 * metrics["property_alignment"]
            + 0.05 * metrics["temporal_correctness"]
            + 0.05 * metrics["causal_correctness"]
        )

        metrics["semantic_alignment"] = round(semantic_alignment, 4)
        metrics["passes_quality_gate"] = semantic_alignment >= SEMANTIC_ALIGNMENT_THRESHOLD

        logger.info(
            f"Semantic alignment: {semantic_alignment:.2%} "
            f"(threshold: {SEMANTIC_ALIGNMENT_THRESHOLD:.0%}, "
            f"passes: {metrics['passes_quality_gate']})"
        )

        return metrics

    def _compute_semantic_entity_matches(
        self,
        discovered: Dict[str, EntityType],
        reference: Dict[str, EntityType],
    ) -> float:
        """
        Compute semantic similarity between entity type sets.

        Uses aliases, descriptions, and examples for matching beyond
        exact name matches.

        Args:
            discovered: Discovered entity types
            reference: Reference entity types

        Returns:
            float: Semantic match score [0, 1]
        """
        if not reference:
            return 1.0

        matches = 0
        total = len(reference)

        for ref_name, ref_entity in reference.items():
            # Exact match
            if ref_name in discovered:
                matches += 1
                continue

            # Check aliases
            for disc_name, disc_entity in discovered.items():
                # Check if reference name is in discovered aliases
                if ref_name.lower() in [a.lower() for a in disc_entity.aliases]:
                    matches += 0.8  # Partial credit for alias match
                    break
                # Check if discovered name is in reference aliases
                if disc_name.lower() in [a.lower() for a in ref_entity.aliases]:
                    matches += 0.8
                    break
                # Check example overlap
                ref_examples = set(e.lower() for e in ref_entity.examples)
                disc_examples = set(e.lower() for e in disc_entity.examples)
                if ref_examples and disc_examples:
                    overlap = len(ref_examples & disc_examples) / len(ref_examples)
                    if overlap > 0.5:
                        matches += overlap * 0.6  # Partial credit for example overlap
                        break

        return matches / total

    def _compute_semantic_relationship_matches(
        self,
        discovered: Dict[str, RelationshipType],
        reference: Dict[str, RelationshipType],
    ) -> float:
        """
        Compute semantic similarity between relationship type sets.

        Uses subject/object type constraints and examples for matching.

        Args:
            discovered: Discovered relationship types
            reference: Reference relationship types

        Returns:
            float: Semantic match score [0, 1]
        """
        if not reference:
            return 1.0

        matches = 0
        total = len(reference)

        for ref_name, ref_rel in reference.items():
            # Exact match
            if ref_name in discovered:
                matches += 1
                continue

            # Check for semantically similar relationships
            for disc_name, disc_rel in discovered.items():
                # Check subject/object type overlap
                ref_subjects = set(ref_rel.subject_types)
                disc_subjects = set(disc_rel.subject_types)
                ref_objects = set(ref_rel.object_types)
                disc_objects = set(disc_rel.object_types)

                subject_overlap = len(ref_subjects & disc_subjects) / max(
                    len(ref_subjects), 1
                )
                object_overlap = len(ref_objects & disc_objects) / max(
                    len(ref_objects), 1
                )

                # If types match well, check for example similarity
                if subject_overlap > 0.5 and object_overlap > 0.5:
                    ref_examples = set(e.lower() for e in ref_rel.examples)
                    disc_examples = set(e.lower() for e in disc_rel.examples)
                    if ref_examples and disc_examples:
                        example_overlap = len(ref_examples & disc_examples) / max(
                            len(ref_examples), 1
                        )
                        if example_overlap > 0.3:
                            matches += 0.7  # Partial credit for semantic match
                            break

        return matches / total

    def merge_schema_versions(
        self,
        base_schema: SchemaVersion,
        new_discoveries: List["SchemaDiscovery"],
        merge_strategy: str = "conservative",
    ) -> SchemaVersion:
        """
        Merge new discoveries into base schema with conflict resolution.

        Merge strategies:
        - "conservative": Only add types with high confidence, never remove
        - "progressive": Add new types, update existing, prune low-discovery types
        - "strict": Only exact matches allowed, reject ambiguous discoveries

        Per AutoSchemaKG: Schema evolution should be additive with pruning
        based on discovery counts and confidence degradation.

        Args:
            base_schema: Current schema version
            new_discoveries: New schema discoveries to merge
            merge_strategy: One of "conservative", "progressive", "strict"

        Returns:
            SchemaVersion: Merged schema
        """
        merged_entities = base_schema.entity_types.copy()
        merged_relationships = base_schema.relationship_types.copy()

        # Define thresholds based on strategy
        thresholds = {
            "conservative": {"min_confidence": 0.85, "min_examples": 5},
            "progressive": {"min_confidence": 0.70, "min_examples": 3},
            "strict": {"min_confidence": 0.95, "min_examples": 10},
        }
        params = thresholds.get(merge_strategy, thresholds["conservative"])

        entity_additions = []
        relationship_additions = []

        for discovery in new_discoveries:
            if discovery.confidence < params["min_confidence"]:
                continue
            if len(discovery.examples) < params["min_examples"]:
                continue

            if discovery.element_type == "entity":
                if discovery.name not in merged_entities:
                    # Add new entity type
                    merged_entities[discovery.name] = EntityType(
                        name=discovery.name,
                        description=discovery.description,
                        examples=discovery.examples[:10],
                        confidence=discovery.confidence,
                        properties={},
                        aliases=[],
                        discovery_count=len(discovery.examples),
                        last_seen=datetime.now(),
                    )
                    entity_additions.append(discovery.name)
                else:
                    # Update existing - merge examples, update confidence
                    existing = merged_entities[discovery.name]
                    existing.discovery_count += len(discovery.examples)
                    existing.last_seen = datetime.now()
                    # Merge examples (deduplicated)
                    existing.examples = list(
                        set(existing.examples + discovery.examples)
                    )[:10]
                    # Update confidence using exponential moving average
                    existing.confidence = (
                        0.7 * existing.confidence + 0.3 * discovery.confidence
                    )

            elif discovery.element_type == "relationship":
                if discovery.name not in merged_relationships:
                    merged_relationships[discovery.name] = RelationshipType(
                        name=discovery.name,
                        description=discovery.description,
                        examples=discovery.examples[:10],
                        confidence=discovery.confidence,
                        subject_types=["Entity"],
                        object_types=["Entity"],
                        temporal=False,
                        causal=False,
                        properties={},
                        discovery_count=len(discovery.examples),
                        last_seen=datetime.now(),
                    )
                    relationship_additions.append(discovery.name)
                else:
                    existing = merged_relationships[discovery.name]
                    existing.discovery_count += len(discovery.examples)
                    existing.last_seen = datetime.now()
                    existing.examples = list(
                        set(existing.examples + discovery.examples)
                    )[:10]
                    existing.confidence = (
                        0.7 * existing.confidence + 0.3 * discovery.confidence
                    )

        # Progressive strategy: prune low-discovery types (except seeds)
        if merge_strategy == "progressive":
            prune_threshold = 3  # Minimum discovery count to keep
            merged_entities = {
                name: entity
                for name, entity in merged_entities.items()
                if entity.confidence == 1.0  # Keep seed types
                or entity.discovery_count >= prune_threshold
            }
            merged_relationships = {
                name: rel
                for name, rel in merged_relationships.items()
                if rel.confidence == 1.0
                or rel.discovery_count >= prune_threshold
            }

        # Create new version
        new_version = SchemaVersion(
            version=base_schema.version + 1,
            created_at=datetime.now(),
            entity_types=merged_entities,
            relationship_types=merged_relationships,
            current_phase=base_schema.current_phase,
            quality_metrics={},
            changes_from_previous=self._generate_changelog(
                base_schema,
                new_entities=merged_entities,
                new_relationships=merged_relationships,
            ),
        )

        logger.info(
            f"Merged schema: added {len(entity_additions)} entities, "
            f"{len(relationship_additions)} relationships "
            f"(strategy: {merge_strategy})"
        )

        return new_version
