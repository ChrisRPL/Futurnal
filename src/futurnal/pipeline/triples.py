"""Semantic triple extraction pipeline for Ghost's experiential memory construction.

Extracts structured relationships from document metadata to build the Ghost's
understanding of the user's personal universe. Triples form the foundational
knowledge representation enabling AI personalization.

Ghost→Animal Evolution Trajectory:
- **Phase 1 (CURRENT - Archivist)**: Mechanical metadata extraction
  - Extract basic document properties, tags, links from structured metadata
  - Build static graph relationships with high precision
  - Establish vocabulary and schema for experiential memory

- **Phase 2 (FUTURE - Analyst)**: AI-learned pattern extraction
  - Use LLM to extract semantic relationships from unstructured content
  - Add confidence scoring based on extraction method
  - Enable incremental learning as Ghost's understanding develops

- **Phase 3 (FUTURE - Guide)**: Causal relationship inference
  - Infer causal relationships from temporal event sequences
  - Generate hypothesis triples for user-guided exploration
  - Link relationships to Aspirational Self for alignment tracking

Current implementation provides the extraction infrastructure that Phase 2/3
will enhance with increasingly sophisticated AI capabilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class SemanticTriple:
    """Represents a semantic triple (Subject, Predicate, Object) with metadata."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_element_id: Optional[str] = None
    source_path: Optional[str] = None
    extraction_method: str = "metadata"
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source_element_id": self.source_element_id,
            "source_path": self.source_path,
            "extraction_method": self.extraction_method,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    name: str
    entity_type: str
    canonical_name: Optional[str] = None
    aliases: List[str] = None
    confidence: float = 1.0
    source_element_id: Optional[str] = None
    source_path: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.canonical_name is None:
            self.canonical_name = self.name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "source_element_id": self.source_element_id,
            "source_path": self.source_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MetadataTripleExtractor:
    """Extracts semantic triples from structured document metadata.

    Phase-Specific Behavior:
    - **Phase 1 (CURRENT)**: Mechanical extraction from known metadata fields
      - Deterministic mapping of frontmatter, tags, links to triples
      - High precision, 100% confidence on extracted relationships
      - Foundation for Ghost's structured understanding

    - **Phase 2 (FUTURE)**: AI-assisted extraction from content
      - LLM-based relationship extraction from unstructured text
      - Confidence scoring based on extraction method and certainty
      - Incremental learning updates as Ghost's understanding evolves

    - **Phase 3 (FUTURE)**: Causal inference and hypothesis generation
      - Temporal relationship analysis for causal pattern detection
      - Generate hypothesis triples linking to Aspirational Self
      - Support user-guided exploration of causal relationships

    Current implementation establishes the extraction vocabulary and
    infrastructure that future phases will enhance with AI capabilities.
    """

    def __init__(self):
        self.extracted_count = 0
    
    def extract_triples(self, element: Dict[str, Any]) -> List[SemanticTriple]:
        """Extract triples from element metadata."""
        triples = []
        
        metadata = element.get("metadata", {})
        source_path = metadata.get("path")
        element_id = element.get("sha256", "unknown")
        
        # Extract document-level triples
        triples.extend(self._extract_document_triples(metadata, element_id, source_path))
        
        # Extract Obsidian-specific triples
        triples.extend(self._extract_obsidian_triples(metadata, element_id, source_path))
        
        # Extract frontmatter triples
        if "frontmatter" in metadata:
            triples.extend(self._extract_frontmatter_triples(
                metadata["frontmatter"], element_id, source_path
            ))
        
        self.extracted_count += len(triples)
        return triples
    
    def _extract_document_triples(
        self, 
        metadata: Dict[str, Any], 
        element_id: str, 
        source_path: Optional[str]
    ) -> List[SemanticTriple]:
        """Extract basic document triples."""
        triples = []
        doc_uri = self._create_document_uri(source_path or element_id)
        
        # Document type
        triples.append(SemanticTriple(
            subject=doc_uri,
            predicate="rdf:type",
            object="futurnal:Document",
            source_element_id=element_id,
            source_path=source_path,
            extraction_method="metadata"
        ))
        
        # Document source
        if source_path:
            triples.append(SemanticTriple(
                subject=doc_uri,
                predicate="futurnal:hasSourcePath",
                object=source_path,
                source_element_id=element_id,
                source_path=source_path,
                extraction_method="metadata"
            ))
        
        # File metadata
        if "size_bytes" in metadata:
            triples.append(SemanticTriple(
                subject=doc_uri,
                predicate="futurnal:hasFileSize",
                object=str(metadata["size_bytes"]),
                source_element_id=element_id,
                source_path=source_path,
                extraction_method="metadata"
            ))
        
        if "modified_at" in metadata:
            triples.append(SemanticTriple(
                subject=doc_uri,
                predicate="futurnal:modifiedAt",
                object=metadata["modified_at"],
                source_element_id=element_id,
                source_path=source_path,
                extraction_method="metadata"
            ))
        
        return triples
    
    def _extract_obsidian_triples(
        self, 
        metadata: Dict[str, Any], 
        element_id: str, 
        source_path: Optional[str]
    ) -> List[SemanticTriple]:
        """Extract Obsidian-specific triples."""
        triples = []
        doc_uri = self._create_document_uri(source_path or element_id)
        
        # Process Obsidian tags
        if "obsidian_tags" in metadata:
            for tag_info in metadata["obsidian_tags"]:
                tag_name = tag_info.get("name", "")
                if tag_name:
                    tag_uri = self._create_tag_uri(tag_name)
                    
                    # Document has tag
                    triples.append(SemanticTriple(
                        subject=doc_uri,
                        predicate="futurnal:hasTag",
                        object=tag_uri,
                        source_element_id=element_id,
                        source_path=source_path,
                        extraction_method="obsidian_metadata"
                    ))
                    
                    # Tag properties
                    triples.append(SemanticTriple(
                        subject=tag_uri,
                        predicate="rdf:type",
                        object="futurnal:Tag",
                        source_element_id=element_id,
                        source_path=source_path,
                        extraction_method="obsidian_metadata"
                    ))
                    
                    triples.append(SemanticTriple(
                        subject=tag_uri,
                        predicate="futurnal:tagName",
                        object=tag_name,
                        source_element_id=element_id,
                        source_path=source_path,
                        extraction_method="obsidian_metadata"
                    ))
                    
                    if tag_info.get("is_nested", False):
                        triples.append(SemanticTriple(
                            subject=tag_uri,
                            predicate="futurnal:isNestedTag",
                            object="true",
                            source_element_id=element_id,
                            source_path=source_path,
                            extraction_method="obsidian_metadata"
                        ))
        
        # Process Obsidian links
        if "obsidian_links" in metadata:
            for link_info in metadata["obsidian_links"]:
                target = link_info.get("target", "")
                if target:
                    target_uri = self._create_document_uri(target)
                    
                    # Document links to target
                    predicate = "futurnal:embeds" if link_info.get("is_embed", False) else "futurnal:linksTo"
                    triples.append(SemanticTriple(
                        subject=doc_uri,
                        predicate=predicate,
                        object=target_uri,
                        source_element_id=element_id,
                        source_path=source_path,
                        extraction_method="obsidian_metadata"
                    ))
                    
                    # Link with section
                    if link_info.get("section"):
                        section_uri = f"{target_uri}#{link_info['section']}"
                        triples.append(SemanticTriple(
                            subject=doc_uri,
                            predicate="futurnal:linksToSection",
                            object=section_uri,
                            source_element_id=element_id,
                            source_path=source_path,
                            extraction_method="obsidian_metadata"
                        ))
                    
                    # Broken links
                    if link_info.get("is_broken", False):
                        triples.append(SemanticTriple(
                            subject=doc_uri,
                            predicate="futurnal:hasBrokenLink",
                            object=target_uri,
                            source_element_id=element_id,
                            source_path=source_path,
                            extraction_method="obsidian_metadata"
                        ))
        
        return triples
    
    def _extract_frontmatter_triples(
        self, 
        frontmatter: Dict[str, Any], 
        element_id: str, 
        source_path: Optional[str]
    ) -> List[SemanticTriple]:
        """Extract triples from YAML frontmatter."""
        triples = []
        doc_uri = self._create_document_uri(source_path or element_id)
        
        # Title
        if "title" in frontmatter:
            triples.append(SemanticTriple(
                subject=doc_uri,
                predicate="dc:title",
                object=str(frontmatter["title"]),
                source_element_id=element_id,
                source_path=source_path,
                extraction_method="frontmatter"
            ))
        
        # Author
        if "author" in frontmatter:
            author_uri = self._create_person_uri(str(frontmatter["author"]))
            triples.append(SemanticTriple(
                subject=doc_uri,
                predicate="dc:creator",
                object=author_uri,
                source_element_id=element_id,
                source_path=source_path,
                extraction_method="frontmatter"
            ))
            
            triples.append(SemanticTriple(
                subject=author_uri,
                predicate="rdf:type",
                object="futurnal:Person",
                source_element_id=element_id,
                source_path=source_path,
                extraction_method="frontmatter"
            ))
        
        # Aliases
        if "aliases" in frontmatter:
            aliases = frontmatter["aliases"]
            if isinstance(aliases, list):
                for alias in aliases:
                    triples.append(SemanticTriple(
                        subject=doc_uri,
                        predicate="futurnal:hasAlias",
                        object=str(alias),
                        source_element_id=element_id,
                        source_path=source_path,
                        extraction_method="frontmatter"
                    ))
        
        # Category/Type
        if "category" in frontmatter:
            category_uri = self._create_category_uri(str(frontmatter["category"]))
            triples.append(SemanticTriple(
                subject=doc_uri,
                predicate="futurnal:hasCategory",
                object=category_uri,
                source_element_id=element_id,
                source_path=source_path,
                extraction_method="frontmatter"
            ))
        
        # Status
        if "status" in frontmatter:
            triples.append(SemanticTriple(
                subject=doc_uri,
                predicate="futurnal:hasStatus",
                object=str(frontmatter["status"]),
                source_element_id=element_id,
                source_path=source_path,
                extraction_method="frontmatter"
            ))
        
        # Dates
        for date_field in ["created", "modified", "published"]:
            if date_field in frontmatter:
                predicate_map = {
                    "created": "dc:created",
                    "modified": "dc:modified", 
                    "published": "dc:issued"
                }
                triples.append(SemanticTriple(
                    subject=doc_uri,
                    predicate=predicate_map.get(date_field, f"futurnal:{date_field}"),
                    object=str(frontmatter[date_field]),
                    source_element_id=element_id,
                    source_path=source_path,
                    extraction_method="frontmatter"
                ))
        
        return triples
    
    def _create_document_uri(self, identifier: str) -> str:
        """Create a URI for a document."""
        # Normalize path separators and create a clean identifier
        clean_id = identifier.replace("\\", "/").replace(" ", "_")
        return f"futurnal:doc/{clean_id}"
    
    def _create_tag_uri(self, tag_name: str) -> str:
        """Create a URI for a tag."""
        clean_tag = tag_name.replace("/", "_").replace(" ", "_")
        return f"futurnal:tag/{clean_tag}"
    
    def _create_person_uri(self, name: str) -> str:
        """Create a URI for a person."""
        clean_name = name.replace(" ", "_")
        return f"futurnal:person/{clean_name}"
    
    def _create_category_uri(self, category: str) -> str:
        """Create a URI for a category."""
        clean_category = category.replace(" ", "_")
        return f"futurnal:category/{clean_category}"


class AdvancedTripleExtractor:
    """Phase 2 (Analyst) extraction with AI-powered relationship detection.
    
    Integrates all advanced extraction modules:
    - Temporal markers (explicit and relative)
    - Event extraction with temporal grounding
    - Causal relationship detection
    - Schema evolution
    - Experiential learning (Training-Free GRPO)
    - Thought templates (TOTAL framework)
    
    This represents the Ghost→Animal evolution where experiential
    knowledge improves extraction quality without parameter updates.
    """
    
    def __init__(self, enable_experiential_learning: bool = True):
        """Initialize advanced extractor with all Phase 2 modules.
        
        Args:
            enable_experiential_learning: Enable GRPO for quality improvement
        """
        from futurnal.extraction.temporal.markers import TemporalMarkerExtractor
        from futurnal.extraction.causal.event_extractor import EventExtractor
        from futurnal.extraction.causal.relationship_detector import CausalRelationshipDetector
        from futurnal.extraction.schema.evolution import SchemaEvolutionEngine
        from futurnal.extraction.schema.seed import create_seed_schema
        from futurnal.extraction.schema.experiential import TrainingFreeGRPO
        from futurnal.extraction.schema.templates import TemplateDatabase
        from futurnal.extraction.local_llm_client import get_test_llm_client
        
        logger.info("Initializing AdvancedTripleExtractor (Phase 2 - Analyst)")
        
        # Initialize LLM client (local, privacy-first)
        try:
            self.llm = get_test_llm_client(fast=True)  # Use lightweight model for now
            logger.info("Local LLM client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            logger.warning("Advanced extraction will be disabled")
            self.llm = None
        
        # Initialize extraction modules
        self.temporal_extractor = TemporalMarkerExtractor()
        logger.info("Temporal extraction initialized")
        
        # Event and causal extraction require LLM
        if self.llm:
            self.event_extractor = EventExtractor(
                llm=self.llm,
                temporal_extractor=self.temporal_extractor
            )
            logger.info("Event extraction initialized")
            
            self.causal_detector = CausalRelationshipDetector(llm=self.llm)
            logger.info("Causal relationship detection initialized")
        else:
            self.event_extractor = None
            self.causal_detector = None
        
        # Initialize schema evolution
        seed_schema = create_seed_schema()
        self.schema_engine = SchemaEvolutionEngine(seed_schema=seed_schema)
        logger.info("Schema evolution initialized")
        
        # Initialize experiential learning (optional)
        self.enable_experiential_learning = enable_experiential_learning and self.llm is not None
        if self.enable_experiential_learning:
            self.grpo = TrainingFreeGRPO(llm=self.llm)
            logger.info("Experiential learning (GRPO) initialized")
        else:
            self.grpo = None
        
        # Initialize thought templates
        self.templates = TemplateDatabase()
        logger.info(f"Thought templates initialized ({len(self.templates.templates)} templates)")
        
        self.extracted_count = 0
    
    def extract_triples(self, element: Dict[str, Any]) -> List[SemanticTriple]:
        """Extract semantic triples using advanced extraction.
        
        Pipeline:
        1. Extract temporal markers from text
        2. Extract events with temporal grounding
        3. Detect causal relationships between events
        4. Generate semantic triples
        
        Args:
            element: Document element with metadata and text
            
        Returns:
            List of SemanticTriple objects with temporal/causal information
        """
        triples = []
        
        metadata = element.get("metadata", {})
        source_path = metadata.get("path")
        element_id = element.get("sha256", "unknown")
        text = element.get("text", "")
        
        # Skip if no text content
        if not text or len(text.strip()) < 10:
            logger.debug(f"Skipping element {element_id}: insufficient text")
            return triples
        
        try:
            # Phase 1: Extract temporal markers
            temporal_markers = self.temporal_extractor.extract_temporal_markers(
                text=text,
                doc_metadata=metadata
            )
            logger.debug(f"Extracted {len(temporal_markers)} temporal markers")
            
            # Convert temporal markers to triples
            for marker in temporal_markers:
                doc_uri = self._create_document_uri(source_path or element_id)
                marker_uri = f"{doc_uri}/temporal/{marker.text.replace(' ', '_')}"
                
                # Temporal marker triple
                triples.append(SemanticTriple(
                    subject=doc_uri,
                    predicate="futurnal:hasTemporalMarker",
                    object=marker_uri,
                    confidence=marker.confidence,
                    source_element_id=element_id,
                    source_path=source_path,
                    extraction_method="temporal_extraction"
                ))
                
                # Marker timestamp triple
                if marker.timestamp:
                    triples.append(SemanticTriple(
                        subject=marker_uri,
                        predicate="futurnal:timestamp",
                        object=marker.timestamp.isoformat(),
                        confidence=marker.confidence,
                        source_element_id=element_id,
                        source_path=source_path,
                        extraction_method="temporal_extraction"
                    ))
            
            # Phase 2: Extract events (requires LLM)
            events = []
            if self.event_extractor and self.llm:
                try:
                    # Create document object for event extractor
                    from futurnal.extraction.causal.event_extractor import Document
                    
                    class SimpleDocument:
                        def __init__(self, content: str, doc_id: str):
                            self.content = content
                            self.doc_id = doc_id
                    
                    doc = SimpleDocument(content=text, doc_id=element_id)
                    events = self.event_extractor.extract_events(doc)
                    logger.debug(f"Extracted {len(events)} events")
                    
                    # Convert events to triples
                    for event in events:
                        doc_uri = self._create_document_uri(source_path or element_id)
                        event_uri = f"{doc_uri}/event/{event.name.replace(' ', '_')}"
                        
                        # Event triple
                        triples.append(SemanticTriple(
                            subject=doc_uri,
                            predicate="futurnal:hasEvent",
                            object=event_uri,
                            confidence=event.extraction_confidence,
                            source_element_id=element_id,
                            source_path=source_path,
                            extraction_method="event_extraction"
                        ))
                        
                        # Event type triple
                        triples.append(SemanticTriple(
                            subject=event_uri,
                            predicate="rdf:type",
                            object=f"futurnal:Event/{event.event_type}",
                            confidence=event.extraction_confidence,
                            source_element_id=element_id,
                            source_path=source_path,
                            extraction_method="event_extraction"
                        ))
                        
                        # Event timestamp triple
                        if event.timestamp:
                            triples.append(SemanticTriple(
                                subject=event_uri,
                                predicate="futurnal:eventTimestamp",
                                object=event.timestamp.isoformat(),
                                confidence=event.extraction_confidence,
                                source_element_id=element_id,
                                source_path=source_path,
                                extraction_method="event_extraction"
                            ))
                
                except Exception as e:
                    logger.error(f"Event extraction failed: {e}")
            
            # Phase 3: Detect causal relationships (requires events and LLM)
            if self.causal_detector and self.llm and len(events) >= 2:
                try:
                    from futurnal.extraction.causal.relationship_detector import Document
                    
                    class SimpleDocument:
                        def __init__(self, content: str, doc_id: str):
                            self.content = content
                            self.doc_id = doc_id
                    
                    doc = SimpleDocument(content=text, doc_id=element_id)
                    causal_candidates = self.causal_detector.detect_causal_candidates(
                        events=events,
                        document=doc
                    )
                    logger.debug(f"Detected {len(causal_candidates)} causal candidates")
                    
                    # Convert causal relationships to triples
                    for candidate in causal_candidates:
                        doc_uri = self._create_document_uri(source_path or element_id)
                        cause_uri = f"{doc_uri}/event/{candidate.cause_event_id.replace(' ', '_')}"
                        effect_uri = f"{doc_uri}/event/{candidate.effect_event_id.replace(' ', '_')}"
                        
                        # Causal relationship triple
                        triples.append(SemanticTriple(
                            subject=cause_uri,
                            predicate=f"futurnal:{candidate.relationship_type.value}",
                            object=effect_uri,
                            confidence=candidate.causal_confidence,
                            source_element_id=element_id,
                            source_path=source_path,
                            extraction_method="causal_detection"
                        ))
                
                except Exception as e:
                    logger.error(f"Causal detection failed: {e}")
        
        except Exception as e:
            logger.error(f"Advanced extraction failed for {element_id}: {e}")
        
        self.extracted_count += len(triples)
        return triples
    
    def _create_document_uri(self, identifier: str) -> str:
        """Create a URI for a document."""
        clean_id = identifier.replace("\\", "/").replace(" ", "_")
        return f"futurnal:doc/{clean_id}"


class TripleEnrichedNormalizationSink:
    """Enhanced normalization sink that extracts semantic triples.
    
    Supports two extraction modes:
    - Phase 1 (Archivist): Metadata-only extraction (backward compatible)
    - Phase 2 (Analyst): AI-powered extraction with temporal/event/causal analysis
    """

    def __init__(self, pkg_writer, vector_writer, enable_advanced_extraction: bool = False):
        """Initialize triple-enriched sink.
        
        Args:
            pkg_writer: PKG writer instance
            vector_writer: Vector writer instance
            enable_advanced_extraction: Enable Phase 2 (Analyst) extraction mode
                                       Requires local LLM and additional compute
        """
        from ..pipeline.stubs import PKGWriter, VectorWriter

        self.pkg_writer: PKGWriter = pkg_writer
        self.vector_writer: VectorWriter = vector_writer
        
        # Phase 1 extractor (always available)
        self.metadata_extractor = MetadataTripleExtractor()
        
        # Phase 2 extractor (optional, requires local LLM)
        self.enable_advanced_extraction = enable_advanced_extraction
        if self.enable_advanced_extraction:
            try:
                self.advanced_extractor = AdvancedTripleExtractor()
                logger.info("Phase 2 (Analyst) extraction enabled")
            except Exception as e:
                logger.error(f"Failed to initialize advanced extraction: {e}")
                logger.warning("Falling back to Phase 1 (Archivist) extraction")
                self.advanced_extractor = None
                self.enable_advanced_extraction = False
        else:
            self.advanced_extractor = None
            logger.info("Phase 1 (Archivist) extraction enabled")
        
    def handle(self, element: dict) -> None:
        """Handle element with semantic triple extraction.
        
        Extracts triples using either Phase 1 (metadata) or Phase 2 (advanced)
        extraction based on configuration.
        """
        # Load element data
        with open(element["element_path"], "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        
        # Extract semantic triples
        # Phase 1: Metadata extraction (always run for backward compat)
        metadata_triples = self.metadata_extractor.extract_triples(payload)
        
        # Phase 2: Advanced extraction (optional)
        advanced_triples = []
        if self.enable_advanced_extraction and self.advanced_extractor:
            try:
                advanced_triples = self.advanced_extractor.extract_triples(payload)
                logger.debug(
                    f"Advanced extraction: {len(advanced_triples)} triples",
                    extra={"element_id": element.get("sha256", "unknown")}
                )
            except Exception as e:
                logger.error(f"Advanced extraction failed: {e}")
        
        # Combine triples
        all_triples = metadata_triples + advanced_triples
        
        # Enrich payload with extracted triples
        payload["semantic_triples"] = [triple.to_dict() for triple in all_triples]
        
        # Create document payload
        document_payload = {
            "sha256": element["sha256"],
            "path": element["path"],
            "source": element["source"],
            "metadata": payload.get("metadata", {}),
            "text": payload.get("text"),
            "semantic_triples": payload["semantic_triples"],
        }
        
        # Write to PKG
        self.pkg_writer.write_document(document_payload)
        
        # Create embedding payload
        embedding_payload = {
            "sha256": element["sha256"],
            "path": element["path"],
            "source": element["source"],
            "text": payload.get("text"),
        }
        if "embedding" in payload:
            embedding_payload["embedding"] = payload["embedding"]
            
        # Write to vector store
        self.vector_writer.write_embedding(embedding_payload)
        
        logger.debug(
            f"Processed element with {len(all_triples)} semantic triples "
            f"({len(metadata_triples)} metadata, {len(advanced_triples)} advanced)",
            extra={
                "element_path": element["path"],
                "triple_count": len(all_triples),
                "metadata_count": len(metadata_triples),
                "advanced_count": len(advanced_triples),
            }
        )
    
    def handle_deletion(self, element: dict) -> None:
        """Handle element deletion."""
        self.pkg_writer.remove_document(element["sha256"])
        self.vector_writer.remove_embedding(element["sha256"])

    def handle_path_change(self, path_change_data: dict) -> None:
        """Handle note path changes by updating PKG relationships.

        Args:
            path_change_data: Dictionary containing path change information
                - vault_id: Vault identifier
                - old_note_id: Previous note ID
                - new_note_id: New note ID
                - old_path: Previous file path
                - new_path: New file path
                - change_type: Type of change (rename, move, etc.)
        """
        try:
            # Check if this is a PKG writer that supports note operations
            if hasattr(self.pkg_writer, 'update_note_path'):
                self.pkg_writer.update_note_path(
                    vault_id=path_change_data["vault_id"],
                    note_id=path_change_data["old_note_id"],
                    old_path=path_change_data["old_path"],
                    new_path=path_change_data["new_path"]
                )

                logger.info(
                    f"Updated note path in PKG: {path_change_data['old_path']} -> {path_change_data['new_path']}",
                    extra={
                        "vault_id": path_change_data["vault_id"],
                        "old_note_id": path_change_data["old_note_id"],
                        "new_note_id": path_change_data["new_note_id"],
                        "change_type": path_change_data["change_type"],
                    }
                )
            else:
                logger.warning(
                    f"PKG writer does not support path updates: {type(self.pkg_writer).__name__}"
                )

        except Exception as e:
            logger.error(
                f"Failed to handle path change in PKG: {e}",
                extra={
                    "vault_id": path_change_data.get("vault_id"),
                    "old_path": path_change_data.get("old_path"),
                    "new_path": path_change_data.get("new_path"),
                    "error": str(e),
                }
            )


