"""Semantic triple extraction pipeline for Personal Knowledge Graph construction."""

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
    """Extracts semantic triples from structured document metadata."""
    
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


class TripleEnrichedNormalizationSink:
    """Enhanced normalization sink that extracts semantic triples."""
    
    def __init__(self, pkg_writer, vector_writer):
        from ..pipeline.stubs import PKGWriter, VectorWriter
        
        self.pkg_writer: PKGWriter = pkg_writer
        self.vector_writer: VectorWriter = vector_writer
        self.triple_extractor = MetadataTripleExtractor()
        
    def handle(self, element: dict) -> None:
        """Handle element with semantic triple extraction."""
        # Load element data
        with open(element["element_path"], "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        
        # Extract semantic triples
        triples = self.triple_extractor.extract_triples(payload)
        
        # Enrich payload with extracted triples
        payload["semantic_triples"] = [triple.to_dict() for triple in triples]
        
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
            f"Processed element with {len(triples)} semantic triples",
            extra={
                "element_path": element["path"],
                "triple_count": len(triples),
                "extraction_count": self.triple_extractor.extracted_count,
            }
        )
    
    def handle_deletion(self, element: dict) -> None:
        """Handle element deletion."""
        self.pkg_writer.remove_document(element["sha256"])
        self.vector_writer.remove_embedding(element["sha256"])
