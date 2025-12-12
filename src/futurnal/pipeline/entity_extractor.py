"""LLM-based entity extraction for Futurnal's Personal Knowledge Graph.

Extracts meaningful Person, Concept, Organization, and Event entities
using local LLM (Llama 3.1 8B via Ollama) for quality extraction.

Phase 1 (Archivist) extracts:
- Person: Email sender/recipient, document authors, mentioned people
- Organization: Companies, institutions mentioned
- Concept: Key topics, themes, technologies discussed
- Document titles from content (not raw filenames)

This replaces the naive spaCy NER approach with LLM-based extraction
that understands context and produces meaningful entities.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from a document or email."""

    id: str
    name: str
    entity_type: str  # Person, Concept, Organization, Event
    canonical_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    extraction_method: str = "llm"
    source_document_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name or self.name,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "source_document_id": self.source_document_id,
        }


@dataclass
class ExtractedRelationship:
    """A relationship between a document and an entity."""

    source_id: str  # Document/email ID
    target_id: str  # Entity ID
    relationship: str  # sent_by, received_by, authored_by, discusses, mentions
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relationship": self.relationship,
            "confidence": self.confidence,
        }


@dataclass
class DocumentEntities:
    """Entities and relationships extracted from a single document."""

    document_id: str
    source: Optional[str] = None
    title: Optional[str] = None  # Extracted document title
    extracted_at: Optional[str] = None
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)

    def __post_init__(self):
        if self.extracted_at is None:
            self.extracted_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "extracted_at": self.extracted_at,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
        }


# LLM prompt for entity extraction
ENTITY_EXTRACTION_PROMPT = """Extract key entities from this document. Return ONLY a JSON object.

Document content:
{content}

Extract:
1. People mentioned (real names only, not generic terms)
2. Organizations (companies, institutions, teams)
3. Key concepts/topics (specific technologies, methodologies, subjects)

Rules:
- Only extract SPECIFIC, MEANINGFUL entities
- Skip generic words like "user", "system", "data", "file"
- Skip partial words or fragments
- Each entity must be clearly identifiable
- Confidence 0.0-1.0 based on how clearly the entity is mentioned

Return JSON:
{{
  "title": "extracted document title or null",
  "entities": [
    {{"name": "Entity Name", "type": "Person|Organization|Concept", "confidence": 0.9}}
  ]
}}

JSON response:"""


class EntityExtractor:
    """LLM-based entity extraction for semantic knowledge graphs.

    Uses Llama 3.1 8B via Ollama for intelligent entity extraction
    that understands context and produces meaningful connections.
    """

    def __init__(self, workspace_dir: Optional[Path] = None):
        """Initialize entity extractor.

        Args:
            workspace_dir: Futurnal workspace directory (~/.futurnal/workspace)
        """
        self.workspace_dir = workspace_dir or Path.home() / ".futurnal" / "workspace"
        self.parsed_dir = self.workspace_dir / "parsed"
        self.entities_dir = self.workspace_dir / "entities"
        self.imap_dir = self.workspace_dir / "imap"

        # Ensure entities directory exists
        self.entities_dir.mkdir(parents=True, exist_ok=True)

        # LLM client (lazy loaded)
        self._llm = None

    def _get_llm(self):
        """Get or create LLM client."""
        if self._llm is None:
            try:
                from futurnal.extraction.ollama_client import OllamaLLMClient, ollama_available

                if ollama_available():
                    self._llm = OllamaLLMClient(
                        model_name="meta-llama/Llama-3.1-8B-Instruct",
                        timeout=60
                    )
                    logger.info("Using Ollama LLM for entity extraction")
                else:
                    logger.warning("Ollama not available, LLM extraction disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
        return self._llm

    def _create_entity_id(self, entity_type: str, name: str) -> str:
        """Create a unique entity ID from type and name."""
        normalized = name.lower().strip()
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        normalized = normalized.strip("_")[:30]
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{entity_type.lower()}:{normalized}_{name_hash}"

    def _extract_title_from_content(self, content: str, metadata: Dict) -> Optional[str]:
        """Extract document title from content or metadata.

        Priority:
        1. Frontmatter title
        2. First markdown heading
        3. Email subject
        4. None (will use LLM or filename later)
        """
        # Try frontmatter title
        frontmatter = metadata.get("frontmatter", {})
        if not frontmatter:
            extra = metadata.get("extra", {})
            frontmatter = extra.get("frontmatter", {})

        if frontmatter.get("title"):
            return str(frontmatter["title"])

        # Try first markdown heading
        heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()

        # Try email subject
        if metadata.get("subject"):
            return metadata["subject"]

        return None

    def _normalize_email_address(self, email_str: str) -> tuple[str, Optional[str]]:
        """Extract name and email from an email address string."""
        if not email_str:
            return "", None

        match = re.match(r'^"?([^"<]+)"?\s*<([^>]+)>', email_str)
        if match:
            name = match.group(1).strip()
            email = match.group(2).strip()
            return name, email

        email_match = re.match(r"^([^@]+)@([^@]+)$", email_str.strip())
        if email_match:
            local_part = email_match.group(1)
            name = local_part.replace(".", " ").replace("_", " ").title()
            return name, email_str.strip()

        return email_str.strip(), None

    def _extract_with_llm(self, content: str, doc_id: str) -> tuple[Optional[str], List[ExtractedEntity]]:
        """Extract entities using LLM.

        Returns:
            Tuple of (extracted_title, list_of_entities)
        """
        llm = self._get_llm()
        if not llm:
            return None, []

        # Truncate content for LLM context
        truncated = content[:3000] if len(content) > 3000 else content

        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(content=truncated)
            response = llm.generate(prompt, temperature=0.1, max_tokens=500)

            # Parse JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                logger.debug(f"No JSON found in LLM response for {doc_id}")
                return None, []

            data = json.loads(json_match.group())
            title = data.get("title")
            entities = []

            for ent in data.get("entities", []):
                name = ent.get("name", "").strip()
                ent_type = ent.get("type", "Concept")
                confidence = float(ent.get("confidence", 0.7))

                # Filter garbage
                if not name or len(name) < 2 or len(name) > 100:
                    continue
                if confidence < 0.5:
                    continue
                # Skip generic terms
                if name.lower() in {"user", "system", "data", "file", "document", "email", "the", "a", "an"}:
                    continue

                # Normalize type
                if ent_type not in {"Person", "Organization", "Concept", "Event"}:
                    ent_type = "Concept"

                entity_id = self._create_entity_id(ent_type, name)
                entity = ExtractedEntity(
                    id=entity_id,
                    name=name,
                    entity_type=ent_type,
                    confidence=confidence,
                    extraction_method="llm",
                    source_document_id=doc_id,
                )
                entities.append(entity)

            return title, entities

        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse LLM JSON for {doc_id}: {e}")
            return None, []
        except Exception as e:
            logger.warning(f"LLM extraction failed for {doc_id}: {e}")
            return None, []

    def extract_from_email(self, element: Dict[str, Any]) -> DocumentEntities:
        """Extract entities from an email element."""
        doc_id = element.get("sha256", "unknown")
        metadata = element.get("metadata", {})
        source = metadata.get("source")
        content = element.get("text", "") or element.get("content", "")

        result = DocumentEntities(document_id=doc_id, source=source)

        # Extract title from subject
        result.title = metadata.get("subject")

        # Extract sender as Person
        sender = metadata.get("sender", "")
        if sender:
            name, email = self._normalize_email_address(sender)
            if name and len(name) > 1:
                entity_id = self._create_entity_id("person", name)
                entity = ExtractedEntity(
                    id=entity_id,
                    name=name,
                    entity_type="Person",
                    aliases=[email] if email else [],
                    confidence=0.95,
                    extraction_method="email_sender",
                    source_document_id=doc_id,
                )
                result.entities.append(entity)
                result.relationships.append(
                    ExtractedRelationship(
                        source_id=doc_id,
                        target_id=entity_id,
                        relationship="sent_by",
                        confidence=0.95,
                    )
                )

        # Extract recipients as Person
        recipient = metadata.get("recipient", "")
        if recipient:
            recipients = [r.strip() for r in recipient.split(",")]
            for recip in recipients[:5]:  # Limit to 5 recipients
                if recip:
                    name, email = self._normalize_email_address(recip)
                    if name and len(name) > 1:
                        entity_id = self._create_entity_id("person", name)
                        entity = ExtractedEntity(
                            id=entity_id,
                            name=name,
                            entity_type="Person",
                            aliases=[email] if email else [],
                            confidence=0.9,
                            extraction_method="email_recipient",
                            source_document_id=doc_id,
                        )
                        result.entities.append(entity)
                        result.relationships.append(
                            ExtractedRelationship(
                                source_id=doc_id,
                                target_id=entity_id,
                                relationship="sent_to",
                                confidence=0.9,
                            )
                        )

        # Extract entities from email body using LLM
        if content and len(content) > 50:
            _, llm_entities = self._extract_with_llm(content, doc_id)
            for entity in llm_entities:
                # Avoid duplicates
                if entity.id not in {e.id for e in result.entities}:
                    result.entities.append(entity)
                    result.relationships.append(
                        ExtractedRelationship(
                            source_id=doc_id,
                            target_id=entity.id,
                            relationship="mentions",
                            confidence=entity.confidence,
                        )
                    )

        return result

    def extract_from_document(self, element: Dict[str, Any]) -> DocumentEntities:
        """Extract entities from a document element."""
        metadata = element.get("metadata", {})
        # Use parent_id to match graph.rs document aggregation (doc:{parent_id})
        doc_id = metadata.get("parent_id") or element.get("element_id") or element.get("sha256", "unknown")
        source = metadata.get("source")
        content = element.get("text", "") or element.get("content", "")

        result = DocumentEntities(document_id=doc_id, source=source)

        # Extract title from content/metadata first
        result.title = self._extract_title_from_content(content, metadata)

        # Check for frontmatter
        frontmatter = metadata.get("frontmatter", {})
        extra = metadata.get("extra", {})
        if not frontmatter and "frontmatter" in extra:
            frontmatter = extra.get("frontmatter", {})

        # Extract author as Person (high confidence - explicitly stated)
        author = frontmatter.get("author") or extra.get("author")
        if author:
            authors = author if isinstance(author, list) else [author]
            for auth in authors:
                if auth and len(str(auth)) > 1:
                    entity_id = self._create_entity_id("person", str(auth))
                    entity = ExtractedEntity(
                        id=entity_id,
                        name=str(auth),
                        entity_type="Person",
                        confidence=0.95,
                        extraction_method="frontmatter_author",
                        source_document_id=doc_id,
                    )
                    result.entities.append(entity)
                    result.relationships.append(
                        ExtractedRelationship(
                            source_id=doc_id,
                            target_id=entity_id,
                            relationship="authored_by",
                            confidence=0.95,
                        )
                    )

        # Extract tags as Concepts (high confidence - explicitly stated)
        tags = frontmatter.get("tags") or extra.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        for tag in tags:
            if tag and isinstance(tag, str) and len(tag) > 1:
                entity_id = self._create_entity_id("concept", tag)
                entity = ExtractedEntity(
                    id=entity_id,
                    name=tag,
                    entity_type="Concept",
                    confidence=0.9,
                    extraction_method="frontmatter_tag",
                    source_document_id=doc_id,
                )
                result.entities.append(entity)
                result.relationships.append(
                    ExtractedRelationship(
                        source_id=doc_id,
                        target_id=entity_id,
                        relationship="discusses",
                        confidence=0.9,
                    )
                )

        # Extract from Obsidian tags
        obsidian_tags = metadata.get("obsidian_tags", [])
        for tag_info in obsidian_tags:
            tag_name = tag_info.get("name") if isinstance(tag_info, dict) else str(tag_info)
            if tag_name and len(tag_name) > 1:
                entity_id = self._create_entity_id("concept", tag_name)
                # Skip if already added
                if entity_id in {e.id for e in result.entities}:
                    continue
                entity = ExtractedEntity(
                    id=entity_id,
                    name=tag_name,
                    entity_type="Concept",
                    confidence=0.9,
                    extraction_method="obsidian_tag",
                    source_document_id=doc_id,
                )
                result.entities.append(entity)
                result.relationships.append(
                    ExtractedRelationship(
                        source_id=doc_id,
                        target_id=entity_id,
                        relationship="discusses",
                        confidence=0.9,
                    )
                )

        # Use LLM for content-based extraction
        if content and len(content) > 100:
            llm_title, llm_entities = self._extract_with_llm(content, doc_id)

            # Use LLM title if we don't have one yet
            if not result.title and llm_title:
                result.title = llm_title

            for entity in llm_entities:
                # Avoid duplicates
                if entity.id not in {e.id for e in result.entities}:
                    result.entities.append(entity)
                    result.relationships.append(
                        ExtractedRelationship(
                            source_id=doc_id,
                            target_id=entity.id,
                            relationship="mentions",
                            confidence=entity.confidence,
                        )
                    )

        # Fallback title: clean filename
        if not result.title:
            filename = metadata.get("filename", "")
            if filename:
                name_without_ext = re.sub(r"\.[^.]+$", "", filename)
                result.title = name_without_ext.replace("-", " ").replace("_", " ").title()

        return result

    def extract_from_element(self, element: Dict[str, Any]) -> DocumentEntities:
        """Extract entities from any parsed element."""
        metadata = element.get("metadata", {})
        source_type = metadata.get("source_type", "")

        # Check if this is an email
        if source_type == "imap" or "sender" in metadata or "@" in metadata.get("source", ""):
            return self.extract_from_email(element)
        else:
            return self.extract_from_document(element)

    def save_entities(self, doc_entities: DocumentEntities) -> Optional[Path]:
        """Save extracted entities to JSON file."""
        if not doc_entities.entities:
            return None

        doc_id = doc_entities.document_id
        filename = f"{doc_id[:32]}.json"
        output_path = self.entities_dir / filename

        with open(output_path, "w") as f:
            json.dump(doc_entities.to_dict(), f, indent=2)

        logger.debug(f"Saved {len(doc_entities.entities)} entities to {output_path}")
        return output_path

    def process_all_parsed(self) -> int:
        """Process all parsed JSON files and extract entities."""
        processed = 0
        total_entities = 0

        # Process parsed documents
        if self.parsed_dir.exists():
            for json_file in self.parsed_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        element = json.load(f)

                    doc_entities = self.extract_from_element(element)

                    if doc_entities.entities:
                        self.save_entities(doc_entities)
                        processed += 1
                        total_entities += len(doc_entities.entities)

                except Exception as e:
                    logger.warning(f"Failed to process {json_file}: {e}")
                    continue

        # Process IMAP emails
        if self.imap_dir.exists():
            for json_file in self.imap_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        element = json.load(f)

                    doc_entities = self.extract_from_email(element)

                    if doc_entities.entities:
                        self.save_entities(doc_entities)
                        processed += 1
                        total_entities += len(doc_entities.entities)

                except Exception as e:
                    logger.warning(f"Failed to process {json_file}: {e}")
                    continue

        logger.info(f"Processed {processed} documents, extracted {total_entities} entities")
        return processed


def extract_entities_for_document(
    element: Dict[str, Any],
    workspace_dir: Optional[Path] = None,
) -> DocumentEntities:
    """Convenience function to extract entities from a single element."""
    extractor = EntityExtractor(workspace_dir)
    return extractor.extract_from_element(element)


def process_workspace_entities(workspace_dir: Optional[Path] = None) -> int:
    """Process all parsed documents in workspace and extract entities."""
    extractor = EntityExtractor(workspace_dir)
    return extractor.process_all_parsed()
