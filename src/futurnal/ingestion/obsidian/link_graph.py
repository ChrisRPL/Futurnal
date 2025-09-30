"""Link Graph Construction for Obsidian vaults.

This module implements the link graph construction subsystem as specified in
04-link-graph-construction.md. It transforms wikilinks and tags into semantic
relationships for the PKG with full provenance tracking.

The implementation builds upon the existing ObsidianLinkParser infrastructure
while adding dedicated graph construction logic for note-to-note relationships.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .normalizer import NormalizedDocument, ObsidianLink, ObsidianTag
from .assets import ObsidianAsset
from .security import PathTraversalValidator, ResourceLimiter, SecurityError

logger = logging.getLogger(__name__)


@dataclass
class NoteNode:
    """Represents a Note node in the PKG with Obsidian-specific metadata."""
    vault_id: str
    note_id: str  # Derived from file path, unique within vault
    title: str
    path: Path
    checksum: str  # Content checksum for change detection
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_uri(self) -> str:
        """Generate a unique URI for this note node."""
        return f"futurnal:note/{self.vault_id}/{self.note_id}"

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "vault_id": self.vault_id,
            "note_id": self.note_id,
            "title": self.title,
            "path": str(self.path),
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "uri": self.to_uri()
        }


@dataclass
class LinkRelationship:
    """Represents a relationship between notes with provenance metadata."""
    source_note: NoteNode
    target_note: NoteNode
    relationship_type: str  # 'links_to', 'references_heading', 'has_tag', 'embeds'
    source_path: str  # Original file path for auditability
    offset: Optional[int] = None  # Character offset in source file
    checksum: str = ""  # Checksum of the relationship context
    heading: Optional[str] = None  # For references_heading relationships
    block_id: Optional[str] = None  # For block references
    display_text: Optional[str] = None  # Original display text
    is_broken: bool = False  # Whether target note exists
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "source_uri": self.source_note.to_uri(),
            "target_uri": self.target_note.to_uri(),
            "relationship_type": self.relationship_type,
            "source_path": self.source_path,
            "offset": self.offset,
            "checksum": self.checksum,
            "heading": self.heading,
            "block_id": self.block_id,
            "display_text": self.display_text,
            "is_broken": self.is_broken,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TagRelationship:
    """Represents a relationship between a note and a tag."""
    note: NoteNode
    tag_name: str
    tag_uri: str
    source_path: str
    offset: Optional[int] = None
    is_nested: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "note_uri": self.note.to_uri(),
            "tag_name": self.tag_name,
            "tag_uri": self.tag_uri,
            "source_path": self.source_path,
            "offset": self.offset,
            "is_nested": self.is_nested,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class AssetRelationship:
    """Represents a relationship between a note and an embedded asset (image, PDF, etc.)."""
    note: NoteNode
    asset: ObsidianAsset
    source_path: str
    offset: Optional[int] = None
    relationship_type: str = "embeds"  # Default to 'embeds' for assets
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_uri(self) -> str:
        """Generate a unique URI for this asset."""
        if self.asset.content_hash:
            return f"futurnal:asset/{self.note.vault_id}/{self.asset.content_hash}"
        else:
            # Fallback to path-based URI if no content hash
            asset_path = str(self.asset.resolved_path) if self.asset.resolved_path else self.asset.target
            safe_path = asset_path.replace('/', '_').replace('\\', '_')
            return f"futurnal:asset/{self.note.vault_id}/{safe_path}"

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "note_uri": self.note.to_uri(),
            "asset_uri": self.to_uri(),
            "asset_target": self.asset.target,
            "asset_path": str(self.asset.resolved_path) if self.asset.resolved_path else None,
            "asset_type": "image" if self.asset.is_image else ("pdf" if self.asset.is_pdf else "unknown"),
            "content_hash": self.asset.content_hash,
            "file_size": self.asset.file_size,
            "mime_type": self.asset.mime_type,
            "is_broken": self.asset.is_broken,
            "is_processable": self.asset.is_processable,
            "relationship_type": self.relationship_type,
            "source_path": self.source_path,
            "offset": self.offset,
            "created_at": self.created_at.isoformat()
        }


class ObsidianLinkGraphConstructor:
    """Constructs note relationship graphs from normalized Obsidian documents.

    This class implements the link graph construction logic as specified in
    04-link-graph-construction.md, transforming wikilinks and tags into
    semantic relationships with full provenance tracking.
    """

    def __init__(
        self,
        vault_id: str,
        vault_root: Path,
        *,
        path_validator: Optional[PathTraversalValidator] = None,
        resource_limiter: Optional[ResourceLimiter] = None,
        enable_backlinks: bool = True,
        enable_cycle_detection: bool = True
    ):
        self.vault_id = vault_id
        self.vault_root = Path(vault_root)
        self.path_validator = path_validator or PathTraversalValidator(vault_root)
        self.resource_limiter = resource_limiter or ResourceLimiter()
        self.enable_backlinks = enable_backlinks
        self.enable_cycle_detection = enable_cycle_detection

        # Graph state tracking
        self.note_registry: Dict[str, NoteNode] = {}
        self.relationship_registry: Set[Tuple[str, str, str]] = set()  # (source_uri, predicate, target_uri)
        self.processing_stack: Set[str] = set()  # For cycle detection

        # Statistics
        self.notes_created = 0
        self.relationships_created = 0
        self.asset_relationships_created = 0
        self.cycles_detected = 0
        self.broken_links_found = 0
        self.broken_assets_found = 0

    def construct_graph(self, normalized_doc: NormalizedDocument) -> Tuple[List[NoteNode], List[LinkRelationship], List[TagRelationship], List[AssetRelationship]]:
        """Construct note relationship graph from normalized document.

        Args:
            normalized_doc: Normalized Obsidian document with parsed metadata

        Returns:
            Tuple of (note_nodes, link_relationships, tag_relationships, asset_relationships)
        """
        logger.debug(f"Constructing graph for document: {normalized_doc.provenance.source_path}")

        # Create primary note node
        source_note = self._create_note_node(normalized_doc)

        # Process wikilinks into relationships
        link_relationships = self._process_wikilinks(source_note, normalized_doc)

        # Process assets into relationships
        asset_relationships = self._process_assets(source_note, normalized_doc)

        # Process tags into relationships
        tag_relationships = self._process_tags(source_note, normalized_doc)

        # Generate backlinks if enabled
        if self.enable_backlinks:
            backlink_relationships = self._infer_backlinks(link_relationships)
            link_relationships.extend(backlink_relationships)

        # Collect all note nodes (source + any created during processing)
        all_notes = [source_note]
        for note_id, note in self.note_registry.items():
            if note.note_id != source_note.note_id:
                all_notes.append(note)

        logger.debug(
            f"Graph construction completed: {len(all_notes)} notes, "
            f"{len(link_relationships)} link relationships, "
            f"{len(asset_relationships)} asset relationships, "
            f"{len(tag_relationships)} tag relationships"
        )

        return all_notes, link_relationships, tag_relationships, asset_relationships

    def _create_note_node(self, normalized_doc: NormalizedDocument) -> NoteNode:
        """Create a note node from normalized document."""
        source_path = normalized_doc.provenance.source_path

        # Generate note ID from path (relative to vault root)
        try:
            relative_path = source_path.relative_to(self.vault_root)
            note_id = str(relative_path.with_suffix(''))  # Remove .md extension
        except ValueError:
            # Path is not relative to vault root, use absolute path
            note_id = str(source_path.with_suffix(''))

        # Extract title (from frontmatter, or filename as fallback)
        title = normalized_doc.metadata.frontmatter.get('title')
        if not title:
            title = source_path.stem  # Filename without extension

        note_node = NoteNode(
            vault_id=self.vault_id,
            note_id=note_id,
            title=str(title),
            path=source_path,
            checksum=normalized_doc.provenance.content_checksum
        )

        # Register in note registry
        self.note_registry[note_id] = note_node
        self.notes_created += 1

        logger.debug(f"Created note node: {note_node.to_uri()}")
        return note_node

    def _process_wikilinks(self, source_note: NoteNode, normalized_doc: NormalizedDocument) -> List[LinkRelationship]:
        """Process wikilinks into link relationships."""
        relationships = []

        for link in normalized_doc.metadata.links:
            try:
                # Validate link security
                self.path_validator.validate_link_path(link.target, source_note.path)

                # Create or retrieve target note node
                target_note = self._create_or_get_target_note(link, source_note)

                # Determine relationship type
                relationship_type = self._determine_relationship_type(link)

                # Create relationship
                relationship = LinkRelationship(
                    source_note=source_note,
                    target_note=target_note,
                    relationship_type=relationship_type,
                    source_path=str(source_note.path),
                    offset=link.start_pos,
                    heading=link.section,
                    block_id=link.block_id,
                    display_text=link.display_text,
                    is_broken=link.is_broken,
                    checksum=self._compute_relationship_checksum(source_note, target_note, link)
                )

                # Check for duplicates and cycles
                if self._should_add_relationship(relationship):
                    relationships.append(relationship)
                    self.relationships_created += 1

                if link.is_broken:
                    self.broken_links_found += 1

            except SecurityError as e:
                logger.warning(f"Skipping unsafe link '{link.target}' from {source_note.path}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process link '{link.target}' from {source_note.path}: {e}")
                continue

        return relationships

    def _process_tags(self, source_note: NoteNode, normalized_doc: NormalizedDocument) -> List[TagRelationship]:
        """Process tags into tag relationships."""
        relationships = []

        for tag in normalized_doc.metadata.tags:
            try:
                # Create tag URI
                tag_uri = self._create_tag_uri(tag.name)

                relationship = TagRelationship(
                    note=source_note,
                    tag_name=tag.name,
                    tag_uri=tag_uri,
                    source_path=str(source_note.path),
                    offset=tag.start_pos,
                    is_nested=tag.is_nested
                )

                relationships.append(relationship)

            except Exception as e:
                logger.error(f"Failed to process tag '{tag.name}' from {source_note.path}: {e}")
                continue

        return relationships

    def _process_assets(self, source_note: NoteNode, normalized_doc: NormalizedDocument) -> List[AssetRelationship]:
        """Process assets into asset relationships."""
        relationships = []

        for asset in normalized_doc.metadata.assets:
            try:
                # Create asset relationship
                relationship = AssetRelationship(
                    note=source_note,
                    asset=asset,
                    source_path=str(source_note.path),
                    offset=asset.start_pos,
                    relationship_type="embeds"
                )

                relationships.append(relationship)
                self.asset_relationships_created += 1

                # Log asset processing and track statistics
                if asset.is_broken:
                    logger.debug(f"Asset '{asset.target}' from {source_note.path} is broken/missing")
                    self.broken_assets_found += 1
                else:
                    logger.debug(f"Processed asset '{asset.target}' from {source_note.path}")

            except Exception as e:
                logger.error(f"Failed to process asset '{asset.target}' from {source_note.path}: {e}")
                continue

        return relationships

    def _create_or_get_target_note(self, link: ObsidianLink, source_note: NoteNode) -> NoteNode:
        """Create or retrieve target note node for a link."""

        # Generate target note ID
        target_note_id = link.target
        if not target_note_id.endswith('.md'):
            target_note_id = link.target  # Keep as-is for note references

        # Check if already exists in registry
        if target_note_id in self.note_registry:
            return self.note_registry[target_note_id]

        # Create new target note node
        if link.resolved_path and link.resolved_path.exists():
            # Target exists, create from actual file
            target_path = link.resolved_path
            title = target_path.stem

            # Compute checksum if file exists
            try:
                content = target_path.read_text(encoding='utf-8')
                checksum = hashlib.sha256(content.encode('utf-8')).hexdigest()
            except Exception:
                checksum = "unknown"
        else:
            # Target doesn't exist, create placeholder node
            target_path = self.vault_root / f"{target_note_id}.md"
            title = target_note_id
            checksum = "missing"

        target_note = NoteNode(
            vault_id=self.vault_id,
            note_id=target_note_id,
            title=title,
            path=target_path,
            checksum=checksum
        )

        # Register the target note
        self.note_registry[target_note_id] = target_note
        self.notes_created += 1

        logger.debug(f"Created target note node: {target_note.to_uri()}")
        return target_note

    def _determine_relationship_type(self, link: ObsidianLink) -> str:
        """Determine the relationship type based on link properties."""
        if link.is_embed:
            return "embeds"
        elif link.section:
            return "references_heading"
        elif link.block_id:
            return "references_block"
        else:
            return "links_to"

    def _compute_relationship_checksum(self, source_note: NoteNode, target_note: NoteNode, link: ObsidianLink) -> str:
        """Compute checksum for relationship deduplication."""
        relationship_data = f"{source_note.to_uri()}|{self._determine_relationship_type(link)}|{target_note.to_uri()}|{link.section or ''}|{link.block_id or ''}"
        return hashlib.sha256(relationship_data.encode('utf-8')).hexdigest()[:16]

    def _should_add_relationship(self, relationship: LinkRelationship) -> bool:
        """Check if relationship should be added (deduplication and cycle detection)."""

        # Create relationship signature for deduplication
        signature = (
            relationship.source_note.to_uri(),
            relationship.relationship_type,
            relationship.target_note.to_uri()
        )

        # Check for duplicates
        if signature in self.relationship_registry:
            logger.debug(f"Skipping duplicate relationship: {signature}")
            return False

        # Cycle detection if enabled
        if self.enable_cycle_detection:
            if self._would_create_cycle(relationship):
                logger.warning(f"Detected cycle in relationship: {signature}")
                self.cycles_detected += 1
                return True  # Still add the relationship, but log the cycle

        # Add to registry
        self.relationship_registry.add(signature)
        return True

    def _would_create_cycle(self, relationship: LinkRelationship) -> bool:
        """Check if adding this relationship would create a cycle."""

        source_uri = relationship.source_note.to_uri()
        target_uri = relationship.target_note.to_uri()

        # Simple cycle detection: check if target already links back to source
        reverse_signature = (target_uri, relationship.relationship_type, source_uri)
        return reverse_signature in self.relationship_registry

    def _infer_backlinks(self, relationships: List[LinkRelationship]) -> List[LinkRelationship]:
        """Infer bidirectional relationships (backlinks)."""
        backlinks = []

        for relationship in relationships:
            # Only create backlinks for certain relationship types
            if relationship.relationship_type in ["links_to", "references_heading"]:

                # Create reverse relationship
                backlink = LinkRelationship(
                    source_note=relationship.target_note,
                    target_note=relationship.source_note,
                    relationship_type="linked_from",
                    source_path=relationship.source_path,  # Keep original source for provenance
                    checksum=f"backlink_{relationship.checksum}",
                    is_broken=relationship.is_broken
                )

                # Check if backlink should be added
                if self._should_add_relationship(backlink):
                    backlinks.append(backlink)

        logger.debug(f"Generated {len(backlinks)} backlink relationships")
        return backlinks

    def _create_tag_uri(self, tag_name: str) -> str:
        """Create a URI for a tag."""
        # Normalize tag name for URI
        clean_tag = tag_name.replace("/", "_").replace(" ", "_").lower()
        return f"futurnal:tag/obsidian/{clean_tag}"

    def get_statistics(self) -> Dict[str, int]:
        """Get construction statistics."""
        return {
            "notes_created": self.notes_created,
            "relationships_created": self.relationships_created,
            "asset_relationships_created": self.asset_relationships_created,
            "cycles_detected": self.cycles_detected,
            "broken_links_found": self.broken_links_found,
            "broken_assets_found": self.broken_assets_found,
            "notes_in_registry": len(self.note_registry),
            "relationships_in_registry": len(self.relationship_registry)
        }

    def clear_state(self) -> None:
        """Clear internal state for processing new documents."""
        self.note_registry.clear()
        self.relationship_registry.clear()
        self.processing_stack.clear()

        # Reset statistics
        self.notes_created = 0
        self.relationships_created = 0
        self.cycles_detected = 0
        self.broken_links_found = 0