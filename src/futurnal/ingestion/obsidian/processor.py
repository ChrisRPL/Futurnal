"""Document processor that integrates MarkdownNormalizer with Unstructured.io pipeline."""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from unstructured.partition.auto import partition

from .normalizer import NormalizedDocument, normalize_obsidian_document
from ..local.state import FileRecord

logger = logging.getLogger(__name__)


class ObsidianDocumentProcessor:
    """Processes Obsidian documents through normalization and Unstructured.io pipeline."""
    
    def __init__(
        self,
        *,
        workspace_dir: Path,
        vault_root: Optional[Path] = None,
        vault_id: Optional[str] = None
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir = self.workspace_dir / "parsed"
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        
        self.vault_root = vault_root
        self.vault_id = vault_id
        
    def process_document(self, file_record: FileRecord, source_name: str) -> Iterator[Dict[str, Any]]:
        """Process a document through the full Obsidian -> Unstructured.io pipeline.
        
        Args:
            file_record: File metadata record
            source_name: Name of the ingestion source
            
        Yields:
            Element dictionaries suitable for the NormalizationSink
        """
        try:
            # Step 1: Read the markdown content
            content = file_record.path.read_text(encoding='utf-8')
            
            # Step 2: Normalize through MarkdownNormalizer
            normalized_doc = normalize_obsidian_document(
                content=content,
                source_path=file_record.path,
                vault_root=self.vault_root,
                vault_id=self.vault_id
            )
            
            # Step 3: Create enriched document for Unstructured.io
            enriched_content = self._create_enriched_content(normalized_doc)
            
            # Step 4: Process through Unstructured.io using temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(enriched_content)
                temp_path = Path(temp_file.name)
            
            try:
                elements = partition(
                    filename=str(temp_path),
                    strategy="fast",
                    include_metadata=True,
                    content_type="text/markdown"
                )
                
                # Step 5: Process each element and enrich with normalized metadata
                for element in elements:
                    element_data = self._create_element_data(
                        element=element,
                        normalized_doc=normalized_doc,
                        file_record=file_record,
                        source_name=source_name
                    )
                    yield element_data
                    
            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to process document {file_record.path}: {e}")
            raise
            
    def _create_enriched_content(self, normalized_doc: NormalizedDocument) -> str:
        """Create enriched content that includes normalized metadata for Unstructured.io.
        
        This combines the normalized text with structured metadata comments
        that can be preserved through the Unstructured.io processing.
        """
        content_parts = []
        
        # Add metadata header as HTML comments (preserved by Unstructured.io)
        if normalized_doc.metadata.frontmatter:
            content_parts.append("<!-- FUTURNAL_FRONTMATTER")
            content_parts.append(json.dumps(normalized_doc.metadata.frontmatter, ensure_ascii=False, default=str))
            content_parts.append("-->")
            content_parts.append("")
        
        # Add structured metadata as comment
        metadata_summary = {
            "futurnal_metadata": {
                "headings_count": len(normalized_doc.metadata.headings),
                "links_count": len(normalized_doc.metadata.links),
                "tags_count": len(normalized_doc.metadata.tags),
                "callouts_count": len(normalized_doc.metadata.callouts),
                "tables_count": len(normalized_doc.metadata.tables),
                "tasks_count": len(normalized_doc.metadata.task_lists),
                "footnotes_count": len(normalized_doc.metadata.footnotes),
                "word_count": normalized_doc.metadata.word_count,
                "reading_time_minutes": normalized_doc.metadata.reading_time_minutes,
                "links": [
                    {
                        "target": link.target,
                        "display_text": link.display_text,
                        "is_embed": link.is_embed,
                        "section": link.section,
                        "block_id": link.block_id,
                        "is_broken": link.is_broken
                    }
                    for link in normalized_doc.metadata.links
                ],
                "tags": [
                    {
                        "name": tag.name,
                        "is_nested": tag.is_nested
                    }
                    for tag in normalized_doc.metadata.tags
                ],
                "callouts": [
                    {
                        "type": callout.type.value,
                        "title": callout.title,
                        "fold_state": callout.fold_state.value
                    }
                    for callout in normalized_doc.metadata.callouts
                ]
            }
        }
        
        content_parts.append("<!-- FUTURNAL_STRUCTURED_METADATA")
        content_parts.append(json.dumps(metadata_summary, ensure_ascii=False))
        content_parts.append("-->")
        content_parts.append("")
        
        # Add the normalized content
        content_parts.append(normalized_doc.content)
        
        return "\n".join(content_parts)
    
    def _create_element_data(
        self,
        element: Any,
        normalized_doc: NormalizedDocument,
        file_record: FileRecord,
        source_name: str
    ) -> Dict[str, Any]:
        """Create element data dictionary for the NormalizationSink."""
        
        # Convert element to dict
        if hasattr(element, "to_dict"):
            payload = element.to_dict()
        elif isinstance(element, dict):
            payload = element
        else:
            payload = {"text": str(element)}
        
        # Ensure metadata exists
        payload.setdefault("metadata", {})
        
        # Enrich with file metadata
        payload["metadata"].update({
            "source": source_name,
            "path": str(file_record.path),
            "sha256": file_record.sha256,
            "size_bytes": file_record.size,
            "modified_at": datetime.utcfromtimestamp(file_record.mtime).isoformat(),
            "ingested_at": datetime.utcnow().isoformat(),
        })
        
        # Add Futurnal-specific metadata
        payload["metadata"]["futurnal"] = {
            "normalizer_version": normalized_doc.provenance.normalizer_version,
            "content_checksum": normalized_doc.provenance.content_checksum,
            "metadata_checksum": normalized_doc.provenance.metadata_checksum,
            "vault_id": normalized_doc.provenance.vault_id,
            "processed_at": normalized_doc.provenance.processed_at.isoformat(),
            "document_metadata": {
                "word_count": normalized_doc.metadata.word_count,
                "reading_time_minutes": normalized_doc.metadata.reading_time_minutes,
                "headings_count": len(normalized_doc.metadata.headings),
                "links_count": len(normalized_doc.metadata.links),
                "tags_count": len(normalized_doc.metadata.tags),
                "callouts_count": len(normalized_doc.metadata.callouts),
                "tables_count": len(normalized_doc.metadata.tables),
                "tasks_count": len(normalized_doc.metadata.task_lists),
                "footnotes_count": len(normalized_doc.metadata.footnotes),
            }
        }
        
        # Extract and preserve structured metadata for semantic triple generation
        if normalized_doc.metadata.frontmatter:
            payload["metadata"]["frontmatter"] = normalized_doc.metadata.frontmatter
            
        if normalized_doc.metadata.tags:
            payload["metadata"]["obsidian_tags"] = [
                {"name": tag.name, "is_nested": tag.is_nested}
                for tag in normalized_doc.metadata.tags
            ]
            
        if normalized_doc.metadata.links:
            payload["metadata"]["obsidian_links"] = [
                {
                    "target": link.target,
                    "display_text": link.display_text,
                    "is_embed": link.is_embed,
                    "section": link.section,
                    "block_id": link.block_id,
                    "resolved_path": str(link.resolved_path) if link.resolved_path else None,
                    "is_broken": link.is_broken
                }
                for link in normalized_doc.metadata.links
            ]
        
        # Save element to disk
        element_id = str(uuid.uuid4())
        storage_path = self.parsed_dir / f"{file_record.sha256}_{element_id}.json"
        storage_path.write_text(json.dumps(payload, ensure_ascii=False, default=str))
        
        return {
            "source": source_name,
            "path": str(file_record.path),
            "sha256": file_record.sha256,
            "element_path": str(storage_path),
            "size_bytes": file_record.size,
        }
