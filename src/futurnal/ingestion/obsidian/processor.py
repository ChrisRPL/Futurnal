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
from .link_graph import ObsidianLinkGraphConstructor
from .sync_metrics import SyncMetricsCollector
from .asset_processor import AssetProcessingPipeline, AssetTextExtractorConfig
from ..local.state import FileRecord

logger = logging.getLogger(__name__)


class ObsidianDocumentProcessor:
    """Processes Obsidian documents through normalization and Unstructured.io pipeline."""
    
    def __init__(
        self,
        *,
        workspace_dir: Path,
        vault_root: Optional[Path] = None,
        vault_id: Optional[str] = None,
        enable_link_graph: bool = True,
        asset_processing_config: Optional[Dict[str, Any]] = None,
        metrics_collector: Optional[SyncMetricsCollector] = None
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir = self.workspace_dir / "parsed"
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

        self.vault_root = vault_root
        self.vault_id = vault_id
        self.asset_processing_config = asset_processing_config or {}
        self.metrics_collector = metrics_collector

        # Initialize link graph constructor if enabled and vault info available
        self.link_graph_constructor = None
        if enable_link_graph and self.vault_root and self.vault_id:
            self.link_graph_constructor = ObsidianLinkGraphConstructor(
                vault_id=self.vault_id,
                vault_root=self.vault_root
            )

        # Initialize asset processing pipeline if vault info available
        self.asset_processing_pipeline = None
        if self.vault_root and self.vault_id:
            extractor_config = AssetTextExtractorConfig()
            self.asset_processing_pipeline = AssetProcessingPipeline(
                vault_root=self.vault_root,
                vault_id=self.vault_id,
                extractor_config=extractor_config,
                metrics_collector=self.metrics_collector
            )

    def update_asset_config(self, config: Dict[str, Any]) -> None:
        """Update asset processing configuration."""
        self.asset_processing_config.update(config)
        
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

            # Collect normalization metrics
            if self.metrics_collector and self.vault_id:
                self._collect_normalization_metrics(normalized_doc)

            # Step 3: Construct link graph relationships
            graph_data = None
            if self.link_graph_constructor:
                try:
                    note_nodes, link_relationships, tag_relationships, asset_relationships = self.link_graph_constructor.construct_graph(normalized_doc)
                    graph_data = {
                        "note_nodes": [node.to_dict() for node in note_nodes],
                        "link_relationships": [rel.to_dict() for rel in link_relationships],
                        "tag_relationships": [rel.to_dict() for rel in tag_relationships],
                        "asset_relationships": [rel.to_dict() for rel in asset_relationships],
                        "statistics": self.link_graph_constructor.get_statistics()
                    }
                    logger.debug(f"Constructed graph for {file_record.path}: {len(note_nodes)} notes, {len(link_relationships)} links, {len(tag_relationships)} tags, {len(asset_relationships)} assets")
                except Exception as e:
                    logger.error(f"Failed to construct graph for {file_record.path}: {e}")
                    graph_data = {"error": str(e)}

            # Step 3.5: Process assets for text extraction if enabled
            asset_processing_results = None
            if (self.asset_processing_pipeline and
                self.asset_processing_config.get("enable_asset_text_extraction", False)):
                try:
                    asset_processing_results = self.asset_processing_pipeline.process_document_assets(
                        content=content,
                        source_file_path=file_record.path,
                        include_text_extraction=True
                    )
                    logger.debug(f"Processed assets for {file_record.path}: {asset_processing_results['statistics']}")
                except Exception as e:
                    logger.error(f"Failed to process assets for {file_record.path}: {e}")
                    asset_processing_results = {"error": str(e)}

            # Step 4: Create enriched document for Unstructured.io
            enriched_content = self._create_enriched_content(normalized_doc, graph_data)
            
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
                        source_name=source_name,
                        graph_data=graph_data,
                        asset_processing_results=asset_processing_results
                    )
                    yield element_data
                    
            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            # Record parse failure metrics
            if self.metrics_collector and self.vault_id:
                self.metrics_collector.increment_counter(
                    "parse_failures",
                    labels={"vault_id": self.vault_id, "error_type": type(e).__name__}
                )
                self.metrics_collector.record_error(
                    "document_parse_error",
                    self.vault_id,
                    str(e),
                    file_path=str(file_record.path)
                )

            logger.error(f"Failed to process document {file_record.path}: {e}")
            raise

    def _collect_normalization_metrics(self, normalized_doc: NormalizedDocument) -> None:
        """Collect metrics from the normalized document."""
        if not self.metrics_collector or not self.vault_id:
            return

        labels = {"vault_id": self.vault_id}

        # Record successful parsing
        self.metrics_collector.increment_counter("parse_successes", labels=labels)

        # Collect link metrics
        total_links = len(normalized_doc.metadata.links)
        broken_links = sum(1 for link in normalized_doc.metadata.links if link.is_broken)

        if total_links > 0:
            self.metrics_collector.increment_counter("total_links", value=total_links, labels=labels)
            self.metrics_collector.increment_counter("broken_links", value=broken_links, labels=labels)
            self.metrics_collector.set_gauge(
                "broken_link_rate",
                broken_links / total_links,
                labels=labels
            )

        # Collect asset metrics
        total_assets = len(normalized_doc.metadata.assets)
        broken_assets = sum(1 for asset in normalized_doc.metadata.assets if asset.is_broken)
        processable_assets = sum(1 for asset in normalized_doc.metadata.assets if asset.is_processable)

        if total_assets > 0:
            self.metrics_collector.increment_counter("total_assets", value=total_assets, labels=labels)
            self.metrics_collector.increment_counter("broken_assets", value=broken_assets, labels=labels)
            self.metrics_collector.increment_counter("processable_assets", value=processable_assets, labels=labels)

        # Collect document complexity metrics
        self.metrics_collector.increment_counter("headings_total", value=len(normalized_doc.metadata.headings), labels=labels)
        self.metrics_collector.increment_counter("tags_total", value=len(normalized_doc.metadata.tags), labels=labels)
        self.metrics_collector.increment_counter("callouts_total", value=len(normalized_doc.metadata.callouts), labels=labels)
        self.metrics_collector.increment_counter("tables_total", value=len(normalized_doc.metadata.tables), labels=labels)
        self.metrics_collector.set_gauge("word_count", normalized_doc.metadata.word_count, labels=labels)

        # Record consent granted (if processing succeeded, consent was likely granted)
        self.metrics_collector.increment_counter("consent_granted_files", labels=labels)

        # Record the successful processing event
        self.metrics_collector.record_event(
            "document_processed",
            self.vault_id,
            word_count=normalized_doc.metadata.word_count,
            links_count=total_links,
            broken_links_count=broken_links,
            assets_count=total_assets,
            broken_assets_count=broken_assets
        )

    def _create_enriched_content(self, normalized_doc: NormalizedDocument, graph_data: Optional[Dict[str, Any]] = None) -> str:
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
                "assets_count": len(normalized_doc.metadata.assets),
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
                ],
                "assets": [
                    {
                        "target": asset.target,
                        "display_text": asset.display_text,
                        "is_embed": asset.is_embed,
                        "resolved_path": str(asset.resolved_path) if asset.resolved_path else None,
                        "is_broken": asset.is_broken,
                        "content_hash": asset.content_hash,
                        "file_size": asset.file_size,
                        "mime_type": asset.mime_type,
                        "is_image": asset.is_image,
                        "is_pdf": asset.is_pdf,
                        "is_processable": asset.is_processable
                    }
                    for asset in normalized_doc.metadata.assets
                ]
            }
        }

        # Add link graph data if available
        if graph_data:
            metadata_summary["futurnal_link_graph"] = graph_data
        
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
        source_name: str,
        graph_data: Optional[Dict[str, Any]] = None,
        asset_processing_results: Optional[Dict[str, Any]] = None
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
        futurnal_metadata = {
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
                "assets_count": len(normalized_doc.metadata.assets),
                "tags_count": len(normalized_doc.metadata.tags),
                "callouts_count": len(normalized_doc.metadata.callouts),
                "tables_count": len(normalized_doc.metadata.tables),
                "tasks_count": len(normalized_doc.metadata.task_lists),
                "footnotes_count": len(normalized_doc.metadata.footnotes),
            }
        }

        # Add link graph data if available
        if graph_data:
            futurnal_metadata["link_graph"] = graph_data

        # Add asset processing results if available
        if asset_processing_results:
            futurnal_metadata["asset_processing"] = asset_processing_results

        payload["metadata"]["futurnal"] = futurnal_metadata
        
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

        if normalized_doc.metadata.assets:
            payload["metadata"]["obsidian_assets"] = [
                {
                    "target": asset.target,
                    "display_text": asset.display_text,
                    "is_embed": asset.is_embed,
                    "resolved_path": str(asset.resolved_path) if asset.resolved_path else None,
                    "is_broken": asset.is_broken,
                    "content_hash": asset.content_hash,
                    "file_size": asset.file_size,
                    "mime_type": asset.mime_type,
                    "is_image": asset.is_image,
                    "is_pdf": asset.is_pdf,
                    "is_processable": asset.is_processable
                }
                for asset in normalized_doc.metadata.assets
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
