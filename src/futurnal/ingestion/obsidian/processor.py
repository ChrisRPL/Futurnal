"""Obsidian Document Processor - Bridge between MarkdownNormalizer and Unstructured.io.

This module processes Obsidian markdown documents through the normalization pipeline
and prepares them for Unstructured.io processing while preserving all metadata.

Research Foundation:
- GFM-RAG: Document processing with metadata preservation
- ProPerSim: Personal knowledge preservation patterns
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from .normalizer import (
    MarkdownNormalizer,
    NormalizedDocument,
    normalize_obsidian_document,
)
from .link_graph import ObsidianLinkGraphConstructor
from .asset_processor import (
    AssetProcessingPipeline,
    AssetTextExtractorConfig,
)

if TYPE_CHECKING:
    from futurnal.ingestion.local.state import FileRecord
    from .sync_metrics import SyncMetricsCollector

logger = logging.getLogger(__name__)

# Try to import partition with graceful fallback
try:
    from unstructured.partition.md import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    logger.warning("Unstructured.io not available - document processing will use fallback")
    UNSTRUCTURED_AVAILABLE = False
    partition = None


@dataclass
class ProcessedElement:
    """Represents a processed document element ready for PKG storage."""

    source: str
    path: str
    sha256: str
    element_path: str
    vault_id: str
    element_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "path": self.path,
            "sha256": self.sha256,
            "element_path": self.element_path,
            "vault_id": self.vault_id,
            "element_type": self.element_type,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class ObsidianDocumentProcessor:
    """Processes Obsidian documents through normalization and Unstructured.io.

    This processor bridges the MarkdownNormalizer output to Unstructured.io,
    enriching content with structured metadata as HTML comments for preservation.

    Example:
        >>> processor = ObsidianDocumentProcessor(
        ...     workspace_dir=Path("/workspace"),
        ...     vault_root=Path("/vault"),
        ...     vault_id="my-vault"
        ... )
        >>> for element in processor.process_document(file_record, "source"):
        ...     print(element["element_path"])
    """

    def __init__(
        self,
        workspace_dir: Path,
        vault_root: Path,
        vault_id: str,
        *,
        enable_link_graph: bool = True,
        asset_processing_config: Optional[Dict[str, Any]] = None,
        metrics_collector: Optional["SyncMetricsCollector"] = None,
    ):
        """Initialize the document processor.

        Args:
            workspace_dir: Directory for workspace files and element storage
            vault_root: Root directory of the Obsidian vault
            vault_id: Unique identifier for this vault
            enable_link_graph: Whether to enable link graph construction
            asset_processing_config: Configuration for asset text extraction
            metrics_collector: Optional metrics collector for tracking
        """
        self.workspace_dir = Path(workspace_dir)
        self.vault_root = Path(vault_root)
        self.vault_id = vault_id
        self.enable_link_graph = enable_link_graph
        self.metrics_collector = metrics_collector

        # Ensure workspace directories exist
        self.elements_dir = self.workspace_dir / "elements"
        self.elements_dir.mkdir(parents=True, exist_ok=True)

        # Initialize normalizer
        self.normalizer = MarkdownNormalizer(
            vault_root=self.vault_root,
            vault_id=self.vault_id,
        )

        # Initialize link graph constructor if enabled
        self.link_graph_constructor: Optional[ObsidianLinkGraphConstructor] = None
        if enable_link_graph:
            try:
                self.link_graph_constructor = ObsidianLinkGraphConstructor(
                    vault_root=self.vault_root,
                    vault_id=self.vault_id,
                )
            except Exception as e:
                logger.warning(f"Could not initialize link graph constructor: {e}")

        # Initialize asset processing pipeline
        self.asset_processing_pipeline: Optional[AssetProcessingPipeline] = None
        if asset_processing_config and asset_processing_config.get("enable_asset_text_extraction"):
            try:
                extractor_config = AssetTextExtractorConfig(
                    enable_image_ocr=asset_processing_config.get("enable_image_ocr", True),
                    enable_pdf_extraction=asset_processing_config.get("enable_pdf_extraction", True),
                )
                self.asset_processing_pipeline = AssetProcessingPipeline(
                    extractor_config=extractor_config,
                    metrics_collector=metrics_collector,
                    vault_id=vault_id,
                )
            except Exception as e:
                logger.warning(f"Could not initialize asset processing pipeline: {e}")

        # Statistics
        self.documents_processed = 0
        self.elements_generated = 0
        self.errors_encountered = 0

    def process_document(
        self,
        file_record: "FileRecord",
        source: str,
    ) -> Generator[Dict[str, Any], None, None]:
        """Process a single document and yield elements.

        Args:
            file_record: File record with path, sha256, etc.
            source: Source identifier for the document

        Yields:
            Dictionary containing element information with keys:
            - source: Source identifier
            - path: Original file path
            - sha256: Content hash
            - element_path: Path to the element JSON file
        """
        file_path = Path(file_record.path)

        try:
            # Read document content
            content = file_path.read_text(encoding='utf-8')

            # Normalize the document
            normalized = self.normalizer.normalize(content, file_path)

            # Process through Unstructured.io (or fallback)
            elements = self._partition_document(file_path, normalized)

            # Enrich and store each element
            for i, element in enumerate(elements):
                enriched = self._enrich_element(
                    element=element,
                    normalized=normalized,
                    file_record=file_record,
                    element_index=i,
                )

                # Store element data
                element_path = self._store_element(enriched, file_record, i)

                # Yield element reference
                yield {
                    "source": source,
                    "path": str(file_path),
                    "sha256": file_record.sha256,
                    "element_path": str(element_path),
                    "element_type": enriched.get("type", "unknown"),
                    "text": enriched.get("text", ""),
                }

                self.elements_generated += 1

            # Process link graph if enabled
            if self.link_graph_constructor and self.enable_link_graph:
                try:
                    self.link_graph_constructor.process_document(normalized, file_path)
                except Exception as e:
                    logger.warning(f"Link graph processing failed for {file_path}: {e}")

            # Process assets if enabled
            if self.asset_processing_pipeline and normalized.metadata.assets:
                try:
                    extractions = self.asset_processing_pipeline.process_assets(
                        normalized.metadata.assets
                    )
                    # Record success metrics
                    if self.metrics_collector:
                        for extraction in extractions:
                            if extraction.success:
                                self.metrics_collector.increment_counter(
                                    "asset_processing_success",
                                    labels={"vault_id": self.vault_id}
                                )
                            else:
                                self.metrics_collector.increment_counter(
                                    "asset_processing_failure",
                                    labels={"vault_id": self.vault_id}
                                )
                except Exception as e:
                    logger.warning(f"Asset processing failed for {file_path}: {e}")

            self.documents_processed += 1

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            self.errors_encountered += 1
            raise

    def _partition_document(
        self,
        file_path: Path,
        normalized: NormalizedDocument,
    ) -> List[Dict[str, Any]]:
        """Partition document using Unstructured.io or fallback.

        Args:
            file_path: Path to the document
            normalized: Normalized document data

        Returns:
            List of element dictionaries
        """
        if UNSTRUCTURED_AVAILABLE and partition is not None:
            # Create temp file with enriched content for Unstructured.io
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.md',
                delete=False,
                encoding='utf-8'
            ) as tmp:
                # Write enriched content
                enriched_content = self._create_enriched_content(normalized)
                tmp.write(enriched_content)
                tmp_path = tmp.name

            try:
                # Process with Unstructured.io
                elements = partition(
                    filename=tmp_path,
                    strategy="fast",
                    include_metadata=True,
                )

                # Convert to dicts
                return [
                    el.to_dict() if hasattr(el, 'to_dict') else {"text": str(el), "type": "unknown"}
                    for el in elements
                ]
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
        else:
            # Fallback: create elements from normalized document
            return self._fallback_partition(normalized)

    def _fallback_partition(self, normalized: NormalizedDocument) -> List[Dict[str, Any]]:
        """Create elements without Unstructured.io.

        Args:
            normalized: Normalized document data

        Returns:
            List of element dictionaries
        """
        elements = []

        # Create elements from blocks
        for block in normalized.metadata.blocks:
            element = {
                "text": block.content,
                "type": self._map_block_type(block.type),
                "metadata": {
                    "block_type": block.type,
                    "level": block.level,
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                },
            }
            elements.append(element)

        # If no blocks, create single element from content
        if not elements:
            elements.append({
                "text": normalized.content,
                "type": "NarrativeText",
                "metadata": {},
            })

        return elements

    def _map_block_type(self, block_type: str) -> str:
        """Map internal block type to Unstructured element type."""
        mapping = {
            "heading": "Title",
            "paragraph": "NarrativeText",
            "code": "CodeSnippet",
            "list_item": "ListItem",
            "blockquote": "Quote",
        }
        return mapping.get(block_type, "NarrativeText")

    def _create_enriched_content(self, normalized: NormalizedDocument) -> str:
        """Create enriched content with metadata comments for Unstructured.io."""
        # Add metadata as HTML comment at the start
        metadata_json = json.dumps({
            "vault_id": self.vault_id,
            "content_checksum": normalized.provenance.content_checksum,
            "normalizer_version": normalized.provenance.normalizer_version,
        })

        header = f"<!-- futurnal_metadata: {metadata_json} -->\n\n"
        return header + normalized.content

    def _enrich_element(
        self,
        element: Dict[str, Any],
        normalized: NormalizedDocument,
        file_record: "FileRecord",
        element_index: int,
    ) -> Dict[str, Any]:
        """Enrich element with Futurnal metadata.

        Args:
            element: Raw element dictionary
            normalized: Normalized document data
            file_record: Original file record
            element_index: Index of this element

        Returns:
            Enriched element dictionary
        """
        # Get or create metadata dict
        metadata = element.get("metadata", {})

        # Add Futurnal-specific metadata
        metadata["futurnal"] = {
            "normalizer_version": normalized.provenance.normalizer_version,
            "content_checksum": normalized.provenance.content_checksum,
            "vault_id": self.vault_id,
            "element_index": element_index,
            "processed_at": datetime.utcnow().isoformat(),
        }

        # Add frontmatter
        metadata["frontmatter"] = normalized.metadata.frontmatter

        # Add Obsidian tags
        metadata["obsidian_tags"] = [
            {
                "name": tag.name,
                "is_nested": tag.is_nested,
            }
            for tag in normalized.metadata.tags
        ]

        # Add Obsidian links
        metadata["obsidian_links"] = [
            {
                "target": link.target,
                "display_text": link.display_text,
                "is_embed": link.is_embed,
                "section": link.section,
                "block_id": link.block_id,
                "is_broken": link.is_broken,
            }
            for link in normalized.metadata.links
        ]

        # Add callouts
        metadata["obsidian_callouts"] = [
            {
                "type": callout.type.value,
                "title": callout.title,
                "content": callout.content,
            }
            for callout in normalized.metadata.callouts
        ]

        # Add tasks
        metadata["obsidian_tasks"] = [
            {
                "text": task.text,
                "checked": task.checked,
                "level": task.level,
            }
            for task in normalized.metadata.task_lists
        ]

        # Add document stats
        metadata["document_stats"] = {
            "word_count": normalized.metadata.word_count,
            "reading_time_minutes": normalized.metadata.reading_time_minutes,
            "link_count": len(normalized.metadata.links),
            "tag_count": len(normalized.metadata.tags),
            "heading_count": len(normalized.metadata.headings),
        }

        element["metadata"] = metadata
        return element

    def _store_element(
        self,
        element: Dict[str, Any],
        file_record: "FileRecord",
        element_index: int,
    ) -> Path:
        """Store element data to JSON file.

        Args:
            element: Enriched element dictionary
            file_record: Original file record
            element_index: Index of this element

        Returns:
            Path to stored element file
        """
        # Create unique filename
        element_id = f"{file_record.sha256}_{element_index}"
        element_path = self.elements_dir / f"{element_id}.json"

        # Write element data
        with open(element_path, 'w', encoding='utf-8') as f:
            json.dump(element, f, indent=2, default=str)

        return element_path

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "documents_processed": self.documents_processed,
            "elements_generated": self.elements_generated,
            "errors_encountered": self.errors_encountered,
        }


class AssetProcessingPipeline:
    """Pipeline for processing assets with text extraction."""

    def __init__(
        self,
        extractor_config: Optional[AssetTextExtractorConfig] = None,
        metrics_collector: Optional["SyncMetricsCollector"] = None,
        vault_id: Optional[str] = None,
    ):
        """Initialize the asset processing pipeline."""
        from .asset_processor import AssetTextExtractor

        self.text_extractor = AssetTextExtractor(
            config=extractor_config,
            metrics_collector=metrics_collector,
            vault_id=vault_id,
        )
        self.metrics_collector = metrics_collector
        self.vault_id = vault_id

    def process_assets(self, assets: List[Any]) -> List[Any]:
        """Process a list of assets and extract text.

        Args:
            assets: List of ObsidianAsset objects

        Returns:
            List of AssetTextExtraction results
        """
        results = []
        for asset in assets:
            extraction = self.text_extractor.extract_text(asset)
            if extraction:
                results.append(extraction)
        return results
