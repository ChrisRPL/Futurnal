Summary: Implement centralized bridge to Unstructured.io with metadata preservation and format-specific strategies.

# 07 · Unstructured.io Integration Bridge

## Purpose
Design and implement a centralized bridge to Unstructured.io that handles format-specific partition strategies, metadata preservation, and element-to-chunk conversion. The bridge abstracts Unstructured.io complexity while maintaining flexibility for format-specific optimizations.

## Scope
- Centralized Unstructured.io invocation interface
- Format-specific partition strategy selection
- Metadata preservation during processing
- Element-based document representation
- Strategy optimization (fast vs hi-res)
- Error handling for parsing failures
- Offline operation with cached models

## Requirements Alignment
- **System Requirements**: "Normalize 60+ document formats using Unstructured.io"
- **Feature Requirement**: "Format-specific adapters leveraging Unstructured.io"
- **Non-Functional**: "Offline operation with caching of models/resources"

## Component Design

### UnstructuredBridge

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import DocumentFormat

logger = logging.getLogger(__name__)

# Type alias for Unstructured elements
UnstructuredElement = Dict[str, Any]


class PartitionStrategy(str, Enum):
    """Unstructured.io partition strategies."""
    FAST = "fast"  # Faster, lower accuracy
    HI_RES = "hi_res"  # Slower, higher accuracy
    OCR_ONLY = "ocr_only"  # For scanned documents


class UnstructuredBridge:
    """Bridge to Unstructured.io document processing library.

    Provides centralized interface to Unstructured.io with format-specific
    optimizations and metadata preservation.

    Example:
        >>> bridge = UnstructuredBridge()
        >>> elements = await bridge.process_document(
        ...     content=document_content,
        ...     format=DocumentFormat.PDF,
        ...     file_path=Path("document.pdf")
        ... )
    """

    def __init__(self):
        self.documents_processed = 0
        self.processing_errors = 0
        self._ensure_unstructured_available()

    def _ensure_unstructured_available(self):
        """Verify Unstructured.io is available."""
        try:
            from unstructured.partition.auto import partition
            self._partition = partition
            logger.info("Unstructured.io library loaded successfully")
        except ImportError as e:
            logger.error(f"Unstructured.io not available: {e}")
            raise RuntimeError(
                "Unstructured.io library not installed. "
                "Install with: pip install unstructured"
            ) from e

    async def process_document(
        self,
        *,
        content: Optional[str] = None,
        file_path: Optional[Path] = None,
        format: DocumentFormat,
        strategy: Optional[PartitionStrategy] = None,
    ) -> List[UnstructuredElement]:
        """Process document using Unstructured.io.

        Args:
            content: Document content (for text-based formats)
            file_path: Path to file (for binary formats)
            format: Document format
            strategy: Partition strategy (auto-selected if None)

        Returns:
            List of Unstructured.io elements

        Raises:
            UnstructuredProcessingError: If processing fails
        """
        try:
            # Auto-select strategy if not specified
            if strategy is None:
                strategy = self._select_strategy(format)

            # Choose processing method based on format
            if format in [DocumentFormat.PDF, DocumentFormat.DOCX, DocumentFormat.PPTX]:
                # Binary formats require file path
                elements = await self._process_from_file(file_path, strategy)
            else:
                # Text formats can use content directly
                elements = await self._process_from_text(content or "", format, strategy)

            # Convert to dictionaries for downstream processing
            element_dicts = [self._element_to_dict(el) for el in elements]

            self.documents_processed += 1
            logger.debug(f"Processed {len(element_dicts)} elements from {format.value}")

            return element_dicts

        except Exception as e:
            self.processing_errors += 1
            logger.error(f"Unstructured processing failed: {e}")
            raise UnstructuredProcessingError(
                f"Failed to process {format.value} document: {str(e)}"
            ) from e

    async def _process_from_file(
        self,
        file_path: Path,
        strategy: PartitionStrategy
    ) -> List[Any]:
        """Process document from file path."""
        from unstructured.partition.auto import partition

        # Format-specific parameters
        kwargs = {
            "filename": str(file_path),
            "strategy": strategy.value,
            "include_metadata": True,
        }

        # Add format-specific options
        if file_path.suffix.lower() == '.pdf':
            kwargs.update({
                "infer_table_structure": True,
                "extract_images_in_pdf": False,  # Privacy: don't extract embedded images
            })

        elements = partition(**kwargs)
        return elements

    async def _process_from_text(
        self,
        content: str,
        format: DocumentFormat,
        strategy: PartitionStrategy
    ) -> List[Any]:
        """Process document from text content."""
        # Choose appropriate partition function
        if format == DocumentFormat.MARKDOWN:
            from unstructured.partition.md import partition_md
            elements = partition_md(text=content)
        elif format == DocumentFormat.HTML:
            from unstructured.partition.html import partition_html
            elements = partition_html(text=content)
        elif format == DocumentFormat.EMAIL:
            from unstructured.partition.email import partition_email
            elements = partition_email(text=content)
        else:
            from unstructured.partition.text import partition_text
            elements = partition_text(text=content)

        return elements

    def _select_strategy(self, format: DocumentFormat) -> PartitionStrategy:
        """Auto-select optimal partition strategy for format."""
        # Binary formats benefit from hi-res
        if format in [DocumentFormat.PDF, DocumentFormat.DOCX]:
            return PartitionStrategy.HI_RES

        # Text formats use fast strategy
        return PartitionStrategy.FAST

    def _element_to_dict(self, element: Any) -> UnstructuredElement:
        """Convert Unstructured element to dictionary."""
        # Elements have to_dict() method
        element_dict = element.to_dict()

        # Preserve important metadata
        metadata = element_dict.get("metadata", {})

        return {
            "type": element_dict.get("type", "Unknown"),
            "text": element_dict.get("text", ""),
            "metadata": metadata,
            "element_id": element_dict.get("element_id"),
        }

    def get_metrics(self) -> dict:
        """Get processing metrics."""
        return {
            "documents_processed": self.documents_processed,
            "processing_errors": self.processing_errors,
            "success_rate": (
                self.documents_processed / (self.documents_processed + self.processing_errors)
                if (self.documents_processed + self.processing_errors) > 0
                else 0
            )
        }


class UnstructuredProcessingError(Exception):
    """Raised when Unstructured.io processing fails."""
    pass
```

### Format-Specific Optimizations

```python
# Configuration for format-specific processing
UNSTRUCTURED_FORMAT_CONFIG = {
    DocumentFormat.PDF: {
        "strategy": "hi_res",
        "infer_table_structure": True,
        "extract_images_in_pdf": False,  # Privacy
        "ocr_languages": ["eng"],  # Can be configured
    },
    DocumentFormat.DOCX: {
        "strategy": "hi_res",
        "infer_table_structure": True,
    },
    DocumentFormat.MARKDOWN: {
        "strategy": "fast",
    },
    DocumentFormat.HTML: {
        "strategy": "fast",
        "include_metadata": True,
    },
    DocumentFormat.EMAIL: {
        "strategy": "fast",
        "process_attachments": False,  # Handled separately
    }
}
```

## Acceptance Criteria

- ✅ Unstructured.io integration working for all supported formats
- ✅ Format-specific partition strategies applied
- ✅ Metadata preserved during processing
- ✅ Element-to-dict conversion working
- ✅ Error handling for parsing failures
- ✅ Offline operation (no network calls)
- ✅ Metrics tracking for telemetry

## Test Plan

### Unit Tests
- Element-to-dict conversion
- Strategy selection per format
- Error handling for invalid inputs

### Integration Tests
- PDF processing with tables
- Markdown processing with frontmatter
- HTML processing with complex structure
- Email processing with headers

### Performance Tests
- Processing throughput per format
- Memory usage during large file processing

## Implementation Notes

### Chunking Integration

Unstructured.io provides built-in chunking:
```python
from unstructured.chunking.title import chunk_by_title

elements = partition(filename="document.pdf")
chunks = chunk_by_title(elements)
```

This can be integrated with our ChunkingEngine for consistency.

### Privacy Considerations

- Don't extract embedded images from PDFs (privacy risk)
- Don't process email attachments inline (handle separately)
- Metadata-only logging (no content exposure)

## Open Questions

- Should we cache Unstructured.io models?
- How to handle OCR for scanned documents?
- Should we support custom partition strategies per connector?
- How to version Unstructured.io dependency?

## Dependencies

- Unstructured.io library (v0.18.15)
- DocumentFormat schema (Task 01)
- Format adapters (Task 03)


