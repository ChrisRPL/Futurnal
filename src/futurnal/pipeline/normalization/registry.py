"""Format adapter registry for pluggable normalization handlers.

This module provides the FormatAdapter protocol and registry system that enables
format-specific normalization logic while maintaining a consistent interface.
Follows the modOpt pattern for extensibility.

Key Features:
- FormatAdapter protocol for consistent interface
- Registry with format → adapter mapping
- Dynamic adapter registration
- Fallback adapter support
- Validation to prevent duplicate registrations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from ..models import DocumentFormat, NormalizedDocument

logger = logging.getLogger(__name__)


class FormatAdapter(Protocol):
    """Protocol for format-specific normalization adapters.

    Each adapter handles format-specific parsing, cleaning, and
    metadata extraction before general-purpose processing.

    Attributes:
        name: Human-readable adapter name
        supported_formats: List of DocumentFormat values this adapter handles
        requires_unstructured_processing: Whether adapter output needs Unstructured.io
    """

    name: str
    supported_formats: List[DocumentFormat]
    requires_unstructured_processing: bool

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize document using format-specific logic.

        Args:
            file_path: Path to source file
            source_id: Connector-specific identifier
            source_type: Source type (e.g., "local_files", "obsidian_vault")
            source_metadata: Additional metadata from connector

        Returns:
            Preliminary NormalizedDocument (may require further processing)

        Raises:
            AdapterError: If format-specific normalization fails
        """
        ...

    async def validate(self, file_path: Path) -> bool:
        """Validate that file is processable by this adapter.

        Args:
            file_path: Path to validate

        Returns:
            True if adapter can process this file
        """
        ...


class AdapterError(Exception):
    """Raised when format adapter processing fails."""

    pass


class AdapterNotFoundError(Exception):
    """Raised when no adapter is registered for requested format."""

    pass


class FormatAdapterRegistry:
    """Registry for format-specific normalization adapters.

    Manages adapter registration, selection, and lifecycle. Supports
    dynamic registration for extensibility.

    Example:
        >>> registry = FormatAdapterRegistry()
        >>> registry.register(MarkdownAdapter())
        >>> registry.register(PDFAdapter())
        >>> adapter = registry.get_adapter(DocumentFormat.MARKDOWN)
        >>> doc = await adapter.normalize(...)
    """

    def __init__(self):
        self._adapters: Dict[DocumentFormat, FormatAdapter] = {}
        self._fallback_adapter: Optional[FormatAdapter] = None

    def register(self, adapter: FormatAdapter) -> None:
        """Register a format adapter.

        Args:
            adapter: Adapter instance to register

        Raises:
            ValueError: If adapter for format already registered
        """
        for fmt in adapter.supported_formats:
            if fmt in self._adapters:
                existing = self._adapters[fmt]
                raise ValueError(
                    f"Adapter for {fmt.value} already registered: {existing.name}"
                )
            self._adapters[fmt] = adapter
            logger.debug(f"Registered {adapter.name} for format {fmt.value}")

    def register_fallback(self, adapter: FormatAdapter) -> None:
        """Register fallback adapter for unknown formats.

        Args:
            adapter: Adapter to use as fallback
        """
        self._fallback_adapter = adapter
        logger.debug(f"Registered fallback adapter: {adapter.name}")

    def get_adapter(self, format: DocumentFormat) -> FormatAdapter:
        """Get adapter for specified format.

        Args:
            format: Document format

        Returns:
            Registered adapter for format

        Raises:
            AdapterNotFoundError: If no adapter registered for format
        """
        if format in self._adapters:
            return self._adapters[format]

        if self._fallback_adapter:
            logger.warning(
                f"No specific adapter for {format.value}, using fallback: "
                f"{self._fallback_adapter.name}"
            )
            return self._fallback_adapter

        raise AdapterNotFoundError(
            f"No adapter registered for format: {format.value}"
        )

    def has_adapter(self, format: DocumentFormat) -> bool:
        """Check if adapter is registered for format.

        Args:
            format: Document format

        Returns:
            True if adapter is registered (or fallback available)
        """
        return format in self._adapters or self._fallback_adapter is not None

    def list_supported_formats(self) -> List[DocumentFormat]:
        """Get list of all supported formats.

        Returns:
            List of DocumentFormat values with registered adapters
        """
        return list(self._adapters.keys())

    def unregister(self, format: DocumentFormat) -> None:
        """Unregister adapter for format.

        Args:
            format: Document format to unregister

        Raises:
            AdapterNotFoundError: If no adapter registered for format
        """
        if format not in self._adapters:
            raise AdapterNotFoundError(
                f"No adapter registered for format: {format.value}"
            )

        adapter = self._adapters.pop(format)
        logger.debug(f"Unregistered {adapter.name} for format {format.value}")

    def clear(self) -> None:
        """Clear all registered adapters (for testing)."""
        self._adapters.clear()
        self._fallback_adapter = None
        logger.debug("Cleared all registered adapters")

    def get_adapter_info(self, format: DocumentFormat) -> Dict[str, object]:
        """Get information about adapter for format.

        Args:
            format: Document format

        Returns:
            Dictionary with adapter information

        Raises:
            AdapterNotFoundError: If no adapter registered for format
        """
        adapter = self.get_adapter(format)

        return {
            "name": adapter.name,
            "supported_formats": [fmt.value for fmt in adapter.supported_formats],
            "requires_unstructured_processing": adapter.requires_unstructured_processing,
        }

    def register_default_adapters(self) -> None:
        """Register default adapters for common formats.

        This method imports and registers all built-in adapters:
        - Text-based: Markdown, Email, HTML, Code
        - Binary: PDF (also handles DOCX, PPTX)
        - Multimodal: Audio, Image, ScannedPDF (Module 08)
        - Fallback: Generic

        Called by factory function to set up standard adapter registry.
        """
        from .adapters.code import CodeAdapter
        from .adapters.email import EmailAdapter
        from .adapters.generic import GenericAdapter
        from .adapters.html import HTMLAdapter
        from .adapters.markdown import MarkdownAdapter
        from .adapters.pdf import PDFAdapter

        # Module 08: Multimodal Integration - Phase 4 (Integration & Polish)
        from .adapters.audio import AudioAdapter
        from .adapters.image import ImageAdapter
        from .adapters.scanned_pdf import ScannedPDFAdapter

        # Register text-based format adapters
        self.register(MarkdownAdapter())
        self.register(EmailAdapter())
        self.register(HTMLAdapter())
        self.register(CodeAdapter())

        # Register binary format adapters
        self.register(PDFAdapter())

        # Register multimodal adapters (Module 08: Multimodal Integration)
        # - AudioAdapter: Whisper V3 transcription for audio files
        # - ImageAdapter: DeepSeek-OCR for image text extraction
        # - ScannedPDFAdapter: PDF→Image→OCR pipeline for scanned documents
        self.register(AudioAdapter())
        self.register(ImageAdapter())
        self.register(ScannedPDFAdapter())

        # Register fallback adapter for unknown formats
        self.register_fallback(GenericAdapter())

        logger.info(
            f"Registered {len(self._adapters)} default adapters "
            f"(including 3 multimodal) with GenericAdapter as fallback"
        )
