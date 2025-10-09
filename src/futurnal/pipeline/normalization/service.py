"""Central normalization service orchestrating the document normalization pipeline.

This module provides the NormalizationService that coordinates format detection,
parsing, chunking, and enrichment to produce standardized normalized documents.
Designed for extensibility, testability, and privacy-first operation.

Key Features:
- Central service interface for connector integration
- Format detection and adapter routing
- Pipeline orchestration (parse → chunk → enrich → deliver)
- Error handling with quarantine integration
- Idempotency via content hashing
- Privacy-aware audit logging
- Performance monitoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...orchestrator.quarantine import QuarantineStore
from ...privacy.audit import AuditEvent, AuditLogger
from ..models import DocumentFormat, NormalizedDocument
from ..stubs import NormalizationSink
from .registry import FormatAdapterRegistry
from .chunking import ChunkingConfig, ChunkingEngine
from .enrichment import MetadataEnrichmentPipeline
from .unstructured_bridge import UnstructuredBridge
from .streaming import StreamingProcessor
from .error_handler import NormalizationErrorHandler

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for normalization pipeline.

    Attributes:
        enable_chunking: Whether to chunk documents
        default_chunk_strategy: Default chunking strategy (by_title, by_page, basic)
        max_chunk_size_chars: Maximum characters per chunk
        chunk_overlap_chars: Character overlap between chunks
        enable_language_detection: Run language detection
        enable_content_classification: Run content type classification
        compute_content_hash: Compute SHA-256 content hash
        enable_streaming: Enable streaming for large files
        streaming_threshold_mb: File size threshold for streaming
        max_memory_mb: Maximum memory usage limit
        quarantine_on_failure: Quarantine failed documents
        retry_on_transient_errors: Retry transient failures
        max_retries: Maximum retry attempts
        audit_logging_enabled: Enable audit logging
        redact_paths_in_logs: Redact file paths in logs
    """

    # Chunking configuration
    enable_chunking: bool = True
    default_chunk_strategy: str = "by_title"
    max_chunk_size_chars: int = 4000
    chunk_overlap_chars: int = 200

    # Metadata enrichment
    enable_language_detection: bool = True
    enable_content_classification: bool = True
    compute_content_hash: bool = True

    # Performance
    enable_streaming: bool = True  # For large files
    streaming_threshold_mb: float = 100.0
    max_memory_mb: float = 2048.0

    # Error handling
    quarantine_on_failure: bool = True
    retry_on_transient_errors: bool = True
    max_retries: int = 3

    # Privacy
    audit_logging_enabled: bool = True
    redact_paths_in_logs: bool = True


class NormalizationError(Exception):
    """Raised when document normalization fails."""

    pass


class NormalizationService:
    """Central orchestrator for document normalization pipeline.

    Coordinates format detection, parsing, chunking, enrichment, and delivery
    to downstream sinks. Designed for extensibility via adapter pattern and
    privacy-first operation.

    Example:
        >>> config = NormalizationConfig()
        >>> service = NormalizationService(
        ...     config=config,
        ...     adapter_registry=registry,
        ...     chunking_engine=engine,
        ...     enrichment_pipeline=pipeline,
        ...     unstructured_bridge=bridge,
        ...     sink=normalization_sink
        ... )
        >>> normalized = await service.normalize_document(
        ...     file_path="document.pdf",
        ...     source_id="local_file_123",
        ...     source_type="local_files"
        ... )
    """

    def __init__(
        self,
        *,
        config: NormalizationConfig,
        adapter_registry: FormatAdapterRegistry,
        chunking_engine: ChunkingEngine,
        enrichment_pipeline: MetadataEnrichmentPipeline,
        unstructured_bridge: UnstructuredBridge,
        sink: Optional[NormalizationSink] = None,
        audit_logger: Optional[AuditLogger] = None,
        quarantine_manager: Optional[QuarantineStore] = None,
        streaming_processor: Optional[StreamingProcessor] = None,
        error_handler: Optional[NormalizationErrorHandler] = None,
    ):
        self.config = config
        self.adapter_registry = adapter_registry
        self.chunking_engine = chunking_engine
        self.enrichment_pipeline = enrichment_pipeline
        self.unstructured_bridge = unstructured_bridge
        self.sink = sink
        self.audit_logger = audit_logger
        self.quarantine_manager = quarantine_manager
        self.streaming_processor = streaming_processor
        self.error_handler = error_handler

        # Metrics
        self.documents_processed = 0
        self.documents_failed = 0
        self.total_processing_time_ms = 0.0

    async def normalize_document(
        self,
        *,
        file_path: Path | str,
        source_id: str,
        source_type: str,
        source_metadata: Optional[dict] = None,
    ) -> NormalizedDocument:
        """Normalize a single document through the complete pipeline.

        Args:
            file_path: Path to source document
            source_id: Connector-specific identifier
            source_type: Source type (e.g., "obsidian_vault", "local_files")
            source_metadata: Additional metadata from connector

        Returns:
            NormalizedDocument ready for PKG ingestion

        Raises:
            NormalizationError: If normalization fails unrecoverably
        """
        start_time = datetime.now(timezone.utc)
        file_path = Path(file_path)

        try:
            # Log start event
            self._log_normalization_start(file_path, source_id, source_type)

            # Step 1: Detect format and select adapter
            format_type = await self._detect_format(file_path, source_metadata)
            adapter = self.adapter_registry.get_adapter(format_type)

            logger.debug(
                f"Processing {file_path.name} as {format_type.value} "
                f"via {adapter.name}"
            )

            # Check if streaming should be used for large files
            if (
                self.config.enable_streaming
                and self.streaming_processor
                and await self.streaming_processor.should_stream(file_path)
            ):
                # Use streaming path for large files
                return await self._process_with_streaming(
                    file_path=file_path,
                    source_id=source_id,
                    source_type=source_type,
                    format_type=format_type,
                    source_metadata=source_metadata,
                )

            # Step 2: Format-specific normalization
            preliminary_normalized = await adapter.normalize(
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                source_metadata=source_metadata or {},
            )

            # Step 3: Unstructured.io processing (if applicable)
            if adapter.requires_unstructured_processing:
                elements = await self.unstructured_bridge.process_document(
                    file_path=file_path, format=format_type
                )
                preliminary_normalized.elements = elements

                # Extract text from elements for adapters that return empty content
                if not preliminary_normalized.content:
                    preliminary_normalized.content = self._extract_text_from_elements(
                        elements
                    )

            # Step 4: Chunking (if enabled and applicable)
            if self.config.enable_chunking and preliminary_normalized.content:
                chunking_config = self._get_chunking_config(format_type, source_type)
                chunks = await self.chunking_engine.chunk_document(
                    content=preliminary_normalized.content,
                    config=chunking_config,
                    elements=preliminary_normalized.elements,
                    parent_document_id=preliminary_normalized.document_id,
                )
                preliminary_normalized.chunks = chunks
                preliminary_normalized.metadata.is_chunked = True
                preliminary_normalized.metadata.chunk_strategy = chunking_config.strategy
                preliminary_normalized.metadata.total_chunks = len(chunks)

            # Step 5: Metadata enrichment
            enriched_metadata = await self.enrichment_pipeline.enrich(
                document=preliminary_normalized,
                enable_language_detection=self.config.enable_language_detection,
                enable_classification=self.config.enable_content_classification,
                compute_hash=self.config.compute_content_hash,
                file_path=file_path,  # Pass file_path for temporal metadata extraction
            )
            preliminary_normalized.metadata = enriched_metadata

            # Step 6: Finalize normalized document
            normalized_doc = preliminary_normalized
            normalized_doc.document_id = normalized_doc.sha256
            normalized_doc.normalized_at = datetime.now(timezone.utc)

            # Step 7: Deliver to sink
            if self.sink:
                sink_payload = normalized_doc.to_sink_format()
                self.sink.handle(sink_payload)

            # Compute processing duration
            duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            normalized_doc.metadata.processing_duration_ms = duration_ms

            # Log success
            self._log_normalization_success(normalized_doc, duration_ms)

            # Update metrics
            self.documents_processed += 1
            self.total_processing_time_ms += duration_ms

            return normalized_doc

        except Exception as e:
            # Log failure
            self._log_normalization_failure(file_path, source_id, e)

            # Quarantine if enabled
            # Note: error_handler internally requires quarantine_manager,
            # so checking for either is sufficient
            if self.config.quarantine_on_failure and (
                self.error_handler or self.quarantine_manager
            ):
                await self._quarantine_document(file_path, source_id, source_type, e)

            # Update metrics
            self.documents_failed += 1

            raise NormalizationError(
                f"Failed to normalize {file_path}: {str(e)}"
            ) from e

    async def _detect_format(
        self, file_path: Path, source_metadata: Optional[dict]
    ) -> DocumentFormat:
        """Detect document format using extension and metadata.

        Args:
            file_path: Path to file
            source_metadata: Additional metadata hints

        Returns:
            Detected DocumentFormat
        """
        # Priority 1: Explicit format in metadata
        if source_metadata and "format" in source_metadata:
            return DocumentFormat(source_metadata["format"])

        # Priority 2: File extension mapping
        extension = file_path.suffix.lower()
        format_map = {
            ".md": DocumentFormat.MARKDOWN,
            ".markdown": DocumentFormat.MARKDOWN,
            ".pdf": DocumentFormat.PDF,
            ".html": DocumentFormat.HTML,
            ".htm": DocumentFormat.HTML,
            ".eml": DocumentFormat.EMAIL,
            ".msg": DocumentFormat.EMAIL,
            ".docx": DocumentFormat.DOCX,
            ".pptx": DocumentFormat.PPTX,
            ".xlsx": DocumentFormat.XLSX,
            ".csv": DocumentFormat.CSV,
            ".json": DocumentFormat.JSON,
            ".yaml": DocumentFormat.YAML,
            ".yml": DocumentFormat.YAML,
            ".txt": DocumentFormat.TEXT,
            ".ipynb": DocumentFormat.JUPYTER,
            ".xml": DocumentFormat.XML,
            ".rtf": DocumentFormat.RTF,
            ".py": DocumentFormat.CODE,
            ".js": DocumentFormat.CODE,
            ".ts": DocumentFormat.CODE,
            ".java": DocumentFormat.CODE,
            ".go": DocumentFormat.CODE,
            ".rs": DocumentFormat.CODE,
            ".cpp": DocumentFormat.CODE,
            ".c": DocumentFormat.CODE,
        }

        return format_map.get(extension, DocumentFormat.UNKNOWN)

    def _get_chunking_config(
        self, format_type: DocumentFormat, source_type: str
    ) -> ChunkingConfig:
        """Get chunking configuration for document type.

        Different formats and sources may require different chunking strategies.

        Args:
            format_type: Document format
            source_type: Source type

        Returns:
            ChunkingConfig for this document
        """
        # Format-specific overrides
        if format_type == DocumentFormat.MARKDOWN:
            return ChunkingConfig(
                strategy="by_title",
                max_chunk_size=self.config.max_chunk_size_chars,
                overlap_size=self.config.chunk_overlap_chars,
            )
        elif format_type == DocumentFormat.PDF:
            return ChunkingConfig(
                strategy="by_page",
                max_chunk_size=self.config.max_chunk_size_chars,
                overlap_size=100,
            )
        elif format_type == DocumentFormat.EMAIL:
            # Emails often chunked by conversation thread
            return ChunkingConfig(strategy="basic", max_chunk_size=2000, overlap_size=0)
        else:
            # Default chunking
            return ChunkingConfig(
                strategy=self.config.default_chunk_strategy,
                max_chunk_size=self.config.max_chunk_size_chars,
                overlap_size=self.config.chunk_overlap_chars,
            )

    def _extract_text_from_elements(self, elements: list[dict]) -> str:
        """Extract text content from Unstructured.io elements.

        Args:
            elements: List of element dictionaries

        Returns:
            Concatenated text content
        """
        text_parts = []
        for element in elements:
            if "text" in element:
                text_parts.append(element["text"])

        return "\n\n".join(text_parts)

    async def _quarantine_document(
        self, file_path: Path, source_id: str, source_type: str, error: Exception
    ) -> None:
        """Quarantine failed document for manual review.

        Uses NormalizationErrorHandler for detailed error classification,
        retry policy selection, and persistence to quarantine store.

        Args:
            file_path: Path to failed document
            source_id: Source identifier
            source_type: Source type
            error: Exception that caused failure
        """
        if not self.error_handler:
            # Fallback: log warning if no error_handler configured
            logger.warning(
                f"Document quarantine requested for {file_path.name}: "
                f"{type(error).__name__} - {str(error)[:100]}"
            )
            return

        # Use error handler for full classification and quarantine
        await self.error_handler.handle_error(
            file_path=file_path,
            source_id=source_id,
            source_type=source_type,
            error=error,
            metadata={
                "processing_stage": "normalization",
                "service": "NormalizationService",
            },
        )

    async def _process_with_streaming(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        format_type: DocumentFormat,
        source_metadata: Optional[dict],
    ) -> NormalizedDocument:
        """Process large document using streaming processor.

        Args:
            file_path: Path to document
            source_id: Source identifier
            source_type: Source type
            format_type: Detected format
            source_metadata: Additional metadata

        Returns:
            NormalizedDocument with chunks

        Raises:
            NormalizationError: If streaming fails
        """
        import hashlib

        logger.info(f"Using streaming processor for large file: {file_path.name}")

        try:
            # Read a small sample for metadata extraction
            with open(file_path, "r", encoding="utf-8") as f:
                sample_content = f.read(10000)  # Read first 10KB for metadata

            # Compute content hash from full file (streaming)
            content_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    content_hash.update(chunk)
            sha256 = content_hash.hexdigest()

            # Get file stats
            file_stats = file_path.stat()
            file_size_bytes = file_stats.st_size

            # Generate chunking config for this file
            chunking_config = self._get_chunking_config(format_type, source_type)

            # Stream chunks
            chunks = await self.streaming_processor.stream_chunks(
                file_path=file_path,
                chunking_config=chunking_config,
                parent_document_id=sha256,
            )

            # Create preliminary metadata
            from ..models import NormalizedMetadata
            from datetime import timezone

            created_at = datetime.fromtimestamp(file_stats.st_birthtime, tz=timezone.utc) if hasattr(file_stats, 'st_birthtime') else None
            modified_at = datetime.fromtimestamp(file_stats.st_mtime, tz=timezone.utc)

            metadata = NormalizedMetadata(
                source_path=str(file_path),
                source_id=source_id,
                source_type=source_type,
                format=format_type,
                content_type=self._get_content_type_for_format(format_type),
                created_at=created_at,
                modified_at=modified_at,
                ingested_at=datetime.now(timezone.utc),
                file_size_bytes=file_size_bytes,
                character_count=sum(chunk.character_count for chunk in chunks),
                word_count=sum(chunk.word_count for chunk in chunks),
                line_count=sum(chunk.content.count("\n") + 1 for chunk in chunks),
                content_hash=sha256,
                is_chunked=True,
                chunk_strategy=chunking_config.strategy,
                total_chunks=len(chunks),
            )

            # Create normalized document
            normalized_doc = NormalizedDocument(
                document_id=sha256,
                sha256=sha256,
                content=None,  # Content is in chunks
                chunks=chunks,
                metadata=metadata,
                normalized_at=datetime.now(timezone.utc),
            )

            # Enrich metadata
            enriched_metadata = await self.enrichment_pipeline.enrich(
                document=normalized_doc,
                enable_language_detection=self.config.enable_language_detection,
                enable_classification=self.config.enable_content_classification,
                compute_hash=False,  # Already computed
                file_path=file_path,
            )
            normalized_doc.metadata = enriched_metadata

            # Deliver to sink
            if self.sink:
                sink_payload = normalized_doc.to_sink_format()
                self.sink.handle(sink_payload)

            logger.info(
                f"Streaming processing complete: {file_path.name}, "
                f"{len(chunks)} chunks, {file_size_bytes / (1024 * 1024):.1f}MB"
            )

            return normalized_doc

        except Exception as e:
            logger.error(f"Streaming processing failed for {file_path}: {e}")
            raise NormalizationError(
                f"Failed to process large file via streaming: {str(e)}"
            ) from e

    def _get_content_type_for_format(self, format_type: DocumentFormat) -> str:
        """Get MIME type for document format.

        Args:
            format_type: Document format

        Returns:
            MIME type string
        """
        mime_map = {
            DocumentFormat.MARKDOWN: "text/markdown",
            DocumentFormat.PDF: "application/pdf",
            DocumentFormat.HTML: "text/html",
            DocumentFormat.TEXT: "text/plain",
            DocumentFormat.CODE: "text/plain",
        }
        return mime_map.get(format_type, "application/octet-stream")

    def _log_normalization_start(
        self, file_path: Path, source_id: str, source_type: str
    ) -> None:
        """Log normalization start event.

        Args:
            file_path: Path to file
            source_id: Source identifier
            source_type: Source type
        """
        if not self.audit_logger:
            return

        self.audit_logger.record(
            AuditEvent(
                job_id=f"normalize_{source_id}",
                source=source_type,
                action="normalization_started",
                status="in_progress",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix,
                },
            )
        )

    def _log_normalization_success(
        self, document: NormalizedDocument, duration_ms: float
    ) -> None:
        """Log successful normalization.

        Args:
            document: Normalized document
            duration_ms: Processing duration in milliseconds
        """
        if not self.audit_logger:
            return

        self.audit_logger.record(
            AuditEvent(
                job_id=f"normalize_{document.document_id[:16]}",
                source=document.metadata.source_type,
                action="normalization_completed",
                status="success",
                timestamp=datetime.now(timezone.utc),
                sha256=document.sha256,
                metadata={
                    "format": document.metadata.format.value,
                    "language": document.metadata.language,
                    "is_chunked": document.is_chunked,
                    "total_chunks": (
                        len(document.chunks) if document.is_chunked else None
                    ),
                    "character_count": document.metadata.character_count,
                    "processing_duration_ms": duration_ms,
                },
            )
        )

    def _log_normalization_failure(
        self, file_path: Path, source_id: str, error: Exception
    ) -> None:
        """Log normalization failure.

        Args:
            file_path: Path to failed file
            source_id: Source identifier
            error: Exception that caused failure
        """
        if not self.audit_logger:
            return

        self.audit_logger.record(
            AuditEvent(
                job_id=f"normalize_{source_id}",
                source="normalization_service",
                action="normalization_failed",
                status="failed",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "file_name": file_path.name,
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:500],  # Truncate for privacy
                },
            )
        )

    def get_metrics(self) -> dict:
        """Get processing metrics for telemetry.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "documents_processed": self.documents_processed,
            "documents_failed": self.documents_failed,
            "total_processing_time_ms": self.total_processing_time_ms,
            "average_processing_time_ms": (
                self.total_processing_time_ms / self.documents_processed
                if self.documents_processed > 0
                else 0
            ),
            "success_rate": (
                self.documents_processed
                / (self.documents_processed + self.documents_failed)
                if (self.documents_processed + self.documents_failed) > 0
                else 0
            ),
        }
