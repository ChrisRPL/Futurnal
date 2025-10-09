"""Document normalization pipeline for the Ghost grounding architecture.

This package provides a complete normalization pipeline that transforms diverse
document formats into standardized, chunked documents ready for PKG ingestion
and semantic embedding.

Main Components:
- NormalizationService: Central orchestrator for the pipeline
- NormalizationProcessor: Orchestrator integration wrapper with state management
- FormatAdapterRegistry: Pluggable format-specific handlers
- ChunkingEngine: Multi-strategy document segmentation
- MetadataEnrichmentPipeline: Language detection, classification, hashing
- UnstructuredBridge: Interface to Unstructured.io library
- NormalizationErrorHandler: Error classification and quarantine integration

Quick Start (Standalone):
    >>> from futurnal.pipeline.normalization import create_normalization_service
    >>> service = create_normalization_service()
    >>> normalized = await service.normalize_document(
    ...     file_path="document.pdf",
    ...     source_id="doc-123",
    ...     source_type="local_files"
    ... )

Quick Start (Orchestrator Integration):
    >>> from futurnal.pipeline.normalization import create_normalization_processor
    >>> processor = create_normalization_processor(
    ...     state_store=state_store,
    ...     audit_logger=audit_logger,
    ...     sink=sink
    ... )
    >>> result = await processor.process_file(
    ...     file_path="document.pdf",
    ...     source_id="doc-123",
    ...     source_type="local_files"
    ... )
"""

from .registry import (
    AdapterError,
    AdapterNotFoundError,
    FormatAdapter,
    FormatAdapterRegistry,
)
from .chunking import ChunkingConfig, ChunkingEngine, ChunkingStrategy
from .enrichment import (
    ContentClassifier,
    LanguageDetector,
    MetadataEnrichmentPipeline,
)
from .factory import (
    create_normalization_service,
    create_normalization_service_with_workspace,
    create_normalization_processor,
    create_normalization_processor_with_workspace,
)
from .orchestrator_integration import (
    NormalizationProcessor,
    ProcessingResult,
)
from .service import NormalizationConfig, NormalizationError, NormalizationService
from .unstructured_bridge import (
    PartitionStrategy,
    UnstructuredBridge,
    UnstructuredProcessingError,
)
from .streaming import (
    MemoryMonitor,
    ProgressCallback,
    StreamingConfig,
    StreamingProcessor,
)
from .error_handler import (
    NormalizationErrorHandler,
    NormalizationErrorType,
)

__all__ = [
    # Service
    "NormalizationService",
    "NormalizationConfig",
    "NormalizationError",
    # Factory
    "create_normalization_service",
    "create_normalization_service_with_workspace",
    "create_normalization_processor",
    "create_normalization_processor_with_workspace",
    # Orchestrator Integration
    "NormalizationProcessor",
    "ProcessingResult",
    # Adapters
    "FormatAdapterRegistry",
    "FormatAdapter",
    "AdapterError",
    "AdapterNotFoundError",
    # Chunking
    "ChunkingEngine",
    "ChunkingConfig",
    "ChunkingStrategy",
    # Enrichment
    "MetadataEnrichmentPipeline",
    "LanguageDetector",
    "ContentClassifier",
    # Unstructured
    "UnstructuredBridge",
    "PartitionStrategy",
    "UnstructuredProcessingError",
    # Streaming
    "StreamingProcessor",
    "StreamingConfig",
    "MemoryMonitor",
    "ProgressCallback",
    # Error Handling
    "NormalizationErrorHandler",
    "NormalizationErrorType",
]
