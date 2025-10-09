"""Document normalization pipeline for the Ghost grounding architecture.

This package provides a complete normalization pipeline that transforms diverse
document formats into standardized, chunked documents ready for PKG ingestion
and semantic embedding.

Main Components:
- NormalizationService: Central orchestrator for the pipeline
- FormatAdapterRegistry: Pluggable format-specific handlers
- ChunkingEngine: Multi-strategy document segmentation
- MetadataEnrichmentPipeline: Language detection, classification, hashing
- UnstructuredBridge: Interface to Unstructured.io library

Quick Start:
    >>> from futurnal.pipeline.normalization import create_normalization_service
    >>> service = create_normalization_service()
    >>> normalized = await service.normalize_document(
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
)
from .service import NormalizationConfig, NormalizationError, NormalizationService
from .unstructured_bridge import (
    PartitionStrategy,
    UnstructuredBridge,
    UnstructuredProcessingError,
)

__all__ = [
    # Service
    "NormalizationService",
    "NormalizationConfig",
    "NormalizationError",
    # Factory
    "create_normalization_service",
    "create_normalization_service_with_workspace",
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
]
