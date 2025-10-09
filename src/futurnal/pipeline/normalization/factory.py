"""Factory function for creating configured NormalizationService instances.

Provides sensible defaults and component wiring for easy service instantiation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ...orchestrator.quarantine import QuarantineStore
from ...privacy.audit import AuditLogger
from ..stubs import NormalizationSink
from .registry import FormatAdapterRegistry
from .chunking import ChunkingEngine
from .enrichment import MetadataEnrichmentPipeline
from .service import NormalizationConfig, NormalizationService
from .unstructured_bridge import UnstructuredBridge

logger = logging.getLogger(__name__)


def create_normalization_service(
    config: Optional[NormalizationConfig] = None,
    *,
    audit_logger: Optional[AuditLogger] = None,
    quarantine_manager: Optional[QuarantineStore] = None,
    sink: Optional[NormalizationSink] = None,
) -> NormalizationService:
    """Factory function to create configured NormalizationService.

    Sets up all required components with sensible defaults.

    Args:
        config: Normalization configuration (uses defaults if None)
        audit_logger: Optional audit logger for privacy-aware logging
        quarantine_manager: Optional quarantine manager for failed documents
        sink: Optional sink for delivering normalized documents

    Returns:
        Fully configured NormalizationService instance

    Example:
        >>> service = create_normalization_service()
        >>> normalized = await service.normalize_document(
        ...     file_path="document.pdf",
        ...     source_id="doc-123",
        ...     source_type="local_files"
        ... )
    """
    if config is None:
        config = NormalizationConfig()

    # Initialize components
    logger.info("Initializing normalization service components...")

    # Create adapter registry and register default adapters
    adapter_registry = FormatAdapterRegistry()
    adapter_registry.register_default_adapters()
    logger.debug(
        f"Registered adapters for {len(adapter_registry.list_supported_formats())} formats"
    )

    # Create chunking engine
    chunking_engine = ChunkingEngine()
    logger.debug("Initialized chunking engine")

    # Create enrichment pipeline
    enrichment_pipeline = MetadataEnrichmentPipeline()
    logger.debug("Initialized metadata enrichment pipeline")

    # Create Unstructured.io bridge
    unstructured_bridge = UnstructuredBridge()
    logger.debug("Initialized Unstructured.io bridge")

    # Create service
    service = NormalizationService(
        config=config,
        adapter_registry=adapter_registry,
        chunking_engine=chunking_engine,
        enrichment_pipeline=enrichment_pipeline,
        unstructured_bridge=unstructured_bridge,
        sink=sink,
        audit_logger=audit_logger,
        quarantine_manager=quarantine_manager,
    )

    logger.info("Normalization service initialized successfully")

    return service


def create_normalization_service_with_workspace(
    workspace_path: Path,
    config: Optional[NormalizationConfig] = None,
    *,
    sink: Optional[NormalizationSink] = None,
) -> NormalizationService:
    """Create NormalizationService with workspace-based audit and quarantine.

    Args:
        workspace_path: Path to workspace directory
        config: Normalization configuration
        sink: Optional normalization sink

    Returns:
        NormalizationService with workspace integration
    """
    # Create audit logger
    audit_dir = workspace_path / "audit"
    audit_logger = AuditLogger(output_dir=audit_dir)

    # Create quarantine manager
    quarantine_dir = workspace_path / "quarantine"
    quarantine_db_path = quarantine_dir / "quarantine.db"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_manager = QuarantineStore(db_path=quarantine_db_path)

    return create_normalization_service(
        config=config,
        audit_logger=audit_logger,
        quarantine_manager=quarantine_manager,
        sink=sink,
    )
