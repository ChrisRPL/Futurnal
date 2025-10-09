"""Factory function for creating configured NormalizationService instances.

Provides sensible defaults and component wiring for easy service instantiation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ...ingestion.local.state import StateStore
from ...orchestrator.quarantine import QuarantineStore
from ...privacy.audit import AuditLogger
from ..stubs import NormalizationSink
from .registry import FormatAdapterRegistry
from .chunking import ChunkingEngine
from .enrichment import MetadataEnrichmentPipeline
from .service import NormalizationConfig, NormalizationService
from .unstructured_bridge import UnstructuredBridge
from .orchestrator_integration import NormalizationProcessor

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
    quarantine_manager = QuarantineStore(quarantine_db_path)

    return create_normalization_service(
        config=config,
        audit_logger=audit_logger,
        quarantine_manager=quarantine_manager,
        sink=sink,
    )


def create_normalization_processor(
    config: Optional[NormalizationConfig] = None,
    *,
    state_store: Optional[StateStore] = None,
    audit_logger: Optional[AuditLogger] = None,
    quarantine_manager: Optional[QuarantineStore] = None,
    sink: Optional[NormalizationSink] = None,
    enable_state_checkpointing: bool = True,
) -> NormalizationProcessor:
    """Create NormalizationProcessor with configured service.

    This is the main factory for orchestrator integration, creating both
    the normalization service and processor wrapper for use with the
    IngestionOrchestrator.

    Args:
        config: Normalization configuration (uses defaults if None)
        state_store: Optional state store for checkpointing
        audit_logger: Optional audit logger for events
        quarantine_manager: Optional quarantine manager for failures
        sink: Optional normalization sink for delivery
        enable_state_checkpointing: Enable state-based caching

    Returns:
        Configured NormalizationProcessor ready for orchestrator use

    Example:
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
    # Create underlying normalization service
    service = create_normalization_service(
        config=config,
        audit_logger=audit_logger,
        quarantine_manager=quarantine_manager,
        sink=sink,
    )

    # Wrap in orchestrator integration processor
    processor = NormalizationProcessor(
        normalization_service=service,
        state_store=state_store,
        audit_logger=audit_logger,
        enable_state_checkpointing=enable_state_checkpointing,
    )

    logger.info("Normalization processor initialized for orchestrator integration")

    return processor


def create_normalization_processor_with_workspace(
    workspace_path: Path,
    config: Optional[NormalizationConfig] = None,
    *,
    state_store: Optional[StateStore] = None,
    sink: Optional[NormalizationSink] = None,
    enable_state_checkpointing: bool = True,
) -> NormalizationProcessor:
    """Create NormalizationProcessor with workspace-based components.

    Convenience factory that sets up audit logging and quarantine from
    workspace directory structure.

    Args:
        workspace_path: Path to workspace directory
        config: Normalization configuration
        state_store: Optional state store (if not provided, checkpointing disabled)
        sink: Optional normalization sink
        enable_state_checkpointing: Enable state-based caching

    Returns:
        Configured NormalizationProcessor with workspace integration
    """
    # Create audit logger
    audit_dir = workspace_path / "audit"
    audit_logger = AuditLogger(output_dir=audit_dir)

    # Create quarantine manager
    quarantine_dir = workspace_path / "quarantine"
    quarantine_db_path = quarantine_dir / "quarantine.db"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_manager = QuarantineStore(quarantine_db_path)

    return create_normalization_processor(
        config=config,
        state_store=state_store,
        audit_logger=audit_logger,
        quarantine_manager=quarantine_manager,
        sink=sink,
        enable_state_checkpointing=enable_state_checkpointing,
    )
