"""Multi-modal router for intelligent file processing.

Module 08: Multimodal Integration & Tool Enhancement - Phase 3
Routes multi-modal inputs to appropriate adapters with intelligent orchestration.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from futurnal.extraction.orchestrator_client import (
    OrchestratorClient,
    ProcessingStrategy,
    ProcessingPlan,
    get_orchestrator_client,
)
from futurnal.pipeline.models import NormalizedDocument, DocumentFormat, NormalizedMetadata
from futurnal.pipeline.normalization.registry import FormatAdapterRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Multi-Modal Router
# =============================================================================


class MultiModalRouter:
    """Routes multi-modal inputs to appropriate adapters with orchestration.

    Features:
    - Intelligent modality detection
    - Automatic adapter selection
    - Parallel and sequential execution
    - Dependency-based processing
    - Result aggregation
    """

    def __init__(
        self,
        adapter_registry: Optional[FormatAdapterRegistry] = None,
        orchestrator: Optional[OrchestratorClient] = None
    ):
        """Initialize multi-modal router.

        Args:
            adapter_registry: Registry of format adapters (default: global registry)
            orchestrator: Orchestrator client (default: auto-select)
        """
        self.registry = adapter_registry or FormatAdapterRegistry()
        self.orchestrator = orchestrator or get_orchestrator_client(backend="auto")
        logger.info(
            f"Initialized MultiModalRouter with {type(self.orchestrator).__name__}"
        )

    async def process(
        self,
        files: Union[Path, List[Path]],
        source_id: str,
        source_type: str,
        source_metadata: Optional[Dict] = None,
        strategy: Optional[str] = None
    ) -> Union[NormalizedDocument, List[NormalizedDocument]]:
        """Process one or more files with intelligent routing.

        Args:
            files: Single file path or list of file paths
            source_id: Source identifier for all files
            source_type: Source type (e.g., "local_files", "obsidian")
            source_metadata: Optional metadata for all files
            strategy: Optional processing strategy override ("auto", "parallel", "sequential")

        Returns:
            Single NormalizedDocument if input is single file,
            List of NormalizedDocuments if input is multiple files
        """
        # Normalize input to list
        if isinstance(files, Path):
            is_single = True
            file_list = [files]
        else:
            is_single = False
            file_list = files

        source_metadata = source_metadata or {}

        # Analyze inputs using orchestrator
        logger.info(f"Analyzing {len(file_list)} files for multi-modal processing")
        analysis = self.orchestrator.analyze_inputs(file_list)

        logger.info(
            f"Analysis: {len(analysis.modalities)} modalities, "
            f"recommended strategy: {analysis.recommended_strategy.value}"
        )
        for insight in analysis.insights:
            logger.info(f"  - {insight}")

        # Create processing plan
        plan_strategy = self._parse_strategy(strategy) if strategy else None
        plan = self.orchestrator.create_processing_plan(analysis, strategy=plan_strategy)

        logger.info(f"Processing plan: {plan.rationale}")
        logger.info(
            f"Estimated time: {plan.estimated_total_time:.1f}s "
            f"using {plan.strategy.value} strategy"
        )

        # Execute plan
        if plan.strategy == ProcessingStrategy.PARALLEL:
            documents = await self._execute_parallel(plan, source_id, source_type, source_metadata)
        elif plan.strategy == ProcessingStrategy.SEQUENTIAL:
            documents = await self._execute_sequential(plan, source_id, source_type, source_metadata)
        else:  # DEPENDENCY_GRAPH
            documents = await self._execute_dependency_graph(
                plan, source_id, source_type, source_metadata
            )

        logger.info(f"Successfully processed {len(documents)} files")

        # Return single document or list based on input
        return documents[0] if is_single else documents

    async def _execute_parallel(
        self,
        plan: ProcessingPlan,
        source_id: str,
        source_type: str,
        source_metadata: Dict
    ) -> List[NormalizedDocument]:
        """Execute processing plan in parallel.

        Args:
            plan: ProcessingPlan from orchestrator
            source_id: Source identifier
            source_type: Source type
            source_metadata: Source metadata

        Returns:
            List of NormalizedDocuments in original file order
        """
        logger.info(f"Executing parallel processing of {len(plan.steps)} files")

        # Create tasks for all steps
        tasks = []
        for step in plan.steps:
            task = self._process_single_file(
                file_path=step.file_path,
                source_id=f"{source_id}_{step.step_id}",
                source_type=source_type,
                source_metadata={
                    **source_metadata,
                    "step_id": step.step_id,
                    "modality": step.modality.value,
                }
            )
            tasks.append(task)

        # Execute all tasks concurrently
        documents = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        results = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, Exception):
                logger.error(f"Failed to process step {idx}: {doc}")
                # Create error placeholder document
                error_doc = self._create_error_document(
                    plan.steps[idx].file_path,
                    f"{source_id}_{idx}",
                    source_type,
                    str(doc)
                )
                results.append(error_doc)
            else:
                results.append(doc)

        return results

    async def _execute_sequential(
        self,
        plan: ProcessingPlan,
        source_id: str,
        source_type: str,
        source_metadata: Dict
    ) -> List[NormalizedDocument]:
        """Execute processing plan sequentially.

        Args:
            plan: ProcessingPlan from orchestrator
            source_id: Source identifier
            source_type: Source type
            source_metadata: Source metadata

        Returns:
            List of NormalizedDocuments in execution order
        """
        logger.info(f"Executing sequential processing of {len(plan.steps)} files")

        documents = []
        for step_idx in plan.execution_order:
            step = plan.steps[step_idx]

            logger.info(
                f"Processing step {step_idx + 1}/{len(plan.steps)}: "
                f"{step.file_path.name} ({step.modality.value})"
            )

            try:
                doc = await self._process_single_file(
                    file_path=step.file_path,
                    source_id=f"{source_id}_{step.step_id}",
                    source_type=source_type,
                    source_metadata={
                        **source_metadata,
                        "step_id": step.step_id,
                        "modality": step.modality.value,
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process step {step_idx}: {e}")
                error_doc = self._create_error_document(
                    step.file_path,
                    f"{source_id}_{step.step_id}",
                    source_type,
                    str(e)
                )
                documents.append(error_doc)

        return documents

    async def _execute_dependency_graph(
        self,
        plan: ProcessingPlan,
        source_id: str,
        source_type: str,
        source_metadata: Dict
    ) -> List[NormalizedDocument]:
        """Execute processing plan based on dependency graph.

        Args:
            plan: ProcessingPlan from orchestrator
            source_id: Source identifier
            source_type: Source type
            source_metadata: Source metadata

        Returns:
            List of NormalizedDocuments respecting dependencies
        """
        logger.info(
            f"Executing dependency-based processing of {len(plan.steps)} files "
            f"with {len(plan.dependencies)} dependency relationships"
        )

        # For now, fall back to sequential if dependencies exist
        # Future: implement proper topological sort + parallel execution
        if plan.dependencies:
            logger.warning(
                "Dependency graph execution not yet fully implemented, "
                "falling back to sequential"
            )

        return await self._execute_sequential(plan, source_id, source_type, source_metadata)

    async def _process_single_file(
        self,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: Dict
    ) -> NormalizedDocument:
        """Process a single file using appropriate adapter.

        Args:
            file_path: Path to file
            source_id: Source identifier
            source_type: Source type
            source_metadata: Source metadata

        Returns:
            NormalizedDocument

        Raises:
            AdapterError: If processing fails
        """
        # Detect document format
        format = self._detect_format(file_path)

        # Get adapter from registry
        adapter = self.registry.get_adapter(format)

        if adapter is None:
            raise ValueError(f"No adapter found for format: {format}")

        logger.debug(
            f"Processing {file_path.name} with {adapter.name} "
            f"(format: {format.value})"
        )

        # Normalize file using adapter
        document = await adapter.normalize(
            file_path=file_path,
            source_id=source_id,
            source_type=source_type,
            source_metadata=source_metadata
        )

        return document

    def _detect_format(self, file_path: Path) -> DocumentFormat:
        """Detect document format from file extension.

        Args:
            file_path: Path to file

        Returns:
            DocumentFormat
        """
        ext = file_path.suffix.lower()

        # Text formats
        if ext in {".txt", ".md", ".rst"}:
            return DocumentFormat.TEXT
        if ext in {".html", ".htm"}:
            return DocumentFormat.HTML
        if ext == ".json":
            return DocumentFormat.JSON
        if ext in {".xml", ".svg"}:
            return DocumentFormat.XML
        if ext == ".csv":
            return DocumentFormat.CSV

        # Audio formats
        if ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}:
            return DocumentFormat.AUDIO

        # Image formats
        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}:
            return DocumentFormat.IMAGE

        # PDF (runtime detection for scanned vs text needed)
        if ext == ".pdf":
            return DocumentFormat.PDF

        # Office formats
        if ext in {".docx", ".doc"}:
            return DocumentFormat.DOCX
        if ext in {".xlsx", ".xls"}:
            return DocumentFormat.XLSX
        if ext in {".pptx", ".ppt"}:
            return DocumentFormat.PPTX

        # Email
        if ext in {".eml", ".msg"}:
            return DocumentFormat.EMAIL

        return DocumentFormat.UNKNOWN

    def _parse_strategy(self, strategy: str) -> ProcessingStrategy:
        """Parse strategy string to ProcessingStrategy enum.

        Args:
            strategy: "auto", "parallel", "sequential", or "dependency_graph"

        Returns:
            ProcessingStrategy
        """
        strategy_lower = strategy.lower()

        if strategy_lower == "parallel":
            return ProcessingStrategy.PARALLEL
        elif strategy_lower == "sequential":
            return ProcessingStrategy.SEQUENTIAL
        elif strategy_lower in {"dependency_graph", "dependency"}:
            return ProcessingStrategy.DEPENDENCY_GRAPH
        else:
            # Return None for "auto" or unknown - let orchestrator decide
            return None

    def _create_error_document(
        self,
        file_path: Path,
        source_id: str,
        source_type: str,
        error_message: str
    ) -> NormalizedDocument:
        """Create error placeholder document for failed processing.

        Args:
            file_path: Path to failed file
            source_id: Source identifier
            source_type: Source type
            error_message: Error message

        Returns:
            NormalizedDocument with error metadata
        """
        import hashlib
        from datetime import datetime, timezone

        # Create minimal error document
        error_content = f"[ERROR] Failed to process: {error_message}"
        content_hash = hashlib.sha256(error_content.encode()).hexdigest()

        metadata = NormalizedMetadata(
            source_path=str(file_path),
            source_id=source_id,
            source_type=source_type,
            format=DocumentFormat.UNKNOWN,
            content_type="text/plain",
            character_count=len(error_content),
            word_count=len(error_content.split()),
            line_count=error_content.count('\n') + 1,
            content_hash=content_hash,
            ingested_at=datetime.now(timezone.utc),
            extra={
                "error": True,
                "error_message": error_message,
                "file_name": file_path.name,
            }
        )

        return NormalizedDocument(
            document_id=content_hash,
            sha256=content_hash,
            content=error_content,
            metadata=metadata
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def process_files(
    files: Union[Path, List[Path]],
    source_id: str = "multimodal",
    source_type: str = "local_files",
    source_metadata: Optional[Dict] = None,
    strategy: Optional[str] = None,
    orchestrator: Optional[OrchestratorClient] = None
) -> Union[NormalizedDocument, List[NormalizedDocument]]:
    """Process one or more files with automatic multi-modal routing.

    Convenience function for quick file processing without manually creating a router.

    Args:
        files: Single file path or list of file paths
        source_id: Source identifier (default: "multimodal")
        source_type: Source type (default: "local_files")
        source_metadata: Optional source metadata
        strategy: Processing strategy ("auto", "parallel", "sequential")
        orchestrator: Optional orchestrator client override

    Returns:
        Single NormalizedDocument if input is single file,
        List of NormalizedDocuments if input is multiple files

    Example:
        # Single file
        doc = await process_files(Path("notes.md"))

        # Multiple files with parallel processing
        docs = await process_files([
            Path("slides.pdf"),
            Path("recording.wav"),
            Path("notes.md")
        ], strategy="parallel")
    """
    router = MultiModalRouter(orchestrator=orchestrator)

    return await router.process(
        files=files,
        source_id=source_id,
        source_type=source_type,
        source_metadata=source_metadata,
        strategy=strategy
    )
