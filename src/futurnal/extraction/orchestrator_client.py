"""Orchestrator client for multi-modal processing coordination.

Module 08: Multimodal Integration & Tool Enhancement - Phase 3
Provides intelligent coordination across audio, image, text, and PDF modalities
using NVIDIA Orchestrator-8B for agentic planning.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ModalityType(str, Enum):
    """Supported modality types."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"
    SCANNED_PDF = "scanned_pdf"
    VIDEO = "video"  # Future support
    UNKNOWN = "unknown"


class ProcessingStrategy(str, Enum):
    """Processing execution strategies."""
    SEQUENTIAL = "sequential"  # Process files one by one
    PARALLEL = "parallel"      # Process all files concurrently
    DEPENDENCY_GRAPH = "dependency_graph"  # Process based on dependencies


@dataclass
class FileInfo:
    """Information about a file to be processed."""
    path: Path
    modality: ModalityType
    size_bytes: int
    estimated_processing_time: float  # seconds
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessingStep:
    """A single processing step in a plan."""
    step_id: int
    file_path: Path
    modality: ModalityType
    adapter_name: str
    estimated_time: float
    priority: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class InputAnalysis:
    """Analysis of input files."""
    files: List[FileInfo]
    modalities: List[ModalityType]
    total_size_bytes: int
    estimated_total_time: float
    recommended_strategy: ProcessingStrategy
    insights: List[str] = field(default_factory=list)


@dataclass
class ProcessingPlan:
    """Coordinated processing plan for multi-modal inputs."""
    steps: List[ProcessingStep]
    execution_order: List[int]  # Order of step_ids
    dependencies: Dict[int, List[int]]  # step_id -> [dependent_step_ids]
    strategy: ProcessingStrategy
    estimated_total_time: float
    rationale: str = ""


# =============================================================================
# Orchestrator Client Protocol
# =============================================================================


class OrchestratorClient:
    """Protocol for orchestrator clients."""

    def analyze_inputs(self, files: List[Path]) -> InputAnalysis:
        """Analyze input files and determine modalities.

        Args:
            files: List of file paths to analyze

        Returns:
            InputAnalysis with modality detection and recommendations
        """
        raise NotImplementedError

    def create_processing_plan(
        self,
        analysis: InputAnalysis,
        strategy: Optional[ProcessingStrategy] = None
    ) -> ProcessingPlan:
        """Create a processing plan from input analysis.

        Args:
            analysis: InputAnalysis from analyze_inputs()
            strategy: Optional override for processing strategy

        Returns:
            ProcessingPlan with steps and execution order
        """
        raise NotImplementedError


# =============================================================================
# NVIDIA Orchestrator-8B Client
# =============================================================================


class NvidiaOrchestratorClient(OrchestratorClient):
    """NVIDIA Orchestrator-8B implementation via Ollama.

    Uses agentic LLM for intelligent multi-modal coordination.
    """

    def __init__(
        self,
        model_name: str = "nvidia/orchestrator-8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60
    ):
        """Initialize NVIDIA Orchestrator client.

        Args:
            model_name: Ollama model name (default: nvidia/orchestrator-8b)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self._llm_client = None

    def _get_llm_client(self):
        """Lazy load Ollama LLM client."""
        if self._llm_client is None:
            try:
                from futurnal.extraction.ollama_client import OllamaLLMClient
                self._llm_client = OllamaLLMClient(
                    model_name=self.model_name,
                    base_url=self.base_url
                )
                logger.info(f"Loaded Orchestrator LLM: {self.model_name}")
            except ImportError as e:
                raise ImportError(
                    f"Failed to load OllamaLLMClient: {e}. "
                    "Ensure futurnal.extraction.ollama_client is available."
                )
        return self._llm_client

    def analyze_inputs(self, files: List[Path]) -> InputAnalysis:
        """Analyze input files using Orchestrator-8B.

        Args:
            files: List of file paths to analyze

        Returns:
            InputAnalysis with intelligent modality detection
        """
        file_infos = []

        for file_path in files:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            # Detect modality from file extension
            modality = self._detect_modality(file_path)
            size_bytes = file_path.stat().st_size
            estimated_time = self._estimate_processing_time(modality, size_bytes)

            file_infos.append(FileInfo(
                path=file_path,
                modality=modality,
                size_bytes=size_bytes,
                estimated_processing_time=estimated_time,
                metadata={"extension": file_path.suffix}
            ))

        # Use LLM to determine recommended strategy
        llm = self._get_llm_client()
        strategy = self._recommend_strategy_via_llm(llm, file_infos)

        total_size = sum(f.size_bytes for f in file_infos)
        total_time = sum(f.estimated_processing_time for f in file_infos)
        modalities = list(set(f.modality for f in file_infos))

        insights = self._generate_insights(file_infos, strategy)

        return InputAnalysis(
            files=file_infos,
            modalities=modalities,
            total_size_bytes=total_size,
            estimated_total_time=total_time,
            recommended_strategy=strategy,
            insights=insights
        )

    def create_processing_plan(
        self,
        analysis: InputAnalysis,
        strategy: Optional[ProcessingStrategy] = None
    ) -> ProcessingPlan:
        """Create processing plan using Orchestrator-8B.

        Args:
            analysis: InputAnalysis from analyze_inputs()
            strategy: Optional strategy override

        Returns:
            ProcessingPlan with execution order
        """
        strategy = strategy or analysis.recommended_strategy

        # Create processing steps
        steps = []
        for idx, file_info in enumerate(analysis.files):
            adapter_name = self._get_adapter_name(file_info.modality)
            step = ProcessingStep(
                step_id=idx,
                file_path=file_info.path,
                modality=file_info.modality,
                adapter_name=adapter_name,
                estimated_time=file_info.estimated_processing_time,
                priority=self._calculate_priority(file_info),
                metadata=file_info.metadata
            )
            steps.append(step)

        # Determine execution order based on strategy
        if strategy == ProcessingStrategy.PARALLEL:
            execution_order = list(range(len(steps)))
            dependencies = {}
            estimated_time = max((s.estimated_time for s in steps), default=0)
        elif strategy == ProcessingStrategy.SEQUENTIAL:
            # Sort by priority (higher priority first)
            sorted_steps = sorted(steps, key=lambda s: s.priority, reverse=True)
            execution_order = [s.step_id for s in sorted_steps]
            dependencies = {}
            estimated_time = sum(s.estimated_time for s in steps)
        else:  # DEPENDENCY_GRAPH
            execution_order, dependencies = self._build_dependency_graph(steps)
            estimated_time = self._calculate_critical_path_time(steps, dependencies)

        rationale = self._generate_rationale(strategy, steps, analysis)

        return ProcessingPlan(
            steps=steps,
            execution_order=execution_order,
            dependencies=dependencies,
            strategy=strategy,
            estimated_total_time=estimated_time,
            rationale=rationale
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _detect_modality(self, file_path: Path) -> ModalityType:
        """Detect modality from file extension."""
        ext = file_path.suffix.lower()

        # Text formats
        if ext in {".txt", ".md", ".rst", ".html", ".xml", ".json", ".csv"}:
            return ModalityType.TEXT

        # Audio formats
        if ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}:
            return ModalityType.AUDIO

        # Image formats
        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}:
            return ModalityType.IMAGE

        # PDF (will need runtime detection for scanned vs text)
        if ext == ".pdf":
            return ModalityType.PDF

        # Video (future support)
        if ext in {".mp4", ".avi", ".mov", ".mkv"}:
            return ModalityType.VIDEO

        return ModalityType.UNKNOWN

    def _estimate_processing_time(self, modality: ModalityType, size_bytes: int) -> float:
        """Estimate processing time in seconds."""
        # Simple heuristic - can be refined with benchmarks
        size_mb = size_bytes / (1024 * 1024)

        if modality == ModalityType.TEXT:
            return size_mb * 0.1  # 0.1s per MB
        elif modality == ModalityType.AUDIO:
            # Assume 1MB ≈ 1 minute audio, Whisper runs at ~2x real-time
            return size_mb * 30  # 30s per MB
        elif modality == ModalityType.IMAGE:
            return size_mb * 2  # 2s per MB for OCR
        elif modality == ModalityType.PDF:
            return size_mb * 5  # 5s per MB (depends on pages)
        else:
            return size_mb * 1

    def _recommend_strategy_via_llm(
        self,
        llm,
        file_infos: List[FileInfo]
    ) -> ProcessingStrategy:
        """Use LLM to recommend processing strategy."""
        # Prepare prompt for LLM
        files_summary = "\n".join([
            f"- {f.path.name} ({f.modality.value}, {f.size_bytes / 1024:.1f} KB, ~{f.estimated_processing_time:.1f}s)"
            for f in file_infos
        ])

        prompt = f"""You are a multi-modal processing coordinator. Analyze these files and recommend a processing strategy.

Files to process:
{files_summary}

Available strategies:
1. PARALLEL: Process all files concurrently (fastest, uses more resources)
2. SEQUENTIAL: Process files one by one (slower, lower resource usage)
3. DEPENDENCY_GRAPH: Process based on dependencies (e.g., audio + slides together)

Consider:
- Total processing time
- Resource availability
- File relationships (e.g., meeting recording + presentation slides)

Respond with ONLY the strategy name: PARALLEL, SEQUENTIAL, or DEPENDENCY_GRAPH"""

        try:
            response = llm.generate(prompt, temperature=0.1, max_tokens=50)
            strategy_text = response.strip().upper()

            if "PARALLEL" in strategy_text:
                return ProcessingStrategy.PARALLEL
            elif "SEQUENTIAL" in strategy_text:
                return ProcessingStrategy.SEQUENTIAL
            elif "DEPENDENCY_GRAPH" in strategy_text or "DEPENDENCY" in strategy_text:
                return ProcessingStrategy.DEPENDENCY_GRAPH
            else:
                # Default fallback
                logger.warning(f"Unclear LLM response: {response}, defaulting to PARALLEL")
                return ProcessingStrategy.PARALLEL

        except Exception as e:
            logger.warning(f"LLM strategy recommendation failed: {e}, defaulting to PARALLEL")
            return ProcessingStrategy.PARALLEL

    def _generate_insights(
        self,
        file_infos: List[FileInfo],
        strategy: ProcessingStrategy
    ) -> List[str]:
        """Generate insights about the input files."""
        insights = []

        # Modality distribution
        modality_counts = {}
        for f in file_infos:
            modality_counts[f.modality] = modality_counts.get(f.modality, 0) + 1

        if len(modality_counts) > 1:
            insights.append(f"Multi-modal input detected: {len(modality_counts)} modalities")

        # Large files
        large_files = [f for f in file_infos if f.size_bytes > 10 * 1024 * 1024]
        if large_files:
            insights.append(f"{len(large_files)} large files (>10MB) may take longer to process")

        # Audio files
        audio_files = [f for f in file_infos if f.modality == ModalityType.AUDIO]
        if audio_files:
            total_audio_time = sum(f.estimated_processing_time for f in audio_files)
            insights.append(f"{len(audio_files)} audio files (~{total_audio_time:.0f}s processing)")

        return insights

    def _get_adapter_name(self, modality: ModalityType) -> str:
        """Get adapter name for modality."""
        mapping = {
            ModalityType.TEXT: "TextAdapter",
            ModalityType.AUDIO: "AudioAdapter",
            ModalityType.IMAGE: "ImageAdapter",
            ModalityType.PDF: "PDFAdapter",
            ModalityType.SCANNED_PDF: "ScannedPDFAdapter",
        }
        return mapping.get(modality, "UnknownAdapter")

    def _calculate_priority(self, file_info: FileInfo) -> int:
        """Calculate processing priority (higher = more important)."""
        # Text files: low priority (fast)
        if file_info.modality == ModalityType.TEXT:
            return 1

        # Audio files: high priority (slow, want to start early)
        if file_info.modality == ModalityType.AUDIO:
            return 10

        # Images/PDFs: medium priority
        if file_info.modality in {ModalityType.IMAGE, ModalityType.PDF}:
            return 5

        return 1

    def _build_dependency_graph(
        self,
        steps: List[ProcessingStep]
    ) -> tuple[List[int], Dict[int, List[int]]]:
        """Build dependency graph for coordinated processing.

        Returns:
            (execution_order, dependencies)
        """
        # Simple heuristic: group related files
        # E.g., if there's audio + PDF with similar names, process together

        # For now, just use priority-based ordering
        # Future: detect file name patterns, temporal relationships, etc.
        sorted_steps = sorted(steps, key=lambda s: s.priority, reverse=True)
        execution_order = [s.step_id for s in sorted_steps]
        dependencies = {}  # No dependencies yet

        return execution_order, dependencies

    def _calculate_critical_path_time(
        self,
        steps: List[ProcessingStep],
        dependencies: Dict[int, List[int]]
    ) -> float:
        """Calculate critical path time through dependency graph."""
        if not dependencies:
            # No dependencies = parallel execution
            return max((s.estimated_time for s in steps), default=0)

        # Simple heuristic: sum of longest dependency chain
        # Future: proper critical path algorithm
        return sum(s.estimated_time for s in steps)

    def _generate_rationale(
        self,
        strategy: ProcessingStrategy,
        steps: List[ProcessingStep],
        analysis: InputAnalysis
    ) -> str:
        """Generate human-readable rationale for the plan."""
        rationale_parts = [
            f"Processing {len(steps)} files using {strategy.value} strategy.",
            f"Total estimated time: {analysis.estimated_total_time:.1f}s.",
        ]

        if strategy == ProcessingStrategy.PARALLEL:
            rationale_parts.append("Files will be processed concurrently for maximum speed.")
        elif strategy == ProcessingStrategy.SEQUENTIAL:
            rationale_parts.append("Files will be processed one at a time to minimize resource usage.")
        else:
            rationale_parts.append("Files will be processed based on dependencies and relationships.")

        return " ".join(rationale_parts)


# =============================================================================
# Rule-Based Orchestrator Client (Fallback)
# =============================================================================


class RuleBasedOrchestratorClient(OrchestratorClient):
    """Simple rule-based orchestrator without LLM dependency.

    Uses heuristics for modality detection and strategy selection.
    """

    def analyze_inputs(self, files: List[Path]) -> InputAnalysis:
        """Analyze inputs using simple heuristics."""
        file_infos = []

        for file_path in files:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            modality = self._detect_modality(file_path)
            size_bytes = file_path.stat().st_size
            estimated_time = self._estimate_processing_time(modality, size_bytes)

            file_infos.append(FileInfo(
                path=file_path,
                modality=modality,
                size_bytes=size_bytes,
                estimated_processing_time=estimated_time,
                metadata={"extension": file_path.suffix}
            ))

        # Simple rule: if multiple files, use parallel; if single, use sequential
        if len(file_infos) > 3:
            strategy = ProcessingStrategy.PARALLEL
        else:
            strategy = ProcessingStrategy.SEQUENTIAL

        total_size = sum(f.size_bytes for f in file_infos)
        total_time = sum(f.estimated_processing_time for f in file_infos)
        modalities = list(set(f.modality for f in file_infos))

        return InputAnalysis(
            files=file_infos,
            modalities=modalities,
            total_size_bytes=total_size,
            estimated_total_time=total_time,
            recommended_strategy=strategy,
            insights=[f"Detected {len(modalities)} modalities across {len(file_infos)} files"]
        )

    def create_processing_plan(
        self,
        analysis: InputAnalysis,
        strategy: Optional[ProcessingStrategy] = None
    ) -> ProcessingPlan:
        """Create simple processing plan."""
        strategy = strategy or analysis.recommended_strategy

        steps = []
        for idx, file_info in enumerate(analysis.files):
            adapter_name = self._get_adapter_name(file_info.modality)
            step = ProcessingStep(
                step_id=idx,
                file_path=file_info.path,
                modality=file_info.modality,
                adapter_name=adapter_name,
                estimated_time=file_info.estimated_processing_time,
                metadata=file_info.metadata
            )
            steps.append(step)

        execution_order = list(range(len(steps)))
        dependencies = {}

        if strategy == ProcessingStrategy.PARALLEL:
            estimated_time = max((s.estimated_time for s in steps), default=0)
        else:
            estimated_time = sum(s.estimated_time for s in steps)

        return ProcessingPlan(
            steps=steps,
            execution_order=execution_order,
            dependencies=dependencies,
            strategy=strategy,
            estimated_total_time=estimated_time,
            rationale=f"Simple {strategy.value} processing of {len(steps)} files"
        )

    def _detect_modality(self, file_path: Path) -> ModalityType:
        """Same as NvidiaOrchestratorClient."""
        ext = file_path.suffix.lower()

        if ext in {".txt", ".md", ".rst", ".html", ".xml", ".json", ".csv"}:
            return ModalityType.TEXT
        if ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}:
            return ModalityType.AUDIO
        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}:
            return ModalityType.IMAGE
        if ext == ".pdf":
            return ModalityType.PDF
        if ext in {".mp4", ".avi", ".mov", ".mkv"}:
            return ModalityType.VIDEO

        return ModalityType.UNKNOWN

    def _estimate_processing_time(self, modality: ModalityType, size_bytes: int) -> float:
        """Same as NvidiaOrchestratorClient."""
        size_mb = size_bytes / (1024 * 1024)

        if modality == ModalityType.TEXT:
            return size_mb * 0.1
        elif modality == ModalityType.AUDIO:
            return size_mb * 30
        elif modality == ModalityType.IMAGE:
            return size_mb * 2
        elif modality == ModalityType.PDF:
            return size_mb * 5
        else:
            return size_mb * 1

    def _get_adapter_name(self, modality: ModalityType) -> str:
        """Same as NvidiaOrchestratorClient."""
        mapping = {
            ModalityType.TEXT: "TextAdapter",
            ModalityType.AUDIO: "AudioAdapter",
            ModalityType.IMAGE: "ImageAdapter",
            ModalityType.PDF: "PDFAdapter",
            ModalityType.SCANNED_PDF: "ScannedPDFAdapter",
        }
        return mapping.get(modality, "UnknownAdapter")


# =============================================================================
# Factory Function
# =============================================================================


def get_orchestrator_client(backend: str = "auto") -> OrchestratorClient:
    """Get orchestrator client with auto-selection.

    Args:
        backend: "auto", "nvidia", or "rule_based"

    Returns:
        OrchestratorClient instance
    """
    # Check environment variable override
    import os
    env_backend = os.getenv("FUTURNAL_ORCHESTRATOR_BACKEND", backend)

    if env_backend == "nvidia":
        return NvidiaOrchestratorClient()
    elif env_backend == "rule_based":
        return RuleBasedOrchestratorClient()
    elif env_backend == "auto":
        # Try NVIDIA Orchestrator first, fall back to rule-based
        try:
            # Check if Ollama is available
            client = NvidiaOrchestratorClient()
            client._get_llm_client()  # Test loading
            logger.info("Using NVIDIA Orchestrator-8B backend")
            return client
        except Exception as e:
            logger.warning(f"NVIDIA Orchestrator unavailable: {e}, falling back to rule-based")
            return RuleBasedOrchestratorClient()
    else:
        # Unknown backend, fall back to rule-based
        logger.warning(f"Unknown orchestrator backend '{env_backend}', using rule-based")
        return RuleBasedOrchestratorClient()


def nvidia_orchestrator_available() -> bool:
    """Check if NVIDIA Orchestrator-8B is available via Ollama."""
    try:
        client = NvidiaOrchestratorClient()
        client._get_llm_client()
        return True
    except Exception:
        return False
