"""Futurnal extraction package.

Module 08: Multimodal Integration & Tool Enhancement
Production Plan: docs/phase-1/entity-relationship-extraction-production-plan/08-multimodal-integration.md

This package provides unified extraction capabilities for all modalities:

Unified API:
    extract_from_any_source() - Single entry point for text, audio, image, PDF

Specialized Clients:
    WhisperTranscriptionClient - Audio transcription via Whisper V3
    OCRClient - Image/document OCR via DeepSeek-OCR
    OrchestratorClient - Multi-modal coordination via NVIDIA Orchestrator-8B
    LLMClient - Text extraction via Llama/Qwen/Phi models

Example:
    >>> from futurnal.extraction import extract_from_any_source
    >>>
    >>> # Process any file type automatically
    >>> doc = await extract_from_any_source("recording.wav")
    >>> print(doc.content)
    >>>
    >>> # Or use specialized clients directly
    >>> from futurnal.extraction import get_transcription_client
    >>> client = get_transcription_client()
    >>> result = client.transcribe("audio.mp3")
"""

# Unified API entry point (Module 08 Phase 4)
from .unified import extract_from_any_source, extract, process

# Audio transcription (Module 08 Phase 1)
from .whisper_client import (
    get_transcription_client,
    OllamaWhisperClient,
    LocalWhisperClient,
    TranscriptionResult,
    TimestampedSegment,
)

# OCR extraction (Module 08 Phase 2)
from .ocr_client import (
    get_ocr_client,
    DeepSeekOCRClient,
    TesseractOCRClient,
    OCRResult,
    TextRegion,
    BoundingBox,
    LayoutInfo,
)

# Multi-modal orchestration (Module 08 Phase 3)
from .orchestrator_client import (
    get_orchestrator_client,
    NvidiaOrchestratorClient,
    RuleBasedOrchestratorClient,
    ModalityType,
    ProcessingStrategy,
    ProcessingPlan,
    ProcessingStep,
    FileInfo,
    InputAnalysis,
)

# LLM clients for text extraction
from .ollama_client import OllamaLLMClient
from .local_llm_client import (
    QuantizedLocalLLM,
    LocalLLMBackend,
    get_test_llm_client,
)

# Link Prediction and Knowledge Base Completion (Phase 3)
from .link_prediction import (
    LinkPredictor,
    KnowledgeBaseCompleter,
    PredictedLink,
    PredictionMethod,
    LinkType,
    CompletionCandidate,
)


__all__ = [
    # Unified API
    "extract_from_any_source",
    "extract",
    "process",
    # Audio transcription
    "get_transcription_client",
    "OllamaWhisperClient",
    "LocalWhisperClient",
    "TranscriptionResult",
    "TimestampedSegment",
    # OCR
    "get_ocr_client",
    "DeepSeekOCRClient",
    "TesseractOCRClient",
    "OCRResult",
    "TextRegion",
    "BoundingBox",
    "LayoutInfo",
    # Orchestration
    "get_orchestrator_client",
    "NvidiaOrchestratorClient",
    "RuleBasedOrchestratorClient",
    "ModalityType",
    "ProcessingStrategy",
    "ProcessingPlan",
    "ProcessingStep",
    "FileInfo",
    "InputAnalysis",
    # LLM
    "OllamaLLMClient",
    "QuantizedLocalLLM",
    "LocalLLMBackend",
    "get_test_llm_client",
    # Link Prediction (Phase 3)
    "LinkPredictor",
    "KnowledgeBaseCompleter",
    "PredictedLink",
    "PredictionMethod",
    "LinkType",
    "CompletionCandidate",
]
