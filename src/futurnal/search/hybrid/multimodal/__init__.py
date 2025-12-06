"""Multimodal Query Handling for Hybrid Search API.

Module 07: Enables hybrid search across multimodal content with
source-aware retrieval for OCR-extracted documents, audio transcriptions,
and mixed-source queries.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Components:
- ContentSource: Content modality types
- SourceMetadata: Extraction metadata for retrieval optimization
- ModalityHintDetector: Query analysis for modality hints
- MultimodalQueryHandler: Execution orchestration
- OCRContentProcessor: OCR integration
- TranscriptionProcessor: Audio integration
- CrossModalFusion: Result combination

Option B Compliance:
- Ghost model frozen (OCR/Whisper for extraction only)
- Local-first processing
- Quality gates: OCR >80%, Audio >75% relevance
"""

from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    ExtractionQuality,
    ModalityHint,
    MultimodalQueryPlan,
    RetrievalMode,
    SourceMetadata,
)
from futurnal.search.hybrid.multimodal.hint_detector import ModalityHintDetector
from futurnal.search.hybrid.multimodal.ocr_processor import (
    OCRContentProcessor,
    OCRLayoutType,
)
from futurnal.search.hybrid.multimodal.transcription_processor import (
    TranscriptionProcessor,
    AudioQuality,
)
from futurnal.search.hybrid.multimodal.handler import (
    MultimodalQueryHandler,
    MultimodalSearchResult,
    create_multimodal_handler,
)
from futurnal.search.hybrid.multimodal.fusion import (
    CrossModalFusion,
    FusionConfig,
    FusedResult,
    create_fusion_config,
)

__all__ = [
    # Core types
    "ContentSource",
    "ExtractionQuality",
    "SourceMetadata",
    "ModalityHint",
    # Query planning
    "RetrievalMode",
    "MultimodalQueryPlan",
    # Detection
    "ModalityHintDetector",
    # OCR processing
    "OCRContentProcessor",
    "OCRLayoutType",
    # Transcription processing
    "TranscriptionProcessor",
    "AudioQuality",
    # Query handling
    "MultimodalQueryHandler",
    "MultimodalSearchResult",
    "create_multimodal_handler",
    # Cross-modal fusion
    "CrossModalFusion",
    "FusionConfig",
    "FusedResult",
    "create_fusion_config",
]
