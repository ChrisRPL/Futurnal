"""Unified extraction API for all modalities.

Module 08: Multimodal Integration & Tool Enhancement - Phase 4 (Integration & Polish)
Production Plan: docs/phase-1/entity-relationship-extraction-production-plan/08-multimodal-integration.md

Provides a single entry point for extracting content from any source:
- Text files (.md, .txt, .py, etc.)
- Audio files (.mp3, .wav, .m4a, etc.) - via Whisper V3
- Images (.png, .jpg, etc.) - via DeepSeek-OCR
- PDFs (.pdf) - including scanned documents via OCR
- Mixed batches - intelligently orchestrated

Example:
    >>> from futurnal.extraction import extract_from_any_source
    >>>
    >>> # Single file
    >>> result = await extract_from_any_source("meeting_notes.md")
    >>>
    >>> # Audio transcription
    >>> result = await extract_from_any_source("recording.wav")
    >>>
    >>> # Image OCR
    >>> result = await extract_from_any_source("document.pdf")
    >>>
    >>> # Mixed batch (Orchestrator decides optimal strategy)
    >>> results = await extract_from_any_source([
    ...     "slides.pdf",
    ...     "recording.wav",
    ...     "notes.md"
    ... ])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from futurnal.pipeline.models import NormalizedDocument
from futurnal.pipeline.multimodal.router import MultiModalRouter, process_files
from futurnal.pipeline.normalization.registry import FormatAdapterRegistry

logger = logging.getLogger(__name__)


async def extract_from_any_source(
    source: Union[str, Path, List[Union[str, Path]]],
    source_id: str = "unified",
    source_type: str = "local_files",
    source_metadata: Optional[Dict] = None,
    strategy: Optional[str] = None,
) -> Union[NormalizedDocument, List[NormalizedDocument]]:
    """Extract content from any source with automatic modality detection.

    This is the unified entry point for all extraction operations as specified in
    Module 08: Multimodal Integration production plan. It automatically:
    - Detects file modality (text, audio, image, PDF)
    - Routes to appropriate extraction backend (Whisper, OCR, text parser)
    - Orchestrates multi-file batches optimally
    - Returns normalized documents ready for entity extraction

    Args:
        source: File path(s) to process. Can be:
            - Single string path: "meeting.mp3"
            - Single Path object: Path("meeting.mp3")
            - List of paths: ["slides.pdf", "recording.wav", "notes.md"]
        source_id: Identifier for the source (default: "unified")
        source_type: Type of source (default: "local_files")
        source_metadata: Optional metadata dict to pass through
        strategy: Processing strategy for multi-file batches:
            - "auto": Let orchestrator decide (default)
            - "parallel": Process all files concurrently
            - "sequential": Process files one-by-one
            - "dependency_graph": Respect file dependencies

    Returns:
        Single NormalizedDocument if input is single file,
        List[NormalizedDocument] if input is multiple files.

        Each document contains:
        - content: Extracted text content
        - metadata: Rich metadata including source info, timestamps, etc.
        - metadata.extra: Modality-specific data:
            - Audio: temporal_segments, audio_duration_seconds
            - Image: ocr_regions, layout_info
            - Scanned PDF: pages, page_count

    Raises:
        FileNotFoundError: If source file does not exist
        AdapterError: If extraction fails for a file

    Example:
        >>> # Text file
        >>> doc = await extract_from_any_source("notes.md")
        >>> print(doc.content[:100])

        >>> # Audio with temporal segments
        >>> doc = await extract_from_any_source("meeting.wav")
        >>> segments = doc.metadata.extra.get("temporal_segments", [])
        >>> for seg in segments:
        ...     print(f"[{seg['start']:.1f}s] {seg['text']}")

        >>> # Mixed batch
        >>> docs = await extract_from_any_source([
        ...     "slides.pdf",
        ...     "recording.wav",
        ...     "notes.md"
        ... ], strategy="parallel")
        >>> print(f"Processed {len(docs)} files")
    """
    # Normalize input to Path objects
    if isinstance(source, str):
        paths = Path(source)
    elif isinstance(source, Path):
        paths = source
    elif isinstance(source, list):
        paths = [Path(s) if isinstance(s, str) else s for s in source]
    else:
        raise TypeError(
            f"source must be str, Path, or List[str|Path], got {type(source)}"
        )

    # Validate files exist (single file or all files in list)
    if isinstance(paths, Path):
        if not paths.exists():
            raise FileNotFoundError(f"Source file not found: {paths}")
    else:
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Source file not found: {p}")

    # Initialize router with default registry (includes multimodal adapters)
    registry = FormatAdapterRegistry()
    registry.register_default_adapters()

    router = MultiModalRouter(adapter_registry=registry)

    logger.info(
        f"Extracting from {'single file' if isinstance(paths, Path) else f'{len(paths)} files'} "
        f"with strategy={strategy or 'auto'}"
    )

    # Process through multimodal router
    result = await router.process(
        files=paths,
        source_id=source_id,
        source_type=source_type,
        source_metadata=source_metadata or {},
        strategy=strategy,
    )

    return result


# Convenience aliases
extract = extract_from_any_source
process = extract_from_any_source


__all__ = [
    "extract_from_any_source",
    "extract",
    "process",
]
