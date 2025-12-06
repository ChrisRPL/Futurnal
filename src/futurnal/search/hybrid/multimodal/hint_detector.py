"""Modality Hint Detector for query analysis.

Detects modality hints in user queries to route them to
appropriate content sources (OCR documents, audio transcriptions, etc.).

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/07-multimodal-query-handling.md

Examples:
- "in my voice notes" -> AUDIO_TRANSCRIPTION (0.95)
- "from the scanned document" -> OCR_DOCUMENT (0.95)
- "what I said in the meeting" -> AUDIO_TRANSCRIPTION (0.85)
- "text from that image" -> OCR_IMAGE (0.90)

Option B Compliance:
- Ghost model frozen (pattern-based detection, no LLM calls)
- Local-first processing
- Quality target: >90% hint accuracy
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from futurnal.search.hybrid.multimodal.types import (
    ContentSource,
    ModalityHint,
)


class ModalityHintDetector:
    """Detects modality hints in user queries.

    Uses regex patterns to identify phrases indicating the user
    wants to search content from a specific modality.

    Pattern Confidence Levels:
    - 0.95: Explicit source reference ("from the scanned PDF")
    - 0.90: Strong source indicator ("in my voice notes")
    - 0.85: Contextual indicator ("what I said in the meeting")
    - 0.80: Weak indicator ("recording", "document")

    Integration Points:
    - QueryRouter: Provides modality hints for routing decisions
    - MultimodalQueryHandler: Configures source filtering

    Attributes:
        MODALITY_PATTERNS: Pattern definitions per content source
    """

    # Pattern definitions: (regex_pattern, base_confidence)
    MODALITY_PATTERNS: Dict[ContentSource, List[Tuple[str, float]]] = {
        ContentSource.AUDIO_TRANSCRIPTION: [
            # Explicit voice/recording references with preposition
            (
                r"(?:in|from)\s+(?:my|the)\s+(?:voice\s+)?(?:notes?|memos?|recordings?)",
                0.95,
            ),
            # Voice notes/memos without preposition - requires "voice" for notes/memos
            (r"\bmy\s+voice\s+(?:notes?|memos?)\b", 0.90),
            # My recordings without preposition (recordings are audio-specific)
            (r"\bmy\s+recordings?\b", 0.90),
            # The recordings (e.g., "search the recordings")
            (r"\b(?:the|my)\s+recordings?\b", 0.85),
            # Action-based indicators with flexible verb structure
            (r"\bwhat\s+(?:\w+\s+)?(?:I|we)\s+(?:say|said|mentioned|discussed)\b", 0.85),
            (r"\bwhen\s+(?:\w+\s+)?(?:I|we)\s+(?:say|said|mentioned|discussed)\b", 0.85),
            # Meeting/call context
            (
                r"(?:in|during|from)\s+(?:the|that|my)\s+"
                r"(?:meeting|call|conversation|interview)",
                0.90,
            ),
            # General audio terms
            (r"\b(?:audio|recording|podcast|episode)\b", 0.80),
            # Speech-based indicators
            (r"(?:I|we)\s+(?:talked|spoke)\s+about", 0.85),
            # Transcription mention
            (r"\btranscri(?:pt|ption|bed)\b", 0.90),
        ],
        ContentSource.OCR_DOCUMENT: [
            # Explicit scanned document reference
            (
                r"(?:in|from)\s+(?:the|that)\s+(?:scanned|uploaded)\s+"
                r"(?:document|pdf|page)",
                0.95,
            ),
            # Scanned PDF without preposition (e.g., "check the scanned pdf")
            (r"\b(?:the|that)\s+scanned\s+(?:pdf|document)\b", 0.90),
            # PDF/scan reference with preposition
            (r"(?:in|from)\s+(?:the|that)\s+(?:pdf|scan)\b", 0.90),
            # PDF without preposition (e.g., "check the pdf")
            (r"\b(?:the|that)\s+pdf\b", 0.85),
            # Handwritten/printed notes
            (r"(?:handwritten|printed)\s+(?:notes?|document)", 0.85),
            # Physical document types
            (r"\b(?:the|that)\s+(?:paper|letter|form|receipt|invoice)\b", 0.80),
            # Scanning action
            (r"\b(?:scanned|digitized)\b", 0.85),
        ],
        ContentSource.OCR_IMAGE: [
            # Explicit image reference
            (
                r"(?:in|from)\s+(?:the|that)\s+(?:image|photo|picture|screenshot)",
                0.90,
            ),
            # Text in image
            (
                r"(?:text|words)\s+(?:in|on|from)\s+(?:the|that)\s+"
                r"(?:image|photo|picture)",
                0.95,
            ),
            # Visual context with text
            (r"\b(?:whiteboard|blackboard|sign|poster|banner)\b", 0.85),
            # Screenshot reference
            (r"\b(?:screenshot|screen\s*shot|capture)\b", 0.90),
        ],
        ContentSource.VIDEO_TRANSCRIPTION: [
            # Explicit video reference
            (r"(?:in|from)\s+(?:the|that)\s+video", 0.90),
            # Video platform mentions
            (r"\b(?:youtube|vimeo|video\s+recording)\b", 0.85),
            # Viewing action
            (r"(?:watched|viewed)\s+(?:the\s+)?(?:video|clip)", 0.80),
        ],
    }

    def __init__(self) -> None:
        """Initialize detector with compiled patterns."""
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(
        self,
    ) -> Dict[ContentSource, List[Tuple[re.Pattern, float]]]:
        """Pre-compile regex patterns for efficiency.

        Returns:
            Dictionary mapping content sources to compiled patterns
        """
        compiled: Dict[ContentSource, List[Tuple[re.Pattern, float]]] = {}

        for source, patterns in self.MODALITY_PATTERNS.items():
            compiled[source] = [
                (re.compile(pattern, re.IGNORECASE), conf)
                for pattern, conf in patterns
            ]

        return compiled

    def detect(self, query: str) -> List[ModalityHint]:
        """Detect modality hints in query.

        Scans the query for patterns indicating specific content sources.
        Returns all detected hints sorted by confidence.

        Args:
            query: User search query

        Returns:
            List of detected hints, sorted by confidence descending

        Example:
            >>> detector = ModalityHintDetector()
            >>> hints = detector.detect("what did I say in my voice notes about budget?")
            >>> hints[0].modality
            ContentSource.AUDIO_TRANSCRIPTION
            >>> hints[0].confidence
            0.95
        """
        hints: List[ModalityHint] = []

        for source, patterns in self._compiled_patterns.items():
            for pattern, base_confidence in patterns:
                matches = pattern.finditer(query)
                for match in matches:
                    hints.append(
                        ModalityHint(
                            modality=source,
                            confidence=base_confidence,
                            hint_phrase=match.group(0),
                            query_position=(match.start(), match.end()),
                        )
                    )

        # Sort by confidence descending, then by position
        hints.sort(key=lambda h: (-h.confidence, h.query_position[0]))

        return hints

    def get_primary_modality(self, query: str) -> Optional[ContentSource]:
        """Get the most likely intended modality, if any.

        Returns the modality from the highest-confidence hint,
        but only if confidence >= 0.80 (strong indicator).

        Args:
            query: User search query

        Returns:
            Most likely content source, or None if no strong hint

        Example:
            >>> detector = ModalityHintDetector()
            >>> detector.get_primary_modality("check my voice recording")
            ContentSource.AUDIO_TRANSCRIPTION
            >>> detector.get_primary_modality("find project notes")
            None
        """
        hints = self.detect(query)
        if hints and hints[0].confidence >= 0.80:
            return hints[0].modality
        return None

    def should_filter_by_modality(self, query: str) -> bool:
        """Determine if query should filter by modality.

        Returns True if there's a confident enough hint to
        justify filtering search to specific content sources.
        Threshold is 0.75 to include moderate hints.

        Args:
            query: User search query

        Returns:
            True if modality filtering is recommended
        """
        hints = self.detect(query)
        return len(hints) > 0 and hints[0].confidence >= 0.75

    def get_all_modalities(self, query: str) -> List[ContentSource]:
        """Get all detected modalities with confidence >= 0.75.

        Useful for queries that might reference multiple modalities.

        Args:
            query: User search query

        Returns:
            List of content sources with confidence >= 0.75
        """
        hints = self.detect(query)
        # Deduplicate while preserving order
        seen: set = set()
        modalities: List[ContentSource] = []
        for hint in hints:
            if hint.confidence >= 0.75 and hint.modality not in seen:
                seen.add(hint.modality)
                modalities.append(hint.modality)
        return modalities

    def extract_query_without_hints(self, query: str) -> str:
        """Remove modality hints from query for cleaner search.

        Removes detected hint phrases to get the core search intent.
        Useful for semantic search where modality context adds noise.
        Handles overlapping matches by merging regions before removal.

        Args:
            query: User search query

        Returns:
            Query with high-confidence hint phrases removed

        Example:
            >>> detector = ModalityHintDetector()
            >>> detector.extract_query_without_hints("in my voice notes about budget")
            "about budget"
        """
        hints = self.detect(query)
        if not hints:
            return query

        # Collect high-confidence regions to remove
        regions_to_remove: List[Tuple[int, int]] = []
        for hint in hints:
            if hint.confidence >= 0.85:
                regions_to_remove.append(hint.query_position)

        if not regions_to_remove:
            return query

        # Merge overlapping regions
        regions_to_remove.sort(key=lambda r: r[0])
        merged: List[Tuple[int, int]] = []
        for start, end in regions_to_remove:
            if merged and start <= merged[-1][1]:
                # Overlaps with previous - extend it
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Remove regions from end to start to preserve positions
        result = query
        for start, end in reversed(merged):
            result = result[:start] + result[end:]

        # Clean up whitespace
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def get_confidence_summary(self, query: str) -> Dict[ContentSource, float]:
        """Get confidence summary for all detected modalities.

        Returns maximum confidence per modality, useful for
        building weighted query plans.

        Args:
            query: User search query

        Returns:
            Dictionary mapping content sources to max confidence
        """
        hints = self.detect(query)
        summary: Dict[ContentSource, float] = {}

        for hint in hints:
            if hint.modality not in summary:
                summary[hint.modality] = hint.confidence
            else:
                summary[hint.modality] = max(summary[hint.modality], hint.confidence)

        return summary
