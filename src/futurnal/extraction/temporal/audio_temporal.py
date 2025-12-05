"""Audio temporal extraction for converting Whisper segments to temporal markers.

This module integrates audio transcription with the temporal extraction pipeline,
converting time-stamped Whisper segments into TemporalMark instances for
entity-relationship extraction.

Module 08: Multimodal Integration & Tool Enhancement - Phase 1
Production Plan: docs/phase-1/entity-relationship-extraction-production-plan/08-multimodal-integration.md
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .models import TemporalMark, TemporalRelationshipType, TemporalSourceType, TemporalSource

logger = logging.getLogger(__name__)


class AudioTemporalExtractor:
    """Extract temporal markers from audio transcription segments.

    Converts Whisper timestamped segments into TemporalMark instances,
    enabling temporal-first entity extraction from audio sources.

    Key Features:
    - Segment-level temporal grounding (second-precision)
    - Sequential temporal relationships (BEFORE/AFTER)
    - Confidence propagation from transcription
    - Base timestamp alignment for absolute time anchoring

    Example:
        >>> extractor = AudioTemporalExtractor()
        >>> segments = [
        ...     {"text": "Meeting started at 9 AM", "start": 0.0, "end": 2.5, "confidence": 0.95},
        ...     {"text": "Discussed quarterly goals", "start": 2.5, "end": 45.0, "confidence": 0.98}
        ... ]
        >>> base_time = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        >>> markers = extractor.extract_from_audio_segments(segments, base_time)
        >>> print(f"Extracted {len(markers)} temporal markers")
    """

    def extract_from_audio_segments(
        self,
        segments: List[Dict],
        base_timestamp: Optional[datetime] = None,
        source_type: str = "audio_transcription"
    ) -> List[TemporalMark]:
        """Convert Whisper segments to temporal markers.

        Args:
            segments: List of segment dictionaries with keys:
                - text: Transcribed text for segment
                - start: Start time in seconds (float)
                - end: End time in seconds (float)
                - confidence: Transcription confidence (0.0-1.0)
            base_timestamp: Optional base timestamp for absolute time anchoring
                (e.g., recording start time). If None, uses current time.
            source_type: Source type identifier for provenance tracking

        Returns:
            List of TemporalMark instances with temporal grounding

        Example:
            >>> segments = [
            ...     {"text": "Introduction", "start": 0.0, "end": 5.2, "confidence": 0.95}
            ... ]
            >>> markers = extractor.extract_from_audio_segments(segments)
        """
        if not segments:
            logger.warning("No audio segments provided for temporal extraction")
            return []

        # Use current time if no base timestamp provided
        if base_timestamp is None:
            base_timestamp = datetime.now(timezone.utc)
            logger.debug(f"No base timestamp provided, using current time: {base_timestamp}")

        markers: List[TemporalMark] = []

        for idx, seg in enumerate(segments):
            # Calculate absolute timestamp for this segment
            segment_start_seconds = seg.get("start", 0.0)
            segment_timestamp = base_timestamp + timedelta(seconds=segment_start_seconds)

            # Create temporal marker for this segment
            marker = TemporalMark(
                text=seg.get("text", ""),
                timestamp=segment_timestamp,
                temporal_type=TemporalSourceType.EXPLICIT,  # Audio timestamps are explicit
                confidence=seg.get("confidence", 1.0),
                span_start=int(segment_start_seconds),      # Store as character offset equivalent
                span_end=int(seg.get("end", 0.0))
            )

            markers.append(marker)

            logger.debug(
                f"Audio segment {idx}: [{segment_start_seconds:.2f}s - {seg.get('end', 0.0):.2f}s] "
                f"'{seg.get('text', '')[:50]}...' → {segment_timestamp}"
            )

        logger.info(
            f"Extracted {len(markers)} temporal markers from audio segments "
            f"(base: {base_timestamp}, duration: {segments[-1].get('end', 0.0):.1f}s)"
        )

        return markers

    def extract_temporal_relationships(
        self,
        segments: List[Dict],
        base_timestamp: Optional[datetime] = None
    ) -> List[Dict]:
        """Extract temporal relationships between consecutive segments.

        Creates BEFORE/AFTER relationships between sequential audio segments,
        enabling temporal graph construction.

        Args:
            segments: List of segment dictionaries (same format as extract_from_audio_segments)
            base_timestamp: Optional base timestamp for absolute time anchoring

        Returns:
            List of temporal relationship dictionaries with keys:
                - source_segment: Index of source segment
                - target_segment: Index of target segment
                - relationship_type: TemporalRelationshipType value
                - confidence: Relationship confidence (0.0-1.0)

        Example:
            >>> relationships = extractor.extract_temporal_relationships(segments)
            >>> print(f"Segment 0 BEFORE Segment 1: {relationships[0]['relationship_type']}")
        """
        if len(segments) < 2:
            logger.debug("Less than 2 segments, no relationships to extract")
            return []

        relationships = []

        for idx in range(len(segments) - 1):
            current_seg = segments[idx]
            next_seg = segments[idx + 1]

            # Determine relationship type based on timing
            current_end = current_seg.get("end", 0.0)
            next_start = next_seg.get("start", 0.0)

            if abs(current_end - next_start) < 0.1:  # Within 100ms = MEETS
                rel_type = TemporalRelationshipType.MEETS
            elif current_end < next_start:  # Gap between segments = BEFORE
                rel_type = TemporalRelationshipType.BEFORE
            else:  # Overlap = OVERLAPS
                rel_type = TemporalRelationshipType.OVERLAPS

            # Relationship confidence is minimum of both segments
            confidence = min(
                current_seg.get("confidence", 1.0),
                next_seg.get("confidence", 1.0)
            )

            relationship = {
                "source_segment": idx,
                "target_segment": idx + 1,
                "relationship_type": rel_type.value,
                "confidence": confidence,
                "source_text": current_seg.get("text", ""),
                "target_text": next_seg.get("text", "")
            }

            relationships.append(relationship)

        logger.info(f"Extracted {len(relationships)} temporal relationships from {len(segments)} segments")

        return relationships

    def create_temporal_source(
        self,
        segment: Dict,
        source_file: Optional[str] = None
    ) -> TemporalSource:
        """Create TemporalSource for provenance tracking.

        Args:
            segment: Segment dictionary with text and timing
            source_file: Optional source audio file path

        Returns:
            TemporalSource instance for provenance tracking
        """
        evidence = f"Audio segment: '{segment.get('text', '')[:100]}'"
        if source_file:
            evidence = f"{source_file} | {evidence}"

        return TemporalSource(
            source_type=TemporalSourceType.EXPLICIT,
            evidence=evidence,
            inference_method="whisper_v3_transcription"
        )

    def segment_to_datetime(
        self,
        segment: Dict,
        base_timestamp: Optional[datetime] = None
    ) -> datetime:
        """Convert segment start time to datetime.

        Args:
            segment: Segment dictionary with 'start' key (seconds)
            base_timestamp: Base timestamp to offset from

        Returns:
            datetime representing absolute time of segment start
        """
        if base_timestamp is None:
            base_timestamp = datetime.now(timezone.utc)

        segment_start_seconds = segment.get("start", 0.0)
        return base_timestamp + timedelta(seconds=segment_start_seconds)
