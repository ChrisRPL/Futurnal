"""Temporal Enricher - Pipeline Integration for Temporal Extraction.

This module provides the pipeline integration component that enriches
documents with temporal information:
- Temporal marker extraction from text
- Event extraction (with LLM support)
- Relationship detection between events
- Consistency validation

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/01-temporal-extraction.md

Research Foundation:
- Time-R1 (ArXiv 2505.13508v2): Three-stage temporal reasoning
- Temporal KG Extrapolation (IJCAI 2024): Causal subhistory identification

Option B Compliance:
- Temporal-first design (all events have timestamps)
- Consistency validation before Phase 3
- No hardcoded patterns (discoverable via experiential learning)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from futurnal.extraction.temporal.consistency import (
    TemporalConsistencyValidator,
    validate_temporal_consistency,
)
from futurnal.extraction.temporal.markers import TemporalMarkerExtractor
from futurnal.extraction.temporal.models import (
    Event,
    TemporalExtractionResult,
    TemporalMark,
    TemporalRelationship,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client interactions."""

    def extract(self, prompt: str) -> Any:
        """Run extraction on a prompt."""
        ...


class Document(Protocol):
    """Protocol for document structure."""

    content: str
    doc_id: str


class TemporalEnricher:
    """Enrich documents with temporal information.

    This class provides the main pipeline integration point for temporal
    extraction. It orchestrates:
    1. Temporal marker extraction from text
    2. Event extraction (when LLM is available)
    3. Causal relationship detection
    4. Temporal consistency validation

    The enricher can operate in two modes:
    - Lightweight mode (no LLM): Only marker extraction
    - Full mode (with LLM): Complete temporal analysis including events

    Example:
        >>> enricher = TemporalEnricher()
        >>> result = await enricher.enrich_document(document)
        >>> print(f"Found {len(result.temporal_markers)} markers")
        >>> if result.events:
        ...     print(f"Extracted {len(result.events)} events")
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        confidence_threshold: float = 0.7,
        reference_time: Optional[datetime] = None,
        strict_mode: bool = True,
    ):
        """Initialize temporal enricher.

        Args:
            llm_client: Optional LLM client for event/relationship extraction.
                       If None, only marker extraction is performed.
            confidence_threshold: Minimum confidence for including results.
            reference_time: Reference time for relative expressions.
                           Defaults to current time.
            strict_mode: If True, consistency violations are errors.
        """
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.reference_time = reference_time

        # Initialize components
        self.marker_extractor = TemporalMarkerExtractor(
            reference_time=reference_time
        )
        self.consistency_validator = TemporalConsistencyValidator(
            strict_mode=strict_mode
        )

        # Optional components (require LLM)
        self._event_extractor = None
        self._relationship_detector = None

        if llm_client:
            self._init_llm_components(llm_client, confidence_threshold)

    def _init_llm_components(
        self,
        llm_client: LLMClient,
        confidence_threshold: float,
    ) -> None:
        """Initialize LLM-dependent components.

        Args:
            llm_client: LLM client for extraction
            confidence_threshold: Confidence threshold for filtering
        """
        try:
            from futurnal.extraction.causal.event_extractor import EventExtractor
            from futurnal.extraction.causal.relationship_detector import (
                CausalRelationshipDetector,
            )

            self._event_extractor = EventExtractor(
                llm=llm_client,
                marker_extractor=self.marker_extractor,
                confidence_threshold=confidence_threshold,
            )
            self._relationship_detector = CausalRelationshipDetector(
                llm=llm_client,
                confidence_threshold=confidence_threshold,
            )
            logger.info("Initialized LLM-based temporal components")

        except ImportError as e:
            logger.warning(
                f"Could not initialize LLM components: {e}. "
                "Running in lightweight mode (marker extraction only)."
            )

    async def enrich_document(
        self,
        document: Document,
        extract_events: bool = True,
        detect_relationships: bool = True,
        validate_consistency: bool = True,
    ) -> TemporalExtractionResult:
        """Enrich document with temporal information.

        This is the main entry point for temporal enrichment. It performs:
        1. Temporal marker extraction (always)
        2. Event extraction (if LLM available and extract_events=True)
        3. Relationship detection (if events exist and detect_relationships=True)
        4. Consistency validation (if validate_consistency=True)

        Args:
            document: Document to enrich
            extract_events: Whether to extract events (requires LLM)
            detect_relationships: Whether to detect relationships
            validate_consistency: Whether to validate temporal consistency

        Returns:
            TemporalExtractionResult with all extracted temporal information
        """
        logger.info(f"Enriching document {getattr(document, 'doc_id', 'unknown')}")

        # 1. Extract temporal markers from text (always performed)
        markers = self._extract_markers(document)
        logger.debug(f"Extracted {len(markers)} temporal markers")

        # 2. Extract events if LLM is available
        events: List[Event] = []
        if extract_events and self._event_extractor:
            events = self._extract_events(document, markers)
            logger.debug(f"Extracted {len(events)} events")

        # 3. Detect causal relationships if events exist
        relationships: List[TemporalRelationship] = []
        if detect_relationships and events and self._relationship_detector:
            relationships = self._detect_relationships(events, document)
            logger.debug(f"Detected {len(relationships)} relationships")

        # 4. Validate temporal consistency
        validation_result: Optional[ValidationResult] = None
        if validate_consistency and (events or relationships):
            validation_result = self._validate_consistency(events, relationships)
            if not validation_result.valid:
                logger.warning(
                    f"Temporal inconsistencies found: {len(validation_result.errors)} errors"
                )

        # Build result
        result = TemporalExtractionResult(
            temporal_markers=markers,
            temporal_relationships=(
                validation_result.relationships if validation_result else relationships
            ),
            events=events,
            temporal_triples=[],  # Converted to triples in PKG storage phase
            summary=self._generate_summary(markers, events, relationships),
        )

        logger.info(
            f"Temporal enrichment complete: "
            f"{len(markers)} markers, {len(events)} events, "
            f"{len(relationships)} relationships"
        )

        return result

    def _extract_markers(self, document: Document) -> List[TemporalMark]:
        """Extract temporal markers from document text.

        Args:
            document: Document to extract from

        Returns:
            List of extracted temporal markers
        """
        try:
            # Get document metadata if available
            doc_metadata = None
            if hasattr(document, 'metadata'):
                metadata = document.metadata
                if hasattr(metadata, 'model_dump'):
                    doc_metadata = metadata.model_dump()
                elif isinstance(metadata, dict):
                    doc_metadata = metadata

            markers = self.marker_extractor.extract_temporal_markers(
                document.content,
                doc_metadata=doc_metadata,
            )

            # Filter by confidence threshold
            return [m for m in markers if m.confidence >= self.confidence_threshold]

        except Exception as e:
            logger.error(f"Error extracting temporal markers: {e}")
            return []

    def _extract_events(
        self,
        document: Document,
        markers: List[TemporalMark],
    ) -> List[Event]:
        """Extract events from document using LLM.

        Args:
            document: Document to extract from
            markers: Already extracted temporal markers

        Returns:
            List of extracted events
        """
        if not self._event_extractor:
            return []

        try:
            events = self._event_extractor.extract_events(document)

            # Filter by confidence threshold
            return [e for e in events if e.extraction_confidence >= self.confidence_threshold]

        except Exception as e:
            logger.error(f"Error extracting events: {e}")
            return []

    def _detect_relationships(
        self,
        events: List[Event],
        document: Document,
    ) -> List[TemporalRelationship]:
        """Detect causal relationships between events.

        Args:
            events: List of extracted events
            document: Source document

        Returns:
            List of detected relationships
        """
        if not self._relationship_detector or len(events) < 2:
            return []

        try:
            candidates = self._relationship_detector.detect_causal_candidates(
                events, document
            )

            # Convert CausalCandidate to TemporalRelationship
            relationships = []
            for candidate in candidates:
                relationships.append(TemporalRelationship(
                    entity1_id=candidate.cause_event_id,
                    entity2_id=candidate.effect_event_id,
                    relationship_type=self._map_causal_type(candidate.relationship_type),
                    confidence=candidate.causal_confidence,
                    evidence=candidate.causal_evidence,
                ))

            return relationships

        except Exception as e:
            logger.error(f"Error detecting relationships: {e}")
            return []

    def _map_causal_type(self, causal_type) -> Any:
        """Map CausalRelationshipType to TemporalRelationshipType.

        Args:
            causal_type: CausalRelationshipType enum value

        Returns:
            Corresponding TemporalRelationshipType
        """
        from futurnal.extraction.temporal.models import TemporalRelationshipType

        mapping = {
            "causes": TemporalRelationshipType.CAUSES,
            "enables": TemporalRelationshipType.ENABLES,
            "prevents": TemporalRelationshipType.PREVENTS,
            "triggers": TemporalRelationshipType.TRIGGERS,
            "leads_to": TemporalRelationshipType.CAUSES,
            "contributes_to": TemporalRelationshipType.ENABLES,
        }

        type_value = causal_type.value if hasattr(causal_type, 'value') else str(causal_type)
        return mapping.get(type_value, TemporalRelationshipType.CAUSES)

    def _validate_consistency(
        self,
        events: List[Event],
        relationships: List[TemporalRelationship],
    ) -> ValidationResult:
        """Validate temporal consistency.

        Args:
            events: List of events
            relationships: List of relationships

        Returns:
            ValidationResult with validity status and any errors
        """
        return self.consistency_validator.validate(events, relationships)

    def _generate_summary(
        self,
        markers: List[TemporalMark],
        events: List[Event],
        relationships: List[TemporalRelationship],
    ) -> str:
        """Generate human-readable summary of extraction.

        Args:
            markers: Extracted temporal markers
            events: Extracted events
            relationships: Detected relationships

        Returns:
            Summary string
        """
        parts = []

        if markers:
            parts.append(f"Found {len(markers)} temporal markers")

        if events:
            parts.append(f"{len(events)} events extracted")

        if relationships:
            parts.append(f"{len(relationships)} relationships detected")

        if not parts:
            return "No temporal information extracted"

        return "; ".join(parts)


# Convenience function for simple enrichment
async def enrich_with_temporal(
    document: Document,
    llm_client: Optional[LLMClient] = None,
    confidence_threshold: float = 0.7,
) -> TemporalExtractionResult:
    """Convenience function for temporal enrichment.

    Args:
        document: Document to enrich
        llm_client: Optional LLM client
        confidence_threshold: Confidence threshold

    Returns:
        TemporalExtractionResult
    """
    enricher = TemporalEnricher(
        llm_client=llm_client,
        confidence_threshold=confidence_threshold,
    )
    return await enricher.enrich_document(document)
