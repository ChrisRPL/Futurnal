"""Multi-modal coordination for cross-modal entity linking and result merging.

Module 08: Multimodal Integration & Tool Enhancement - Phase 3
Coordinates results across modalities for unified knowledge graph construction.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from futurnal.pipeline.models import NormalizedDocument, DocumentFormat

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CrossModalEntity:
    """Entity detected across multiple modalities."""
    entity_text: str
    entity_type: str  # "person", "organization", "location", etc.
    modalities: Set[DocumentFormat] = field(default_factory=set)
    occurrences: List[Dict] = field(default_factory=list)  # [{source, position, confidence}]
    confidence: float = 0.0


@dataclass
class TemporalAlignment:
    """Temporal alignment between modalities."""
    audio_timestamp: float  # seconds
    document_position: int  # character offset or page number
    alignment_type: str  # "explicit", "inferred", "synchronized"
    confidence: float = 0.0


@dataclass
class CoordinatedResult:
    """Coordinated result from multi-modal processing."""
    documents: List[NormalizedDocument]
    cross_modal_entities: List[CrossModalEntity] = field(default_factory=list)
    temporal_alignments: List[TemporalAlignment] = field(default_factory=list)
    modality_summary: Dict[DocumentFormat, int] = field(default_factory=dict)
    merged_metadata: Dict = field(default_factory=dict)


# =============================================================================
# Multi-Modal Coordinator
# =============================================================================


class MultiModalCoordinator:
    """Coordinates multi-modal processing results.

    Features:
    - Cross-modal entity linking
    - Temporal synchronization
    - Result merging and aggregation
    - Unified metadata generation
    """

    def __init__(self, enable_entity_linking: bool = True):
        """Initialize multi-modal coordinator.

        Args:
            enable_entity_linking: Whether to perform cross-modal entity linking
        """
        self.enable_entity_linking = enable_entity_linking
        logger.info(
            f"Initialized MultiModalCoordinator "
            f"(entity linking: {enable_entity_linking})"
        )

    def coordinate(
        self,
        documents: List[NormalizedDocument],
        link_entities: Optional[bool] = None
    ) -> CoordinatedResult:
        """Coordinate results from multiple modalities.

        Args:
            documents: List of NormalizedDocuments from different modalities
            link_entities: Override for entity linking (default: use init setting)

        Returns:
            CoordinatedResult with linked entities and alignments
        """
        if not documents:
            logger.warning("No documents to coordinate")
            return CoordinatedResult(documents=[])

        logger.info(f"Coordinating {len(documents)} documents across modalities")

        # Analyze modality distribution
        modality_summary = self._analyze_modalities(documents)
        logger.info(f"Modalities: {dict(modality_summary)}")

        # Extract cross-modal entities if enabled
        cross_modal_entities = []
        if (link_entities if link_entities is not None else self.enable_entity_linking):
            cross_modal_entities = self._extract_cross_modal_entities(documents)
            logger.info(f"Found {len(cross_modal_entities)} cross-modal entities")

        # Detect temporal alignments
        temporal_alignments = self._detect_temporal_alignments(documents)
        logger.info(f"Found {len(temporal_alignments)} temporal alignments")

        # Merge metadata
        merged_metadata = self._merge_metadata(documents)

        return CoordinatedResult(
            documents=documents,
            cross_modal_entities=cross_modal_entities,
            temporal_alignments=temporal_alignments,
            modality_summary=modality_summary,
            merged_metadata=merged_metadata
        )

    def _analyze_modalities(
        self,
        documents: List[NormalizedDocument]
    ) -> Dict[DocumentFormat, int]:
        """Analyze distribution of modalities.

        Args:
            documents: List of documents

        Returns:
            Dict mapping DocumentFormat to count
        """
        modality_counts = {}
        for doc in documents:
            format = doc.metadata.format
            modality_counts[format] = modality_counts.get(format, 0) + 1

        return modality_counts

    def _extract_cross_modal_entities(
        self,
        documents: List[NormalizedDocument]
    ) -> List[CrossModalEntity]:
        """Extract entities that appear across multiple modalities.

        Args:
            documents: List of documents

        Returns:
            List of CrossModalEntity instances
        """
        # Simple implementation: look for common named entities
        # Future: use proper NER and entity linking

        # Extract entities from each document
        document_entities = []
        for doc in documents:
            entities = self._extract_entities_from_document(doc)
            document_entities.append((doc, entities))

        # Find entities appearing in multiple modalities
        entity_map: Dict[str, CrossModalEntity] = {}

        for doc, entities in document_entities:
            for entity_text, entity_type in entities:
                # Normalize entity text
                normalized_text = entity_text.strip().lower()

                if normalized_text not in entity_map:
                    entity_map[normalized_text] = CrossModalEntity(
                        entity_text=entity_text,
                        entity_type=entity_type,
                        modalities=set(),
                        occurrences=[],
                        confidence=0.0
                    )

                cross_entity = entity_map[normalized_text]
                cross_entity.modalities.add(doc.metadata.format)
                cross_entity.occurrences.append({
                    "source_id": doc.metadata.source_id,
                    "format": doc.metadata.format.value,
                    "position": 0,  # Future: track actual position
                    "confidence": 1.0  # Future: use NER confidence
                })

        # Filter to only cross-modal entities (appearing in 2+ modalities)
        cross_modal_entities = [
            entity for entity in entity_map.values()
            if len(entity.modalities) >= 2
        ]

        # Calculate confidence as average of occurrence confidences
        for entity in cross_modal_entities:
            if entity.occurrences:
                entity.confidence = sum(
                    occ["confidence"] for occ in entity.occurrences
                ) / len(entity.occurrences)

        return cross_modal_entities

    def _extract_entities_from_document(
        self,
        document: NormalizedDocument
    ) -> List[Tuple[str, str]]:
        """Extract named entities from document content.

        Args:
            document: NormalizedDocument

        Returns:
            List of (entity_text, entity_type) tuples
        """
        # Simple heuristic-based entity extraction
        # Future: use proper NER model

        entities = []

        # Capitalized words (potential proper nouns)
        words = document.content.split()
        for word in words:
            # Simple heuristic: capitalized words that aren't at sentence start
            if word and word[0].isupper() and len(word) > 3:
                # Filter common words
                if word.lower() not in {"the", "this", "that", "these", "those"}:
                    entities.append((word, "UNKNOWN"))

        # Extract from metadata if available
        if "entities" in document.metadata.extra:
            for entity in document.metadata.extra["entities"]:
                entities.append((entity.get("text", ""), entity.get("type", "UNKNOWN")))

        return entities

    def _detect_temporal_alignments(
        self,
        documents: List[NormalizedDocument]
    ) -> List[TemporalAlignment]:
        """Detect temporal alignments between modalities.

        Args:
            documents: List of documents

        Returns:
            List of TemporalAlignment instances
        """
        alignments = []

        # Find audio documents with temporal segments
        audio_docs = [
            doc for doc in documents
            if doc.metadata.format == DocumentFormat.AUDIO
        ]

        # Find other documents that might have temporal markers
        other_docs = [
            doc for doc in documents
            if doc.metadata.format != DocumentFormat.AUDIO
        ]

        # Simple heuristic: align audio segments with document sections
        for audio_doc in audio_docs:
            if "temporal_segments" not in audio_doc.metadata.extra:
                continue

            segments = audio_doc.metadata.extra["temporal_segments"]

            for segment in segments:
                # Check if segment mentions document names or sections
                segment_text = segment.get("text", "").lower()

                for other_doc in other_docs:
                    doc_name = Path(other_doc.metadata.source_path).stem.lower()

                    if doc_name in segment_text:
                        # Found potential alignment
                        alignment = TemporalAlignment(
                            audio_timestamp=segment.get("start", 0),
                            document_position=0,  # Future: find specific section
                            alignment_type="inferred",
                            confidence=0.7  # Medium confidence for name-based inference
                        )
                        alignments.append(alignment)

        logger.debug(f"Detected {len(alignments)} temporal alignments")
        return alignments

    def _merge_metadata(self, documents: List[NormalizedDocument]) -> Dict:
        """Merge metadata from multiple documents.

        Args:
            documents: List of documents

        Returns:
            Merged metadata dictionary
        """
        merged = {
            "document_count": len(documents),
            "total_characters": sum(doc.metadata.character_count for doc in documents),
            "total_words": sum(doc.metadata.word_count for doc in documents),
            "formats": list(set(doc.metadata.format.value for doc in documents)),
            "source_ids": [doc.metadata.source_id for doc in documents],
            "processed_at": datetime.now().isoformat(),
        }

        # Aggregate format-specific metadata
        audio_metadata = []
        ocr_metadata = []

        for doc in documents:
            if doc.metadata.format == DocumentFormat.AUDIO:
                if "audio" in doc.metadata.extra:
                    audio_metadata.append(doc.metadata.extra["audio"])

            if doc.metadata.format in {DocumentFormat.IMAGE, DocumentFormat.SCANNED_PDF}:
                if "ocr" in doc.metadata.extra:
                    ocr_metadata.append(doc.metadata.extra["ocr"])

        if audio_metadata:
            merged["audio"] = {
                "count": len(audio_metadata),
                "languages": list(set(a.get("language", "unknown") for a in audio_metadata)),
                "avg_confidence": sum(a.get("confidence", 0) for a in audio_metadata) / len(audio_metadata),
            }

        if ocr_metadata:
            merged["ocr"] = {
                "count": len(ocr_metadata),
                "avg_confidence": sum(o.get("confidence", 0) for o in ocr_metadata) / len(ocr_metadata),
                "total_regions": sum(o.get("region_count", 0) for o in ocr_metadata),
            }

        return merged


# =============================================================================
# Result Merger
# =============================================================================


class ResultMerger:
    """Merges multi-modal results into unified output.

    Prepares coordinated results for PKG storage.
    """

    def merge_for_pkg(
        self,
        coordinated: CoordinatedResult
    ) -> Dict:
        """Merge coordinated results for PKG storage.

        Args:
            coordinated: CoordinatedResult from MultiModalCoordinator

        Returns:
            Dictionary ready for PKG ingestion
        """
        merged = {
            "documents": [],
            "cross_modal_entities": [],
            "temporal_alignments": [],
            "summary": coordinated.merged_metadata,
        }

        # Convert documents to PKG format
        for doc in coordinated.documents:
            merged["documents"].append({
                "source_id": doc.metadata.source_id,
                "format": doc.metadata.format.value,
                "content": doc.content,
                "metadata": {
                    "source_path": doc.metadata.source_path,
                    "content_hash": doc.metadata.content_hash,
                    "character_count": doc.metadata.character_count,
                    "word_count": doc.metadata.word_count,
                    "extra": doc.metadata.extra,
                }
            })

        # Convert cross-modal entities
        for entity in coordinated.cross_modal_entities:
            merged["cross_modal_entities"].append({
                "text": entity.entity_text,
                "type": entity.entity_type,
                "modalities": [m.value for m in entity.modalities],
                "occurrences": entity.occurrences,
                "confidence": entity.confidence,
            })

        # Convert temporal alignments
        for alignment in coordinated.temporal_alignments:
            merged["temporal_alignments"].append({
                "audio_timestamp": alignment.audio_timestamp,
                "document_position": alignment.document_position,
                "alignment_type": alignment.alignment_type,
                "confidence": alignment.confidence,
            })

        return merged


# =============================================================================
# Convenience Functions
# =============================================================================


def coordinate_documents(
    documents: List[NormalizedDocument],
    link_entities: bool = True
) -> CoordinatedResult:
    """Coordinate multi-modal documents.

    Convenience function for quick coordination without creating a coordinator.

    Args:
        documents: List of NormalizedDocuments
        link_entities: Whether to perform entity linking

    Returns:
        CoordinatedResult

    Example:
        result = coordinate_documents([audio_doc, image_doc, text_doc])
        print(f"Found {len(result.cross_modal_entities)} cross-modal entities")
    """
    coordinator = MultiModalCoordinator(enable_entity_linking=link_entities)
    return coordinator.coordinate(documents)


def merge_for_pkg(coordinated: CoordinatedResult) -> Dict:
    """Merge coordinated results for PKG storage.

    Convenience function for quick merging.

    Args:
        coordinated: CoordinatedResult

    Returns:
        Dictionary ready for PKG ingestion
    """
    merger = ResultMerger()
    return merger.merge_for_pkg(coordinated)
