"""PKG Integration Bridge for Causal Extraction.

Bridges the extraction layer's CausalCandidate to PKG storage layer,
transforming extraction results into PKG-compatible structures.

Implementation follows:
- Step 07: Causal Structure Preparation (this module)
- PKG Schema: docs/phase-1/pkg-graph-storage-production-plan/01-graph-schema-design.md

Option B Compliance:
- Causal candidates flagged for Phase 3 validation
- Temporal ordering validated before storage
- Bradford Hill criteria structure preserved

Research Foundation:
- CausalRAG (ACL 2025): Causal graphs integrated into RAG
- Temporal KG Extrapolation (IJCAI 2024): Temporal ordering preservation
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional, TYPE_CHECKING

from futurnal.extraction.causal.models import (
    CausalCandidate,
    CausalRelationshipType as ExtractionCausalType,
)
from futurnal.pkg.schema.models import (
    CausalRelationType as PKGCausalType,
    CausalRelationshipProps,
)

if TYPE_CHECKING:
    from futurnal.pkg.repository.relationships import RelationshipRepository

logger = logging.getLogger(__name__)


# Mapping from extraction layer types to PKG layer types
CAUSAL_TYPE_MAPPING: Dict[ExtractionCausalType, PKGCausalType] = {
    ExtractionCausalType.CAUSES: PKGCausalType.CAUSES,
    ExtractionCausalType.ENABLES: PKGCausalType.ENABLES,
    ExtractionCausalType.PREVENTS: PKGCausalType.PREVENTS,
    ExtractionCausalType.TRIGGERS: PKGCausalType.TRIGGERS,
    # LEADS_TO and CONTRIBUTES_TO map to CAUSES (weaker causation)
    ExtractionCausalType.LEADS_TO: PKGCausalType.CAUSES,
    ExtractionCausalType.CONTRIBUTES_TO: PKGCausalType.CAUSES,
}


class CausalPKGIntegration:
    """Bridge between extraction CausalCandidate and PKG storage.

    Transforms extraction-layer causal candidates into PKG-layer
    relationship structures for graph storage.

    Phase 1 Scope:
    - Store causal candidates with is_causal_candidate=True
    - Preserve Bradford Hill criteria structure
    - All candidates marked as is_validated=False

    Phase 3 Scope (future):
    - Update is_validated=True after user confirmation
    - Populate additional Bradford Hill criteria
    - Track validation method

    Example:
        >>> from futurnal.extraction.causal.pkg_integration import CausalPKGIntegration
        >>> from futurnal.pkg.repository.relationships import RelationshipRepository
        >>>
        >>> integration = CausalPKGIntegration(relationship_repo)
        >>> rel_id = integration.store_causal_candidate(
        ...     candidate=causal_candidate,
        ...     cause_event_pkg_id="event-001",
        ...     effect_event_pkg_id="event-002",
        ... )
    """

    def __init__(self, repository: "RelationshipRepository"):
        """Initialize PKG integration bridge.

        Args:
            repository: PKG RelationshipRepository for storing relationships
        """
        self.repository = repository

    def store_causal_candidate(
        self,
        candidate: CausalCandidate,
        cause_event_pkg_id: str,
        effect_event_pkg_id: str,
    ) -> str:
        """Store extraction CausalCandidate in PKG.

        Transforms extraction-layer CausalCandidate to PKG-layer
        CausalRelationshipProps and creates the relationship.

        Args:
            candidate: Extraction layer CausalCandidate
            cause_event_pkg_id: PKG Event node ID for cause
            effect_event_pkg_id: PKG Event node ID for effect

        Returns:
            Relationship ID in PKG

        Raises:
            TemporalValidationError: If temporal ordering is invalid
            EntityNotFoundError: If event nodes don't exist
        """
        # Map extraction relationship type to PKG type
        pkg_rel_type = CAUSAL_TYPE_MAPPING.get(
            candidate.relationship_type,
            PKGCausalType.CAUSES,
        )

        # Build PKG relationship properties
        props = CausalRelationshipProps(
            # Provenance
            source_document=candidate.source_document,
            extraction_method="causal_extraction",

            # Temporal fields
            temporal_gap=candidate.temporal_gap,
            temporal_ordering_valid=candidate.temporal_ordering_valid,

            # Causal metadata
            causal_confidence=candidate.causal_confidence,
            causal_evidence=candidate.causal_evidence,
            is_causal_candidate=True,
            is_validated=False,  # Phase 3 will validate
            validation_method=None,

            # Bradford Hill criteria (Phase 1: only temporality)
            temporality_satisfied=candidate.temporality_satisfied,
            strength=candidate.strength,  # May be None
            dose_response=None,  # Phase 3
            consistency=candidate.consistency,  # May be None
            plausibility=candidate.plausibility,  # May be None
        )

        # Store in PKG via repository
        relationship_id = self.repository.create_causal_relationship(
            cause_event_id=cause_event_pkg_id,
            effect_event_id=effect_event_pkg_id,
            relationship_type=pkg_rel_type,
            properties=props,
        )

        logger.info(
            "Stored causal candidate: %s -%s-> %s (confidence: %.2f)",
            cause_event_pkg_id,
            pkg_rel_type.value,
            effect_event_pkg_id,
            candidate.causal_confidence,
        )

        return relationship_id

    def store_causal_candidates_bulk(
        self,
        candidates: List[CausalCandidate],
        event_id_mapping: Dict[str, str],
    ) -> List[str]:
        """Store multiple causal candidates in PKG.

        Args:
            candidates: List of extraction CausalCandidate objects
            event_id_mapping: Mapping from extraction event names to PKG event IDs

        Returns:
            List of relationship IDs created

        Note:
            Candidates with missing event ID mappings are skipped with warning.
        """
        relationship_ids = []

        for candidate in candidates:
            cause_pkg_id = event_id_mapping.get(candidate.cause_event_id)
            effect_pkg_id = event_id_mapping.get(candidate.effect_event_id)

            if not cause_pkg_id:
                logger.warning(
                    "Skipping candidate: cause event '%s' not in PKG mapping",
                    candidate.cause_event_id,
                )
                continue

            if not effect_pkg_id:
                logger.warning(
                    "Skipping candidate: effect event '%s' not in PKG mapping",
                    candidate.effect_event_id,
                )
                continue

            try:
                rel_id = self.store_causal_candidate(
                    candidate=candidate,
                    cause_event_pkg_id=cause_pkg_id,
                    effect_event_pkg_id=effect_pkg_id,
                )
                relationship_ids.append(rel_id)
            except Exception as e:
                logger.error(
                    "Failed to store causal candidate %s -> %s: %s",
                    candidate.cause_event_id,
                    candidate.effect_event_id,
                    str(e),
                )

        logger.info(
            "Stored %d/%d causal candidates in PKG",
            len(relationship_ids),
            len(candidates),
        )

        return relationship_ids

    def candidate_to_props(
        self,
        candidate: CausalCandidate,
    ) -> CausalRelationshipProps:
        """Convert CausalCandidate to PKG CausalRelationshipProps.

        Utility method for cases where relationship creation is handled
        externally but property conversion is needed.

        Args:
            candidate: Extraction layer CausalCandidate

        Returns:
            PKG-compatible CausalRelationshipProps
        """
        return CausalRelationshipProps(
            source_document=candidate.source_document,
            extraction_method="causal_extraction",
            temporal_gap=candidate.temporal_gap,
            temporal_ordering_valid=candidate.temporal_ordering_valid,
            causal_confidence=candidate.causal_confidence,
            causal_evidence=candidate.causal_evidence,
            is_causal_candidate=True,
            is_validated=False,
            validation_method=None,
            temporality_satisfied=candidate.temporality_satisfied,
            strength=candidate.strength,
            dose_response=None,
            consistency=candidate.consistency,
            plausibility=candidate.plausibility,
        )

    def get_pkg_relationship_type(
        self,
        extraction_type: ExtractionCausalType,
    ) -> PKGCausalType:
        """Map extraction relationship type to PKG relationship type.

        Args:
            extraction_type: Extraction layer CausalRelationshipType

        Returns:
            PKG layer CausalRelationType
        """
        return CAUSAL_TYPE_MAPPING.get(extraction_type, PKGCausalType.CAUSES)
