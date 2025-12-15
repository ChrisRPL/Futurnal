"""Unit tests for PKG integration bridge.

Tests implementation from Step 07: Causal Structure Preparation.

Test Coverage:
- CausalPKGIntegration class
- Extraction-to-PKG type mapping
- CausalCandidate to CausalRelationshipProps conversion
- Bulk storage operations
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from futurnal.extraction.causal import (
    CausalCandidate,
    CausalPKGIntegration,
    CausalRelationshipType,
)
from futurnal.pkg.schema.models import CausalRelationType as PKGCausalType


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_repository():
    """Create mock RelationshipRepository."""
    repo = MagicMock()
    repo.create_causal_relationship.return_value = "test-rel-id-001"
    return repo


@pytest.fixture
def sample_candidate() -> CausalCandidate:
    """Create a sample CausalCandidate for testing."""
    return CausalCandidate(
        id="candidate_001",
        cause_event_id="event_cause",
        effect_event_id="event_effect",
        relationship_type=CausalRelationshipType.CAUSES,
        temporal_gap=timedelta(hours=2),
        temporal_ordering_valid=True,
        causal_evidence="The meeting led to the decision.",
        causal_confidence=0.85,
        temporality_satisfied=True,
        source_document="test_doc_001",
    )


@pytest.fixture
def integration(mock_repository) -> CausalPKGIntegration:
    """Create CausalPKGIntegration with mock repository."""
    return CausalPKGIntegration(mock_repository)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestCausalPKGIntegrationInit:
    """Test CausalPKGIntegration initialization."""

    def test_initialization_with_repository(self, mock_repository):
        """Validate initialization with repository."""
        integration = CausalPKGIntegration(mock_repository)
        assert integration.repository is mock_repository


# ---------------------------------------------------------------------------
# Type Mapping Tests
# ---------------------------------------------------------------------------


class TestCausalTypeMapping:
    """Test extraction-to-PKG type mapping."""

    def test_causes_maps_to_causes(self, integration):
        """CAUSES maps to PKG CAUSES."""
        result = integration.get_pkg_relationship_type(CausalRelationshipType.CAUSES)
        assert result == PKGCausalType.CAUSES

    def test_enables_maps_to_enables(self, integration):
        """ENABLES maps to PKG ENABLES."""
        result = integration.get_pkg_relationship_type(CausalRelationshipType.ENABLES)
        assert result == PKGCausalType.ENABLES

    def test_prevents_maps_to_prevents(self, integration):
        """PREVENTS maps to PKG PREVENTS."""
        result = integration.get_pkg_relationship_type(CausalRelationshipType.PREVENTS)
        assert result == PKGCausalType.PREVENTS

    def test_triggers_maps_to_triggers(self, integration):
        """TRIGGERS maps to PKG TRIGGERS."""
        result = integration.get_pkg_relationship_type(CausalRelationshipType.TRIGGERS)
        assert result == PKGCausalType.TRIGGERS

    def test_leads_to_maps_to_causes(self, integration):
        """LEADS_TO maps to PKG CAUSES (weaker causation)."""
        result = integration.get_pkg_relationship_type(CausalRelationshipType.LEADS_TO)
        assert result == PKGCausalType.CAUSES

    def test_contributes_to_maps_to_causes(self, integration):
        """CONTRIBUTES_TO maps to PKG CAUSES (partial causation)."""
        result = integration.get_pkg_relationship_type(CausalRelationshipType.CONTRIBUTES_TO)
        assert result == PKGCausalType.CAUSES


# ---------------------------------------------------------------------------
# Candidate to Props Conversion Tests
# ---------------------------------------------------------------------------


class TestCandidateToPropsConversion:
    """Test CausalCandidate to CausalRelationshipProps conversion."""

    def test_basic_conversion(self, integration, sample_candidate):
        """Basic conversion preserves key fields."""
        props = integration.candidate_to_props(sample_candidate)

        assert props.source_document == sample_candidate.source_document
        assert props.causal_confidence == sample_candidate.causal_confidence
        assert props.causal_evidence == sample_candidate.causal_evidence
        assert props.temporal_ordering_valid == sample_candidate.temporal_ordering_valid
        assert props.temporality_satisfied == sample_candidate.temporality_satisfied

    def test_temporal_gap_preserved(self, integration, sample_candidate):
        """Temporal gap is preserved in conversion."""
        props = integration.candidate_to_props(sample_candidate)
        assert props.temporal_gap == timedelta(hours=2)

    def test_phase1_defaults(self, integration, sample_candidate):
        """Phase 1 defaults: is_causal_candidate=True, is_validated=False."""
        props = integration.candidate_to_props(sample_candidate)

        assert props.is_causal_candidate is True
        assert props.is_validated is False
        assert props.validation_method is None

    def test_bradford_hill_criteria_preserved(self, integration, sample_candidate):
        """Bradford Hill criteria structure preserved."""
        props = integration.candidate_to_props(sample_candidate)

        # Temporality should be set (Phase 1)
        assert props.temporality_satisfied is True

        # Other criteria should be None (Phase 3)
        assert props.dose_response is None

    def test_extraction_method_set(self, integration, sample_candidate):
        """Extraction method is set to causal_extraction."""
        props = integration.candidate_to_props(sample_candidate)
        assert props.extraction_method == "causal_extraction"


# ---------------------------------------------------------------------------
# Store Causal Candidate Tests
# ---------------------------------------------------------------------------


class TestStoreCausalCandidate:
    """Test storing single causal candidate."""

    def test_store_creates_relationship(self, integration, mock_repository, sample_candidate):
        """Store creates relationship via repository."""
        rel_id = integration.store_causal_candidate(
            candidate=sample_candidate,
            cause_event_pkg_id="pkg-cause-001",
            effect_event_pkg_id="pkg-effect-001",
        )

        assert rel_id == "test-rel-id-001"
        mock_repository.create_causal_relationship.assert_called_once()

    def test_store_passes_correct_event_ids(self, integration, mock_repository, sample_candidate):
        """Store passes correct event IDs to repository."""
        integration.store_causal_candidate(
            candidate=sample_candidate,
            cause_event_pkg_id="pkg-cause-001",
            effect_event_pkg_id="pkg-effect-001",
        )

        call_args = mock_repository.create_causal_relationship.call_args
        assert call_args.kwargs["cause_event_id"] == "pkg-cause-001"
        assert call_args.kwargs["effect_event_id"] == "pkg-effect-001"

    def test_store_maps_relationship_type(self, integration, mock_repository, sample_candidate):
        """Store maps extraction type to PKG type."""
        integration.store_causal_candidate(
            candidate=sample_candidate,
            cause_event_pkg_id="pkg-cause-001",
            effect_event_pkg_id="pkg-effect-001",
        )

        call_args = mock_repository.create_causal_relationship.call_args
        assert call_args.kwargs["relationship_type"] == PKGCausalType.CAUSES


# ---------------------------------------------------------------------------
# Bulk Storage Tests
# ---------------------------------------------------------------------------


class TestBulkStorage:
    """Test bulk storage operations."""

    def test_bulk_stores_multiple_candidates(self, integration, mock_repository):
        """Bulk operation stores multiple candidates."""
        candidates = [
            CausalCandidate(
                id=f"candidate_{i}",
                cause_event_id=f"event_{i}",
                effect_event_id=f"event_{i+1}",
                relationship_type=CausalRelationshipType.CAUSES,
                temporal_gap=timedelta(hours=1),
                temporal_ordering_valid=True,
                causal_evidence="Test evidence",
                causal_confidence=0.8,
                temporality_satisfied=True,
                source_document="test_doc",
            )
            for i in range(3)
        ]

        event_mapping = {
            "event_0": "pkg-event-0",
            "event_1": "pkg-event-1",
            "event_2": "pkg-event-2",
            "event_3": "pkg-event-3",
        }

        rel_ids = integration.store_causal_candidates_bulk(candidates, event_mapping)

        assert len(rel_ids) == 3
        assert mock_repository.create_causal_relationship.call_count == 3

    def test_bulk_skips_missing_cause_mapping(self, integration, mock_repository):
        """Bulk skips candidates with missing cause event mapping."""
        candidates = [
            CausalCandidate(
                id="candidate_1",
                cause_event_id="unknown_event",  # Not in mapping
                effect_event_id="event_1",
                relationship_type=CausalRelationshipType.CAUSES,
                temporal_gap=timedelta(hours=1),
                temporal_ordering_valid=True,
                causal_evidence="Test evidence",
                causal_confidence=0.8,
                temporality_satisfied=True,
                source_document="test_doc",
            )
        ]

        event_mapping = {"event_1": "pkg-event-1"}  # Missing unknown_event

        rel_ids = integration.store_causal_candidates_bulk(candidates, event_mapping)

        assert len(rel_ids) == 0
        mock_repository.create_causal_relationship.assert_not_called()

    def test_bulk_skips_missing_effect_mapping(self, integration, mock_repository):
        """Bulk skips candidates with missing effect event mapping."""
        candidates = [
            CausalCandidate(
                id="candidate_1",
                cause_event_id="event_0",
                effect_event_id="unknown_event",  # Not in mapping
                relationship_type=CausalRelationshipType.CAUSES,
                temporal_gap=timedelta(hours=1),
                temporal_ordering_valid=True,
                causal_evidence="Test evidence",
                causal_confidence=0.8,
                temporality_satisfied=True,
                source_document="test_doc",
            )
        ]

        event_mapping = {"event_0": "pkg-event-0"}  # Missing unknown_event

        rel_ids = integration.store_causal_candidates_bulk(candidates, event_mapping)

        assert len(rel_ids) == 0
        mock_repository.create_causal_relationship.assert_not_called()

    def test_bulk_handles_repository_errors(self, integration, mock_repository):
        """Bulk continues after repository errors."""
        candidates = [
            CausalCandidate(
                id=f"candidate_{i}",
                cause_event_id=f"event_{i}",
                effect_event_id=f"event_{i+1}",
                relationship_type=CausalRelationshipType.CAUSES,
                temporal_gap=timedelta(hours=1),
                temporal_ordering_valid=True,
                causal_evidence="Test evidence",
                causal_confidence=0.8,
                temporality_satisfied=True,
                source_document="test_doc",
            )
            for i in range(3)
        ]

        event_mapping = {
            f"event_{i}": f"pkg-event-{i}"
            for i in range(4)
        }

        # First call fails, second and third succeed
        mock_repository.create_causal_relationship.side_effect = [
            Exception("Test error"),
            "rel-id-2",
            "rel-id-3",
        ]

        rel_ids = integration.store_causal_candidates_bulk(candidates, event_mapping)

        assert len(rel_ids) == 2  # Only 2 succeeded
        assert mock_repository.create_causal_relationship.call_count == 3
