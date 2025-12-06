"""Causal search module test fixtures.

Provides fixtures for testing CausalChainRetrieval:
- Mock PKG queries with causal relationships
- Test event data with causal structure
- Integration test fixtures

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock

import pytest

from futurnal.pkg.schema.models import EventNode
from futurnal.pkg.queries.models import CausalChainResult, CausalPath
from futurnal.search.config import CausalSearchConfig, SearchConfig
from tests.search.conftest import create_test_event, MockTemporalGraphQueries


# ---------------------------------------------------------------------------
# Causal Event Factory
# ---------------------------------------------------------------------------


def create_causal_chain_events() -> List[EventNode]:
    """Create events forming a causal chain: meeting -> decision -> publication.

    Returns:
        List of 3 EventNodes with proper temporal ordering
    """
    return [
        create_test_event(
            "m1", "Initial Meeting", "meeting", datetime(2024, 1, 1, 10, 0)
        ),
        create_test_event(
            "d1", "Key Decision", "decision", datetime(2024, 1, 5, 14, 0)
        ),
        create_test_event(
            "p1", "Final Publication", "publication", datetime(2024, 1, 15, 9, 0)
        ),
    ]


def create_multi_chain_events() -> List[EventNode]:
    """Create events with multiple causal chains.

    Chain 1: m1 -> d1 -> p1
    Chain 2: m2 -> d2 -> a1
    Shared: m1 -> d2 (meeting leads to different decisions)
    """
    return [
        # Chain 1
        create_test_event(
            "m1", "Meeting Alpha", "meeting", datetime(2024, 1, 1, 10, 0)
        ),
        create_test_event(
            "d1", "Decision Alpha", "decision", datetime(2024, 1, 5, 14, 0)
        ),
        create_test_event(
            "p1", "Publication Alpha", "publication", datetime(2024, 1, 15, 9, 0)
        ),
        # Chain 2
        create_test_event(
            "m2", "Meeting Beta", "meeting", datetime(2024, 1, 3, 11, 0)
        ),
        create_test_event(
            "d2", "Decision Beta", "decision", datetime(2024, 1, 8, 15, 0)
        ),
        create_test_event(
            "a1", "Action Beta", "action", datetime(2024, 1, 20, 10, 0)
        ),
    ]


def create_invalid_temporal_events() -> List[EventNode]:
    """Create events with invalid temporal ordering for testing violations.

    d1 happens BEFORE m1 (invalid for causal chain m1 -> d1)
    """
    return [
        create_test_event(
            "m1", "Meeting", "meeting", datetime(2024, 1, 10, 10, 0)  # Later
        ),
        create_test_event(
            "d1", "Decision", "decision", datetime(2024, 1, 5, 14, 0)  # Earlier
        ),
    ]


# ---------------------------------------------------------------------------
# Mock PKG with Causal Support
# ---------------------------------------------------------------------------


class MockCausalGraphQueries(MockTemporalGraphQueries):
    """Extended mock with causal relationship support.

    Simulates Neo4j causal queries for unit testing without database.
    """

    def __init__(self, events: Optional[List[EventNode]] = None):
        """Initialize with optional preset events."""
        super().__init__(events)
        self._causal_relationships: List[Dict[str, Any]] = []
        # Mock database manager for session access
        self._db = MagicMock()
        self._db.session.return_value.__enter__ = MagicMock(
            return_value=self._create_mock_session()
        )
        self._db.session.return_value.__exit__ = MagicMock(return_value=False)

    def _create_mock_session(self) -> MagicMock:
        """Create mock Neo4j session."""
        session = MagicMock()
        session.run = MagicMock(side_effect=self._handle_query)
        return session

    def _handle_query(self, query: str, params: Dict[str, Any]) -> MagicMock:
        """Handle mock Cypher queries."""
        result = MagicMock()

        # Handle timestamp fetch queries
        if "e.timestamp AS timestamp" in query:
            records = []
            event_ids = params.get("event_ids", [])
            for event in self._events:
                if event.id in event_ids:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, e=event: (
                        e.id if k == "id" else e.timestamp
                    )
                    records.append(record)
            result.__iter__ = lambda _: iter(records)
            return result

        # Handle find causes query
        if "CAUSES|ENABLES|TRIGGERS" in query and "effect.id = $event_id" in query:
            records = self._find_causes(params.get("event_id"))
            result.__iter__ = lambda _: iter(records)
            return result

        # Handle find effects query
        if "CAUSES|ENABLES|TRIGGERS" in query and "cause.id = $event_id" in query:
            records = self._find_effects(params.get("event_id"))
            result.__iter__ = lambda _: iter(records)
            return result

        # Handle causal path query
        if "shortestPath" in query:
            record = self._find_path(
                params.get("start_id"), params.get("end_id")
            )
            result.single = lambda: record
            return result

        # Handle get event by id
        if "Event {id: $id}" in query:
            event = self._get_event_by_id(params.get("id"))
            if event:
                record = MagicMock()
                record.__getitem__ = lambda _, k: {
                    "id": event.id,
                    "name": event.name,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "source_document": event.source_document,
                    "description": event.description,
                    "confidence": event.confidence,
                }
                result.single = lambda: record
            else:
                result.single = lambda: None
            return result

        # Default: return empty
        result.__iter__ = lambda _: iter([])
        result.single = lambda: None
        return result

    def _find_causes(self, event_id: str) -> List[MagicMock]:
        """Find mock causes for event."""
        records = []
        for rel in self._causal_relationships:
            if rel["effect_id"] == event_id:
                cause = self._get_event_by_id(rel["cause_id"])
                if cause:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, c=cause, r=rel: {
                        "cause_id": c.id,
                        "cause_name": c.name,
                        "cause_timestamp": c.timestamp,
                        "distance": 1,
                        "confidence_scores": [r["confidence"]],
                        "path_ids": [c.id, event_id],
                    }.get(k)
                    records.append(record)
        return records

    def _find_effects(self, event_id: str) -> List[MagicMock]:
        """Find mock effects for event."""
        records = []
        for rel in self._causal_relationships:
            if rel["cause_id"] == event_id:
                effect = self._get_event_by_id(rel["effect_id"])
                if effect:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, e=effect, r=rel: {
                        "effect_id": e.id,
                        "effect_name": e.name,
                        "effect_timestamp": e.timestamp,
                        "distance": 1,
                        "confidence_scores": [r["confidence"]],
                        "path_ids": [event_id, e.id],
                    }.get(k)
                    records.append(record)
        return records

    def _find_path(
        self, start_id: str, end_id: str
    ) -> Optional[MagicMock]:
        """Find mock path between events."""
        # Check for direct relationship
        for rel in self._causal_relationships:
            if rel["cause_id"] == start_id and rel["effect_id"] == end_id:
                record = MagicMock()
                record.__getitem__ = lambda _, k: {
                    "event_ids": [start_id, end_id],
                    "confidences": [rel["confidence"]],
                    "evidence": [rel.get("evidence", "")],
                    "path_length": 1,
                }.get(k)
                return record

        # Check for 2-hop path
        for rel1 in self._causal_relationships:
            if rel1["cause_id"] == start_id:
                mid_id = rel1["effect_id"]
                for rel2 in self._causal_relationships:
                    if rel2["cause_id"] == mid_id and rel2["effect_id"] == end_id:
                        record = MagicMock()
                        record.__getitem__ = lambda _, k: {
                            "event_ids": [start_id, mid_id, end_id],
                            "confidences": [rel1["confidence"], rel2["confidence"]],
                            "evidence": [
                                rel1.get("evidence", ""),
                                rel2.get("evidence", ""),
                            ],
                            "path_length": 2,
                        }.get(k)
                        return record

        return None

    def add_causal_relationship(
        self,
        cause_id: str,
        effect_id: str,
        confidence: float = 0.8,
        evidence: str = "",
    ) -> None:
        """Add a causal relationship between events.

        Args:
            cause_id: ID of the cause event
            effect_id: ID of the effect event
            confidence: Causal confidence (0.0-1.0)
            evidence: Text evidence for the relationship
        """
        self._causal_relationships.append({
            "cause_id": cause_id,
            "effect_id": effect_id,
            "confidence": confidence,
            "evidence": evidence,
        })

    def _convert_neo4j_props(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock conversion of Neo4j properties."""
        return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def causal_config() -> CausalSearchConfig:
    """Provide default causal search config."""
    return CausalSearchConfig(
        default_max_hops=3,
        default_min_confidence=0.6,
    )


@pytest.fixture
def causal_chain_events() -> List[EventNode]:
    """Provide events forming a causal chain: meeting -> decision -> publication."""
    return create_causal_chain_events()


@pytest.fixture
def multi_chain_events() -> List[EventNode]:
    """Provide events with multiple causal chains."""
    return create_multi_chain_events()


@pytest.fixture
def invalid_temporal_events() -> List[EventNode]:
    """Provide events with invalid temporal ordering."""
    return create_invalid_temporal_events()


@pytest.fixture
def mock_causal_pkg(causal_chain_events) -> MockCausalGraphQueries:
    """Provide mock PKG with causal relationships.

    Chain: m1 -> d1 -> p1
    """
    mock = MockCausalGraphQueries(events=causal_chain_events)
    mock.add_causal_relationship("m1", "d1", confidence=0.9, evidence="Meeting led to decision")
    mock.add_causal_relationship("d1", "p1", confidence=0.85, evidence="Decision resulted in publication")
    return mock


@pytest.fixture
def mock_causal_pkg_multi(multi_chain_events) -> MockCausalGraphQueries:
    """Provide mock PKG with multiple causal chains."""
    mock = MockCausalGraphQueries(events=multi_chain_events)
    # Chain 1: m1 -> d1 -> p1
    mock.add_causal_relationship("m1", "d1", confidence=0.9)
    mock.add_causal_relationship("d1", "p1", confidence=0.85)
    # Chain 2: m2 -> d2 -> a1
    mock.add_causal_relationship("m2", "d2", confidence=0.8)
    mock.add_causal_relationship("d2", "a1", confidence=0.75)
    # Cross-chain: m1 -> d2
    mock.add_causal_relationship("m1", "d2", confidence=0.7)
    return mock


@pytest.fixture
def mock_causal_pkg_invalid(invalid_temporal_events) -> MockCausalGraphQueries:
    """Provide mock PKG with invalid temporal relationships."""
    mock = MockCausalGraphQueries(events=invalid_temporal_events)
    # Add relationship with wrong temporal ordering
    mock.add_causal_relationship("m1", "d1", confidence=0.9)
    return mock
