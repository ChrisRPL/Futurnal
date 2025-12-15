"""Performance benchmarks for causal extraction.

Tests for Step 07: Causal Structure Preparation performance.

Success Criteria from Step 07 Spec:
- Causal candidate detection efficient for 100+ events
- Bradford Hill preparation efficient
- End-to-end extraction flow responsive

Research Foundation:
- CausalRAG (ACL 2025): Real-time causal retrieval
- Temporal KG Extrapolation (IJCAI 2024): Efficient temporal processing

Note: PKG-level performance tests (Neo4j queries <100ms) are in:
tests/pkg/queries/test_temporal_integration.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, List

import pytest

from futurnal.extraction.causal import (
    BradfordHillPreparation,
    CausalRelationshipDetector,
    EventExtractor,
)
from futurnal.extraction.temporal.models import Event, TemporalMark


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


class MockLLM:
    """Mock LLM for performance testing without actual LLM calls."""

    def __init__(self, default_confidence: float = 0.8):
        self.default_confidence = default_confidence
        self.call_count = 0

    def extract(self, prompt: str) -> dict:
        """Return mock extraction results instantly.

        Simulates fast LLM response for benchmarking extraction logic.
        """
        self.call_count += 1

        # Return different responses based on prompt type
        if "Event Types:" in prompt:
            # Event extraction
            return {
                "name": f"Mock Event {self.call_count}",
                "event_type": "action",
                "timestamp": datetime.utcnow().isoformat(),
                "duration": None,
                "participants": [],
                "confidence": self.default_confidence,
            }
        else:
            # Causal detection
            return {
                "evidence": "Mock causal evidence for performance testing",
                "relationship_type": "causes",
                "confidence": self.default_confidence,
            }


class MockTemporalExtractor:
    """Mock temporal extractor for testing."""

    def extract_explicit_timestamps(self, text: str) -> List[TemporalMark]:
        """Return empty markers - we inject events directly."""
        return []


class MockDocument:
    """Mock document for testing."""

    def __init__(self, content: str, doc_id: str = "perf_test_doc"):
        self.content = content
        self.doc_id = doc_id


def generate_test_events(count: int, base_timestamp: datetime = None) -> List[Event]:
    """Generate test events with sequential timestamps.

    Args:
        count: Number of events to generate
        base_timestamp: Starting timestamp (defaults to 2024-01-01)

    Returns:
        List of Event objects with valid temporal ordering
    """
    if base_timestamp is None:
        base_timestamp = datetime(2024, 1, 1)

    events = []
    event_types = ["meeting", "decision", "action", "communication", "state_change"]

    for i in range(count):
        event = Event(
            name=f"Performance Event {i}",
            event_type=event_types[i % len(event_types)],
            timestamp=base_timestamp + timedelta(hours=i),
            source_document="perf_test_doc",
            extraction_confidence=0.9,
        )
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Performance Benchmark Tests
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestCausalExtractionPerformance:
    """Performance benchmarks for causal extraction layer.

    Step 07 Success Criteria: Efficient causal candidate detection.
    """

    def test_causal_detection_100_events(self):
        """Benchmark: Detect causal candidates among 100 events.

        Target: <1 second for 100 events (without LLM latency).
        """
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)

        events = generate_test_events(100)
        doc = MockDocument("Performance test document with causal evidence.")

        start_time = time.perf_counter()

        candidates = detector.detect_causal_candidates(events, doc)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete quickly (under 1 second)
        assert elapsed_ms < 1000, f"Detection took {elapsed_ms:.1f}ms (target <1000ms)"

        # Should find multiple candidates
        assert len(candidates) > 0

        print(f"\nPerformance: Detected {len(candidates)} candidates "
              f"from {len(events)} events in {elapsed_ms:.1f}ms")

    def test_causal_detection_scaling(self):
        """Benchmark: Verify detection scales reasonably with event count.

        Test O(n²) pairwise comparison scaling.
        """
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)

        results = []

        for event_count in [10, 25, 50]:
            events = generate_test_events(event_count)
            doc = MockDocument("Performance scaling test.")

            start_time = time.perf_counter()
            candidates = detector.detect_causal_candidates(events, doc)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            results.append({
                "events": event_count,
                "candidates": len(candidates),
                "time_ms": elapsed_ms,
                "pairs_checked": event_count * (event_count - 1) // 2,
            })

        # Print scaling results
        print("\n--- Causal Detection Scaling ---")
        for r in results:
            print(f"  {r['events']} events: {r['candidates']} candidates "
                  f"({r['pairs_checked']} pairs) in {r['time_ms']:.1f}ms")

        # Verify reasonable scaling (not exponential)
        ratio_10_to_50 = results[2]["time_ms"] / max(results[0]["time_ms"], 0.1)
        assert ratio_10_to_50 < 100, f"Scaling ratio {ratio_10_to_50:.1f}x is too high"

    def test_bradford_hill_preparation_bulk(self):
        """Benchmark: Bradford Hill preparation for 100 candidates.

        Target: <100ms for bulk preparation.
        """
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        prep = BradfordHillPreparation()

        # Generate candidates
        events = generate_test_events(50)
        doc = MockDocument("Bradford Hill performance test.")
        candidates = detector.detect_causal_candidates(events, doc)

        # Ensure we have enough candidates
        assert len(candidates) > 0, "Need candidates for benchmark"

        start_time = time.perf_counter()

        # Prepare Bradford Hill criteria for all candidates
        criteria_list = [prep.prepare_for_validation(c) for c in candidates]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should be very fast (no LLM calls in Phase 1)
        assert elapsed_ms < 100, f"Preparation took {elapsed_ms:.1f}ms (target <100ms)"

        # All criteria should have temporality validated
        assert all(c.temporality for c in criteria_list)

        print(f"\nPerformance: Prepared {len(criteria_list)} Bradford Hill criteria "
              f"in {elapsed_ms:.1f}ms")

    def test_temporal_filtering_efficiency(self):
        """Benchmark: Temporal gap filtering performance.

        Step 07 Requirement: 100% temporal ordering validation.
        """
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm, max_temporal_gap_days=30)

        # Create events spanning a year
        events = generate_test_events(100)
        doc = MockDocument("Temporal filtering test.")

        start_time = time.perf_counter()

        candidates = detector.detect_causal_candidates(events, doc)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify all candidates satisfy temporal constraints
        for candidate in candidates:
            assert candidate.temporal_ordering_valid is True
            assert candidate.temporality_satisfied is True
            assert candidate.temporal_gap <= timedelta(days=30)

        print(f"\nPerformance: Temporal filtering on {len(events)} events "
              f"produced {len(candidates)} valid candidates in {elapsed_ms:.1f}ms")


@pytest.mark.performance
class TestEventExtractionPerformance:
    """Performance benchmarks for event extraction."""

    def test_event_indicator_detection_bulk(self):
        """Benchmark: Event indicator detection on large text.

        Tests the _has_event_indicators method performance.
        """
        llm = MockLLM()
        temporal_extractor = MockTemporalExtractor()
        extractor = EventExtractor(llm, temporal_extractor)

        # Generate test sentences
        test_sentences = [
            f"We met with the team on day {i} to discuss the project."
            for i in range(100)
        ]

        start_time = time.perf_counter()

        has_indicators = [
            extractor._has_event_indicators(sentence)
            for sentence in test_sentences
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should be very fast (simple string matching)
        assert elapsed_ms < 50, f"Indicator detection took {elapsed_ms:.1f}ms (target <50ms)"

        # Most sentences should have indicators (they contain "met")
        assert sum(has_indicators) > 90

        print(f"\nPerformance: Checked {len(test_sentences)} sentences "
              f"for event indicators in {elapsed_ms:.1f}ms")


@pytest.mark.performance
class TestEndToEndCausalPerformance:
    """End-to-end performance benchmarks for complete causal workflow."""

    def test_full_causal_workflow(self):
        """Benchmark: Complete extraction → detection → preparation flow.

        Step 07 Success Criteria: End-to-end causal structure preparation.
        """
        llm = MockLLM()
        detector = CausalRelationshipDetector(llm)
        prep = BradfordHillPreparation()

        # Simulate events extracted from document
        events = generate_test_events(50)
        doc = MockDocument("Full workflow performance test with causal relationships.")

        start_time = time.perf_counter()

        # Step 1: Detect causal candidates
        candidates = detector.detect_causal_candidates(events, doc)

        # Step 2: Prepare Bradford Hill criteria
        criteria_list = [prep.prepare_for_validation(c) for c in candidates]

        # Step 3: Validate all temporality satisfied
        valid_count = sum(1 for c in criteria_list if c.temporality)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Complete workflow should be fast
        assert elapsed_ms < 2000, f"Full workflow took {elapsed_ms:.1f}ms (target <2000ms)"

        # All criteria should be valid (Phase 1 only validates temporality)
        assert valid_count == len(criteria_list)

        print(f"\nPerformance: Full causal workflow "
              f"({len(events)} events → {len(candidates)} candidates "
              f"→ {len(criteria_list)} Bradford Hill criteria) "
              f"in {elapsed_ms:.1f}ms")
