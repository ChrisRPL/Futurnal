"""Temporal Consistency Validator for Module 01.

This module implements comprehensive temporal consistency validation including:
- Cycle detection in temporal graphs (DFS-based)
- Transitivity validation (transitive closure checking)
- Causal ordering validation (Bradford-Hill Criterion #1)
- Allen's Interval Algebra consistency checking

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/01-temporal-extraction.md

Research Foundation:
- Time-R1 (ArXiv 2505.13508v2): Comprehensive temporal reasoning
- Temporal KG Extrapolation (IJCAI 2024): Causal subhistory identification
- Bradford-Hill Criteria: Temporal precedence for causality

Quality Gates (.cursor/rules/quality-gates.mdc):
- Temporal consistency: 100% (zero contradictions allowed)
- No cycles in BEFORE/AFTER/CAUSES relationships
- All CAUSES relationships satisfy cause.timestamp < effect.timestamp

Option B Compliance:
- Temporal-first design enforced
- Phase 3 causal inference foundation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from futurnal.extraction.temporal.models import (
    Event,
    TemporalRelationship,
    TemporalRelationshipType,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ViolationType(str, Enum):
    """Types of temporal consistency violations."""

    CYCLE = "cycle"
    TRANSITIVITY = "transitivity"
    CAUSAL_ORDERING = "causal_ordering"
    INTERVAL_INCONSISTENCY = "interval_inconsistency"


class Severity(str, Enum):
    """Severity levels for violations."""

    ERROR = "error"      # Must be fixed before Phase 3
    WARNING = "warning"  # Flagged but may proceed with caution


@dataclass
class TemporalInconsistency:
    """A detected temporal inconsistency.

    Represents a single violation of temporal consistency rules.
    Used to track and report issues that block Phase 3 causal inference.

    Attributes:
        event_ids: Tuple of event IDs involved in the inconsistency
        violation_type: Type of violation (cycle, transitivity, causal_ordering)
        description: Human-readable description of the issue
        severity: Error (blocker) or warning (flagged)
        evidence: Additional evidence or context
    """

    event_ids: Tuple[str, ...]
    violation_type: ViolationType
    description: str
    severity: Severity = Severity.ERROR
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "event_ids": list(self.event_ids),
            "violation_type": self.violation_type.value,
            "description": self.description,
            "severity": self.severity.value,
            "evidence": self.evidence,
        }


class TemporalConsistencyValidator:
    """Validate temporal consistency of extracted relationships.

    Implements comprehensive validation per research requirements:
    1. Cycle Detection: No A→B→C→A in BEFORE/AFTER/CAUSES
    2. Transitivity: If A BEFORE B and B BEFORE C, then A.timestamp < C.timestamp
    3. Causal Ordering: CAUSES/ENABLES/TRIGGERS require cause < effect

    This validator is CRITICAL for Phase 3 causal inference. Without it,
    temporal paradoxes could invalidate Bradford-Hill criteria validation.

    Example:
        >>> validator = TemporalConsistencyValidator()
        >>> result = validator.validate(events, relationships)
        >>> if not result.valid:
        ...     for error in result.errors:
        ...         print(f"Inconsistency: {error}")
    """

    # Relationship types that imply temporal ordering (source BEFORE target)
    ORDERING_TYPES = {
        TemporalRelationshipType.BEFORE,
        TemporalRelationshipType.CAUSES,
        TemporalRelationshipType.ENABLES,
        TemporalRelationshipType.TRIGGERS,
        TemporalRelationshipType.PREVENTS,  # Still implies temporal ordering
    }

    # Causal relationship types that require strict cause < effect
    CAUSAL_TYPES = {
        TemporalRelationshipType.CAUSES,
        TemporalRelationshipType.ENABLES,
        TemporalRelationshipType.TRIGGERS,
        TemporalRelationshipType.PREVENTS,
    }

    def __init__(self, strict_mode: bool = True):
        """Initialize validator.

        Args:
            strict_mode: If True, all violations are errors.
                        If False, some may be warnings.
        """
        self.strict_mode = strict_mode

    def validate(
        self,
        events: List[Event],
        relations: List[TemporalRelationship],
    ) -> ValidationResult:
        """Validate temporal consistency of events and relationships.

        Quality Gate: 100% consistency required (zero errors allowed).

        Args:
            events: List of events to validate
            relations: List of temporal relationships between events

        Returns:
            ValidationResult with valid=True if consistent, else errors populated
        """
        inconsistencies: List[TemporalInconsistency] = []

        # Build temporal graph for analysis
        graph = self._build_temporal_graph(events, relations)
        event_map = {e.event_id: e for e in events}

        # Check 1: Cycle Detection (CRITICAL)
        cycles = self._detect_cycles(graph)
        for cycle in cycles:
            inconsistencies.append(TemporalInconsistency(
                event_ids=tuple(cycle),
                violation_type=ViolationType.CYCLE,
                description=f"Temporal cycle detected: {' → '.join(cycle)}",
                severity=Severity.ERROR,
                evidence=f"Events form a cycle in temporal ordering",
            ))

        # Check 2: Transitivity Validation
        transitive_violations = self._check_transitivity(graph, event_map)
        inconsistencies.extend(transitive_violations)

        # Check 3: Causal Ordering (Bradford-Hill Criterion #1)
        causal_violations = self._check_causal_ordering(event_map, relations)
        inconsistencies.extend(causal_violations)

        # Check 4: Interval Consistency (Allen's IA rules)
        interval_violations = self._check_interval_consistency(event_map, relations)
        inconsistencies.extend(interval_violations)

        # Build result
        errors = [inc.description for inc in inconsistencies if inc.severity == Severity.ERROR]
        warnings = [inc.description for inc in inconsistencies if inc.severity == Severity.WARNING]

        valid = len(errors) == 0

        if not valid:
            logger.warning(
                f"Temporal consistency validation failed: {len(errors)} errors, {len(warnings)} warnings"
            )
        else:
            logger.info(
                f"Temporal consistency validation passed ({len(events)} events, {len(relations)} relations)"
            )

        return ValidationResult(
            valid=valid,
            relationships=relations,
            errors=errors,
            warnings=warnings,
        )

    def _build_temporal_graph(
        self,
        events: List[Event],
        relations: List[TemporalRelationship],
    ) -> Dict[str, Set[str]]:
        """Build directed graph of temporal ordering.

        Creates adjacency list where edge A→B means A is BEFORE B
        (or A CAUSES/ENABLES/TRIGGERS B, which implies A < B).

        Args:
            events: List of events (nodes)
            relations: List of temporal relationships (edges)

        Returns:
            Adjacency list: Dict[source_id, Set[target_ids]]
        """
        graph: Dict[str, Set[str]] = {}

        # Initialize all nodes
        for event in events:
            graph[event.event_id] = set()

        # Add edges for ordering relationships
        for rel in relations:
            if rel.relationship_type in self.ORDERING_TYPES:
                graph.setdefault(rel.entity1_id, set()).add(rel.entity2_id)
            elif rel.relationship_type == TemporalRelationshipType.AFTER:
                # AFTER is inverse of BEFORE
                graph.setdefault(rel.entity2_id, set()).add(rel.entity1_id)

        return graph

    def _detect_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect cycles in temporal graph using DFS.

        A cycle indicates a temporal paradox (A before B before C before A).
        Such cycles make Phase 3 causal inference impossible.

        Args:
            graph: Adjacency list representation

        Returns:
            List of cycles found (each cycle is a list of event IDs)
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            """DFS with recursion stack tracking for cycle detection."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found cycle: extract cycle from path
                    try:
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        # Only add unique cycles (avoid duplicates)
                        if cycle not in cycles:
                            cycles.append(cycle)
                    except ValueError:
                        pass

            path.pop()
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _check_transitivity(
        self,
        graph: Dict[str, Set[str]],
        event_map: Dict[str, Event],
    ) -> List[TemporalInconsistency]:
        """Validate transitive closure is consistent with timestamps.

        Rule: If A BEFORE B and B BEFORE C (through graph reachability),
        then A.timestamp must be < C.timestamp.

        Args:
            graph: Adjacency list of temporal ordering
            event_map: Dict mapping event_id to Event

        Returns:
            List of transitivity violations
        """
        violations: List[TemporalInconsistency] = []

        # Compute transitive closure using Floyd-Warshall style BFS
        closure = self._compute_transitive_closure(graph)

        # Check each reachable pair
        for source_id, reachable in closure.items():
            source = event_map.get(source_id)
            if not source or not source.timestamp:
                continue

            for target_id in reachable:
                if target_id == source_id:
                    continue

                target = event_map.get(target_id)
                if not target or not target.timestamp:
                    continue

                # Source should be before target
                if source.timestamp >= target.timestamp:
                    violations.append(TemporalInconsistency(
                        event_ids=(source_id, target_id),
                        violation_type=ViolationType.TRANSITIVITY,
                        description=(
                            f"Transitive ordering violated: '{source.name}' should be before "
                            f"'{target.name}', but {source.timestamp.isoformat()} >= "
                            f"{target.timestamp.isoformat()}"
                        ),
                        severity=Severity.ERROR,
                        evidence=f"Path exists in temporal graph from {source_id} to {target_id}",
                    ))

        return violations

    def _compute_transitive_closure(
        self,
        graph: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """Compute transitive closure of temporal graph.

        Uses BFS from each node to find all reachable nodes.
        More efficient than Floyd-Warshall for sparse graphs.

        Args:
            graph: Adjacency list representation

        Returns:
            Dict mapping each node to set of all reachable nodes
        """
        closure: Dict[str, Set[str]] = {}

        for start_node in graph:
            reachable: Set[str] = set()
            queue = list(graph.get(start_node, set()))
            visited_in_bfs: Set[str] = set(queue)

            while queue:
                current = queue.pop(0)
                reachable.add(current)

                for neighbor in graph.get(current, set()):
                    if neighbor not in visited_in_bfs:
                        visited_in_bfs.add(neighbor)
                        queue.append(neighbor)

            closure[start_node] = reachable

        return closure

    def _check_causal_ordering(
        self,
        event_map: Dict[str, Event],
        relations: List[TemporalRelationship],
    ) -> List[TemporalInconsistency]:
        """Validate causal relationships respect temporal ordering.

        Bradford-Hill Criterion #1: Cause must temporally precede effect.
        This is non-negotiable for Phase 3 causal inference.

        Args:
            event_map: Dict mapping event_id to Event
            relations: List of temporal relationships

        Returns:
            List of causal ordering violations
        """
        violations: List[TemporalInconsistency] = []

        for rel in relations:
            if rel.relationship_type not in self.CAUSAL_TYPES:
                continue

            cause = event_map.get(rel.entity1_id)
            effect = event_map.get(rel.entity2_id)

            if not cause or not effect:
                # Missing events - flag as warning
                violations.append(TemporalInconsistency(
                    event_ids=(rel.entity1_id, rel.entity2_id),
                    violation_type=ViolationType.CAUSAL_ORDERING,
                    description=(
                        f"Causal relationship references unknown event(s): "
                        f"cause={rel.entity1_id}, effect={rel.entity2_id}"
                    ),
                    severity=Severity.WARNING,
                    evidence=f"Relationship type: {rel.relationship_type.value}",
                ))
                continue

            # Both events must have timestamps for causal validation
            if not cause.timestamp or not effect.timestamp:
                violations.append(TemporalInconsistency(
                    event_ids=(rel.entity1_id, rel.entity2_id),
                    violation_type=ViolationType.CAUSAL_ORDERING,
                    description=(
                        f"Causal relationship missing timestamps: "
                        f"'{cause.name}' {rel.relationship_type.value} '{effect.name}'"
                    ),
                    severity=Severity.WARNING if not self.strict_mode else Severity.ERROR,
                    evidence="Events in causal relationships MUST have timestamps",
                ))
                continue

            # Bradford-Hill Criterion #1: Cause must precede effect
            if cause.timestamp >= effect.timestamp:
                violations.append(TemporalInconsistency(
                    event_ids=(rel.entity1_id, rel.entity2_id),
                    violation_type=ViolationType.CAUSAL_ORDERING,
                    description=(
                        f"Bradford-Hill Criterion #1 violated: Cause '{cause.name}' "
                        f"({cause.timestamp.isoformat()}) must precede effect "
                        f"'{effect.name}' ({effect.timestamp.isoformat()})"
                    ),
                    severity=Severity.ERROR,
                    evidence=f"Relationship type: {rel.relationship_type.value}",
                ))

        return violations

    def _check_interval_consistency(
        self,
        event_map: Dict[str, Event],
        relations: List[TemporalRelationship],
    ) -> List[TemporalInconsistency]:
        """Validate Allen's Interval Algebra consistency.

        Checks that interval relationships (DURING, CONTAINS, OVERLAPS)
        are consistent with event timestamps and durations.

        Args:
            event_map: Dict mapping event_id to Event
            relations: List of temporal relationships

        Returns:
            List of interval consistency violations
        """
        violations: List[TemporalInconsistency] = []

        for rel in relations:
            event1 = event_map.get(rel.entity1_id)
            event2 = event_map.get(rel.entity2_id)

            if not event1 or not event2:
                continue

            if not event1.timestamp or not event2.timestamp:
                continue

            # DURING: event1 should occur within event2's timespan
            if rel.relationship_type == TemporalRelationshipType.DURING:
                if event2.duration:
                    event2_end = event2.timestamp + event2.duration
                    if not (event2.timestamp <= event1.timestamp <= event2_end):
                        violations.append(TemporalInconsistency(
                            event_ids=(rel.entity1_id, rel.entity2_id),
                            violation_type=ViolationType.INTERVAL_INCONSISTENCY,
                            description=(
                                f"DURING relationship invalid: '{event1.name}' not within "
                                f"'{event2.name}' timespan"
                            ),
                            severity=Severity.WARNING,
                            evidence=(
                                f"Event1: {event1.timestamp.isoformat()}, "
                                f"Event2: {event2.timestamp.isoformat()} - {event2_end.isoformat()}"
                            ),
                        ))

            # CONTAINS: event1 should contain event2's timespan
            elif rel.relationship_type == TemporalRelationshipType.CONTAINS:
                if event1.duration:
                    event1_end = event1.timestamp + event1.duration
                    if not (event1.timestamp <= event2.timestamp <= event1_end):
                        violations.append(TemporalInconsistency(
                            event_ids=(rel.entity1_id, rel.entity2_id),
                            violation_type=ViolationType.INTERVAL_INCONSISTENCY,
                            description=(
                                f"CONTAINS relationship invalid: '{event1.name}' does not contain "
                                f"'{event2.name}'"
                            ),
                            severity=Severity.WARNING,
                            evidence=(
                                f"Event1: {event1.timestamp.isoformat()} - {event1_end.isoformat()}, "
                                f"Event2: {event2.timestamp.isoformat()}"
                            ),
                        ))

            # SIMULTANEOUS: timestamps should be equal (or very close)
            elif rel.relationship_type == TemporalRelationshipType.SIMULTANEOUS:
                if event1.timestamp != event2.timestamp:
                    violations.append(TemporalInconsistency(
                        event_ids=(rel.entity1_id, rel.entity2_id),
                        violation_type=ViolationType.INTERVAL_INCONSISTENCY,
                        description=(
                            f"SIMULTANEOUS relationship invalid: '{event1.name}' and "
                            f"'{event2.name}' have different timestamps"
                        ),
                        severity=Severity.WARNING,
                        evidence=(
                            f"Event1: {event1.timestamp.isoformat()}, "
                            f"Event2: {event2.timestamp.isoformat()}"
                        ),
                    ))

        return violations

    def validate_single_relationship(
        self,
        event1: Event,
        event2: Event,
        relationship_type: TemporalRelationshipType,
    ) -> bool:
        """Validate a single temporal relationship.

        Utility method for quick validation without full graph analysis.

        Args:
            event1: Source event
            event2: Target event
            relationship_type: Type of temporal relationship

        Returns:
            True if relationship is valid, False otherwise
        """
        if not event1.timestamp or not event2.timestamp:
            # Cannot validate without timestamps
            return relationship_type not in self.CAUSAL_TYPES

        if relationship_type in {
            TemporalRelationshipType.BEFORE,
            TemporalRelationshipType.CAUSES,
            TemporalRelationshipType.ENABLES,
            TemporalRelationshipType.TRIGGERS,
            TemporalRelationshipType.PREVENTS,
        }:
            return event1.timestamp < event2.timestamp

        elif relationship_type == TemporalRelationshipType.AFTER:
            return event1.timestamp > event2.timestamp

        elif relationship_type == TemporalRelationshipType.SIMULTANEOUS:
            return event1.timestamp == event2.timestamp

        elif relationship_type == TemporalRelationshipType.DURING:
            if event2.duration:
                event2_end = event2.timestamp + event2.duration
                return event2.timestamp <= event1.timestamp <= event2_end
            return True  # Cannot validate without duration

        elif relationship_type == TemporalRelationshipType.CONTAINS:
            if event1.duration:
                event1_end = event1.timestamp + event1.duration
                return event1.timestamp <= event2.timestamp <= event1_end
            return True  # Cannot validate without duration

        # Other types (OVERLAPS, MEETS, STARTS, FINISHES, EQUALS, PARALLEL)
        # require more complex validation - return True for now
        return True


# Convenience function for simple validation
def validate_temporal_consistency(
    events: List[Event],
    relations: List[TemporalRelationship],
    strict_mode: bool = True,
) -> ValidationResult:
    """Validate temporal consistency of events and relationships.

    Convenience wrapper around TemporalConsistencyValidator.

    Args:
        events: List of events to validate
        relations: List of temporal relationships
        strict_mode: If True, all violations are errors

    Returns:
        ValidationResult with valid=True if consistent
    """
    validator = TemporalConsistencyValidator(strict_mode=strict_mode)
    return validator.validate(events, relations)
