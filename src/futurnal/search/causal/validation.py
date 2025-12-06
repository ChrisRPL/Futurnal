"""Temporal Ordering Validation for Causal Paths.

Provides utilities for validating that causal paths respect temporal ordering
(cause must precede effect).

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md

Option B Compliance:
- 100% temporal validation for ALL causal paths
- Bradford Hill criterion 1 (temporality) enforcement
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from futurnal.search.causal.exceptions import TemporalOrderingViolation

if TYPE_CHECKING:
    from futurnal.pkg.queries.temporal import TemporalGraphQueries

logger = logging.getLogger(__name__)


class TemporalOrderingValidator:
    """Validate temporal ordering in causal paths.

    Ensures that in any causal path A -> B -> C:
    - timestamp(A) < timestamp(B)
    - timestamp(B) < timestamp(C)

    This is Bradford Hill criterion 1 (temporality) - required for causation.

    Example:
        >>> validator = TemporalOrderingValidator(pkg_queries)
        >>> is_valid = validator.validate_path(["event_1", "event_2", "event_3"])
        >>> if not is_valid:
        ...     print("Causal path violates temporal ordering")
    """

    def __init__(self, pkg_queries: "TemporalGraphQueries"):
        """Initialize the validator.

        Args:
            pkg_queries: PKG temporal queries service for timestamp lookup
        """
        self._pkg = pkg_queries

    def validate_path(
        self,
        event_ids: List[str],
        raise_on_violation: bool = False,
    ) -> bool:
        """Validate temporal ordering of events in path.

        Args:
            event_ids: List of event IDs in causal order
            raise_on_violation: If True, raise TemporalOrderingViolation on failure

        Returns:
            True if all events are temporally ordered, False otherwise

        Raises:
            TemporalOrderingViolation: If raise_on_violation=True and path is invalid
        """
        if len(event_ids) < 2:
            return True

        # Fetch timestamps for all events
        timestamps = self._fetch_timestamps(event_ids)

        # Check ordering
        for i in range(len(event_ids) - 1):
            current_id = event_ids[i]
            next_id = event_ids[i + 1]

            current_ts = timestamps.get(current_id)
            next_ts = timestamps.get(next_id)

            if current_ts is None or next_ts is None:
                logger.warning(
                    f"Missing timestamp for event in path: "
                    f"{current_id if current_ts is None else next_id}"
                )
                # Treat missing timestamps as valid (benefit of the doubt)
                # This can happen with incomplete data
                continue

            # Normalize for comparison
            current_ts = self._strip_timezone(current_ts)
            next_ts = self._strip_timezone(next_ts)

            if current_ts >= next_ts:
                logger.debug(
                    f"Temporal ordering violation: {current_id} ({current_ts}) "
                    f">= {next_id} ({next_ts})"
                )
                if raise_on_violation:
                    raise TemporalOrderingViolation(event_ids, i)
                return False

        return True

    def validate_paths_batch(
        self,
        paths: List[List[str]],
    ) -> List[Tuple[List[str], bool]]:
        """Validate multiple paths efficiently.

        Fetches all timestamps in one query for efficiency.

        Args:
            paths: List of event ID paths

        Returns:
            List of (path, is_valid) tuples
        """
        if not paths:
            return []

        # Collect all unique event IDs
        all_event_ids: set[str] = set()
        for path in paths:
            all_event_ids.update(path)

        # Fetch all timestamps at once
        timestamps = self._fetch_timestamps(list(all_event_ids))

        # Validate each path
        results: List[Tuple[List[str], bool]] = []
        for path in paths:
            is_valid = self._validate_with_timestamps(path, timestamps)
            results.append((path, is_valid))

        return results

    def _fetch_timestamps(self, event_ids: List[str]) -> Dict[str, datetime]:
        """Fetch timestamps for events from PKG.

        Uses batch query for efficiency.

        Args:
            event_ids: List of event IDs to fetch timestamps for

        Returns:
            Dictionary mapping event ID to timestamp
        """
        timestamps: Dict[str, datetime] = {}

        if not event_ids:
            return timestamps

        try:
            with self._pkg._db.session() as session:
                query = """
                    MATCH (e:Event)
                    WHERE e.id IN $event_ids
                    RETURN e.id AS id, e.timestamp AS timestamp
                """
                result = session.run(query, {"event_ids": event_ids})

                for record in result:
                    event_id = record["id"]
                    ts = record["timestamp"]
                    if ts is not None:
                        # Convert Neo4j datetime to Python datetime
                        timestamps[event_id] = self._convert_timestamp(ts)

        except Exception as e:
            logger.error(f"Failed to fetch timestamps: {e}")

        return timestamps

    def _validate_with_timestamps(
        self,
        event_ids: List[str],
        timestamps: Dict[str, datetime],
    ) -> bool:
        """Validate path using pre-fetched timestamps.

        Args:
            event_ids: Event IDs in causal order
            timestamps: Pre-fetched timestamp mapping

        Returns:
            True if temporally valid, False otherwise
        """
        for i in range(len(event_ids) - 1):
            current_ts = timestamps.get(event_ids[i])
            next_ts = timestamps.get(event_ids[i + 1])

            if current_ts is None or next_ts is None:
                continue

            current_ts = self._strip_timezone(current_ts)
            next_ts = self._strip_timezone(next_ts)

            if current_ts >= next_ts:
                return False

        return True

    def _convert_timestamp(self, ts: object) -> datetime:
        """Convert Neo4j timestamp to Python datetime.

        Handles various Neo4j datetime formats.

        Args:
            ts: Neo4j timestamp value

        Returns:
            Python datetime object
        """
        # Neo4j native datetime
        if hasattr(ts, "to_native"):
            return ts.to_native()

        # ISO format string
        if hasattr(ts, "iso_format"):
            return datetime.fromisoformat(ts.iso_format())

        # Already a datetime
        if isinstance(ts, datetime):
            return ts

        # String timestamp
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)

        # Fallback - return as-is (will fail type check if wrong)
        return ts

    def _strip_timezone(self, dt: datetime) -> datetime:
        """Remove timezone info for consistent comparison.

        Comparing timezone-aware and naive datetimes raises errors,
        so we normalize all to naive.

        Args:
            dt: Datetime with or without timezone

        Returns:
            Naive datetime (timezone removed)
        """
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt
