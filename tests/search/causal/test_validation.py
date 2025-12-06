"""Tests for temporal ordering validation.

Tests TemporalOrderingValidator for Option B compliance.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md

Option B Compliance:
- 100% temporal validation for ALL causal paths
- Bradford Hill criterion 1 (temporality) enforcement
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from futurnal.search.causal.validation import TemporalOrderingValidator
from futurnal.search.causal.exceptions import TemporalOrderingViolation


class TestTemporalOrderingValidator:
    """Test TemporalOrderingValidator."""

    @pytest.fixture
    def mock_pkg(self):
        """Create mock PKG queries."""
        mock = MagicMock()
        mock._db = MagicMock()
        return mock

    @pytest.fixture
    def validator(self, mock_pkg):
        """Create validator with mock PKG."""
        return TemporalOrderingValidator(mock_pkg)

    def test_empty_path_valid(self, validator):
        """Test empty path is valid."""
        assert validator.validate_path([]) is True

    def test_single_event_valid(self, validator):
        """Test single event path is valid."""
        assert validator.validate_path(["e1"]) is True

    def test_valid_temporal_ordering(self, validator, mock_pkg):
        """Test valid temporal ordering passes."""
        # Setup mock to return timestamps in correct order
        timestamps = {
            "e1": datetime(2024, 1, 1, 10, 0),
            "e2": datetime(2024, 1, 5, 14, 0),
            "e3": datetime(2024, 1, 15, 9, 0),
        }

        def mock_query(query, params):
            records = []
            for event_id in params.get("event_ids", []):
                if event_id in timestamps:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, eid=event_id: (
                        eid if k == "id" else timestamps[eid]
                    )
                    records.append(record)
            result = MagicMock()
            result.__iter__ = lambda _: iter(records)
            return result

        mock_session = MagicMock()
        mock_session.run = mock_query
        mock_pkg._db.session.return_value.__enter__.return_value = mock_session

        # Test validation
        assert validator.validate_path(["e1", "e2", "e3"]) is True

    def test_invalid_temporal_ordering(self, validator, mock_pkg):
        """Test invalid temporal ordering fails."""
        # Setup mock with e2 before e1 (violation)
        timestamps = {
            "e1": datetime(2024, 1, 10, 10, 0),  # Later
            "e2": datetime(2024, 1, 5, 14, 0),   # Earlier
        }

        def mock_query(query, params):
            records = []
            for event_id in params.get("event_ids", []):
                if event_id in timestamps:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, eid=event_id: (
                        eid if k == "id" else timestamps[eid]
                    )
                    records.append(record)
            result = MagicMock()
            result.__iter__ = lambda _: iter(records)
            return result

        mock_session = MagicMock()
        mock_session.run = mock_query
        mock_pkg._db.session.return_value.__enter__.return_value = mock_session

        # Path e1 -> e2 should fail because e1 is after e2
        assert validator.validate_path(["e1", "e2"]) is False

    def test_raise_on_violation(self, validator, mock_pkg):
        """Test raise_on_violation parameter."""
        # Setup mock with invalid ordering
        timestamps = {
            "e1": datetime(2024, 1, 10, 10, 0),
            "e2": datetime(2024, 1, 5, 14, 0),
        }

        def mock_query(query, params):
            records = []
            for event_id in params.get("event_ids", []):
                if event_id in timestamps:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, eid=event_id: (
                        eid if k == "id" else timestamps[eid]
                    )
                    records.append(record)
            result = MagicMock()
            result.__iter__ = lambda _: iter(records)
            return result

        mock_session = MagicMock()
        mock_session.run = mock_query
        mock_pkg._db.session.return_value.__enter__.return_value = mock_session

        with pytest.raises(TemporalOrderingViolation) as exc_info:
            validator.validate_path(["e1", "e2"], raise_on_violation=True)

        assert exc_info.value.path == ["e1", "e2"]
        assert exc_info.value.violation_index == 0

    def test_missing_timestamp_treated_as_valid(self, validator, mock_pkg):
        """Test missing timestamps are treated as valid (benefit of doubt)."""
        # Setup mock with one missing timestamp
        timestamps = {
            "e1": datetime(2024, 1, 1, 10, 0),
            # "e2" missing
            "e3": datetime(2024, 1, 15, 9, 0),
        }

        def mock_query(query, params):
            records = []
            for event_id in params.get("event_ids", []):
                if event_id in timestamps:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, eid=event_id: (
                        eid if k == "id" else timestamps[eid]
                    )
                    records.append(record)
            result = MagicMock()
            result.__iter__ = lambda _: iter(records)
            return result

        mock_session = MagicMock()
        mock_session.run = mock_query
        mock_pkg._db.session.return_value.__enter__.return_value = mock_session

        # Should still be valid because we give benefit of doubt
        assert validator.validate_path(["e1", "e2", "e3"]) is True

    def test_batch_validation(self, validator, mock_pkg):
        """Test batch validation of multiple paths."""
        timestamps = {
            "e1": datetime(2024, 1, 1, 10, 0),
            "e2": datetime(2024, 1, 5, 14, 0),
            "e3": datetime(2024, 1, 15, 9, 0),
            "e4": datetime(2024, 1, 3, 10, 0),  # Between e1 and e2
        }

        def mock_query(query, params):
            records = []
            for event_id in params.get("event_ids", []):
                if event_id in timestamps:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, eid=event_id: (
                        eid if k == "id" else timestamps[eid]
                    )
                    records.append(record)
            result = MagicMock()
            result.__iter__ = lambda _: iter(records)
            return result

        mock_session = MagicMock()
        mock_session.run = mock_query
        mock_pkg._db.session.return_value.__enter__.return_value = mock_session

        paths = [
            ["e1", "e2", "e3"],  # Valid
            ["e1", "e4", "e2"],  # Valid (e1 < e4 < e2)
            ["e2", "e1"],        # Invalid (e2 > e1)
        ]

        results = validator.validate_paths_batch(paths)

        assert results[0] == (["e1", "e2", "e3"], True)
        assert results[1] == (["e1", "e4", "e2"], True)
        assert results[2] == (["e2", "e1"], False)

    def test_equal_timestamps_invalid(self, validator, mock_pkg):
        """Test equal timestamps are treated as invalid (need strict ordering)."""
        same_time = datetime(2024, 1, 5, 14, 0)
        timestamps = {
            "e1": same_time,
            "e2": same_time,
        }

        def mock_query(query, params):
            records = []
            for event_id in params.get("event_ids", []):
                if event_id in timestamps:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, eid=event_id: (
                        eid if k == "id" else timestamps[eid]
                    )
                    records.append(record)
            result = MagicMock()
            result.__iter__ = lambda _: iter(records)
            return result

        mock_session = MagicMock()
        mock_session.run = mock_query
        mock_pkg._db.session.return_value.__enter__.return_value = mock_session

        # Equal timestamps should fail (need cause < effect)
        assert validator.validate_path(["e1", "e2"]) is False

    def test_timezone_handling(self, validator, mock_pkg):
        """Test timezone-aware timestamps are normalized."""
        from datetime import timezone

        # One naive, one with timezone
        timestamps = {
            "e1": datetime(2024, 1, 1, 10, 0),
            "e2": datetime(2024, 1, 5, 14, 0, tzinfo=timezone.utc),
        }

        def mock_query(query, params):
            records = []
            for event_id in params.get("event_ids", []):
                if event_id in timestamps:
                    record = MagicMock()
                    record.__getitem__ = lambda _, k, eid=event_id: (
                        eid if k == "id" else timestamps[eid]
                    )
                    records.append(record)
            result = MagicMock()
            result.__iter__ = lambda _: iter(records)
            return result

        mock_session = MagicMock()
        mock_session.run = mock_query
        mock_pkg._db.session.return_value.__enter__.return_value = mock_session

        # Should handle mixed timezone/naive datetimes
        assert validator.validate_path(["e1", "e2"]) is True
