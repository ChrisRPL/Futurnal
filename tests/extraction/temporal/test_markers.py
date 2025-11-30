"""Unit tests for temporal marker extraction (Week 1).

Tests for:
- Explicit timestamp detection (target: >95% accuracy)
- Relative time expression parsing (target: >85% accuracy)
- Document metadata temporal inference

Production plan reference:
docs/phase-1/entity-relationship-extraction-production-plan/01-temporal-extraction.md
"""

from datetime import datetime, timedelta

import pytest

from futurnal.extraction.temporal.markers import TemporalMarkerExtractor
from futurnal.extraction.temporal.models import TemporalSourceType


# ---------------------------------------------------------------------------
# Explicit Timestamp Detection Tests
# ---------------------------------------------------------------------------


class TestExplicitTimestampDetection:
    """Tests for explicit timestamp extraction.

    Target: >95% accuracy on explicit timestamps
    """

    def test_iso8601_basic_date(self):
        """Test ISO 8601 basic date format (2024-01-15)."""
        extractor = TemporalMarkerExtractor()
        text = "The meeting was scheduled for 2024-01-15"

        markers = extractor.extract_explicit_timestamps(text)

        assert len(markers) == 1
        assert markers[0].text == "2024-01-15"
        assert markers[0].timestamp == datetime(2024, 1, 15)
        assert markers[0].temporal_type == TemporalSourceType.EXPLICIT
        assert markers[0].confidence >= 0.95

    def test_iso8601_datetime_with_timezone(self):
        """Test ISO 8601 with timezone (2024-01-15T14:30:00Z)."""
        extractor = TemporalMarkerExtractor()
        text = "Deadline: 2024-01-15T14:30:00Z for submission"

        markers = extractor.extract_explicit_timestamps(text)

        assert len(markers) == 1
        assert markers[0].text == "2024-01-15T14:30:00Z"
        assert markers[0].timestamp.year == 2024
        assert markers[0].timestamp.month == 1
        assert markers[0].timestamp.day == 15
        assert markers[0].timestamp.hour == 14
        assert markers[0].timestamp.minute == 30
        assert markers[0].confidence >= 0.95

    def test_iso8601_with_offset_timezone(self):
        """Test ISO 8601 with offset timezone (+00:00)."""
        extractor = TemporalMarkerExtractor()
        text = "Meeting at 2024-03-20T09:00:00+00:00"

        markers = extractor.extract_explicit_timestamps(text)

        assert len(markers) == 1
        assert markers[0].timestamp.year == 2024
        assert markers[0].timestamp.month == 3
        assert markers[0].timestamp.day == 20

    def test_natural_language_date_month_day_year(self):
        """Test natural language date (January 15, 2024)."""
        extractor = TemporalMarkerExtractor()
        text = "On January 15, 2024, we completed the project"

        markers = extractor.extract_explicit_timestamps(text)

        # Should extract at least one timestamp
        assert len(markers) >= 1

        # Check if we found the correct date
        found_correct_date = any(
            m.timestamp and
            m.timestamp.year == 2024 and
            m.timestamp.month == 1 and
            m.timestamp.day == 15
            for m in markers
        )
        assert found_correct_date

    def test_time_expression_12hour_am(self):
        """Test 12-hour time format AM (9:00 AM)."""
        extractor = TemporalMarkerExtractor()
        text = "Call scheduled for 9:00 AM tomorrow"

        markers = extractor.extract_explicit_timestamps(text)

        # Find time marker
        time_markers = [m for m in markers if "9:00 AM" in m.text]
        assert len(time_markers) >= 1

        time_marker = time_markers[0]
        assert time_marker.timestamp.hour == 9
        assert time_marker.timestamp.minute == 0

    def test_time_expression_12hour_pm(self):
        """Test 12-hour time format PM (2:30 PM)."""
        extractor = TemporalMarkerExtractor()
        text = "Meeting at 2:30 PM in conference room"

        markers = extractor.extract_explicit_timestamps(text)

        # Find time marker
        time_markers = [m for m in markers if "2:30 PM" in m.text]
        assert len(time_markers) >= 1

        time_marker = time_markers[0]
        assert time_marker.timestamp.hour == 14  # 2 PM = 14:00
        assert time_marker.timestamp.minute == 30

    def test_time_expression_24hour(self):
        """Test 24-hour time format (14:30)."""
        extractor = TemporalMarkerExtractor()
        text = "Server restart at 14:30 today"

        markers = extractor.extract_explicit_timestamps(text)

        # Find time marker
        time_markers = [m for m in markers if "14:30" in m.text]
        assert len(time_markers) >= 1

        time_marker = time_markers[0]
        assert time_marker.timestamp.hour == 14
        assert time_marker.timestamp.minute == 30

    def test_multiple_timestamps_in_text(self):
        """Test extraction of multiple timestamps from same text."""
        extractor = TemporalMarkerExtractor()
        text = "Meeting on 2024-01-15 at 2:30 PM, follow-up on 2024-01-20"

        markers = extractor.extract_explicit_timestamps(text)

        # Should extract at least 2 timestamps (2 dates, possibly the time)
        assert len(markers) >= 2

        # Check for both dates
        dates = [m.text for m in markers]
        assert "2024-01-15" in dates
        assert "2024-01-20" in dates

    def test_no_false_positives_on_numbers(self):
        """Test that standalone numbers don't trigger extraction."""
        extractor = TemporalMarkerExtractor()
        text = "The project cost $2024 and took 15 days"

        markers = extractor.extract_explicit_timestamps(text)

        # Should not extract standalone numbers as timestamps
        # (though some natural language parsers might be aggressive)
        # At minimum, confidence should be lower if extracted
        for marker in markers:
            if marker.timestamp and marker.timestamp.year == 2024:
                # If extracted, confidence should be lower
                assert marker.confidence < 0.95

    def test_extraction_accuracy_on_test_corpus(self, explicit_timestamp_test_cases, accuracy_calculator):
        """Test overall accuracy on test corpus.

        Target: >95% accuracy
        """
        extractor = TemporalMarkerExtractor()

        total_accuracy = []

        for text, expected in explicit_timestamp_test_cases:
            markers = extractor.extract_explicit_timestamps(text)
            accuracy = accuracy_calculator.calculate_timestamp_accuracy(markers, expected)
            total_accuracy.append(accuracy)

        overall_accuracy = sum(total_accuracy) / len(total_accuracy)

        # Target: >95% accuracy
        assert overall_accuracy > 0.95, f"Accuracy {overall_accuracy:.2%} below 95% target"


# ---------------------------------------------------------------------------
# Relative Expression Parsing Tests
# ---------------------------------------------------------------------------


class TestRelativeExpressionParsing:
    """Tests for relative time expression parsing.

    Target: >85% accuracy on relative expressions
    """

    def test_relative_day_yesterday(self):
        """Test parsing 'yesterday'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("yesterday", reference)

        assert marker is not None
        assert marker.timestamp == datetime(2024, 1, 14, 14, 30)
        assert marker.temporal_type == TemporalSourceType.RELATIVE
        assert marker.confidence >= 0.90

    def test_relative_day_today(self):
        """Test parsing 'today'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("today", reference)

        assert marker is not None
        assert marker.timestamp == reference
        assert marker.temporal_type == TemporalSourceType.RELATIVE

    def test_relative_day_tomorrow(self):
        """Test parsing 'tomorrow'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("tomorrow", reference)

        assert marker is not None
        assert marker.timestamp == datetime(2024, 1, 16, 14, 30)

    def test_relative_week_last_week(self):
        """Test parsing 'last week'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("last week", reference)

        assert marker is not None
        expected = reference - timedelta(weeks=1)
        assert marker.timestamp == expected

    def test_relative_week_next_week(self):
        """Test parsing 'next week'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("next week", reference)

        assert marker is not None
        expected = reference + timedelta(weeks=1)
        assert marker.timestamp == expected

    def test_duration_ago_days(self):
        """Test parsing 'X days ago'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("2 days ago", reference)

        assert marker is not None
        expected = reference - timedelta(days=2)
        assert marker.timestamp == expected

    def test_duration_ago_weeks(self):
        """Test parsing 'X weeks ago'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("3 weeks ago", reference)

        assert marker is not None
        expected = reference - timedelta(weeks=3)
        assert marker.timestamp == expected

    def test_duration_in_future_days(self):
        """Test parsing 'in X days'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("in 3 days", reference)

        assert marker is not None
        expected = reference + timedelta(days=3)
        assert marker.timestamp == expected

    def test_duration_in_future_weeks(self):
        """Test parsing 'in X weeks'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("in 2 weeks", reference)

        assert marker is not None
        expected = reference + timedelta(weeks=2)
        assert marker.timestamp == expected

    def test_relative_month_last_month(self):
        """Test parsing 'last month'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 2, 15, 14, 30)

        marker = extractor.parse_relative_expression("last month", reference)

        assert marker is not None
        # Should be approximately 1 month earlier
        # (exact delta depends on month length)
        assert marker.timestamp.month == 1
        assert marker.timestamp.year == 2024

    def test_relative_month_next_month(self):
        """Test parsing 'next month'."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("next month", reference)

        assert marker is not None
        assert marker.timestamp.month == 2
        assert marker.timestamp.year == 2024

    def test_unparseable_expression_returns_none(self):
        """Test that unparseable expressions return None."""
        extractor = TemporalMarkerExtractor()
        reference = datetime(2024, 1, 15, 14, 30)

        marker = extractor.parse_relative_expression("some random text", reference)

        assert marker is None

    def test_relative_accuracy_on_test_corpus(self, relative_expression_test_cases, accuracy_calculator):
        """Test overall accuracy on relative expressions.

        Target: >85% accuracy
        """
        extractor = TemporalMarkerExtractor()

        correct = 0
        total = 0

        for expr, reference, expected_offset_days in relative_expression_test_cases:
            marker = extractor.parse_relative_expression(expr, reference)

            if marker and marker.timestamp:
                is_correct = accuracy_calculator.calculate_relative_accuracy(
                    marker.timestamp,
                    reference,
                    expected_offset_days,
                    tolerance_days=2  # Allow 2-day tolerance for month approximations
                )
                if is_correct:
                    correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0.0

        # Target: >85% accuracy
        assert accuracy > 0.85, f"Accuracy {accuracy:.2%} below 85% target"


# ---------------------------------------------------------------------------
# Document Metadata Inference Tests
# ---------------------------------------------------------------------------


class TestDocumentMetadataInference:
    """Tests for document metadata temporal inference."""

    def test_infer_from_frontmatter_created(self):
        """Test inference from frontmatter 'created' field."""
        extractor = TemporalMarkerExtractor()
        metadata = {
            "frontmatter": {
                "created": "2024-01-15",
                "title": "My Note"
            }
        }

        timestamp = extractor.infer_from_document_metadata(metadata)

        assert timestamp is not None
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 15

    def test_infer_from_frontmatter_date(self):
        """Test inference from frontmatter 'date' field."""
        extractor = TemporalMarkerExtractor()
        metadata = {
            "frontmatter": {
                "date": "2024-03-20",
            }
        }

        timestamp = extractor.infer_from_document_metadata(metadata)

        assert timestamp is not None
        assert timestamp.year == 2024
        assert timestamp.month == 3
        assert timestamp.day == 20

    def test_infer_from_file_created_at(self):
        """Test inference from file 'created_at' timestamp."""
        extractor = TemporalMarkerExtractor()
        created = datetime(2024, 2, 10, 8, 30)
        metadata = {
            "created_at": created,
            "modified_at": datetime(2024, 2, 15, 14, 20)
        }

        timestamp = extractor.infer_from_document_metadata(metadata)

        assert timestamp == created  # Should prefer created_at

    def test_infer_from_file_modified_at(self):
        """Test inference from file 'modified_at' when created_at missing."""
        extractor = TemporalMarkerExtractor()
        modified = datetime(2024, 2, 15, 14, 20)
        metadata = {
            "modified_at": modified
        }

        timestamp = extractor.infer_from_document_metadata(metadata)

        assert timestamp == modified

    def test_infer_from_email_headers(self):
        """Test inference from email 'Date' header."""
        extractor = TemporalMarkerExtractor()
        metadata = {
            "email_date": "2024-01-15T14:30:00Z"
        }

        timestamp = extractor.infer_from_document_metadata(metadata)

        assert timestamp is not None
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 15

    def test_infer_from_git_commit(self):
        """Test inference from git commit timestamp."""
        extractor = TemporalMarkerExtractor()
        metadata = {
            "git_commit_timestamp": "2024-01-20T16:45:00Z"
        }

        timestamp = extractor.infer_from_document_metadata(metadata)

        assert timestamp is not None
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 20

    def test_no_metadata_returns_none(self):
        """Test that missing metadata returns None."""
        extractor = TemporalMarkerExtractor()
        metadata = {
            "title": "My Note",
            "tags": ["work", "meeting"]
        }

        timestamp = extractor.infer_from_document_metadata(metadata)

        assert timestamp is None

    def test_metadata_inference_accuracy(self, document_metadata_test_cases):
        """Test accuracy on document metadata inference."""
        extractor = TemporalMarkerExtractor()

        correct = 0
        total = len(document_metadata_test_cases)

        for metadata, expected in document_metadata_test_cases:
            result = extractor.infer_from_document_metadata(metadata)

            if expected is None:
                if result is None:
                    correct += 1
            else:
                if result and abs((result - expected).total_seconds()) < 1:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        # Target: >90% accuracy (metadata inference should be highly reliable)
        assert accuracy > 0.90, f"Metadata inference accuracy {accuracy:.2%} below 90% target"


# ---------------------------------------------------------------------------
# Production Readiness Tests
# ---------------------------------------------------------------------------


@pytest.mark.production_readiness
class TestTemporalMarkerProductionReadiness:
    """Production readiness tests for temporal marker extraction.

    Validates that Week 1 success metrics are met:
    - >95% explicit timestamp accuracy
    - >85% relative expression accuracy
    """

    def test_overall_explicit_accuracy_target(self, explicit_timestamp_test_cases, accuracy_calculator):
        """Validate >95% explicit timestamp accuracy (PRODUCTION GATE)."""
        extractor = TemporalMarkerExtractor()

        total_accuracy = []

        for text, expected in explicit_timestamp_test_cases:
            markers = extractor.extract_explicit_timestamps(text)
            accuracy = accuracy_calculator.calculate_timestamp_accuracy(markers, expected)
            total_accuracy.append(accuracy)

        overall_accuracy = sum(total_accuracy) / len(total_accuracy)

        # PRODUCTION GATE: >95% accuracy
        assert overall_accuracy > 0.95, (
            f"PRODUCTION GATE FAILED: Explicit timestamp accuracy {overall_accuracy:.2%} "
            f"below 95% target"
        )

    def test_overall_relative_accuracy_target(self, relative_expression_test_cases, accuracy_calculator):
        """Validate >85% relative expression accuracy (PRODUCTION GATE)."""
        extractor = TemporalMarkerExtractor()

        correct = 0
        total = 0

        for expr, reference, expected_offset_days in relative_expression_test_cases:
            marker = extractor.parse_relative_expression(expr, reference)

            if marker and marker.timestamp:
                is_correct = accuracy_calculator.calculate_relative_accuracy(
                    marker.timestamp,
                    reference,
                    expected_offset_days,
                    tolerance_days=2
                )
                if is_correct:
                    correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0.0

        # PRODUCTION GATE: >85% accuracy
        assert accuracy > 0.85, (
            f"PRODUCTION GATE FAILED: Relative expression accuracy {accuracy:.2%} "
            f"below 85% target"
        )

    def test_no_crashes_on_edge_cases(self):
        """Test robustness on edge cases."""
        extractor = TemporalMarkerExtractor()

        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "No timestamps at all!",  # No temporal content
            "🎉 Unicode emoji content 🚀",  # Unicode
            "Very " * 1000 + "long text",  # Very long text
            "2024-13-45",  # Invalid date
            "99:99 PM",  # Invalid time
        ]

        for case in edge_cases:
            try:
                markers = extractor.extract_explicit_timestamps(case)
                # Should not crash, even if returns empty list
                assert isinstance(markers, list)
            except Exception as e:
                pytest.fail(f"Crashed on edge case '{case[:50]}...': {e}")
