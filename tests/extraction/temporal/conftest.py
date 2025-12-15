"""Test fixtures for temporal extraction tests.

Provides:
- Mock data for temporal marker extraction
- Golden dataset samples
- Test utilities for accuracy measurement
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any

import pytest

from futurnal.extraction.temporal.models import (
    TemporalMark,
    TemporalSourceType,
)


# ---------------------------------------------------------------------------
# Explicit Timestamp Test Cases
# ---------------------------------------------------------------------------

@pytest.fixture
def explicit_timestamp_test_cases():
    """Test cases for explicit timestamp extraction.

    Target: >95% accuracy

    Returns:
        List of (text, expected_timestamps) tuples
    """
    return [
        # ISO 8601 formats
        (
            "The meeting was scheduled for 2024-01-15",
            [{"text": "2024-01-15", "timestamp": datetime(2024, 1, 15)}]
        ),
        (
            "Deadline: 2024-01-15T14:30:00Z",
            [{"text": "2024-01-15T14:30:00Z", "timestamp": datetime(2024, 1, 15, 14, 30, 0)}]
        ),
        (
            "Meeting at 2024-03-20T09:00:00+00:00 in conference room",
            [{"text": "2024-03-20T09:00:00+00:00", "timestamp": datetime(2024, 3, 20, 9, 0, 0)}]
        ),

        # Natural language dates
        (
            "On January 15, 2024, we completed the project",
            [{"text": "January 15, 2024", "timestamp": datetime(2024, 1, 15)}]
        ),
        (
            "The deadline is March 20th, 2024",
            [{"text": "March 20th, 2024", "timestamp": datetime(2024, 3, 20)}]
        ),

        # Time expressions - 12-hour format
        (
            "Meeting at 2:30 PM",
            [{"text": "at 2:30 PM", "hour": 14, "minute": 30}]
        ),
        (
            "Call scheduled for 9:00 AM",
            [{"text": "9:00 AM", "hour": 9, "minute": 0}]
        ),

        # Time expressions - 24-hour format
        (
            "Server restart at 14:30",
            [{"text": "14:30", "hour": 14, "minute": 30}]
        ),

        # Multiple timestamps in one text
        (
            "Meeting on 2024-01-15 at 2:30 PM, follow-up on 2024-01-20",
            [
                {"text": "2024-01-15", "timestamp": datetime(2024, 1, 15)},
                {"text": "at 2:30 PM", "hour": 14, "minute": 30},
                {"text": "2024-01-20", "timestamp": datetime(2024, 1, 20)},
            ]
        ),

        # Edge cases
        (
            "No timestamps here, just numbers like 2024 and 15",
            []  # Should not extract standalone numbers
        ),
    ]


# ---------------------------------------------------------------------------
# Relative Expression Test Cases
# ---------------------------------------------------------------------------

@pytest.fixture
def relative_expression_test_cases():
    """Test cases for relative time expression parsing.

    Target: >85% accuracy

    Returns:
        List of (expression, reference_time, expected_offset_days) tuples
    """
    reference = datetime(2024, 1, 15, 14, 30)  # Fixed reference time

    return [
        # Relative days
        ("yesterday", reference, -1),
        ("today", reference, 0),
        ("tomorrow", reference, 1),
        ("the day before yesterday", reference, -2),
        ("the day after tomorrow", reference, 2),

        # Relative weeks
        ("last week", reference, -7),
        ("this week", reference, 0),
        ("next week", reference, 7),
        ("two weeks ago", reference, -14),
        ("in two weeks", reference, 14),

        # Relative months (approximate)
        ("last month", reference, -30),  # Approximate
        ("this month", reference, 0),
        ("next month", reference, 30),   # Approximate

        # Duration-based: X ago
        ("2 days ago", reference, -2),
        ("3 weeks ago", reference, -21),
        ("1 month ago", reference, -30),  # Approximate
        ("5 hours ago", reference, 0),    # Same day

        # Duration-based: in X
        ("in 3 days", reference, 3),
        ("in 2 weeks", reference, 14),
        ("in 1 month", reference, 30),    # Approximate

        # Edge cases
        ("5 minutes ago", reference, 0),  # Same day
        ("in 2 hours", reference, 0),     # Same day
    ]


# ---------------------------------------------------------------------------
# Document Metadata Test Cases
# ---------------------------------------------------------------------------

@pytest.fixture
def document_metadata_test_cases():
    """Test cases for document metadata temporal inference.

    Returns:
        List of (metadata, expected_timestamp) tuples
    """
    return [
        # Frontmatter with created date
        (
            {
                "frontmatter": {
                    "created": "2024-01-15",
                    "title": "My Note"
                }
            },
            datetime(2024, 1, 15)
        ),

        # Frontmatter with published date
        (
            {
                "frontmatter": {
                    "published": "2024-03-20T10:00:00Z",
                }
            },
            datetime(2024, 3, 20, 10, 0, 0)
        ),

        # File timestamps
        (
            {
                "created_at": datetime(2024, 2, 10, 8, 30),
                "modified_at": datetime(2024, 2, 15, 14, 20)
            },
            datetime(2024, 2, 10, 8, 30)  # Prefer created_at
        ),

        # Email headers
        (
            {
                "email_date": "Mon, 15 Jan 2024 14:30:00 +0000"
            },
            datetime(2024, 1, 15, 14, 30, 0)
        ),

        # Git commit timestamp
        (
            {
                "git_commit_timestamp": "2024-01-20T16:45:00Z"
            },
            datetime(2024, 1, 20, 16, 45, 0)
        ),

        # No temporal metadata
        (
            {
                "title": "My Note",
                "tags": ["work", "meeting"]
            },
            None  # Should return None
        ),
    ]


# ---------------------------------------------------------------------------
# Test Utilities
# ---------------------------------------------------------------------------

@pytest.fixture
def accuracy_calculator():
    """Utility for calculating temporal extraction accuracy."""

    class AccuracyCalculator:
        @staticmethod
        def calculate_timestamp_accuracy(
            predicted: List[TemporalMark],
            expected: List[Dict[str, Any]]
        ) -> float:
            """Calculate accuracy for timestamp extraction.

            Considers a match if:
            - Timestamps are within 1 second of each other
            - Or if hour/minute match for time-only expressions
            """
            if not expected:
                return 1.0 if not predicted else 0.0

            matches = 0
            for exp in expected:
                for pred in predicted:
                    if "timestamp" in exp:
                        # Full timestamp comparison
                        if pred.timestamp:
                            delta = abs((pred.timestamp - exp["timestamp"]).total_seconds())
                            if delta < 1:  # Within 1 second
                                matches += 1
                                break
                    elif "hour" in exp and "minute" in exp:
                        # Time-only comparison
                        if pred.timestamp:
                            if (pred.timestamp.hour == exp["hour"] and
                                pred.timestamp.minute == exp["minute"]):
                                matches += 1
                                break

            return matches / len(expected) if expected else 0.0

        @staticmethod
        def calculate_relative_accuracy(
            parsed_timestamp: datetime,
            reference_time: datetime,
            expected_offset_days: int,
            tolerance_days: int = 2
        ) -> bool:
            """Check if relative expression was parsed correctly.

            Args:
                parsed_timestamp: Parsed result
                reference_time: Reference time used
                expected_offset_days: Expected day offset
                tolerance_days: Tolerance for month approximations

            Returns:
                True if within tolerance
            """
            expected_timestamp = reference_time + timedelta(days=expected_offset_days)
            delta_days = abs((parsed_timestamp - expected_timestamp).days)

            return delta_days <= tolerance_days

    return AccuracyCalculator()


# ---------------------------------------------------------------------------
# Mock LLM for Testing
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Mock LLM for testing without actual model."""

    class MockLLM:
        def generate(self, prompt: str) -> str:
            """Mock generation - returns deterministic responses."""
            if "BEFORE" in prompt.upper():
                return '{"relationship": "BEFORE", "confidence": 0.9, "reasoning": "Mock reasoning"}'
            elif "AFTER" in prompt.upper():
                return '{"relationship": "AFTER", "confidence": 0.9, "reasoning": "Mock reasoning"}'
            else:
                return '{"relationship": "UNKNOWN", "confidence": 0.5, "reasoning": "Mock reasoning"}'

    return MockLLM()


# ---------------------------------------------------------------------------
# Golden Dataset Samples
# ---------------------------------------------------------------------------

@pytest.fixture
def golden_dataset_sample():
    """Sample from golden dataset for integration testing."""

    return {
        "document": """# Meeting Notes - Product Planning

On January 15, 2024, we held our quarterly planning meeting. The team discussed
the roadmap for Q1 2024.

## Action Items
- Review proposal (deadline: 2024-01-20)
- Schedule follow-up meeting for next week
- Send summary email yesterday evening

## Timeline
After reviewing the proposal, we'll make a decision. The implementation will
begin in 2 weeks.""",

        "expected_markers": [
            {"text": "January 15, 2024", "timestamp": datetime(2024, 1, 15)},
            {"text": "2024-01-20", "timestamp": datetime(2024, 1, 20)},
        ],

        "expected_relative_expressions": [
            "next week",
            "yesterday",
            "in 2 weeks",
        ],

        "expected_relationships": [
            {
                "entity1": "reviewing proposal",
                "entity2": "making decision",
                "relationship": "BEFORE"
            }
        ]
    }
