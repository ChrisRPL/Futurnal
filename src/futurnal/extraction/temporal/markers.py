"""Temporal marker extraction for Module 01 - Week 1.

This module implements temporal marker detection including:
- Explicit timestamp extraction (ISO 8601, natural language dates)
- Relative time expression parsing
- Document metadata temporal inference

Target accuracy:
- >95% explicit timestamp detection
- >85% relative expression parsing

Implementation follows production plan:
docs/phase-1/entity-relationship-extraction-production-plan/01-temporal-extraction.md
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta

from futurnal.extraction.temporal.models import (
    TemporalMark,
    TemporalSourceType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relative Time Expression Mappings
# ---------------------------------------------------------------------------

RELATIVE_DAY_MAPPING = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
    "the day before yesterday": -2,
    "the day after tomorrow": 2,
}

RELATIVE_WEEK_MAPPING = {
    "this week": 0,
    "last week": -1,
    "next week": 1,
    "two weeks ago": -2,
    "in two weeks": 2,
}

RELATIVE_MONTH_MAPPING = {
    "this month": 0,
    "last month": -1,
    "next month": 1,
}

RELATIVE_YEAR_MAPPING = {
    "this year": 0,
    "last year": -1,
    "next year": 1,
}


# ---------------------------------------------------------------------------
# Regular Expression Patterns
# ---------------------------------------------------------------------------

# ISO 8601 patterns
ISO_8601_PATTERN = re.compile(
    r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?\b'
)

# Time expressions (12-hour and 24-hour)
TIME_12HR_PATTERN = re.compile(
    r'\b(?:at\s+)?(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)\b'
)
TIME_24HR_PATTERN = re.compile(
    r'\b(?:at\s+)?(\d{2}):(\d{2})\b'
)

# Relative duration patterns
DURATION_AGO_PATTERN = re.compile(
    r'\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b',
    re.IGNORECASE
)
DURATION_IN_PATTERN = re.compile(
    r'\bin\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Main Temporal Marker Extractor
# ---------------------------------------------------------------------------


class TemporalMarkerExtractor:
    """Extract temporal markers from text.

    Implements Week 1 deliverable: Core temporal marker extraction

    Supports:
    - ISO 8601 timestamps
    - Natural language dates
    - Time expressions (12-hour and 24-hour)
    - Relative expressions (yesterday, last week, 2 weeks ago)
    - Document metadata inference

    Target accuracy:
    - >95% explicit timestamp detection
    - >85% relative expression parsing
    """

    def __init__(self):
        """Initialize temporal marker extractor."""
        self.extracted_count = 0

    def extract_explicit_timestamps(self, text: str) -> List[TemporalMark]:
        """Extract explicit temporal markers from text.

        Detects:
        - ISO 8601 formats (2024-01-15, 2024-01-15T14:30:00Z)
        - Natural language dates (January 15, 2024, 15th of January)
        - Time expressions (2:30 PM, 14:30)

        Args:
            text: Input text to extract temporal markers from

        Returns:
            List of extracted temporal markers

        Target: >95% accuracy on explicit timestamps
        """
        markers = []

        # Extract ISO 8601 timestamps
        markers.extend(self._extract_iso8601(text))

        # Extract natural language dates
        markers.extend(self._extract_natural_dates(text))

        # Extract time expressions
        markers.extend(self._extract_time_expressions(text))

        self.extracted_count += len(markers)
        return markers

    def parse_relative_expression(
        self,
        expr: str,
        reference_time: Optional[datetime] = None
    ) -> Optional[TemporalMark]:
        """Parse relative temporal expressions into absolute timestamps.

        Handles:
        - Relative days (yesterday, today, tomorrow)
        - Relative periods (last week, this month, next year)
        - Duration offsets (2 weeks ago, in 3 months, 5 days from now)

        Args:
            expr: Relative expression to parse
            reference_time: Reference time for relative calculation (defaults to now)

        Returns:
            TemporalMark with parsed timestamp or None if unparseable

        Target: >85% accuracy on relative expressions
        """
        if reference_time is None:
            reference_time = datetime.utcnow()

        expr_lower = expr.lower().strip()

        # Relative days
        if expr_lower in RELATIVE_DAY_MAPPING:
            return self._parse_relative_day(expr_lower, reference_time)

        # Relative weeks
        if expr_lower in RELATIVE_WEEK_MAPPING:
            return self._parse_relative_week(expr_lower, reference_time)

        # Relative months
        if expr_lower in RELATIVE_MONTH_MAPPING:
            return self._parse_relative_month(expr_lower, reference_time)

        # Relative years
        if expr_lower in RELATIVE_YEAR_MAPPING:
            return self._parse_relative_year(expr_lower, reference_time)

        # Duration-based: "X days/weeks/months ago"
        if "ago" in expr_lower:
            return self._parse_duration_ago(expr_lower, reference_time)

        # Duration-based: "in X days/weeks/months"
        if expr_lower.startswith("in "):
            return self._parse_future_duration(expr_lower, reference_time)

        return None

    def infer_from_document_metadata(self, doc_metadata: Dict[str, Any]) -> Optional[datetime]:
        """Extract temporal reference from document metadata.

        Checks (in priority order):
        1. Frontmatter date fields (created, modified, published)
        2. File timestamps (creation, modification times)
        3. Email headers (Date:, Received:)
        4. Git commit timestamps

        Args:
            doc_metadata: Document metadata dictionary

        Returns:
            Inferred timestamp or None
        """
        # Check frontmatter (Obsidian)
        frontmatter = doc_metadata.get("frontmatter", {})
        for field in ["created", "date", "published", "modified"]:
            if field in frontmatter:
                try:
                    return self._parse_flexible_timestamp(frontmatter[field])
                except Exception as e:
                    logger.debug(f"Failed to parse frontmatter {field}: {e}")

        # Check file timestamps
        if "created_at" in doc_metadata and doc_metadata["created_at"]:
            return doc_metadata["created_at"]
        if "modified_at" in doc_metadata and doc_metadata["modified_at"]:
            return doc_metadata["modified_at"]

        # Check email headers
        if "email_date" in doc_metadata:
            try:
                return self._parse_flexible_timestamp(doc_metadata["email_date"])
            except Exception as e:
                logger.debug(f"Failed to parse email date: {e}")

        # Check git commit timestamp
        if "git_commit_timestamp" in doc_metadata:
            try:
                return self._parse_flexible_timestamp(doc_metadata["git_commit_timestamp"])
            except Exception as e:
                logger.debug(f"Failed to parse git timestamp: {e}")

        return None

    # -----------------------------------------------------------------------
    # Private: ISO 8601 Extraction
    # -----------------------------------------------------------------------

    def _extract_iso8601(self, text: str) -> List[TemporalMark]:
        """Extract ISO 8601 format timestamps."""
        markers = []
        for match in ISO_8601_PATTERN.finditer(text):
            timestamp_str = match.group(0)
            try:
                # Parse ISO 8601 timestamp
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                # Convert to naive UTC datetime for consistency
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)

                markers.append(TemporalMark(
                    text=timestamp_str,
                    timestamp=timestamp,
                    temporal_type=TemporalSourceType.EXPLICIT,
                    confidence=0.98,  # High confidence for ISO 8601
                    span_start=match.start(),
                    span_end=match.end()
                ))
            except Exception as e:
                logger.debug(f"Failed to parse ISO 8601 timestamp '{timestamp_str}': {e}")

        return markers

    def _extract_natural_dates(self, text: str) -> List[TemporalMark]:
        """Extract natural language dates using dateutil parser."""
        markers = []

        # Track positions of already extracted ISO 8601 timestamps to avoid duplicates
        iso_matches = list(ISO_8601_PATTERN.finditer(text))
        iso_ranges = [(m.start(), m.end()) for m in iso_matches]

        # Split text into sentences to avoid false positives
        sentences = re.split(r'[.!?\n]', text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Skip if sentence contains ISO 8601 timestamp (already extracted)
            sentence_in_text = text.find(sentence)
            # Check if any ISO range overlaps with this sentence's range
            sentence_end = sentence_in_text + len(sentence)
            if any(not (iso_end <= sentence_in_text or iso_start >= sentence_end)
                   for iso_start, iso_end in iso_ranges):
                continue

            try:
                # Try to parse natural language date
                # dateutil.parser is quite liberal, so we need to be careful
                parsed_date, tokens = dateutil_parser.parse(sentence, fuzzy_with_tokens=True)

                # Convert to naive datetime for consistency
                if parsed_date.tzinfo is not None:
                    parsed_date = parsed_date.replace(tzinfo=None)

                # Check if significant date components were found
                # (avoid matching pure numbers or unrelated text)
                if self._is_likely_date(sentence, tokens):
                    # Reconstruct the matched text
                    matched_text = sentence
                    for token in tokens:
                        matched_text = matched_text.replace(token, '', 1)
                    matched_text = matched_text.strip()

                    markers.append(TemporalMark(
                        text=matched_text[:50],  # Limit length
                        timestamp=parsed_date,
                        temporal_type=TemporalSourceType.EXPLICIT,
                        confidence=0.85,  # Lower confidence than ISO 8601
                    ))
            except (ValueError, OverflowError, AttributeError):
                # Not a parseable date - continue
                continue

        return markers

    def _is_likely_date(self, sentence: str, non_date_tokens: List[str]) -> bool:
        """Check if parsed result is likely a real date."""
        # If most of the sentence was consumed by non-date tokens, it's probably not a date
        total_len = len(sentence)
        non_date_len = sum(len(token) for token in non_date_tokens)

        # At least 30% of sentence should be date-related
        return (total_len - non_date_len) / total_len >= 0.3

    def _extract_time_expressions(self, text: str) -> List[TemporalMark]:
        """Extract time expressions (12-hour and 24-hour formats)."""
        markers = []

        # Track ISO 8601 timestamp positions to avoid extracting time parts
        iso_matches = list(ISO_8601_PATTERN.finditer(text))
        iso_ranges = [(m.start(), m.end()) for m in iso_matches]

        # Extract 12-hour format times
        for match in TIME_12HR_PATTERN.finditer(text):
            # Skip if within ISO 8601 timestamp
            if any(start <= match.start() < end for start, end in iso_ranges):
                continue

            time_str = match.group(0)
            hour = int(match.group(1))
            minute = int(match.group(2))
            am_pm = match.group(3).upper()

            # Validate hour (1-12 for 12-hour format)
            if hour < 1 or hour > 12 or minute > 59:
                continue

            # Convert to 24-hour
            if am_pm == "PM" and hour != 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0

            # Create timestamp (date is unknown, use today as placeholder)
            today = datetime.utcnow().date()
            timestamp = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))

            markers.append(TemporalMark(
                text=time_str,
                timestamp=timestamp,
                temporal_type=TemporalSourceType.EXPLICIT,
                confidence=0.90,
                span_start=match.start(),
                span_end=match.end()
            ))

        # Extract 24-hour format times
        for match in TIME_24HR_PATTERN.finditer(text):
            # Skip if within ISO 8601 timestamp
            if any(start <= match.start() < end for start, end in iso_ranges):
                continue

            time_str = match.group(0)
            hour = int(match.group(1))
            minute = int(match.group(2))

            # Validate time
            if hour > 23 or minute > 59:
                continue

            today = datetime.utcnow().date()
            timestamp = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))

            markers.append(TemporalMark(
                text=time_str,
                timestamp=timestamp,
                temporal_type=TemporalSourceType.EXPLICIT,
                confidence=0.90,
                span_start=match.start(),
                span_end=match.end()
            ))

        return markers

    # -----------------------------------------------------------------------
    # Private: Relative Expression Parsing
    # -----------------------------------------------------------------------

    def _parse_relative_day(self, expr: str, reference_time: datetime) -> TemporalMark:
        """Parse relative day expression (yesterday, today, tomorrow)."""
        offset_days = RELATIVE_DAY_MAPPING[expr]
        timestamp = reference_time + timedelta(days=offset_days)

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.95
        )

    def _parse_relative_week(self, expr: str, reference_time: datetime) -> TemporalMark:
        """Parse relative week expression (last week, this week, next week)."""
        offset_weeks = RELATIVE_WEEK_MAPPING[expr]
        timestamp = reference_time + timedelta(weeks=offset_weeks)

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.90
        )

    def _parse_relative_month(self, expr: str, reference_time: datetime) -> TemporalMark:
        """Parse relative month expression (last month, this month, next month)."""
        offset_months = RELATIVE_MONTH_MAPPING[expr]
        timestamp = reference_time + relativedelta(months=offset_months)

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.90
        )

    def _parse_relative_year(self, expr: str, reference_time: datetime) -> TemporalMark:
        """Parse relative year expression (last year, this year, next year)."""
        offset_years = RELATIVE_YEAR_MAPPING[expr]
        timestamp = reference_time + relativedelta(years=offset_years)

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.90
        )

    def _parse_duration_ago(self, expr: str, reference_time: datetime) -> Optional[TemporalMark]:
        """Parse 'X days/weeks/months ago' expressions."""
        match = DURATION_AGO_PATTERN.search(expr)
        if not match:
            return None

        amount = int(match.group(1))
        unit = match.group(2).lower()

        # Calculate offset
        if unit == "second":
            timestamp = reference_time - timedelta(seconds=amount)
        elif unit == "minute":
            timestamp = reference_time - timedelta(minutes=amount)
        elif unit == "hour":
            timestamp = reference_time - timedelta(hours=amount)
        elif unit == "day":
            timestamp = reference_time - timedelta(days=amount)
        elif unit == "week":
            timestamp = reference_time - timedelta(weeks=amount)
        elif unit == "month":
            timestamp = reference_time - relativedelta(months=amount)
        elif unit == "year":
            timestamp = reference_time - relativedelta(years=amount)
        else:
            return None

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.88
        )

    def _parse_future_duration(self, expr: str, reference_time: datetime) -> Optional[TemporalMark]:
        """Parse 'in X days/weeks/months' expressions."""
        match = DURATION_IN_PATTERN.search(expr)
        if not match:
            return None

        amount = int(match.group(1))
        unit = match.group(2).lower()

        # Calculate offset
        if unit == "second":
            timestamp = reference_time + timedelta(seconds=amount)
        elif unit == "minute":
            timestamp = reference_time + timedelta(minutes=amount)
        elif unit == "hour":
            timestamp = reference_time + timedelta(hours=amount)
        elif unit == "day":
            timestamp = reference_time + timedelta(days=amount)
        elif unit == "week":
            timestamp = reference_time + timedelta(weeks=amount)
        elif unit == "month":
            timestamp = reference_time + relativedelta(months=amount)
        elif unit == "year":
            timestamp = reference_time + relativedelta(years=amount)
        else:
            return None

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.88
        )

    # -----------------------------------------------------------------------
    # Private: Utility Methods
    # -----------------------------------------------------------------------

    def _parse_flexible_timestamp(self, value: Any) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(value, datetime):
            # Convert to naive UTC if timezone-aware
            if value.tzinfo is not None:
                return value.replace(tzinfo=None)
            return value
        elif isinstance(value, str):
            # Try ISO 8601 first
            try:
                timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
                # Convert to naive UTC
                if timestamp.tzinfo is not None:
                    return timestamp.replace(tzinfo=None)
                return timestamp
            except ValueError:
                # Fall back to dateutil parser
                timestamp = dateutil_parser.parse(value)
                # Convert to naive UTC
                if timestamp.tzinfo is not None:
                    return timestamp.replace(tzinfo=None)
                return timestamp
        else:
            raise ValueError(f"Unsupported timestamp type: {type(value)}")
