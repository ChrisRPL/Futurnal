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

# ISO 8601 patterns (enhanced for better coverage)
ISO_8601_PATTERN = re.compile(
    r'\b\d{4}[-/.]\d{2}[-/.]\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b'
)

# Additional common date formats
US_DATE_PATTERN = re.compile(
    r'\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b'  # MM/DD/YYYY or MM/DD/YY
)
EU_DATE_PATTERN = re.compile(
    r'\b(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})\b'  # DD.MM.YYYY or DD.MM.YY
)

# Time expressions (12-hour and 24-hour) - enhanced
TIME_12HR_PATTERN = re.compile(
    r'\b(?:at\s+)?(\d{1,2}):(\d{2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.)\b'
)
TIME_24HR_PATTERN = re.compile(
    r'\b(?:at\s+)?(\d{1,2}):(\d{2})(?::(\d{2}))?\b'  # Added optional seconds
)

# Relative duration patterns (enhanced with more variations)
DURATION_AGO_PATTERN = re.compile(
    r'\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b',
    re.IGNORECASE
)
DURATION_IN_PATTERN = re.compile(
    r'\bin\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b',
    re.IGNORECASE
)
DURATION_FROM_NOW_PATTERN = re.compile(
    r'\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+from\s+now\b',
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

    def extract_temporal_markers(
        self,
        text: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
        reference_time: Optional[datetime] = None
    ) -> List[TemporalMark]:
        """Extract all temporal markers from text and metadata.

        This is the main entry point for comprehensive temporal extraction.
        Combines:
        - Explicit timestamps (ISO 8601, natural language dates, times)
        - Relative expressions (yesterday, last week, 2 weeks ago)
        - Document metadata inference

        Args:
            text: Input text to extract temporal markers from
            doc_metadata: Optional document metadata dictionary
            reference_time: Optional reference time for relative expressions
                          (defaults to current time)

        Returns:
            List of all extracted temporal markers

        Target accuracy:
        - >95% for explicit timestamps
        - >85% for relative expressions
        """
        if reference_time is None:
            reference_time = datetime.now()

        markers = []

        # 1. Extract explicit timestamps
        markers.extend(self.extract_explicit_timestamps(text))

        # 2. Parse relative expressions in text with time-of-day modifiers
        text_lower = text.lower()

        # Time-of-day mapping
        time_of_day = {
            "morning": 9, "afternoon": 14, "evening": 18, "night": 21
        }

        # Look for compound expressions (e.g., "tomorrow evening", "yesterday morning")
        for day_expr in ["yesterday", "today", "tomorrow"]:
            for time_expr, hour in time_of_day.items():
                compound = f"{day_expr} {time_expr}"
                if compound in text_lower:
                    relative_marker = self.parse_relative_expression(day_expr, reference_time)
                    if relative_marker:
                        # Adjust time to specified time of day
                        relative_marker.timestamp = relative_marker.timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
                        relative_marker.text = compound
                        markers.append(relative_marker)

        # Look for simple relative expressions
        for expr in ["yesterday", "today", "tomorrow", "last week", "next week",
                     "this week", "last month", "next month", "this month",
                     "last year", "next year", "this year"]:
            # Skip if already matched as compound expression
            if any(f"{expr} {tod}" in text_lower for tod in time_of_day.keys()):
                continue
            if expr in text_lower:
                relative_marker = self.parse_relative_expression(expr, reference_time)
                if relative_marker:
                    markers.append(relative_marker)

        # Parse duration patterns (e.g., "2 weeks ago", "in 3 days", "3 months later")
        import re

        # Pattern: "N time_unit ago"
        duration_ago_matches = re.findall(
            r'(Three|two|one|\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago',
            text,
            re.IGNORECASE
        )
        for match in duration_ago_matches:
            # Convert word numbers to digits
            amount = match[0].lower()
            if amount == "one": amount = "1"
            elif amount == "two": amount = "2"
            elif amount == "three": amount = "3"

            expr = f"{amount} {match[1]}s ago"
            relative_marker = self.parse_relative_expression(expr, reference_time)
            if relative_marker:
                markers.append(relative_marker)

        # Pattern: "in N time_unit"
        duration_in_matches = re.findall(
            r'in\s+(one|two|three|\d+)\s+(second|minute|hour|day|week|month|year)s?',
            text,
            re.IGNORECASE
        )
        for match in duration_in_matches:
            # Convert word numbers to digits
            amount = match[0].lower()
            if amount == "one": amount = "1"
            elif amount == "two": amount = "2"
            elif amount == "three": amount = "3"

            expr = f"in {amount} {match[1]}s"
            relative_marker = self.parse_relative_expression(expr, reference_time)
            if relative_marker:
                markers.append(relative_marker)

        # Pattern: "N time_unit later" (relative to a previous date)
        duration_later_matches = re.findall(
            r'(one|two|three|\d+)\s+(second|minute|hour|day|week|month|year)s?\s+later',
            text,
            re.IGNORECASE
        )
        for match in duration_later_matches:
            # Convert word numbers to digits
            amount = match[0].lower()
            if amount == "one": amount = "1"
            elif amount == "two": amount = "2"
            elif amount == "three": amount = "3"

            # "later" means forward from reference_time
            expr = f"in {amount} {match[1]}s"
            relative_marker = self.parse_relative_expression(expr, reference_time)
            if relative_marker:
                markers.append(relative_marker)

        # 3. Infer from document metadata
        # Note: We extract metadata timestamp for reference_time purposes,
        # but don't add it as a separate marker since it's not mentioned in the text
        # If needed, users can call infer_from_document_metadata() separately

        self.extracted_count += len(markers)
        return markers

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

        # Extract US/EU date formats (MM/DD/YYYY, DD.MM.YYYY)
        markers.extend(self._extract_common_date_formats(text))

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
        # Check top-level date fields first (common in test data and simple metadata)
        for field in ["created", "date", "published", "modified"]:
            if field in doc_metadata:
                try:
                    return self._parse_flexible_timestamp(doc_metadata[field])
                except Exception as e:
                    logger.debug(f"Failed to parse top-level {field}: {e}")

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
        """Extract ISO 8601 format timestamps and date+time combinations."""
        markers = []

        # First, extract ISO 8601 date + time combinations (e.g., "2024-01-20 at 2:30 PM")
        date_at_time_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2})\s+at\s+(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)',
            re.IGNORECASE
        )
        for match in date_at_time_pattern.finditer(text):
            date_str = match.group(1)
            hour = int(match.group(2))
            minute = int(match.group(3))
            am_pm = match.group(4).upper()

            # Convert to 24-hour
            if am_pm == "PM" and hour != 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0

            try:
                date_part = datetime.fromisoformat(date_str)
                timestamp = date_part.replace(hour=hour, minute=minute)

                markers.append(TemporalMark(
                    text=match.group(0),
                    timestamp=timestamp,
                    temporal_type=TemporalSourceType.EXPLICIT,
                    confidence=0.98,
                    span_start=match.start(),
                    span_end=match.end()
                ))
            except Exception as e:
                logger.debug(f"Failed to parse date+time '{match.group(0)}': {e}")

        # Track already extracted date+time combinations to avoid duplicates
        extracted_ranges = [(m.span_start, m.span_end) for m in markers if hasattr(m, 'span_start')]

        # Then extract standard ISO 8601 timestamps
        for match in ISO_8601_PATTERN.finditer(text):
            # Skip if already part of a date+time combination
            if any(start <= match.start() < end for start, end in extracted_ranges):
                continue

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

    def _extract_common_date_formats(self, text: str) -> List[TemporalMark]:
        """Extract common US and EU date formats (MM/DD/YYYY, DD.MM.YYYY)."""
        markers = []

        # Track already extracted positions
        iso_matches = list(ISO_8601_PATTERN.finditer(text))
        iso_ranges = [(m.start(), m.end()) for m in iso_matches]

        # Extract US format dates (MM/DD/YYYY or MM/DD/YY)
        for match in US_DATE_PATTERN.finditer(text):
            # Skip if overlaps with ISO 8601
            if any(not (iso_end <= match.start() or iso_start >= match.end())
                   for iso_start, iso_end in iso_ranges):
                continue

            month = int(match.group(1))
            day = int(match.group(2))
            year_str = match.group(3)
            year = int(year_str) if len(year_str) == 4 else (2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str))

            # Validate date components
            if month < 1 or month > 12 or day < 1 or day > 31:
                continue

            try:
                timestamp = datetime(year, month, day)
                markers.append(TemporalMark(
                    text=match.group(0),
                    timestamp=timestamp,
                    temporal_type=TemporalSourceType.EXPLICIT,
                    confidence=0.92,
                    span_start=match.start(),
                    span_end=match.end()
                ))
            except ValueError:
                continue  # Invalid date

        # Extract EU format dates (DD.MM.YYYY or DD.MM.YY)
        extracted_ranges = [(m.span_start, m.span_end) for m in markers if hasattr(m, 'span_start')]
        
        for match in EU_DATE_PATTERN.finditer(text):
            # Skip if overlaps with ISO 8601 or already extracted
            if any(not (iso_end <= match.start() or iso_start >= match.end())
                   for iso_start, iso_end in iso_ranges):
                continue
            if any(start <= match.start() < end for start, end in extracted_ranges):
                continue

            day = int(match.group(1))
            month = int(match.group(2))
            year_str = match.group(3)
            year = int(year_str) if len(year_str) == 4 else (2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str))

            # Validate date components
            if month < 1 or month > 12 or day < 1 or day > 31:
                continue

            try:
                timestamp = datetime(year, month, day)
                markers.append(TemporalMark(
                    text=match.group(0),
                    timestamp=timestamp,
                    temporal_type=TemporalSourceType.EXPLICIT,
                    confidence=0.92,
                    span_start=match.start(),
                    span_end=match.end()
                ))
            except ValueError:
                continue  # Invalid date

        return markers

    def _extract_natural_dates(self, text: str) -> List[TemporalMark]:
        """Extract natural language dates using pattern matching and dateutil parser."""
        markers = []

        # Track positions of already extracted ISO 8601 timestamps to avoid duplicates
        iso_matches = list(ISO_8601_PATTERN.finditer(text))
        iso_ranges = [(m.start(), m.end()) for m in iso_matches]

        # Explicit natural date patterns (more precise than fuzzy parsing)
        # Format: Month Day, Year (e.g., "January 15, 2024", "March 15, 2024")
        month_day_year = re.compile(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
            r'(\d{1,2})(?:st|nd|rd|th)?,?\s+'
            r'(\d{4})\b',
            re.IGNORECASE
        )

        # Format: Day Month Year (e.g., "15 January 2024")
        day_month_year = re.compile(
            r'\b(\d{1,2})(?:st|nd|rd|th)?\s+'
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
            r'(\d{4})\b',
            re.IGNORECASE
        )

        # Also check for natural date + time combinations (e.g., "January 17, 2024 at 2:00 PM")
        natural_date_time = re.compile(
            r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+'
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
            r'(\d{1,2})(?:st|nd|rd|th)?,?\s+'
            r'(\d{4})\s+at\s+'
            r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)',
            re.IGNORECASE
        )

        for match in natural_date_time.finditer(text):
            # Skip if overlaps with ISO 8601
            if any(not (iso_end <= match.start() or iso_start >= match.end())
                   for iso_start, iso_end in iso_ranges):
                continue

            try:
                # Parse the date part
                date_str = f"{match.group(1)} {match.group(2)}, {match.group(3)}"
                parsed_date = dateutil_parser.parse(date_str)

                # Parse the time part
                hour = int(match.group(4))
                minute = int(match.group(5))
                am_pm = match.group(6).upper()

                # Convert to 24-hour
                if am_pm == "PM" and hour != 12:
                    hour += 12
                elif am_pm == "AM" and hour == 12:
                    hour = 0

                timestamp = parsed_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)

                markers.append(TemporalMark(
                    text=match.group(0),
                    timestamp=timestamp,
                    temporal_type=TemporalSourceType.EXPLICIT,
                    confidence=0.95,
                    span_start=match.start(),
                    span_end=match.end()
                ))
            except Exception as e:
                logger.debug(f"Failed to parse natural date+time '{match.group(0)}': {e}")

        # Extract using explicit patterns (date only)
        extracted_ranges = [(m.span_start, m.span_end) for m in markers if hasattr(m, 'span_start')]

        for pattern in [month_day_year, day_month_year]:
            for match in pattern.finditer(text):
                # Skip if overlaps with ISO 8601 or already extracted
                if any(not (iso_end <= match.start() or iso_start >= match.end())
                       for iso_start, iso_end in iso_ranges):
                    continue
                if any(start <= match.start() < end for start, end in extracted_ranges):
                    continue

                date_str = match.group(0)
                try:
                    parsed_date = dateutil_parser.parse(date_str)
                    if parsed_date.tzinfo is not None:
                        parsed_date = parsed_date.replace(tzinfo=None)

                    markers.append(TemporalMark(
                        text=date_str,
                        timestamp=parsed_date,
                        temporal_type=TemporalSourceType.EXPLICIT,
                        confidence=0.90,  # High confidence for explicit patterns
                        span_start=match.start(),
                        span_end=match.end()
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse natural date '{date_str}': {e}")

        return markers

    def _is_likely_date(self, sentence: str, non_date_tokens: List[str]) -> bool:
        """Check if parsed result is likely a real date."""
        # If most of the sentence was consumed by non-date tokens, it's probably not a date
        total_len = len(sentence)
        non_date_len = sum(len(token) for token in non_date_tokens)

        # At least 15% of sentence should be date-related (lowered from 30% to catch
        # dates in longer sentences like "On January 15, 2024, we held...")
        return (total_len - non_date_len) / total_len >= 0.15

    def _extract_time_expressions(self, text: str) -> List[TemporalMark]:
        """Extract time expressions (12-hour and 24-hour formats)."""
        markers = []

        # Track ISO 8601 timestamp positions to avoid extracting time parts
        iso_matches = list(ISO_8601_PATTERN.finditer(text))
        iso_ranges = [(m.start(), m.end()) for m in iso_matches]

        # Also track date+time combinations to avoid duplicate extraction
        date_at_time_pattern = re.compile(
            r'\d{4}-\d{2}-\d{2}\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)',
            re.IGNORECASE
        )
        date_time_matches = list(date_at_time_pattern.finditer(text))
        date_time_ranges = [(m.start(), m.end()) for m in date_time_matches]

        # Extract 12-hour format times
        for match in TIME_12HR_PATTERN.finditer(text):
            # Skip if within ISO 8601 timestamp or date+time combination
            if any(start <= match.start() < end for start, end in iso_ranges):
                continue
            if any(start <= match.start() < end for start, end in date_time_ranges):
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
        # Use midnight (00:00) for date-only expressions
        timestamp = reference_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=offset_days)

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.95
        )

    def _parse_relative_week(self, expr: str, reference_time: datetime) -> TemporalMark:
        """Parse relative week expression (last week, this week, next week)."""
        offset_weeks = RELATIVE_WEEK_MAPPING[expr]
        # Use midnight (00:00) for date-only expressions
        timestamp = reference_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(weeks=offset_weeks)

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.90
        )

    def _parse_relative_month(self, expr: str, reference_time: datetime) -> TemporalMark:
        """Parse relative month expression (last month, this month, next month)."""
        offset_months = RELATIVE_MONTH_MAPPING[expr]
        # Use midnight (00:00) for date-only expressions
        timestamp = reference_time.replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(months=offset_months)

        return TemporalMark(
            text=expr,
            timestamp=timestamp,
            temporal_type=TemporalSourceType.RELATIVE,
            confidence=0.90
        )

    def _parse_relative_year(self, expr: str, reference_time: datetime) -> TemporalMark:
        """Parse relative year expression (last year, this year, next year)."""
        offset_years = RELATIVE_YEAR_MAPPING[expr]
        # Use midnight (00:00) for date-only expressions
        timestamp = reference_time.replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(years=offset_years)

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

        # For date-level units (day+), use midnight; for time units, keep precision
        base_time = reference_time if unit in ["second", "minute", "hour"] else reference_time.replace(hour=0, minute=0, second=0, microsecond=0)

        # Calculate offset
        if unit == "second":
            timestamp = base_time - timedelta(seconds=amount)
        elif unit == "minute":
            timestamp = base_time - timedelta(minutes=amount)
        elif unit == "hour":
            timestamp = base_time - timedelta(hours=amount)
        elif unit == "day":
            timestamp = base_time - timedelta(days=amount)
        elif unit == "week":
            timestamp = base_time - timedelta(weeks=amount)
        elif unit == "month":
            timestamp = base_time - relativedelta(months=amount)
        elif unit == "year":
            timestamp = base_time - relativedelta(years=amount)
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

        # For date-level units (day+), use midnight; for time units, keep precision
        base_time = reference_time if unit in ["second", "minute", "hour"] else reference_time.replace(hour=0, minute=0, second=0, microsecond=0)

        # Calculate offset
        if unit == "second":
            timestamp = base_time + timedelta(seconds=amount)
        elif unit == "minute":
            timestamp = base_time + timedelta(minutes=amount)
        elif unit == "hour":
            timestamp = base_time + timedelta(hours=amount)
        elif unit == "day":
            timestamp = base_time + timedelta(days=amount)
        elif unit == "week":
            timestamp = base_time + timedelta(weeks=amount)
        elif unit == "month":
            timestamp = base_time + relativedelta(months=amount)
        elif unit == "year":
            timestamp = base_time + relativedelta(years=amount)
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
