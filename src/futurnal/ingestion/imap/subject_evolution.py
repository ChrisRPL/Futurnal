"""Subject evolution tracking for email threads.

Tracks how subject lines evolve throughout an email conversation thread,
identifying subject variations, topic drift, and conversation branching
patterns. This enables the Ghost to understand when conversations shift
focus or split into multiple topics.

Key Features:
- Subject variation detection across thread messages
- Normalization for meaningful comparison (remove Re:/Fwd:, case, whitespace)
- Subject change point identification for topic drift analysis
- Subject clustering for conversation grouping

Ghost→Animal Evolution:
- **Phase 1 (CURRENT)**: Mechanical subject variation tracking
- **Phase 2 (FUTURE)**: Semantic similarity analysis for topic drift detection
- **Phase 3 (FUTURE)**: Conversation split prediction and topic clustering
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

from .email_parser import EmailMessage
from .thread_models import EmailThread

logger = logging.getLogger(__name__)


class SubjectEvolutionTracker:
    """Track how subject line evolves in a thread.

    Analyzes subject variations across messages in an email thread to
    identify topic changes, conversation branching, and subject drift.

    Usage:
        tracker = SubjectEvolutionTracker()

        # Analyze subject evolution for a thread
        variations = tracker.analyze_subject_evolution(thread, messages)

        # Get subject change points
        change_points = tracker.identify_subject_changes(thread, messages)
    """

    def __init__(self):
        """Initialize subject evolution tracker."""
        pass

    def analyze_subject_evolution(
        self, thread: EmailThread, messages: Dict[str, EmailMessage]
    ) -> List[str]:
        """Analyze subject variations in thread.

        Identifies all unique subject line variations in the thread
        (after normalization) while preserving original formatting.

        Args:
            thread: EmailThread to analyze
            messages: Dictionary mapping Message-ID to EmailMessage

        Returns:
            List of unique subject variations (original format)
        """
        variations = []
        seen_subjects: Set[str] = set()

        # Traverse messages in thread order (chronologically)
        sorted_message_ids = self._sort_messages_chronologically(thread, messages)

        for message_id in sorted_message_ids:
            if message_id not in messages:
                continue

            msg = messages[message_id]
            subject = msg.subject or ""

            # Normalize for comparison
            normalized = self._normalize_subject(subject)

            # Add if this is a new variation
            if normalized and normalized not in seen_subjects:
                seen_subjects.add(normalized)
                variations.append(subject)  # Store original format

        logger.debug(
            f"Found {len(variations)} subject variations in thread {thread.thread_id}",
            extra={
                "thread_id": thread.thread_id,
                "variation_count": len(variations),
                "message_count": thread.message_count,
            },
        )

        return variations

    def identify_subject_changes(
        self, thread: EmailThread, messages: Dict[str, EmailMessage]
    ) -> List[Dict[str, any]]:
        """Identify subject change points in thread.

        Detects when subject line changes during conversation, which
        may indicate topic drift or conversation branching.

        Args:
            thread: EmailThread to analyze
            messages: Dictionary mapping Message-ID to EmailMessage

        Returns:
            List of change point dictionaries with keys:
                - message_id: Message where change occurred
                - previous_subject: Previous normalized subject
                - new_subject: New normalized subject
                - change_type: Type of change (topic_shift, branch, etc.)
        """
        change_points = []
        previous_normalized = None

        sorted_message_ids = self._sort_messages_chronologically(thread, messages)

        for i, message_id in enumerate(sorted_message_ids):
            if message_id not in messages:
                continue

            msg = messages[message_id]
            subject = msg.subject or ""
            normalized = self._normalize_subject(subject)

            # Detect change from previous message
            if previous_normalized and normalized != previous_normalized:
                change_type = self._classify_subject_change(
                    previous_normalized, normalized
                )

                change_points.append(
                    {
                        "message_id": message_id,
                        "message_index": i,
                        "previous_subject": previous_normalized,
                        "new_subject": normalized,
                        "change_type": change_type,
                        "date": msg.date,
                    }
                )

            previous_normalized = normalized

        logger.debug(
            f"Found {len(change_points)} subject changes in thread {thread.thread_id}",
            extra={
                "thread_id": thread.thread_id,
                "change_count": len(change_points),
            },
        )

        return change_points

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject for comparison.

        Applies aggressive normalization:
        - Remove Re:/Fwd: prefixes
        - Convert to lowercase
        - Remove extra whitespace
        - Remove punctuation (for semantic comparison)

        Args:
            subject: Raw subject line

        Returns:
            Normalized subject for comparison
        """
        if not subject:
            return ""

        # Remove Re:/Fwd: prefixes (case-insensitive, with optional [N])
        pattern = r"^(Re|RE|re|Fwd|FWD|fwd)(\[\d+\])?:\s*"
        while re.match(pattern, subject):
            subject = re.sub(pattern, "", subject, count=1).strip()

        # Lowercase and strip
        subject = subject.lower().strip()

        # Normalize whitespace (multiple spaces -> single space)
        subject = re.sub(r"\s+", " ", subject)

        return subject

    def _sort_messages_chronologically(
        self, thread: EmailThread, messages: Dict[str, EmailMessage]
    ) -> List[str]:
        """Sort thread messages chronologically by date.

        Args:
            thread: EmailThread containing message IDs
            messages: Dictionary mapping Message-ID to EmailMessage

        Returns:
            List of Message-IDs sorted by date (oldest first)
        """
        # Create list of (message_id, date) tuples
        message_dates = [
            (mid, messages[mid].date)
            for mid in thread.message_ids
            if mid in messages
        ]

        # Sort by date
        message_dates.sort(key=lambda x: x[1])

        return [mid for mid, _ in message_dates]

    def _classify_subject_change(
        self, previous_subject: str, new_subject: str
    ) -> str:
        """Classify type of subject change.

        Categorizes subject changes to understand conversation dynamics:
        - minor_edit: Small changes (typo fixes, clarifications)
        - topic_shift: Significant topic change
        - branch: Conversation branching (based on content difference)

        Args:
            previous_subject: Previous normalized subject
            new_subject: New normalized subject

        Returns:
            Change type classification
        """
        # Calculate similarity using simple word overlap
        prev_words = set(previous_subject.split())
        new_words = set(new_subject.split())

        # Calculate Jaccard similarity
        if not prev_words and not new_words:
            return "no_change"

        intersection = len(prev_words & new_words)
        union = len(prev_words | new_words)

        if union == 0:
            similarity = 0.0
        else:
            similarity = intersection / union

        # Classify based on similarity
        if similarity > 0.7:
            return "minor_edit"
        elif similarity > 0.3:
            return "topic_shift"
        else:
            return "branch"

    def get_subject_evolution_summary(
        self, thread: EmailThread, messages: Dict[str, EmailMessage]
    ) -> Dict[str, any]:
        """Get comprehensive subject evolution summary.

        Provides complete analysis of subject evolution including
        variations, change points, and evolution metrics.

        Args:
            thread: EmailThread to analyze
            messages: Dictionary mapping Message-ID to EmailMessage

        Returns:
            Dictionary with evolution metrics:
                - total_variations: Number of unique subject variations
                - subject_changes: Number of subject change points
                - change_frequency: Changes per message ratio
                - most_common_subject: Most frequently used subject
                - subject_stability: Measure of subject consistency (0-1)
        """
        variations = self.analyze_subject_evolution(thread, messages)
        change_points = self.identify_subject_changes(thread, messages)

        # Calculate subject frequency
        subject_counts: Dict[str, int] = {}
        for message_id in thread.message_ids:
            if message_id not in messages:
                continue

            subject = self._normalize_subject(messages[message_id].subject or "")
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

        most_common_subject = (
            max(subject_counts, key=subject_counts.get) if subject_counts else ""
        )

        # Calculate stability (1 = no changes, 0 = constant changes)
        if thread.message_count > 1:
            stability = 1 - (len(change_points) / (thread.message_count - 1))
        else:
            stability = 1.0

        return {
            "total_variations": len(variations),
            "subject_changes": len(change_points),
            "change_frequency": len(change_points) / max(thread.message_count, 1),
            "most_common_subject": most_common_subject,
            "subject_stability": stability,
            "variations": variations,
            "change_points": change_points,
        }


__all__ = ["SubjectEvolutionTracker"]
