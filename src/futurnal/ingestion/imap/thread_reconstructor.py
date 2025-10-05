"""Email thread reconstruction engine.

Reconstructs email conversation threads from Message-ID, References, and In-Reply-To
headers per RFC 2822, building a conversation graph that enables the Ghost to
understand communication patterns, relationship dynamics, and conversational
context evolution over time.

Key Features:
- Message-ID graph construction with parent-child relationships
- References/In-Reply-To header parsing and validation
- Thread assembly algorithm (tree construction)
- Participant extraction and role identification
- Response time statistics calculation
- Handling of orphan messages (missing parents)
- Out-of-order message arrival support

Ghost→Animal Evolution:
- **Phase 1 (CURRENT)**: Mechanical thread reconstruction from headers
- **Phase 2 (FUTURE)**: Temporal pattern detection and participant behavior analysis
- **Phase 3 (FUTURE)**: Causal inference from conversation dynamics
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .email_parser import EmailMessage
from .thread_models import (
    EmailThread,
    ParticipantRole,
    ThreadNode,
    ThreadParticipant,
)

logger = logging.getLogger(__name__)


class ThreadReconstructor:
    """Reconstruct conversation threads from email messages.

    Builds a directed acyclic graph (DAG) of email messages using Message-ID,
    In-Reply-To, and References headers, then assembles complete conversation
    threads with participants, temporal data, and structural analysis.

    Usage:
        reconstructor = ThreadReconstructor()

        # Add messages to graph
        for email in emails:
            reconstructor.add_message(email)

        # Reconstruct all threads
        threads = reconstructor.reconstruct_threads()

        # Calculate response times
        for thread in threads:
            reconstructor.calculate_response_times(thread, email_dict)
    """

    def __init__(self):
        """Initialize thread reconstructor."""
        self.message_graph: Dict[str, ThreadNode] = {}
        self.threads: Dict[str, EmailThread] = {}

    def add_message(self, email_message: EmailMessage) -> None:
        """Add message to thread graph.

        Creates a ThreadNode for the message and links it to its parent
        using In-Reply-To header (preferred) or References header (fallback).
        Handles out-of-order message arrival gracefully.

        Args:
            email_message: Parsed email message to add to graph
        """
        # Create node for this message
        node = ThreadNode(
            message_id=email_message.message_id,
            parent_message_id=email_message.in_reply_to,
            date=email_message.date,
            from_address=email_message.from_address.address,
            subject=email_message.subject or "",
        )

        # Add to graph
        self.message_graph[email_message.message_id] = node

        # Link to parent if In-Reply-To is present and parent exists
        if email_message.in_reply_to and email_message.in_reply_to in self.message_graph:
            parent = self.message_graph[email_message.in_reply_to]
            if email_message.message_id not in parent.children:
                parent.children.append(email_message.message_id)

        # Try to link via References header if In-Reply-To missing or parent not found
        elif email_message.references:
            self._link_via_references(email_message)

        # If this message is a parent for any existing orphan, link them
        self._link_orphan_children(email_message.message_id)

    def _link_via_references(self, email_message: EmailMessage) -> None:
        """Link message to parent via References header.

        Per RFC 2822, References header contains ordered list of Message-IDs
        from root to immediate parent. We traverse in reverse order to find
        the first existing parent.

        Args:
            email_message: Email message to link
        """
        # References are ordered from oldest to newest
        # Last reference is the immediate parent
        for ref in reversed(email_message.references):
            if ref in self.message_graph:
                node = self.message_graph[email_message.message_id]
                node.parent_message_id = ref
                parent = self.message_graph[ref]
                if email_message.message_id not in parent.children:
                    parent.children.append(email_message.message_id)
                break

    def _link_orphan_children(self, parent_message_id: str) -> None:
        """Link orphan children to newly added parent.

        When a parent message arrives after its children, we need to
        update the orphan children to link to this parent.

        Args:
            parent_message_id: Message-ID of the newly added parent
        """
        parent_node = self.message_graph.get(parent_message_id)
        if not parent_node:
            return

        # Find all nodes that claim this as their parent
        for message_id, node in self.message_graph.items():
            if message_id == parent_message_id:
                continue

            # If node's parent is this message and not already linked
            if (
                node.parent_message_id == parent_message_id
                and message_id not in parent_node.children
            ):
                parent_node.children.append(message_id)

    def reconstruct_threads(self) -> List[EmailThread]:
        """Reconstruct all threads from message graph.

        Identifies root messages (messages with no parent) and builds
        complete thread structures for each root. Handles orphan messages
        by treating them as separate threads.

        Returns:
            List of reconstructed EmailThread objects
        """
        # Find root messages (no parent)
        root_messages = [
            node for node in self.message_graph.values() if node.is_root
        ]

        threads = []
        for root in root_messages:
            thread = self._build_thread(root)
            threads.append(thread)

        self.threads = {t.thread_id: t for t in threads}
        logger.info(
            f"Reconstructed {len(threads)} threads from {len(self.message_graph)} messages"
        )

        return threads

    def _build_thread(self, root: ThreadNode) -> EmailThread:
        """Build thread from root message.

        Recursively traverses the thread tree to collect all messages,
        identify participants, calculate metadata, and analyze structure.

        Args:
            root: Root ThreadNode of the thread

        Returns:
            Complete EmailThread object with all metadata
        """
        # Collect all messages and participants in thread
        message_ids = []
        participants_map: Dict[str, ThreadParticipant] = {}
        max_depth = 0

        def traverse(node: ThreadNode, depth: int = 0) -> None:
            nonlocal max_depth
            message_ids.append(node.message_id)
            max_depth = max(max_depth, depth)

            # Update node depth
            node.depth = depth

            # Track participant
            if node.from_address not in participants_map:
                # Determine role based on position in thread
                if depth == 0:
                    role = ParticipantRole.INITIATOR
                else:
                    role = ParticipantRole.PARTICIPANT

                participants_map[node.from_address] = ThreadParticipant(
                    email_address=node.from_address,
                    role=role,
                    message_count=0,
                    first_message_date=node.date,
                    last_message_date=node.date,
                )

            # Update participant statistics
            participant = participants_map[node.from_address]
            participant.message_count += 1
            participant.last_message_date = max(participant.last_message_date, node.date)
            participant.first_message_date = min(participant.first_message_date, node.date)

            # Traverse children
            for child_id in node.children:
                if child_id in self.message_graph:
                    traverse(self.message_graph[child_id], depth + 1)

        # Start traversal from root
        traverse(root)

        # Calculate thread temporal metadata
        dates = [
            self.message_graph[mid].date
            for mid in message_ids
            if mid in self.message_graph
        ]
        start_date = min(dates) if dates else root.date
        last_date = max(dates) if dates else root.date
        duration_days = (last_date - start_date).total_seconds() / 86400

        # Count branches (nodes with more than one child)
        branch_count = sum(
            1
            for mid in message_ids
            if mid in self.message_graph and len(self.message_graph[mid].children) > 1
        )

        # Normalize subject (remove Re:, Fwd:, etc.)
        subject = self._normalize_subject(root.subject)

        # Build EmailThread
        thread = EmailThread(
            thread_id=root.message_id,
            root_message_id=root.message_id,
            message_ids=message_ids,
            message_count=len(message_ids),
            participants=list(participants_map.values()),
            participant_count=len(participants_map),
            subject=subject,
            start_date=start_date,
            last_message_date=last_date,
            duration_days=duration_days,
            depth=max_depth,
            branch_count=branch_count,
            mailbox_id="",  # Set by caller
            reconstructed_at=datetime.utcnow(),
        )

        return thread

    def _normalize_subject(self, subject: str) -> str:
        """Remove Re:/Fwd: prefixes to get original subject.

        Handles various formats:
        - Re: Subject
        - RE: Subject
        - Fwd: Subject
        - FWD: Subject
        - Re: Re: Subject (multiple prefixes)
        - Re[2]: Subject (Gmail style)

        Args:
            subject: Raw subject line

        Returns:
            Normalized subject without prefixes
        """
        if not subject:
            return ""

        # Remove common prefixes (case-insensitive)
        # Pattern matches: Re, RE, re, Fwd, FWD, fwd, with optional [N] suffix
        pattern = r"^(Re|RE|re|Fwd|FWD|fwd)(\[\d+\])?:\s*"

        # Keep removing prefixes until none remain
        while re.match(pattern, subject):
            subject = re.sub(pattern, "", subject, count=1).strip()

        return subject

    def calculate_response_times(
        self,
        thread: EmailThread,
        messages: Dict[str, EmailMessage],
    ) -> None:
        """Calculate response time statistics for thread.

        Computes time deltas between parent and child messages to
        measure conversation velocity. Updates thread with total
        and average response times.

        Args:
            thread: EmailThread to analyze
            messages: Dictionary mapping Message-ID to EmailMessage
        """
        response_times = []

        for message_id in thread.message_ids:
            if message_id not in self.message_graph:
                continue

            node = self.message_graph[message_id]

            # Skip if no parent or parent not in graph
            if not node.parent_message_id or node.parent_message_id not in self.message_graph:
                continue

            parent = self.message_graph[node.parent_message_id]

            # Calculate response time in minutes
            response_time_minutes = (node.date - parent.date).total_seconds() / 60

            # Only count positive response times (ignore clock skew)
            if response_time_minutes > 0:
                response_times.append(response_time_minutes)

        # Update thread statistics
        if response_times:
            thread.total_response_time_minutes = sum(response_times)
            thread.average_response_time_minutes = sum(response_times) / len(response_times)

    def handle_orphan_message(
        self, email_message: EmailMessage, placeholder_offset_minutes: int = 30
    ) -> None:
        """Handle message whose parent is not in mailbox.

        Creates a placeholder parent node to maintain thread structure
        when parent messages are missing (e.g., deleted, different mailbox).

        Args:
            email_message: Orphan email message
            placeholder_offset_minutes: Estimated time before orphan (default: 30 min)
        """
        if not email_message.in_reply_to:
            return

        # Only create placeholder if parent truly doesn't exist
        if email_message.in_reply_to in self.message_graph:
            return

        # Create placeholder parent node
        placeholder = ThreadNode(
            message_id=email_message.in_reply_to,
            parent_message_id=None,  # Assume it's a root
            date=email_message.date - timedelta(minutes=placeholder_offset_minutes),
            from_address="unknown@unknown",
            subject=email_message.subject or "",
        )

        self.message_graph[email_message.in_reply_to] = placeholder

        logger.debug(
            f"Created placeholder parent node for orphan message {email_message.message_id}",
            extra={
                "orphan_id": email_message.message_id,
                "placeholder_parent_id": email_message.in_reply_to,
            },
        )

    def get_thread_for_message(self, message_id: str) -> Optional[EmailThread]:
        """Get the thread containing a specific message.

        Args:
            message_id: Message-ID to search for

        Returns:
            EmailThread containing the message, or None if not found
        """
        for thread in self.threads.values():
            if message_id in thread.message_ids:
                return thread
        return None

    def get_thread_statistics(self) -> Dict[str, any]:
        """Get overall threading statistics.

        Returns:
            Dictionary with threading metrics:
                - total_threads: Total number of threads
                - total_messages: Total messages in all threads
                - avg_messages_per_thread: Average messages per thread
                - avg_participants_per_thread: Average participants per thread
                - deepest_thread_depth: Maximum thread depth
                - longest_thread_days: Longest thread duration
        """
        if not self.threads:
            return {
                "total_threads": 0,
                "total_messages": 0,
                "avg_messages_per_thread": 0,
                "avg_participants_per_thread": 0,
                "deepest_thread_depth": 0,
                "longest_thread_days": 0,
            }

        threads = list(self.threads.values())
        total_messages = sum(t.message_count for t in threads)
        total_participants = sum(t.participant_count for t in threads)

        return {
            "total_threads": len(threads),
            "total_messages": total_messages,
            "avg_messages_per_thread": total_messages / len(threads),
            "avg_participants_per_thread": total_participants / len(threads),
            "deepest_thread_depth": max((t.depth for t in threads), default=0),
            "longest_thread_days": max((t.duration_days for t in threads), default=0),
        }


__all__ = ["ThreadReconstructor"]
