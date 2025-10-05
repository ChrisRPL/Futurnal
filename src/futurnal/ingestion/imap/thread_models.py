"""Email thread reconstruction data models.

This module defines the data models for reconstructing email conversation threads
from Message-ID, References, and In-Reply-To headers per RFC 2822. These models
enable the Ghost to understand communication patterns, relationship dynamics, and
conversational context evolution over time.

Models:
- ParticipantRole: Enum defining participant roles in threads
- ThreadParticipant: Individual participant in a conversation thread
- ThreadNode: Node in the thread tree structure
- EmailThread: Complete reconstructed conversation thread

Ghost→Animal Evolution:
- **Phase 1 (CURRENT)**: Mechanical thread reconstruction from headers
- **Phase 2 (FUTURE)**: Participant behavior pattern analysis
- **Phase 3 (FUTURE)**: Conversational dynamics and causal relationship inference
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ParticipantRole(str, Enum):
    """Role of a participant in an email thread.

    Roles are assigned based on position in the conversation:
    - INITIATOR: Started the thread (sender of root message)
    - PRIMARY_RECIPIENT: In To: field of root message
    - PARTICIPANT: Active participant who sent messages in thread
    - CC_RECIPIENT: Only appears in Cc: fields, never sent messages
    - OBSERVER: Only in Bcc: fields (rarely known)
    """

    INITIATOR = "initiator"
    PRIMARY_RECIPIENT = "primary_recipient"
    PARTICIPANT = "participant"
    CC_RECIPIENT = "cc_recipient"
    OBSERVER = "observer"


class ThreadParticipant(BaseModel):
    """Participant in an email thread with engagement metrics.

    Tracks individual's involvement in a conversation thread including
    their role, message count, and temporal participation boundaries.
    """

    email_address: str = Field(..., description="Participant email address")
    display_name: Optional[str] = Field(
        default=None, description="Display name from email headers"
    )
    role: ParticipantRole = Field(..., description="Participant's role in thread")
    message_count: int = Field(default=0, ge=0, description="Number of messages sent")
    first_message_date: datetime = Field(
        ..., description="Date of first participation"
    )
    last_message_date: datetime = Field(..., description="Date of last participation")

    @property
    def participation_duration_days(self) -> float:
        """Calculate duration of participation in days."""
        delta = self.last_message_date - self.first_message_date
        return delta.total_seconds() / 86400


class ThreadNode(BaseModel):
    """Node in email thread tree structure.

    Represents a single message in the conversation tree with links to
    parent and children messages, enabling tree traversal and depth calculation.
    """

    message_id: str = Field(..., description="Message-ID of this message")
    parent_message_id: Optional[str] = Field(
        default=None, description="Message-ID of parent (None for root)"
    )
    children: List[str] = Field(
        default_factory=list, description="Message-IDs of child messages"
    )
    depth: int = Field(default=0, ge=0, description="Depth in thread tree (0 for root)")
    date: datetime = Field(..., description="Message sent date")
    from_address: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Message subject line")

    @property
    def is_root(self) -> bool:
        """Check if this is a root message (no parent)."""
        return self.parent_message_id is None

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf message (no children)."""
        return len(self.children) == 0

    @property
    def child_count(self) -> int:
        """Number of direct children."""
        return len(self.children)


class EmailThread(BaseModel):
    """Reconstructed email conversation thread with metadata and analytics.

    Represents a complete conversation thread reconstructed from email
    Message-ID graph, including all messages, participants, temporal data,
    and structural analysis (depth, branches, response times).
    """

    # Identity
    thread_id: str = Field(..., description="Root Message-ID or generated ID")
    root_message_id: str = Field(..., description="First message in thread")

    # Messages
    message_ids: List[str] = Field(
        default_factory=list, description="All Message-IDs in thread"
    )
    message_count: int = Field(default=0, ge=0, description="Total message count")

    # Participants
    participants: List[ThreadParticipant] = Field(
        default_factory=list, description="All thread participants"
    )
    participant_count: int = Field(default=0, ge=0, description="Total participant count")

    # Thread metadata
    subject: str = Field(..., description="Original subject (without Re:/Fwd:)")
    subject_variations: List[str] = Field(
        default_factory=list, description="Subject line variations in thread"
    )
    start_date: datetime = Field(..., description="Date of first message")
    last_message_date: datetime = Field(..., description="Date of last message")
    duration_days: float = Field(default=0.0, ge=0.0, description="Thread duration in days")

    # Thread structure
    depth: int = Field(default=0, ge=0, description="Maximum depth of thread tree")
    branch_count: int = Field(default=0, ge=0, description="Number of branches")

    # Analysis
    total_response_time_minutes: float = Field(
        default=0.0, ge=0.0, description="Sum of all response times"
    )
    average_response_time_minutes: float = Field(
        default=0.0, ge=0.0, description="Average response time"
    )
    has_attachments: bool = Field(
        default=False, description="True if any message has attachments"
    )

    # Provenance
    mailbox_id: str = Field(..., description="Source mailbox identifier")
    reconstructed_at: datetime = Field(..., description="Timestamp of reconstruction")

    @property
    def is_multi_participant(self) -> bool:
        """Check if thread has multiple participants."""
        return self.participant_count > 1

    @property
    def is_long_thread(self) -> bool:
        """Check if thread is long (>10 messages)."""
        return self.message_count > 10

    @property
    def is_deep_thread(self) -> bool:
        """Check if thread is deep (>5 levels)."""
        return self.depth > 5

    @property
    def is_branching_thread(self) -> bool:
        """Check if thread has branches."""
        return self.branch_count > 0

    @property
    def messages_per_day(self) -> float:
        """Calculate average messages per day."""
        if self.duration_days == 0:
            return float(self.message_count)
        return self.message_count / max(self.duration_days, 0.01)


__all__ = [
    "ParticipantRole",
    "ThreadParticipant",
    "ThreadNode",
    "EmailThread",
]
