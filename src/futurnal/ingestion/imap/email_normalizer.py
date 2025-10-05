"""Email normalizer for Unstructured.io processing pipeline.

This module converts parsed EmailMessage objects into a normalized text format
optimized for Unstructured.io document processing, preserving metadata headers
and conversation context while maintaining clean, structured output.

The normalized format includes:
- Structured metadata headers (From, To, Cc, Date, Subject)
- Threading context (In-Reply-To, References)
- Clean body content
- Attachment metadata listing

This format enables the Ghost to learn from email content while preserving
conversational structure, participant relationships, and temporal context.
"""

from __future__ import annotations

import logging
from typing import List

from .email_parser import EmailMessage, EmailAddress

logger = logging.getLogger(__name__)


class EmailNormalizer:
    """Normalize email content for Unstructured.io processing.

    Converts EmailMessage objects into a clean, structured text format that:
    - Preserves all metadata as structured headers
    - Maintains threading context for conversation reconstruction
    - Includes attachment information without content
    - Provides clean, parseable format for LLM ingestion
    """

    def __init__(self):
        """Initialize email normalizer."""
        pass

    def normalize(self, email_message: EmailMessage) -> str:
        """Convert email to normalized text format for Unstructured.io.

        Creates a structured text document with:
        1. Metadata header section (From, To, Cc, Date, Subject)
        2. Threading context (In-Reply-To, References)
        3. Separator line
        4. Body content
        5. Attachment metadata (if present)

        Args:
            email_message: Parsed email message

        Returns:
            Normalized text suitable for Unstructured.io processing
        """
        lines = []

        # Add sender information
        lines.extend(self._format_sender(email_message.from_address))

        # Add recipients
        lines.extend(self._format_recipients(email_message))

        # Add date
        lines.append(f"Date: {email_message.date.isoformat()}")

        # Add subject
        if email_message.subject:
            lines.append(f"Subject: {email_message.subject}")

        # Add threading context
        lines.extend(self._format_threading(email_message))

        # Add separator
        lines.append("")
        lines.append("---")
        lines.append("")

        # Add body content
        if email_message.body_normalized:
            lines.append(email_message.body_normalized)

        # Add attachment information
        if email_message.attachments:
            lines.extend(self._format_attachments(email_message.attachments))

        return "\n".join(lines)

    def _format_sender(self, from_address: EmailAddress) -> List[str]:
        """Format sender information.

        Args:
            from_address: Sender email address

        Returns:
            List of formatted lines
        """
        lines = []
        lines.append(f"From: {from_address.address}")

        if from_address.display_name:
            lines.append(f"From Name: {from_address.display_name}")

        return lines

    def _format_recipients(self, email_message: EmailMessage) -> List[str]:
        """Format recipient information.

        Args:
            email_message: Email message

        Returns:
            List of formatted lines
        """
        lines = []

        # Format To addresses
        if email_message.to_addresses:
            to_addrs = ", ".join(addr.address for addr in email_message.to_addresses)
            lines.append(f"To: {to_addrs}")

        # Format Cc addresses
        if email_message.cc_addresses:
            cc_addrs = ", ".join(addr.address for addr in email_message.cc_addresses)
            lines.append(f"Cc: {cc_addrs}")

        return lines

    def _format_threading(self, email_message: EmailMessage) -> List[str]:
        """Format threading context.

        Args:
            email_message: Email message

        Returns:
            List of formatted lines
        """
        lines = []

        # Add In-Reply-To
        if email_message.in_reply_to:
            lines.append(f"In-Reply-To: {email_message.in_reply_to}")

        # Add References
        if email_message.references:
            refs = ", ".join(email_message.references)
            lines.append(f"References: {refs}")

        return lines

    def _format_attachments(self, attachments) -> List[str]:
        """Format attachment metadata.

        Args:
            attachments: List of AttachmentMetadata objects

        Returns:
            List of formatted lines
        """
        lines = []
        lines.append("")
        lines.append("Attachments:")

        for att in attachments:
            # Format size in human-readable format
            size_str = self._format_size(att.size_bytes)
            inline_marker = " (inline)" if att.is_inline else ""
            lines.append(
                f"  - {att.filename} ({att.content_type}, {size_str}){inline_marker}"
            )

        return lines

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string (e.g., "1.5 MB", "512 KB")
        """
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


__all__ = ["EmailNormalizer"]
