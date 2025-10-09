"""Email format adapter.

Leverages existing EmailParser and EmailNormalizer for email document processing.
Handles .eml and .msg files with full metadata extraction including threading,
participants, and attachments.
"""

from __future__ import annotations

import logging
from email import message_from_bytes
from email.policy import default as email_policy
from pathlib import Path

from ...models import DocumentFormat, NormalizedDocument
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class EmailAdapter(BaseAdapter):
    """Adapter for email documents.

    Leverages existing EmailParser and EmailNormalizer for robust email parsing
    with support for RFC822/MIME format, threading reconstruction, and participant
    tracking. Does not require Unstructured.io processing.

    Example:
        >>> adapter = EmailAdapter()
        >>> doc = await adapter.normalize(
        ...     file_path=Path("message.eml"),
        ...     source_id="email-123",
        ...     source_type="local_files",
        ...     source_metadata={}
        ... )
    """

    def __init__(self):
        super().__init__(
            name="EmailAdapter",
            supported_formats=[DocumentFormat.EMAIL],
        )
        self.requires_unstructured_processing = False

        # Import email processing components
        try:
            from ....ingestion.imap.email_parser import EmailParser
            from ....ingestion.imap.email_normalizer import EmailNormalizer

            self._email_parser = EmailParser()
            self._email_normalizer = EmailNormalizer()
        except ImportError as e:
            logger.warning(f"Email processing components not available: {e}")
            self._email_parser = None
            self._email_normalizer = None

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize email document.

        Args:
            file_path: Path to email file (.eml or .msg)
            source_id: Connector-specific identifier
            source_type: Source type
            source_metadata: Additional metadata

        Returns:
            NormalizedDocument with parsed email content and metadata

        Raises:
            AdapterError: If email parsing fails
        """
        try:
            # Validate email parser is available
            if not self._email_parser or not self._email_normalizer:
                from ..registry import AdapterError

                raise AdapterError(
                    "Email processing components not available. "
                    "Ensure ingestion.imap modules are properly installed."
                )

            # Read raw email file
            with open(file_path, "rb") as f:
                raw_message = f.read()

            # Parse email using existing EmailParser
            # Note: We use dummy UID/folder since this is file-based, not IMAP
            email_message = self._email_parser.parse_message(
                raw_message=raw_message,
                uid=0,  # Not from IMAP
                folder="local",
                mailbox_id=source_id,
            )

            # Normalize email for pipeline processing
            normalized_text = self._email_normalizer.normalize(email_message)

            # Extract participants for metadata
            participants = self._extract_participants(email_message)

            # Build tags from email metadata
            tags = []
            if email_message.labels:
                tags.extend(email_message.labels)
            if email_message.flags:
                # Convert IMAP flags to tags
                flag_tags = [f"flag:{flag.lower()}" for flag in email_message.flags]
                tags.extend(flag_tags)

            # Create normalized document
            document = self.create_normalized_document(
                content=normalized_text,
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                format=DocumentFormat.EMAIL,
                source_metadata=source_metadata,
                tags=tags,
            )

            # Add email-specific metadata
            document.metadata.extra["email"] = {
                "message_id": email_message.message_id,
                "from": email_message.from_address.address,
                "from_name": email_message.from_address.display_name,
                "to": [addr.address for addr in email_message.to_addresses],
                "cc": [addr.address for addr in email_message.cc_addresses],
                "subject": email_message.subject,
                "date": email_message.date.isoformat(),
                "has_attachments": email_message.has_attachments,
                "attachment_count": len(email_message.attachments),
                "participant_count": email_message.participant_count,
                "is_reply": email_message.is_reply,
                "in_reply_to": email_message.in_reply_to,
                "references": email_message.references,
                "contains_sensitive": email_message.contains_sensitive_keywords,
                "participants": participants,
            }

            # Add attachment metadata if present
            if email_message.attachments:
                document.metadata.extra["email"]["attachments"] = [
                    {
                        "filename": att.filename,
                        "content_type": att.content_type,
                        "size_bytes": att.size_bytes,
                        "is_inline": att.is_inline,
                    }
                    for att in email_message.attachments
                ]

            logger.debug(
                f"Normalized email document: {file_path.name} "
                f"(from: {email_message.from_address.address}, "
                f"participants: {email_message.participant_count})"
            )

            return document

        except Exception as e:
            logger.error(f"Email normalization failed for {file_path.name}: {e}")
            from ..registry import AdapterError

            raise AdapterError(f"Failed to normalize email document: {str(e)}") from e

    def _extract_participants(self, email_message) -> list[dict]:
        """Extract all email participants with roles.

        Args:
            email_message: Parsed EmailMessage

        Returns:
            List of participant dictionaries with address and role
        """
        participants = []

        # Add sender
        participants.append(
            {
                "address": email_message.from_address.address,
                "name": email_message.from_address.display_name,
                "role": "from",
            }
        )

        # Add recipients
        for addr in email_message.to_addresses:
            participants.append(
                {"address": addr.address, "name": addr.display_name, "role": "to"}
            )

        # Add CC recipients
        for addr in email_message.cc_addresses:
            participants.append(
                {"address": addr.address, "name": addr.display_name, "role": "cc"}
            )

        # Add BCC recipients (rarely present)
        for addr in email_message.bcc_addresses:
            participants.append(
                {"address": addr.address, "name": addr.display_name, "role": "bcc"}
            )

        return participants

    async def validate(self, file_path: Path) -> bool:
        """Validate email file.

        Checks file extension and attempts to parse as email message.

        Args:
            file_path: Path to validate

        Returns:
            True if file is a valid email
        """
        # Check extension
        if file_path.suffix.lower() not in [".eml", ".msg"]:
            return False

        # Check file exists
        if not file_path.exists():
            return False

        # Try to parse as email (basic validation)
        try:
            with open(file_path, "rb") as f:
                raw = f.read()

            # Attempt to parse with email module
            msg = message_from_bytes(raw, policy=email_policy)

            # Check for required email headers
            has_from = msg.get("From") is not None
            has_date = msg.get("Date") is not None

            return has_from or has_date  # At least one core header present

        except Exception as e:
            logger.debug(f"Email validation failed for {file_path.name}: {e}")
            return False
