"""Email parser for RFC822/MIME messages with privacy-aware processing.

This module parses raw email messages into structured EmailMessage objects,
extracting headers, threading information, body content, and attachment metadata
while maintaining strict privacy compliance through audit logging without content
exposure.

Privacy-First Design:
- Uses Python standard library email.parser (no external parsing dependencies)
- Audit logs contain hashed message IDs only, never content
- Respects MailboxPrivacySettings for keyword detection and anonymization
- Attachment metadata only, no content downloaded

Integration:
- Feeds normalized content to Unstructured.io pipeline
- Generates semantic triples for PKG construction
- Compatible with existing privacy framework (AuditLogger, ConsentRegistry)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from email import message_from_bytes
from email.header import decode_header
from email.message import EmailMessage as StdEmailMessage
from email.policy import default as email_policy
from email.utils import getaddresses, parsedate_to_datetime
from hashlib import sha256
from typing import Dict, List, Optional, Tuple

import html2text
from pydantic import BaseModel, Field, field_validator

from .descriptor import MailboxPrivacySettings
from ...privacy.audit import AuditEvent, AuditLogger
from ...privacy.consent import ConsentRequiredError

# Import privacy components (optional to maintain backward compatibility)
try:
    from .consent_manager import ImapConsentManager, ImapConsentScopes
    from .email_redaction import EmailHeaderRedactionPolicy
    from .audit_events import log_email_processing_event, log_consent_check_failed
    PRIVACY_COMPONENTS_AVAILABLE = True
except ImportError:
    PRIVACY_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmailAddress(BaseModel):
    """Parsed email address with display name.

    Represents a single email participant with optional display name,
    following RFC 5322 address format.
    """

    address: str = Field(..., description="Email address (user@domain.com)")
    display_name: Optional[str] = Field(
        default=None, description="Display name (e.g., 'John Doe')"
    )

    @field_validator("address")
    @classmethod
    def _validate_address(cls, value: str) -> str:  # type: ignore[override]
        """Validate email address has @ symbol."""
        if "@" not in value or value.count("@") != 1:
            raise ValueError(f"Invalid email address: {value}")
        return value.lower()

    @classmethod
    def from_header(cls, header_value: str) -> List[EmailAddress]:
        """Parse email addresses from header value.

        Args:
            header_value: Raw header value (e.g., "John Doe <john@example.com>, jane@example.com")

        Returns:
            List of parsed EmailAddress objects
        """
        if not header_value or not header_value.strip():
            return []

        addresses = getaddresses([header_value])
        result = []

        for display_name, addr in addresses:
            if not addr or "@" not in addr:
                continue

            result.append(
                cls(
                    address=addr.strip().lower(),
                    display_name=display_name.strip() if display_name else None,
                )
            )

        return result


class AttachmentMetadata(BaseModel):
    """Metadata for email attachment (no content stored).

    Privacy-first design: stores only metadata, never downloads or stores
    actual attachment content.
    """

    filename: str = Field(..., description="Attachment filename")
    content_type: str = Field(..., description="MIME content type")
    size_bytes: int = Field(..., ge=0, description="Attachment size in bytes")
    part_id: str = Field(..., description="MIME part identifier")
    is_inline: bool = Field(default=False, description="True if inline attachment")
    content_id: Optional[str] = Field(
        default=None, description="Content-ID for inline images"
    )


class EmailMessage(BaseModel):
    """Parsed email message with metadata and privacy compliance.

    Comprehensive email representation extracted from RFC822/MIME format,
    including threading information, participants, content, and privacy
    classification.
    """

    # Identity
    message_id: str = Field(..., description="Unique message identifier (Message-ID)")
    uid: int = Field(..., description="IMAP UID")
    folder: str = Field(..., description="IMAP folder path")

    # Headers
    subject: Optional[str] = Field(default=None, description="Email subject")
    from_address: EmailAddress = Field(..., description="Sender address")
    to_addresses: List[EmailAddress] = Field(
        default_factory=list, description="To recipients"
    )
    cc_addresses: List[EmailAddress] = Field(
        default_factory=list, description="Cc recipients"
    )
    bcc_addresses: List[EmailAddress] = Field(
        default_factory=list, description="Bcc recipients"
    )
    reply_to_addresses: List[EmailAddress] = Field(
        default_factory=list, description="Reply-To addresses"
    )
    date: datetime = Field(..., description="Email sent date")

    # Threading headers for conversation reconstruction
    in_reply_to: Optional[str] = Field(
        default=None, description="Message-ID of parent message"
    )
    references: List[str] = Field(
        default_factory=list, description="Thread chain of Message-IDs"
    )

    # Content (plain text preferred, HTML fallback)
    body_plain: Optional[str] = Field(default=None, description="Plain text body")
    body_html: Optional[str] = Field(default=None, description="HTML body")
    body_normalized: Optional[str] = Field(
        default=None, description="Normalized text for Unstructured.io"
    )

    # Metadata
    size_bytes: int = Field(..., ge=0, description="Raw message size")
    flags: List[str] = Field(default_factory=list, description="IMAP flags")
    labels: List[str] = Field(default_factory=list, description="Gmail labels")
    attachments: List[AttachmentMetadata] = Field(
        default_factory=list, description="Attachment metadata"
    )

    # Privacy classification
    contains_sensitive_keywords: bool = Field(
        default=False, description="True if privacy keywords detected"
    )
    privacy_classification: str = Field(
        default="standard", description="Privacy level classification"
    )

    # Provenance
    retrieved_at: datetime = Field(..., description="Timestamp of retrieval")
    mailbox_id: str = Field(..., description="Source mailbox identifier")

    @property
    def has_attachments(self) -> bool:
        """Check if message has attachments."""
        return len(self.attachments) > 0

    @property
    def participant_count(self) -> int:
        """Total number of participants (from + to + cc)."""
        return 1 + len(self.to_addresses) + len(self.cc_addresses)

    @property
    def is_reply(self) -> bool:
        """Check if message is a reply (has in_reply_to)."""
        return self.in_reply_to is not None


class EmailParser:
    """Parse RFC822/MIME emails into structured EmailMessage format.

    Privacy-aware email parser that extracts headers, body content, and
    threading information while respecting privacy settings and emitting
    audit events without content exposure.

    Uses Python standard library email.parser for maximum privacy and
    reliability.
    """

    def __init__(
        self,
        *,
        privacy_policy: Optional[MailboxPrivacySettings] = None,
        audit_logger: Optional[AuditLogger] = None,
        consent_manager: Optional[object] = None,  # ImapConsentManager (optional)
    ):
        """Initialize email parser.

        Args:
            privacy_policy: Privacy settings for keyword detection and anonymization
            audit_logger: Audit logger for privacy-compliant event recording
            consent_manager: Optional consent manager for enforcement (ImapConsentManager)
        """
        self.privacy_policy = privacy_policy or MailboxPrivacySettings()
        self.audit_logger = audit_logger
        self.consent_manager = consent_manager

        # Create redaction policy for audit logging
        if PRIVACY_COMPONENTS_AVAILABLE and privacy_policy:
            self.redaction_policy = EmailHeaderRedactionPolicy(
                redact_sender=privacy_policy.enable_sender_anonymization,
                redact_recipients=privacy_policy.enable_recipient_anonymization,
                redact_subject=privacy_policy.enable_subject_redaction,
            )
        else:
            self.redaction_policy = None

        # Configure html2text for clean conversion
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # No line wrapping

    def parse_message(
        self,
        *,
        raw_message: bytes,
        uid: int,
        folder: str,
        mailbox_id: str,
        flags: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> EmailMessage:
        """Parse raw RFC822 message into EmailMessage.

        Args:
            raw_message: Raw RFC822/MIME message bytes
            uid: IMAP UID
            folder: IMAP folder path
            mailbox_id: Source mailbox identifier
            flags: Optional IMAP flags
            labels: Optional Gmail labels

        Returns:
            Parsed EmailMessage object

        Raises:
            ValueError: If message cannot be parsed
            ConsentRequiredError: If required consent not granted
        """
        # Consent enforcement: Check required consents before parsing
        if self.consent_manager and PRIVACY_COMPONENTS_AVAILABLE:
            try:
                # Require email content analysis consent
                self.consent_manager.require_consent(
                    mailbox_id=mailbox_id,
                    scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
                )
            except ConsentRequiredError as e:
                # Log consent check failure
                if self.audit_logger and PRIVACY_COMPONENTS_AVAILABLE:
                    log_consent_check_failed(
                        self.audit_logger,
                        mailbox_id=mailbox_id,
                        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
                        operation="email_parsing",
                    )
                raise

        try:
            # Parse with Python email.parser (handles encoding automatically)
            msg = message_from_bytes(raw_message, policy=email_policy)

            # Extract all components
            message_id = self._extract_message_id(msg)
            subject = self._extract_subject(msg)
            from_addr = self._extract_from(msg)
            to_addrs = self._extract_to(msg)
            cc_addrs = self._extract_cc(msg)
            bcc_addrs = self._extract_bcc(msg)
            reply_to_addrs = self._extract_reply_to(msg)
            date = self._extract_date(msg)
            in_reply_to = self._extract_in_reply_to(msg)
            references = self._extract_references(msg)

            # Extract body content
            body_plain, body_html = self._extract_body(msg)
            body_normalized = self._normalize_body(body_plain, body_html)

            # Extract attachments
            attachments = self._extract_attachments(msg)

            # Privacy classification
            contains_sensitive = self._check_sensitive_keywords(
                subject, body_normalized
            )

            # Build EmailMessage
            email_message = EmailMessage(
                message_id=message_id,
                uid=uid,
                folder=folder,
                subject=subject,
                from_address=from_addr,
                to_addresses=to_addrs,
                cc_addresses=cc_addrs,
                bcc_addresses=bcc_addrs,
                reply_to_addresses=reply_to_addrs,
                date=date,
                in_reply_to=in_reply_to,
                references=references,
                body_plain=body_plain,
                body_html=body_html,
                body_normalized=body_normalized,
                size_bytes=len(raw_message),
                flags=flags or [],
                labels=labels or [],
                attachments=attachments,
                contains_sensitive_keywords=contains_sensitive,
                privacy_classification="sensitive" if contains_sensitive else "standard",
                retrieved_at=datetime.utcnow(),
                mailbox_id=mailbox_id,
            )

            # Log parsing event (without content)
            self._log_parse_event(email_message)

            return email_message

        except Exception as e:
            logger.error(
                f"Failed to parse email message: {e}",
                extra={"uid": uid, "folder": folder, "error": str(e)},
            )
            raise ValueError(f"Email parsing failed: {e}")

    def _extract_message_id(self, msg: StdEmailMessage) -> str:
        """Extract Message-ID header with fallback generation.

        Args:
            msg: Parsed email message

        Returns:
            Message-ID (generated if missing)
        """
        message_id = msg.get("Message-ID", "").strip("<>").strip()

        if not message_id:
            # Generate fallback ID for messages without Message-ID
            fallback_id = f"generated-{uuid.uuid4()}@futurnal.local"
            logger.warning(
                f"Message missing Message-ID, generated fallback: {fallback_id}"
            )
            return fallback_id

        return message_id

    def _extract_subject(self, msg: StdEmailMessage) -> Optional[str]:
        """Extract and decode subject header.

        Handles encoded words (RFC 2047) and various character encodings.

        Args:
            msg: Parsed email message

        Returns:
            Decoded subject or None if missing
        """
        subject = msg.get("Subject", "")
        if not subject:
            return None

        # Decode encoded words (=?utf-8?B?...?=)
        decoded_parts = decode_header(subject)
        result = ""

        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                # Decode bytes with specified charset or fallback to utf-8
                try:
                    result += part.decode(charset or "utf-8", errors="replace")
                except (UnicodeDecodeError, LookupError):
                    result += part.decode("utf-8", errors="replace")
            else:
                result += part

        return result.strip() if result.strip() else None

    def _extract_from(self, msg: StdEmailMessage) -> EmailAddress:
        """Extract From address.

        Args:
            msg: Parsed email message

        Returns:
            EmailAddress object (fallback to unknown@unknown if missing)
        """
        from_header = msg.get("From", "")
        addresses = EmailAddress.from_header(from_header)

        if not addresses:
            logger.warning("Message missing From header, using fallback")
            return EmailAddress(address="unknown@unknown", display_name=None)

        return addresses[0]

    def _extract_to(self, msg: StdEmailMessage) -> List[EmailAddress]:
        """Extract To addresses."""
        to_header = msg.get("To", "")
        return EmailAddress.from_header(to_header)

    def _extract_cc(self, msg: StdEmailMessage) -> List[EmailAddress]:
        """Extract Cc addresses."""
        cc_header = msg.get("Cc", "")
        return EmailAddress.from_header(cc_header)

    def _extract_bcc(self, msg: StdEmailMessage) -> List[EmailAddress]:
        """Extract Bcc addresses (usually not present in received messages)."""
        bcc_header = msg.get("Bcc", "")
        return EmailAddress.from_header(bcc_header)

    def _extract_reply_to(self, msg: StdEmailMessage) -> List[EmailAddress]:
        """Extract Reply-To addresses."""
        reply_to_header = msg.get("Reply-To", "")
        return EmailAddress.from_header(reply_to_header)

    def _extract_date(self, msg: StdEmailMessage) -> datetime:
        """Extract and parse Date header.

        Args:
            msg: Parsed email message

        Returns:
            Parsed datetime (fallback to current time if missing/invalid)
        """
        date_header = msg.get("Date")

        if date_header:
            try:
                return parsedate_to_datetime(date_header)
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse Date header '{date_header}': {e}, using current time"
                )

        return datetime.utcnow()

    def _extract_in_reply_to(self, msg: StdEmailMessage) -> Optional[str]:
        """Extract In-Reply-To header for threading.

        Args:
            msg: Parsed email message

        Returns:
            Parent message ID or None
        """
        in_reply_to = msg.get("In-Reply-To", "").strip("<>").strip()
        return in_reply_to if in_reply_to else None

    def _extract_references(self, msg: StdEmailMessage) -> List[str]:
        """Extract References header (thread chain).

        Args:
            msg: Parsed email message

        Returns:
            List of Message-IDs in thread chain
        """
        references = msg.get("References", "")

        if not references:
            return []

        # Parse space-separated Message-IDs
        return [ref.strip("<>").strip() for ref in references.split() if ref.strip()]

    def _extract_body(
        self, msg: StdEmailMessage
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract plain text and HTML body parts.

        Handles multipart MIME messages, preferring text/plain over text/html.

        Args:
            msg: Parsed email message

        Returns:
            Tuple of (plain_text_body, html_body)
        """
        body_plain = None
        body_html = None

        if msg.is_multipart():
            # Walk through all parts
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get_content_disposition()

                # Skip attachments
                if content_disposition == "attachment":
                    continue

                # Extract plain text (first occurrence)
                if content_type == "text/plain" and body_plain is None:
                    try:
                        body_plain = part.get_content()
                    except Exception as e:
                        logger.warning(f"Failed to extract plain text body: {e}")

                # Extract HTML (first occurrence)
                elif content_type == "text/html" and body_html is None:
                    try:
                        body_html = part.get_content()
                    except Exception as e:
                        logger.warning(f"Failed to extract HTML body: {e}")

        else:
            # Single-part message
            content_type = msg.get_content_type()

            try:
                if content_type == "text/plain":
                    body_plain = msg.get_content()
                elif content_type == "text/html":
                    body_html = msg.get_content()
            except Exception as e:
                logger.warning(f"Failed to extract body from single-part message: {e}")

        return body_plain, body_html

    def _normalize_body(
        self, plain: Optional[str], html: Optional[str]
    ) -> Optional[str]:
        """Normalize body for Unstructured.io processing.

        Prefers plain text, falls back to HTML-to-text conversion.

        Args:
            plain: Plain text body
            html: HTML body

        Returns:
            Normalized text or empty string if no body content
        """
        if plain is not None:
            return plain.strip()

        if html is not None:
            try:
                # Convert HTML to plain text using html2text
                text = self.html_converter.handle(html)
                return text.strip()
            except Exception as e:
                logger.warning(f"HTML to text conversion failed: {e}")
                # Fallback: basic HTML stripping
                import re
                from html import unescape

                text = re.sub("<[^<]+?>", "", html)  # Strip tags
                text = unescape(text)
                return text.strip()

        return ""

    def _extract_attachments(self, msg: StdEmailMessage) -> List[AttachmentMetadata]:
        """Extract attachment metadata (no content).

        Privacy-first: only extracts metadata, never downloads or stores
        actual attachment content.

        Args:
            msg: Parsed email message

        Returns:
            List of AttachmentMetadata objects
        """
        attachments = []

        if not msg.is_multipart():
            return attachments

        for part in msg.walk():
            content_disposition = part.get_content_disposition()

            # Check for attachments and inline parts
            if content_disposition in ("attachment", "inline"):
                filename = part.get_filename()

                if filename:
                    try:
                        # Get attachment size
                        payload = part.get_payload(decode=True)
                        size = len(payload) if payload else 0

                        # Extract Content-ID for inline images
                        content_id = part.get("Content-ID", "").strip("<>").strip()

                        attachments.append(
                            AttachmentMetadata(
                                filename=filename,
                                content_type=part.get_content_type(),
                                size_bytes=size,
                                part_id=content_id or str(len(attachments)),
                                is_inline=content_disposition == "inline",
                                content_id=content_id if content_id else None,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract attachment metadata for '{filename}': {e}"
                        )

        return attachments

    def _check_sensitive_keywords(
        self, subject: Optional[str], body: Optional[str]
    ) -> bool:
        """Check for sensitive keywords in subject/body.

        Args:
            subject: Email subject
            body: Email body (normalized)

        Returns:
            True if sensitive keywords detected
        """
        if not self.privacy_policy or not self.privacy_policy.privacy_subject_keywords:
            return False

        keywords = self.privacy_policy.privacy_subject_keywords
        text = f"{subject or ''} {body or ''}".lower()

        return any(keyword.lower() in text for keyword in keywords)

    def _log_parse_event(self, email_message: EmailMessage) -> None:
        """Log parsing event without email content.

        Privacy-compliant logging: hashes message IDs, never logs content.
        Uses enhanced redaction if available.

        Args:
            email_message: Parsed email message
        """
        if not self.audit_logger:
            return

        # Use enhanced logging with redaction if available
        if PRIVACY_COMPONENTS_AVAILABLE and self.redaction_policy:
            log_email_processing_event(
                self.audit_logger,
                email_message=email_message,
                redaction_policy=self.redaction_policy,
                status="success",
            )
        else:
            # Fallback to basic logging
            message_id_hash = sha256(email_message.message_id.encode()).hexdigest()[:16]

            self.audit_logger.record(
                AuditEvent(
                    job_id=f"email_parse_{email_message.uid}",
                    source="imap_email_parser",
                    action="email_parsed",
                    status="success",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "message_id_hash": message_id_hash,
                        "folder": email_message.folder,
                        "uid": email_message.uid,
                        "has_attachments": email_message.has_attachments,
                        "attachment_count": len(email_message.attachments),
                        "body_length": len(email_message.body_normalized or ""),
                        "contains_sensitive": email_message.contains_sensitive_keywords,
                        "participant_count": email_message.participant_count,
                        "is_reply": email_message.is_reply,
                        "mailbox_id": email_message.mailbox_id,
                    },
                )
            )


__all__ = [
    "EmailAddress",
    "AttachmentMetadata",
    "EmailMessage",
    "EmailParser",
]
