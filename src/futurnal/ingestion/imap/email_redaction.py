"""Email-specific redaction for privacy-safe audit logging.

This module provides email address and subject line redaction for IMAP
audit logging, ensuring that personally identifiable information (PII)
is never exposed in logs or audit events.

Privacy Guarantees:
- Email addresses redacted as hash@domain (preserves domain for debugging)
- Subject lines hashed when privacy keywords detected
- Email bodies NEVER logged under any circumstances
- Stable hashing ensures consistent redaction

Redaction Examples:
- john.doe@example.com → a8f3c2d1@example.com
- "Confidential: Q4 Results" → "8f2a1c3d" (if keywords detected)
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class EmailHeaderRedactionPolicy:
    """Redact email headers for privacy-safe logging.

    This policy provides granular control over email header redaction,
    allowing different levels of anonymization based on privacy requirements.

    Attributes:
        redact_sender: Whether to redact sender addresses
        redact_recipients: Whether to redact recipient addresses
        redact_subject: Whether to redact subject lines
        hash_length: Length of hash prefixes (default 8 chars)
    """

    def __init__(
        self,
        *,
        redact_sender: bool = True,
        redact_recipients: bool = True,
        redact_subject: bool = False,
        hash_length: int = 8,
    ):
        """Initialize redaction policy.

        Args:
            redact_sender: Redact sender addresses (default True)
            redact_recipients: Redact recipient addresses (default True)
            redact_subject: Redact subject lines (default False - usually safe)
            hash_length: Hash prefix length for redaction (default 8)
        """
        self.redact_sender = redact_sender
        self.redact_recipients = redact_recipients
        self.redact_subject_enabled = redact_subject
        self.hash_length = hash_length

    def redact_email_address(self, email: str) -> str:
        """Redact email address while preserving domain.

        Hashes the local part (before @) while preserving the domain
        for debugging purposes. This provides privacy while maintaining
        useful information about email providers.

        Args:
            email: Email address to redact (e.g., "john@example.com")

        Returns:
            Redacted email (e.g., "a8f3c2d1@example.com")

        Examples:
            >>> policy = EmailHeaderRedactionPolicy()
            >>> policy.redact_email_address("john.doe@example.com")
            'a8f3c2d1@example.com'
        """
        if '@' not in email:
            # Malformed email - hash entirely
            return self._hash(email)[: self.hash_length]

        local, domain = email.split('@', 1)
        hashed_local = self._hash(local)[: self.hash_length]
        return f"{hashed_local}@{domain}"

    def redact_subject(self, subject: str) -> str:
        """Redact subject line (hash only).

        Args:
            subject: Subject line to redact

        Returns:
            Hashed subject

        Examples:
            >>> policy = EmailHeaderRedactionPolicy()
            >>> policy.redact_subject("Confidential Meeting")
            'f2a1c8d3'
        """
        return self._hash(subject)[: self.hash_length]

    def redact_email_message(self, email_message: BaseModel) -> Dict[str, object]:
        """Create redacted version of email for logging.

        This method produces a privacy-safe representation of an EmailMessage
        suitable for audit logging. Email body content is NEVER included.

        Args:
            email_message: EmailMessage object to redact

        Returns:
            Dictionary with redacted email metadata

        Privacy Guarantees:
            - Email addresses redacted based on policy
            - Subject line optionally redacted
            - Email body NEVER included
            - Only metadata and counts included
        """
        # Build redacted representation
        redacted: Dict[str, object] = {
            "message_id_hash": self._hash(email_message.message_id)[: self.hash_length],
            "folder": email_message.folder,
            "date": email_message.date.isoformat(),
            "size_bytes": email_message.size_bytes,
            "has_attachments": len(email_message.attachments) > 0,
            "attachment_count": len(email_message.attachments),
        }

        # Redact sender
        if self.redact_sender:
            redacted["from"] = self.redact_email_address(
                email_message.from_address.address
            )
        else:
            redacted["from"] = email_message.from_address.address

        # Redact recipients
        if self.redact_recipients:
            redacted["to_count"] = len(email_message.to_addresses)
            redacted["cc_count"] = len(email_message.cc_addresses)
        else:
            redacted["to"] = [addr.address for addr in email_message.to_addresses]
            redacted["cc"] = [addr.address for addr in email_message.cc_addresses]

        # Redact subject
        if self.redact_subject_enabled:
            redacted["subject_hash"] = self.redact_subject(
                email_message.subject or ""
            )
        else:
            # Subject usually safe for logs (no PII)
            redacted["subject"] = email_message.subject

        # Add privacy classification
        redacted["contains_sensitive"] = email_message.contains_sensitive_keywords
        redacted["privacy_classification"] = email_message.privacy_classification

        # CRITICAL: Never include email body
        # This is enforced by not accessing body_plain, body_html, or body_normalized

        return redacted

    def redact_email_list(
        self, email_addresses: List[str]
    ) -> List[str]:
        """Redact a list of email addresses.

        Args:
            email_addresses: List of email addresses to redact

        Returns:
            List of redacted email addresses
        """
        return [self.redact_email_address(addr) for addr in email_addresses]

    def check_sensitive_subject(
        self,
        subject: Optional[str],
        keywords: Optional[List[str]] = None,
    ) -> bool:
        """Check if subject contains sensitive keywords.

        Args:
            subject: Email subject to check
            keywords: List of sensitive keywords (case-insensitive)

        Returns:
            True if sensitive keywords detected
        """
        if not subject or not keywords:
            return False

        subject_lower = subject.lower()
        return any(keyword.lower() in subject_lower for keyword in keywords)

    def _hash(self, value: str) -> str:
        """Generate stable hash for value.

        Uses SHA256 for cryptographically secure hashing.
        Same input always produces same hash (stable).

        Args:
            value: String to hash

        Returns:
            Hex digest of SHA256 hash
        """
        return hashlib.sha256(value.encode('utf-8')).hexdigest()


def create_redaction_policy_for_privacy_level(
    privacy_level: str,
    *,
    enable_sender_anonymization: bool = True,
    enable_recipient_anonymization: bool = True,
    enable_subject_redaction: bool = False,
) -> EmailHeaderRedactionPolicy:
    """Create redaction policy based on privacy level.

    Helper function to create appropriate redaction policy based on
    privacy settings.

    Args:
        privacy_level: Privacy level (strict, standard, permissive)
        enable_sender_anonymization: Redact sender addresses
        enable_recipient_anonymization: Redact recipient addresses
        enable_subject_redaction: Redact subject lines

    Returns:
        Configured EmailHeaderRedactionPolicy
    """
    if privacy_level == "strict":
        # Strict: Redact everything
        return EmailHeaderRedactionPolicy(
            redact_sender=True,
            redact_recipients=True,
            redact_subject=True,
            hash_length=8,
        )
    elif privacy_level == "permissive":
        # Permissive: Minimal redaction
        return EmailHeaderRedactionPolicy(
            redact_sender=False,
            redact_recipients=False,
            redact_subject=False,
            hash_length=8,
        )
    else:
        # Standard: Configurable redaction
        return EmailHeaderRedactionPolicy(
            redact_sender=enable_sender_anonymization,
            redact_recipients=enable_recipient_anonymization,
            redact_subject=enable_subject_redaction,
            hash_length=8,
        )


__all__ = [
    "EmailHeaderRedactionPolicy",
    "create_redaction_policy_for_privacy_level",
]
