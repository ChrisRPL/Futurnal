"""IMAP-specific consent management with interactive flows.

This module provides consent orchestration for IMAP mailbox operations,
implementing the consent requirements from the privacy audit integration plan.
It wraps the generic ConsentRegistry with IMAP-specific consent scopes and
provides interactive consent request flows.

Privacy-First Design:
- Explicit consent required for all mailbox access
- Granular consent scopes for different operations
- Audit logging of all consent decisions
- Integration with existing ConsentRegistry and AuditLogger
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from ...privacy.audit import AuditLogger
from ...privacy.consent import ConsentRegistry, ConsentRequiredError


class ImapConsentScopes(str, Enum):
    """Consent scopes for IMAP operations.

    These scopes define granular permissions for different IMAP operations,
    allowing users to control exactly what the system can do with their email data.
    """

    # Base access
    MAILBOX_ACCESS = "imap:mailbox:access"

    # Content analysis
    EMAIL_CONTENT_ANALYSIS = "imap:email:content_analysis"
    EMAIL_METADATA_EXTRACTION = "imap:email:metadata_extraction"

    # Feature-specific
    ATTACHMENT_EXTRACTION = "imap:email:attachment_extraction"
    THREAD_RECONSTRUCTION = "imap:email:thread_reconstruction"
    PARTICIPANT_ANALYSIS = "imap:email:participant_analysis"

    # External processing
    CLOUD_MODELS = "imap:email:cloud_models"

    @classmethod
    def get_default_scopes(cls) -> list[str]:
        """Get default consent scopes for standard mailbox processing.

        Returns:
            List of default scope values for standard email processing
        """
        return [
            cls.MAILBOX_ACCESS.value,
            cls.EMAIL_METADATA_EXTRACTION.value,
            cls.EMAIL_CONTENT_ANALYSIS.value,
            cls.THREAD_RECONSTRUCTION.value,
        ]

    @classmethod
    def get_minimal_scopes(cls) -> list[str]:
        """Get minimal consent scopes for privacy-strict mode.

        Returns:
            List of minimal scope values (metadata only)
        """
        return [
            cls.MAILBOX_ACCESS.value,
            cls.EMAIL_METADATA_EXTRACTION.value,
        ]


class ImapConsentManager:
    """Manage consent for IMAP mailbox operations.

    This class orchestrates consent flows for IMAP operations, providing:
    - Interactive consent request flows
    - Consent checking and enforcement
    - Audit logging of consent decisions
    - Integration with ConsentRegistry

    Attributes:
        consent_registry: Backend consent storage
        audit_logger: Optional audit logger for consent events
    """

    def __init__(
        self,
        consent_registry: ConsentRegistry,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """Initialize consent manager.

        Args:
            consent_registry: Backend consent registry for storage
            audit_logger: Optional audit logger for consent events
        """
        self.consent_registry = consent_registry
        self.audit_logger = audit_logger

    def request_mailbox_consent(
        self,
        *,
        mailbox_id: str,
        email_address: str,
        required_scopes: list[str],
        operator: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Request consent for mailbox operations (interactive).

        This method presents an interactive consent prompt to the user,
        explaining what permissions are being requested and recording
        their decisions.

        Args:
            mailbox_id: Unique mailbox identifier
            email_address: Email address being accessed (for display)
            required_scopes: List of scope values being requested
            operator: Optional operator name for audit trail

        Returns:
            Dictionary mapping scope values to granted status
        """
        consent_results: Dict[str, bool] = {}

        print(f"\n{'='*70}")
        print(f"MAILBOX ACCESS CONSENT REQUEST")
        print(f"{'='*70}")
        print(f"Email Address: {email_address}")
        print(f"Mailbox ID: {mailbox_id}")
        print(f"\nThe Futurnal AI will learn from your email communications to understand:")
        print("  • Communication patterns and relationship dynamics")
        print("  • Conversational context and topic evolution")
        print("  • Your professional and personal interaction patterns")
        print(f"{'='*70}\n")

        for scope in required_scopes:
            description = self._get_scope_description(scope)
            print(f"\n{description}")

            # Interactive prompt
            while True:
                response = input(f"Grant consent for '{scope}'? (yes/no): ").strip().lower()
                if response in ['yes', 'y', 'no', 'n']:
                    break
                print("Please enter 'yes' or 'no'")

            granted = response in ['yes', 'y']
            consent_results[scope] = granted

            # Record in registry
            if granted:
                self.consent_registry.grant(
                    source=f"mailbox:{mailbox_id}",
                    scope=scope,
                    operator=operator,
                )
            else:
                self.consent_registry.revoke(
                    source=f"mailbox:{mailbox_id}",
                    scope=scope,
                    operator=operator,
                )

            # Log consent decision
            self._log_consent_decision(
                mailbox_id=mailbox_id,
                scope=scope,
                granted=granted,
                operator=operator,
            )

        print(f"\n{'='*70}")
        granted_count = sum(1 for v in consent_results.values() if v)
        print(f"Consent granted for {granted_count}/{len(required_scopes)} scopes")
        print(f"{'='*70}\n")

        return consent_results

    def check_consent(
        self,
        *,
        mailbox_id: str,
        scope: str,
    ) -> bool:
        """Check if consent is granted for operation.

        Args:
            mailbox_id: Mailbox identifier
            scope: Consent scope to check

        Returns:
            True if consent is granted and active
        """
        record = self.consent_registry.get(
            source=f"mailbox:{mailbox_id}",
            scope=scope,
        )
        return record is not None and record.is_active()

    def require_consent(
        self,
        *,
        mailbox_id: str,
        scope: str,
    ) -> None:
        """Require consent or raise error.

        This method enforces consent requirements, raising an exception
        if consent has not been granted for the specified scope.

        Args:
            mailbox_id: Mailbox identifier
            scope: Consent scope required

        Raises:
            ConsentRequiredError: If consent not granted
        """
        if not self.check_consent(mailbox_id=mailbox_id, scope=scope):
            raise ConsentRequiredError(
                f"Consent required for scope '{scope}' on mailbox {mailbox_id}"
            )

    def grant_consent(
        self,
        *,
        mailbox_id: str,
        scope: str,
        operator: Optional[str] = None,
        duration_hours: Optional[int] = None,
    ) -> None:
        """Grant consent programmatically (non-interactive).

        Args:
            mailbox_id: Mailbox identifier
            scope: Consent scope to grant
            operator: Optional operator name for audit trail
            duration_hours: Optional expiration duration
        """
        self.consent_registry.grant(
            source=f"mailbox:{mailbox_id}",
            scope=scope,
            operator=operator,
            duration_hours=duration_hours,
        )

        self._log_consent_decision(
            mailbox_id=mailbox_id,
            scope=scope,
            granted=True,
            operator=operator,
        )

    def revoke_consent(
        self,
        *,
        mailbox_id: str,
        scope: str,
        operator: Optional[str] = None,
    ) -> None:
        """Revoke consent programmatically.

        Args:
            mailbox_id: Mailbox identifier
            scope: Consent scope to revoke
            operator: Optional operator name for audit trail
        """
        self.consent_registry.revoke(
            source=f"mailbox:{mailbox_id}",
            scope=scope,
            operator=operator,
        )

        self._log_consent_decision(
            mailbox_id=mailbox_id,
            scope=scope,
            granted=False,
            operator=operator,
        )

    def _get_scope_description(self, scope: str) -> str:
        """Get human-readable description of consent scope.

        Args:
            scope: Consent scope value

        Returns:
            Human-readable description
        """
        descriptions = {
            ImapConsentScopes.MAILBOX_ACCESS.value: (
                "📬 MAILBOX ACCESS\n"
                "   Access your mailbox to read emails"
            ),
            ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value: (
                "🔍 EMAIL CONTENT ANALYSIS\n"
                "   Analyze email body content for AI learning and pattern detection"
            ),
            ImapConsentScopes.EMAIL_METADATA_EXTRACTION.value: (
                "📋 EMAIL METADATA EXTRACTION\n"
                "   Extract email headers (from, to, subject, date) for context understanding"
            ),
            ImapConsentScopes.ATTACHMENT_EXTRACTION.value: (
                "📎 ATTACHMENT EXTRACTION\n"
                "   Extract and analyze email attachments"
            ),
            ImapConsentScopes.THREAD_RECONSTRUCTION.value: (
                "🧵 THREAD RECONSTRUCTION\n"
                "   Reconstruct conversation threads for context understanding"
            ),
            ImapConsentScopes.PARTICIPANT_ANALYSIS.value: (
                "👥 PARTICIPANT ANALYSIS\n"
                "   Analyze communication patterns with specific people"
            ),
            ImapConsentScopes.CLOUD_MODELS.value: (
                "☁️  CLOUD AI MODELS\n"
                "   Use cloud AI models for enhanced processing (data leaves device)"
            ),
        }
        return descriptions.get(scope, f"UNKNOWN SCOPE: {scope}")

    def _log_consent_decision(
        self,
        *,
        mailbox_id: str,
        scope: str,
        granted: bool,
        operator: Optional[str],
    ) -> None:
        """Log consent decision to audit logger.

        Args:
            mailbox_id: Mailbox identifier
            scope: Consent scope
            granted: Whether consent was granted
            operator: Optional operator name
        """
        if not self.audit_logger:
            return

        # Generate consent token hash for audit trail
        timestamp = datetime.utcnow().isoformat()
        token = f"{mailbox_id}:{scope}:{timestamp}"
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Use existing audit logger consent event method
        self.audit_logger.record_consent_event(
            job_id=f"imap_consent_{mailbox_id}_{int(datetime.utcnow().timestamp())}",
            source=f"mailbox:{mailbox_id}",
            scope=scope,
            granted=granted,
            operator=operator,
            token_hash=token_hash,
        )


__all__ = [
    "ImapConsentScopes",
    "ImapConsentManager",
]
