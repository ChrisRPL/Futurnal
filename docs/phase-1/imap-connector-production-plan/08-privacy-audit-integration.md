Summary: Implement privacy safeguards and audit logging for IMAP email ingestion without exposing email content.

# 08 · Privacy & Audit Integration

## Purpose
Integrate comprehensive privacy safeguards and audit logging for IMAP email ingestion, ensuring explicit consent, participant anonymization, and audit trails without exposing email content or personally identifiable information (PII).

## Scope
- Consent flow for mailbox access and AI learning
- Email header anonymization in logs (sender, recipient, subject)
- No email body content in logs or audit events
- Participant email redaction policies
- GDPR-compliant audit trails
- Privacy classification based on keywords/patterns
- Audit event generation for all IMAP operations
- Integration with existing ConsentRegistry and AuditLogger

## Requirements Alignment
- **Explicit consent**: All mailbox access requires explicit AI learning consent
- **Privacy by default**: No PII in logs unless explicitly approved
- **GDPR compliance**: Audit trails support data access/deletion requests
- **Content protection**: Email bodies never logged or exposed
- **Participant privacy**: Email addresses anonymized in telemetry

## Consent Scopes

### IMAP-Specific Consent Requirements
```python
class ImapConsentScopes:
    """Consent scopes for IMAP operations."""

    # Base access
    MAILBOX_ACCESS = "imap:mailbox:access"  # Required for any access

    # Content analysis
    EMAIL_CONTENT_ANALYSIS = "imap:email:content_analysis"  # AI learning from bodies
    EMAIL_METADATA_EXTRACTION = "imap:email:metadata_extraction"  # Headers only

    # Feature-specific
    ATTACHMENT_EXTRACTION = "imap:email:attachment_extraction"
    THREAD_RECONSTRUCTION = "imap:email:thread_reconstruction"
    PARTICIPANT_ANALYSIS = "imap:email:participant_analysis"

    # External processing
    CLOUD_MODELS = "imap:email:cloud_models"  # Cloud AI processing

    @classmethod
    def get_default_scopes(cls) -> List[str]:
        """Get default consent scopes for standard mailbox processing."""
        return [
            cls.MAILBOX_ACCESS,
            cls.EMAIL_METADATA_EXTRACTION,
            cls.EMAIL_CONTENT_ANALYSIS,
            cls.THREAD_RECONSTRUCTION,
        ]

    @classmethod
    def get_minimal_scopes(cls) -> List[str]:
        """Get minimal consent scopes for privacy-strict mode."""
        return [
            cls.MAILBOX_ACCESS,
            cls.EMAIL_METADATA_EXTRACTION,
        ]
```

### Consent Flow Integration
```python
class ImapConsentManager:
    """Manage consent for IMAP mailbox operations."""

    def __init__(
        self,
        consent_registry: ConsentRegistry,
        audit_logger: AuditLogger,
    ):
        self.consent_registry = consent_registry
        self.audit_logger = audit_logger

    async def request_mailbox_consent(
        self,
        mailbox_descriptor: ImapMailboxDescriptor,
        operator: str,
    ) -> Dict[str, bool]:
        """Request consent for mailbox operations (interactive)."""
        required_scopes = mailbox_descriptor.privacy_settings.required_consent_scopes
        consent_results = {}

        print(f"\nMailbox Access Consent Request: {mailbox_descriptor.email_address}")
        print(f"The Ghost AI will learn from your email communications to understand:")
        print("  - Communication patterns and relationship dynamics")
        print("  - Conversational context and topic evolution")
        print("  - Your professional and personal interaction patterns\n")

        for scope in required_scopes:
            description = self._get_scope_description(scope)
            print(f"\n{description}")
            response = input(f"Grant consent for '{scope.value}'? (yes/no): ").strip().lower()

            granted = response in ['yes', 'y']
            consent_results[scope.value] = granted

            if granted:
                self.consent_registry.grant(
                    source=f"mailbox:{mailbox_descriptor.id}",
                    scope=scope.value,
                    operator=operator,
                )
            else:
                self.consent_registry.revoke(
                    source=f"mailbox:{mailbox_descriptor.id}",
                    scope=scope.value,
                    operator=operator,
                )

        # Log consent decision
        self._log_consent_decision(mailbox_descriptor, consent_results, operator)

        return consent_results

    def check_consent(
        self,
        mailbox_id: str,
        scope: str,
    ) -> bool:
        """Check if consent is granted for operation."""
        record = self.consent_registry.get(
            source=f"mailbox:{mailbox_id}",
            scope=scope,
        )
        return record is not None and record.is_active()

    def require_consent(
        self,
        mailbox_id: str,
        scope: str,
    ) -> None:
        """Require consent or raise error."""
        if not self.check_consent(mailbox_id, scope):
            raise ConsentRequiredError(
                f"Consent required for scope '{scope}' on mailbox {mailbox_id}"
            )

    def _get_scope_description(self, scope: ConsentScope) -> str:
        """Get human-readable description of consent scope."""
        descriptions = {
            ImapConsentScopes.MAILBOX_ACCESS: "Access your mailbox to read emails",
            ImapConsentScopes.EMAIL_CONTENT_ANALYSIS: "Analyze email body content for AI learning",
            ImapConsentScopes.EMAIL_METADATA_EXTRACTION: "Extract email headers (from, to, subject, date)",
            ImapConsentScopes.ATTACHMENT_EXTRACTION: "Extract and analyze email attachments",
            ImapConsentScopes.THREAD_RECONSTRUCTION: "Reconstruct conversation threads for context understanding",
            ImapConsentScopes.PARTICIPANT_ANALYSIS: "Analyze communication patterns with specific people",
            ImapConsentScopes.CLOUD_MODELS: "Use cloud AI models for enhanced processing (data leaves device)",
        }
        return descriptions.get(scope.value, scope.value)
```

## Privacy Redaction

### Email Header Redaction
```python
class EmailHeaderRedactionPolicy:
    """Redact email headers for privacy-safe logging."""

    def __init__(
        self,
        redact_sender: bool = True,
        redact_recipients: bool = True,
        redact_subject: bool = False,  # Usually safe for logs
        hash_length: int = 8,
    ):
        self.redact_sender = redact_sender
        self.redact_recipients = redact_recipients
        self.redact_subject = redact_subject
        self.hash_length = hash_length

    def redact_email_address(self, email: str) -> str:
        """Redact email address while preserving domain."""
        if '@' not in email:
            return self._hash(email)

        local, domain = email.split('@', 1)
        hashed_local = self._hash(local)[:self.hash_length]
        return f"{hashed_local}@{domain}"

    def redact_subject(self, subject: str) -> str:
        """Redact subject line (hash only)."""
        return self._hash(subject)[:self.hash_length]

    def _hash(self, value: str) -> str:
        """Generate stable hash for value."""
        return hashlib.sha256(value.encode()).hexdigest()

    def redact_email_message(self, email_message: EmailMessage) -> Dict[str, str]:
        """Create redacted version of email for logging."""
        redacted = {
            "message_id_hash": self._hash(email_message.message_id)[:self.hash_length],
            "folder": email_message.folder,
            "date": email_message.date.isoformat(),
            "size_bytes": email_message.size_bytes,
            "has_attachments": len(email_message.attachments) > 0,
        }

        if self.redact_sender:
            redacted["from"] = self.redact_email_address(email_message.from_address.address)
        else:
            redacted["from"] = email_message.from_address.address

        if self.redact_recipients:
            redacted["to_count"] = len(email_message.to_addresses)
            redacted["cc_count"] = len(email_message.cc_addresses)
        else:
            redacted["to"] = [addr.address for addr in email_message.to_addresses]
            redacted["cc"] = [addr.address for addr in email_message.cc_addresses]

        if self.redact_subject:
            redacted["subject_hash"] = self.redact_subject(email_message.subject or "")
        else:
            # Subject usually safe for logs (no PII)
            redacted["subject"] = email_message.subject

        return redacted
```

## Audit Event Structure

### IMAP-Specific Audit Events
```python
class ImapAuditEvents:
    """Audit event types for IMAP operations."""

    # Connection events
    CONNECTION_ESTABLISHED = "imap_connection_established"
    CONNECTION_FAILED = "imap_connection_failed"
    CONNECTION_CLOSED = "imap_connection_closed"

    # Sync events
    SYNC_STARTED = "imap_sync_started"
    SYNC_COMPLETED = "imap_sync_completed"
    SYNC_FAILED = "imap_sync_failed"

    # Email processing
    EMAIL_FETCHED = "imap_email_fetched"
    EMAIL_PARSED = "imap_email_parsed"
    EMAIL_PROCESSING_FAILED = "imap_email_processing_failed"

    # Attachment processing
    ATTACHMENT_EXTRACTED = "imap_attachment_extracted"
    ATTACHMENT_PROCESSED = "imap_attachment_processed"
    ATTACHMENT_SKIPPED = "imap_attachment_skipped"

    # Thread reconstruction
    THREAD_RECONSTRUCTED = "imap_thread_reconstructed"

    # Privacy events
    CONSENT_GRANTED = "imap_consent_granted"
    CONSENT_REVOKED = "imap_consent_revoked"
    CONSENT_CHECK_FAILED = "imap_consent_check_failed"

def log_email_sync_event(
    audit_logger: AuditLogger,
    mailbox_id: str,
    folder: str,
    sync_result: SyncResult,
    redaction_policy: EmailHeaderRedactionPolicy,
) -> None:
    """Log email sync event with privacy redaction."""

    audit_logger.record(AuditEvent(
        job_id=f"imap_sync_{mailbox_id}_{int(datetime.utcnow().timestamp())}",
        source="imap_sync_engine",
        action=ImapAuditEvents.SYNC_COMPLETED,
        status="success",
        timestamp=datetime.utcnow(),
        metadata={
            "mailbox_id": mailbox_id,
            "folder": folder,
            "new_messages": len(sync_result.new_messages),
            "updated_messages": len(sync_result.updated_messages),
            "deleted_messages": len(sync_result.deleted_messages),
            "sync_duration_seconds": sync_result.sync_duration_seconds,
            "errors": len(sync_result.errors),
        },
    ))

def log_email_processing_event(
    audit_logger: AuditLogger,
    email_message: EmailMessage,
    redaction_policy: EmailHeaderRedactionPolicy,
    status: str = "success",
) -> None:
    """Log email processing event with redaction."""

    redacted_email = redaction_policy.redact_email_message(email_message)

    audit_logger.record(AuditEvent(
        job_id=f"imap_email_{email_message.uid}",
        source="imap_email_processor",
        action=ImapAuditEvents.EMAIL_PARSED,
        status=status,
        timestamp=datetime.utcnow(),
        metadata=redacted_email,
    ))
```

## Acceptance Criteria

- ✅ Explicit consent required before mailbox access
- ✅ Consent checked before each operation requiring it
- ✅ Email addresses redacted in all logs (hash + domain)
- ✅ Email subject lines never logged unless privacy level permits
- ✅ Email bodies never logged under any circumstances
- ✅ Audit events generated for all IMAP operations
- ✅ Audit events contain no PII
- ✅ Consent decisions logged with timestamps
- ✅ GDPR-compliant audit retention (configurable, default 90 days)
- ✅ Privacy classification based on keywords enforced

## Test Plan

### Unit Tests
- Consent scope validation
- Email address redaction
- Subject line redaction
- Audit event structure validation
- Privacy policy enforcement

### Integration Tests
- End-to-end consent flow (interactive)
- Consent check enforcement
- Audit event generation during sync
- Redaction policy application
- ConsentRegistry integration

### Security Tests
- No PII in log output (grep for email addresses)
- No email bodies in log output
- Consent enforcement (reject operations without consent)
- Audit log integrity (tamper detection)

### Privacy Tests
- GDPR data export (retrieve all audit events for mailbox)
- GDPR data deletion (remove all records for mailbox)
- Redaction effectiveness (verify hashes stable)

## Implementation Notes

### Consent Token Pattern
```python
# Store consent decision hash for audit trail
token_hash = hashlib.sha256(f"{mailbox_id}:{scope}:{timestamp}".encode()).hexdigest()
```

### Audit Log Retention
```python
# Configurable per mailbox privacy settings
retain_audit_days = mailbox_descriptor.privacy_settings.retain_audit_days  # Default 90
```

## Open Questions

- Should we support consent expiration (renew every N days)?
- How to handle consent withdrawal mid-sync?
- Should we allow per-folder consent (INBOX yes, Sent no)?
- How to export audit logs for GDPR data access requests?
- Should we implement audit log encryption at rest?

## Dependencies
- ConsentRegistry from privacy module
- AuditLogger from privacy module
- ImapMailboxDescriptor from task 01


