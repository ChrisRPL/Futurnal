Summary: Implement RFC822/MIME email parsing and normalization for Unstructured.io processing.

# 04 · Email Parser & Normalizer

## Purpose
Parse RFC822/MIME email messages and normalize content for Unstructured.io processing, extracting headers, body text (plain/HTML), and metadata. Ensures the Ghost learns from email content while preserving conversational structure, participant relationships, and temporal context.

## Scope
- RFC822/MIME parsing via Python email.parser
- Header extraction (Message-ID, References, In-Reply-To, Date, From, To, Cc, Subject)
- Body text extraction (plain text preference, HTML fallback)
- Multipart MIME handling
- Character encoding detection and normalization
- Email metadata extraction for semantic triple generation
- Feed normalized content to Unstructured.io for element extraction
- Privacy-aware content handling (no PII in logs)

## Requirements Alignment
- **Content fidelity**: Preserve email structure for Ghost understanding
- **Privacy-first**: Extract metadata without logging email bodies
- **Unstructured.io integration**: Feed processed content to existing pipeline
- **Semantic triples**: Extract relationships for PKG construction
- **Thread awareness**: Preserve threading headers for conversation reconstruction

## Component Design

### EmailMessage Model
```python
class EmailMessage(BaseModel):
    """Parsed email message with metadata."""

    # Identity
    message_id: str  # Unique message identifier
    uid: int  # IMAP UID
    folder: str  # IMAP folder path

    # Headers
    subject: Optional[str] = None
    from_address: EmailAddress
    to_addresses: List[EmailAddress] = Field(default_factory=list)
    cc_addresses: List[EmailAddress] = Field(default_factory=list)
    bcc_addresses: List[EmailAddress] = Field(default_factory=list)
    reply_to_addresses: List[EmailAddress] = Field(default_factory=list)
    date: datetime

    # Threading headers
    in_reply_to: Optional[str] = None  # Message-ID of parent
    references: List[str] = Field(default_factory=list)  # Thread chain

    # Content
    body_plain: Optional[str] = None
    body_html: Optional[str] = None
    body_normalized: Optional[str] = None  # For Unstructured.io

    # Metadata
    size_bytes: int
    flags: List[str] = Field(default_factory=list)  # IMAP flags
    labels: List[str] = Field(default_factory=list)  # Gmail labels
    attachments: List[AttachmentMetadata] = Field(default_factory=list)

    # Privacy
    contains_sensitive_keywords: bool = False
    privacy_classification: str = "standard"

    # Provenance
    retrieved_at: datetime
    mailbox_id: str


class EmailAddress(BaseModel):
    """Parsed email address with display name."""
    address: str  # email@example.com
    display_name: Optional[str] = None  # "John Doe"

    @classmethod
    def from_header(cls, header_value: str) -> List['EmailAddress']:
        """Parse email addresses from header value."""
        # Use email.utils.getaddresses
        from email.utils import getaddresses
        addresses = getaddresses([header_value])
        return [
            cls(display_name=name or None, address=addr)
            for name, addr in addresses
            if addr
        ]


class AttachmentMetadata(BaseModel):
    """Metadata for email attachment."""
    filename: str
    content_type: str
    size_bytes: int
    part_id: str  # MIME part identifier
    is_inline: bool = False
    content_id: Optional[str] = None  # For inline images
```

### EmailParser
```python
class EmailParser:
    """Parse RFC822/MIME emails into structured format."""

    def __init__(
        self,
        *,
        privacy_policy: Optional[MailboxPrivacySettings] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.privacy_policy = privacy_policy
        self.audit_logger = audit_logger

    def parse_message(
        self,
        *,
        raw_message: bytes,
        uid: int,
        folder: str,
        mailbox_id: str,
    ) -> EmailMessage:
        """Parse raw RFC822 message into EmailMessage."""
        # Parse with Python email.parser
        from email import message_from_bytes
        from email.policy import default

        msg = message_from_bytes(raw_message, policy=default)

        # Extract headers
        message_id = self._extract_message_id(msg)
        subject = self._extract_subject(msg)
        from_addr = self._extract_from(msg)
        to_addrs = self._extract_to(msg)
        cc_addrs = self._extract_cc(msg)
        date = self._extract_date(msg)
        in_reply_to = self._extract_in_reply_to(msg)
        references = self._extract_references(msg)

        # Extract body
        body_plain, body_html = self._extract_body(msg)
        body_normalized = self._normalize_body(body_plain, body_html)

        # Extract attachments
        attachments = self._extract_attachments(msg)

        # Privacy classification
        contains_sensitive = self._check_sensitive_keywords(subject, body_normalized)

        email_message = EmailMessage(
            message_id=message_id,
            uid=uid,
            folder=folder,
            subject=subject,
            from_address=from_addr,
            to_addresses=to_addrs,
            cc_addresses=cc_addrs,
            date=date,
            in_reply_to=in_reply_to,
            references=references,
            body_plain=body_plain,
            body_html=body_html,
            body_normalized=body_normalized,
            size_bytes=len(raw_message),
            attachments=attachments,
            contains_sensitive_keywords=contains_sensitive,
            retrieved_at=datetime.utcnow(),
            mailbox_id=mailbox_id,
        )

        # Log parsing event (without content)
        self._log_parse_event(email_message)

        return email_message

    def _extract_message_id(self, msg: EmailMessage) -> str:
        """Extract Message-ID header."""
        message_id = msg.get('Message-ID', '').strip('<>')
        if not message_id:
            # Generate fallback ID
            message_id = f"generated-{uuid.uuid4()}@futurnal.local"
        return message_id

    def _extract_subject(self, msg: EmailMessage) -> Optional[str]:
        """Extract and decode subject."""
        from email.header import decode_header

        subject = msg.get('Subject', '')
        if not subject:
            return None

        # Decode encoded words
        decoded_parts = decode_header(subject)
        subject = ''
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                subject += part.decode(charset or 'utf-8', errors='replace')
            else:
                subject += part

        return subject.strip()

    def _extract_from(self, msg: EmailMessage) -> EmailAddress:
        """Extract From address."""
        from_header = msg.get('From', '')
        addresses = EmailAddress.from_header(from_header)
        return addresses[0] if addresses else EmailAddress(address='unknown@unknown')

    def _extract_to(self, msg: EmailMessage) -> List[EmailAddress]:
        """Extract To addresses."""
        to_header = msg.get('To', '')
        return EmailAddress.from_header(to_header)

    def _extract_cc(self, msg: EmailMessage) -> List[EmailAddress]:
        """Extract Cc addresses."""
        cc_header = msg.get('Cc', '')
        return EmailAddress.from_header(cc_header)

    def _extract_date(self, msg: EmailMessage) -> datetime:
        """Extract and parse Date header."""
        from email.utils import parsedate_to_datetime

        date_header = msg.get('Date')
        if date_header:
            try:
                return parsedate_to_datetime(date_header)
            except Exception:
                pass
        return datetime.utcnow()

    def _extract_in_reply_to(self, msg: EmailMessage) -> Optional[str]:
        """Extract In-Reply-To header."""
        in_reply_to = msg.get('In-Reply-To', '').strip('<>')
        return in_reply_to if in_reply_to else None

    def _extract_references(self, msg: EmailMessage) -> List[str]:
        """Extract References header (thread chain)."""
        references = msg.get('References', '')
        if not references:
            return []

        # Parse space-separated Message-IDs
        return [ref.strip('<>') for ref in references.split() if ref]

    def _extract_body(self, msg: EmailMessage) -> Tuple[Optional[str], Optional[str]]:
        """Extract plain text and HTML body parts."""
        body_plain = None
        body_html = None

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain' and not body_plain:
                    body_plain = part.get_content()
                elif content_type == 'text/html' and not body_html:
                    body_html = part.get_content()
        else:
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                body_plain = msg.get_content()
            elif content_type == 'text/html':
                body_html = msg.get_content()

        return body_plain, body_html

    def _normalize_body(self, plain: Optional[str], html: Optional[str]) -> str:
        """Normalize body for Unstructured.io processing."""
        if plain:
            return plain.strip()
        elif html:
            # Convert HTML to plain text (basic)
            from html import unescape
            import re
            text = re.sub('<[^<]+?>', '', html)  # Strip tags
            text = unescape(text)
            return text.strip()
        return ""

    def _extract_attachments(self, msg: EmailMessage) -> List[AttachmentMetadata]:
        """Extract attachment metadata."""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == 'attachment':
                    filename = part.get_filename()
                    if filename:
                        attachments.append(AttachmentMetadata(
                            filename=filename,
                            content_type=part.get_content_type(),
                            size_bytes=len(part.get_payload(decode=True) or b''),
                            part_id=part.get('Content-ID', '').strip('<>'),
                            is_inline=False,
                        ))

        return attachments

    def _check_sensitive_keywords(self, subject: Optional[str], body: str) -> bool:
        """Check for sensitive keywords in subject/body."""
        if not self.privacy_policy:
            return False

        keywords = self.privacy_policy.privacy_subject_keywords
        text = f"{subject or ''} {body}".lower()

        return any(keyword.lower() in text for keyword in keywords)

    def _log_parse_event(self, email_message: EmailMessage) -> None:
        """Log parsing event without email content."""
        if not self.audit_logger:
            return

        self.audit_logger.record(AuditEvent(
            job_id=f"email_parse_{email_message.uid}",
            source="imap_email_parser",
            action="email_parsed",
            status="success",
            timestamp=datetime.utcnow(),
            metadata={
                "message_id_hash": sha256(email_message.message_id.encode()).hexdigest()[:16],
                "folder": email_message.folder,
                "has_attachments": len(email_message.attachments) > 0,
                "attachment_count": len(email_message.attachments),
                "body_length": len(email_message.body_normalized or ""),
                "contains_sensitive": email_message.contains_sensitive_keywords,
            },
        ))
```

### EmailNormalizer (for Unstructured.io)
```python
class EmailNormalizer:
    """Normalize email content for Unstructured.io processing."""

    def normalize(self, email_message: EmailMessage) -> str:
        """Convert email to normalized text format for Unstructured.io."""
        lines = []

        # Add metadata header
        lines.append(f"From: {email_message.from_address.address}")
        if email_message.from_address.display_name:
            lines.append(f"From Name: {email_message.from_address.display_name}")

        lines.append(f"To: {', '.join(a.address for a in email_message.to_addresses)}")

        if email_message.cc_addresses:
            lines.append(f"Cc: {', '.join(a.address for a in email_message.cc_addresses)}")

        lines.append(f"Date: {email_message.date.isoformat()}")

        if email_message.subject:
            lines.append(f"Subject: {email_message.subject}")

        # Add thread context
        if email_message.in_reply_to:
            lines.append(f"In-Reply-To: {email_message.in_reply_to}")

        if email_message.references:
            lines.append(f"References: {', '.join(email_message.references)}")

        # Separator
        lines.append("")
        lines.append("---")
        lines.append("")

        # Add body
        if email_message.body_normalized:
            lines.append(email_message.body_normalized)

        # Add attachment info
        if email_message.attachments:
            lines.append("")
            lines.append("Attachments:")
            for att in email_message.attachments:
                lines.append(f"  - {att.filename} ({att.content_type}, {att.size_bytes} bytes)")

        return "\n".join(lines)
```

### Unstructured.io Integration
```python
async def process_email_with_unstructured(
    email_message: EmailMessage,
    normalizer: EmailNormalizer,
) -> List[Dict]:
    """Process email through Unstructured.io pipeline."""
    from unstructured.partition.text import partition_text

    # Normalize email
    normalized_text = normalizer.normalize(email_message)

    # Process with Unstructured.io
    elements = partition_text(text=normalized_text)

    # Convert to element dicts
    element_dicts = []
    for element in elements:
        element_dict = element.to_dict()
        element_dict['metadata']['source_message_id'] = email_message.message_id
        element_dict['metadata']['source_folder'] = email_message.folder
        element_dict['metadata']['source_date'] = email_message.date.isoformat()
        element_dicts.append(element_dict)

    return element_dicts
```

## Semantic Triple Extraction

### Email-Specific Triples
```python
def extract_email_triples(email_message: EmailMessage) -> List[SemanticTriple]:
    """Extract semantic triples from email metadata."""
    triples = []

    # Email identity
    email_uri = f"email:{email_message.message_id}"

    # Type triple
    triples.append(SemanticTriple(
        subject=email_uri,
        predicate="rdf:type",
        object="futurnal:Email",
        source_element_id=email_message.message_id,
        extraction_method="metadata",
    ))

    # Subject triple
    if email_message.subject:
        triples.append(SemanticTriple(
            subject=email_uri,
            predicate="email:subject",
            object=email_message.subject,
            extraction_method="metadata",
        ))

    # From triple
    from_person_uri = f"person:{email_message.from_address.address}"
    triples.append(SemanticTriple(
        subject=email_uri,
        predicate="email:from",
        object=from_person_uri,
        extraction_method="metadata",
    ))

    # To triples
    for to_addr in email_message.to_addresses:
        to_person_uri = f"person:{to_addr.address}"
        triples.append(SemanticTriple(
            subject=email_uri,
            predicate="email:to",
            object=to_person_uri,
            extraction_method="metadata",
        ))

    # Date triple
    triples.append(SemanticTriple(
        subject=email_uri,
        predicate="email:sentDate",
        object=email_message.date.isoformat(),
        extraction_method="metadata",
    ))

    # Thread triples (in-reply-to)
    if email_message.in_reply_to:
        parent_uri = f"email:{email_message.in_reply_to}"
        triples.append(SemanticTriple(
            subject=email_uri,
            predicate="email:inReplyTo",
            object=parent_uri,
            extraction_method="metadata",
        ))

    # Person metadata triples
    if email_message.from_address.display_name:
        triples.append(SemanticTriple(
            subject=from_person_uri,
            predicate="person:displayName",
            object=email_message.from_address.display_name,
            extraction_method="metadata",
        ))

    return triples
```

## Acceptance Criteria

- ✅ RFC822/MIME messages parsed correctly
- ✅ All threading headers extracted (Message-ID, References, In-Reply-To)
- ✅ Multi-part MIME messages handled (plain + HTML)
- ✅ Character encoding detected and normalized
- ✅ Email addresses parsed with display names
- ✅ Attachment metadata extracted (no content)
- ✅ Normalized content feeds to Unstructured.io successfully
- ✅ Semantic triples generated from email metadata
- ✅ Privacy keywords detected in subject/body
- ✅ No email body content logged in audit events
- ✅ HTML to plain text conversion preserves readability

## Test Plan

### Unit Tests
- Message-ID extraction and generation
- Subject decoding (encoded words)
- Email address parsing (with/without display names)
- Date parsing (various formats)
- Threading header extraction
- Body extraction (plain, HTML, multipart)
- Attachment metadata extraction
- Sensitive keyword detection

### Integration Tests
- End-to-end parsing of real email samples
- Unstructured.io element extraction
- Semantic triple generation
- Gmail message format
- Office 365 message format
- Generic IMAP message format

### Edge Case Tests
- Missing Message-ID (fallback generation)
- Malformed headers
- Non-ASCII characters in subject/body
- Empty body messages
- Messages with only HTML body
- Inline images and attachments
- Very large messages (>10MB)

## Implementation Notes

### Character Encoding
```python
# Python email.parser handles encoding automatically with policy=default
from email.policy import default
msg = message_from_bytes(raw_message, policy=default)
```

### HTML to Plain Text
Consider using `html2text` library for better HTML conversion:
```python
import html2text
h = html2text.HTML2Text()
h.ignore_links = False
plain_text = h.handle(html_body)
```

## Open Questions

- Should we extract inline images as separate elements?
- How to handle very large email bodies (>1MB)?
- Should we parse email signatures separately?
- How to detect and handle forwarded messages?
- Should we extract quoted text separately for thread reconstruction?
- How to handle encrypted emails (S/MIME, PGP)?

## Dependencies
- Python email module (standard library)
- Unstructured.io for element extraction
- html2text for HTML conversion (optional)
- SemanticTripleExtractor from pipeline
- AuditLogger for parsing event tracking


