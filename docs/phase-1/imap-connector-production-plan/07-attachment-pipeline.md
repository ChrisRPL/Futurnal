Summary: Extract and process email attachments with size filtering and Unstructured.io routing.

# 07 · Attachment Pipeline

## Purpose
Extract email attachments from MIME parts, filter by size and type, and route supported formats to Unstructured.io for content extraction. Ensures the Ghost learns from attachment content (PDFs, documents, images) while respecting privacy and resource constraints.

## Scope
- Attachment extraction from MIME multipart messages
- Size and type filtering
- Supported format detection
- Route to Unstructured.io for text extraction
- Link attachments to parent email in PKG
- Attachment metadata storage
- Privacy-aware handling (no inline PII)
- Quarantine for unsupported/failed attachments

## Requirements Alignment
- **Content understanding**: Extract text from attachments for Ghost learning
- **Resource constraints**: Limit processing to reasonable sizes (<50MB default)
- **Privacy-first**: No attachment content in logs
- **PKG integration**: Link attachments to emails in graph
- **Unstructured.io integration**: Reuse existing asset pipeline patterns

## Data Model

```python
class EmailAttachment(BaseModel):
    """Email attachment with metadata and content reference."""

    # Identity
    attachment_id: str  # Generated UUID
    message_id: str  # Parent email Message-ID
    part_id: str  # MIME part identifier

    # Metadata
    filename: str
    content_type: str
    size_bytes: int
    is_inline: bool = False
    content_id: Optional[str] = None  # For inline images

    # Content reference
    content_hash: str  # SHA256 of content
    storage_path: Optional[Path] = None  # Local storage path

    # Processing status
    processing_status: AttachmentProcessingStatus = AttachmentProcessingStatus.PENDING
    processed_at: Optional[datetime] = None
    extraction_elements: int = 0

    # Privacy
    contains_sensitive: bool = False

    # Provenance
    extracted_at: datetime
    mailbox_id: str


class AttachmentProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Too large, unsupported type
    QUARANTINED = "quarantined"
```

## Component Design

### AttachmentExtractor
```python
class AttachmentExtractor:
    """Extract attachments from email MIME parts."""

    SUPPORTED_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.txt', '.rtf',
        '.xls', '.xlsx', '.ppt', '.pptx',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',
        '.html', '.htm', '.md', '.csv',
    }

    def __init__(
        self,
        *,
        max_size_bytes: int = 50 * 1024 * 1024,  # 50MB
        storage_dir: Path,
        supported_extensions: Optional[Set[str]] = None,
    ):
        self.max_size_bytes = max_size_bytes
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.supported_extensions = supported_extensions or self.SUPPORTED_EXTENSIONS

    def extract_attachments(
        self,
        raw_message: bytes,
        message_id: str,
        mailbox_id: str,
    ) -> List[EmailAttachment]:
        """Extract all attachments from email message."""
        from email import message_from_bytes
        from email.policy import default

        msg = message_from_bytes(raw_message, policy=default)
        attachments = []

        if not msg.is_multipart():
            return attachments

        for part in msg.walk():
            if part.get_content_disposition() in ['attachment', 'inline']:
                attachment = self._extract_part(part, message_id, mailbox_id)
                if attachment:
                    attachments.append(attachment)

        return attachments

    def _extract_part(
        self,
        part: Any,
        message_id: str,
        mailbox_id: str,
    ) -> Optional[EmailAttachment]:
        """Extract single attachment part."""
        filename = part.get_filename()
        if not filename:
            return None

        # Get content
        content = part.get_payload(decode=True)
        if not content:
            return None

        size_bytes = len(content)

        # Check size limit
        if size_bytes > self.max_size_bytes:
            logger.info(
                f"Skipping large attachment: {filename} ({size_bytes} bytes)",
                extra={"message_id": message_id}
            )
            return self._create_skipped_attachment(
                filename, part, message_id, mailbox_id, size_bytes, "too_large"
            )

        # Check supported format
        extension = Path(filename).suffix.lower()
        if extension not in self.supported_extensions:
            logger.debug(
                f"Skipping unsupported attachment: {filename}",
                extra={"message_id": message_id, "extension": extension}
            )
            return self._create_skipped_attachment(
                filename, part, message_id, mailbox_id, size_bytes, "unsupported"
            )

        # Calculate content hash
        content_hash = hashlib.sha256(content).hexdigest()

        # Store attachment
        storage_path = self._store_attachment(content, content_hash, filename)

        # Create attachment record
        attachment = EmailAttachment(
            attachment_id=str(uuid.uuid4()),
            message_id=message_id,
            part_id=part.get('Content-ID', '').strip('<>') or str(uuid.uuid4()),
            filename=filename,
            content_type=part.get_content_type(),
            size_bytes=size_bytes,
            is_inline=part.get_content_disposition() == 'inline',
            content_id=part.get('Content-ID', '').strip('<>'),
            content_hash=content_hash,
            storage_path=storage_path,
            extracted_at=datetime.utcnow(),
            mailbox_id=mailbox_id,
        )

        return attachment

    def _store_attachment(
        self,
        content: bytes,
        content_hash: str,
        filename: str,
    ) -> Path:
        """Store attachment content to disk."""
        # Use content hash as filename to deduplicate
        extension = Path(filename).suffix.lower()
        storage_filename = f"{content_hash}{extension}"
        storage_path = self.storage_dir / storage_filename

        if not storage_path.exists():
            storage_path.write_bytes(content)

        return storage_path

    def _create_skipped_attachment(
        self,
        filename: str,
        part: Any,
        message_id: str,
        mailbox_id: str,
        size_bytes: int,
        skip_reason: str,
    ) -> EmailAttachment:
        """Create attachment record for skipped attachment."""
        return EmailAttachment(
            attachment_id=str(uuid.uuid4()),
            message_id=message_id,
            part_id=part.get('Content-ID', '').strip('<>') or str(uuid.uuid4()),
            filename=filename,
            content_type=part.get_content_type(),
            size_bytes=size_bytes,
            is_inline=False,
            content_hash="",
            processing_status=AttachmentProcessingStatus.SKIPPED,
            extracted_at=datetime.utcnow(),
            mailbox_id=mailbox_id,
        )
```

### AttachmentProcessor
```python
class AttachmentProcessor:
    """Process attachments through Unstructured.io pipeline."""

    def __init__(
        self,
        *,
        ocr_languages: str = "eng",
        processing_timeout: int = 60,
    ):
        self.ocr_languages = ocr_languages
        self.processing_timeout = processing_timeout

    async def process_attachment(
        self,
        attachment: EmailAttachment,
    ) -> List[Dict]:
        """Process attachment through Unstructured.io."""
        if attachment.processing_status != AttachmentProcessingStatus.PENDING:
            return []

        if not attachment.storage_path or not attachment.storage_path.exists():
            logger.error(f"Attachment file not found: {attachment.filename}")
            attachment.processing_status = AttachmentProcessingStatus.FAILED
            return []

        try:
            attachment.processing_status = AttachmentProcessingStatus.PROCESSING

            # Process with Unstructured.io
            elements = await self._process_with_unstructured(attachment)

            attachment.processing_status = AttachmentProcessingStatus.COMPLETED
            attachment.processed_at = datetime.utcnow()
            attachment.extraction_elements = len(elements)

            return elements

        except Exception as e:
            logger.error(f"Attachment processing failed: {e}", extra={
                "attachment_id": attachment.attachment_id,
                "filename": attachment.filename,
            })
            attachment.processing_status = AttachmentProcessingStatus.FAILED
            return []

    async def _process_with_unstructured(
        self,
        attachment: EmailAttachment,
    ) -> List[Dict]:
        """Process file with Unstructured.io."""
        from unstructured.partition.auto import partition

        # Partition with timeout
        elements = await asyncio.wait_for(
            asyncio.to_thread(
                partition,
                filename=str(attachment.storage_path),
                languages=[self.ocr_languages],
            ),
            timeout=self.processing_timeout,
        )

        # Convert to dicts and add metadata
        element_dicts = []
        for element in elements:
            element_dict = element.to_dict()
            element_dict['metadata']['source_attachment_id'] = attachment.attachment_id
            element_dict['metadata']['source_message_id'] = attachment.message_id
            element_dict['metadata']['attachment_filename'] = attachment.filename
            element_dicts.append(element_dict)

        return element_dicts
```

### Attachment-Email Linking (PKG)
```python
def extract_attachment_triples(
    attachment: EmailAttachment,
    email_message_id: str,
) -> List[SemanticTriple]:
    """Extract semantic triples linking attachment to email."""
    triples = []

    attachment_uri = f"attachment:{attachment.attachment_id}"
    email_uri = f"email:{email_message_id}"

    # Attachment type
    triples.append(SemanticTriple(
        subject=attachment_uri,
        predicate="rdf:type",
        object="futurnal:EmailAttachment",
        extraction_method="metadata",
    ))

    # Link to email
    triples.append(SemanticTriple(
        subject=attachment_uri,
        predicate="attachment:partOfEmail",
        object=email_uri,
        extraction_method="metadata",
    ))

    # Filename
    triples.append(SemanticTriple(
        subject=attachment_uri,
        predicate="attachment:filename",
        object=attachment.filename,
        extraction_method="metadata",
    ))

    # Content type
    triples.append(SemanticTriple(
        subject=attachment_uri,
        predicate="attachment:contentType",
        object=attachment.content_type,
        extraction_method="metadata",
    ))

    # Size
    triples.append(SemanticTriple(
        subject=attachment_uri,
        predicate="attachment:sizeBytes",
        object=str(attachment.size_bytes),
        extraction_method="metadata",
    ))

    return triples
```

## Acceptance Criteria

- ✅ Attachments extracted from multipart MIME messages
- ✅ Size filtering enforced (skip >50MB by default)
- ✅ Format filtering enforced (only supported extensions)
- ✅ Attachment content stored with deduplication (hash-based)
- ✅ Unstructured.io processing successful for supported formats
- ✅ Inline images detected and marked
- ✅ Semantic triples link attachments to emails in PKG
- ✅ Processing timeout enforced (60s default)
- ✅ Failed attachments quarantined with error details
- ✅ No attachment content in logs or audit events

## Test Plan

### Unit Tests
- MIME attachment extraction
- Size limit enforcement
- Format detection and filtering
- Content hash calculation
- Deduplication logic

### Integration Tests
- End-to-end attachment processing
- Unstructured.io extraction for PDFs
- Unstructured.io extraction for images (OCR)
- Unstructured.io extraction for Office documents
- Semantic triple generation
- PKG storage of attachment relationships

### Edge Case Tests
- Attachments without filename
- Inline images (Content-ID)
- Very large attachments (>100MB)
- Corrupted files
- Password-protected files
- Empty attachments

## Implementation Notes

### Content Deduplication
```python
# Use SHA256 hash as storage filename
# Multiple emails with same attachment → single file
# Saves storage and processing time
```

### OCR Language Configuration
```python
# Configure OCR languages per mailbox
# Example: "eng+fra" for English and French
# Passed to Unstructured.io partition()
```

## Open Questions

- Should we extract attachments from nested emails (message/rfc822)?
- How to handle attachments in encrypted emails?
- Should we support cloud storage for attachments (S3, etc.)?
- How to handle very large attachment backlogs (thousands)?
- Should we implement attachment virus scanning?

## Dependencies
- Unstructured.io for content extraction
- EmailParser from task 04
- SemanticTripleExtractor from pipeline
- Quarantine system for failed processing


