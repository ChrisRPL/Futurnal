Summary: Implement pluggable adapter registry for format-specific normalization handlers.

# 03 · Format Adapter Registry

## Purpose
Design and implement a pluggable adapter registry system that enables format-specific normalization logic while maintaining a consistent interface. The registry allows modular addition of new format handlers following the modOpt pattern, leveraging existing normalizers (Obsidian, Email, GitHub) and adding new ones for PDF, HTML, code, and other formats.

## Scope
- Define `FormatAdapter` protocol for consistent interface
- Implement `FormatAdapterRegistry` for adapter management
- Integrate existing normalizers (Obsidian, Email, GitHub)
- Create new adapters for common formats (PDF, HTML, code, text)
- Support adapter discovery and registration
- Enable format-specific configuration

## Requirements Alignment
- **Feature Requirement**: "Format-specific adapters (markdown, PDF, HTML, email, code comment blocks)"
- **Implementation Guide**: "Adapter Library: Implement modular adapters per format; leverage modOpt modular patterns"
- **Extensibility**: Plug-and-play addition of new format types

## Component Design

### FormatAdapter Protocol

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Optional
from .schema import NormalizedDocument, DocumentFormat


class FormatAdapter(Protocol):
    """Protocol for format-specific normalization adapters.

    Each adapter handles format-specific parsing, cleaning, and
    metadata extraction before general-purpose processing.
    """

    # Adapter metadata
    name: str
    supported_formats: list[DocumentFormat]
    requires_unstructured_processing: bool

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize document using format-specific logic.

        Args:
            file_path: Path to source file
            source_id: Connector-specific identifier
            source_type: Source type (e.g., "local_files")
            source_metadata: Additional metadata from connector

        Returns:
            Preliminary NormalizedDocument (may require further processing)

        Raises:
            AdapterError: If format-specific normalization fails
        """
        ...

    async def validate(self, file_path: Path) -> bool:
        """Validate that file is processable by this adapter.

        Args:
            file_path: Path to validate

        Returns:
            True if adapter can process this file
        """
        ...
```

### FormatAdapterRegistry

```python
import logging
from typing import Dict, List, Optional
from .schema import DocumentFormat

logger = logging.getLogger(__name__)


class FormatAdapterRegistry:
    """Registry for format-specific normalization adapters.

    Manages adapter registration, selection, and lifecycle. Supports
    dynamic registration for extensibility.

    Example:
        >>> registry = FormatAdapterRegistry()
        >>> registry.register(MarkdownAdapter())
        >>> registry.register(PDFAdapter())
        >>> adapter = registry.get_adapter(DocumentFormat.MARKDOWN)
    """

    def __init__(self):
        self._adapters: Dict[DocumentFormat, FormatAdapter] = {}
        self._fallback_adapter: Optional[FormatAdapter] = None

    def register(self, adapter: FormatAdapter) -> None:
        """Register a format adapter.

        Args:
            adapter: Adapter instance to register

        Raises:
            ValueError: If adapter for format already registered
        """
        for fmt in adapter.supported_formats:
            if fmt in self._adapters:
                existing = self._adapters[fmt]
                raise ValueError(
                    f"Adapter for {fmt.value} already registered: {existing.name}"
                )
            self._adapters[fmt] = adapter
            logger.debug(f"Registered {adapter.name} for format {fmt.value}")

    def register_fallback(self, adapter: FormatAdapter) -> None:
        """Register fallback adapter for unknown formats."""
        self._fallback_adapter = adapter
        logger.debug(f"Registered fallback adapter: {adapter.name}")

    def get_adapter(self, format: DocumentFormat) -> FormatAdapter:
        """Get adapter for specified format.

        Args:
            format: Document format

        Returns:
            Registered adapter for format

        Raises:
            AdapterNotFoundError: If no adapter registered for format
        """
        if format in self._adapters:
            return self._adapters[format]

        if self._fallback_adapter:
            logger.warning(
                f"No specific adapter for {format.value}, using fallback"
            )
            return self._fallback_adapter

        raise AdapterNotFoundError(
            f"No adapter registered for format: {format.value}"
        )

    def list_supported_formats(self) -> List[DocumentFormat]:
        """Get list of all supported formats."""
        return list(self._adapters.keys())

    def register_default_adapters(self) -> None:
        """Register all default adapters.

        Called during service initialization to set up common adapters.
        """
        from .adapters import (
            MarkdownAdapter,
            EmailAdapter,
            PDFAdapter,
            HTMLAdapter,
            CodeAdapter,
            TextAdapter,
            GenericAdapter,
        )

        # Text-based formats
        self.register(MarkdownAdapter())
        self.register(EmailAdapter())
        self.register(HTMLAdapter())
        self.register(CodeAdapter())
        self.register(TextAdapter())

        # Binary formats
        self.register(PDFAdapter())

        # Fallback for unknown formats
        self.register_fallback(GenericAdapter())


class AdapterNotFoundError(Exception):
    """Raised when no adapter found for format."""
    pass


class AdapterError(Exception):
    """Base exception for adapter-specific errors."""
    pass
```

### Built-in Adapters

#### MarkdownAdapter (Leverage Existing)

```python
from ...ingestion.obsidian.normalizer import MarkdownNormalizer
from .schema import NormalizedDocument, NormalizedMetadata, DocumentFormat


class MarkdownAdapter:
    """Adapter for Markdown documents using existing ObsidianNormalizer."""

    name = "markdown_adapter"
    supported_formats = [DocumentFormat.MARKDOWN]
    requires_unstructured_processing = False  # Already handles parsing

    def __init__(self):
        self.normalizer = MarkdownNormalizer()

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize markdown using existing MarkdownNormalizer."""
        # Read content
        content = file_path.read_text(encoding='utf-8')

        # Use existing normalizer
        normalized_result = self.normalizer.normalize(content, file_path)

        # Convert to NormalizedDocument schema
        metadata = NormalizedMetadata(
            source_path=str(file_path),
            source_id=source_id,
            source_type=source_type,
            format=DocumentFormat.MARKDOWN,
            content_type="text/markdown",
            file_size_bytes=file_path.stat().st_size,
            character_count=len(content),
            word_count=len(content.split()),
            line_count=content.count('\n') + 1,
            content_hash="",  # Will be computed by enrichment pipeline
            frontmatter=normalized_result.get("frontmatter"),
            tags=normalized_result.get("tags", []),
            aliases=normalized_result.get("aliases", []),
            extra=normalized_result.get("extra", {})
        )

        return NormalizedDocument(
            document_id="",  # Will be set by service
            sha256="",  # Will be computed by enrichment
            content=normalized_result.get("normalized_content", content),
            metadata=metadata
        )

    async def validate(self, file_path: Path) -> bool:
        """Validate markdown file."""
        return file_path.suffix.lower() in ['.md', '.markdown']
```

#### EmailAdapter (Leverage Existing)

```python
from ...ingestion.imap.email_normalizer import EmailNormalizer as IMAPEmailNormalizer
from ...ingestion.imap.email_parser import EmailParser


class EmailAdapter:
    """Adapter for email documents using existing EmailNormalizer."""

    name = "email_adapter"
    supported_formats = [DocumentFormat.EMAIL]
    requires_unstructured_processing = False

    def __init__(self):
        self.parser = EmailParser()
        self.normalizer = IMAPEmailNormalizer()

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize email using existing parsers."""
        # Read raw email
        raw_message = file_path.read_bytes()

        # Parse email
        email_message = self.parser.parse_message(
            raw_message=raw_message,
            uid=0,  # Not from IMAP
            folder="local",
            mailbox_id=source_id
        )

        # Normalize for Unstructured
        normalized_text = self.normalizer.normalize(email_message)

        # Build metadata
        metadata = NormalizedMetadata(
            source_path=str(file_path),
            source_id=source_id,
            source_type=source_type,
            format=DocumentFormat.EMAIL,
            content_type="message/rfc822",
            file_size_bytes=len(raw_message),
            character_count=len(normalized_text),
            word_count=len(normalized_text.split()),
            line_count=normalized_text.count('\n') + 1,
            content_hash="",
            extra={
                "from": email_message.from_address.address,
                "subject": email_message.subject,
                "date": email_message.date.isoformat(),
                "has_attachments": len(email_message.attachments) > 0
            }
        )

        return NormalizedDocument(
            document_id="",
            sha256="",
            content=normalized_text,
            metadata=metadata
        )

    async def validate(self, file_path: Path) -> bool:
        """Validate email file."""
        return file_path.suffix.lower() in ['.eml', '.msg']
```

#### PDFAdapter (New)

```python
class PDFAdapter:
    """Adapter for PDF documents via Unstructured.io."""

    name = "pdf_adapter"
    supported_formats = [DocumentFormat.PDF]
    requires_unstructured_processing = True  # Needs Unstructured.io

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Prepare PDF for Unstructured.io processing."""
        # Read file for hashing (content extracted by Unstructured.io)
        file_size = file_path.stat().st_size

        metadata = NormalizedMetadata(
            source_path=str(file_path),
            source_id=source_id,
            source_type=source_type,
            format=DocumentFormat.PDF,
            content_type="application/pdf",
            file_size_bytes=file_size,
            character_count=0,  # Will be updated after Unstructured processing
            word_count=0,
            line_count=0,
            content_hash="",
            extra=source_metadata
        )

        # Return placeholder - Unstructured.io will extract content
        return NormalizedDocument(
            document_id="",
            sha256="",
            content=None,  # Content extracted by Unstructured.io
            metadata=metadata
        )

    async def validate(self, file_path: Path) -> bool:
        """Validate PDF file."""
        if file_path.suffix.lower() != '.pdf':
            return False

        # Check PDF magic number
        try:
            header = file_path.read_bytes(0, 4)
            return header == b'%PDF'
        except Exception:
            return False
```

#### HTMLAdapter (New)

```python
from html.parser import HTMLParser


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML."""

    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self) -> str:
        return ''.join(self.text)


class HTMLAdapter:
    """Adapter for HTML documents."""

    name = "html_adapter"
    supported_formats = [DocumentFormat.HTML]
    requires_unstructured_processing = True

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize HTML document."""
        html_content = file_path.read_text(encoding='utf-8')

        # Extract text for metadata (Unstructured.io handles full parsing)
        extractor = HTMLTextExtractor()
        extractor.feed(html_content)
        text_content = extractor.get_text()

        metadata = NormalizedMetadata(
            source_path=str(file_path),
            source_id=source_id,
            source_type=source_type,
            format=DocumentFormat.HTML,
            content_type="text/html",
            file_size_bytes=file_path.stat().st_size,
            character_count=len(text_content),
            word_count=len(text_content.split()),
            line_count=html_content.count('\n') + 1,
            content_hash=""
        )

        return NormalizedDocument(
            document_id="",
            sha256="",
            content=html_content,  # Unstructured.io will parse
            metadata=metadata
        )

    async def validate(self, file_path: Path) -> bool:
        """Validate HTML file."""
        return file_path.suffix.lower() in ['.html', '.htm']
```

#### CodeAdapter (New)

```python
import re


class CodeAdapter:
    """Adapter for source code files with comment extraction."""

    name = "code_adapter"
    supported_formats = [DocumentFormat.CODE]
    requires_unstructured_processing = False

    # Language-specific comment patterns
    COMMENT_PATTERNS = {
        'python': (r'#.*$', r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"),
        'javascript': (r'//.*$', r'/\*[\s\S]*?\*/'),
        'java': (r'//.*$', r'/\*[\s\S]*?\*/'),
        'go': (r'//.*$', r'/\*[\s\S]*?\*/'),
        'rust': (r'//.*$', r'/\*[\s\S]*?\*/'),
    }

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize code file, extracting comments."""
        content = file_path.read_text(encoding='utf-8')
        language = self._detect_language(file_path)

        # Extract comments
        comments = self._extract_comments(content, language)

        metadata = NormalizedMetadata(
            source_path=str(file_path),
            source_id=source_id,
            source_type=source_type,
            format=DocumentFormat.CODE,
            content_type="text/plain",
            file_size_bytes=file_path.stat().st_size,
            character_count=len(content),
            word_count=len(content.split()),
            line_count=content.count('\n') + 1,
            content_hash="",
            extra={
                "language": language,
                "comment_count": len(comments)
            }
        )

        # Content is comments + code structure (for PKG)
        enriched_content = f"# Comments from {file_path.name}\n\n"
        enriched_content += "\n\n".join(comments)
        enriched_content += f"\n\n# Original Code\n\n{content}"

        return NormalizedDocument(
            document_id="",
            sha256="",
            content=enriched_content,
            metadata=metadata
        )

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
        }
        return ext_map.get(file_path.suffix.lower(), 'unknown')

    def _extract_comments(self, content: str, language: str) -> List[str]:
        """Extract comments from code."""
        if language not in self.COMMENT_PATTERNS:
            return []

        comments = []
        for pattern in self.COMMENT_PATTERNS[language]:
            matches = re.findall(pattern, content, re.MULTILINE)
            comments.extend(matches)

        return comments

    async def validate(self, file_path: Path) -> bool:
        """Validate code file."""
        code_extensions = ['.py', '.js', '.java', '.go', '.rs', '.c', '.cpp', '.h']
        return file_path.suffix.lower() in code_extensions
```

## Acceptance Criteria

- ✅ FormatAdapter protocol defines consistent interface
- ✅ FormatAdapterRegistry manages adapter lifecycle
- ✅ Existing normalizers integrated (Markdown, Email, GitHub)
- ✅ New adapters created (PDF, HTML, Code)
- ✅ Adapter selection by format working correctly
- ✅ Fallback adapter for unknown formats
- ✅ Dynamic adapter registration supported
- ✅ Validation logic per adapter
- ✅ Format-specific metadata extraction

## Test Plan

### Unit Tests
- Adapter registration and retrieval
- Format validation per adapter
- Fallback adapter selection
- Duplicate registration rejection
- Metadata extraction accuracy

### Integration Tests
- Each adapter with sample documents
- Adapter chaining with Unstructured.io
- Error handling per adapter type
- Multi-format document batches

## Open Questions

- Should adapters support batch processing?
- How to handle format-specific configuration?
- Should adapters be dynamically loadable (plugins)?
- How to version adapters independently?

## Dependencies

- NormalizedDocument schema (Task 01)
- Existing normalizers (Obsidian, Email)
- Unstructured.io bridge (Task 07)


