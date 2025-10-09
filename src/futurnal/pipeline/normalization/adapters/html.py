"""HTML format adapter.

Processes HTML documents with text extraction and metadata parsing.
Requires Unstructured.io for advanced table and form extraction.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

from ...models import DocumentFormat, NormalizedDocument
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class HTMLMetadataExtractor(HTMLParser):
    """Extract metadata and text content from HTML."""

    def __init__(self):
        super().__init__()
        self.title: Optional[str] = None
        self.meta_tags: Dict[str, str] = {}
        self.text_chunks: List[str] = []
        self.in_title = False
        self.in_script = False
        self.in_style = False
        self.heading_count = 0
        self.link_count = 0
        self.image_count = 0
        self.table_count = 0
        self.form_count = 0

    def handle_starttag(self, tag: str, attrs: List[tuple]):
        """Handle opening tags."""
        # Track title tag
        if tag == "title":
            self.in_title = True

        # Track script/style to skip content
        elif tag == "script":
            self.in_script = True
        elif tag == "style":
            self.in_style = True

        # Extract meta tags
        elif tag == "meta":
            attrs_dict = dict(attrs)
            name = attrs_dict.get("name") or attrs_dict.get("property")
            content = attrs_dict.get("content")
            if name and content:
                self.meta_tags[name] = content

        # Count structural elements
        elif tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.heading_count += 1
        elif tag == "a":
            self.link_count += 1
        elif tag == "img":
            self.image_count += 1
        elif tag == "table":
            self.table_count += 1
        elif tag == "form":
            self.form_count += 1

    def handle_endtag(self, tag: str):
        """Handle closing tags."""
        if tag == "title":
            self.in_title = False
        elif tag == "script":
            self.in_script = False
        elif tag == "style":
            self.in_style = False

    def handle_data(self, data: str):
        """Handle text content."""
        # Extract title
        if self.in_title:
            self.title = data.strip()

        # Skip script/style content
        if self.in_script or self.in_style:
            return

        # Collect text content
        text = data.strip()
        if text:
            self.text_chunks.append(text)

    def get_text(self) -> str:
        """Get extracted text content."""
        return "\n".join(self.text_chunks)

    def get_metadata(self) -> Dict:
        """Get extracted metadata."""
        return {
            "title": self.title,
            "meta_tags": self.meta_tags,
            "heading_count": self.heading_count,
            "link_count": self.link_count,
            "image_count": self.image_count,
            "table_count": self.table_count,
            "form_count": self.form_count,
        }


class HTMLAdapter(BaseAdapter):
    """Adapter for HTML documents.

    Parses HTML documents to extract text content and metadata. Uses
    Unstructured.io processing for advanced features like table extraction,
    form parsing, and structured content detection.

    Example:
        >>> adapter = HTMLAdapter()
        >>> doc = await adapter.normalize(
        ...     file_path=Path("document.html"),
        ...     source_id="doc-123",
        ...     source_type="local_files",
        ...     source_metadata={}
        ... )
    """

    def __init__(self):
        super().__init__(
            name="HTMLAdapter",
            supported_formats=[DocumentFormat.HTML],
        )
        self.requires_unstructured_processing = True

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize HTML document.

        Args:
            file_path: Path to HTML file
            source_id: Connector-specific identifier
            source_type: Source type
            source_metadata: Additional metadata

        Returns:
            Preliminary NormalizedDocument (content will be enhanced by Unstructured.io)

        Raises:
            AdapterError: If HTML parsing fails
        """
        try:
            # Read HTML content
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                html_content = f.read()

            # Extract metadata and text using HTML parser
            extractor = HTMLMetadataExtractor()
            try:
                extractor.feed(html_content)
            except Exception as e:
                logger.warning(
                    f"HTML parsing error for {file_path.name}, using fallback: {e}"
                )
                # Fallback: basic text extraction
                text_content = self._fallback_text_extraction(html_content)
                html_metadata = {
                    "title": None,
                    "meta_tags": {},
                    "heading_count": 0,
                    "link_count": 0,
                    "image_count": 0,
                    "table_count": 0,
                    "form_count": 0,
                }
            else:
                text_content = extractor.get_text()
                html_metadata = extractor.get_metadata()

            # Detect language from HTML attributes
            language = self._detect_html_language(html_content)

            # Build tags from meta keywords
            tags = []
            if "keywords" in html_metadata["meta_tags"]:
                keywords = html_metadata["meta_tags"]["keywords"]
                # Split by comma and clean
                tags = [kw.strip() for kw in keywords.split(",") if kw.strip()]

            # Create normalized document
            # Content will be enhanced by Unstructured.io processing
            document = self.create_normalized_document(
                content=html_content,  # Full HTML for Unstructured.io
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                format=DocumentFormat.HTML,
                source_metadata=source_metadata,
                tags=tags,
            )

            # Add HTML-specific metadata
            document.metadata.extra["html"] = {
                "title": html_metadata["title"],
                "meta_description": html_metadata["meta_tags"].get("description"),
                "meta_author": html_metadata["meta_tags"].get("author"),
                "meta_keywords": html_metadata["meta_tags"].get("keywords"),
                "heading_count": html_metadata["heading_count"],
                "link_count": html_metadata["link_count"],
                "image_count": html_metadata["image_count"],
                "table_count": html_metadata["table_count"],
                "form_count": html_metadata["form_count"],
                "has_tables": html_metadata["table_count"] > 0,
                "has_forms": html_metadata["form_count"] > 0,
                "detected_language": language,
                "extracted_text_length": len(text_content),
            }

            # Override language if detected from HTML
            if language:
                document.metadata.language = language

            logger.debug(
                f"Normalized HTML document: {file_path.name} "
                f"(title: {html_metadata['title']}, "
                f"tables: {html_metadata['table_count']}, "
                f"forms: {html_metadata['form_count']})"
            )

            return document

        except Exception as e:
            logger.error(f"HTML normalization failed for {file_path.name}: {e}")
            from ..registry import AdapterError

            raise AdapterError(f"Failed to normalize HTML document: {str(e)}") from e

    def _detect_html_language(self, html_content: str) -> Optional[str]:
        """Detect language from HTML lang attribute.

        Args:
            html_content: HTML content

        Returns:
            ISO 639-1 language code or None
        """
        # Look for <html lang="..."> or <html xml:lang="...">
        lang_pattern = re.compile(
            r'<html[^>]+(?:lang|xml:lang)=["\']([\w-]+)["\']',
            re.IGNORECASE,
        )
        match = lang_pattern.search(html_content)
        if match:
            lang_code = match.group(1).lower()
            # Extract primary language code (e.g., "en" from "en-US")
            if "-" in lang_code:
                lang_code = lang_code.split("-")[0]
            # Validate it's 2 characters
            if len(lang_code) == 2 and lang_code.isalpha():
                return lang_code

        return None

    def _fallback_text_extraction(self, html_content: str) -> str:
        """Fallback text extraction using regex when parser fails.

        Args:
            html_content: HTML content

        Returns:
            Extracted text
        """
        # Remove script and style tags with content
        text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)

        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode HTML entities
        try:
            from html import unescape

            text = unescape(text)
        except ImportError:
            pass

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    async def validate(self, file_path: Path) -> bool:
        """Validate HTML file.

        Args:
            file_path: Path to validate

        Returns:
            True if file is valid HTML
        """
        # Check extension
        if file_path.suffix.lower() not in [".html", ".htm"]:
            return False

        # Check file exists
        if not file_path.exists():
            return False

        # Basic content validation
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(1024)  # Read first 1KB

            # Check for HTML markers
            content_lower = content.lower()
            has_html_tag = "<html" in content_lower or "<!doctype html" in content_lower
            has_body_tag = "<body" in content_lower
            has_head_tag = "<head" in content_lower

            # Valid if has HTML structure or at least some HTML tags
            return has_html_tag or has_body_tag or has_head_tag or "<" in content

        except Exception as e:
            logger.debug(f"HTML validation failed for {file_path.name}: {e}")
            return False
