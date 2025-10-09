"""Markdown format adapter.

Wraps existing ObsidianNormalizer for markdown document processing with
frontmatter, wikilinks, and tag extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from ....ingestion.obsidian.normalizer import MarkdownNormalizer
from ...models import DocumentFormat, NormalizedDocument
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class MarkdownAdapter(BaseAdapter):
    """Adapter for Markdown documents.

    Leverages existing ObsidianNormalizer for robust markdown parsing with
    support for frontmatter, wikilinks, tags, callouts, and other Obsidian
    extensions. Does not require Unstructured.io processing.

    Example:
        >>> adapter = MarkdownAdapter()
        >>> doc = await adapter.normalize(
        ...     file_path=Path("note.md"),
        ...     source_id="note-123",
        ...     source_type="local_files",
        ...     source_metadata={}
        ... )
    """

    def __init__(self):
        super().__init__(
            name="MarkdownAdapter",
            supported_formats=[DocumentFormat.MARKDOWN],
        )
        self.requires_unstructured_processing = False
        self._normalizer = MarkdownNormalizer()

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize markdown document.

        Args:
            file_path: Path to markdown file
            source_id: Connector-specific identifier
            source_type: Source type
            source_metadata: Additional metadata

        Returns:
            NormalizedDocument with parsed content and metadata

        Raises:
            AdapterError: If markdown parsing fails
        """
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            # Parse markdown using ObsidianNormalizer
            parse_result = self._normalizer.parse(markdown_content)

            # Extract frontmatter
            frontmatter = None
            if parse_result.frontmatter:
                frontmatter = parse_result.frontmatter

            # Extract tags
            tags = []
            if parse_result.tags:
                tags = [tag.name for tag in parse_result.tags]

            # Extract aliases from frontmatter
            aliases = []
            if frontmatter and "aliases" in frontmatter:
                alias_value = frontmatter["aliases"]
                if isinstance(alias_value, list):
                    aliases = alias_value
                elif isinstance(alias_value, str):
                    aliases = [alias_value]

            # Get clean content (without frontmatter)
            content = parse_result.content

            # Create normalized document
            document = self.create_normalized_document(
                content=content,
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                format=DocumentFormat.MARKDOWN,
                source_metadata=source_metadata,
                frontmatter=frontmatter,
                tags=tags,
                aliases=aliases,
            )

            # Add markdown-specific metadata
            document.metadata.extra["markdown"] = {
                "has_frontmatter": parse_result.frontmatter is not None,
                "has_links": len(parse_result.wikilinks) > 0,
                "has_tags": len(parse_result.tags) > 0,
                "link_count": len(parse_result.wikilinks),
                "tag_count": len(parse_result.tags),
                "has_tables": len(parse_result.tables) > 0,
                "has_code_blocks": len(parse_result.code_blocks) > 0,
            }

            logger.debug(f"Normalized markdown document: {file_path.name}")

            return document

        except Exception as e:
            logger.error(f"Markdown normalization failed for {file_path.name}: {e}")
            from ..registry import AdapterError

            raise AdapterError(
                f"Failed to normalize markdown document: {str(e)}"
            ) from e
