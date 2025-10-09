"""Generic format adapter (fallback).

Ultimate fallback adapter for unknown or unsupported file formats.
Attempts text extraction with best-effort encoding detection.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...models import DocumentFormat, NormalizedDocument
from .text import TextAdapter

logger = logging.getLogger(__name__)


class GenericAdapter(TextAdapter):
    """Generic fallback adapter for unknown formats.

    Serves as the ultimate fallback when no specific adapter is registered
    for a format. Extends TextAdapter with more permissive validation and
    additional logging for unknown formats.

    Example:
        >>> adapter = GenericAdapter()
        >>> doc = await adapter.normalize(
        ...     file_path=Path("unknown.dat"),
        ...     source_id="doc-123",
        ...     source_type="local_files",
        ...     source_metadata={}
        ... )
    """

    def __init__(self):
        # Call parent __init__ to set up TextAdapter base
        super().__init__()

        # Override name and formats
        self.name = "GenericAdapter"
        self.supported_formats = [
            DocumentFormat.TEXT,
            DocumentFormat.CODE,
            DocumentFormat.UNKNOWN,
        ]

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize document with unknown format.

        Args:
            file_path: Path to file
            source_id: Connector-specific identifier
            source_type: Source type
            source_metadata: Additional metadata

        Returns:
            NormalizedDocument with best-effort text extraction

        Raises:
            AdapterError: If normalization fails
        """
        # Log that we're using fallback adapter
        logger.info(
            f"Using GenericAdapter fallback for {file_path.name} "
            f"(extension: {file_path.suffix})"
        )

        try:
            # Use parent TextAdapter normalize method
            document = await super().normalize(
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                source_metadata=source_metadata,
            )

            # Add generic adapter metadata
            document.metadata.extra["generic_adapter"] = {
                "used_fallback": True,
                "original_extension": file_path.suffix,
                "adapter_name": self.name,
            }

            # If format was detected as CODE or TEXT, override to UNKNOWN
            # to indicate this was a fallback
            if file_path.suffix.lower() not in [".txt", ".text"]:
                document.metadata.format = DocumentFormat.UNKNOWN
                document.metadata.content_type = "application/octet-stream"

            logger.debug(
                f"Generic normalization completed for {file_path.name} "
                f"({len(document.content)} chars)"
            )

            return document

        except Exception as e:
            logger.error(f"Generic normalization failed for {file_path.name}: {e}")
            from ..registry import AdapterError

            raise AdapterError(
                f"Failed to normalize document with generic adapter: {str(e)}"
            ) from e

    async def validate(self, file_path: Path) -> bool:
        """Validate file for generic processing.

        As the ultimate fallback, this accepts any file that exists
        and is readable.

        Args:
            file_path: Path to validate

        Returns:
            True if file exists and is readable
        """
        if not file_path.exists():
            return False

        # Check if file is readable
        try:
            # Try to open and read a small sample
            with open(file_path, "rb") as f:
                sample = f.read(1024)

            # Accept any file that can be read
            return True

        except Exception as e:
            logger.debug(f"Generic validation failed for {file_path.name}: {e}")
            return False
