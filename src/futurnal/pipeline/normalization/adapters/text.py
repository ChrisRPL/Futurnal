"""Text format adapter (fallback).

Simple text extraction adapter that serves as fallback for unknown or
plain text formats. Provides minimal processing with basic metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...models import DocumentFormat, NormalizedDocument
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class TextAdapter(BaseAdapter):
    """Adapter for plain text documents.

    Serves as fallback adapter for unknown formats. Performs simple text
    extraction with minimal processing. Does not require Unstructured.io.

    Example:
        >>> adapter = TextAdapter()
        >>> doc = await adapter.normalize(
        ...     file_path=Path("document.txt"),
        ...     source_id="doc-123",
        ...     source_type="local_files",
        ...     source_metadata={}
        ... )
    """

    def __init__(self):
        super().__init__(
            name="TextAdapter",
            supported_formats=[
                DocumentFormat.TEXT,
                DocumentFormat.CODE,
                DocumentFormat.UNKNOWN,
            ],
        )
        self.requires_unstructured_processing = False

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize text document.

        Args:
            file_path: Path to text file
            source_id: Connector-specific identifier
            source_type: Source type
            source_metadata: Additional metadata

        Returns:
            NormalizedDocument with text content

        Raises:
            AdapterError: If text reading fails
        """
        try:
            # Read file content with multiple encoding attempts
            content = self._read_file_with_encoding(file_path)

            # Detect if this is a code file
            format = self._detect_format(file_path, content)

            # Create normalized document
            document = self.create_normalized_document(
                content=content,
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                format=format,
                source_metadata=source_metadata,
            )

            # Add text-specific metadata
            document.metadata.extra["text"] = {
                "encoding": "utf-8",  # Simplified - actual encoding detection can be added
                "is_code": format == DocumentFormat.CODE,
                "file_extension": file_path.suffix,
            }

            logger.debug(
                f"Normalized text document ({format.value}): {file_path.name}"
            )

            return document

        except Exception as e:
            logger.error(f"Text normalization failed for {file_path.name}: {e}")
            from ..registry import AdapterError

            raise AdapterError(f"Failed to normalize text document: {str(e)}") from e

    def _read_file_with_encoding(self, file_path: Path) -> str:
        """Read file with multiple encoding attempts.

        Args:
            file_path: Path to file

        Returns:
            File content as string

        Raises:
            UnicodeDecodeError: If file cannot be decoded with any encoding
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # If all encodings fail, read as binary and decode with errors='replace'
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")

    def _detect_format(self, file_path: Path, content: str) -> DocumentFormat:
        """Detect if file is code or plain text.

        Args:
            file_path: Path to file
            content: File content

        Returns:
            DocumentFormat.CODE or DocumentFormat.TEXT
        """
        # Code file extensions
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".sh",
            ".bash",
            ".sql",
            ".r",
            ".m",
            ".lua",
            ".pl",
            ".pm",
            ".tcl",
        }

        if file_path.suffix.lower() in code_extensions:
            return DocumentFormat.CODE

        # Heuristic: if content has typical code patterns, consider it code
        if self._looks_like_code(content):
            return DocumentFormat.CODE

        return DocumentFormat.TEXT

    def _looks_like_code(self, content: str) -> bool:
        """Heuristic check if content looks like code.

        Args:
            content: Text content

        Returns:
            True if content appears to be code
        """
        # Simple heuristics
        lines = content.split("\n")
        if not lines:
            return False

        # Count lines with code-like patterns
        code_indicators = 0
        total_lines = min(len(lines), 50)  # Check first 50 lines

        for line in lines[:total_lines]:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Check for common code patterns
            if any(
                [
                    stripped.startswith(("import ", "from ", "#include", "using ")),
                    " = " in stripped,
                    stripped.endswith((";", "{", "}", ":")),
                    stripped.startswith(("def ", "class ", "function ", "var ", "let ", "const ")),
                    "=>" in stripped,
                    "->" in stripped,
                ]
            ):
                code_indicators += 1

        # If >30% of lines look like code, consider it code
        non_empty_lines = sum(1 for line in lines[:total_lines] if line.strip())
        if non_empty_lines == 0:
            return False

        return (code_indicators / non_empty_lines) > 0.3

    async def validate(self, file_path: Path) -> bool:
        """Validate text file.

        As fallback adapter, accepts any file that exists.

        Args:
            file_path: Path to validate

        Returns:
            True if file exists
        """
        return file_path.exists()
