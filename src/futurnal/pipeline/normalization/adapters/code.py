"""Code format adapter.

Specialized adapter for source code files with comment extraction and
language-specific parsing. Extracts documentation and preserves code structure.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...models import DocumentFormat, NormalizedDocument
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class CommentExtractor:
    """Extract comments from source code based on language syntax."""

    # Language-specific comment patterns
    LANGUAGE_PATTERNS = {
        "python": {
            "single_line": r"#.*$",
            "multi_line": [r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
            "extensions": {".py", ".pyw", ".pyi"},
        },
        "javascript": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".js", ".jsx", ".mjs", ".cjs"},
        },
        "typescript": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".ts", ".tsx"},
        },
        "java": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".java"},
        },
        "go": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".go"},
        },
        "rust": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".rs"},
        },
        "c": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".c", ".h"},
        },
        "cpp": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h++"},
        },
        "csharp": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".cs"},
        },
        "ruby": {
            "single_line": r"#.*$",
            "multi_line": [r"=begin[\s\S]*?=end"],
            "extensions": {".rb"},
        },
        "php": {
            "single_line": r"(?://.*$|#.*$)",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".php"},
        },
        "swift": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".swift"},
        },
        "kotlin": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".kt", ".kts"},
        },
        "scala": {
            "single_line": r"//.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".scala"},
        },
        "shell": {
            "single_line": r"#.*$",
            "multi_line": [],
            "extensions": {".sh", ".bash", ".zsh"},
        },
        "sql": {
            "single_line": r"--.*$",
            "multi_line": [r"/\*[\s\S]*?\*/"],
            "extensions": {".sql"},
        },
        "r": {
            "single_line": r"#.*$",
            "multi_line": [],
            "extensions": {".r", ".R"},
        },
        "lua": {
            "single_line": r"--.*$",
            "multi_line": [r"--\[\[[\s\S]*?--\]\]"],
            "extensions": {".lua"},
        },
    }

    def detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension.

        Args:
            file_path: Path to source file

        Returns:
            Language name or None if unknown
        """
        extension = file_path.suffix.lower()

        for lang, config in self.LANGUAGE_PATTERNS.items():
            if extension in config["extensions"]:
                return lang

        return None

    def extract_comments(
        self, content: str, language: str
    ) -> Tuple[List[str], Dict[str, int]]:
        """Extract comments from source code.

        Args:
            content: Source code content
            language: Programming language

        Returns:
            Tuple of (comment_list, statistics)
        """
        if language not in self.LANGUAGE_PATTERNS:
            return [], {"total": 0, "single_line": 0, "multi_line": 0}

        patterns = self.LANGUAGE_PATTERNS[language]
        comments = []
        stats = {"total": 0, "single_line": 0, "multi_line": 0}

        # Extract single-line comments
        if patterns["single_line"]:
            single_line_pattern = re.compile(patterns["single_line"], re.MULTILINE)
            for match in single_line_pattern.finditer(content):
                comment = match.group().strip()
                if comment:
                    comments.append(comment)
                    stats["single_line"] += 1

        # Extract multi-line comments
        for multi_pattern in patterns["multi_line"]:
            multi_line_pattern = re.compile(multi_pattern, re.MULTILINE)
            for match in multi_line_pattern.finditer(content):
                comment = match.group().strip()
                if comment:
                    comments.append(comment)
                    stats["multi_line"] += 1

        stats["total"] = len(comments)
        return comments, stats


class CodeAdapter(BaseAdapter):
    """Adapter for source code files.

    Specialized adapter that extracts comments, docstrings, and preserves code
    structure for semantic understanding. Supports multiple programming languages
    with language-specific comment syntax.

    Example:
        >>> adapter = CodeAdapter()
        >>> doc = await adapter.normalize(
        ...     file_path=Path("script.py"),
        ...     source_id="code-123",
        ...     source_type="local_files",
        ...     source_metadata={}
        ... )
    """

    def __init__(self):
        super().__init__(
            name="CodeAdapter",
            supported_formats=[DocumentFormat.CODE],
        )
        self.requires_unstructured_processing = False
        self.comment_extractor = CommentExtractor()

    async def normalize(
        self,
        *,
        file_path: Path,
        source_id: str,
        source_type: str,
        source_metadata: dict,
    ) -> NormalizedDocument:
        """Normalize code file.

        Args:
            file_path: Path to source code file
            source_id: Connector-specific identifier
            source_type: Source type
            source_metadata: Additional metadata

        Returns:
            NormalizedDocument with code and extracted comments

        Raises:
            AdapterError: If code normalization fails
        """
        try:
            # Read code content with encoding detection
            content = self._read_code_file(file_path)

            # Detect programming language
            language = self.comment_extractor.detect_language(file_path)
            if not language:
                language = "unknown"
                logger.debug(
                    f"Unknown language for {file_path.name}, using generic code processing"
                )

            # Extract comments
            comments, comment_stats = self.comment_extractor.extract_comments(
                content, language
            )

            # Extract code structure metadata
            structure_info = self._extract_structure_info(content, language)

            # Build enriched content for PKG
            # Format: Comments section + Original code
            enriched_content = self._build_enriched_content(
                file_path, content, comments, language
            )

            # Create tags from language and file type
            tags = [f"language:{language}", "code"]
            if structure_info["has_classes"]:
                tags.append("has:classes")
            if structure_info["has_functions"]:
                tags.append("has:functions")
            if structure_info["has_imports"]:
                tags.append("has:imports")

            # Create normalized document
            document = self.create_normalized_document(
                content=enriched_content,
                file_path=file_path,
                source_id=source_id,
                source_type=source_type,
                format=DocumentFormat.CODE,
                source_metadata=source_metadata,
                tags=tags,
            )

            # Add code-specific metadata
            document.metadata.extra["code"] = {
                "language": language,
                "file_extension": file_path.suffix,
                "comment_count": comment_stats["total"],
                "single_line_comments": comment_stats["single_line"],
                "multi_line_comments": comment_stats["multi_line"],
                "has_comments": comment_stats["total"] > 0,
                "line_count": content.count("\n") + 1,
                "has_classes": structure_info["has_classes"],
                "has_functions": structure_info["has_functions"],
                "has_imports": structure_info["has_imports"],
                "estimated_complexity": structure_info["complexity_score"],
            }

            logger.debug(
                f"Normalized code document: {file_path.name} "
                f"(language: {language}, comments: {comment_stats['total']})"
            )

            return document

        except Exception as e:
            logger.error(f"Code normalization failed for {file_path.name}: {e}")
            from ..registry import AdapterError

            raise AdapterError(f"Failed to normalize code document: {str(e)}") from e

    def _read_code_file(self, file_path: Path) -> str:
        """Read code file with multiple encoding attempts.

        Args:
            file_path: Path to code file

        Returns:
            File content as string
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue

        # Final fallback: read with errors='replace'
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _extract_structure_info(self, content: str, language: str) -> Dict:
        """Extract code structure information.

        Args:
            content: Source code content
            language: Programming language

        Returns:
            Dictionary with structure information
        """
        info = {
            "has_classes": False,
            "has_functions": False,
            "has_imports": False,
            "complexity_score": 0,
        }

        # Common patterns across languages
        class_patterns = [
            r"\bclass\s+\w+",  # Python, Java, C++, etc.
            r"\bstruct\s+\w+",  # C, C++, Go, Rust
            r"\binterface\s+\w+",  # Java, TypeScript, Go
        ]

        function_patterns = [
            r"\bdef\s+\w+",  # Python
            r"\bfunction\s+\w+",  # JavaScript
            r"\bfunc\s+\w+",  # Go
            r"\bfn\s+\w+",  # Rust
            r"\w+\s+\w+\s*\([^\)]*\)\s*\{",  # C-style functions
        ]

        import_patterns = [
            r"\bimport\s+",  # Python, Java, Go
            r"\bfrom\s+\w+\s+import\s+",  # Python
            r"\brequire\s*\(",  # JavaScript/Node
            r"\buse\s+",  # Rust, PHP
            r"#include\s+",  # C/C++
        ]

        # Check for classes
        for pattern in class_patterns:
            if re.search(pattern, content, re.MULTILINE):
                info["has_classes"] = True
                break

        # Check for functions
        for pattern in function_patterns:
            if re.search(pattern, content, re.MULTILINE):
                info["has_functions"] = True
                break

        # Check for imports
        for pattern in import_patterns:
            if re.search(pattern, content, re.MULTILINE):
                info["has_imports"] = True
                break

        # Simple complexity score based on structure
        complexity = 0
        if info["has_classes"]:
            complexity += 2
        if info["has_functions"]:
            complexity += 1
        if info["has_imports"]:
            complexity += 1

        # Add complexity based on control flow keywords
        control_keywords = [
            "if",
            "else",
            "elif",
            "for",
            "while",
            "switch",
            "case",
            "try",
            "catch",
            "except",
        ]
        for keyword in control_keywords:
            # Use word boundary to avoid partial matches
            pattern = rf"\b{keyword}\b"
            matches = len(re.findall(pattern, content))
            complexity += min(matches, 10)  # Cap contribution per keyword

        info["complexity_score"] = min(complexity, 100)  # Cap at 100

        return info

    def _build_enriched_content(
        self, file_path: Path, content: str, comments: List[str], language: str
    ) -> str:
        """Build enriched content with comments and code.

        Args:
            file_path: Path to file
            content: Original code content
            comments: Extracted comments
            language: Programming language

        Returns:
            Enriched content string
        """
        lines = []

        # Header
        lines.append(f"# Code File: {file_path.name}")
        lines.append(f"# Language: {language}")
        lines.append("")

        # Comments section
        if comments:
            lines.append("# Extracted Comments and Documentation:")
            lines.append("")
            for i, comment in enumerate(comments, 1):
                lines.append(f"## Comment {i}")
                lines.append(comment)
                lines.append("")

        # Original code section
        lines.append("# Original Source Code:")
        lines.append("")
        lines.append(content)

        return "\n".join(lines)

    async def validate(self, file_path: Path) -> bool:
        """Validate code file.

        Args:
            file_path: Path to validate

        Returns:
            True if file is a valid code file
        """
        # Check if language is supported
        language = self.comment_extractor.detect_language(file_path)
        if not language:
            return False

        # Check file exists
        if not file_path.exists():
            return False

        # Try to read file
        try:
            self._read_code_file(file_path)
            return True
        except Exception as e:
            logger.debug(f"Code validation failed for {file_path.name}: {e}")
            return False
