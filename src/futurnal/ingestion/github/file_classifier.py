"""File classification engine for GitHub repositories.

This module implements a multi-stage classification system that routes repository
files to appropriate processing pipelines based on their category and characteristics.
"""

from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Optional

from .classifier_models import FileCategory, FileClassification, ProgrammingLanguage
from .language_detector import detect_language
from .secret_detector import SecretDetector


class FileClassifier:
    """Classifies repository files into semantic categories.

    Implements a three-stage classification waterfall:
    1. Extension-based (fast, high confidence)
    2. Path-based (medium confidence)
    3. Content-based (slow, variable confidence)

    Attributes:
        use_codebert: Whether to use CodeBERT for content classification
        secret_detector: Secret detection instance
        extension_map: File extension to category mapping
        path_patterns: Path pattern to category mapping
    """

    def __init__(
        self,
        *,
        use_codebert: bool = False,
        secret_detector: Optional[SecretDetector] = None,
    ):
        """Initialize the file classifier.

        Args:
            use_codebert: Enable expensive CodeBERT classification (default: False)
            secret_detector: Optional secret detector instance
        """
        self.use_codebert = use_codebert
        self.secret_detector = secret_detector or SecretDetector()

        # Load classification rules
        self.extension_map = self._load_extension_map()
        self.path_patterns = self._load_path_patterns()

    def classify_file(
        self, file_path: str, content: Optional[bytes] = None
    ) -> FileClassification:
        """Classify a single file.

        Args:
            file_path: Relative path within the repository
            content: Optional file content for analysis

        Returns:
            FileClassification with category, language, and processing directives
        """
        # Determine file size
        size_bytes = len(content) if content else 0

        # Check if binary
        is_binary = self._is_binary(content) if content else False

        # Try path-based classification FIRST for test files and special configs
        # This ensures test files and setup.py are correctly categorized
        path_result = self._classify_by_path(file_path)

        # Try extension-based classification
        extension_result = self._classify_by_extension(file_path)

        # Priority logic:
        # 1. Path-based wins for test patterns and special files
        # 2. Extension-based wins for high-confidence matches
        # 3. Fall back to content-based or unknown

        # Determine if path should take priority
        filename_lower = Path(file_path).name.lower()
        special_config_files = [
            'setup.py', 'setup.cfg', 'dockerfile', 'makefile',
            'requirements.txt', 'pipfile', 'gemfile', 'package.json'
        ]

        if path_result and (
            path_result.category == FileCategory.TEST_CODE or
            filename_lower in special_config_files or
            (path_result.category == FileCategory.CONFIGURATION and
             filename_lower in special_config_files)
        ):
            # Test files and special config files: path wins
            classification = path_result
        elif extension_result and extension_result.confidence > 0.8:
            # High confidence extension match
            classification = extension_result
        elif path_result and path_result.confidence > 0.7:
            # Medium confidence path match
            classification = path_result
        else:
            # Fallback to content-based or unknown
            if content and not is_binary:
                # Try content-based classification
                classification = self._classify_by_content(file_path, content)
            else:
                # Use best available or mark as unknown
                classification = extension_result or path_result or FileClassification(
                    file_path=file_path,
                    category=FileCategory.UNKNOWN,
                    confidence=0.0,
                    classified_by="fallback",
                    size_bytes=size_bytes,
                    is_binary=is_binary,
                )

        # Update size and binary flag
        classification.size_bytes = size_bytes
        classification.is_binary = is_binary

        # Count lines for text files
        if content and not is_binary:
            try:
                text = content.decode("utf-8", errors="ignore")
                classification.line_count = text.count("\n") + 1
            except (UnicodeDecodeError, AttributeError):
                classification.line_count = None

        # Detect secrets if configuration
        if classification.category == FileCategory.CONFIGURATION and content:
            classification.has_secrets = self.secret_detector.detect(content)

        # Set embedding strategy
        classification.embedding_type = self._determine_embedding_type(classification)
        classification.should_embed = classification.embedding_type != "none"

        return classification

    def _classify_by_extension(self, file_path: str) -> Optional[FileClassification]:
        """Classify based on file extension.

        Args:
            file_path: Path to the file

        Returns:
            FileClassification or None if extension not recognized
        """
        ext = Path(file_path).suffix.lower()

        if ext in self.extension_map:
            mapping = self.extension_map[ext]
            return FileClassification(
                file_path=file_path,
                category=mapping["category"],
                language=mapping.get("language"),
                confidence=0.9,
                classified_by="extension",
                size_bytes=0,  # Will be set later
            )

        return None

    def _classify_by_path(self, file_path: str) -> Optional[FileClassification]:
        """Classify based on file path patterns.

        Args:
            file_path: Path to the file

        Returns:
            FileClassification or None if no pattern matches
        """
        path_lower = file_path.lower()
        path_basename = Path(file_path).name.lower()

        for pattern, category in self.path_patterns.items():
            # Check for directory in path (patterns ending with /)
            if pattern.endswith("/") and pattern in path_lower:
                return FileClassification(
                    file_path=file_path,
                    category=category,
                    confidence=0.75,
                    classified_by="path",
                    size_bytes=0,
                )
            # Check for filename pattern match
            elif fnmatch(path_basename, pattern):
                return FileClassification(
                    file_path=file_path,
                    category=category,
                    confidence=0.75,
                    classified_by="path",
                    size_bytes=0,
                )
            # Check full path pattern match
            elif fnmatch(path_lower, pattern):
                return FileClassification(
                    file_path=file_path,
                    category=category,
                    confidence=0.75,
                    classified_by="path",
                    size_bytes=0,
                )

        return None

    def _classify_by_content(
        self, file_path: str, content: bytes
    ) -> FileClassification:
        """Classify based on file content (stub for future CodeBERT integration).

        Args:
            file_path: Path to the file
            content: File content as bytes

        Returns:
            FileClassification based on content analysis
        """
        # This is a stub for future CodeBERT integration
        # For now, use basic heuristics
        try:
            text = content.decode("utf-8", errors="ignore")[:1000]  # First 1KB

            # Try to detect language from content
            language = detect_language(file_path, text)

            if language:
                return FileClassification(
                    file_path=file_path,
                    category=FileCategory.SOURCE_CODE,
                    language=language,
                    confidence=0.6,
                    classified_by="content",
                    size_bytes=len(content),
                )
        except Exception:
            pass

        # Unknown
        return FileClassification(
            file_path=file_path,
            category=FileCategory.UNKNOWN,
            confidence=0.3,
            classified_by="content",
            size_bytes=len(content),
        )

    def _determine_embedding_type(self, classification: FileClassification) -> str:
        """Determine which embedding model to use.

        Args:
            classification: File classification result

        Returns:
            Embedding type: "codebert", "standard", or "none"
        """
        if classification.is_binary:
            return "none"

        if classification.category == FileCategory.SOURCE_CODE:
            return "codebert"  # Code-specific embeddings

        if classification.category == FileCategory.DOCUMENTATION:
            return "standard"  # Standard text embeddings

        if classification.category == FileCategory.CONFIGURATION:
            return "standard"  # Config as structured text

        if classification.category == FileCategory.TEST_CODE:
            return "codebert"  # Test code uses code embeddings

        if classification.category == FileCategory.ASSET:
            return "none"

        return "standard"  # Default

    def _is_binary(self, content: Optional[bytes]) -> bool:
        """Detect if content is binary.

        Args:
            content: File content as bytes

        Returns:
            True if binary, False otherwise
        """
        if not content:
            return False

        # Check first 8KB for null bytes
        sample = content[:8192]
        return b"\x00" in sample

    def _load_extension_map(self) -> Dict[str, Dict]:
        """Load file extension to category mapping.

        Returns:
            Dictionary mapping extensions to category and language
        """
        return {
            # Source code - Python
            ".py": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.PYTHON,
            },
            ".pyw": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.PYTHON,
            },
            ".pyi": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.PYTHON,
            },
            # Source code - JavaScript/TypeScript
            ".js": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.JAVASCRIPT,
            },
            ".mjs": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.JAVASCRIPT,
            },
            ".cjs": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.JAVASCRIPT,
            },
            ".jsx": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.JAVASCRIPT,
            },
            ".ts": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.TYPESCRIPT,
            },
            ".tsx": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.TYPESCRIPT,
            },
            # Source code - Java/JVM
            ".java": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.JAVA,
            },
            ".kt": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.KOTLIN,
            },
            ".scala": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.SCALA,
            },
            ".groovy": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.GROOVY,
            },
            ".clj": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.CLOJURE,
            },
            # Source code - Systems
            ".go": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.GO,
            },
            ".rs": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.RUST,
            },
            ".c": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.C,
            },
            ".h": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.C,
            },
            ".cpp": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.CPP,
            },
            ".cc": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.CPP,
            },
            ".cxx": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.CPP,
            },
            ".hpp": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.CPP,
            },
            ".hxx": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.CPP,
            },
            # Source code - Mobile
            ".swift": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.SWIFT,
            },
            ".m": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.OBJECTIVE_C,
            },
            ".mm": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.OBJECTIVE_C,
            },
            ".dart": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.DART,
            },
            # Source code - Web/scripting
            ".php": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.PHP,
            },
            ".rb": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.RUBY,
            },
            ".rake": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.RUBY,
            },
            ".pl": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.PERL,
            },
            ".lua": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.LUA,
            },
            # Source code - .NET
            ".cs": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.CSHARP,
            },
            ".fs": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.F_SHARP,
            },
            # Source code - Functional
            ".hs": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.HASKELL,
            },
            ".ml": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.OCAML,
            },
            ".ex": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.ELIXIR,
            },
            ".erl": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.ERLANG,
            },
            # Source code - Data science
            ".r": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.R,
            },
            ".R": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.R,
            },
            ".jl": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.JULIA,
            },
            # Source code - Shell
            ".sh": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.BASH,
            },
            ".bash": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.BASH,
            },
            ".zsh": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.SHELL,
            },
            ".ps1": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.POWERSHELL,
            },
            # Source code - Other
            ".sql": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.SQL,
            },
            ".proto": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.PROTO,
            },
            ".sol": {
                "category": FileCategory.SOURCE_CODE,
                "language": ProgrammingLanguage.SOLIDITY,
            },
            # Documentation
            ".md": {"category": FileCategory.DOCUMENTATION},
            ".markdown": {"category": FileCategory.DOCUMENTATION},
            ".rst": {"category": FileCategory.DOCUMENTATION},
            ".txt": {"category": FileCategory.DOCUMENTATION},
            ".adoc": {"category": FileCategory.DOCUMENTATION},
            ".org": {"category": FileCategory.DOCUMENTATION},
            ".tex": {"category": FileCategory.DOCUMENTATION},
            # Configuration
            ".json": {"category": FileCategory.CONFIGURATION},
            ".yaml": {"category": FileCategory.CONFIGURATION},
            ".yml": {"category": FileCategory.CONFIGURATION},
            ".toml": {"category": FileCategory.CONFIGURATION},
            ".ini": {"category": FileCategory.CONFIGURATION},
            ".cfg": {"category": FileCategory.CONFIGURATION},
            ".conf": {"category": FileCategory.CONFIGURATION},
            ".config": {"category": FileCategory.CONFIGURATION},
            ".xml": {"category": FileCategory.CONFIGURATION},
            ".env": {"category": FileCategory.CONFIGURATION},
            ".properties": {"category": FileCategory.CONFIGURATION},
            # Assets - Images
            ".jpg": {"category": FileCategory.ASSET},
            ".jpeg": {"category": FileCategory.ASSET},
            ".png": {"category": FileCategory.ASSET},
            ".gif": {"category": FileCategory.ASSET},
            ".bmp": {"category": FileCategory.ASSET},
            ".ico": {"category": FileCategory.ASSET},
            ".svg": {"category": FileCategory.ASSET},
            ".webp": {"category": FileCategory.ASSET},
            # Assets - Video/Audio
            ".mp4": {"category": FileCategory.ASSET},
            ".avi": {"category": FileCategory.ASSET},
            ".mov": {"category": FileCategory.ASSET},
            ".wmv": {"category": FileCategory.ASSET},
            ".mp3": {"category": FileCategory.ASSET},
            ".wav": {"category": FileCategory.ASSET},
            ".ogg": {"category": FileCategory.ASSET},
            # Assets - Archives
            ".zip": {"category": FileCategory.ASSET},
            ".tar": {"category": FileCategory.ASSET},
            ".gz": {"category": FileCategory.ASSET},
            ".bz2": {"category": FileCategory.ASSET},
            ".7z": {"category": FileCategory.ASSET},
            ".rar": {"category": FileCategory.ASSET},
            # Assets - Documents
            ".pdf": {"category": FileCategory.ASSET},
            ".doc": {"category": FileCategory.ASSET},
            ".docx": {"category": FileCategory.ASSET},
            ".xls": {"category": FileCategory.ASSET},
            ".xlsx": {"category": FileCategory.ASSET},
            ".ppt": {"category": FileCategory.ASSET},
            ".pptx": {"category": FileCategory.ASSET},
            # Assets - Fonts
            ".ttf": {"category": FileCategory.ASSET},
            ".otf": {"category": FileCategory.ASSET},
            ".woff": {"category": FileCategory.ASSET},
            ".woff2": {"category": FileCategory.ASSET},
        }

    def _load_path_patterns(self) -> Dict[str, FileCategory]:
        """Load path pattern to category mapping.

        Returns:
            Dictionary mapping path patterns to categories
        """
        return {
            # Documentation paths
            "readme*": FileCategory.DOCUMENTATION,
            "changelog*": FileCategory.DOCUMENTATION,
            "changes*": FileCategory.DOCUMENTATION,
            "contributing*": FileCategory.DOCUMENTATION,
            "contributors*": FileCategory.DOCUMENTATION,
            "license*": FileCategory.DOCUMENTATION,
            "authors*": FileCategory.DOCUMENTATION,
            "code_of_conduct*": FileCategory.DOCUMENTATION,
            "security*": FileCategory.DOCUMENTATION,
            "docs/": FileCategory.DOCUMENTATION,
            "doc/": FileCategory.DOCUMENTATION,
            "documentation/": FileCategory.DOCUMENTATION,
            # Test paths
            "tests/": FileCategory.TEST_CODE,
            "test/": FileCategory.TEST_CODE,
            "__tests__/": FileCategory.TEST_CODE,
            "spec/": FileCategory.TEST_CODE,
            "specs/": FileCategory.TEST_CODE,
            "*_test.*": FileCategory.TEST_CODE,
            "*_spec.*": FileCategory.TEST_CODE,
            "test_*.*": FileCategory.TEST_CODE,
            "*.test.*": FileCategory.TEST_CODE,
            "*.spec.*": FileCategory.TEST_CODE,
            # Configuration paths
            ".github/": FileCategory.CONFIGURATION,
            ".gitlab/": FileCategory.CONFIGURATION,
            ".circleci/": FileCategory.CONFIGURATION,
            ".travis.yml": FileCategory.CONFIGURATION,
            ".gitlab-ci.yml": FileCategory.CONFIGURATION,
            "azure-pipelines.yml": FileCategory.CONFIGURATION,
            "jenkinsfile": FileCategory.CONFIGURATION,
            ".gitignore": FileCategory.CONFIGURATION,
            ".gitattributes": FileCategory.CONFIGURATION,
            ".dockerignore": FileCategory.CONFIGURATION,
            "dockerfile": FileCategory.CONFIGURATION,
            "docker-compose*": FileCategory.CONFIGURATION,
            "makefile": FileCategory.CONFIGURATION,
            "cmakelists.txt": FileCategory.CONFIGURATION,
            "package.json": FileCategory.CONFIGURATION,
            "package-lock.json": FileCategory.CONFIGURATION,
            "yarn.lock": FileCategory.CONFIGURATION,
            "pnpm-lock.yaml": FileCategory.CONFIGURATION,
            "requirements.txt": FileCategory.CONFIGURATION,
            "pipfile": FileCategory.CONFIGURATION,
            "poetry.lock": FileCategory.CONFIGURATION,
            "pyproject.toml": FileCategory.CONFIGURATION,
            "setup.py": FileCategory.CONFIGURATION,
            "setup.cfg": FileCategory.CONFIGURATION,
            "cargo.toml": FileCategory.CONFIGURATION,
            "cargo.lock": FileCategory.CONFIGURATION,
            "go.mod": FileCategory.CONFIGURATION,
            "go.sum": FileCategory.CONFIGURATION,
            "pom.xml": FileCategory.CONFIGURATION,
            "build.gradle": FileCategory.CONFIGURATION,
            "gradle.properties": FileCategory.CONFIGURATION,
            ".env*": FileCategory.CONFIGURATION,
            "setup.py": FileCategory.CONFIGURATION,
            "setup.cfg": FileCategory.CONFIGURATION,
        }
