"""Data models for GitHub repository file classification.

This module defines the classification taxonomy and result structures for routing
repository files to appropriate processing pipelines.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class FileCategory(str, Enum):
    """Semantic category of a repository file."""

    SOURCE_CODE = "source_code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    TEST_CODE = "test_code"
    ASSET = "asset"
    UNKNOWN = "unknown"


class ProgrammingLanguage(str, Enum):
    """Programming language detected for source code files."""

    # Popular languages (top tier)
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"

    # JVM languages
    KOTLIN = "kotlin"
    SCALA = "scala"
    GROOVY = "groovy"
    CLOJURE = "clojure"

    # Mobile
    SWIFT = "swift"
    OBJECTIVE_C = "objective_c"
    DART = "dart"

    # Web/scripting
    PHP = "php"
    RUBY = "ruby"
    PERL = "perl"
    LUA = "lua"

    # Systems
    ASSEMBLY = "assembly"
    FORTRAN = "fortran"

    # Functional
    HASKELL = "haskell"
    OCAML = "ocaml"
    ELIXIR = "elixir"
    ERLANG = "erlang"
    F_SHARP = "fsharp"

    # Data science
    R = "r"
    JULIA = "julia"
    MATLAB = "matlab"

    # Shell
    BASH = "bash"
    SHELL = "shell"
    POWERSHELL = "powershell"

    # Other
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    GRAPHQL = "graphql"
    PROTO = "protobuf"
    THRIFT = "thrift"
    SOLIDITY = "solidity"
    VERILOG = "verilog"
    VHDL = "vhdl"
    LATEX = "latex"
    MARKDOWN = "markdown"


class FileClassification(BaseModel):
    """Classification result for a repository file.

    Attributes:
        file_path: Relative path within the repository
        category: Semantic category (code, docs, config, etc.)
        language: Programming language (if applicable)
        confidence: Classification confidence score (0.0-1.0)
        classified_by: Method used for classification
        classification_metadata: Additional classification context
        is_binary: Whether the file is binary
        size_bytes: File size in bytes
        line_count: Number of lines (text files only)
        has_secrets: Whether secret patterns were detected
        should_embed: Whether file should be embedded
        embedding_type: Type of embedding to use
        should_parse: Whether file should be parsed for entities
    """

    file_path: str
    category: FileCategory
    language: Optional[ProgrammingLanguage] = None
    confidence: float = Field(ge=0.0, le=1.0)

    # Classification rationale
    classified_by: str  # "extension", "path", "content", "codebert"
    classification_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Content characteristics
    is_binary: bool = False
    size_bytes: int = Field(ge=0)
    line_count: Optional[int] = Field(default=None, ge=0)
    has_secrets: bool = False

    # Processing directives
    should_embed: bool = True
    embedding_type: str = "standard"  # "standard", "codebert", "none"
    should_parse: bool = True

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
