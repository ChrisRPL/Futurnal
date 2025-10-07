"""Programming language detection for source code files.

This module implements multi-stage language detection using file extensions
as the primary method and content-based heuristics as fallback.
"""

from pathlib import Path
from typing import Dict, Optional

from .classifier_models import ProgrammingLanguage


def detect_language(
    file_path: str, content: Optional[str] = None
) -> Optional[ProgrammingLanguage]:
    """Detect programming language from file path and optionally content.

    Args:
        file_path: Path to the file
        content: Optional file content for content-based detection

    Returns:
        Detected programming language or None if unknown
    """
    # Try extension-based detection first (most accurate)
    language = _detect_by_extension(file_path)
    if language:
        return language

    # Fallback to content-based detection
    if content:
        return _detect_by_content(content)

    return None


def _detect_by_extension(file_path: str) -> Optional[ProgrammingLanguage]:
    """Detect language based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        Detected programming language or None
    """
    ext = Path(file_path).suffix.lower()

    # Comprehensive extension mapping
    language_map: Dict[str, ProgrammingLanguage] = {
        # Python
        ".py": ProgrammingLanguage.PYTHON,
        ".pyw": ProgrammingLanguage.PYTHON,
        ".pyi": ProgrammingLanguage.PYTHON,
        # JavaScript/TypeScript
        ".js": ProgrammingLanguage.JAVASCRIPT,
        ".mjs": ProgrammingLanguage.JAVASCRIPT,
        ".cjs": ProgrammingLanguage.JAVASCRIPT,
        ".jsx": ProgrammingLanguage.JAVASCRIPT,
        ".ts": ProgrammingLanguage.TYPESCRIPT,
        ".tsx": ProgrammingLanguage.TYPESCRIPT,
        ".mts": ProgrammingLanguage.TYPESCRIPT,
        ".cts": ProgrammingLanguage.TYPESCRIPT,
        # Java/JVM
        ".java": ProgrammingLanguage.JAVA,
        ".kt": ProgrammingLanguage.KOTLIN,
        ".kts": ProgrammingLanguage.KOTLIN,
        ".scala": ProgrammingLanguage.SCALA,
        ".sc": ProgrammingLanguage.SCALA,
        ".groovy": ProgrammingLanguage.GROOVY,
        ".gvy": ProgrammingLanguage.GROOVY,
        ".clj": ProgrammingLanguage.CLOJURE,
        ".cljs": ProgrammingLanguage.CLOJURE,
        ".cljc": ProgrammingLanguage.CLOJURE,
        # Go
        ".go": ProgrammingLanguage.GO,
        # Rust
        ".rs": ProgrammingLanguage.RUST,
        # C/C++
        ".c": ProgrammingLanguage.C,
        ".h": ProgrammingLanguage.C,  # Default to C, could be C++
        ".cpp": ProgrammingLanguage.CPP,
        ".cc": ProgrammingLanguage.CPP,
        ".cxx": ProgrammingLanguage.CPP,
        ".c++": ProgrammingLanguage.CPP,
        ".hpp": ProgrammingLanguage.CPP,
        ".hh": ProgrammingLanguage.CPP,
        ".hxx": ProgrammingLanguage.CPP,
        ".h++": ProgrammingLanguage.CPP,
        # C#
        ".cs": ProgrammingLanguage.CSHARP,
        ".csx": ProgrammingLanguage.CSHARP,
        # Swift
        ".swift": ProgrammingLanguage.SWIFT,
        # Objective-C
        ".m": ProgrammingLanguage.OBJECTIVE_C,
        ".mm": ProgrammingLanguage.OBJECTIVE_C,
        # Dart
        ".dart": ProgrammingLanguage.DART,
        # PHP
        ".php": ProgrammingLanguage.PHP,
        ".php3": ProgrammingLanguage.PHP,
        ".php4": ProgrammingLanguage.PHP,
        ".php5": ProgrammingLanguage.PHP,
        ".phtml": ProgrammingLanguage.PHP,
        # Ruby
        ".rb": ProgrammingLanguage.RUBY,
        ".rake": ProgrammingLanguage.RUBY,
        ".gemspec": ProgrammingLanguage.RUBY,
        # Perl
        ".pl": ProgrammingLanguage.PERL,
        ".pm": ProgrammingLanguage.PERL,
        ".pod": ProgrammingLanguage.PERL,
        # Lua
        ".lua": ProgrammingLanguage.LUA,
        # Assembly
        ".asm": ProgrammingLanguage.ASSEMBLY,
        ".s": ProgrammingLanguage.ASSEMBLY,
        ".S": ProgrammingLanguage.ASSEMBLY,
        # Fortran
        ".f": ProgrammingLanguage.FORTRAN,
        ".f90": ProgrammingLanguage.FORTRAN,
        ".f95": ProgrammingLanguage.FORTRAN,
        ".for": ProgrammingLanguage.FORTRAN,
        # Haskell
        ".hs": ProgrammingLanguage.HASKELL,
        ".lhs": ProgrammingLanguage.HASKELL,
        # OCaml
        ".ml": ProgrammingLanguage.OCAML,
        ".mli": ProgrammingLanguage.OCAML,
        # Elixir
        ".ex": ProgrammingLanguage.ELIXIR,
        ".exs": ProgrammingLanguage.ELIXIR,
        # Erlang
        ".erl": ProgrammingLanguage.ERLANG,
        ".hrl": ProgrammingLanguage.ERLANG,
        # F#
        ".fs": ProgrammingLanguage.F_SHARP,
        ".fsi": ProgrammingLanguage.F_SHARP,
        ".fsx": ProgrammingLanguage.F_SHARP,
        # R
        ".r": ProgrammingLanguage.R,
        ".R": ProgrammingLanguage.R,
        ".rmd": ProgrammingLanguage.R,
        ".Rmd": ProgrammingLanguage.R,
        # Julia
        ".jl": ProgrammingLanguage.JULIA,
        # MATLAB
        ".m": ProgrammingLanguage.MATLAB,  # Conflicts with Objective-C
        ".mat": ProgrammingLanguage.MATLAB,
        # Shell scripts
        ".sh": ProgrammingLanguage.BASH,
        ".bash": ProgrammingLanguage.BASH,
        ".zsh": ProgrammingLanguage.SHELL,
        ".fish": ProgrammingLanguage.SHELL,
        ".ksh": ProgrammingLanguage.SHELL,
        ".csh": ProgrammingLanguage.SHELL,
        ".ps1": ProgrammingLanguage.POWERSHELL,
        ".psm1": ProgrammingLanguage.POWERSHELL,
        # Web
        ".html": ProgrammingLanguage.HTML,
        ".htm": ProgrammingLanguage.HTML,
        ".css": ProgrammingLanguage.CSS,
        ".scss": ProgrammingLanguage.CSS,
        ".sass": ProgrammingLanguage.CSS,
        ".less": ProgrammingLanguage.CSS,
        # SQL
        ".sql": ProgrammingLanguage.SQL,
        # GraphQL
        ".graphql": ProgrammingLanguage.GRAPHQL,
        ".gql": ProgrammingLanguage.GRAPHQL,
        # Protocol Buffers
        ".proto": ProgrammingLanguage.PROTO,
        # Thrift
        ".thrift": ProgrammingLanguage.THRIFT,
        # Solidity
        ".sol": ProgrammingLanguage.SOLIDITY,
        # Verilog/VHDL
        ".v": ProgrammingLanguage.VERILOG,
        ".sv": ProgrammingLanguage.VERILOG,
        ".vhd": ProgrammingLanguage.VHDL,
        ".vhdl": ProgrammingLanguage.VHDL,
        # LaTeX
        ".tex": ProgrammingLanguage.LATEX,
        # Markdown (for syntax highlighting)
        ".md": ProgrammingLanguage.MARKDOWN,
        ".markdown": ProgrammingLanguage.MARKDOWN,
    }

    return language_map.get(ext)


def _detect_by_content(content: str) -> Optional[ProgrammingLanguage]:
    """Detect language based on file content heuristics.

    Args:
        content: File content as string

    Returns:
        Detected programming language or None
    """
    # Check shebang line first
    if content.startswith("#!"):
        first_line = content.split("\n")[0].lower()

        if "python" in first_line:
            return ProgrammingLanguage.PYTHON
        elif "node" in first_line or "javascript" in first_line:
            return ProgrammingLanguage.JAVASCRIPT
        elif "ruby" in first_line:
            return ProgrammingLanguage.RUBY
        elif "perl" in first_line:
            return ProgrammingLanguage.PERL
        elif "bash" in first_line or "/sh" in first_line:
            return ProgrammingLanguage.BASH
        elif "php" in first_line:
            return ProgrammingLanguage.PHP

    # Use keyword-based detection (first 1000 chars)
    sample = content[:1000]

    # Python detection
    if "def " in sample and ("import " in sample or "from " in sample):
        return ProgrammingLanguage.PYTHON

    # JavaScript detection
    if (
        "function " in sample
        or "const " in sample
        or "let " in sample
        or "var " in sample
        or "console.log" in sample
    ):
        # Additional JavaScript indicators
        if "=>" in sample or "console." in sample or "{" in sample:
            return ProgrammingLanguage.JAVASCRIPT

    # TypeScript detection (requires type annotations)
    if (
        ("function " in sample or "const " in sample)
        and ": " in sample
        and ("interface " in sample or "type " in sample)
    ):
        return ProgrammingLanguage.TYPESCRIPT

    # Java detection
    if "public class " in sample and "import java." in sample:
        return ProgrammingLanguage.JAVA

    # Go detection
    if "package " in sample and "func " in sample:
        return ProgrammingLanguage.GO

    # Rust detection
    if "fn " in sample and ("use " in sample or "impl " in sample):
        return ProgrammingLanguage.RUST

    # C/C++ detection
    if "#include " in sample:
        # Check for C++ indicators
        if (
            "std::" in sample
            or "namespace " in sample
            or "class " in sample
            or "template " in sample
        ):
            return ProgrammingLanguage.CPP
        return ProgrammingLanguage.C

    # Ruby detection
    if ("def " in sample or "class " in sample) and ("end" in sample):
        return ProgrammingLanguage.RUBY

    # PHP detection
    if "<?php" in sample:
        return ProgrammingLanguage.PHP

    # No confident detection
    return None
