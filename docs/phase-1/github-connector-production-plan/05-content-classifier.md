Summary: Implement file classification to route code, documentation, and configuration files appropriately.

# 05 · Content Classifier

## Purpose
Classify repository files into semantic categories (source code, documentation, configuration, assets) to enable appropriate processing pathways. Code files receive code-specific embeddings (CodeBERT), documentation uses standard embeddings, and configurations are analyzed for dependencies and settings.

## Scope
- File type classification (code, docs, config, assets, tests)
- Programming language detection for code files
- Heuristic-based classification (file extensions, paths)
- CodeBERT-based semantic classification for ambiguous cases
- Documentation quality assessment
- Binary file detection and exclusion
- Secret detection in configuration files
- Classification metadata for PKG enrichment

## Requirements Alignment
- **Appropriate processing**: Route files to correct embedding/analysis pipeline
- **Code understanding**: Specialized embeddings for source code
- **Documentation extraction**: README, wiki, inline docs → documentation nodes
- **Security**: Detect and exclude sensitive files (secrets, credentials)
- **Performance**: Fast classification for large repositories

## File Categories

### Source Code
**Extensions**: `.py`, `.js`, `.ts`, `.java`, `.go`, `.rb`, `.rs`, `.cpp`, `.c`, `.h`, `.swift`, `.kt`, `.scala`, `.php`, `.cs`

**Processing**:
- Language-specific parsing
- CodeBERT embeddings for semantic understanding
- Function/class extraction
- Import dependency analysis
- Inline comment extraction

### Documentation
**Extensions**: `.md`, `.rst`, `.txt`, `.adoc`, `.org`
**Paths**: `docs/`, `README*`, `CONTRIBUTING*`, `LICENSE*`, `CHANGELOG*`

**Processing**:
- Standard text embeddings
- Markdown structure parsing
- Documentation quality scoring
- Cross-reference extraction
- Version-specific documentation tagging

### Configuration
**Extensions**: `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.conf`, `.xml`, `.env`
**Files**: `package.json`, `pyproject.toml`, `Cargo.toml`, `pom.xml`, `.github/workflows/`

**Processing**:
- Dependency extraction
- Configuration schema validation
- Secret pattern detection
- Version constraint analysis

### Test Code
**Paths**: `tests/`, `test/`, `__tests__/`, `spec/`, `**/*_test.*`, `**/*Test.*`

**Processing**:
- Separate from main code
- Test coverage analysis
- Test framework detection

### Assets (Excluded from Content)
**Extensions**: `.jpg`, `.png`, `.gif`, `.svg`, `.mp4`, `.mp3`, `.pdf`, `.zip`, `.tar`, `.gz`

**Processing**:
- Metadata only (size, type, path)
- No content embedding
- Asset inventory for PKG

## Data Model

### FileClassification
```python
class FileCategory(str, Enum):
    SOURCE_CODE = "source_code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    TEST_CODE = "test_code"
    ASSET = "asset"
    UNKNOWN = "unknown"

class ProgrammingLanguage(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "csharp"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    # ... more languages

class FileClassification(BaseModel):
    """Classification result for a file."""

    file_path: str
    category: FileCategory
    language: Optional[ProgrammingLanguage] = None
    confidence: float  # 0.0-1.0

    # Classification rationale
    classified_by: str  # "extension", "path", "content", "codebert"
    classification_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Content characteristics
    is_binary: bool = False
    size_bytes: int
    line_count: Optional[int] = None
    has_secrets: bool = False  # Secret patterns detected

    # Processing directives
    should_embed: bool = True
    embedding_type: str = "standard"  # "standard", "codebert", "none"
    should_parse: bool = True
```

## Component Design

### FileClassifier
```python
class FileClassifier:
    """Classifies repository files into categories."""

    def __init__(
        self,
        *,
        use_codebert: bool = False,  # Expensive, use only if needed
        secret_detector: Optional[SecretDetector] = None,
    ):
        self.use_codebert = use_codebert
        self.secret_detector = secret_detector or SecretDetector()

        # Load classification rules
        self.extension_map = self._load_extension_map()
        self.path_patterns = self._load_path_patterns()

    def classify_file(
        self,
        file_path: str,
        content: Optional[bytes] = None,
    ) -> FileClassification:
        """Classify a single file."""

        # Check if binary
        is_binary = self._is_binary(content) if content else False

        # Try extension-based classification first (fast)
        extension_result = self._classify_by_extension(file_path)
        if extension_result and extension_result.confidence > 0.8:
            classification = extension_result
        else:
            # Try path-based classification
            path_result = self._classify_by_path(file_path)
            if path_result and path_result.confidence > 0.7:
                classification = path_result
            else:
                # Fallback to content-based (expensive)
                if content and not is_binary and self.use_codebert:
                    classification = self._classify_by_content(file_path, content)
                else:
                    classification = extension_result or path_result or FileClassification(
                        file_path=file_path,
                        category=FileCategory.UNKNOWN,
                        confidence=0.0,
                        classified_by="fallback",
                        size_bytes=len(content) if content else 0,
                        is_binary=is_binary,
                    )

        # Detect secrets if configuration
        if classification.category == FileCategory.CONFIGURATION and content:
            classification.has_secrets = self.secret_detector.detect(content)

        # Set embedding strategy
        classification.embedding_type = self._determine_embedding_type(classification)

        return classification

    def _classify_by_extension(
        self,
        file_path: str,
    ) -> Optional[FileClassification]:
        """Classify based on file extension."""
        ext = Path(file_path).suffix.lower()

        if ext in self.extension_map:
            mapping = self.extension_map[ext]
            return FileClassification(
                file_path=file_path,
                category=mapping["category"],
                language=mapping.get("language"),
                confidence=0.9,
                classified_by="extension",
                size_bytes=0,  # Set later
            )

        return None

    def _classify_by_path(
        self,
        file_path: str,
    ) -> Optional[FileClassification]:
        """Classify based on file path patterns."""
        path_lower = file_path.lower()

        for pattern, category in self.path_patterns.items():
            if pattern in path_lower or fnmatch(path_lower, pattern):
                return FileClassification(
                    file_path=file_path,
                    category=category,
                    confidence=0.75,
                    classified_by="path",
                    size_bytes=0,
                )

        return None

    def _classify_by_content(
        self,
        file_path: str,
        content: bytes,
    ) -> FileClassification:
        """Classify based on file content using CodeBERT."""
        # Use CodeBERT to analyze content
        # This is expensive, only used when other methods fail

        text = content.decode('utf-8', errors='ignore')[:1000]  # First 1KB

        # CodeBERT inference (pseudo-code)
        prediction = self._codebert_classify(text)

        return FileClassification(
            file_path=file_path,
            category=prediction["category"],
            language=prediction.get("language"),
            confidence=prediction["confidence"],
            classified_by="codebert",
            size_bytes=len(content),
        )

    def _determine_embedding_type(
        self,
        classification: FileClassification,
    ) -> str:
        """Determine which embedding model to use."""

        if classification.is_binary:
            return "none"

        if classification.category == FileCategory.SOURCE_CODE:
            return "codebert"  # Code-specific embeddings

        if classification.category == FileCategory.DOCUMENTATION:
            return "standard"  # Standard text embeddings

        if classification.category == FileCategory.CONFIGURATION:
            return "standard"  # Config as structured text

        if classification.category == FileCategory.ASSET:
            return "none"

        return "standard"  # Default

    def _is_binary(self, content: Optional[bytes]) -> bool:
        """Detect if content is binary."""
        if not content:
            return False

        # Check first 8KB for null bytes
        sample = content[:8192]
        return b'\x00' in sample

    def _load_extension_map(self) -> Dict[str, Dict]:
        """Load file extension to category mapping."""
        return {
            # Source code
            ".py": {"category": FileCategory.SOURCE_CODE, "language": ProgrammingLanguage.PYTHON},
            ".js": {"category": FileCategory.SOURCE_CODE, "language": ProgrammingLanguage.JAVASCRIPT},
            ".ts": {"category": FileCategory.SOURCE_CODE, "language": ProgrammingLanguage.TYPESCRIPT},
            ".java": {"category": FileCategory.SOURCE_CODE, "language": ProgrammingLanguage.JAVA},
            ".go": {"category": FileCategory.SOURCE_CODE, "language": ProgrammingLanguage.GO},
            ".rs": {"category": FileCategory.SOURCE_CODE, "language": ProgrammingLanguage.RUST},
            # Documentation
            ".md": {"category": FileCategory.DOCUMENTATION},
            ".rst": {"category": FileCategory.DOCUMENTATION},
            ".txt": {"category": FileCategory.DOCUMENTATION},
            # Configuration
            ".json": {"category": FileCategory.CONFIGURATION},
            ".yaml": {"category": FileCategory.CONFIGURATION},
            ".yml": {"category": FileCategory.CONFIGURATION},
            ".toml": {"category": FileCategory.CONFIGURATION},
            # Assets
            ".jpg": {"category": FileCategory.ASSET},
            ".png": {"category": FileCategory.ASSET},
            ".gif": {"category": FileCategory.ASSET},
            ".svg": {"category": FileCategory.ASSET},
            # ... more extensions
        }

    def _load_path_patterns(self) -> Dict[str, FileCategory]:
        """Load path patterns to category mapping."""
        return {
            "readme*": FileCategory.DOCUMENTATION,
            "changelog*": FileCategory.DOCUMENTATION,
            "contributing*": FileCategory.DOCUMENTATION,
            "license*": FileCategory.DOCUMENTATION,
            "docs/": FileCategory.DOCUMENTATION,
            "tests/": FileCategory.TEST_CODE,
            "test/": FileCategory.TEST_CODE,
            "__tests__/": FileCategory.TEST_CODE,
            "spec/": FileCategory.TEST_CODE,
            ".github/workflows/": FileCategory.CONFIGURATION,
            # ... more patterns
        }
```

### SecretDetector
```python
class SecretDetector:
    """Detects secrets and sensitive data in files."""

    def __init__(self):
        self.secret_patterns = self._load_secret_patterns()

    def detect(self, content: bytes) -> bool:
        """Detect if content contains secrets."""
        text = content.decode('utf-8', errors='ignore')

        for pattern in self.secret_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _load_secret_patterns(self) -> List[re.Pattern]:
        """Load regex patterns for secret detection."""
        return [
            re.compile(r"(?i)(api[_-]?key|apikey)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})"),
            re.compile(r"(?i)(password|passwd|pwd)[\s]*[=:]+[\s]*['\"]?([^\s'\"]{8,})"),
            re.compile(r"(?i)(token)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})"),
            re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
            re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub Personal Access Token
            re.compile(r"gho_[a-zA-Z0-9]{36}"),  # GitHub OAuth Token
            # ... more patterns
        ]
```

## Language Detection

### Programming Language Detection
```python
def detect_language(file_path: str, content: Optional[str] = None) -> Optional[ProgrammingLanguage]:
    """Detect programming language from file."""

    # Extension-based detection (most accurate)
    ext = Path(file_path).suffix.lower()
    language_map = {
        ".py": ProgrammingLanguage.PYTHON,
        ".js": ProgrammingLanguage.JAVASCRIPT,
        ".ts": ProgrammingLanguage.TYPESCRIPT,
        ".java": ProgrammingLanguage.JAVA,
        ".go": ProgrammingLanguage.GO,
        # ... more mappings
    }

    if ext in language_map:
        return language_map[ext]

    # Content-based detection (fallback)
    if content:
        # Check shebang
        if content.startswith("#!"):
            first_line = content.split("\n")[0]
            if "python" in first_line:
                return ProgrammingLanguage.PYTHON
            if "node" in first_line or "javascript" in first_line:
                return ProgrammingLanguage.JAVASCRIPT

        # Check for language-specific keywords
        if "def " in content and "import " in content:
            return ProgrammingLanguage.PYTHON
        if "function " in content and "const " in content:
            return ProgrammingLanguage.JAVASCRIPT

    return None
```

## CodeBERT Integration (Optional)

### CodeBERT Classification
```python
from transformers import AutoTokenizer, AutoModel
import torch

class CodeBERTClassifier:
    """Optional CodeBERT-based file classification."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

    def classify(self, code_snippet: str) -> Dict[str, Any]:
        """Classify code snippet using CodeBERT."""
        # Tokenize
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Classification logic (simplified)
        # In practice, you'd have a trained classifier on top
        return {
            "category": FileCategory.SOURCE_CODE,
            "language": self._detect_language_from_embedding(embeddings),
            "confidence": 0.85,
        }
```

## Acceptance Criteria

- ✅ Extension-based classification accuracy > 95% for common file types
- ✅ Path-based classification correctly identifies docs/, tests/, etc.
- ✅ Binary file detection prevents content processing
- ✅ Secret detection flags configuration files with sensitive data
- ✅ Language detection accurate for major programming languages
- ✅ Embedding type assignment appropriate for file category
- ✅ Test files correctly separated from main code
- ✅ Documentation files (README, CHANGELOG) classified correctly
- ✅ Configuration files parsed for dependencies
- ✅ Asset files (images, videos) excluded from content processing

## Test Plan

### Unit Tests
- Extension mapping coverage
- Path pattern matching
- Binary file detection
- Secret pattern matching
- Language detection accuracy
- Embedding type assignment logic

### Integration Tests
- Classification of entire test repository
- Ambiguous file handling (no extension)
- Mixed content files
- Large file classification (>10MB)
- Non-UTF-8 encoded files

### Accuracy Tests
- Measure classification accuracy on labeled dataset
- Language detection accuracy
- False positive rate for secret detection
- Binary vs text detection accuracy

## Implementation Notes

### Extension Map Completeness
- Cover top 50 programming languages
- Include major config formats
- Handle ambiguous extensions (.h for C/C++)

### Performance Optimization
- Cache classification results
- Lazy-load CodeBERT model (only if needed)
- Batch processing for repository scans

## Open Questions

- Should we use pygments for language detection?
- Should we analyze file content for documentation quality scoring?
- How to handle Jupyter notebooks (.ipynb)?
- Should we detect generated code (protobuf, OpenAPI)?
- How to classify polyglot files (HTML with inline JS/CSS)?

## Dependencies
- `pathlib` for path operations
- `fnmatch` for pattern matching
- `re` for regex-based detection
- (Optional) `transformers` for CodeBERT (`pip install transformers torch`)
- (Optional) `pygments` for language detection (`pip install pygments`)


