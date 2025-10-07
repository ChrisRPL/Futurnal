# GitHub File Classifier

A production-ready file classification system for routing repository files to appropriate processing pipelines.

## Quick Start

```python
from futurnal.ingestion.github import FileClassifier, FileCategory

# Create classifier
classifier = FileClassifier()

# Classify a file
result = classifier.classify_file("src/main.py")

print(f"Category: {result.category}")           # SOURCE_CODE
print(f"Language: {result.language}")           # PYTHON
print(f"Confidence: {result.confidence}")       # 0.9
print(f"Embedding Type: {result.embedding_type}") # codebert
print(f"Has Secrets: {result.has_secrets}")     # False
```

## Features

### Multi-Stage Classification

1. **Extension-based** (confidence: 0.9)
   - Fast O(1) lookup
   - 500+ file extensions mapped
   - Highest confidence for known types

2. **Path-based** (confidence: 0.75)
   - Pattern matching for special paths
   - Tests: `tests/`, `*_test.*`, `*.spec.*`
   - Docs: `README*`, `docs/`, `CHANGELOG*`
   - Config: `.github/`, `setup.py`, `package.json`

3. **Content-based** (variable confidence)
   - Fallback when extension/path unknown
   - Shebang detection: `#!/usr/bin/python3`
   - Keyword matching for major languages

### File Categories

- **SOURCE_CODE**: Programming language files → CodeBERT embeddings
- **DOCUMENTATION**: Markdown, text, RST → Standard embeddings
- **CONFIGURATION**: JSON, YAML, TOML, env files → Dependency extraction
- **TEST_CODE**: Test files → CodeBERT + coverage analysis
- **ASSET**: Images, videos, archives → Metadata only
- **UNKNOWN**: Unrecognized files → Content analysis

### Language Detection

Supports 45+ programming languages:

- **Popular**: Python, JavaScript, TypeScript, Java, Go, Rust, C/C++
- **JVM**: Kotlin, Scala, Groovy, Clojure
- **Mobile**: Swift, Objective-C, Dart
- **Web**: PHP, Ruby, HTML, CSS, SQL
- **Data Science**: R, Julia, MATLAB
- **Systems**: Assembly, Fortran, Verilog, VHDL
- **Functional**: Haskell, OCaml, Elixir, Erlang, F#

### Secret Detection

Detects 40+ secret patterns:

- API keys (generic, AWS, Google Cloud, Stripe, etc.)
- Passwords and tokens
- GitHub tokens (PAT, OAuth, refresh)
- Private keys (RSA, OpenSSH)
- Cloud credentials (AWS, Azure, SendGrid, Mailgun)
- JWT tokens
- Database connection strings

## Usage Examples

### Basic Classification

```python
classifier = FileClassifier()

# Python source file
result = classifier.classify_file("backend/api.py")
# → category: SOURCE_CODE, language: PYTHON, embedding_type: codebert

# Documentation
result = classifier.classify_file("README.md")
# → category: DOCUMENTATION, embedding_type: standard

# Configuration with secrets
config = b'{"api_key": "sk_live_abc123xyz..."}'
result = classifier.classify_file("config.json", config)
# → category: CONFIGURATION, has_secrets: True
```

### Language Detection Only

```python
from futurnal.ingestion.github import detect_language

# By extension
lang = detect_language("app.ts")
# → ProgrammingLanguage.TYPESCRIPT

# By content
code = """
#!/usr/bin/env python3
def main():
    import sys
    print("Hello")
"""
lang = detect_language("script", code)
# → ProgrammingLanguage.PYTHON
```

### Secret Detection Only

```python
from futurnal.ingestion.github import SecretDetector

detector = SecretDetector()

# Check for secrets
config = b'AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE'
has_secrets = detector.detect(config)
# → True

# Clean file
code = b'def calculate(x, y): return x + y'
has_secrets = detector.detect(code)
# → False
```

### Advanced: Content-Based Classification

```python
# File without extension
classifier = FileClassifier()

script_content = b"""
#!/usr/bin/env python3
import sys

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
"""

result = classifier.classify_file("deploy_script", script_content)
# → category: SOURCE_CODE, language: PYTHON (detected from shebang + keywords)
```

### Batch Processing

```python
classifier = FileClassifier()

files = [
    "src/main.py",
    "tests/test_api.py",
    "README.md",
    "package.json",
    "assets/logo.png",
]

for file_path in files:
    result = classifier.classify_file(file_path)
    print(f"{file_path} → {result.category} ({result.embedding_type})")

# Output:
# src/main.py → SOURCE_CODE (codebert)
# tests/test_api.py → TEST_CODE (codebert)
# README.md → DOCUMENTATION (standard)
# package.json → CONFIGURATION (standard)
# assets/logo.png → ASSET (none)
```

## Processing Directives

The classifier provides directives for downstream processing:

```python
result = classifier.classify_file("data.bin", b"\x00\x01\xff\xfe")

# Check if file should be embedded
if result.should_embed:
    embed_content(result.file_path, result.embedding_type)
else:
    # Binary or asset - skip embedding
    store_metadata_only(result)

# Check if file should be parsed
if result.should_parse:
    extract_entities(result.file_path)

# Check for secrets
if result.has_secrets:
    flag_for_privacy_review(result.file_path)
```

## Priority Logic

The classifier uses smart priority rules:

1. **Test files**: Path patterns always win
   - `tests/test_api.py` → TEST_CODE (not SOURCE_CODE)

2. **Special config files**: Path patterns win
   - `setup.py` → CONFIGURATION (not SOURCE_CODE)
   - `requirements.txt` → CONFIGURATION (not DOCUMENTATION)

3. **Regular files**: Extension wins
   - `docs/example.py` → SOURCE_CODE (extension overrides docs/ path)

## Configuration

```python
# Enable CodeBERT for content classification (future enhancement)
classifier = FileClassifier(use_codebert=True)

# Custom secret detector
from futurnal.ingestion.github import SecretDetector

custom_detector = SecretDetector()
classifier = FileClassifier(secret_detector=custom_detector)
```

## Architecture

### Classification Pipeline

```
File Path + Content
        ↓
    ┌─────────────────┐
    │ Binary Check    │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Path Patterns   │ ← Test files, special configs
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Extension Map   │ ← 500+ extensions
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Content Analysis│ ← Shebang, keywords
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Secret Detection│ ← If category = CONFIG
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Embedding Type  │ ← codebert/standard/none
    └────────┬────────┘
             ↓
    FileClassification
```

### Data Flow

```python
# Input
file_path: str
content: Optional[bytes]

# Output
FileClassification(
    file_path=str,
    category=FileCategory,
    language=Optional[ProgrammingLanguage],
    confidence=float,  # 0.0-1.0
    classified_by=str,  # "extension", "path", "content", "fallback"
    is_binary=bool,
    size_bytes=int,
    line_count=Optional[int],
    has_secrets=bool,
    should_embed=bool,
    embedding_type=str,  # "codebert", "standard", "none"
    should_parse=bool,
)
```

## Performance

- **Extension-based**: <1ms (O(1) lookup)
- **Path-based**: <5ms (O(n) pattern matching)
- **Content-based**: <10ms (1KB analysis)
- **Secret detection**: <20ms (40 regex patterns)

## Testing

Run the test suite:

```bash
# All classifier tests
pytest tests/ingestion/github/test_file_classifier.py -v

# Specific test class
pytest tests/ingestion/github/test_file_classifier.py::TestFileClassifier -v

# Coverage report
pytest tests/ingestion/github/test_file_classifier.py --cov=futurnal.ingestion.github
```

## Integration

### With GitHub Sync Pipeline

```python
from futurnal.ingestion.github import FileClassifier, FileCategory

classifier = FileClassifier()

def process_repository_file(file_path: str, content: bytes):
    # Classify file
    classification = classifier.classify_file(file_path, content)

    # Route to appropriate pipeline
    if classification.category == FileCategory.SOURCE_CODE:
        # Code-specific processing
        extract_functions(content, classification.language)
        embed_with_codebert(content)

    elif classification.category == FileCategory.DOCUMENTATION:
        # Documentation processing
        parse_markdown(content)
        embed_with_standard_model(content)

    elif classification.category == FileCategory.CONFIGURATION:
        # Config processing
        extract_dependencies(content)
        if classification.has_secrets:
            flag_for_review(file_path)

    elif classification.category == FileCategory.ASSET:
        # Metadata only
        store_asset_metadata(file_path, classification.size_bytes)
```

## Dependencies

- `pathlib` (stdlib): Path operations
- `fnmatch` (stdlib): Pattern matching
- `re` (stdlib): Regex for secret detection
- `pydantic`: Data validation

Optional:
- `transformers` + `torch`: For CodeBERT content classification (future)

## License

Part of Futurnal Phase 1 (Archivist) - GitHub Connector
