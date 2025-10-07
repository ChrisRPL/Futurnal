"""Tests for GitHub file classification system.

This module tests the content classifier components including file categorization,
language detection, secret detection, and embedding type assignment.
"""

import pytest

from futurnal.ingestion.github import (
    FileCategory,
    FileClassification,
    FileClassifier,
    ProgrammingLanguage,
    SecretDetector,
    detect_language,
)


class TestFileCategory:
    """Tests for FileCategory enum."""

    def test_all_categories_exist(self):
        """Test that all expected categories are defined."""
        expected = {
            "source_code",
            "documentation",
            "configuration",
            "test_code",
            "asset",
            "unknown",
        }
        actual = {cat.value for cat in FileCategory}
        assert actual == expected


class TestProgrammingLanguage:
    """Tests for ProgrammingLanguage enum."""

    def test_major_languages_exist(self):
        """Test that major programming languages are defined."""
        major_languages = {
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "cpp",
            "c",
            "csharp",
        }
        actual = {lang.value for lang in ProgrammingLanguage}
        assert major_languages.issubset(actual)

    def test_language_count(self):
        """Test that we have a comprehensive language set."""
        # Should have at least 30 languages
        assert len(list(ProgrammingLanguage)) >= 30


class TestSecretDetector:
    """Tests for secret detection."""

    def test_detect_api_key(self):
        """Test detection of API keys."""
        detector = SecretDetector()
        content = b"API_KEY=sk_test_abcdefghijklmnopqrstuvwxyz123456"
        assert detector.detect(content) is True

    def test_detect_password(self):
        """Test detection of passwords."""
        detector = SecretDetector()
        content = b"password=super_secret_password_123"
        assert detector.detect(content) is True

    def test_detect_github_token(self):
        """Test detection of GitHub tokens."""
        detector = SecretDetector()
        content = b"token: ghp_1234567890abcdefghijklmnopqrstuvwxyz"
        assert detector.detect(content) is True

    def test_detect_aws_key(self):
        """Test detection of AWS credentials."""
        detector = SecretDetector()
        content = b"AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        assert detector.detect(content) is True

    def test_detect_private_key(self):
        """Test detection of private keys."""
        detector = SecretDetector()
        content = b"-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK..."
        assert detector.detect(content) is True

    def test_detect_jwt_token(self):
        """Test detection of JWT tokens."""
        detector = SecretDetector()
        content = b"Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert detector.detect(content) is True

    def test_no_secrets_in_clean_code(self):
        """Test that clean code is not flagged."""
        detector = SecretDetector()
        content = b"""
def calculate_sum(a, b):
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a + b

if __name__ == "__main__":
    result = calculate_sum(5, 3)
    print(f"Result: {result}")
"""
        assert detector.detect(content) is False

    def test_no_secrets_in_documentation(self):
        """Test that documentation is not flagged."""
        detector = SecretDetector()
        content = b"""
# API Documentation

To use the API, you need an API key. You can get one from the dashboard.

Example:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com
```
"""
        assert detector.detect(content) is False

    def test_binary_content_handling(self):
        """Test that binary content doesn't cause errors."""
        detector = SecretDetector()
        content = b"\x00\x01\x02\x03\x04\x05\xff\xfe"
        assert detector.detect(content) is False


class TestLanguageDetector:
    """Tests for programming language detection."""

    def test_detect_python_by_extension(self):
        """Test Python detection by .py extension."""
        assert detect_language("script.py") == ProgrammingLanguage.PYTHON
        assert detect_language("module.pyw") == ProgrammingLanguage.PYTHON

    def test_detect_javascript_by_extension(self):
        """Test JavaScript detection by extension."""
        assert detect_language("app.js") == ProgrammingLanguage.JAVASCRIPT
        assert detect_language("module.mjs") == ProgrammingLanguage.JAVASCRIPT

    def test_detect_typescript_by_extension(self):
        """Test TypeScript detection by extension."""
        assert detect_language("app.ts") == ProgrammingLanguage.TYPESCRIPT
        assert detect_language("component.tsx") == ProgrammingLanguage.TYPESCRIPT

    def test_detect_python_by_shebang(self):
        """Test Python detection by shebang."""
        content = "#!/usr/bin/env python3\nprint('hello')"
        assert detect_language("script", content) == ProgrammingLanguage.PYTHON

    def test_detect_python_by_content(self):
        """Test Python detection by keywords."""
        content = """
def main():
    import sys
    print("Hello, World!")
"""
        assert detect_language("unknown_file", content) == ProgrammingLanguage.PYTHON

    def test_detect_javascript_by_content(self):
        """Test JavaScript detection by keywords."""
        content = """
function hello() {
    const message = "Hello, World!";
    console.log(message);
}
"""
        assert detect_language("script", content) == ProgrammingLanguage.JAVASCRIPT

    def test_detect_go_by_content(self):
        """Test Go detection by keywords."""
        content = """
package main

func main() {
    fmt.Println("Hello, World!")
}
"""
        assert detect_language("main", content) == ProgrammingLanguage.GO

    def test_unknown_language(self):
        """Test handling of unknown languages."""
        assert detect_language("file.xyz") is None
        assert detect_language("unknown", "random content") is None


class TestFileClassifier:
    """Tests for file classification."""

    def test_classify_python_file(self):
        """Test classification of Python source file."""
        classifier = FileClassifier()
        result = classifier.classify_file("src/main.py")

        assert result.category == FileCategory.SOURCE_CODE
        assert result.language == ProgrammingLanguage.PYTHON
        assert result.confidence == 0.9
        assert result.classified_by == "extension"
        assert result.embedding_type == "codebert"
        assert result.should_embed is True

    def test_classify_javascript_file(self):
        """Test classification of JavaScript file."""
        classifier = FileClassifier()
        result = classifier.classify_file("app.js")

        assert result.category == FileCategory.SOURCE_CODE
        assert result.language == ProgrammingLanguage.JAVASCRIPT
        assert result.embedding_type == "codebert"

    def test_classify_markdown_documentation(self):
        """Test classification of Markdown documentation."""
        classifier = FileClassifier()
        result = classifier.classify_file("README.md")

        assert result.category == FileCategory.DOCUMENTATION
        assert result.language is None
        assert result.embedding_type == "standard"

    def test_classify_readme_by_path(self):
        """Test README classification by path pattern."""
        classifier = FileClassifier()
        result = classifier.classify_file("readme.txt")

        assert result.category == FileCategory.DOCUMENTATION
        # Note: .txt extension also classifies as DOCUMENTATION, so this could be either
        assert result.classified_by in ["path", "extension"]

    def test_classify_json_config(self):
        """Test classification of JSON configuration."""
        classifier = FileClassifier()
        result = classifier.classify_file("package.json")

        assert result.category == FileCategory.CONFIGURATION
        assert result.embedding_type == "standard"

    def test_classify_yaml_config(self):
        """Test classification of YAML configuration."""
        classifier = FileClassifier()
        result = classifier.classify_file(".github/workflows/ci.yml")

        assert result.category == FileCategory.CONFIGURATION

    def test_classify_test_file_by_path(self):
        """Test classification of test files by path."""
        classifier = FileClassifier()

        # Path-based detection
        result = classifier.classify_file("tests/test_main.py")
        assert result.category == FileCategory.TEST_CODE

        # Filename pattern detection
        result = classifier.classify_file("src/utils_test.go")
        assert result.category == FileCategory.TEST_CODE

        result = classifier.classify_file("component.spec.ts")
        assert result.category == FileCategory.TEST_CODE

    def test_classify_image_asset(self):
        """Test classification of image assets."""
        classifier = FileClassifier()
        result = classifier.classify_file("logo.png")

        assert result.category == FileCategory.ASSET
        assert result.embedding_type == "none"
        assert result.should_embed is False

    def test_classify_binary_content(self):
        """Test binary file detection."""
        classifier = FileClassifier()
        binary_content = b"\x00\x01\x02\x03\xff\xfe\xfd"
        result = classifier.classify_file("data.bin", binary_content)

        assert result.is_binary is True
        assert result.embedding_type == "none"
        assert result.should_embed is False

    def test_classify_text_content(self):
        """Test text file processing."""
        classifier = FileClassifier()
        text_content = b"Hello, World!\nThis is a test."
        result = classifier.classify_file("hello.txt", text_content)

        assert result.is_binary is False
        assert result.size_bytes == len(text_content)
        assert result.line_count == 2

    def test_secret_detection_in_config(self):
        """Test secret detection in configuration files."""
        classifier = FileClassifier()
        config_content = b'{"api_key": "sk_live_abcdefghijklmnopqrstuvwxyz123456"}'
        result = classifier.classify_file(".env", config_content)

        assert result.category == FileCategory.CONFIGURATION
        assert result.has_secrets is True

    def test_no_secrets_in_clean_config(self):
        """Test that clean config is not flagged."""
        classifier = FileClassifier()
        config_content = b'{"app_name": "MyApp", "version": "1.0.0"}'
        result = classifier.classify_file("config.json", config_content)

        assert result.has_secrets is False

    def test_unknown_file_classification(self):
        """Test classification of unknown file types."""
        classifier = FileClassifier()
        result = classifier.classify_file("file.xyz")

        assert result.category == FileCategory.UNKNOWN
        assert result.confidence == 0.0
        assert result.classified_by == "fallback"

    def test_extension_overrides_path(self):
        """Test that extension has higher priority than path."""
        classifier = FileClassifier()
        # Even though it's in docs/, .py should classify as source code
        result = classifier.classify_file("docs/example.py")

        assert result.category == FileCategory.SOURCE_CODE
        assert result.classified_by == "extension"

    def test_confidence_scoring(self):
        """Test confidence scores for different methods."""
        classifier = FileClassifier()

        # Extension-based: 0.9
        ext_result = classifier.classify_file("main.py")
        assert ext_result.confidence == 0.9

        # Path-based: 0.75 (use a file that matches pattern)
        path_result = classifier.classify_file("README")
        assert path_result.confidence == 0.75
        assert path_result.classified_by == "path"

    def test_embedding_type_assignment(self):
        """Test correct embedding type assignment."""
        classifier = FileClassifier()

        # Source code -> codebert
        assert (
            classifier.classify_file("main.py").embedding_type
            == "codebert"
        )

        # Documentation -> standard
        assert (
            classifier.classify_file("README.md").embedding_type
            == "standard"
        )

        # Configuration -> standard
        assert (
            classifier.classify_file("config.json").embedding_type
            == "standard"
        )

        # Asset -> none
        assert (
            classifier.classify_file("image.png").embedding_type
            == "none"
        )

        # Test code -> codebert
        assert (
            classifier.classify_file("test_main.py").embedding_type
            == "codebert"
        )


class TestClassificationIntegration:
    """Integration tests for full classification workflow."""

    def test_classify_repository_files(self):
        """Test classification of various repository files."""
        classifier = FileClassifier()

        # Common repository files
        files = {
            "README.md": FileCategory.DOCUMENTATION,
            "LICENSE": FileCategory.DOCUMENTATION,
            "CONTRIBUTING.md": FileCategory.DOCUMENTATION,
            "setup.py": FileCategory.CONFIGURATION,
            "requirements.txt": FileCategory.CONFIGURATION,
            "src/main.py": FileCategory.SOURCE_CODE,
            "src/utils.js": FileCategory.SOURCE_CODE,
            "tests/test_main.py": FileCategory.TEST_CODE,
            "docs/api.md": FileCategory.DOCUMENTATION,
            "assets/logo.png": FileCategory.ASSET,
            ".github/workflows/ci.yml": FileCategory.CONFIGURATION,
            "package.json": FileCategory.CONFIGURATION,
            "Dockerfile": FileCategory.CONFIGURATION,
        }

        for file_path, expected_category in files.items():
            result = classifier.classify_file(file_path)
            assert result.category == expected_category, (
                f"Failed for {file_path}: expected {expected_category}, "
                f"got {result.category}"
            )

    def test_content_based_classification(self):
        """Test content-based classification when extension is missing."""
        classifier = FileClassifier()

        # Python script without extension
        python_content = b"""
#!/usr/bin/env python3
def main():
    import sys
    print("Hello")

if __name__ == "__main__":
    main()
"""
        result = classifier.classify_file("script", python_content)
        # Should detect as Python by shebang or keywords
        assert result.language == ProgrammingLanguage.PYTHON

    def test_file_classification_model_validation(self):
        """Test Pydantic model validation."""
        # Valid classification
        classification = FileClassification(
            file_path="test.py",
            category=FileCategory.SOURCE_CODE,
            language=ProgrammingLanguage.PYTHON,
            confidence=0.9,
            classified_by="extension",
            size_bytes=1024,
        )
        assert classification.confidence == 0.9

        # Invalid confidence (out of range)
        with pytest.raises(Exception):  # Pydantic ValidationError
            FileClassification(
                file_path="test.py",
                category=FileCategory.SOURCE_CODE,
                confidence=1.5,  # Invalid: > 1.0
                classified_by="extension",
                size_bytes=1024,
            )

    def test_large_file_handling(self):
        """Test handling of large files."""
        classifier = FileClassifier()
        large_content = b"x" * (10 * 1024 * 1024)  # 10MB
        result = classifier.classify_file("large.txt", large_content)

        assert result.size_bytes == 10 * 1024 * 1024
        assert result.is_binary is False

    def test_multilanguage_repository(self):
        """Test classification in a multi-language repository."""
        classifier = FileClassifier()

        files = [
            ("backend/main.py", ProgrammingLanguage.PYTHON),
            ("frontend/app.tsx", ProgrammingLanguage.TYPESCRIPT),
            ("mobile/App.swift", ProgrammingLanguage.SWIFT),
            ("services/handler.go", ProgrammingLanguage.GO),
            ("lib/core.rs", ProgrammingLanguage.RUST),
        ]

        for file_path, expected_lang in files:
            result = classifier.classify_file(file_path)
            assert result.category == FileCategory.SOURCE_CODE
            assert result.language == expected_lang
