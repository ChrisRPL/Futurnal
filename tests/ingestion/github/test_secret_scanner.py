"""Comprehensive tests for SecretScanner with real secret patterns.

Tests cover:
- Secret detection in file content
- File pattern exclusion
- File extension exclusion
- File size limits
- Privacy-aware logging
- Integration with RepositoryPrivacySettings
"""

import tempfile
from pathlib import Path

import pytest

from futurnal.ingestion.github import (
    PrivacyLevel,
    RepositoryPrivacySettings,
    SecretScanner,
    create_secret_scanner,
)


class TestSecretScanner:
    """Test SecretScanner with various secret patterns."""

    def setup_method(self):
        """Set up test environment."""
        self.privacy_settings = RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.STANDARD,
            detect_secrets=True,
        )
        self.scanner = SecretScanner(self.privacy_settings)

    def test_detect_github_pat(self):
        """Test detection of GitHub Personal Access Token."""
        content = b"GITHUB_TOKEN=ghp_1234567890123456789012345678901234"

        detected = self.scanner.scan_file("test.txt", content)
        assert detected is True

    def test_detect_github_oauth_token(self):
        """Test detection of GitHub OAuth token."""
        content = b"oauth_token: gho_1234567890123456789012345678901234"

        detected = self.scanner.scan_file("config.yml", content)
        assert detected is True

    def test_detect_api_key(self):
        """Test detection of generic API key."""
        content = b'api_key = "sk_live_1234567890abcdefghijklmnop"'

        detected = self.scanner.scan_file("settings.py", content)
        assert detected is True

    def test_detect_aws_access_key(self):
        """Test detection of AWS access key."""
        content = b"AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"

        detected = self.scanner.scan_file("env.txt", content)
        assert detected is True

    def test_detect_private_key(self):
        """Test detection of RSA private key."""
        content = b"""-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890abcdefghijklmnop
-----END RSA PRIVATE KEY-----"""

        detected = self.scanner.scan_file("key.pem", content)
        assert detected is True

    def test_detect_password_in_config(self):
        """Test detection of password in configuration."""
        content = b'password: "MySecretPassword123!"'

        detected = self.scanner.scan_file("config.yml", content)
        assert detected is True

    def test_detect_jwt_token(self):
        """Test detection of JWT token."""
        content = b"token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

        detected = self.scanner.scan_file("auth.txt", content)
        assert detected is True

    def test_detect_database_connection_string(self):
        """Test detection of database connection string with credentials."""
        content = b"mongodb://admin:password123@localhost:27017/mydb"

        detected = self.scanner.scan_file("database.cfg", content)
        assert detected is True

    def test_detect_slack_token(self):
        """Test detection of Slack token."""
        content = b"slack_token = xoxb-1234567890-1234567890123-abcdefghijklmnopqrstuvwx"

        detected = self.scanner.scan_file("slack.py", content)
        assert detected is True

    def test_detect_stripe_key(self):
        """Test detection of Stripe secret key."""
        content = b"STRIPE_SECRET_KEY=sk_live_abcdefghijklmnopqrstuvwxyz123456"

        detected = self.scanner.scan_file("payment.py", content)
        assert detected is True

    def test_no_detection_clean_file(self):
        """Test that clean files pass without detection."""
        content = b"""
def hello_world():
    print("Hello, World!")
    return True
"""

        detected = self.scanner.scan_file("hello.py", content)
        assert detected is False

    def test_no_detection_when_disabled(self):
        """Test that detection is skipped when disabled in settings."""
        privacy_settings = RepositoryPrivacySettings(
            detect_secrets=False,
        )
        scanner = SecretScanner(privacy_settings)

        content = b"GITHUB_TOKEN=ghp_1234567890123456789012345678901234"

        detected = scanner.scan_file("test.txt", content)
        assert detected is False

    def test_binary_file_handling(self):
        """Test that binary files don't cause errors."""
        # Binary content that can't be decoded as UTF-8
        content = b"\x00\x01\x02\x03\x04\x05"

        detected = self.scanner.scan_file("binary.dat", content)
        assert detected is False


class TestFilePatternExclusion:
    """Test file pattern-based exclusion."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.privacy_settings = RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.STANDARD,
            redact_file_patterns=[
                "*secret*",
                "*password*",
                "*token*",
                ".env*",
                "credentials.*",
            ],
        )
        self.scanner = SecretScanner(self.privacy_settings)

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_exclude_secret_pattern(self):
        """Test exclusion of files matching *secret* pattern."""
        should_exclude = self.scanner.should_exclude_file("my_secret_key.txt")
        assert should_exclude is True

    def test_exclude_password_pattern(self):
        """Test exclusion of files matching *password* pattern."""
        should_exclude = self.scanner.should_exclude_file("passwords.txt")
        assert should_exclude is True

    def test_exclude_token_pattern(self):
        """Test exclusion of files matching *token* pattern."""
        should_exclude = self.scanner.should_exclude_file("auth_token.json")
        assert should_exclude is True

    def test_exclude_env_file(self):
        """Test exclusion of .env files."""
        should_exclude = self.scanner.should_exclude_file(".env")
        assert should_exclude is True

    def test_exclude_env_local(self):
        """Test exclusion of .env.local files."""
        should_exclude = self.scanner.should_exclude_file(".env.local")
        assert should_exclude is True

    def test_exclude_credentials_file(self):
        """Test exclusion of credentials files."""
        should_exclude = self.scanner.should_exclude_file("credentials.json")
        assert should_exclude is True

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case-insensitive."""
        should_exclude = self.scanner.should_exclude_file("MY_SECRET_KEY.TXT")
        assert should_exclude is True

    def test_allow_normal_file(self):
        """Test that normal files are not excluded."""
        should_exclude = self.scanner.should_exclude_file("README.md")
        assert should_exclude is False


class TestFileExtensionExclusion:
    """Test file extension-based exclusion."""

    def setup_method(self):
        """Set up test environment."""
        self.privacy_settings = RepositoryPrivacySettings(
            exclude_extensions=[
                ".exe",
                ".dll",
                ".so",
                ".key",
                ".pem",
                ".jpg",
                ".zip",
            ],
        )
        self.scanner = SecretScanner(self.privacy_settings)

    def test_exclude_binary_extensions(self):
        """Test exclusion of binary file extensions."""
        assert self.scanner.should_exclude_file("app.exe") is True
        assert self.scanner.should_exclude_file("library.dll") is True
        assert self.scanner.should_exclude_file("module.so") is True

    def test_exclude_key_files(self):
        """Test exclusion of key file extensions."""
        assert self.scanner.should_exclude_file("private.key") is True
        assert self.scanner.should_exclude_file("certificate.pem") is True

    def test_exclude_media_files(self):
        """Test exclusion of media file extensions."""
        assert self.scanner.should_exclude_file("image.jpg") is True

    def test_exclude_archive_files(self):
        """Test exclusion of archive file extensions."""
        assert self.scanner.should_exclude_file("backup.zip") is True

    def test_case_insensitive_extension(self):
        """Test that extension matching is case-insensitive."""
        assert self.scanner.should_exclude_file("app.EXE") is True
        assert self.scanner.should_exclude_file("image.JPG") is True

    def test_allow_code_files(self):
        """Test that code files are not excluded."""
        assert self.scanner.should_exclude_file("main.py") is False
        assert self.scanner.should_exclude_file("app.js") is False
        assert self.scanner.should_exclude_file("README.md") is False


class TestFileSizeLimits:
    """Test file size-based exclusion."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.privacy_settings = RepositoryPrivacySettings(
            max_file_size_mb=1,  # 1 MB limit
        )
        self.scanner = SecretScanner(self.privacy_settings)

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_exclude_large_file(self):
        """Test exclusion of files exceeding size limit."""
        # Create file larger than 1 MB
        large_file = self.temp_path / "large_file.txt"
        large_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

        should_exclude = self.scanner.should_exclude_file(large_file)
        assert should_exclude is True

    def test_allow_small_file(self):
        """Test that small files are not excluded."""
        # Create file smaller than 1 MB
        small_file = self.temp_path / "small_file.txt"
        small_file.write_bytes(b"x" * 1024)  # 1 KB

        should_exclude = self.scanner.should_exclude_file(small_file)
        assert should_exclude is False

    def test_non_existent_file(self):
        """Test that non-existent files don't cause errors."""
        non_existent = self.temp_path / "does_not_exist.txt"

        # Should not raise error, and should not exclude based on size
        # (other exclusion rules may still apply)
        try:
            self.scanner.should_exclude_file(non_existent)
        except Exception as e:
            pytest.fail(f"Should not raise exception for non-existent file: {e}")


class TestContentBasedExclusion:
    """Test content-based secret detection for exclusion."""

    def setup_method(self):
        """Set up test environment."""
        self.privacy_settings = RepositoryPrivacySettings(
            detect_secrets=True,
        )
        self.scanner = SecretScanner(self.privacy_settings)

    def test_exclude_file_with_secrets(self):
        """Test that files with detected secrets are excluded."""
        content = b"GITHUB_TOKEN=ghp_1234567890123456789012345678901234"

        should_exclude = self.scanner.should_exclude_file(
            "config.txt", content=content
        )
        assert should_exclude is True

    def test_allow_file_without_secrets(self):
        """Test that clean files are not excluded."""
        content = b"Just some normal configuration values"

        should_exclude = self.scanner.should_exclude_file(
            "config.txt", content=content
        )
        assert should_exclude is False


class TestExclusionReason:
    """Test getting detailed exclusion reasons."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.privacy_settings = RepositoryPrivacySettings(
            redact_file_patterns=["*secret*"],
            exclude_extensions=[".key"],
            max_file_size_mb=1,
            detect_secrets=True,
        )
        self.scanner = SecretScanner(self.privacy_settings)

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_reason_file_pattern(self):
        """Test getting exclusion reason for file pattern match."""
        reason = self.scanner.get_exclusion_reason("my_secret.txt")
        assert reason == "file_name_pattern"

    def test_reason_file_extension(self):
        """Test getting exclusion reason for extension match."""
        reason = self.scanner.get_exclusion_reason("private.key")
        assert reason == "file_extension"

    def test_reason_file_size(self):
        """Test getting exclusion reason for size limit."""
        large_file = self.temp_path / "large.txt"
        large_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

        reason = self.scanner.get_exclusion_reason(large_file)
        assert reason is not None
        assert "file_size_limit" in reason

    def test_reason_secret_detected(self):
        """Test getting exclusion reason for secret detection."""
        content = b"GITHUB_TOKEN=ghp_1234567890123456789012345678901234"

        reason = self.scanner.get_exclusion_reason("config.txt", content=content)
        assert reason == "secret_detected"

    def test_no_reason_for_allowed_file(self):
        """Test that allowed files have no exclusion reason."""
        reason = self.scanner.get_exclusion_reason("README.md")
        assert reason is None


class TestSecretScannerFactory:
    """Test SecretScanner factory function."""

    def test_create_secret_scanner(self):
        """Test factory function creates scanner correctly."""
        privacy_settings = RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
        )

        scanner = create_secret_scanner(privacy_settings)

        assert isinstance(scanner, SecretScanner)
        assert scanner.privacy_settings == privacy_settings


class TestPrivacyAwareLogging:
    """Test that scanner logs paths securely."""

    def setup_method(self):
        """Set up test environment."""
        self.privacy_settings = RepositoryPrivacySettings(
            detect_secrets=True,
        )
        self.scanner = SecretScanner(self.privacy_settings)

    def test_hash_path_consistency(self):
        """Test that path hashing is consistent."""
        path = "/path/to/secret/file.txt"

        hash1 = self.scanner._hash_path(path)
        hash2 = self.scanner._hash_path(path)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_paths_different_hashes(self):
        """Test that different paths produce different hashes."""
        hash1 = self.scanner._hash_path("/path/to/file1.txt")
        hash2 = self.scanner._hash_path("/path/to/file2.txt")

        assert hash1 != hash2


class TestRealWorldScenarios:
    """Test real-world scenario combinations."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.privacy_settings = RepositoryPrivacySettings(
            privacy_level=PrivacyLevel.STRICT,
            redact_file_patterns=["*secret*", ".env*", "credentials.*"],
            exclude_extensions=[".key", ".pem"],
            max_file_size_mb=5,
            detect_secrets=True,
        )
        self.scanner = SecretScanner(self.privacy_settings)

    def teardown_method(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_dotenv_file_with_secrets(self):
        """Test .env file with actual secrets is excluded."""
        env_file = self.temp_path / ".env"
        env_file.write_bytes(
            b"""
DATABASE_URL=postgres://user:password@localhost/db
AWS_SECRET_ACCESS_KEY=abcdefghijklmnopqrstuvwxyz123456
API_KEY=sk_live_1234567890abcdefghijklmnop
"""
        )

        # Should be excluded by pattern alone
        should_exclude = self.scanner.should_exclude_file(env_file)
        assert should_exclude is True

        reason = self.scanner.get_exclusion_reason(env_file)
        assert reason == "file_name_pattern"

    def test_aws_credentials_file(self):
        """Test AWS credentials file is excluded."""
        creds_file = self.temp_path / "credentials.json"
        creds_file.write_bytes(
            b"""
{
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
}
"""
        )

        should_exclude = self.scanner.should_exclude_file(creds_file)
        assert should_exclude is True

    def test_private_key_file(self):
        """Test private key file is excluded."""
        key_file = self.temp_path / "id_rsa.key"
        key_file.write_bytes(
            b"""-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890abcdefghijklmnop
-----END RSA PRIVATE KEY-----"""
        )

        should_exclude = self.scanner.should_exclude_file(key_file)
        assert should_exclude is True

        # Should be excluded by extension
        reason = self.scanner.get_exclusion_reason(key_file)
        assert reason == "file_extension"

    def test_normal_config_file_allowed(self):
        """Test that normal config files without secrets are allowed."""
        config_file = self.temp_path / "config.yml"
        config_file.write_bytes(
            b"""
app_name: MyApp
version: 1.0.0
debug: false
"""
        )

        content = config_file.read_bytes()
        should_exclude = self.scanner.should_exclude_file(config_file, content=content)
        assert should_exclude is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
