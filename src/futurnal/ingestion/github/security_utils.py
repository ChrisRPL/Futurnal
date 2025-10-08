"""Security utilities for GitHub connector.

Provides credential redaction, secure logging, and security validation utilities
to prevent credential leakage and ensure privacy compliance.
"""

import logging
import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Credential Patterns
# ---------------------------------------------------------------------------

# GitHub token patterns
GITHUB_TOKEN_PATTERNS = [
    r'ghp_[a-zA-Z0-9]{36,}',  # Personal Access Token
    r'gho_[a-zA-Z0-9]{36,}',  # OAuth Access Token
    r'ghu_[a-zA-Z0-9]{36,}',  # User-to-server token
    r'ghs_[a-zA-Z0-9]{36,}',  # Server-to-server token
    r'ghr_[a-zA-Z0-9]{36,}',  # Refresh token
    r'github_pat_[a-zA-Z0-9_]{82,}',  # Fine-grained PAT
]

# Generic credential patterns
CREDENTIAL_PATTERNS = [
    r'bearer\s+[a-zA-Z0-9\-._~+/]+=*',  # Bearer tokens (case insensitive)
    r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-._~+/]{20,}',  # token= or token:
    r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-._~+/]{20,}',  # API keys
    r'password["\']?\s*[:=]\s*["\']?[^\s"\']{8,}',  # Passwords
    r'secret["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-._~+/]{20,}',  # Secrets
]


# ---------------------------------------------------------------------------
# Credential Redaction Filter
# ---------------------------------------------------------------------------


class CredentialRedactionFilter(logging.Filter):
    """Logging filter that redacts credentials from log messages.

    Prevents credential leakage by replacing sensitive patterns with
    redacted placeholders before messages are logged.
    """

    def __init__(self, redaction_placeholder: str = "***REDACTED***"):
        """Initialize filter.

        Args:
            redaction_placeholder: String to replace credentials with
        """
        super().__init__()
        self.redaction_placeholder = redaction_placeholder

        # Compile all patterns for efficiency
        self.patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in GITHUB_TOKEN_PATTERNS + CREDENTIAL_PATTERNS
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record by redacting credentials.

        Args:
            record: Log record to filter

        Returns:
            True (always allow the record, just redact it)
        """
        # Redact message
        if isinstance(record.msg, str):
            record.msg = self._redact_string(record.msg)

        # Redact arguments
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact_value(v)
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(self._redact_value(v) for v in record.args)

        return True

    def _redact_string(self, text: str) -> str:
        """Redact credentials from string.

        Args:
            text: Text to redact

        Returns:
            Text with credentials replaced
        """
        for pattern in self.patterns:
            text = pattern.sub(self.redaction_placeholder, text)
        return text

    def _redact_value(self, value: Any) -> Any:
        """Redact credentials from any value type.

        Args:
            value: Value to redact (string, dict, list, etc.)

        Returns:
            Value with credentials redacted
        """
        if isinstance(value, str):
            return self._redact_string(value)
        elif isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(self._redact_value(v) for v in value)
        return value


# ---------------------------------------------------------------------------
# Exception Sanitization
# ---------------------------------------------------------------------------


class SanitizedException(Exception):
    """Exception wrapper that sanitizes error messages."""

    def __init__(self, original_exception: Exception):
        """Initialize sanitized exception.

        Args:
            original_exception: Original exception to sanitize
        """
        self.original_type = type(original_exception).__name__
        self.sanitized_message = self._sanitize_message(str(original_exception))
        super().__init__(self.sanitized_message)

    def _sanitize_message(self, message: str) -> str:
        """Sanitize exception message.

        Args:
            message: Original error message

        Returns:
            Sanitized message
        """
        filter = CredentialRedactionFilter()
        return filter._redact_string(message)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.original_type}: {self.sanitized_message}"


def sanitize_exception(exc: Exception) -> SanitizedException:
    """Sanitize an exception to prevent credential leakage.

    Args:
        exc: Exception to sanitize

    Returns:
        Sanitized exception safe to log/display
    """
    return SanitizedException(exc)


# ---------------------------------------------------------------------------
# Secure Logging Setup
# ---------------------------------------------------------------------------


def setup_secure_logging(logger: logging.Logger) -> logging.Logger:
    """Configure logger with credential redaction.

    Args:
        logger: Logger to configure

    Returns:
        Configured logger with redaction filter
    """
    # Add redaction filter to all handlers
    redaction_filter = CredentialRedactionFilter()

    for handler in logger.handlers:
        handler.addFilter(redaction_filter)

    # If no handlers, add to logger itself
    if not logger.handlers:
        logger.addFilter(redaction_filter)

    return logger


def get_secure_logger(name: str) -> logging.Logger:
    """Get a logger with credential redaction enabled.

    Args:
        name: Logger name

    Returns:
        Logger with redaction filter
    """
    logger = logging.getLogger(name)
    return setup_secure_logging(logger)


# ---------------------------------------------------------------------------
# Token Validation
# ---------------------------------------------------------------------------


def is_valid_github_token(token: str) -> bool:
    """Validate GitHub token format.

    Args:
        token: Token to validate

    Returns:
        True if token matches valid GitHub token pattern
    """
    for pattern in GITHUB_TOKEN_PATTERNS:
        if re.match(pattern, token):
            return True
    return False


def mask_token(token: str, visible_chars: int = 4) -> str:
    """Mask token for safe display.

    Args:
        token: Token to mask
        visible_chars: Number of characters to show at start/end

    Returns:
        Masked token (e.g., "ghp_****...****abcd")
    """
    if len(token) <= visible_chars * 2:
        return "*" * len(token)

    return f"{token[:visible_chars]}****...****{token[-visible_chars:]}"


# ---------------------------------------------------------------------------
# URL Sanitization
# ---------------------------------------------------------------------------


def sanitize_url(url: str) -> str:
    """Sanitize URL by removing credentials.

    Args:
        url: URL that might contain credentials

    Returns:
        URL with credentials removed
    """
    # Replace credentials in URLs like https://token@github.com
    url = re.sub(
        r'https://[^@:]+:[^@]+@',
        'https://***:***@',
        url
    )
    url = re.sub(
        r'https://[^@]+@',
        'https://***@',
        url
    )
    return url


# ---------------------------------------------------------------------------
# Audit Log Sanitization
# ---------------------------------------------------------------------------


def sanitize_for_audit(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize data for audit logging.

    Removes sensitive values while preserving structure for audit trail.

    Args:
        data: Data to sanitize

    Returns:
        Sanitized data safe for audit logs
    """
    sensitive_keys = {
        'token', 'password', 'secret', 'api_key', 'apikey',
        'access_token', 'refresh_token', 'bearer', 'authorization',
        'credential', 'private_key', 'secret_key',
    }

    def sanitize_value(key: str, value: Any) -> Any:
        """Sanitize individual value."""
        # Check if key is sensitive
        if isinstance(key, str) and any(
            sensitive in key.lower() for sensitive in sensitive_keys
        ):
            if isinstance(value, str) and len(value) > 0:
                return mask_token(value)
            return "***REDACTED***"

        # Recursively sanitize nested structures
        if isinstance(value, dict):
            return {k: sanitize_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [sanitize_value(key, item) for item in value]

        return value

    return {k: sanitize_value(k, v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# HTTPS Enforcement
# ---------------------------------------------------------------------------


def enforce_https(url: str) -> str:
    """Enforce HTTPS for URL.

    Args:
        url: URL to check

    Returns:
        HTTPS URL

    Raises:
        ValueError: If URL cannot be secured
    """
    if url.startswith('http://'):
        # Upgrade to HTTPS
        return url.replace('http://', 'https://', 1)
    elif url.startswith('https://'):
        return url
    else:
        raise ValueError(f"Invalid URL protocol: {url}")


def validate_https_only(url: str) -> None:
    """Validate that URL uses HTTPS.

    Args:
        url: URL to validate

    Raises:
        ValueError: If URL does not use HTTPS
    """
    if not url.startswith('https://'):
        raise ValueError(f"HTTPS required, got: {url[:20]}...")
