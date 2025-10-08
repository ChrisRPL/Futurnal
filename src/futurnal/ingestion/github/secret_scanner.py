"""Secret scanner with privacy-aware file exclusion.

This module wraps the SecretDetector to provide privacy-aware secret
scanning that integrates with repository privacy settings for file
pattern exclusion and content filtering.

Implements specification from:
docs/phase-1/github-connector-production-plan/09-privacy-consent-integration.md
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import List, Optional

from .descriptor import RepositoryPrivacySettings
from .secret_detector import SecretDetector

logger = logging.getLogger(__name__)


class SecretScanner:
    """Scans files for secrets and sensitive data with privacy controls.

    Wraps the SecretDetector to add privacy-aware file filtering based on:
    - File name patterns (e.g., *.env, *secret*, credentials.*)
    - File extensions (e.g., .key, .pem)
    - Content-based secret detection
    - File size limits

    Uses RepositoryPrivacySettings to determine exclusion rules.
    """

    def __init__(self, privacy_settings: RepositoryPrivacySettings):
        """Initialize secret scanner with privacy settings.

        Args:
            privacy_settings: Repository privacy settings defining
                exclusion patterns and secret detection rules
        """
        self.privacy_settings = privacy_settings
        self.detector = SecretDetector()

        # Convert pattern strings to lowercase for case-insensitive matching
        self.redact_patterns = [
            pattern.lower() for pattern in privacy_settings.redact_file_patterns
        ]
        self.exclude_extensions = [
            ext.lower() for ext in privacy_settings.exclude_extensions
        ]

    def scan_file(self, file_path: str | Path, content: bytes) -> bool:
        """Check if file contains secrets using content analysis.

        Args:
            file_path: Path to the file (used for logging only)
            content: File content as bytes

        Returns:
            True if secrets are detected, False otherwise
        """
        if not self.privacy_settings.detect_secrets:
            return False

        detected = self.detector.detect(content)

        if detected:
            logger.warning(
                f"Secret pattern detected in file "
                f"(hash: {self._hash_path(str(file_path))})"
            )

        return detected

    def should_exclude_file(
        self,
        file_path: str | Path,
        content: Optional[bytes] = None,
    ) -> bool:
        """Determine if file should be excluded due to privacy/security concerns.

        Checks multiple exclusion criteria in order:
        1. File name pattern matching (e.g., *secret*, .env*)
        2. File extension exclusion (e.g., .key, .pem)
        3. File size limits
        4. Content-based secret detection (if content provided)

        Args:
            file_path: Path to the file
            content: Optional file content for secret detection

        Returns:
            True if file should be excluded, False otherwise
        """
        path = Path(file_path)
        filename = path.name.lower()

        # Check file name patterns
        if self._matches_redact_patterns(filename):
            logger.debug(
                f"Excluding file due to name pattern: "
                f"{self._hash_path(str(file_path))}"
            )
            return True

        # Check file extension
        if self._has_excluded_extension(filename):
            logger.debug(
                f"Excluding file due to extension: "
                f"{self._hash_path(str(file_path))}"
            )
            return True

        # Check file size if file exists
        if path.exists():
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.privacy_settings.max_file_size_mb:
                logger.debug(
                    f"Excluding file due to size ({file_size_mb:.2f}MB > "
                    f"{self.privacy_settings.max_file_size_mb}MB): "
                    f"{self._hash_path(str(file_path))}"
                )
                return True

        # Check content for secrets if provided
        if content and self.privacy_settings.detect_secrets:
            if self.scan_file(file_path, content):
                return True

        return False

    def get_exclusion_reason(
        self,
        file_path: str | Path,
        content: Optional[bytes] = None,
    ) -> Optional[str]:
        """Get the reason why a file would be excluded.

        Useful for debugging and audit logging.

        Args:
            file_path: Path to the file
            content: Optional file content

        Returns:
            Exclusion reason string, or None if file wouldn't be excluded
        """
        path = Path(file_path)
        filename = path.name.lower()

        if self._matches_redact_patterns(filename):
            return "file_name_pattern"

        if self._has_excluded_extension(filename):
            return "file_extension"

        if path.exists():
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.privacy_settings.max_file_size_mb:
                return f"file_size_limit_{file_size_mb:.2f}mb"

        if content and self.privacy_settings.detect_secrets:
            if self.scan_file(file_path, content):
                return "secret_detected"

        return None

    def _matches_redact_patterns(self, filename: str) -> bool:
        """Check if filename matches any redaction patterns.

        Args:
            filename: Lowercase filename to check

        Returns:
            True if filename matches any redaction pattern
        """
        for pattern in self.redact_patterns:
            # Support glob patterns like *secret*, *.env
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def _has_excluded_extension(self, filename: str) -> bool:
        """Check if filename has an excluded extension.

        Args:
            filename: Lowercase filename to check

        Returns:
            True if filename has excluded extension
        """
        for ext in self.exclude_extensions:
            if filename.endswith(ext):
                return True
        return False

    def _hash_path(self, file_path: str) -> str:
        """Hash file path for privacy-aware logging.

        Args:
            file_path: File path to hash

        Returns:
            First 16 characters of SHA256 hash
        """
        from hashlib import sha256

        return sha256(file_path.encode()).hexdigest()[:16]


def create_secret_scanner(
    privacy_settings: RepositoryPrivacySettings,
) -> SecretScanner:
    """Factory function to create a SecretScanner instance.

    Args:
        privacy_settings: Repository privacy settings

    Returns:
        Configured SecretScanner instance
    """
    return SecretScanner(privacy_settings)


__all__ = [
    "SecretScanner",
    "create_secret_scanner",
]
