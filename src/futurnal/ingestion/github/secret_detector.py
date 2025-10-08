"""Secret and sensitive data detection for repository files.

This module implements pattern-based secret detection to flag configuration
files containing credentials, API keys, tokens, and other sensitive data.
"""

import re
from typing import List, Pattern


class SecretDetector:
    """Detects secrets and sensitive data in file content.

    Uses regex patterns to identify common secret formats including API keys,
    passwords, tokens, private keys, and cloud provider credentials.
    """

    def __init__(self):
        """Initialize the secret detector with pre-compiled patterns."""
        self.secret_patterns = self._load_secret_patterns()

    def detect(self, content: bytes) -> bool:
        """Detect if content contains secrets.

        Args:
            content: File content as bytes

        Returns:
            True if secrets are detected, False otherwise
        """
        try:
            text = content.decode("utf-8", errors="ignore")
        except (UnicodeDecodeError, AttributeError):
            # If we can't decode, assume no secrets (binary files)
            return False

        for pattern in self.secret_patterns:
            if pattern.search(text):
                return True

        return False

    def _load_secret_patterns(self) -> List[Pattern]:
        """Load and compile regex patterns for secret detection.

        Returns:
            List of compiled regex patterns
        """
        patterns = [
            # Generic API keys
            r"(?i)(api[_-]?key|apikey)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})",
            # Passwords
            r"(?i)(password|passwd|pwd)[\s]*[=:]+[\s]*['\"]?([^\s'\"]{8,})",
            # Generic tokens
            r"(?i)(token|auth[_-]?token)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})",
            # Generic secrets
            r"(?i)(secret|secret[_-]?key)[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{20,})",
            # Private keys
            r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
            r"-----BEGIN\s+ENCRYPTED\s+PRIVATE\s+KEY-----",
            r"-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----",
            # GitHub tokens (flexible length for various formats)
            r"ghp_[a-zA-Z0-9]{30,}",  # Personal Access Token
            r"gho_[a-zA-Z0-9]{30,}",  # OAuth Access Token
            r"ghu_[a-zA-Z0-9]{30,}",  # User-to-server token
            r"ghs_[a-zA-Z0-9]{30,}",  # Server-to-server token
            r"ghr_[a-zA-Z0-9]{30,}",  # Refresh token
            r"github_pat_[a-zA-Z0-9_]{25,}",  # Fine-grained PAT (flexible)
            # AWS credentials
            r"AKIA[0-9A-Z]{16}",  # AWS Access Key ID
            r"(?i)aws[_-]?secret[_-]?access[_-]?key[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9/+=]{40})",
            # Google Cloud
            r"(?i)google[_-]?api[_-]?key[\s]*[=:]+[\s]*['\"]?([a-zA-Z0-9_\-]{39})",
            # Slack tokens (various formats)
            r"xox[baprs]-[0-9]{8,}-[0-9]{8,}-[a-zA-Z0-9]{20,}",
            r"xox[baprs]-[a-zA-Z0-9-]{20,}",  # Alternative format
            # Stripe keys
            r"sk_live_[0-9a-zA-Z]{24,}",
            r"rk_live_[0-9a-zA-Z]{24,}",
            # Azure
            r"(?i)azure[_-]?subscription[_-]?key[\s]*[=:]+[\s]*['\"]?([a-f0-9]{32})",
            # JWT tokens (basic detection)
            r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
            # Database connection strings
            r"(?i)(mongodb|mysql|postgres|postgresql)://[^\s]+:[^\s]+@",
            # SSH private key indicators (content)
            r"(?i)^ssh-rsa\s+AAAA[0-9A-Za-z+/]+[=]{0,3}\s*",
            # Generic bearer tokens
            r"(?i)bearer\s+[a-zA-Z0-9_\-\.=]{20,}",
            # NPM tokens
            r"npm_[a-zA-Z0-9]{36}",
            # PyPI tokens
            r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9-_]{50,}",
            # Docker Hub tokens
            r"(?i)dockerhub[_-]?token[\s]*[=:]+[\s]*['\"]?([a-f0-9-]{36})",
            # Telegram bot tokens
            r"[0-9]{8,10}:[a-zA-Z0-9_-]{35}",
            # Twilio
            r"SK[a-f0-9]{32}",
            # SendGrid
            r"SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}",
            # Mailgun
            r"key-[a-f0-9]{32}",
            # Generic credentials in JSON/YAML format
            r'(?i)"(password|secret|token|api_key)"[\s]*:[\s]*"[^"]{8,}"',
            r"(?i)(password|secret|token|api_key):\s+['\"]?[^\s'\"]{8,}",
        ]

        return [re.compile(pattern) for pattern in patterns]
