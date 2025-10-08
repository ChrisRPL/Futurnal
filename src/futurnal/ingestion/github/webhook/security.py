"""Security components for GitHub webhook integration.

This module implements HMAC-SHA256 signature verification and rate limiting
to protect against invalid webhooks and DDoS attacks.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signature Verification
# ---------------------------------------------------------------------------


def verify_webhook_signature(
    payload_body: bytes,
    signature_header: str,
    secret: str,
) -> bool:
    """Verify GitHub webhook signature using HMAC-SHA256.

    This function implements secure webhook signature verification as specified
    by GitHub's webhook documentation. It uses constant-time comparison to
    prevent timing attacks.

    Args:
        payload_body: Raw webhook payload body (bytes)
        signature_header: Value of X-Hub-Signature-256 header
        secret: Webhook secret configured in GitHub

    Returns:
        True if signature is valid, False otherwise

    Example:
        >>> body = b'{"ref": "refs/heads/main", ...}'
        >>> signature = "sha256=abc123..."
        >>> secret = "my-webhook-secret"
        >>> is_valid = verify_webhook_signature(body, signature, secret)
    """
    # Validate signature header format
    if not signature_header or not signature_header.startswith("sha256="):
        logger.warning("Invalid signature header format")
        return False

    # Extract signature (remove "sha256=" prefix)
    try:
        received_signature = signature_header[7:]  # len("sha256=") == 7
    except (IndexError, AttributeError):
        logger.warning("Failed to extract signature from header")
        return False

    # Compute expected signature
    expected_signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison to prevent timing attacks
    is_valid = hmac.compare_digest(received_signature, expected_signature)

    if not is_valid:
        logger.warning(
            "Webhook signature verification failed",
            extra={"expected_length": len(expected_signature)},
        )

    return is_valid


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class WebhookRateLimiter:
    """Rate limiter for webhook requests per repository.

    This class implements sliding window rate limiting per repository to
    prevent abuse and DDoS attacks. Each repository has independent rate
    limits to ensure one noisy repository doesn't block others.

    Attributes:
        max_requests_per_minute: Maximum requests allowed per minute per repository
        request_times: Tracking dict of request timestamps per repository

    Example:
        >>> limiter = WebhookRateLimiter(max_requests_per_minute=60)
        >>> if limiter.allow_request("octocat/Hello-World"):
        ...     # Process webhook
        ...     pass
        ... else:
        ...     # Return 429 Too Many Requests
        ...     pass
    """

    def __init__(self, max_requests_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum requests per minute per repository
        """
        if max_requests_per_minute < 1:
            raise ValueError("max_requests_per_minute must be at least 1")

        self.max_requests = max_requests_per_minute
        self.request_times: Dict[str, deque] = {}

    def allow_request(self, repo_full_name: str) -> bool:
        """Check if request should be allowed based on rate limits.

        This method implements a sliding window algorithm that tracks requests
        in the last 60 seconds. Old requests are automatically cleaned up.

        Args:
            repo_full_name: Repository identifier (owner/repo)

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        now = time.time()

        # Initialize request tracking for new repositories
        if repo_full_name not in self.request_times:
            self.request_times[repo_full_name] = deque()

        times = self.request_times[repo_full_name]

        # Remove requests older than 60 seconds (sliding window)
        while times and times[0] < now - 60:
            times.popleft()

        # Check if limit exceeded
        if len(times) >= self.max_requests:
            logger.warning(
                f"Rate limit exceeded for repository {repo_full_name}",
                extra={
                    "repository": repo_full_name,
                    "requests_in_window": len(times),
                    "max_requests": self.max_requests,
                },
            )
            return False

        # Record this request
        times.append(now)
        return True

    def get_current_rate(self, repo_full_name: str) -> int:
        """Get current request count for repository in the last minute.

        Args:
            repo_full_name: Repository identifier (owner/repo)

        Returns:
            Number of requests in the last 60 seconds
        """
        if repo_full_name not in self.request_times:
            return 0

        now = time.time()
        times = self.request_times[repo_full_name]

        # Clean old requests
        while times and times[0] < now - 60:
            times.popleft()

        return len(times)

    def reset(self, repo_full_name: Optional[str] = None) -> None:
        """Reset rate limiting for repository or all repositories.

        Args:
            repo_full_name: Repository to reset, or None to reset all
        """
        if repo_full_name is None:
            # Reset all
            self.request_times.clear()
            logger.info("Rate limiter reset for all repositories")
        elif repo_full_name in self.request_times:
            # Reset specific repository
            del self.request_times[repo_full_name]
            logger.info(f"Rate limiter reset for repository {repo_full_name}")

    def cleanup_old_entries(self, max_age_seconds: int = 300) -> None:
        """Clean up repositories with no recent requests.

        This method removes tracking data for repositories that haven't had
        requests in the specified time period to prevent memory bloat.

        Args:
            max_age_seconds: Remove tracking for repos inactive this long
        """
        now = time.time()
        repos_to_remove = []

        for repo_full_name, times in self.request_times.items():
            # Remove old requests from this repo's window
            while times and times[0] < now - 60:
                times.popleft()

            # If no requests in max_age_seconds, mark for removal
            if not times or times[-1] < now - max_age_seconds:
                repos_to_remove.append(repo_full_name)

        # Remove inactive repositories
        for repo in repos_to_remove:
            del self.request_times[repo]

        if repos_to_remove:
            logger.debug(
                f"Cleaned up rate limiter tracking for {len(repos_to_remove)} inactive repositories"
            )


__all__ = [
    "verify_webhook_signature",
    "WebhookRateLimiter",
]
