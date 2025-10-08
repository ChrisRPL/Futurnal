"""Tests for webhook security components."""

import hashlib
import hmac
import time

import pytest

from futurnal.ingestion.github.webhook.security import (
    verify_webhook_signature,
    WebhookRateLimiter,
)


# ---------------------------------------------------------------------------
# Signature Verification Tests
# ---------------------------------------------------------------------------


def test_verify_webhook_signature_valid():
    """Test signature verification with valid signature."""
    secret = "my-webhook-secret"
    payload = b'{"ref": "refs/heads/main"}'

    # Compute valid signature
    signature = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    signature_header = f"sha256={signature}"

    assert verify_webhook_signature(payload, signature_header, secret) is True


def test_verify_webhook_signature_invalid():
    """Test signature verification with invalid signature."""
    secret = "my-webhook-secret"
    payload = b'{"ref": "refs/heads/main"}'
    signature_header = "sha256=invalid_signature_here"

    assert verify_webhook_signature(payload, signature_header, secret) is False


def test_verify_webhook_signature_wrong_secret():
    """Test signature verification with wrong secret."""
    correct_secret = "correct-secret"
    wrong_secret = "wrong-secret"
    payload = b'{"ref": "refs/heads/main"}'

    # Compute signature with correct secret
    signature = hmac.new(
        correct_secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    signature_header = f"sha256={signature}"

    # Verify with wrong secret should fail
    assert verify_webhook_signature(payload, signature_header, wrong_secret) is False


def test_verify_webhook_signature_modified_payload():
    """Test signature verification with modified payload."""
    secret = "my-webhook-secret"
    original_payload = b'{"ref": "refs/heads/main"}'
    modified_payload = b'{"ref": "refs/heads/develop"}'

    # Compute signature with original payload
    signature = hmac.new(
        secret.encode("utf-8"),
        original_payload,
        hashlib.sha256,
    ).hexdigest()
    signature_header = f"sha256={signature}"

    # Verify with modified payload should fail
    assert verify_webhook_signature(modified_payload, signature_header, secret) is False


def test_verify_webhook_signature_missing_prefix():
    """Test signature verification with missing sha256= prefix."""
    secret = "my-webhook-secret"
    payload = b'{"ref": "refs/heads/main"}'

    signature = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    # No sha256= prefix
    signature_header = signature

    assert verify_webhook_signature(payload, signature_header, secret) is False


def test_verify_webhook_signature_empty_header():
    """Test signature verification with empty header."""
    secret = "my-webhook-secret"
    payload = b'{"ref": "refs/heads/main"}'

    assert verify_webhook_signature(payload, "", secret) is False


def test_verify_webhook_signature_empty_payload():
    """Test signature verification with empty payload."""
    secret = "my-webhook-secret"
    payload = b""

    signature = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    signature_header = f"sha256={signature}"

    assert verify_webhook_signature(payload, signature_header, secret) is True


def test_verify_webhook_signature_unicode_secret():
    """Test signature verification with unicode secret."""
    secret = "my-webhook-secret-with-émojis-🎉"
    payload = b'{"ref": "refs/heads/main"}'

    signature = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    signature_header = f"sha256={signature}"

    assert verify_webhook_signature(payload, signature_header, secret) is True


# ---------------------------------------------------------------------------
# Rate Limiter Tests
# ---------------------------------------------------------------------------


def test_rate_limiter_creation():
    """Test rate limiter creation."""
    limiter = WebhookRateLimiter(max_requests_per_minute=60)
    assert limiter.max_requests == 60
    assert len(limiter.request_times) == 0


def test_rate_limiter_invalid_max_requests():
    """Test rate limiter with invalid max requests."""
    with pytest.raises(ValueError):
        WebhookRateLimiter(max_requests_per_minute=0)


def test_rate_limiter_allow_request():
    """Test rate limiter allows requests under limit."""
    limiter = WebhookRateLimiter(max_requests_per_minute=10)
    repo = "owner/repo"

    # First 10 requests should be allowed
    for _ in range(10):
        assert limiter.allow_request(repo) is True


def test_rate_limiter_block_request():
    """Test rate limiter blocks requests over limit."""
    limiter = WebhookRateLimiter(max_requests_per_minute=5)
    repo = "owner/repo"

    # First 5 requests allowed
    for _ in range(5):
        assert limiter.allow_request(repo) is True

    # 6th request should be blocked
    assert limiter.allow_request(repo) is False


def test_rate_limiter_per_repository():
    """Test rate limiter is per-repository."""
    limiter = WebhookRateLimiter(max_requests_per_minute=3)
    repo1 = "owner1/repo1"
    repo2 = "owner2/repo2"

    # 3 requests to repo1
    for _ in range(3):
        assert limiter.allow_request(repo1) is True

    # repo1 should be blocked
    assert limiter.allow_request(repo1) is False

    # repo2 should still be allowed
    assert limiter.allow_request(repo2) is True


def test_rate_limiter_window_expiration():
    """Test rate limiter sliding window expiration."""
    limiter = WebhookRateLimiter(max_requests_per_minute=2)
    repo = "owner/repo"

    # Make 2 requests
    assert limiter.allow_request(repo) is True
    assert limiter.allow_request(repo) is True

    # Should be blocked
    assert limiter.allow_request(repo) is False

    # Manually manipulate time (simulate 61 seconds passing)
    current_time = time.time()
    limiter.request_times[repo][0] = current_time - 61
    limiter.request_times[repo][1] = current_time - 61

    # Should be allowed again (old requests expired)
    assert limiter.allow_request(repo) is True


def test_rate_limiter_get_current_rate():
    """Test getting current rate for repository."""
    limiter = WebhookRateLimiter(max_requests_per_minute=10)
    repo = "owner/repo"

    # Initially 0
    assert limiter.get_current_rate(repo) == 0

    # After 3 requests
    for _ in range(3):
        limiter.allow_request(repo)

    assert limiter.get_current_rate(repo) == 3


def test_rate_limiter_reset_specific_repo():
    """Test resetting specific repository."""
    limiter = WebhookRateLimiter(max_requests_per_minute=2)
    repo1 = "owner1/repo1"
    repo2 = "owner2/repo2"

    # Make requests to both repos
    limiter.allow_request(repo1)
    limiter.allow_request(repo1)
    limiter.allow_request(repo2)

    # repo1 should be at limit
    assert limiter.allow_request(repo1) is False

    # Reset repo1
    limiter.reset(repo1)

    # repo1 should be allowed again
    assert limiter.allow_request(repo1) is True

    # repo2 should still have 1 request counted
    assert limiter.get_current_rate(repo2) == 1


def test_rate_limiter_reset_all():
    """Test resetting all repositories."""
    limiter = WebhookRateLimiter(max_requests_per_minute=2)
    repo1 = "owner1/repo1"
    repo2 = "owner2/repo2"

    # Make requests to both repos
    limiter.allow_request(repo1)
    limiter.allow_request(repo2)

    # Reset all
    limiter.reset()

    # Both should be at 0
    assert limiter.get_current_rate(repo1) == 0
    assert limiter.get_current_rate(repo2) == 0


def test_rate_limiter_cleanup_old_entries():
    """Test cleaning up old inactive repositories."""
    limiter = WebhookRateLimiter(max_requests_per_minute=10)
    repo = "owner/repo"

    # Make a request
    limiter.allow_request(repo)

    assert len(limiter.request_times) == 1

    # Manually set old timestamp (6 minutes ago)
    current_time = time.time()
    limiter.request_times[repo][0] = current_time - 360

    # Cleanup with 5 minute threshold
    limiter.cleanup_old_entries(max_age_seconds=300)

    # Should be cleaned up
    assert len(limiter.request_times) == 0


def test_rate_limiter_cleanup_keeps_recent():
    """Test cleanup doesn't remove recent repositories."""
    limiter = WebhookRateLimiter(max_requests_per_minute=10)
    repo = "owner/repo"

    # Make a request
    limiter.allow_request(repo)

    assert len(limiter.request_times) == 1

    # Cleanup with 5 minute threshold
    limiter.cleanup_old_entries(max_age_seconds=300)

    # Should NOT be cleaned up (recent activity)
    assert len(limiter.request_times) == 1


def test_rate_limiter_high_volume():
    """Test rate limiter with high volume scenario."""
    limiter = WebhookRateLimiter(max_requests_per_minute=100)
    repo = "owner/repo"

    # Make 100 requests (at limit)
    for _ in range(100):
        assert limiter.allow_request(repo) is True

    # 101st should be blocked
    assert limiter.allow_request(repo) is False

    # Check count
    assert limiter.get_current_rate(repo) == 100


def test_rate_limiter_multiple_repos_concurrent():
    """Test rate limiter with multiple repositories concurrently."""
    limiter = WebhookRateLimiter(max_requests_per_minute=5)
    repos = [f"owner{i}/repo{i}" for i in range(10)]

    # Each repo should be able to make 5 requests
    for repo in repos:
        for _ in range(5):
            assert limiter.allow_request(repo) is True

        # 6th should be blocked
        assert limiter.allow_request(repo) is False

    # Verify all repos are tracked
    assert len(limiter.request_times) == 10
