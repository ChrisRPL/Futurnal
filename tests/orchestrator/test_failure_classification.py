"""Tests for enhanced failure classification system."""

import pytest

from futurnal.orchestrator.quarantine import (
    QuarantineReason,
    classify_failure,
    quarantine_reason_to_failure_type,
)
from futurnal.orchestrator.retry_policy import FailureType


class MockHTTPException(Exception):
    """Mock HTTP exception with status_code attribute."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(message)


class TestFailureClassificationByException:
    """Test failure classification using exception types."""

    def test_permission_error_classification(self):
        """Test that PermissionError is classified as PERMISSION_DENIED."""
        result = classify_failure(
            error_message="Permission denied",
            exception_type=PermissionError,
        )
        assert result == QuarantineReason.PERMISSION_DENIED

    def test_memory_error_classification(self):
        """Test that MemoryError is classified as RESOURCE_EXHAUSTED."""
        result = classify_failure(
            error_message="Out of memory",
            exception_type=MemoryError,
        )
        assert result == QuarantineReason.RESOURCE_EXHAUSTED

    def test_timeout_error_classification(self):
        """Test that TimeoutError is classified as TIMEOUT."""
        result = classify_failure(
            error_message="Operation timed out",
            exception_type=TimeoutError,
        )
        assert result == QuarantineReason.TIMEOUT


class TestFailureClassificationByStatusCode:
    """Test failure classification using HTTP status codes."""

    def test_status_429_rate_limited(self):
        """Test that HTTP 429 is classified as RATE_LIMITED."""
        exc = MockHTTPException(429, "Too many requests")
        result = classify_failure(
            error_message="Too many requests",
            exception=exc,
        )
        assert result == QuarantineReason.RATE_LIMITED

    def test_status_502_timeout(self):
        """Test that HTTP 502 is classified as TIMEOUT (transient)."""
        exc = MockHTTPException(502, "Bad Gateway")
        result = classify_failure(
            error_message="Bad Gateway",
            exception=exc,
        )
        assert result == QuarantineReason.TIMEOUT

    def test_status_503_timeout(self):
        """Test that HTTP 503 is classified as TIMEOUT (transient)."""
        exc = MockHTTPException(503, "Service Unavailable")
        result = classify_failure(
            error_message="Service Unavailable",
            exception=exc,
        )
        assert result == QuarantineReason.TIMEOUT

    def test_status_504_timeout(self):
        """Test that HTTP 504 is classified as TIMEOUT (transient)."""
        exc = MockHTTPException(504, "Gateway Timeout")
        result = classify_failure(
            error_message="Gateway Timeout",
            exception=exc,
        )
        assert result == QuarantineReason.TIMEOUT

    def test_status_401_permission_denied(self):
        """Test that HTTP 401 is classified as PERMISSION_DENIED."""
        exc = MockHTTPException(401, "Unauthorized")
        result = classify_failure(
            error_message="Unauthorized",
            exception=exc,
        )
        assert result == QuarantineReason.PERMISSION_DENIED

    def test_status_403_permission_denied(self):
        """Test that HTTP 403 is classified as PERMISSION_DENIED."""
        exc = MockHTTPException(403, "Forbidden")
        result = classify_failure(
            error_message="Forbidden",
            exception=exc,
        )
        assert result == QuarantineReason.PERMISSION_DENIED

    def test_status_404_permission_denied(self):
        """Test that HTTP 404 is classified as PERMISSION_DENIED."""
        exc = MockHTTPException(404, "Not Found")
        result = classify_failure(
            error_message="Not Found",
            exception=exc,
        )
        assert result == QuarantineReason.PERMISSION_DENIED


class TestFailureClassificationByMessage:
    """Test failure classification using message patterns."""

    def test_rate_limit_patterns(self):
        """Test various rate limit message patterns."""
        rate_limit_messages = [
            "rate limit exceeded",
            "too many requests",
            "quota exceeded",
            "API rate-limit reached",
            "throttle limit hit",
            "Error 429: Rate limited",
        ]

        for message in rate_limit_messages:
            result = classify_failure(message)
            assert result == QuarantineReason.RATE_LIMITED, f"Failed for: {message}"

    def test_permission_denied_patterns(self):
        """Test various permission/access message patterns."""
        permission_messages = [
            "permission denied",
            "access denied",
            "authentication failed",
            "invalid credentials",
            "unauthorized access",
            "forbidden resource",
            "Error 401: Unauthorized",
            "Error 403: Forbidden",
        ]

        for message in permission_messages:
            result = classify_failure(message)
            assert result == QuarantineReason.PERMISSION_DENIED, f"Failed for: {message}"

    def test_parse_error_patterns(self):
        """Test various parsing error message patterns."""
        parse_messages = [
            "parse error in document",
            "parsing failed",
            "malformed JSON",
            "invalid format detected",
            "decode error occurred",
        ]

        for message in parse_messages:
            result = classify_failure(message)
            assert result == QuarantineReason.PARSE_ERROR, f"Failed for: {message}"

    def test_resource_exhausted_patterns(self):
        """Test various resource exhaustion message patterns."""
        resource_messages = [
            "out of memory",
            "disk space exhausted",
            "out of space",
            "resource limit exceeded",
            "file too large",
        ]

        for message in resource_messages:
            result = classify_failure(message)
            assert result == QuarantineReason.RESOURCE_EXHAUSTED, f"Failed for: {message}"

    def test_timeout_patterns(self):
        """Test various timeout message patterns."""
        timeout_messages = [
            "operation timed out",
            "request timeout",
            "deadline exceeded",
            "connection refused",
            "temporary failure",
            "service unavailable",
            "Error 503: Unavailable",
            "Error 502: Bad Gateway",
            "Error 504: Gateway Timeout",
        ]

        for message in timeout_messages:
            result = classify_failure(message)
            assert result == QuarantineReason.TIMEOUT, f"Failed for: {message}"

    def test_dependency_failure_patterns(self):
        """Test various dependency failure message patterns."""
        dependency_messages = [
            "neo4j connection failed",
            "chromadb is unavailable",
            "database connection error",
            "service unavailable: neo4j",
        ]

        for message in dependency_messages:
            result = classify_failure(message)
            assert result == QuarantineReason.DEPENDENCY_FAILURE, f"Failed for: {message}"

    def test_connector_error_pattern(self):
        """Test connector-specific error pattern."""
        result = classify_failure("connector initialization failed")
        assert result == QuarantineReason.CONNECTOR_ERROR

    def test_state_corruption_patterns(self):
        """Test various state corruption message patterns."""
        state_messages = [
            "state corruption detected",
            "corrupt database",
            "inconsistent state",
        ]

        for message in state_messages:
            result = classify_failure(message)
            assert result == QuarantineReason.INVALID_STATE, f"Failed for: {message}"

    def test_unknown_classification(self):
        """Test that unrecognized errors are classified as UNKNOWN."""
        result = classify_failure("Something went wrong")
        assert result == QuarantineReason.UNKNOWN


class TestFailureClassificationPrecedence:
    """Test classification precedence (exception type > status code > message)."""

    def test_exception_type_takes_precedence(self):
        """Test that exception type takes precedence over message."""
        # Message suggests parse error, but exception type says timeout
        result = classify_failure(
            error_message="parse error occurred",
            exception_type=TimeoutError,
        )
        assert result == QuarantineReason.TIMEOUT

    def test_status_code_takes_precedence_over_message(self):
        """Test that status code takes precedence over message."""
        # Message suggests parse error, but status code says rate limit
        exc = MockHTTPException(429, "parse error in response")
        result = classify_failure(
            error_message="parse error in response",
            exception=exc,
        )
        assert result == QuarantineReason.RATE_LIMITED


class TestQuarantineReasonMapping:
    """Test mapping from QuarantineReason to FailureType."""

    def test_rate_limited_mapping(self):
        """Test RATE_LIMITED maps correctly."""
        result = quarantine_reason_to_failure_type(QuarantineReason.RATE_LIMITED)
        assert result == FailureType.RATE_LIMITED

    def test_timeout_mapping(self):
        """Test TIMEOUT maps to TRANSIENT."""
        result = quarantine_reason_to_failure_type(QuarantineReason.TIMEOUT)
        assert result == FailureType.TRANSIENT

    def test_dependency_failure_mapping(self):
        """Test DEPENDENCY_FAILURE maps to TRANSIENT."""
        result = quarantine_reason_to_failure_type(QuarantineReason.DEPENDENCY_FAILURE)
        assert result == FailureType.TRANSIENT

    def test_connector_error_mapping(self):
        """Test CONNECTOR_ERROR maps to TRANSIENT."""
        result = quarantine_reason_to_failure_type(QuarantineReason.CONNECTOR_ERROR)
        assert result == FailureType.TRANSIENT

    def test_permission_denied_mapping(self):
        """Test PERMISSION_DENIED maps to PERMANENT."""
        result = quarantine_reason_to_failure_type(QuarantineReason.PERMISSION_DENIED)
        assert result == FailureType.PERMANENT

    def test_invalid_state_mapping(self):
        """Test INVALID_STATE maps to PERMANENT."""
        result = quarantine_reason_to_failure_type(QuarantineReason.INVALID_STATE)
        assert result == FailureType.PERMANENT

    def test_parse_error_mapping(self):
        """Test PARSE_ERROR maps to UNKNOWN."""
        result = quarantine_reason_to_failure_type(QuarantineReason.PARSE_ERROR)
        assert result == FailureType.UNKNOWN

    def test_resource_exhausted_mapping(self):
        """Test RESOURCE_EXHAUSTED maps to UNKNOWN."""
        result = quarantine_reason_to_failure_type(QuarantineReason.RESOURCE_EXHAUSTED)
        assert result == FailureType.UNKNOWN

    def test_unknown_mapping(self):
        """Test UNKNOWN maps to UNKNOWN."""
        result = quarantine_reason_to_failure_type(QuarantineReason.UNKNOWN)
        assert result == FailureType.UNKNOWN


class TestRealWorldScenarios:
    """Test classification with real-world error messages."""

    def test_github_rate_limit_error(self):
        """Test GitHub API rate limit error."""
        error = "API rate limit exceeded for user. Retry after 2024-01-01T00:00:00Z"
        result = classify_failure(error)
        assert result == QuarantineReason.RATE_LIMITED

    def test_imap_connection_timeout(self):
        """Test IMAP connection timeout error."""
        error = "IMAP connection timed out after 30 seconds"
        result = classify_failure(error)
        assert result == QuarantineReason.TIMEOUT

    def test_obsidian_permission_denied(self):
        """Test Obsidian vault permission denied error."""
        error = "Permission denied: Cannot read /Users/john/vault/.obsidian/config"
        result = classify_failure(error)
        assert result == QuarantineReason.PERMISSION_DENIED

    def test_github_invalid_token(self):
        """Test GitHub invalid token error."""
        error = "Invalid credentials: Bad credentials (401)"
        result = classify_failure(error)
        assert result == QuarantineReason.PERMISSION_DENIED

    def test_neo4j_connection_failure(self):
        """Test Neo4j connection failure."""
        error = "Failed to establish connection to Neo4j: Connection refused"
        result = classify_failure(error)
        assert result == QuarantineReason.DEPENDENCY_FAILURE

    def test_unstructured_parse_error(self):
        """Test Unstructured.io parsing error."""
        error = "Failed to parse PDF: Malformed file structure"
        result = classify_failure(error)
        assert result == QuarantineReason.PARSE_ERROR
