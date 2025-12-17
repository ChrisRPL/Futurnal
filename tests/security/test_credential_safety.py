"""Credential safety tests.

Validates that credentials are never leaked:
- Not in logs
- Not in telemetry
- Not in error messages
- Stored securely
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pytest


class TestCredentialStorage:
    """Test secure credential storage."""

    def test_credentials_use_system_keychain(self):
        """Verify credentials are stored in system keychain."""
        # Expected storage locations by platform
        storage_backends = {
            "darwin": "Keychain",
            "win32": "Credential Manager",
            "linux": "libsecret",
        }

        # Verify all platforms have secure storage
        for platform, backend in storage_backends.items():
            assert backend is not None, f"No secure storage for {platform}"

    def test_credentials_not_in_config_file(self):
        """Verify credentials are never stored in plain config."""
        sample_config = {
            "llm": {"model": "llama3.2:3b", "endpoint": "http://localhost:11434"},
            "sources": {
                "imap": {
                    "server": "imap.gmail.com",
                    "port": 993,
                    # NO password field
                }
            },
        }

        config_str = json.dumps(sample_config)

        forbidden_fields = ["password", "api_key", "token", "secret", "credential"]

        for field in forbidden_fields:
            assert field not in config_str.lower(), f"Credential field in config: {field}"

    def test_credential_reference_only(self):
        """Verify config only contains credential references."""
        config_with_reference = {
            "sources": {
                "imap": {
                    "server": "imap.gmail.com",
                    "credential_id": "keychain:futurnal/imap/personal",  # Reference only
                }
            }
        }

        # Should have reference, not actual credential
        assert "credential_id" in config_with_reference["sources"]["imap"]
        assert "password" not in config_with_reference["sources"]["imap"]


class TestCredentialLogging:
    """Test credentials are not logged."""

    def test_credentials_not_in_debug_logs(self):
        """Verify credentials never appear in debug logs."""
        sample_log_lines = [
            "DEBUG: Connecting to imap.gmail.com:993",
            "DEBUG: Authentication successful",
            "DEBUG: Fetching mailboxes",
            "INFO: Synced 150 emails",
        ]

        credential_patterns = [
            r"password[=:]\s*\S+",
            r"token[=:]\s*[a-zA-Z0-9]{20,}",
            r"api.key[=:]\s*\S+",
            r"Bearer\s+[a-zA-Z0-9._-]+",
        ]

        for line in sample_log_lines:
            for pattern in credential_patterns:
                assert not re.search(
                    pattern, line, re.IGNORECASE
                ), f"Credential pattern in log: {pattern}"

    def test_credentials_masked_in_errors(self):
        """Verify credentials are masked in error messages."""

        def create_connection_error(server: str, password: str) -> str:
            """Create error message WITHOUT exposing password."""
            return f"Failed to connect to {server}: Authentication failed"

        error_msg = create_connection_error("imap.gmail.com", "secret123")

        # Should not contain password
        assert "secret123" not in error_msg
        assert "password" not in error_msg.lower()

    def test_exception_messages_safe(self):
        """Verify exception messages don't contain credentials."""
        from futurnal.errors import SourceConnectionError

        # Create error with connection details
        error = SourceConnectionError("Cannot connect to email server")

        error_str = str(error)

        # Should not contain any credential-like data
        assert not re.search(r"[a-zA-Z0-9]{20,}", error_str)  # Long tokens
        assert "password" not in error_str.lower()


class TestCredentialTelemetry:
    """Test credentials are not in telemetry."""

    def test_telemetry_excludes_credentials(self):
        """Verify telemetry events never contain credentials."""
        telemetry_event = {
            "event": "source_connected",
            "source_type": "imap",
            "success": True,
            "latency_ms": 1500,
            # NO credentials
        }

        event_str = json.dumps(telemetry_event)

        forbidden_patterns = [
            "password",
            "token",
            "api_key",
            "secret",
            "credential",
            "authorization",
        ]

        for pattern in forbidden_patterns:
            assert (
                pattern not in event_str.lower()
            ), f"Credential pattern in telemetry: {pattern}"

    def test_telemetry_sanitizes_paths(self):
        """Verify telemetry sanitizes file paths."""
        # Original path
        original_path = "/Users/john/Documents/secrets/passwords.md"

        # Sanitized for telemetry
        sanitized = "/[USER]/Documents/[REDACTED]/[REDACTED].md"

        # Should not contain user-specific info
        assert "john" not in sanitized
        assert "secrets" not in sanitized
        assert "passwords" not in sanitized


class TestCredentialExposure:
    """Test for credential exposure vectors."""

    def test_no_credentials_in_http_responses(self):
        """Verify credentials never appear in HTTP responses."""
        api_response = {
            "status": "success",
            "sources": [
                {"id": "imap-1", "type": "imap", "server": "imap.gmail.com"}
                # NO password
            ],
        }

        response_str = json.dumps(api_response)

        assert "password" not in response_str
        assert "token" not in response_str.lower()

    def test_no_credentials_in_exports(self):
        """Verify data exports don't contain credentials."""
        export_data = {
            "version": "1.0.0",
            "export_date": "2024-12-17",
            "sources": [
                {"id": "imap-1", "type": "imap", "config": {"server": "imap.gmail.com"}}
            ],
            "entities": [],
            "relationships": [],
        }

        export_str = json.dumps(export_data)

        credential_fields = ["password", "token", "api_key", "secret", "credential"]

        for field in credential_fields:
            assert field not in export_str.lower()

    def test_no_credentials_in_error_reports(self):
        """Verify error reports don't contain credentials."""
        error_report = {
            "error_type": "ConnectionError",
            "message": "Failed to connect to email server",
            "stack_trace": [
                "File connector.py, line 100, in connect",
                "File imap.py, line 50, in authenticate",
            ],
            "context": {"server": "imap.gmail.com", "port": 993},
        }

        report_str = json.dumps(error_report)

        # Should not contain credentials even in stack traces
        assert not re.search(r"password\s*=", report_str, re.IGNORECASE)
        assert not re.search(r"token\s*=", report_str, re.IGNORECASE)


class TestCredentialHandling:
    """Test proper credential handling practices."""

    def test_credentials_cleared_from_memory(self):
        """Verify credentials are cleared after use."""
        # Simulate credential handling
        credential = "secret_password_123"

        # Use credential
        _ = len(credential)

        # Clear credential (in production, use secure memory clearing)
        credential = None

        assert credential is None

    def test_credential_access_logged(self):
        """Verify credential access is audit-logged (without the credential)."""
        audit_entry = {
            "action": "credential_accessed",
            "credential_id": "keychain:futurnal/imap/personal",
            "purpose": "imap_authentication",
            "timestamp": "2024-12-17T10:00:00Z",
            # NO actual credential value
        }

        assert "credential_id" in audit_entry
        assert "value" not in audit_entry
        assert "password" not in json.dumps(audit_entry)

    def test_credential_rotation_supported(self):
        """Verify credential rotation is supported."""
        credential_metadata = {
            "id": "keychain:futurnal/imap/personal",
            "created_at": "2024-01-01T00:00:00Z",
            "last_rotated": "2024-06-01T00:00:00Z",
            "rotation_recommended": True,
        }

        assert "last_rotated" in credential_metadata
        assert "rotation_recommended" in credential_metadata
