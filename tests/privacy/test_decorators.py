"""Tests for privacy decorators module."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from futurnal.privacy.decorators import (
    requires_consent,
    audit_action,
    privacy_protected,
    ConsentDeniedError,
    bypass_consent_for_testing,
    _is_safe_for_audit,
)
from futurnal.privacy.policy_engine import (
    PolicyEngine,
    PolicyResult,
    PolicyDecision,
    configure_policy_engine,
    reset_policy_engine,
)
from futurnal.privacy.consent import ConsentRegistry


class TestRequiresConsent:
    """Test requires_consent decorator."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create and configure policy engine."""
        registry = ConsentRegistry(tmp_path / "consent")
        return configure_policy_engine(consent_registry=registry)

    @pytest.fixture
    def registry(self, engine):
        """Get the registry from the engine."""
        return engine._consent_registry

    def teardown_method(self):
        reset_policy_engine()

    def test_static_source_scope_allowed(self, registry):
        registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
        def process_content(doc_id: str):
            return f"processed:{doc_id}"

        result = process_content("doc123")
        assert result == "processed:doc123"

    def test_static_source_scope_denied(self):
        @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
        def process_content(doc_id: str):
            return f"processed:{doc_id}"

        with pytest.raises(ConsentDeniedError) as exc_info:
            process_content("doc123")

        assert exc_info.value.source == "obsidian"
        assert exc_info.value.scope == "CONTENT_ANALYSIS"

    def test_dynamic_source_from_arg(self, registry):
        registry.grant(source="vault_123", scope="CONTENT_ANALYSIS")

        @requires_consent(source_arg="vault_id", scope="CONTENT_ANALYSIS")
        def process_vault(vault_id: str, content: str):
            return f"processed:{vault_id}"

        result = process_vault("vault_123", "content")
        assert result == "processed:vault_123"

    def test_dynamic_source_from_kwarg(self, registry):
        registry.grant(source="vault_456", scope="CONTENT_ANALYSIS")

        @requires_consent(source_arg="vault_id", scope="CONTENT_ANALYSIS")
        def process_vault(vault_id: str, content: str):
            return f"processed:{vault_id}"

        result = process_vault(vault_id="vault_456", content="content")
        assert result == "processed:vault_456"

    def test_dynamic_scope_from_arg(self, registry):
        registry.grant(source="obsidian", scope="METADATA_ACCESS")

        @requires_consent(source="obsidian", scope_arg="operation")
        def perform_operation(doc_id: str, operation: str):
            return f"{operation}:{doc_id}"

        result = perform_operation("doc123", "METADATA_ACCESS")
        assert result == "METADATA_ACCESS:doc123"

    def test_async_function(self, registry):
        registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
        async def async_process(doc_id: str):
            await asyncio.sleep(0)
            return f"async:{doc_id}"

        result = asyncio.run(async_process("doc123"))
        assert result == "async:doc123"

    def test_async_function_denied(self):
        @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
        async def async_process(doc_id: str):
            await asyncio.sleep(0)
            return f"async:{doc_id}"

        with pytest.raises(ConsentDeniedError):
            asyncio.run(async_process("doc123"))

    def test_missing_required_args(self):
        with pytest.raises(ValueError, match="source"):

            @requires_consent(scope="CONTENT_ANALYSIS")
            def func():
                pass

        with pytest.raises(ValueError, match="scope"):

            @requires_consent(source="obsidian")
            def func2():
                pass

    def test_custom_error_class(self):
        class CustomError(Exception):
            def __init__(self, source, scope, reason):
                self.source = source
                self.scope = scope
                super().__init__(reason)

        @requires_consent(
            source="obsidian",
            scope="CONTENT_ANALYSIS",
            error_class=CustomError,
        )
        def process():
            pass

        with pytest.raises(CustomError):
            process()

    def test_with_custom_engine(self, tmp_path):
        # Create a separate engine
        registry = ConsentRegistry(tmp_path / "custom_consent")
        registry.grant(source="custom", scope="CUSTOM_SCOPE")
        custom_engine = PolicyEngine(consent_registry=registry)

        @requires_consent(
            source="custom",
            scope="CUSTOM_SCOPE",
            policy_engine=custom_engine,
        )
        def custom_func():
            return "success"

        result = custom_func()
        assert result == "success"


class TestAuditAction:
    """Test audit_action decorator."""

    @pytest.fixture
    def audit_logger(self):
        return MagicMock()

    @pytest.fixture
    def engine(self, audit_logger, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")
        return configure_policy_engine(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

    def teardown_method(self):
        reset_policy_engine()

    def test_logs_action(self, engine, audit_logger):
        @audit_action("test_action")
        def do_something(item_id: str):
            return f"done:{item_id}"

        result = do_something("item123")
        assert result == "done:item123"

        audit_logger.record.assert_called_once()
        call_args = audit_logger.record.call_args[0][0]  # AuditEvent object
        assert call_args.action == "test_action"
        assert call_args.status == "success"

    def test_uses_function_name_as_action(self, engine, audit_logger):
        @audit_action()
        def my_custom_function():
            return "result"

        my_custom_function()

        call_args = audit_logger.record.call_args[0][0]
        assert call_args.action == "my_custom_function"

    def test_logs_timing(self, engine, audit_logger):
        @audit_action(include_timing=True)
        def slow_function():
            import time

            time.sleep(0.01)
            return "done"

        slow_function()

        call_args = audit_logger.record.call_args[0][0]
        assert "duration_ms" in call_args.metadata
        assert call_args.metadata["duration_ms"] >= 10

    def test_logs_error(self, engine, audit_logger):
        @audit_action()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        call_args = audit_logger.record.call_args[0][0]
        assert call_args.status == "error"
        assert call_args.metadata["error"] == "ValueError"

    def test_no_log_on_error_when_disabled(self, engine, audit_logger):
        @audit_action(on_error=False)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        audit_logger.record.assert_not_called()

    def test_include_safe_args(self, engine, audit_logger):
        @audit_action(include_args=True)
        def process_item(item_id: str, count: int):
            return "done"

        process_item("item123", 5)

        call_args = audit_logger.record.call_args[0][0]
        assert "args" in call_args.metadata
        assert call_args.metadata["args"]["item_id"] == "item123"
        assert call_args.metadata["args"]["count"] == "5"

    def test_excludes_sensitive_args(self, engine, audit_logger):
        @audit_action(include_args=True)
        def login(username: str, password: str, token: str):
            return "done"

        login("user", "secret123", "token456")

        call_args = audit_logger.record.call_args[0][0]
        args = call_args.metadata.get("args", {})
        assert "username" in args
        assert "password" not in args
        assert "token" not in args

    def test_include_result_count(self, engine, audit_logger):
        @audit_action(include_result=True)
        def get_items():
            return [1, 2, 3, 4, 5]

        get_items()

        call_args = audit_logger.record.call_args[0][0]
        assert call_args.metadata["result_count"] == 5

    def test_async_function(self, engine, audit_logger):
        @audit_action("async_action")
        async def async_func():
            await asyncio.sleep(0)
            return "done"

        result = asyncio.run(async_func())
        assert result == "done"

        audit_logger.record.assert_called_once()

    def test_source_from_arg(self, engine, audit_logger):
        @audit_action(source_arg="source_id")
        def process_source(source_id: str):
            return "done"

        process_source("my_source")

        call_args = audit_logger.record.call_args[0][0]
        assert call_args.source == "my_source"


class TestPrivacyProtected:
    """Test privacy_protected combined decorator."""

    @pytest.fixture
    def audit_logger(self):
        return MagicMock()

    @pytest.fixture
    def engine(self, audit_logger, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")
        registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")
        return configure_policy_engine(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

    @pytest.fixture
    def registry(self, engine):
        return engine._consent_registry

    def teardown_method(self):
        reset_policy_engine()

    def test_combined_consent_and_audit(self, audit_logger, registry):
        @privacy_protected(source="obsidian", scope="CONTENT_ANALYSIS")
        def analyze_content(doc_id: str):
            return f"analyzed:{doc_id}"

        result = analyze_content("doc123")
        assert result == "analyzed:doc123"

        # Should have audit log entry
        audit_logger.record.assert_called()

    def test_denied_without_consent(self, audit_logger):
        @privacy_protected(source="unknown", scope="CONTENT_ANALYSIS")
        def analyze_content(doc_id: str):
            return f"analyzed:{doc_id}"

        with pytest.raises(ConsentDeniedError):
            analyze_content("doc123")

    def test_async_combined(self, audit_logger, registry):
        @privacy_protected(source="obsidian", scope="CONTENT_ANALYSIS")
        async def async_analyze(doc_id: str):
            await asyncio.sleep(0)
            return f"async:{doc_id}"

        result = asyncio.run(async_analyze("doc123"))
        assert result == "async:doc123"


class TestIsSafeForAudit:
    """Test _is_safe_for_audit helper function."""

    def test_safe_values(self):
        assert _is_safe_for_audit("item_id", "123") is True
        assert _is_safe_for_audit("count", 42) is True
        assert _is_safe_for_audit("enabled", True) is True
        assert _is_safe_for_audit("ratio", 3.14) is True

    def test_sensitive_names(self):
        assert _is_safe_for_audit("password", "secret") is False
        assert _is_safe_for_audit("api_token", "abc123") is False
        assert _is_safe_for_audit("secret_key", "key") is False
        assert _is_safe_for_audit("content", "hello") is False
        assert _is_safe_for_audit("auth_header", "bearer xyz") is False

    def test_complex_types(self):
        assert _is_safe_for_audit("items", [1, 2, 3]) is False
        assert _is_safe_for_audit("data", {"key": "value"}) is False

    def test_long_strings(self):
        long_string = "x" * 200
        assert _is_safe_for_audit("description", long_string) is False


class TestBypassConsentForTesting:
    """Test bypass_consent_for_testing context manager."""

    def teardown_method(self):
        reset_policy_engine()

    def test_bypass_allows_all(self, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")
        configure_policy_engine(consent_registry=registry)

        @requires_consent(source="any", scope="any")
        def protected_func():
            return "success"

        # Should fail without bypass
        with pytest.raises(ConsentDeniedError):
            protected_func()

        # Should succeed with bypass
        with bypass_consent_for_testing():
            result = protected_func()
            assert result == "success"

        # Should fail again after bypass
        with pytest.raises(ConsentDeniedError):
            protected_func()

    def test_bypass_restores_on_exception(self, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")
        configure_policy_engine(consent_registry=registry)

        @requires_consent(source="any", scope="any")
        def protected_func():
            return "success"

        try:
            with bypass_consent_for_testing():
                raise RuntimeError("test")
        except RuntimeError:
            pass

        # Should be back to normal after exception
        with pytest.raises(ConsentDeniedError):
            protected_func()


class TestDecoratorCombination:
    """Test combining decorators with other decorators."""

    @pytest.fixture
    def engine(self, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")
        registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")
        return configure_policy_engine(consent_registry=registry)

    def teardown_method(self):
        reset_policy_engine()

    def test_with_functools_wraps(self, engine):
        @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
        def documented_function(arg: str) -> str:
            """This is a docstring."""
            return arg

        # Should preserve function metadata
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a docstring."

    def test_stacked_decorators(self, tmp_path):
        audit_logger = MagicMock()
        registry = ConsentRegistry(tmp_path / "consent2")
        registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")
        configure_policy_engine(consent_registry=registry, audit_logger=audit_logger)

        @audit_action("outer_action")
        @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
        def doubly_decorated(item: str):
            return f"result:{item}"

        result = doubly_decorated("test")
        assert result == "result:test"

        # Both decorators should work
        audit_logger.record.assert_called()
