"""Privacy-aware decorators for functions and methods.

Provides decorators that enforce consent requirements and audit logging
on functions/methods that access sensitive data.

Example:
    from futurnal.privacy.decorators import requires_consent, audit_action

    @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
    @audit_action("extract_entities")
    async def extract_entities(vault_id: str, document_id: str):
        # This will only run if consent is granted
        # The action will be logged to the audit trail
        ...

    # You can also use dynamic source/scope extraction
    @requires_consent(source_arg="vault_id", scope="CONTENT_ANALYSIS")
    def process_vault(vault_id: str, content: str):
        # source is extracted from vault_id argument
        ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    TypeVar,
    Union,
    overload,
)

from .consent import ConsentRequiredError
from .policy_engine import PolicyEngine, get_policy_engine

if TYPE_CHECKING:
    from .audit import AuditLogger

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ConsentDeniedError(ConsentRequiredError):
    """Raised when a decorated function is called without required consent."""

    def __init__(
        self,
        source: str,
        scope: str,
        reason: Optional[str] = None,
    ):
        self.source = source
        self.scope = scope
        self.reason = reason or f"Consent required for {source}:{scope}"
        super().__init__(self.reason)


def requires_consent(
    source: Optional[str] = None,
    scope: Optional[str] = None,
    *,
    source_arg: Optional[str] = None,
    scope_arg: Optional[str] = None,
    source_kwarg: Optional[str] = None,
    scope_kwarg: Optional[str] = None,
    policy_engine: Optional[PolicyEngine] = None,
    error_class: type = ConsentDeniedError,
) -> Callable[[F], F]:
    """Decorator that requires consent before function execution.

    The source and scope can be specified directly or extracted from
    function arguments.

    Args:
        source: Static source identifier
        scope: Static scope identifier
        source_arg: Name of positional/keyword arg containing source
        scope_arg: Name of positional/keyword arg containing scope
        source_kwarg: Name of keyword-only arg containing source (deprecated, use source_arg)
        scope_kwarg: Name of keyword-only arg containing scope (deprecated, use scope_arg)
        policy_engine: PolicyEngine to use (defaults to global singleton)
        error_class: Exception class to raise on denial

    Returns:
        Decorated function

    Raises:
        ConsentDeniedError: If consent is not granted (or specified error_class)
        ValueError: If neither static value nor arg name is provided

    Example:
        @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
        def analyze_content(doc_id: str):
            ...

        @requires_consent(source_arg="source_id", scope="METADATA_ACCESS")
        def get_metadata(source_id: str, item_id: str):
            ...
    """
    # Handle deprecated kwargs
    actual_source_arg = source_arg or source_kwarg
    actual_scope_arg = scope_arg or scope_kwarg

    if source is None and actual_source_arg is None:
        raise ValueError("Either 'source' or 'source_arg' must be provided")
    if scope is None and actual_scope_arg is None:
        raise ValueError("Either 'scope' or 'scope_arg' must be provided")

    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        def _extract_arg_value(
            arg_name: str, args: tuple, kwargs: dict
        ) -> Optional[str]:
            """Extract argument value from args/kwargs."""
            # Check kwargs first
            if arg_name in kwargs:
                return str(kwargs[arg_name])

            # Check positional args
            try:
                idx = param_names.index(arg_name)
                if idx < len(args):
                    return str(args[idx])
            except ValueError:
                pass

            return None

        def _get_source_scope(args: tuple, kwargs: dict) -> tuple[str, str]:
            """Get source and scope values."""
            actual_source = source
            actual_scope = scope

            if actual_source_arg:
                extracted = _extract_arg_value(actual_source_arg, args, kwargs)
                if extracted is None:
                    raise ValueError(
                        f"Could not extract source from argument '{actual_source_arg}'"
                    )
                actual_source = extracted

            if actual_scope_arg:
                extracted = _extract_arg_value(actual_scope_arg, args, kwargs)
                if extracted is None:
                    raise ValueError(
                        f"Could not extract scope from argument '{actual_scope_arg}'"
                    )
                actual_scope = extracted

            return actual_source, actual_scope  # type: ignore

        def _check_consent(actual_source: str, actual_scope: str) -> None:
            """Check consent and raise if denied."""
            engine = policy_engine or get_policy_engine()
            result = engine.check_consent(actual_source, actual_scope)

            if not result.allowed:
                raise error_class(actual_source, actual_scope, result.reason)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                actual_source, actual_scope = _get_source_scope(args, kwargs)
                _check_consent(actual_source, actual_scope)
                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                actual_source, actual_scope = _get_source_scope(args, kwargs)
                _check_consent(actual_source, actual_scope)
                return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator


def audit_action(
    action: Optional[str] = None,
    *,
    action_arg: Optional[str] = None,
    source_arg: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False,
    include_timing: bool = True,
    audit_logger: Optional["AuditLogger"] = None,
    on_error: bool = True,
) -> Callable[[F], F]:
    """Decorator that logs function calls to the audit trail.

    Args:
        action: Static action name (defaults to function name)
        action_arg: Name of arg containing action name
        source_arg: Name of arg containing source identifier
        include_args: Include function arguments in audit (be careful with PII!)
        include_result: Include function return value in audit
        include_timing: Include execution time in audit
        audit_logger: AuditLogger to use (will use policy engine's logger if not provided)
        on_error: Log on error too

    Returns:
        Decorated function

    Example:
        @audit_action("search_query")
        def search(query: str):
            ...

        @audit_action(include_timing=True)
        async def process_document(doc_id: str):
            ...
    """

    def decorator(func: F) -> F:
        func_name = func.__name__
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        def _extract_arg_value(
            arg_name: str, args: tuple, kwargs: dict
        ) -> Optional[str]:
            """Extract argument value from args/kwargs."""
            if arg_name in kwargs:
                return str(kwargs[arg_name])

            try:
                idx = param_names.index(arg_name)
                if idx < len(args):
                    return str(args[idx])
            except ValueError:
                pass

            return None

        def _get_audit_logger() -> Optional["AuditLogger"]:
            """Get audit logger."""
            if audit_logger is not None:
                return audit_logger

            # Try to get from policy engine
            engine = get_policy_engine()
            return engine._audit_logger

        def _build_metadata(
            args: tuple,
            kwargs: dict,
            result: Any = None,
            error: Optional[Exception] = None,
            duration_ms: Optional[float] = None,
        ) -> Dict[str, Any]:
            """Build audit metadata."""
            metadata: Dict[str, Any] = {}

            if include_args:
                # Be careful - don't include sensitive data
                # Only include safe argument names
                safe_args = {}
                for i, value in enumerate(args):
                    if i < len(param_names):
                        name = param_names[i]
                        # Only include if it looks like an ID or safe value
                        if _is_safe_for_audit(name, value):
                            safe_args[name] = str(value)

                for name, value in kwargs.items():
                    if _is_safe_for_audit(name, value):
                        safe_args[name] = str(value)

                if safe_args:
                    metadata["args"] = safe_args

            if include_result and result is not None and error is None:
                # Only include simple results
                if isinstance(result, (str, int, float, bool)):
                    metadata["result"] = result
                elif isinstance(result, (list, tuple)):
                    metadata["result_count"] = len(result)
                elif hasattr(result, "__len__"):
                    metadata["result_count"] = len(result)

            if include_timing and duration_ms is not None:
                metadata["duration_ms"] = round(duration_ms, 2)

            if error is not None:
                metadata["error"] = type(error).__name__
                metadata["error_message"] = str(error)[:200]  # Truncate

            return metadata

        def _log_audit(
            args: tuple,
            kwargs: dict,
            result: Any = None,
            error: Optional[Exception] = None,
            duration_ms: Optional[float] = None,
        ) -> None:
            """Log to audit trail."""
            from datetime import datetime
            from .audit import AuditEvent

            audit = _get_audit_logger()
            if audit is None:
                return

            actual_action = action or func_name
            if action_arg:
                extracted = _extract_arg_value(action_arg, args, kwargs)
                if extracted:
                    actual_action = extracted

            source = None
            if source_arg:
                source = _extract_arg_value(source_arg, args, kwargs)

            status = "success" if error is None else "error"
            metadata = _build_metadata(args, kwargs, result, error, duration_ms)

            audit.record(
                AuditEvent(
                    job_id=f"audit_{actual_action}_{int(datetime.utcnow().timestamp())}",
                    source=source or "decorator",
                    action=actual_action,
                    status=status,
                    timestamp=datetime.utcnow(),
                    metadata=metadata if metadata else {},
                )
            )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter() if include_timing else 0
                error = None
                result = None

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    if error is None or on_error:
                        duration_ms = (
                            (time.perf_counter() - start_time) * 1000
                            if include_timing
                            else None
                        )
                        _log_audit(args, kwargs, result, error, duration_ms)

            return async_wrapper  # type: ignore

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter() if include_timing else 0
                error = None
                result = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    if error is None or on_error:
                        duration_ms = (
                            (time.perf_counter() - start_time) * 1000
                            if include_timing
                            else None
                        )
                        _log_audit(args, kwargs, result, error, duration_ms)

            return sync_wrapper  # type: ignore

    return decorator


def _is_safe_for_audit(name: str, value: Any) -> bool:
    """Check if a value is safe to include in audit logs.

    This is a heuristic to avoid logging sensitive data.
    """
    # Skip if name suggests sensitive data
    sensitive_names = {
        "password",
        "secret",
        "token",
        "key",
        "credential",
        "auth",
        "content",
        "body",
        "text",
        "data",
        "payload",
    }

    name_lower = name.lower()
    for sensitive in sensitive_names:
        if sensitive in name_lower:
            return False

    # Skip complex types
    if not isinstance(value, (str, int, float, bool)):
        return False

    # Skip long strings (might be content)
    if isinstance(value, str) and len(value) > 100:
        return False

    return True


def privacy_protected(
    source: Optional[str] = None,
    scope: Optional[str] = None,
    action: Optional[str] = None,
    *,
    source_arg: Optional[str] = None,
    scope_arg: Optional[str] = None,
) -> Callable[[F], F]:
    """Combined decorator for consent checking and audit logging.

    Convenience decorator that applies both @requires_consent and @audit_action.

    Args:
        source: Static source identifier
        scope: Static scope identifier
        action: Audit action name (defaults to function name)
        source_arg: Name of arg containing source
        scope_arg: Name of arg containing scope

    Returns:
        Decorated function

    Example:
        @privacy_protected(source="obsidian", scope="CONTENT_ANALYSIS")
        async def analyze_vault(vault_id: str):
            # Requires consent and logs the action
            ...
    """

    def decorator(func: F) -> F:
        # Apply audit first (inner), then consent (outer)
        audited = audit_action(
            action=action,
            source_arg=source_arg,
            include_timing=True,
        )(func)

        protected = requires_consent(
            source=source,
            scope=scope,
            source_arg=source_arg,
            scope_arg=scope_arg,
        )(audited)

        return protected  # type: ignore

    return decorator


# Context manager for temporary consent bypass (testing only)
class _ConsentBypass:
    """Context manager for bypassing consent checks (testing only).

    WARNING: Only use this in tests!
    """

    def __init__(self):
        self._original_check = None

    def __enter__(self):
        from .policy_engine import PolicyEngine, PolicyResult, PolicyDecision

        # Store original method
        self._original_check = PolicyEngine.check_consent

        # Replace with bypass
        def bypass_check(
            self_engine,
            source: str,
            scope: str,
            **kwargs,
        ) -> PolicyResult:
            return PolicyResult(
                decision=PolicyDecision.ALLOW,
                source=source,
                scope=scope,
                reason="Bypassed for testing",
            )

        PolicyEngine.check_consent = bypass_check
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from .policy_engine import PolicyEngine

        if self._original_check is not None:
            PolicyEngine.check_consent = self._original_check

        return False


def bypass_consent_for_testing():
    """Context manager to bypass consent checks in tests.

    WARNING: Only use this in tests!

    Example:
        with bypass_consent_for_testing():
            # Consent checks are bypassed here
            result = some_protected_function()
    """
    return _ConsentBypass()
