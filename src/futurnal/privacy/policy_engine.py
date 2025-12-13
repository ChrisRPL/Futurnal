"""Centralized Policy Engine for privacy policy enforcement.

Provides a singleton-based policy engine that:
- Caches consent checks for performance (<1ms with cache)
- Evaluates policies with consistent semantics
- Integrates with consent registry and audit logging
- Supports policy rules with conditions

Example:
    from futurnal.privacy.policy_engine import get_policy_engine

    engine = get_policy_engine()
    if engine.check_consent("obsidian", "CONTENT_ANALYSIS"):
        # Proceed with content analysis
        ...
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .consent import ConsentRegistry
    from .anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Result of a policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    DENY_NO_CONSENT = "deny_no_consent"
    DENY_EXPIRED = "deny_expired"
    DENY_POLICY_RULE = "deny_policy_rule"


@dataclass
class PolicyResult:
    """Result of a policy evaluation with details."""

    decision: PolicyDecision
    source: str
    scope: str
    reason: Optional[str] = None
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    cached: bool = False

    @property
    def allowed(self) -> bool:
        """Check if the policy allows the operation."""
        return self.decision == PolicyDecision.ALLOW


@dataclass
class PolicyRule:
    """A rule that can modify policy decisions.

    Rules are evaluated in priority order (highest first).
    """

    name: str
    priority: int
    condition: Callable[[str, str, Dict[str, Any]], bool]
    decision: PolicyDecision
    reason: str
    enabled: bool = True

    def evaluate(
        self, source: str, scope: str, context: Dict[str, Any]
    ) -> Optional[PolicyDecision]:
        """Evaluate the rule.

        Returns:
            PolicyDecision if rule matches, None otherwise
        """
        if not self.enabled:
            return None

        try:
            if self.condition(source, scope, context):
                return self.decision
        except Exception as e:
            logger.warning(f"Rule '{self.name}' evaluation failed: {e}")

        return None


class ConsentCache:
    """Thread-safe cache for consent decisions.

    Provides <1ms policy evaluation through caching.
    """

    def __init__(self, ttl_seconds: int = 60):
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached entries
        """
        self._cache: Dict[str, tuple[PolicyResult, datetime]] = {}
        self._lock = threading.RLock()
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, source: str, scope: str) -> Optional[PolicyResult]:
        """Get cached result if valid."""
        key = self._key(source, scope)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            result, timestamp = entry
            if datetime.utcnow() - timestamp > self._ttl:
                del self._cache[key]
                return None

            # Mark as cached
            return PolicyResult(
                decision=result.decision,
                source=result.source,
                scope=result.scope,
                reason=result.reason,
                evaluated_at=result.evaluated_at,
                cached=True,
            )

    def set(self, result: PolicyResult) -> None:
        """Cache a policy result."""
        key = self._key(result.source, result.scope)
        with self._lock:
            self._cache[key] = (result, datetime.utcnow())

    def invalidate(self, source: Optional[str] = None, scope: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            source: Invalidate entries for this source (None = all sources)
            scope: Invalidate entries for this scope (None = all scopes)

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if source is None and scope is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            to_remove = []
            for key in self._cache:
                key_source, key_scope = key.split(":", 1)
                if source is not None and key_source != source:
                    continue
                if scope is not None and key_scope != scope:
                    continue
                to_remove.append(key)

            for key in to_remove:
                del self._cache[key]

            return len(to_remove)

    def _key(self, source: str, scope: str) -> str:
        return f"{source}:{scope}"


class PolicyEngine:
    """Centralized policy enforcement engine.

    Provides unified policy evaluation with:
    - Consent checking via ConsentRegistry
    - Custom policy rules
    - Caching for performance
    - Audit integration
    - Anomaly detection integration
    """

    def __init__(
        self,
        consent_registry: Optional["ConsentRegistry"] = None,
        audit_logger: Optional["AuditLogger"] = None,
        anomaly_detector: Optional["AnomalyDetector"] = None,
        cache_ttl_seconds: int = 60,
    ):
        """Initialize policy engine.

        Args:
            consent_registry: ConsentRegistry for consent checks
            audit_logger: AuditLogger for audit events
            anomaly_detector: AnomalyDetector for anomaly tracking
            cache_ttl_seconds: TTL for consent cache
        """
        self._consent_registry = consent_registry
        self._audit_logger = audit_logger
        self._anomaly_detector = anomaly_detector
        self._cache = ConsentCache(ttl_seconds=cache_ttl_seconds)
        self._rules: List[PolicyRule] = []
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "total_checks": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "denials": 0,
            "rule_denials": 0,
        }

    def configure(
        self,
        consent_registry: Optional["ConsentRegistry"] = None,
        audit_logger: Optional["AuditLogger"] = None,
        anomaly_detector: Optional["AnomalyDetector"] = None,
    ) -> None:
        """Configure the policy engine with dependencies.

        Can be called after initialization to set or update dependencies.
        """
        if consent_registry is not None:
            self._consent_registry = consent_registry
            self._cache.invalidate()  # Clear cache on registry change

        if audit_logger is not None:
            self._audit_logger = audit_logger

        if anomaly_detector is not None:
            self._anomaly_detector = anomaly_detector

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a policy rule.

        Rules are evaluated in priority order (highest first).
        """
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: r.priority, reverse=True)
            logger.debug(f"Added policy rule: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        """Remove a policy rule by name.

        Returns:
            True if rule was removed, False if not found
        """
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.name == name:
                    self._rules.pop(i)
                    logger.debug(f"Removed policy rule: {name}")
                    return True
        return False

    def check_consent(
        self,
        source: str,
        scope: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False,
        audit: bool = True,
    ) -> PolicyResult:
        """Check if an operation is allowed by policy.

        This is the main entry point for policy checks.

        Args:
            source: Data source identifier
            scope: Operation scope (e.g., CONTENT_ANALYSIS, METADATA_ACCESS)
            context: Additional context for rule evaluation
            skip_cache: Skip cache lookup
            audit: Log the check to audit log

        Returns:
            PolicyResult with decision and details
        """
        self._stats["total_checks"] += 1
        context = context or {}

        # Check cache first
        if not skip_cache:
            cached = self._cache.get(source, scope)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return cached

        self._stats["cache_misses"] += 1

        # Evaluate policy rules first
        result = self._evaluate_rules(source, scope, context)

        # If no rule matched, check consent registry
        if result is None:
            result = self._check_consent_registry(source, scope)

        # Cache the result
        self._cache.set(result)

        # Track denials
        if not result.allowed:
            self._stats["denials"] += 1

            # Record anomaly for consent violations
            if (
                self._anomaly_detector is not None
                and result.decision == PolicyDecision.DENY_NO_CONSENT
            ):
                self._anomaly_detector.record_consent_violation(source, scope=scope)

        # Audit the check
        if audit and self._audit_logger is not None:
            self._audit_policy_check(result)

        return result

    def require_consent(
        self,
        source: str,
        scope: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyResult:
        """Require consent, raising ConsentRequiredError if not granted.

        Args:
            source: Data source identifier
            scope: Operation scope

        Returns:
            PolicyResult (always ALLOW if no exception raised)

        Raises:
            ConsentRequiredError: If consent is not granted
        """
        from .consent import ConsentRequiredError

        result = self.check_consent(source, scope, context=context)
        if not result.allowed:
            reason = result.reason or f"Consent required for {source}:{scope}"
            raise ConsentRequiredError(reason)

        return result

    def invalidate_cache(
        self, source: Optional[str] = None, scope: Optional[str] = None
    ) -> int:
        """Invalidate cache entries.

        Call this when consent state changes.

        Returns:
            Number of entries invalidated
        """
        return self._cache.invalidate(source, scope)

    def get_stats(self) -> Dict[str, int]:
        """Get policy engine statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0

    def _evaluate_rules(
        self, source: str, scope: str, context: Dict[str, Any]
    ) -> Optional[PolicyResult]:
        """Evaluate policy rules.

        Returns:
            PolicyResult if a rule matched, None otherwise
        """
        with self._lock:
            for rule in self._rules:
                decision = rule.evaluate(source, scope, context)
                if decision is not None:
                    if decision != PolicyDecision.ALLOW:
                        self._stats["rule_denials"] += 1

                    return PolicyResult(
                        decision=decision,
                        source=source,
                        scope=scope,
                        reason=rule.reason,
                    )

        return None

    def _check_consent_registry(self, source: str, scope: str) -> PolicyResult:
        """Check consent registry for consent."""
        if self._consent_registry is None:
            # No registry configured - default deny
            return PolicyResult(
                decision=PolicyDecision.DENY_NO_CONSENT,
                source=source,
                scope=scope,
                reason="No consent registry configured",
            )

        record = self._consent_registry.get(source=source, scope=scope)

        if record is None:
            return PolicyResult(
                decision=PolicyDecision.DENY_NO_CONSENT,
                source=source,
                scope=scope,
                reason="No consent record found",
            )

        if not record.granted:
            return PolicyResult(
                decision=PolicyDecision.DENY_NO_CONSENT,
                source=source,
                scope=scope,
                reason="Consent not granted",
            )

        if not record.is_active():
            return PolicyResult(
                decision=PolicyDecision.DENY_EXPIRED,
                source=source,
                scope=scope,
                reason="Consent expired",
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            source=source,
            scope=scope,
        )

    def _audit_policy_check(self, result: PolicyResult) -> None:
        """Log policy check to audit log."""
        if self._audit_logger is None:
            return

        from .audit import AuditEvent

        action = "consent_check"
        status = "allowed" if result.allowed else "denied"

        self._audit_logger.record(
            AuditEvent(
                job_id=f"consent_check_{result.source}_{int(datetime.utcnow().timestamp())}",
                source=result.source,
                action=action,
                status=status,
                timestamp=datetime.utcnow(),
                metadata={
                    "scope": result.scope,
                    "decision": result.decision.value,
                    "reason": result.reason,
                    "cached": result.cached,
                },
            )
        )


# Global singleton instance
_policy_engine: Optional[PolicyEngine] = None
_engine_lock = threading.Lock()


def get_policy_engine() -> PolicyEngine:
    """Get the global policy engine singleton.

    Creates a new instance if one doesn't exist.

    Returns:
        PolicyEngine singleton
    """
    global _policy_engine

    if _policy_engine is None:
        with _engine_lock:
            if _policy_engine is None:
                _policy_engine = PolicyEngine()

    return _policy_engine


def configure_policy_engine(
    consent_registry: Optional["ConsentRegistry"] = None,
    audit_logger: Optional["AuditLogger"] = None,
    anomaly_detector: Optional["AnomalyDetector"] = None,
    cache_ttl_seconds: int = 60,
) -> PolicyEngine:
    """Configure the global policy engine.

    Creates a new engine instance with the provided configuration.

    Args:
        consent_registry: ConsentRegistry for consent checks
        audit_logger: AuditLogger for audit events
        anomaly_detector: AnomalyDetector for anomaly tracking
        cache_ttl_seconds: TTL for consent cache

    Returns:
        Configured PolicyEngine singleton
    """
    global _policy_engine

    with _engine_lock:
        _policy_engine = PolicyEngine(
            consent_registry=consent_registry,
            audit_logger=audit_logger,
            anomaly_detector=anomaly_detector,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    return _policy_engine


def reset_policy_engine() -> None:
    """Reset the global policy engine.

    Useful for testing.
    """
    global _policy_engine

    with _engine_lock:
        _policy_engine = None


# Common policy rules

def create_time_restriction_rule(
    name: str,
    allowed_hours: List[tuple[int, int]],
    sources: Optional[Set[str]] = None,
    priority: int = 100,
) -> PolicyRule:
    """Create a rule that restricts access to certain hours.

    Args:
        name: Rule name
        allowed_hours: List of (start_hour, end_hour) tuples (24-hour format)
        sources: Sources to apply to (None = all sources)
        priority: Rule priority

    Returns:
        PolicyRule for time restriction
    """

    def condition(source: str, scope: str, context: Dict[str, Any]) -> bool:
        if sources is not None and source not in sources:
            return False

        now = datetime.now()
        current_hour = now.hour

        for start, end in allowed_hours:
            if start <= current_hour < end:
                return False  # Within allowed hours, don't deny

        return True  # Outside allowed hours, deny

    return PolicyRule(
        name=name,
        priority=priority,
        condition=condition,
        decision=PolicyDecision.DENY_POLICY_RULE,
        reason=f"Access restricted to hours: {allowed_hours}",
    )


def create_scope_restriction_rule(
    name: str,
    denied_scopes: Set[str],
    sources: Optional[Set[str]] = None,
    priority: int = 100,
) -> PolicyRule:
    """Create a rule that denies certain scopes.

    Args:
        name: Rule name
        denied_scopes: Set of scopes to deny
        sources: Sources to apply to (None = all sources)
        priority: Rule priority

    Returns:
        PolicyRule for scope restriction
    """

    def condition(source: str, scope: str, context: Dict[str, Any]) -> bool:
        if sources is not None and source not in sources:
            return False

        return scope in denied_scopes

    return PolicyRule(
        name=name,
        priority=priority,
        condition=condition,
        decision=PolicyDecision.DENY_POLICY_RULE,
        reason="Scope is restricted by policy rule",
    )


def create_emergency_lockdown_rule(priority: int = 1000) -> PolicyRule:
    """Create an emergency lockdown rule that denies all access.

    Args:
        priority: Rule priority (default very high)

    Returns:
        PolicyRule for emergency lockdown
    """

    def condition(source: str, scope: str, context: Dict[str, Any]) -> bool:
        return True  # Always match

    return PolicyRule(
        name="emergency_lockdown",
        priority=priority,
        condition=condition,
        decision=PolicyDecision.DENY_POLICY_RULE,
        reason="Emergency lockdown active - all access denied",
        enabled=False,  # Disabled by default
    )
