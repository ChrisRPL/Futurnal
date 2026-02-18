"""Tests for policy engine module."""

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from futurnal.privacy.policy_engine import (
    PolicyEngine,
    PolicyResult,
    PolicyDecision,
    PolicyRule,
    ConsentCache,
    get_policy_engine,
    configure_policy_engine,
    reset_policy_engine,
    create_time_restriction_rule,
    create_scope_restriction_rule,
    create_emergency_lockdown_rule,
)
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError


class TestConsentCache:
    """Test ConsentCache functionality."""

    def test_get_miss(self):
        cache = ConsentCache(ttl_seconds=60)
        result = cache.get("source", "scope")
        assert result is None

    def test_set_and_get(self):
        cache = ConsentCache(ttl_seconds=60)

        result = PolicyResult(
            decision=PolicyDecision.ALLOW,
            source="test_source",
            scope="test_scope",
        )
        cache.set(result)

        cached = cache.get("test_source", "test_scope")
        assert cached is not None
        assert cached.decision == PolicyDecision.ALLOW
        assert cached.cached is True

    def test_ttl_expiration(self):
        cache = ConsentCache(ttl_seconds=0)  # Immediate expiration

        result = PolicyResult(
            decision=PolicyDecision.ALLOW,
            source="test_source",
            scope="test_scope",
        )
        cache.set(result)

        # Should be expired
        time.sleep(0.01)
        cached = cache.get("test_source", "test_scope")
        assert cached is None

    def test_invalidate_all(self):
        cache = ConsentCache(ttl_seconds=60)

        for i in range(3):
            result = PolicyResult(
                decision=PolicyDecision.ALLOW,
                source=f"source_{i}",
                scope="scope",
            )
            cache.set(result)

        count = cache.invalidate()
        assert count == 3

        # All entries should be gone
        for i in range(3):
            assert cache.get(f"source_{i}", "scope") is None

    def test_invalidate_by_source(self):
        cache = ConsentCache(ttl_seconds=60)

        # Add entries for multiple sources
        for source in ["a", "b"]:
            for scope in ["x", "y"]:
                result = PolicyResult(
                    decision=PolicyDecision.ALLOW,
                    source=source,
                    scope=scope,
                )
                cache.set(result)

        count = cache.invalidate(source="a")
        assert count == 2

        # Source "a" entries should be gone
        assert cache.get("a", "x") is None
        assert cache.get("a", "y") is None

        # Source "b" entries should remain
        assert cache.get("b", "x") is not None
        assert cache.get("b", "y") is not None


class TestPolicyResult:
    """Test PolicyResult functionality."""

    def test_allowed_property(self):
        allow = PolicyResult(
            decision=PolicyDecision.ALLOW,
            source="src",
            scope="scope",
        )
        assert allow.allowed is True

        deny = PolicyResult(
            decision=PolicyDecision.DENY,
            source="src",
            scope="scope",
        )
        assert deny.allowed is False


class TestPolicyRule:
    """Test PolicyRule functionality."""

    def test_evaluate_match(self):
        rule = PolicyRule(
            name="test_rule",
            priority=100,
            condition=lambda s, sc, ctx: s == "blocked_source",
            decision=PolicyDecision.DENY_POLICY_RULE,
            reason="Source is blocked",
        )

        result = rule.evaluate("blocked_source", "scope", {})
        assert result == PolicyDecision.DENY_POLICY_RULE

    def test_evaluate_no_match(self):
        rule = PolicyRule(
            name="test_rule",
            priority=100,
            condition=lambda s, sc, ctx: s == "blocked_source",
            decision=PolicyDecision.DENY_POLICY_RULE,
            reason="Source is blocked",
        )

        result = rule.evaluate("allowed_source", "scope", {})
        assert result is None

    def test_evaluate_disabled(self):
        rule = PolicyRule(
            name="test_rule",
            priority=100,
            condition=lambda s, sc, ctx: True,
            decision=PolicyDecision.DENY_POLICY_RULE,
            reason="Always deny",
            enabled=False,
        )

        result = rule.evaluate("any_source", "scope", {})
        assert result is None

    def test_evaluate_exception(self):
        rule = PolicyRule(
            name="test_rule",
            priority=100,
            condition=lambda s, sc, ctx: 1 / 0,  # Will raise
            decision=PolicyDecision.DENY_POLICY_RULE,
            reason="Broken rule",
        )

        result = rule.evaluate("source", "scope", {})
        assert result is None  # Should return None on exception


class TestPolicyEngine:
    """Test PolicyEngine functionality."""

    @pytest.fixture
    def consent_registry(self, tmp_path):
        """Create a consent registry."""
        return ConsentRegistry(tmp_path / "consent")

    @pytest.fixture
    def engine(self, consent_registry):
        """Create a policy engine."""
        return PolicyEngine(consent_registry=consent_registry)

    def test_check_consent_allowed(self, engine, consent_registry):
        # Grant consent
        consent_registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        result = engine.check_consent("obsidian", "CONTENT_ANALYSIS")

        assert result.allowed is True
        assert result.decision == PolicyDecision.ALLOW

    def test_check_consent_denied_no_record(self, engine):
        result = engine.check_consent("unknown", "CONTENT_ANALYSIS")

        assert result.allowed is False
        assert result.decision == PolicyDecision.DENY_NO_CONSENT

    def test_check_consent_denied_not_granted(self, engine, consent_registry):
        # Revoke consent (creates record with granted=False)
        consent_registry.revoke(source="obsidian", scope="CONTENT_ANALYSIS")

        result = engine.check_consent("obsidian", "CONTENT_ANALYSIS")

        assert result.allowed is False
        assert result.decision == PolicyDecision.DENY_NO_CONSENT

    def test_check_consent_caching(self, engine, consent_registry):
        consent_registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        # First check
        result1 = engine.check_consent("obsidian", "CONTENT_ANALYSIS")
        assert result1.cached is False

        # Second check should be cached
        result2 = engine.check_consent("obsidian", "CONTENT_ANALYSIS")
        assert result2.cached is True

        # Stats should reflect cache hit
        stats = engine.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

    def test_check_consent_skip_cache(self, engine, consent_registry):
        consent_registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        # First check
        engine.check_consent("obsidian", "CONTENT_ANALYSIS")

        # Second check with skip_cache
        result = engine.check_consent("obsidian", "CONTENT_ANALYSIS", skip_cache=True)
        assert result.cached is False

    def test_require_consent_success(self, engine, consent_registry):
        consent_registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        result = engine.require_consent("obsidian", "CONTENT_ANALYSIS")
        assert result.allowed is True

    def test_require_consent_raises(self, engine):
        with pytest.raises(ConsentRequiredError):
            engine.require_consent("unknown", "CONTENT_ANALYSIS")

    def test_add_and_evaluate_rule(self, engine, consent_registry):
        # Grant consent
        consent_registry.grant(source="blocked", scope="CONTENT_ANALYSIS")

        # Add blocking rule
        rule = PolicyRule(
            name="block_source",
            priority=100,
            condition=lambda s, sc, ctx: s == "blocked",
            decision=PolicyDecision.DENY_POLICY_RULE,
            reason="Source is blocked",
        )
        engine.add_rule(rule)

        # Check - should be denied by rule even though consent exists
        result = engine.check_consent("blocked", "CONTENT_ANALYSIS")

        assert result.allowed is False
        assert result.decision == PolicyDecision.DENY_POLICY_RULE
        assert "blocked" in result.reason

    def test_rule_priority_order(self, engine):
        # Add rules in non-priority order
        low_priority = PolicyRule(
            name="low",
            priority=10,
            condition=lambda s, sc, ctx: True,
            decision=PolicyDecision.DENY,
            reason="Low priority",
        )

        high_priority = PolicyRule(
            name="high",
            priority=100,
            condition=lambda s, sc, ctx: True,
            decision=PolicyDecision.ALLOW,
            reason="High priority",
        )

        engine.add_rule(low_priority)
        engine.add_rule(high_priority)

        # High priority should win
        result = engine.check_consent("any", "any")
        assert result.decision == PolicyDecision.ALLOW

    def test_remove_rule(self, engine):
        rule = PolicyRule(
            name="test_rule",
            priority=100,
            condition=lambda s, sc, ctx: True,
            decision=PolicyDecision.DENY,
            reason="Test",
        )
        engine.add_rule(rule)

        assert engine.remove_rule("test_rule") is True
        assert engine.remove_rule("test_rule") is False  # Already removed

    def test_invalidate_cache(self, engine, consent_registry):
        consent_registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        # Cache the result
        engine.check_consent("obsidian", "CONTENT_ANALYSIS")

        # Invalidate
        count = engine.invalidate_cache()
        assert count == 1

        # Next check should miss cache
        result = engine.check_consent("obsidian", "CONTENT_ANALYSIS")
        assert result.cached is False

    def test_no_registry_configured(self):
        engine = PolicyEngine()  # No consent registry

        result = engine.check_consent("any", "any")

        assert result.allowed is False
        assert result.decision == PolicyDecision.DENY_NO_CONSENT
        assert "No consent registry" in result.reason

    def test_configure_after_init(self, consent_registry):
        engine = PolicyEngine()

        # Initially denied
        result1 = engine.check_consent("obsidian", "CONTENT_ANALYSIS")
        assert result1.allowed is False

        # Configure with registry and grant consent
        engine.configure(consent_registry=consent_registry)
        consent_registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        # Now should be allowed
        result2 = engine.check_consent("obsidian", "CONTENT_ANALYSIS")
        assert result2.allowed is True

    def test_stats_tracking(self, engine, consent_registry):
        consent_registry.grant(source="allowed", scope="CONTENT_ANALYSIS")

        engine.check_consent("allowed", "CONTENT_ANALYSIS")  # Allow
        engine.check_consent("denied", "CONTENT_ANALYSIS")  # Deny

        stats = engine.get_stats()
        assert stats["total_checks"] == 2
        assert stats["denials"] == 1

    def test_reset_stats(self, engine):
        engine._stats["total_checks"] = 100
        engine.reset_stats()
        assert engine.get_stats()["total_checks"] == 0


class TestSingletonFunctions:
    """Test singleton management functions."""

    def setup_method(self):
        reset_policy_engine()

    def teardown_method(self):
        reset_policy_engine()

    def test_get_policy_engine(self):
        engine1 = get_policy_engine()
        engine2 = get_policy_engine()

        assert engine1 is engine2

    def test_configure_policy_engine(self, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")

        engine = configure_policy_engine(consent_registry=registry)

        assert engine._consent_registry is registry
        assert get_policy_engine() is engine

    def test_reset_policy_engine(self):
        engine1 = get_policy_engine()
        reset_policy_engine()
        engine2 = get_policy_engine()

        assert engine1 is not engine2


class TestPredefinedRules:
    """Test predefined rule factory functions."""

    def test_time_restriction_rule_outside_hours(self):
        # Create rule allowing only 9-17
        rule = create_time_restriction_rule(
            name="business_hours",
            allowed_hours=[(9, 17)],
        )

        # Test at midnight (should deny)
        with patch("futurnal.privacy.policy_engine.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.hour = 0  # Midnight
            mock_dt.now.return_value = mock_now

            result = rule.evaluate("any", "any", {})
            assert result == PolicyDecision.DENY_POLICY_RULE

    def test_time_restriction_rule_within_hours(self):
        rule = create_time_restriction_rule(
            name="business_hours",
            allowed_hours=[(9, 17)],
        )

        # Test at noon (should allow)
        with patch("futurnal.privacy.policy_engine.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.hour = 12  # Noon
            mock_dt.now.return_value = mock_now

            result = rule.evaluate("any", "any", {})
            assert result is None  # No denial

    def test_scope_restriction_rule(self):
        rule = create_scope_restriction_rule(
            name="deny_cloud",
            denied_scopes={"CLOUD_PROCESSING"},
        )

        # Denied scope
        result = rule.evaluate("any", "CLOUD_PROCESSING", {})
        assert result == PolicyDecision.DENY_POLICY_RULE

        # Allowed scope
        result = rule.evaluate("any", "LOCAL_PROCESSING", {})
        assert result is None

    def test_emergency_lockdown_rule(self):
        rule = create_emergency_lockdown_rule()

        # Should be disabled by default
        result = rule.evaluate("any", "any", {})
        assert result is None

        # Enable it
        rule.enabled = True
        result = rule.evaluate("any", "any", {})
        assert result == PolicyDecision.DENY_POLICY_RULE


class TestAnomalyIntegration:
    """Test integration with anomaly detector."""

    def test_denial_records_anomaly(self, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")
        anomaly_detector = MagicMock()

        engine = PolicyEngine(
            consent_registry=registry,
            anomaly_detector=anomaly_detector,
        )

        # Check without consent
        engine.check_consent("source", "scope")

        # Should have recorded anomaly
        anomaly_detector.record_consent_violation.assert_called_once_with(
            "source", scope="scope"
        )


class TestAuditIntegration:
    """Test integration with audit logger."""

    def test_check_logs_to_audit(self, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")
        audit_logger = MagicMock()

        engine = PolicyEngine(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

        engine.check_consent("source", "scope")

        audit_logger.record.assert_called_once()
        call_args = audit_logger.record.call_args[0][0]  # AuditEvent object
        assert call_args.action == "consent_check"
        assert call_args.source == "source"

    def test_check_without_audit(self, tmp_path):
        registry = ConsentRegistry(tmp_path / "consent")
        audit_logger = MagicMock()

        engine = PolicyEngine(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

        engine.check_consent("source", "scope", audit=False)

        audit_logger.record.assert_not_called()
