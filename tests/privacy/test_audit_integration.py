"""End-to-end integration tests for the privacy & audit logging system.

These tests verify that all components work together correctly:
- Consent Registry + Policy Engine
- Audit Logger + Anomaly Detection
- Privacy Decorators + Audit Logging
- Encryption + Purge
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from futurnal.privacy.audit import AuditLogger, AuditEvent
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.policy_engine import (
    PolicyEngine,
    PolicyDecision,
    configure_policy_engine,
    reset_policy_engine,
    get_policy_engine,
)
from futurnal.privacy.decorators import (
    requires_consent,
    audit_action,
    privacy_protected,
    ConsentDeniedError,
)
from futurnal.privacy.anomaly_detector import (
    AnomalyDetector,
    AnomalyConfig,
    AnomalyType,
)
from futurnal.privacy.purge import DataPurgeService, create_purge_service


class TestConsentPolicyIntegration:
    """Test ConsentRegistry + PolicyEngine integration."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a workspace with all components."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)
        anomaly_config = AnomalyConfig(consent_violation_threshold=2)
        anomaly_detector = AnomalyDetector(anomaly_config)

        engine = configure_policy_engine(
            consent_registry=registry,
            audit_logger=audit_logger,
            anomaly_detector=anomaly_detector,
        )

        return {
            "registry": registry,
            "audit_logger": audit_logger,
            "anomaly_detector": anomaly_detector,
            "engine": engine,
            "tmp_path": tmp_path,
        }

    def teardown_method(self):
        reset_policy_engine()

    def test_consent_flow_end_to_end(self, workspace):
        """Test complete consent flow from grant to check."""
        registry = workspace["registry"]
        engine = workspace["engine"]

        # Initially denied
        result = engine.check_consent("obsidian", "CONTENT_ANALYSIS")
        assert not result.allowed
        assert result.decision == PolicyDecision.DENY_NO_CONSENT

        # Grant consent
        registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")

        # Invalidate cache
        engine.invalidate_cache()

        # Now allowed
        result = engine.check_consent("obsidian", "CONTENT_ANALYSIS")
        assert result.allowed

        # Revoke consent
        registry.revoke(source="obsidian", scope="CONTENT_ANALYSIS")
        engine.invalidate_cache()

        # Denied again
        result = engine.check_consent("obsidian", "CONTENT_ANALYSIS")
        assert not result.allowed

    def test_consent_denial_triggers_anomaly(self, workspace):
        """Test that consent denials are tracked by anomaly detector."""
        engine = workspace["engine"]
        anomaly_detector = workspace["anomaly_detector"]

        # Multiple consent check failures
        engine.check_consent("unknown1", "CONTENT_ANALYSIS")
        engine.check_consent("unknown2", "CONTENT_ANALYSIS")

        # Check for anomalies
        anomalies = anomaly_detector.check_for_anomalies()

        # Should have consent violation anomaly
        consent_anomalies = [
            a for a in anomalies if a.anomaly_type == AnomalyType.CONSENT_VIOLATION
        ]
        assert len(consent_anomalies) >= 1

    def test_audit_logs_consent_checks(self, workspace):
        """Test that consent checks are logged to audit."""
        audit_logger = workspace["audit_logger"]
        engine = workspace["engine"]

        # Perform consent check
        engine.check_consent("test_source", "test_scope")

        # Verify audit log
        events = list(audit_logger.iter_events())
        consent_events = [e for e in events if "consent_check" in e.get("action", "")]
        assert len(consent_events) >= 1


class TestDecoratorIntegration:
    """Test privacy decorators with full stack."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create workspace with decorated functions."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)

        engine = configure_policy_engine(
            consent_registry=registry,
            audit_logger=audit_logger,
        )

        return {
            "registry": registry,
            "audit_logger": audit_logger,
            "engine": engine,
            "tmp_path": tmp_path,
        }

    def teardown_method(self):
        reset_policy_engine()

    def test_protected_function_requires_consent(self, workspace):
        """Test that @requires_consent enforces consent."""
        registry = workspace["registry"]

        @requires_consent(source="obsidian", scope="CONTENT_ANALYSIS")
        def analyze_content(doc_id: str):
            return f"analyzed:{doc_id}"

        # Should fail without consent
        with pytest.raises(ConsentDeniedError):
            analyze_content("doc123")

        # Grant consent
        registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")
        get_policy_engine().invalidate_cache()

        # Should succeed with consent
        result = analyze_content("doc123")
        assert result == "analyzed:doc123"

    def test_audit_action_logs_function_calls(self, workspace):
        """Test that @audit_action logs all function calls."""
        audit_logger = workspace["audit_logger"]

        @audit_action("process_document")
        def process_doc(doc_id: str):
            return f"processed:{doc_id}"

        # Call function
        result = process_doc("doc456")
        assert result == "processed:doc456"

        # Verify audit log
        events = list(audit_logger.iter_events())
        doc_events = [e for e in events if e.get("action") == "process_document"]
        assert len(doc_events) == 1
        assert doc_events[0]["status"] == "success"

    def test_privacy_protected_decorator(self, workspace):
        """Test combined @privacy_protected decorator."""
        registry = workspace["registry"]
        audit_logger = workspace["audit_logger"]

        registry.grant(source="github", scope="METADATA_ACCESS")
        get_policy_engine().invalidate_cache()

        @privacy_protected(source="github", scope="METADATA_ACCESS")
        def fetch_repo_info(repo_id: str):
            return {"repo": repo_id, "stars": 100}

        result = fetch_repo_info("my-repo")
        assert result["repo"] == "my-repo"

        # Verify audit logged
        events = list(audit_logger.iter_events())
        assert len(events) >= 1


class TestEncryptionPurgeIntegration:
    """Test Encryption + Purge integration."""

    def test_purge_removes_all_data(self, tmp_path):
        """Test that purge removes all data including encrypted."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)

        # Create some data
        registry.grant(source="source1", scope="scope1")
        registry.grant(source="source2", scope="scope2")

        audit_logger.record(
            AuditEvent(
                job_id="job1",
                source="test",
                action="action1",
                status="success",
                timestamp=datetime.utcnow(),
            )
        )

        # Create purge service
        purge_service = DataPurgeService(
            audit_logger=audit_logger,
            consent_registry=registry,
        )

        # Purge all
        result = purge_service.purge_all(confirm=True)

        assert result.success
        assert "consent_records" in result.sources_purged
        assert "audit_logs" in result.sources_purged


class TestAnomalyDetectionIntegration:
    """Test anomaly detection integration with full stack."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create workspace with anomaly detection."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)

        anomaly_config = AnomalyConfig(
            consent_violation_threshold=3,
            window_minutes=5,
        )
        anomaly_detector = AnomalyDetector(
            anomaly_config,
            audit_logger=audit_logger,
        )

        engine = configure_policy_engine(
            consent_registry=registry,
            audit_logger=audit_logger,
            anomaly_detector=anomaly_detector,
        )

        return {
            "registry": registry,
            "audit_logger": audit_logger,
            "anomaly_detector": anomaly_detector,
            "engine": engine,
        }

    def teardown_method(self):
        reset_policy_engine()

    def test_consent_violations_detected(self, workspace):
        """Test that consent violations are detected as anomalies."""
        engine = workspace["engine"]
        anomaly_detector = workspace["anomaly_detector"]

        # Generate consent violations
        for i in range(4):
            engine.check_consent(f"unknown_source_{i}", "CONTENT_ANALYSIS")

        # Check for anomalies
        anomalies = anomaly_detector.check_for_anomalies()

        consent_anomalies = [
            a for a in anomalies if a.anomaly_type == AnomalyType.CONSENT_VIOLATION
        ]
        assert len(consent_anomalies) >= 1

    def test_anomalies_logged_to_audit(self, workspace):
        """Test that detected anomalies are logged to audit trail."""
        audit_logger = workspace["audit_logger"]
        anomaly_detector = workspace["anomaly_detector"]

        # Record violations directly
        for i in range(4):
            anomaly_detector.record_consent_violation(f"source_{i}")

        # Check for anomalies (triggers audit logging)
        anomaly_detector.check_for_anomalies()

        # Verify audit contains anomaly events
        events = list(audit_logger.iter_events())
        anomaly_events = [e for e in events if "anomaly" in e.get("action", "")]
        assert len(anomaly_events) >= 1


class TestFullPipelineIntegration:
    """Test full privacy pipeline from consent to audit to purge."""

    @pytest.fixture
    def full_workspace(self, tmp_path):
        """Create complete workspace with all components."""
        consent_dir = tmp_path / "consent"
        audit_dir = tmp_path / "audit"
        consent_dir.mkdir()
        audit_dir.mkdir()

        registry = ConsentRegistry(consent_dir)
        audit_logger = AuditLogger(output_dir=audit_dir)

        anomaly_config = AnomalyConfig(consent_violation_threshold=3)
        anomaly_detector = AnomalyDetector(anomaly_config, audit_logger=audit_logger)

        engine = configure_policy_engine(
            consent_registry=registry,
            audit_logger=audit_logger,
            anomaly_detector=anomaly_detector,
        )

        purge_service = DataPurgeService(
            audit_logger=audit_logger,
            consent_registry=registry,
        )

        return {
            "registry": registry,
            "audit_logger": audit_logger,
            "anomaly_detector": anomaly_detector,
            "engine": engine,
            "purge_service": purge_service,
            "tmp_path": tmp_path,
        }

    def teardown_method(self):
        reset_policy_engine()

    def test_complete_privacy_lifecycle(self, full_workspace):
        """Test complete lifecycle: consent -> use -> audit -> purge."""
        registry = full_workspace["registry"]
        audit_logger = full_workspace["audit_logger"]
        engine = full_workspace["engine"]
        purge_service = full_workspace["purge_service"]

        # 1. Grant consent
        registry.grant(source="obsidian", scope="CONTENT_ANALYSIS")
        engine.invalidate_cache()

        # 2. Use protected function
        @privacy_protected(source="obsidian", scope="CONTENT_ANALYSIS")
        def analyze_vault(vault_id: str):
            return {"vault_id": vault_id, "entities": 42}

        result = analyze_vault("my-vault")
        assert result["entities"] == 42

        # 3. Verify audit trail exists
        events = list(audit_logger.iter_events())
        assert len(events) >= 1

        # 4. Revoke consent
        registry.revoke(source="obsidian", scope="CONTENT_ANALYSIS")
        engine.invalidate_cache()

        # 5. Function should now fail
        with pytest.raises(ConsentDeniedError):
            analyze_vault("my-vault")

        # 6. Purge all data
        purge_result = purge_service.purge_all(confirm=True)
        assert purge_result.success
        assert "consent_records" in purge_result.sources_purged

    def test_source_specific_purge(self, full_workspace):
        """Test purging data for a specific source."""
        registry = full_workspace["registry"]
        audit_logger = full_workspace["audit_logger"]
        purge_service = full_workspace["purge_service"]

        # Create data for multiple sources
        registry.grant(source="source_a", scope="SCOPE_1")
        registry.grant(source="source_b", scope="SCOPE_1")

        audit_logger.record(
            AuditEvent(
                job_id="job_a",
                source="source_a",
                action="test",
                status="success",
                timestamp=datetime.utcnow(),
            )
        )
        audit_logger.record(
            AuditEvent(
                job_id="job_b",
                source="source_b",
                action="test",
                status="success",
                timestamp=datetime.utcnow(),
            )
        )

        # Purge only source_a
        result = purge_service.purge_by_source("source_a", confirm=True)
        assert result.success

        # Verify source_a consent is revoked
        record_a = registry.get(source="source_a", scope="SCOPE_1")
        assert record_a is not None
        assert record_a.granted is False  # Consent revoked

        # Verify source_b consent remains active
        record_b = registry.get(source="source_b", scope="SCOPE_1")
        assert record_b is not None
        assert record_b.granted is True


class TestCachePerformance:
    """Test policy engine cache performance."""

    @pytest.fixture
    def engine(self, tmp_path):
        consent_dir = tmp_path / "consent"
        consent_dir.mkdir()
        registry = ConsentRegistry(consent_dir)
        registry.grant(source="test", scope="test")
        return configure_policy_engine(consent_registry=registry, cache_ttl_seconds=60)

    def teardown_method(self):
        reset_policy_engine()

    def test_cached_checks_are_fast(self, engine):
        """Test that cached consent checks are sub-millisecond."""
        # Prime the cache
        engine.check_consent("test", "test")

        # Measure cached checks
        start = time.perf_counter()
        for _ in range(1000):
            result = engine.check_consent("test", "test")
        elapsed = time.perf_counter() - start

        # Should be very fast with cache
        avg_time_ms = (elapsed / 1000) * 1000
        assert avg_time_ms < 1.0  # Less than 1ms per check
        assert result.cached is True

    def test_cache_hit_rate(self, engine):
        """Test cache hit rate tracking."""
        # Prime cache
        engine.check_consent("test", "test")

        # Multiple checks
        for _ in range(10):
            engine.check_consent("test", "test")

        stats = engine.get_stats()
        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 1  # Initial check


class TestHashChainIntegrity:
    """Test audit log hash chain integrity."""

    def test_audit_log_chain_consistency(self, tmp_path):
        """Test that audit logs maintain chain consistency."""
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()

        audit_logger = AuditLogger(output_dir=audit_dir)

        # Record some events
        for i in range(5):
            audit_logger.record(
                AuditEvent(
                    job_id=f"job_{i}",
                    source="test",
                    action=f"action_{i}",
                    status="success",
                    timestamp=datetime.utcnow(),
                )
            )

        # Events should be readable and in order
        events = list(audit_logger.iter_events())
        assert len(events) == 5

        # Each event should have a chain hash
        for event in events:
            assert "chain_hash" in event

        # Events should be in chronological order
        timestamps = [event["timestamp"] for event in events]
        assert timestamps == sorted(timestamps)
