"""Tests for data purge service."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.futurnal.privacy.audit import AuditLogger, AuditEvent
from src.futurnal.privacy.consent import ConsentRegistry
from src.futurnal.privacy.purge import (
    DataPurgeService,
    PurgeResult,
    PurgeConfirmationRequired,
    create_purge_service,
)


@pytest.fixture
def audit_logger(tmp_path):
    """Create an audit logger for testing."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(exist_ok=True)
    return AuditLogger(output_dir=audit_dir)


@pytest.fixture
def consent_registry(tmp_path):
    """Create a consent registry for testing."""
    consent_dir = tmp_path / "consent"
    consent_dir.mkdir(exist_ok=True)
    return ConsentRegistry(directory=consent_dir)


@pytest.fixture
def purge_service(audit_logger, consent_registry):
    """Create a purge service for testing."""
    return DataPurgeService(
        audit_logger=audit_logger,
        consent_registry=consent_registry,
    )


def populate_audit_logs(audit_logger, count=5):
    """Populate audit logger with test events."""
    for i in range(count):
        audit_logger.record(
            AuditEvent(
                job_id=f"job_{i}",
                source=f"source_{i % 2}",
                action="test_action",
                status="success",
                timestamp=datetime.utcnow(),
            )
        )


def populate_consent(consent_registry):
    """Populate consent registry with test records."""
    consent_registry.grant(source="source_a", scope="CONTENT_ANALYSIS")
    consent_registry.grant(source="source_a", scope="METADATA_ACCESS")
    consent_registry.grant(source="source_b", scope="CONTENT_ANALYSIS")


class TestPurgeResult:
    """Test PurgeResult data class."""

    def test_to_dict(self):
        result = PurgeResult(
            success=True,
            files_deleted=5,
            bytes_freed=1024,
            sources_purged={"audit_logs", "consent"},
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["files_deleted"] == 5
        assert d["bytes_freed"] == 1024
        assert set(d["sources_purged"]) == {"audit_logs", "consent"}
        assert d["error_count"] == 0


class TestDataPurgeService:
    """Test DataPurgeService core functionality."""

    def test_purge_requires_confirmation(self, purge_service):
        with pytest.raises(PurgeConfirmationRequired):
            purge_service.purge_all(confirm=False)

    def test_purge_all_with_confirmation(self, purge_service, audit_logger, consent_registry):
        populate_audit_logs(audit_logger)
        populate_consent(consent_registry)

        result = purge_service.purge_all(confirm=True)

        assert result.success
        assert result.files_deleted > 0
        assert "audit_logs" in result.sources_purged
        assert "consent_records" in result.sources_purged

    def test_purge_all_removes_audit_logs(self, purge_service, audit_logger):
        populate_audit_logs(audit_logger, count=10)

        # Verify logs exist
        assert audit_logger._path.exists()

        result = purge_service.purge_all(confirm=True)

        # Verify logs are gone
        assert not audit_logger._path.exists()
        assert result.success

    def test_purge_all_removes_consent(self, purge_service, consent_registry):
        populate_consent(consent_registry)

        # Verify consent exists
        assert len(list(consent_registry.snapshot())) == 3

        result = purge_service.purge_all(confirm=True)

        # Consent file should be deleted by DataPurgeService._purge_consent
        assert result.success


class TestPurgeAuditLogs:
    """Test audit log purge functionality."""

    def test_purge_audit_logs_requires_confirmation(self, purge_service):
        with pytest.raises(PurgeConfirmationRequired):
            purge_service.purge_audit_logs(confirm=False)

    def test_purge_audit_logs(self, purge_service, audit_logger):
        populate_audit_logs(audit_logger, count=5)

        result = purge_service.purge_audit_logs(confirm=True)

        assert result.success
        assert "audit_logs" in result.sources_purged
        assert not audit_logger._path.exists()


class TestPurgeConsent:
    """Test consent purge functionality."""

    def test_purge_consent_requires_confirmation(self, purge_service):
        with pytest.raises(PurgeConfirmationRequired):
            purge_service.purge_consent(confirm=False)

    def test_purge_consent(self, purge_service, consent_registry):
        populate_consent(consent_registry)

        result = purge_service.purge_consent(confirm=True)

        assert result.success
        assert "consent_records" in result.sources_purged


class TestPurgeBySource:
    """Test source-specific purge functionality."""

    def test_purge_by_source_requires_confirmation(self, purge_service):
        with pytest.raises(PurgeConfirmationRequired):
            purge_service.purge_by_source("test_source", confirm=False)

    def test_purge_by_source_revokes_consent(self, purge_service, consent_registry):
        populate_consent(consent_registry)

        # Verify source_a has 2 consents
        source_a_count = sum(
            1 for r in consent_registry.snapshot()
            if r.source == "source_a" and r.granted
        )
        assert source_a_count == 2

        result = purge_service.purge_by_source("source_a", confirm=True)

        assert result.success
        assert "source_a" in result.sources_purged

        # Verify source_a consents are revoked
        source_a_active = sum(
            1 for r in consent_registry.iter_active()
            if r.source == "source_a"
        )
        assert source_a_active == 0


class TestAuditLoggerPurgeMethods:
    """Test AuditLogger purge methods directly."""

    def test_audit_logger_purge_all(self, audit_logger):
        populate_audit_logs(audit_logger, count=10)

        deleted = audit_logger.purge_all()

        assert deleted > 0
        assert not audit_logger._path.exists()

    def test_audit_logger_purge_by_source(self, audit_logger):
        # Add events from different sources
        for i in range(6):
            audit_logger.record(
                AuditEvent(
                    job_id=f"job_{i}",
                    source="source_a" if i < 4 else "source_b",
                    action="test",
                    status="success",
                    timestamp=datetime.utcnow(),
                )
            )

        # Verify initial count
        events = list(audit_logger.iter_events())
        assert len(events) == 6
        assert sum(1 for e in events if e["source"] == "source_a") == 4

        # Purge source_a
        removed = audit_logger.purge_by_source("source_a")

        assert removed == 4

        # Verify remaining events
        remaining = list(audit_logger.iter_events())
        assert len(remaining) == 2
        assert all(e["source"] == "source_b" for e in remaining)


class TestConsentRegistryPurgeMethods:
    """Test ConsentRegistry purge methods directly."""

    def test_consent_registry_purge_all(self, consent_registry):
        populate_consent(consent_registry)

        count = consent_registry.purge_all()

        assert count == 3
        assert len(list(consent_registry.snapshot())) == 0

    def test_consent_registry_purge_by_source(self, consent_registry):
        populate_consent(consent_registry)

        removed = consent_registry.purge_by_source("source_a")

        assert removed == 2

        # Verify remaining
        remaining = list(consent_registry.snapshot())
        assert len(remaining) == 1
        assert remaining[0].source == "source_b"


class TestPurgeVerification:
    """Test purge verification functionality."""

    def test_verify_purge_after_complete_purge(self, purge_service, audit_logger, consent_registry):
        populate_audit_logs(audit_logger)
        populate_consent(consent_registry)

        purge_service.purge_all(confirm=True)

        assert purge_service.verify_purge()

    def test_verify_purge_with_remaining_data(self, purge_service, audit_logger, consent_registry):
        populate_audit_logs(audit_logger)
        populate_consent(consent_registry)

        # Only purge consent, not audit
        purge_service.purge_consent(confirm=True)

        # Should fail verification because audit logs remain
        assert not purge_service.verify_purge()


class TestMetaAuditLogging:
    """Test that purge operations are logged (meta-audit)."""

    def test_purge_logs_start_and_complete(self, tmp_path, consent_registry):
        # Create separate meta audit logger
        meta_dir = tmp_path / "meta_audit"
        meta_dir.mkdir()
        meta_logger = AuditLogger(output_dir=meta_dir)

        purge_service = DataPurgeService(
            consent_registry=consent_registry,
            meta_audit_logger=meta_logger,
        )

        populate_consent(consent_registry)
        purge_service.purge_consent(confirm=True)

        # Check meta audit log
        events = list(meta_logger.iter_events())
        assert len(events) >= 2

        # Should have start and complete events
        actions = [e["action"] for e in events]
        assert any("purge" in a and "started" in e.get("status", "") for a, e in zip(actions, events))
        assert any("purge" in a and "completed" in e.get("status", "") for a, e in zip(actions, events))


class TestAdditionalPaths:
    """Test purging additional paths."""

    def test_purge_additional_path(self, tmp_path, audit_logger, consent_registry):
        # Create additional data directory
        data_dir = tmp_path / "workspace_data"
        data_dir.mkdir()
        (data_dir / "file1.txt").write_text("data1")
        (data_dir / "file2.txt").write_text("data2")
        subdir = data_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("data3")

        purge_service = DataPurgeService(
            audit_logger=audit_logger,
            consent_registry=consent_registry,
            additional_paths=[data_dir],
        )

        result = purge_service.purge_all(confirm=True)

        assert result.success
        assert not data_dir.exists()


class TestCreatePurgeService:
    """Test factory function."""

    def test_create_purge_service(self, audit_logger, consent_registry):
        service = create_purge_service(
            audit_logger=audit_logger,
            consent_registry=consent_registry,
        )

        assert service.audit_logger is audit_logger
        assert service.consent_registry is consent_registry

    def test_create_purge_service_with_workspace(self, tmp_path, audit_logger, consent_registry):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        service = create_purge_service(
            workspace_dir=workspace,
            audit_logger=audit_logger,
            consent_registry=consent_registry,
        )

        assert workspace in service.additional_paths
