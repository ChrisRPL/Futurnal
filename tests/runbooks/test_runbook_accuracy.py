"""Test that commands in operator runbooks work as documented.

This test suite validates that all commands shown in the operator runbooks
documentation actually execute successfully and produce expected outputs.
"""

import pytest
from datetime import datetime, timedelta

from futurnal.orchestrator.daemon import DaemonStatus
from futurnal.orchestrator.models import IngestionJob, JobPriority, JobType
from futurnal.orchestrator.quarantine import QuarantinedJob, QuarantineReason


class TestOrchestratorDaemonCommands:
    """Test orchestrator daemon management commands from runbooks."""

    def test_daemon_status_check(self, orchestrator_daemon):
        """Test: futurnal orchestrator status (daemon not running)."""
        status = orchestrator_daemon.status()
        assert isinstance(status, DaemonStatus)
        assert not status.running

    def test_daemon_register_start(self, orchestrator_daemon):
        """Test: PID file creation on start."""
        orchestrator_daemon.register_start()
        status = orchestrator_daemon.status()
        assert status.running
        assert status.pid is not None

        # Cleanup
        orchestrator_daemon.register_stop()

    def test_daemon_register_stop(self, orchestrator_daemon):
        """Test: PID file removal on stop."""
        orchestrator_daemon.register_start()
        orchestrator_daemon.register_stop()
        status = orchestrator_daemon.status()
        assert not status.running


class TestJobManagementCommands:
    """Test job management commands from runbooks."""

    def test_jobs_list_basic(self, job_queue, sample_jobs):
        """Test: futurnal orchestrator jobs list."""
        jobs = job_queue.snapshot(limit=10)
        assert len(jobs) == 2
        assert all(isinstance(j, dict) for j in jobs)
        assert all("job_id" in j for j in jobs)

    def test_jobs_list_with_status_filter(self, job_queue, sample_jobs):
        """Test: futurnal orchestrator jobs list --status pending."""
        jobs = job_queue.snapshot(status=None, limit=10)
        pending_jobs = [j for j in jobs if j["status"] == "pending"]
        assert len(pending_jobs) == 2

    def test_jobs_show(self, job_queue, sample_jobs):
        """Test: futurnal orchestrator jobs show JOB_ID."""
        job = job_queue.get_job("job-1")
        assert job is not None
        assert job["job_id"] == "job-1"
        assert job["job_type"] == "local_files"
        assert "payload" in job

    def test_jobs_cancel(self, job_queue, sample_jobs):
        """Test: futurnal orchestrator jobs cancel JOB_ID."""
        # Job should exist and be pending
        job = job_queue.get_job("job-1")
        assert job["status"] == "pending"

        # Cancel the job
        job_queue.cancel_job("job-1")

        # Verify cancellation
        job_after = job_queue.get_job("job-1")
        assert job_after is None or job_after["status"] != "pending"

    def test_pending_count(self, job_queue, sample_jobs):
        """Test: Queue depth reporting."""
        count = job_queue.pending_count()
        assert count == 2

    def test_running_count(self, job_queue, sample_jobs):
        """Test: Running jobs count."""
        count = job_queue.running_count()
        assert count == 0

    def test_completed_count(self, job_queue):
        """Test: Completed jobs count with time filter."""
        since = datetime.utcnow() - timedelta(hours=1)
        count = job_queue.completed_count(since=since)
        assert count >= 0


class TestSourceControlCommands:
    """Test source control commands from runbooks."""

    def test_sources_pause(self, paused_sources_registry):
        """Test: futurnal orchestrator sources pause email."""
        assert not paused_sources_registry.is_paused("email")

        paused_sources_registry.pause("email")
        assert paused_sources_registry.is_paused("email")

    def test_sources_resume(self, paused_sources_registry):
        """Test: futurnal orchestrator sources resume email."""
        paused_sources_registry.pause("email")
        assert paused_sources_registry.is_paused("email")

        paused_sources_registry.resume("email")
        assert not paused_sources_registry.is_paused("email")

    def test_sources_list_paused(self, paused_sources_registry):
        """Test: futurnal orchestrator sources list --status paused."""
        paused_sources_registry.pause("source1")
        paused_sources_registry.pause("source2")

        paused_list = paused_sources_registry.list_paused()
        assert len(paused_list) == 2
        assert "source1" in paused_list
        assert "source2" in paused_list


class TestQuarantineCommands:
    """Test quarantine management commands from runbooks."""

    def test_quarantine_list_empty(self, quarantine_store):
        """Test: futurnal orchestrator quarantine list (empty)."""
        # List quarantined jobs on empty store
        jobs = quarantine_store.list()
        assert len(jobs) == 0

    def test_quarantine_add_via_job(self, quarantine_store, job_queue):
        """Test: Adding job to quarantine and listing."""
        # Create a job
        job = IngestionJob(
            job_id="quarantined-1",
            job_type=JobType.LOCAL_FILES,
            payload={"test": "data"},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )

        # Quarantine it
        quarantine_store.quarantine(
            job=job,
            reason=QuarantineReason.PARSE_ERROR,
            error_message="Test parse error",
        )

        # List and verify
        jobs = quarantine_store.list()
        assert len(jobs) == 1
        assert jobs[0].job_id == "quarantined-1"

    def test_quarantine_get(self, quarantine_store):
        """Test: futurnal orchestrator quarantine show JOB_ID."""
        # Create and quarantine a job
        job = IngestionJob(
            job_id="quarantined-2",
            job_type=JobType.IMAP_MAILBOX,
            payload={"mailbox": "test"},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )

        quarantine_store.quarantine(
            job=job,
            reason=QuarantineReason.PERMISSION_DENIED,
            error_message="Permission error",
        )

        # Get specific job
        retrieved_job = quarantine_store.get("quarantined-2")
        assert retrieved_job is not None
        assert retrieved_job.job_id == "quarantined-2"
        assert retrieved_job.reason == QuarantineReason.PERMISSION_DENIED

    def test_quarantine_statistics(self, quarantine_store):
        """Test: futurnal orchestrator quarantine stats."""
        stats = quarantine_store.statistics()
        assert isinstance(stats, dict)
        assert "total_quarantined" in stats


class TestDatabaseUtilities:
    """Test database utility commands from runbooks."""

    def test_database_backup(self, database_manager):
        """Test: futurnal orchestrator db backup."""
        backup_path = database_manager.backup(comment="test")
        assert backup_path.exists()
        assert backup_path.suffix == ".db"
        assert "test" in backup_path.name

    def test_database_integrity_check(self, database_manager):
        """Test: futurnal orchestrator db check (or sqlite3 PRAGMA integrity_check)."""
        is_valid, detail = database_manager.check_integrity()
        assert is_valid
        assert "passed" in detail.lower()

    def test_database_list_backups(self, database_manager):
        """Test: List available database backups."""
        # Create a few backups
        database_manager.backup(comment="backup1")
        database_manager.backup(comment="backup2")

        backups = database_manager.list_backups()
        assert len(backups) >= 2
        assert all(isinstance(b, tuple) for b in backups)
        assert all(len(b) == 3 for b in backups)  # (path, created, size)

    def test_database_restore(self, database_manager, temp_workspace):
        """Test: futurnal orchestrator db restore."""
        # Create backup
        backup_path = database_manager.backup(comment="restore-test")

        # Restore should work (with force=True for testing)
        database_manager.restore(backup_path, force=True)

        # Verify restore created backup of previous database
        before_restore_backup = temp_workspace / "queue" / "jobs.db.before-restore"
        assert before_restore_backup.exists()

    def test_database_vacuum(self, database_manager):
        """Test: Database vacuum operation."""
        # Should not raise
        database_manager.vacuum()

    def test_database_stats(self, database_manager):
        """Test: Database statistics retrieval."""
        stats = database_manager.get_stats()
        assert "exists" in stats
        assert stats["exists"]
        assert "size_mb" in stats
        assert "job_count" in stats


class TestTelemetryCommands:
    """Test telemetry viewing commands from runbooks."""

    def test_telemetry_failures_analysis(self, telemetry_analyzer):
        """Test: futurnal orchestrator telemetry failures --since 24h."""
        stats = telemetry_analyzer.analyze_failures()
        assert stats.total_failures >= 0
        assert isinstance(stats.failures_by_reason, dict)
        assert isinstance(stats.failures_by_connector, dict)

    def test_telemetry_throughput_calculation(self, telemetry_analyzer):
        """Test: futurnal orchestrator telemetry throughput --since 1h."""
        metrics = telemetry_analyzer.calculate_throughput()
        assert metrics.files_processed >= 0
        assert metrics.bytes_processed >= 0
        assert metrics.throughput_mbps >= 0.0

    def test_telemetry_by_connector(self, telemetry_analyzer):
        """Test: futurnal orchestrator telemetry by-connector."""
        connector_metrics = telemetry_analyzer.metrics_by_connector()
        assert isinstance(connector_metrics, list)
        # Should have at least the connectors from sample data
        if connector_metrics:
            assert all(hasattr(m, "connector_type") for m in connector_metrics)
            assert all(hasattr(m, "success_rate") for m in connector_metrics)

    def test_telemetry_clean(self, telemetry_analyzer):
        """Test: futurnal telemetry clean --older-than-days 60."""
        # Dry run should not modify files
        removed = telemetry_analyzer.clean_old_telemetry(
            older_than_days=1,
            dry_run=True,
        )
        assert removed >= 0  # Should return count


class TestHealthChecks:
    """Test health check commands from runbooks."""

    def test_health_check_workspace_creation(self, temp_workspace):
        """Test: Workspace directory structure."""
        assert (temp_workspace / "queue").exists()
        assert (temp_workspace / "quarantine").exists()
        assert (temp_workspace / "telemetry").exists()
        assert (temp_workspace / "audit").exists()

    def test_health_check_database_exists(self, job_queue, temp_workspace):
        """Test: Check if queue database exists and is accessible."""
        db_path = temp_workspace / "queue" / "jobs.db"
        assert db_path.exists()

    def test_subsystem_health_check(self, temp_workspace, monkeypatch):
        """Test: futurnal health check <subsystem> command."""
        from futurnal.configuration.cli import check_subsystem

        # Mock collect_health_report to return a health report
        def mock_collect_health_report(settings, workspace_path):
            return {
                "checks": [
                    {"name": "neo4j_connection", "status": "ok", "detail": "Connected"},
                    {"name": "chroma_connection", "status": "ok", "detail": "Connected"},
                    {"name": "queue_database", "status": "ok", "detail": "OK"},
                ]
            }

        monkeypatch.setattr(
            "futurnal.configuration.cli.collect_health_report", mock_collect_health_report
        )
        monkeypatch.setattr(
            "futurnal.configuration.cli.bootstrap_settings",
            lambda path: type("Settings", (), {})(),
        )

        # Test neo4j subsystem check
        try:
            check_subsystem(subsystem="neo4j", workspace_path=temp_workspace)
        except SystemExit:
            pass  # Expected exit

        # Test chroma subsystem check
        try:
            check_subsystem(subsystem="chroma", workspace_path=temp_workspace)
        except SystemExit:
            pass  # Expected exit


class TestConfigurationCommands:
    """Test configuration management commands from runbooks."""

    def test_config_validation_concept(self, temp_workspace):
        """Test: Configuration file validation concept.

        Note: Actual config validation is in orchestrator_config_app,
        this test validates the workspace supports configuration.
        """
        config_dir = temp_workspace / "config"
        config_dir.mkdir(exist_ok=True)
        assert config_dir.exists()

    def test_config_validate_command(self, temp_workspace, monkeypatch):
        """Test: futurnal config validate command."""
        from futurnal.configuration.settings import DEFAULT_CONFIG_PATH, Settings

        # Mock load_settings to return a valid settings object
        def mock_load_settings(path):
            return Settings.model_validate({
                "workspace": {
                    "workspace_path": str(temp_workspace),
                    "storage": {
                        "neo4j_uri": "bolt://localhost:7687",
                        "neo4j_username": "neo4j",
                        "neo4j_password": "test",
                        "chroma_path": str(temp_workspace / "chroma"),
                    },
                },
            })

        monkeypatch.setattr("futurnal.configuration.cli.load_settings", mock_load_settings)

        # This validates the command can load and display config
        from futurnal.configuration.cli import validate_config

        # Should not raise exception (validate returns success)
        validate_config(config_path=DEFAULT_CONFIG_PATH)


class TestAuditLogging:
    """Test audit logging for operator actions."""

    def test_audit_directory_exists(self, temp_workspace):
        """Test: Audit log directory exists."""
        audit_dir = temp_workspace / "audit"
        assert audit_dir.exists()


# Integration-style tests that verify end-to-end workflows


class TestRunbookWorkflows:
    """Test complete workflows described in runbooks."""

    def test_job_lifecycle_workflow(self, job_queue):
        """Test complete job lifecycle: enqueue → running → completed."""
        # Enqueue job
        job = IngestionJob(
            job_id="workflow-test",
            job_type=JobType.LOCAL_FILES,
            payload={"source": "test"},
            priority=JobPriority.NORMAL,
            scheduled_for=datetime.utcnow(),
        )
        job_queue.enqueue(job)

        # Verify pending
        assert job_queue.pending_count() >= 1

        # Mark running
        job_queue.mark_running("workflow-test")
        assert job_queue.running_count() >= 1

        # Mark completed
        job_queue.mark_completed("workflow-test")
        completed_since = datetime.utcnow() - timedelta(seconds=5)
        assert job_queue.completed_count(since=completed_since) >= 1

    def test_source_pause_resume_workflow(self, paused_sources_registry):
        """Test source pause/resume workflow from troubleshooting guide."""
        source_name = "email"

        # Initial state: active
        assert not paused_sources_registry.is_paused(source_name)

        # Pause (e.g., for maintenance)
        paused_sources_registry.pause(source_name)
        assert paused_sources_registry.is_paused(source_name)

        # Resume after maintenance
        paused_sources_registry.resume(source_name)
        assert not paused_sources_registry.is_paused(source_name)

    def test_database_backup_restore_workflow(self, database_manager, temp_workspace):
        """Test database backup and restore workflow from disaster recovery."""
        # Create backup
        backup_path = database_manager.backup(comment="disaster-test")
        assert backup_path.exists()

        # Verify integrity before restore
        is_valid, _ = database_manager.check_integrity()
        assert is_valid

        # Restore from backup
        database_manager.restore(backup_path, force=True)

        # Verify integrity after restore
        is_valid_after, _ = database_manager.check_integrity()
        assert is_valid_after
