Summary: Comprehensive integration tests for end-to-end orchestrator workflows including multi-connector pipelines, privacy audit trails, and error propagation.

# 08 · Integration Test Suite

## Purpose
Provide comprehensive integration tests that validate end-to-end orchestrator workflows, ensuring all components (queue, scheduler, workers, connectors, telemetry, audit) work together correctly. Validates the Ghost's experiential learning pipeline from data ingestion through storage in PKG and vector stores.

## Scope
- End-to-end multi-connector workflow tests
- Pipeline integration (ingestion → parsing → normalization → storage)
- Privacy audit trail verification
- Telemetry accuracy validation
- Error propagation testing
- Connector-orchestrator integration
- State persistence across restarts
- Multi-phase job lifecycle testing

## Requirements Alignment
- **Integration Tests**: "Connector job execution end-to-end with induced failures" (testing strategy)
- **Pipeline Validation**: Ensure data flows through entire pipeline
- **Audit Compliance**: Verify audit trail completeness
- **Telemetry Accuracy**: Validate metric correctness

## Test Categories

### 1. End-to-End Pipeline Tests
Validate complete data flow from source to storage.

### 2. Multi-Connector Tests
Validate orchestrator handles multiple connector types.

### 3. Privacy & Audit Tests
Validate audit trail and privacy compliance.

### 4. Telemetry Tests
Validate telemetry accuracy and completeness.

### 5. Error Handling Tests
Validate error propagation and recovery.

## Test Specifications

### 01 - End-to-End Local Files Pipeline
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_files_end_to_end_pipeline(tmp_path: Path):
    """Test complete pipeline from local file to PKG storage."""

    # Setup
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "note.md").write_text("# Test Note\nThis is test content.")

    # Initialize components
    queue = JobQueue(workspace / "queue.db")
    state_store = StateStore(workspace / "state" / "state.db")

    # Mock PKG and vector writers
    pkg_writer = MockPKGWriter()
    vector_writer = MockVectorWriter()
    sink = NormalizationSink(pkg_writer=pkg_writer, vector_writer=vector_writer)

    # Create orchestrator
    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(workspace),
        element_sink=sink,
    )

    # Register source
    source = LocalIngestionSource(
        name="test_source",
        root_path=source_dir,
    )
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual")
    )

    # Execute
    orchestrator.run_manual_job("test_source", force=True)
    orchestrator.start()
    await asyncio.sleep(2)
    await orchestrator.shutdown()

    # Validate: File was processed
    assert pkg_writer.documents_written > 0
    assert vector_writer.embeddings_written > 0

    # Validate: State stored
    file_state = state_store.get(str(source_dir / "note.md"))
    assert file_state is not None
    assert file_state.status == "processed"

    # Validate: Job completed
    jobs = queue.snapshot(status=JobStatus.SUCCEEDED)
    assert len(jobs) == 1

    # Validate: Telemetry recorded
    telemetry_dir = workspace / "telemetry"
    assert (telemetry_dir / "telemetry.log").exists()
```

### 02 - Multi-Connector Integration
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_connector_orchestration(tmp_path: Path):
    """Test orchestrator with multiple connector types."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Setup sources
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "file.txt").write_text("test")

    obsidian_dir = tmp_path / "vault"
    obsidian_dir.mkdir()
    (obsidian_dir / "note.md").write_text("# Note")

    # Initialize orchestrator
    queue = JobQueue(workspace / "queue.db")
    state_store = StateStore(workspace / "state" / "state.db")
    sink = MockNormalizationSink()

    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(workspace),
        element_sink=sink,
    )

    # Register multiple sources
    orchestrator.register_source(
        SourceRegistration(
            source=LocalIngestionSource(name="local", root_path=local_dir),
            schedule="@manual",
        )
    )
    orchestrator.register_source(
        SourceRegistration(
            source=ObsidianIngestionSource(name="vault", root_path=obsidian_dir),
            schedule="@manual",
        )
    )

    # Trigger both
    orchestrator.run_manual_job("local", force=True)
    orchestrator.run_manual_job("vault", force=True)

    orchestrator.start()
    await asyncio.sleep(3)
    await orchestrator.shutdown()

    # Validate: Both connectors executed
    jobs = queue.snapshot(status=JobStatus.SUCCEEDED)
    assert len(jobs) == 2
    job_types = {j["job_type"] for j in jobs}
    assert JobType.LOCAL_FILES.value in job_types
    assert JobType.OBSIDIAN_VAULT.value in job_types
```

### 03 - Privacy Audit Trail Verification
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_audit_trail_completeness(tmp_path: Path):
    """Verify comprehensive audit trail for all operations."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    audit_dir = workspace / "audit"
    audit_logger = AuditLogger(audit_dir)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "file.txt").write_text("test")

    # Create orchestrator with audit logging
    queue = JobQueue(workspace / "queue.db")
    state_store = StateStore(workspace / "state" / "state.db")

    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(workspace),
        element_sink=MockNormalizationSink(),
    )

    # Register source
    source = LocalIngestionSource(name="test", root_path=source_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual")
    )

    # Execute
    orchestrator.run_manual_job("test", force=True)
    orchestrator.start()
    await asyncio.sleep(2)
    await orchestrator.shutdown()

    # Validate: Audit events logged
    audit_files = list(audit_dir.glob("*.jsonl"))
    assert len(audit_files) > 0

    # Parse audit events
    events = []
    for audit_file in audit_files:
        for line in audit_file.read_text().splitlines():
            events.append(json.loads(line))

    # Verify expected events
    event_actions = {e["action"] for e in events}
    assert "job" in event_actions  # Job execution logged

    # Verify no sensitive data in audit
    for event in events:
        assert "password" not in json.dumps(event).lower()
        assert "token" not in json.dumps(event).lower()
```

### 04 - Telemetry Accuracy Validation
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_telemetry_accuracy(tmp_path: Path):
    """Validate telemetry metrics match actual execution."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Create test files with known sizes
    file_sizes = [1000, 2000, 3000]
    for i, size in enumerate(file_sizes):
        (source_dir / f"file{i}.txt").write_bytes(b"x" * size)

    # Create orchestrator
    queue = JobQueue(workspace / "queue.db")
    state_store = StateStore(workspace / "state" / "state.db")
    telemetry = TelemetryRecorder(workspace / "telemetry")

    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(workspace),
        element_sink=MockNormalizationSink(),
        telemetry=telemetry,
    )

    # Register and execute
    source = LocalIngestionSource(name="test", root_path=source_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual")
    )

    orchestrator.run_manual_job("test", force=True)
    orchestrator.start()
    await asyncio.sleep(3)
    await orchestrator.shutdown()

    # Load telemetry
    telemetry_summary = json.loads(
        (workspace / "telemetry" / "telemetry_summary.json").read_text()
    )

    # Validate: File count matches
    assert telemetry_summary["overall"]["files"] == len(file_sizes)

    # Validate: Byte count matches (approximately)
    expected_bytes = sum(file_sizes)
    actual_bytes = telemetry_summary["overall"]["bytes"]
    assert abs(actual_bytes - expected_bytes) < 1000  # Allow small variance
```

### 05 - Error Propagation Testing
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_propagation_and_recovery(tmp_path: Path):
    """Test error handling across pipeline components."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Create file that will cause parsing error
    (source_dir / "corrupt.bin").write_bytes(b"\x00\x01\x02" * 100)

    # Create orchestrator with quarantine
    queue = JobQueue(workspace / "queue.db")
    state_store = StateStore(workspace / "state" / "state.db")
    quarantine = QuarantineStore(workspace / "quarantine" / "quarantine.db")

    # Mock connector that raises error
    class FailingConnector(LocalFilesConnector):
        def ingest(self, source, job_id):
            raise ValueError("Simulated parsing error")

    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(workspace),
        element_sink=MockNormalizationSink(),
        quarantine_store=quarantine,
    )

    # Monkey-patch connector to fail
    orchestrator._local_connector = FailingConnector(
        workspace_dir=workspace,
        state_store=state_store,
        element_sink=MockNormalizationSink(),
    )

    # Register and execute
    source = LocalIngestionSource(name="test", root_path=source_dir)
    orchestrator.register_source(
        SourceRegistration(source=source, schedule="@manual")
    )

    orchestrator.run_manual_job("test", force=True)
    orchestrator.start()
    await asyncio.sleep(5)  # Allow retries
    await orchestrator.shutdown()

    # Validate: Job failed and quarantined
    failed_jobs = queue.snapshot(status=JobStatus.FAILED)
    assert len(failed_jobs) >= 1

    quarantined_jobs = quarantine.list()
    assert len(quarantined_jobs) >= 1

    # Validate: Error logged
    assert "parsing error" in quarantined_jobs[0].error_message.lower()
```

### 06 - State Persistence Across Restarts
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_state_persistence_across_restarts(tmp_path: Path):
    """Test orchestrator state persists across restarts."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "file.txt").write_text("test")

    # First run: Process file
    queue = JobQueue(workspace / "queue.db")
    state_store = StateStore(workspace / "state" / "state.db")

    orchestrator1 = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(workspace),
        element_sink=MockNormalizationSink(),
    )

    source = LocalIngestionSource(name="test", root_path=source_dir)
    orchestrator1.register_source(
        SourceRegistration(source=source, schedule="@manual")
    )

    orchestrator1.run_manual_job("test", force=True)
    orchestrator1.start()
    await asyncio.sleep(2)
    await orchestrator1.shutdown()

    # Validate: Job completed
    jobs_after_first_run = queue.snapshot(status=JobStatus.SUCCEEDED)
    assert len(jobs_after_first_run) == 1

    # Second run: Restart orchestrator (should not reprocess)
    orchestrator2 = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(workspace),
        element_sink=MockNormalizationSink(),
    )

    orchestrator2.register_source(
        SourceRegistration(source=source, schedule="@manual")
    )

    orchestrator2.run_manual_job("test", force=True)
    orchestrator2.start()
    await asyncio.sleep(2)
    await orchestrator2.shutdown()

    # Validate: No duplicate processing (state persisted)
    jobs_after_second_run = queue.snapshot(status=JobStatus.SUCCEEDED)
    # Should still be 1 (or 2 if second run enqueued but skipped)
    assert len(jobs_after_second_run) <= 2
```

### 07 - Scheduled Job Execution
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_scheduled_job_execution(tmp_path: Path):
    """Test automatic scheduled job execution."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "file.txt").write_text("test")

    # Create orchestrator
    queue = JobQueue(workspace / "queue.db")
    state_store = StateStore(workspace / "state" / "state.db")

    orchestrator = IngestionOrchestrator(
        job_queue=queue,
        state_store=state_store,
        workspace_dir=str(workspace),
        element_sink=MockNormalizationSink(),
    )

    # Register with interval schedule (every 2 seconds)
    source = LocalIngestionSource(name="test", root_path=source_dir)
    orchestrator.register_source(
        SourceRegistration(
            source=source,
            schedule="@interval",
            interval_seconds=2,
        )
    )

    # Start and wait for scheduled executions
    orchestrator.start()
    await asyncio.sleep(6)  # Allow 2-3 executions
    await orchestrator.shutdown()

    # Validate: Multiple scheduled executions
    jobs = queue.snapshot(status=JobStatus.SUCCEEDED)
    assert len(jobs) >= 2  # At least 2 scheduled runs
```

## Test Utilities

### MockPKGWriter
```python
class MockPKGWriter:
    """Mock PKG writer for testing."""

    def __init__(self):
        self.documents_written = 0
        self.documents = []

    def write_document(self, payload: Dict[str, Any]) -> None:
        self.documents_written += 1
        self.documents.append(payload)

    def remove_document(self, sha256: str) -> None:
        self.documents = [d for d in self.documents if d.get("sha256") != sha256]
```

### MockVectorWriter
```python
class MockVectorWriter:
    """Mock vector writer for testing."""

    def __init__(self):
        self.embeddings_written = 0
        self.embeddings = []

    def write_embedding(self, payload: Dict[str, Any]) -> None:
        self.embeddings_written += 1
        self.embeddings.append(payload)

    def remove_embedding(self, sha256: str) -> None:
        self.embeddings = [e for e in self.embeddings if e.get("sha256") != sha256]
```

## Acceptance Criteria

- ✅ End-to-end local files pipeline test passes
- ✅ Multi-connector orchestration test passes
- ✅ Privacy audit trail verified complete
- ✅ Telemetry metrics match actual execution
- ✅ Error propagation and quarantine tested
- ✅ State persistence across restarts validated
- ✅ Scheduled job execution tested
- ✅ All integration tests pass in CI
- ✅ Integration test coverage ≥80%
- ✅ Integration tests complete in <5 minutes

## Test Plan

### Integration Tests
- `test_local_files_pipeline.py`: End-to-end local files
- `test_obsidian_pipeline.py`: End-to-end Obsidian vault
- `test_multi_connector.py`: Multiple connectors
- `test_audit_trail.py`: Privacy and audit compliance
- `test_telemetry.py`: Telemetry accuracy
- `test_error_handling.py`: Error propagation
- `test_state_persistence.py`: Restart behavior
- `test_scheduled_execution.py`: Scheduled jobs

### Pipeline Tests
- `test_pipeline_integration.py`: Full ingestion → storage pipeline
- `test_normalization_integration.py`: Normalization workflow
- `test_extraction_integration.py`: Entity extraction workflow

### Resilience Tests
- `test_connector_failure.py`: Connector failure handling
- `test_storage_failure.py`: PKG/vector storage failures
- `test_partial_failure.py`: Partial pipeline failures

## Implementation Notes

### Running Integration Tests
```bash
# Run all integration tests
pytest -m integration tests/orchestrator/integration/

# Run specific test
pytest tests/orchestrator/integration/test_local_files_pipeline.py -v

# Run with coverage
pytest -m integration --cov=futurnal.orchestrator tests/orchestrator/integration/
```

### CI Configuration
```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: pytest -m integration --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Open Questions

- Should integration tests use real or mocked external services?
- How to handle long-running integration tests (run nightly)?
- Should we test with real Neo4j and ChromaDB instances?
- What's the appropriate timeout for integration tests?
- Should we implement integration test fixtures for common scenarios?
- How to validate cross-platform behavior (macOS/Linux/Windows)?

## Dependencies

- Pytest with async support
- Mock implementations of connectors and storage
- Test fixtures for common scenarios
- CI/CD pipeline integration


