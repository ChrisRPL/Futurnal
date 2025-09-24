Summary: Plan for integration, performance, and security testing of the Local Files Connector.

# Task · Integration & Performance Test Suite

## Objectives
- Validate end-to-end ingestion across diverse directory structures, file types, and OS-level edge cases.
- Confirm throughput meets or exceeds the ≥5 MB/s target on reference Apple Silicon hardware.
- Ensure permission errors, deletions, and symlink traversal behave predictably without data loss.

## Deliverables
- Automated integration test harness with fixture directories (nested, binary, large files, symlinks, hidden paths).
- Performance benchmark script capturing throughput, CPU, and memory stats; results ingested into telemetry reports.
- Security regression tests verifying graceful handling of permission-denied files and sandbox boundaries.
- Documentation for running the suite locally and in CI, including dataset setup and expected runtime.

## Work Breakdown
1. **Fixture Design**
   - Add fixture generation scripts under `tests/fixtures/local_connector/` that synthesize required directory layouts at test runtime to avoid repo bloat. Scenarios: deep nesting, mixed binaries, sparse large files (>100 MB via sparse allocation), hidden files, valid/broken symlinks, permission-locked items, and concurrent modification simulators.
   - Provide teardown utilities to reset `pkg/`, `vec/`, `telemetry/`, and state stores after each scenario so hybrid indices remain synchronized for subsequent tests.
2. **Integration Harness**
   - Extend pytest suite in `tests/ingestion/local/` with parametrized end-to-end tests that run ingestion against each generated fixture, verifying Unstructured.io parsing, PKG/vector writes via instrumentation fakes, and state-store deletion handling.
   - Capture telemetry and audit artifacts into temp directories; assert audit entries redact paths per privacy rules and that quarantine remains empty for happy paths.
3. **Performance Benchmarks**
   - Introduce pytest marker `performance` plus optional CLI wrapper to time ingestion/bytes processed, leveraging `TelemetryRecorder` to serialize metrics into `telemetry/perf.json` with hardware metadata.
   - Fail benchmarks if sustained throughput drops below ≥5 MB/s on reference Apple Silicon baseline; store rolling baselines for roadmap KPI tracking.
4. **Security & Resilience Tests**
   - Add test cases that force permission-denied errors, locked-file access, and mid-run interruption/restart, confirming quarantine captures failures, audit trails remain intact, and PKG/vector stores stay consistent.
   - Verify privacy defaults (no external escalation) and ensure deletion propagation removes both PKG nodes and vector embeddings across restart boundaries.
5. **CI Integration & Runbook**
   - Add GitHub Actions workflow (`.github/workflows/local-files-tests.yml`) that runs smoke tests on PRs and full integration + performance suite nightly on macOS runners; upload telemetry artifacts for trend analysis.
   - Document local execution commands (`pytest tests/ingestion/local/test_integration_connector.py`, `pytest -m performance`) and expected runtimes (smoke <15s, performance <30s on M2 Pro) including fixture generation notes.
   - Describe telemetry output rotation: benchmark summaries written to `workspace/telemetry/perf.json` per run, with nightly artifacts stored for baseline comparison. Provide guidance on updating baselines when hardware or thresholds change.

## Execution Notes

- **Local prerequisites:** Python 3.11, dependencies from `requirements.txt`, and available disk space for temporary sparse files (sparse creation keeps on-disk usage minimal). Large-file fixtures rely on sparse allocation; on filesystems without sparse support, adjust size via `create_sparse_large_file_fixture(size_bytes=...)`.
- **Running tests locally:**
  - Smoke: `pytest tests/ingestion/local/test_integration_connector.py`
  - Full suite: `pytest tests/ingestion/local/test_integration_connector.py tests/ingestion/local/test_performance_metrics.py`
  - Performance-only: `pytest -m performance tests/ingestion/local/test_performance_metrics.py`
- **Telemetry outputs:** Each run writes parsed elements to `workspace/parsed/` and aggregates throughput statistics under `workspace/telemetry/perf.json`. Nightly CI uploads telemetry artifacts; compare `overall.throughput_bytes_per_second` against the ≥5 MB/s acceptance threshold and update documentation if baselines shift.
- **Fixture cleanup:** Tests automatically reset temporary directories; when recreating fixtures manually, ensure permissions on locked files are restored using the `reset_mode` metadata from fixture builders to avoid residue affecting subsequent runs.

## Open Questions
- What is the largest fixture size we can include without bloating the repo? Consider generating assets during tests.
- Do we need cross-platform validation (Linux/Windows) in Phase 1 or can we target macOS only?
- How should we version-control benchmark baselines (e.g., store in telemetry summary, or separate log)?


