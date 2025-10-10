# Quality Gates & Testing Validation Guide

## Overview

This document describes the comprehensive testing infrastructure for the normalization pipeline quality gates as defined in [12-quality-gates-testing.md](../phase-1/normalization-production-plan/12-quality-gates-testing.md).

The quality gates testing ensures production readiness through:
- **Determinism**: Byte-identical outputs for identical inputs
- **Format Coverage**: All 16+ document formats process successfully
- **Edge Cases**: Robust handling of empty, large, corrupted, and unicode files
- **Privacy Compliance**: No content leakage in logs, metrics, or error messages
- **Performance**: Throughput ≥5 MB/s for mixed document types
- **Production Readiness**: Automated validation of all quality gates

## Test Organization

### Test Suite Structure

```
tests/pipeline/normalization/
├── test_determinism.py              # Determinism validation
├── test_format_coverage.py          # Format coverage validation
├── test_edge_cases.py               # Edge case handling
├── test_privacy_compliance.py       # Privacy audit
├── test_production_readiness.py     # Production readiness checklist
├── test_utils.py                    # Test utilities and assertions
├── fixtures/
│   ├── format_samples.py            # Format test data
│   └── edge_case_data.py            # Edge case test data
└── conftest.py                      # Pytest configuration
```

### Pytest Markers

Tests are organized with custom pytest markers:

```bash
# Run determinism tests only
pytest -m determinism

# Run format coverage tests only
pytest -m format_coverage

# Run edge case tests
pytest -m edge_case

# Run privacy audit tests
pytest -m privacy_audit

# Run production readiness validation
pytest -m production_readiness
```

## Running Tests

### Quick Start

```bash
# Run all quality gates tests
pytest tests/pipeline/normalization/test_*

# Run specific test suite
pytest tests/pipeline/normalization/test_determinism.py -v

# Run with detailed output
pytest tests/pipeline/normalization/ -v --tb=short

# Run production readiness validation
pytest tests/pipeline/normalization/test_production_readiness.py -v
```

### Running Individual Quality Gates

#### 1. Determinism Tests

Validates byte-identical outputs for identical inputs.

```bash
# Run all determinism tests
pytest -m determinism -v

# Run specific determinism test
pytest tests/pipeline/normalization/test_determinism.py::TestContentHashDeterminism -v
```

**Expected Output**: All tests should pass with identical SHA-256 hashes for same content.

#### 2. Format Coverage Tests

Validates all 16 DocumentFormat types process successfully.

```bash
# Run format coverage tests
pytest -m format_coverage -v

# Generate format coverage report
pytest tests/pipeline/normalization/test_format_coverage.py::TestFormatCoverageReport -v
```

**Expected Output**: ≥80% format coverage (ideally 100% for implemented formats).

#### 3. Edge Case Tests

Validates robust handling of edge cases.

```bash
# Run edge case tests (excluding slow tests)
pytest -m edge_case -v

# Run with slow tests (100MB+ files)
pytest -m edge_case --run-slow -v
```

**Expected Output**: Graceful handling of empty files, large files, unicode, and corrupted files.

#### 4. Privacy Compliance Tests

Validates no content leakage in logs, metrics, or error messages.

```bash
# Run privacy audit
pytest -m privacy_audit -v

# Generate privacy compliance report
pytest tests/pipeline/normalization/test_privacy_compliance.py::TestPrivacyComplianceSummary -v
```

**Expected Output**: Zero privacy violations detected.

#### 5. Production Readiness Validation

Comprehensive validation of all quality gates.

```bash
# Run full production readiness validation
pytest tests/pipeline/normalization/test_production_readiness.py -v

# Run quick individual gate checks
pytest tests/pipeline/normalization/test_production_readiness.py::TestIndividualQualityGates -v
```

**Expected Output**: All 10 quality gates passing at 100%.

## Performance Benchmarks

### Running Performance Tests

Performance tests validate throughput ≥5 MB/s target.

```bash
# Run performance benchmarks
pytest -m performance tests/pipeline/normalization/test_performance_benchmarks.py -v

# Run specific benchmark
pytest tests/pipeline/normalization/test_performance_benchmarks.py::TestOverallThroughput::test_mixed_format_throughput -v
```

### Performance Targets

- **Overall Throughput**: ≥5.0 MB/s for mixed formats
- **Markdown**: ≥6.0 MB/s
- **Plain Text**: ≥10.0 MB/s
- **Large Files**: Streaming handles >1GB without OOM

## Test Reports

### Generating Reports

Tests automatically generate JSON reports for CI/CD integration:

```bash
# Run with report generation
pytest tests/pipeline/normalization/test_production_readiness.py -v --tb=short

# Reports are saved to tmp_path during tests
# In production, configure output directory via pytest fixtures
```

### Report Types

1. **Format Coverage Report** (`format_coverage_report.json`)
   - Total formats tested
   - Success/failure breakdown
   - Per-format results

2. **Production Readiness Checklist** (`production_readiness_checklist.json`)
   - 10 quality gate status
   - Overall passing percentage
   - Detailed validation results

3. **Performance Analysis** (embedded in test output)
   - Throughput metrics
   - Per-format performance
   - Memory usage validation

## CI/CD Integration

### Recommended CI Pipeline

```yaml
# Example GitHub Actions workflow
name: Quality Gates

on: [push, pull_request]

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run Determinism Tests
        run: pytest -m determinism -v

      - name: Run Format Coverage
        run: pytest -m format_coverage -v

      - name: Run Privacy Audit
        run: pytest -m privacy_audit -v

      - name: Run Production Readiness
        run: pytest -m production_readiness -v

      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: quality-gates-reports
          path: |
            **/format_coverage_report.json
            **/production_readiness_checklist.json
```

### Pre-Commit Hooks

Run quality gates before committing:

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: quality-gates
        name: Quality Gates Validation
        entry: pytest -m "determinism or format_coverage"
        language: system
        pass_filenames: false
```

## Troubleshooting

### Common Issues

#### Import Errors

If you encounter import errors:

```bash
# Ensure tests are run from project root
cd /path/to/Futurnal
pytest tests/pipeline/normalization/test_determinism.py
```

#### Slow Tests

Skip slow tests (100MB+ files) by default:

```bash
# Default: excludes slow tests
pytest tests/pipeline/normalization/

# Explicitly run slow tests
pytest tests/pipeline/normalization/ --run-slow
```

#### Format Fixtures Missing

If format fixtures are missing, check:

```bash
# Verify fixtures are available
pytest tests/pipeline/normalization/fixtures/format_samples.py --collect-only
```

### Debug Mode

Run with verbose debugging:

```bash
# Maximum verbosity
pytest tests/pipeline/normalization/ -vv --tb=long --log-cli-level=DEBUG

# Capture logs
pytest tests/pipeline/normalization/ --log-file=quality_gates.log
```

## Production Readiness Checklist

The production readiness validation checks all 10 quality gates:

- ✅ **Format Coverage**: All 16+ formats parse successfully
- ✅ **Determinism**: Tests pass 100% (byte-identical outputs)
- ✅ **Performance**: Throughput ≥5 MB/s
- ✅ **Memory**: Usage <2GB for largest test documents
- ✅ **Integration**: All connector types work
- ✅ **Quarantine**: Workflow handles all failure modes
- ✅ **Privacy**: Audit shows no content leakage
- ✅ **Streaming**: Handles >1GB documents without OOM
- ✅ **Offline**: No network calls during normalization
- ✅ **Metrics**: Exported to telemetry correctly

### Validation Command

```bash
# Run comprehensive validation
pytest tests/pipeline/normalization/test_production_readiness.py::TestProductionReadinessValidation::test_comprehensive_production_readiness_validation -v

# Expected output: All 10 gates passing
```

## Test Coverage Goals

- **Unit Tests**: 100% coverage of normalization service
- **Integration Tests**: All connectors → normalization → PKG flows
- **Format Coverage**: 100% of implemented DocumentFormat types
- **Edge Cases**: Comprehensive coverage of failure modes
- **Privacy**: Zero content leakage in all scenarios

## Contributing

When adding new tests:

1. **Follow existing patterns** in test files
2. **Use appropriate pytest markers** (`@pytest.mark.determinism`, etc.)
3. **Add fixtures** to `format_samples.py` or `edge_case_data.py`
4. **Update this documentation** with new test suites
5. **Ensure tests are privacy-safe** (no real sensitive data)

## References

- [Quality Gates Production Plan](../phase-1/normalization-production-plan/12-quality-gates-testing.md)
- [System Requirements](../requirements/system-requirements.md)
- [System Architecture](../architecture/system-architecture.md)
- [Development Guide](../../DEVELOPMENT_GUIDE.md)
