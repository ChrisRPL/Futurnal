# GitHub Connector Testing Guide

This directory contains comprehensive tests for the GitHub Repository Connector, organized according to quality gates and production readiness requirements.

## Test Categories

### Unit Tests (Default)
Standard unit tests for individual components.

```bash
# Run all unit tests
pytest tests/ingestion/github/

# Run with coverage
pytest tests/ingestion/github/ --cov=src/futurnal/ingestion/github --cov-report=html
```

### Integration Tests (`@pytest.mark.github_integration`)
End-to-end workflow tests validating complete sync operations.

```bash
# Run integration tests
pytest tests/ingestion/github/ -m github_integration -v
```

**Coverage:**
- Full repository sync (GraphQL and Git clone)
- Incremental sync workflows
- Webhook-triggered sync
- Issue/PR metadata extraction
- Privacy and consent integration
- Quarantine and recovery workflows
- Multi-repository orchestration

### Security Tests (`@pytest.mark.github_security`)
Security validation tests ensuring credential safety and privacy compliance.

```bash
# Run security tests
pytest tests/ingestion/github/ -m github_security -v
```

**Coverage:**
- Credential leakage detection (logs, exceptions, audit)
- Secret detection effectiveness
- HTTPS-only enforcement
- Token security and rotation
- Privacy-aware logging

### Performance Tests (`@pytest.mark.github_performance`)
Performance benchmarks with defined targets.

```bash
# Run performance tests
pytest tests/ingestion/github/ -m github_performance -v --benchmark-only
```

**Targets:**
- Small repo (<100 files): <10s full sync
- Medium repo (100-1000 files): <60s full sync
- Incremental sync: <5s for 10 files
- API requests: <100 for medium repo
- Memory usage: <500MB peak

### Load Tests (`@pytest.mark.github_load`)
Stress tests for rate limits and concurrent operations.

```bash
# Run load tests
pytest tests/ingestion/github/ -m github_load -v
```

**Coverage:**
- Rate limit compliance (sustained load)
- Concurrent repository sync
- Large repository handling (10k+ files)
- High commit frequency scenarios
- Queue backlog management

### Provider Tests (`@pytest.mark.github_provider_*`)
Provider-specific tests for GitHub.com and Enterprise.

```bash
# Run GitHub.com tests
pytest tests/ingestion/github/ -m github_provider_com -v

# Run GitHub Enterprise tests
pytest tests/ingestion/github/ -m github_provider_enterprise -v
```

### Regression Tests (`@pytest.mark.github_regression`)
Critical path regression tests.

```bash
# Run regression tests
pytest tests/ingestion/github/ -m github_regression -v
```

**Coverage:**
- Repository registration and full sync
- Incremental sync after webhook
- Force push detection and handling
- Consent revocation workflow
- Quarantine and recovery
- Credential refresh during sync

## Test Fixtures

### Repository Fixtures
Located in `tests/ingestion/github/fixtures/repositories.py`:

- `small_test_repo_fixture`: ~10 files for quick tests
- `medium_test_repo_fixture`: ~500 files for realistic testing
- `large_test_repo_fixture`: 10k+ files for load testing
- `repo_with_force_push`: Force push scenario testing
- `repo_with_multiple_branches`: Multi-branch testing

### Enhanced API Mocks
Located in `tests/ingestion/github/fixtures/mock_github_enhanced.py`:

- `enhanced_mock_github_api`: Full-featured GitHub API mock
- `rate_limit_exhausted_api`: Rate limit testing
- `circuit_breaker_open_api`: Circuit breaker testing

## Quality Gates

The connector must meet these quality gates before production deployment:

### Pre-Commit Gates
- ✅ All unit tests pass
- ✅ Code coverage >90%
- ✅ Linting passes (ruff, mypy)
- ✅ Security scans pass

### Pre-Merge Gates
- ✅ All unit tests pass
- ✅ Integration tests pass
- ✅ Security tests pass
- ✅ No credential leakage
- ✅ Code review approved

### Pre-Release Gates
- ✅ All tests pass (unit, integration, security)
- ✅ Performance benchmarks meet targets
- ✅ Load tests pass
- ✅ Manual OAuth flow verified
- ✅ Documentation complete
- ✅ <0.5% failure rate

## Running Quality Gate Evaluation

```bash
# Run quality gate evaluation
python -c "
from futurnal.ingestion.github.quality_gate import (
    GitHubQualityMetrics,
    create_quality_gate_evaluator,
    print_quality_gate_report,
)

# Create metrics
metrics = GitHubQualityMetrics()
metrics.coverage_percentage = 92.5
metrics.total_tests = 600
metrics.passed_tests = 595
metrics.security_tests_total = 23
metrics.security_tests_passed = 23

# Evaluate
evaluator = create_quality_gate_evaluator(metrics=metrics)
result = evaluator.evaluate()

# Print report
print_quality_gate_report(result)
"
```

## Performance Benchmarks

Run performance benchmarks to validate targets:

```bash
# Run with benchmark reporting
pytest tests/ingestion/github/ -m github_performance --benchmark-only --benchmark-columns=min,max,mean,stddev

# Save benchmark results
pytest tests/ingestion/github/ -m github_performance --benchmark-only --benchmark-json=benchmark_results.json
```

## Code Coverage

Generate coverage reports:

```bash
# HTML report
pytest tests/ingestion/github/ --cov=src/futurnal/ingestion/github --cov-report=html
open htmlcov/index.html

# Terminal report
pytest tests/ingestion/github/ --cov=src/futurnal/ingestion/github --cov-report=term-missing

# Check threshold
coverage report --fail-under=90
```

## Continuous Integration

Tests run automatically via GitHub Actions on:
- Every pull request
- Push to main/feat/p1-archivist branches
- Daily at 6 AM UTC (scheduled)

See `.github/workflows/github-connector-tests.yml` for CI/CD configuration.

## Manual Testing

For manual verification procedures, see `MANUAL_TESTING_CHECKLIST.md`.

## Troubleshooting

### Tests Failing Locally

1. **Check Python version**: Must be 3.11+
   ```bash
   python --version
   ```

2. **Update dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Clear pytest cache**:
   ```bash
   pytest --cache-clear
   ```

### Coverage Not Meeting Threshold

1. **Identify uncovered lines**:
   ```bash
   pytest --cov=src/futurnal/ingestion/github --cov-report=term-missing
   ```

2. **Focus on critical paths**: Ensure all main workflows have test coverage

### Performance Tests Failing

1. **Check system resources**: Performance targets assume adequate CPU/memory
2. **Review recent changes**: Performance regressions may indicate inefficiencies
3. **Run with profiling**: Use `pytest-profiling` to identify bottlenecks

## Best Practices

1. **Write tests first**: Follow TDD for new features
2. **Use appropriate markers**: Tag tests with correct pytest markers
3. **Mock external dependencies**: Use fixtures to avoid real API calls
4. **Test edge cases**: Include error scenarios and boundary conditions
5. **Keep tests fast**: Unit tests should complete in <1s each
6. **Document complex tests**: Add docstrings explaining test purpose
7. **Maintain test isolation**: Tests should not depend on each other

## Contributing

When adding new tests:

1. Place in appropriate test file based on component
2. Add appropriate pytest marker (`@pytest.mark.github_integration`, etc.)
3. Use existing fixtures when possible
4. Follow naming convention: `test_<what>_<scenario>()`
5. Add docstring explaining what is being tested
6. Ensure new code maintains >90% coverage

## Reference Documentation

- [Quality Gates Document](../../../docs/phase-1/github-connector-production-plan/11-quality-gates-testing.md)
- [GitHub Connector Architecture](../../../docs/phase-1/github-connector-production-plan/)
- [Manual Testing Checklist](./MANUAL_TESTING_CHECKLIST.md)
