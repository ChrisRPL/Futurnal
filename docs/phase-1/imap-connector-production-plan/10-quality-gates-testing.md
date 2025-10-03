Summary: Define quality gates, testing strategy, and success metrics for production-ready IMAP connector.

# 10 · Quality Gates & Testing

## Purpose
Establish comprehensive quality gates, testing strategy, and success metrics to ensure the IMAP connector meets production-ready standards: <0.5% failure rate, 5-minute detection window, privacy compliance, and seamless integration with existing Futurnal architecture.

## Scope
- Unit test coverage requirements
- Integration test scenarios
- Performance benchmarks and load tests
- Security and privacy validation
- Provider-specific testing (Gmail, Office 365, generic IMAP)
- Connection reliability tests
- Thread reconstruction accuracy tests
- OAuth token refresh tests
- Quality gate metrics and thresholds
- CI/CD integration

## Quality Gates

### Critical Quality Gates (Must Pass)
```python
class ImapQualityGates(BaseModel):
    """Quality gate thresholds for IMAP connector."""

    # Reliability
    max_sync_failure_rate: float = 0.005  # <0.5%
    max_connection_failure_rate: float = 0.01  # <1%
    max_parse_failure_rate: float = 0.02  # <2%

    # Performance
    max_detection_window_minutes: int = 5  # 5-minute detection
    min_throughput_messages_per_second: float = 1.0  # ≥1 msg/s
    max_sync_latency_seconds: float = 30.0  # <30s per folder

    # Accuracy
    min_thread_reconstruction_accuracy: float = 0.95  # 95% correct
    min_attachment_extraction_accuracy: float = 0.98  # 98% correct

    # Privacy
    zero_pii_in_logs: bool = True  # No PII leaked
    require_consent_coverage: float = 1.0  # 100% consent checks

    # Integration
    min_element_sink_success_rate: float = 0.99  # 99% elements processed
    min_state_persistence_success_rate: float = 1.0  # 100% state saved
```

### Quality Gate Evaluation
```python
class QualityGateEvaluator:
    """Evaluate quality gate metrics against thresholds."""

    def __init__(self, gates: ImapQualityGates):
        self.gates = gates

    def evaluate(self, metrics: Dict[str, float]) -> QualityGateResult:
        """Evaluate metrics against quality gates."""
        results = QualityGateResult()

        # Reliability checks
        if metrics.get('sync_failure_rate', 0) > self.gates.max_sync_failure_rate:
            results.add_failure(
                "Sync failure rate exceeds threshold",
                actual=metrics['sync_failure_rate'],
                threshold=self.gates.max_sync_failure_rate,
            )

        if metrics.get('connection_failure_rate', 0) > self.gates.max_connection_failure_rate:
            results.add_failure(
                "Connection failure rate exceeds threshold",
                actual=metrics['connection_failure_rate'],
                threshold=self.gates.max_connection_failure_rate,
            )

        # Performance checks
        if metrics.get('avg_detection_minutes', 999) > self.gates.max_detection_window_minutes:
            results.add_failure(
                "Detection window exceeds 5 minutes",
                actual=metrics['avg_detection_minutes'],
                threshold=self.gates.max_detection_window_minutes,
            )

        # Privacy checks
        if metrics.get('pii_leak_count', 0) > 0:
            results.add_critical_failure(
                "PII detected in logs",
                actual=metrics['pii_leak_count'],
                threshold=0,
            )

        # ... more checks

        return results
```

## Testing Strategy

### Unit Test Coverage
```
Target: ≥90% code coverage

Key areas:
- Credential management (token storage, refresh, deletion)
- Connection lifecycle (connect, authenticate, disconnect, retry)
- Email parsing (headers, body, attachments, threading)
- Sync logic (UID, MODSEQ, deletion detection)
- Thread reconstruction (graph building, participant extraction)
- Privacy redaction (email addresses, subjects, bodies)
- Consent checking (grant, revoke, require)

Test structure:
tests/ingestion/imap/
├── test_credential_manager.py (30+ tests)
├── test_connection_manager.py (25+ tests)
├── test_email_parser.py (40+ tests)
├── test_sync_engine.py (35+ tests)
├── test_thread_reconstructor.py (20+ tests)
├── test_attachment_extractor.py (15+ tests)
├── test_privacy_redaction.py (20+ tests)
└── test_consent_manager.py (15+ tests)
```

### Integration Test Scenarios
```python
@pytest.mark.integration
class TestImapIntegration:
    """Integration tests with mock IMAP server."""

    def test_end_to_end_sync_gmail_format(self):
        """Test complete sync workflow with Gmail-formatted messages."""
        # Setup mock IMAP server with Gmail responses
        # Perform full sync
        # Verify all messages processed
        # Verify threads reconstructed
        # Verify elements sent to sink

    def test_oauth_token_refresh_during_sync(self):
        """Test automatic token refresh during long sync."""
        # Setup expired OAuth token
        # Start sync (should auto-refresh)
        # Verify sync completes successfully

    def test_idle_monitoring_new_message(self):
        """Test IDLE detects new message in real-time."""
        # Start IDLE monitor
        # Simulate new message on server
        # Verify sync triggered within 5 seconds

    def test_connection_retry_on_network_failure(self):
        """Test exponential backoff retry on network errors."""
        # Simulate network failures
        # Verify retry with increasing delays
        # Verify eventual success

    def test_uidvalidity_change_full_resync(self):
        """Test full resync when UIDVALIDITY changes."""
        # Perform initial sync
        # Change UIDVALIDITY on server
        # Perform sync (should detect change)
        # Verify full resync triggered

    def test_multi_folder_concurrent_sync(self):
        """Test concurrent sync of multiple folders."""
        # Setup mailbox with 5 folders
        # Trigger sync
        # Verify all folders synced
        # Verify connection pool managed correctly
```

### Provider-Specific Tests
```python
# Gmail-specific tests
@pytest.mark.provider_gmail
class TestGmailIntegration:
    def test_gmail_labels_extraction(self):
        """Test Gmail label handling."""

    def test_gmail_special_folders(self):
        """Test Gmail [Gmail]/... folders."""

    def test_gmail_oauth_flow(self):
        """Test Gmail OAuth2 consent flow."""

# Office 365-specific tests
@pytest.mark.provider_office365
class TestOffice365Integration:
    def test_office365_oauth_flow(self):
        """Test Office 365 OAuth2 consent flow."""

    def test_office365_folder_structure(self):
        """Test Office 365 folder naming."""

# Generic IMAP tests
@pytest.mark.provider_generic
class TestGenericImapIntegration:
    def test_app_password_auth(self):
        """Test app password authentication."""

    def test_starttls_connection(self):
        """Test STARTTLS upgrade."""
```

### Performance Tests
```python
@pytest.mark.performance
class TestImapPerformance:
    """Performance and load tests."""

    def test_large_mailbox_initial_sync(self):
        """Test initial sync of mailbox with 10,000 messages."""
        # Setup mock server with 10k messages
        # Measure sync time
        # Assert: <3 hours for 10k messages
        # Assert: >1 msg/second throughput

    def test_high_volume_idle_events(self):
        """Test IDLE handling under high message volume."""
        # Simulate 100 messages arriving during IDLE
        # Verify all detected within 5 minutes

    def test_connection_pool_efficiency(self):
        """Test connection pool reuse and limits."""
        # Perform 100 concurrent operations
        # Verify max connections respected
        # Verify connection reuse

    def test_memory_usage_long_running_sync(self):
        """Test memory doesn't leak during long sync."""
        # Run 1-hour sync simulation
        # Monitor memory usage
        # Assert: <500MB peak memory
```

### Security & Privacy Tests
```python
@pytest.mark.security
class TestImapSecurity:
    """Security and privacy validation."""

    def test_tls_enforcement(self):
        """Verify TLS-only connections."""
        # Attempt port 143 connection
        # Assert: Rejected or STARTTLS required

    def test_credentials_never_logged(self):
        """Verify credentials never appear in logs."""
        # Perform operations with logging
        # Parse all log output
        # Assert: No tokens, passwords in logs

    def test_pii_redaction_in_logs(self):
        """Verify email addresses redacted in logs."""
        # Perform sync with logging
        # Parse all log output
        # Assert: No plaintext email addresses

    def test_email_bodies_never_logged(self):
        """Verify email bodies never logged."""
        # Process emails with logging
        # Parse all log output
        # Assert: No email body text in logs

    def test_consent_enforcement(self):
        """Verify operations blocked without consent."""
        # Revoke consent
        # Attempt sync
        # Assert: ConsentRequiredError raised

    def test_audit_log_integrity(self):
        """Verify audit logs tamper-evident."""
        # Generate audit events
        # Attempt to modify audit log
        # Verify detection (hash chain)
```

### Thread Reconstruction Accuracy Tests
```python
@pytest.mark.accuracy
class TestThreadAccuracy:
    """Thread reconstruction accuracy tests."""

    def test_simple_thread_reconstruction(self):
        """Test 3-message thread reconstruction."""
        # Setup: A → B → C
        # Verify: Correct parent-child relationships

    def test_branching_thread_reconstruction(self):
        """Test thread with multiple branches."""
        # Setup: A → B → C
        #            → D → E
        # Verify: Correct tree structure

    def test_out_of_order_arrival(self):
        """Test thread reconstruction with out-of-order messages."""
        # Arrive in order: C, A, B
        # Verify: Correct thread after all arrived

    def test_missing_parent_handling(self):
        """Test orphan message (parent not in mailbox)."""
        # Setup: Only B and C (A missing)
        # Verify: Placeholder parent created

    def test_thread_accuracy_real_world_dataset(self):
        """Test accuracy on real-world email dataset."""
        # Load 1000 real emails from test dataset
        # Reconstruct threads
        # Compare to ground truth
        # Assert: ≥95% accuracy
```

## Test Infrastructure

### Mock IMAP Server
```python
class MockImapServer:
    """Mock IMAP server for testing."""

    def __init__(self):
        self.messages: Dict[int, bytes] = {}
        self.uidvalidity = 1
        self.capabilities = [b'IMAP4rev1', b'IDLE', b'CONDSTORE']

    def add_message(self, uid: int, raw_message: bytes):
        """Add message to server."""
        self.messages[uid] = raw_message

    def simulate_new_message(self, raw_message: bytes) -> int:
        """Simulate new message arrival."""
        uid = max(self.messages.keys()) + 1 if self.messages else 1
        self.messages[uid] = raw_message
        return uid

    def simulate_deletion(self, uid: int):
        """Simulate message deletion."""
        if uid in self.messages:
            del self.messages[uid]

    def change_uidvalidity(self):
        """Simulate UIDVALIDITY change (e.g., mailbox migration)."""
        self.uidvalidity += 1
        self.messages.clear()
```

### Test Fixtures
```python
@pytest.fixture
def mock_imap_server():
    """Provide mock IMAP server."""
    return MockImapServer()

@pytest.fixture
def sample_email_message():
    """Provide sample RFC822 email."""
    return b"""From: sender@example.com
To: recipient@example.com
Subject: Test Email
Message-ID: <test123@example.com>
Date: Mon, 1 Jan 2024 12:00:00 +0000

This is a test email body.
"""

@pytest.fixture
def gmail_threaded_messages():
    """Provide Gmail-style threaded messages."""
    # Return list of messages forming a thread
    pass
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
name: IMAP Connector Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/ingestion/imap/ -m "not integration and not performance"

  integration-tests:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
      - name: Run integration tests
        run: pytest tests/ingestion/imap/ -m integration

  security-tests:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security tests
        run: pytest tests/ingestion/imap/ -m security
      - name: Check for PII in logs
        run: |
          grep -r '@' test_logs/ && exit 1 || exit 0
```

## Acceptance Criteria

- ✅ ≥90% unit test coverage
- ✅ All critical quality gates pass
- ✅ <0.5% sync failure rate in integration tests
- ✅ 5-minute detection window validated
- ✅ ≥95% thread reconstruction accuracy
- ✅ Zero PII in logs (security tests pass)
- ✅ OAuth token refresh works automatically
- ✅ Provider-specific tests pass (Gmail, Office 365)
- ✅ Performance benchmarks met (≥1 msg/s)
- ✅ CI/CD pipeline green

## Success Metrics

### Phase 1 Targets (MVP)
- 200+ unit tests passing
- 50+ integration tests passing
- 10+ security tests passing
- All quality gates green
- Successfully sync 3 mailboxes (Gmail, Office 365, generic)

### Production Readiness Targets
- 1-month field test with 10+ users
- <0.5% failure rate in production
- 95%+ user satisfaction
- Zero security incidents
- Zero privacy violations

## Open Questions

- Should we implement chaos testing (random network failures)?
- How to test OAuth flows without real credentials?
- Should we create a test email dataset for public use?
- How to benchmark against commercial email clients?
- Should we implement automated performance regression detection?

## Dependencies
- pytest for test framework
- pytest-asyncio for async tests
- pytest-mock for mocking
- Mock IMAP server implementation
- All connector components from tasks 01-09


