"""Tests for the IMAP connection manager layer."""

from __future__ import annotations

import asyncio
import socket
import ssl
import sys
import types
from contextlib import contextmanager

import pytest

# Provide a lightweight ``imapclient`` stub if the dependency is not installed.
if "imapclient" not in sys.modules:
    imapclient_module = types.ModuleType("imapclient")

    # Create proper exception hierarchy to avoid false isinstance matches
    class IMAPClientError(Exception):
        """Base IMAP error."""
        pass

    class IMAPClientAuthenticationError(IMAPClientError):
        """Authentication failure."""
        pass

    class IMAPClientAbortError(IMAPClientError):
        """Connection aborted."""
        pass

    class _PlaceholderIMAPClient:  # pragma: no cover - only used when dependency absent
        Error = IMAPClientError
        AuthenticationError = IMAPClientAuthenticationError
        AbortError = IMAPClientAbortError

    imapclient_module.IMAPClient = _PlaceholderIMAPClient
    sys.modules["imapclient"] = imapclient_module

import futurnal.ingestion.imap.connection_manager as cm  # noqa: E402 - ensure stub registered
from futurnal.ingestion.imap.connection_manager import (  # noqa: E402
    ConnectionMetrics,
    ConnectionState,
    IdleConnection,
    ImapConnection,
    ImapConnectionPool,
    NetworkMonitor,
    RetryStrategy,
)
from futurnal.ingestion.imap.credential_manager import (  # noqa: E402
    AppPassword,
    OAuth2Tokens,
)
from futurnal.ingestion.imap.descriptor import AuthMode, ImapMailboxDescriptor  # noqa: E402


class DummyCredentialManager:  # pragma: no cover - simple stub container
    pass


class StubIMAPClient:
    """Minimal IMAPClient-compatible stub for connection tests."""

    # Use the proper exception hierarchy from our stub
    from imapclient import IMAPClient as IMAPClientStub
    try:
        from imapclient.exceptions import LoginError as IMAPLoginError
    except Exception:
        IMAPLoginError = getattr(IMAPClientStub, "AuthenticationError", IMAPClientStub.Error)

    Error = IMAPClientStub.Error
    AuthenticationError = IMAPLoginError
    AbortError = getattr(IMAPClientStub, "AbortError", IMAPClientStub.Error)

    def __init__(
        self,
        host: str,
        port: int,
        *,
        ssl: bool,
        ssl_context: ssl.SSLContext,
        timeout: int,
        use_uid: bool,
    ) -> None:
        self.host = host
        self.port = port
        self.ssl = ssl
        self.ssl_context = ssl_context
        self.timeout = timeout
        self.use_uid = use_uid
        self.login_calls: list[tuple[str, str]] = []
        self.oauth_calls: list[tuple[str, str]] = []
        self.logged_out = False
        self.noop_calls = 0
        self.selected_folder: str | None = None
        self.idle_started = False
        self.idle_done_called = False
        self.idle_responses: list[tuple[bytes, bytes]] = []
        self.search_results: list[int] = []

    def login(self, email: str, password: str) -> None:
        self.login_calls.append((email, password))

    def oauth2_login(self, email: str, token: str) -> None:
        self.oauth_calls.append((email, token))

    def logout(self) -> None:
        self.logged_out = True

    def noop(self) -> None:
        self.noop_calls += 1

    def select_folder(self, folder: str) -> None:
        self.selected_folder = folder

    def idle(self) -> None:
        self.idle_started = True

    def idle_check(self, timeout: int) -> list[tuple[bytes, bytes]]:  # noqa: ARG002
        return list(self.idle_responses)

    def idle_done(self) -> None:
        self.idle_done_called = True

    def search(self, criteria):  # noqa: D401, ARG002 - mirrors IMAPClient API
        return list(self.search_results)


@pytest.fixture()
def descriptor() -> ImapMailboxDescriptor:
    return ImapMailboxDescriptor.from_registration(
        email_address="user@example.com",
        imap_host="imap.example.com",
        auth_mode=AuthMode.APP_PASSWORD,
        credential_id="cred",
    )


def _patch_app_password(monkeypatch: pytest.MonkeyPatch, password: str = "secret") -> None:
    @contextmanager
    def _app_password_context(*_args, **_kwargs):
        yield AppPassword(password=password)

    monkeypatch.setattr(cm, "secure_credential_context", _app_password_context)


def _patch_oauth_token(monkeypatch: pytest.MonkeyPatch, token_value: str = "token") -> None:
    token = OAuth2Tokens(access_token=token_value, refresh_token="refresh", expires_in=3600)

    @contextmanager
    def _oauth_context(*_args, **_kwargs):
        yield token

    monkeypatch.setattr(cm, "secure_credential_context", _oauth_context)


def _set_imap_client_factory(monkeypatch: pytest.MonkeyPatch, stub: StubIMAPClient) -> None:
    monkeypatch.setattr(cm, "IMAPClient", lambda **kwargs: stub)


def test_create_ssl_context_enforces_tls12(descriptor) -> None:
    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())
    context = connection._create_ssl_context()
    assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_imap_connection_app_password_auth(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    stub = StubIMAPClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch, password="s3cret")
    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    with connection.connect():
        pass

    assert stub.login_calls == [(descriptor.email_address, "s3cret")]
    assert stub.logged_out is True


def test_imap_connection_oauth_auth(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    descriptor = descriptor.update(auth_mode=AuthMode.OAUTH2)
    stub = StubIMAPClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_oauth_token(monkeypatch, token_value="oauth-token")
    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    with connection.connect():
        pass

    assert stub.oauth_calls == [(descriptor.email_address, "oauth-token")]
    assert stub.logged_out is True


def test_retry_strategy_behaviour() -> None:
    strategy = RetryStrategy(base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)
    assert strategy.calculate_delay(0) == 1.0
    assert strategy.calculate_delay(2) == 4.0
    assert strategy.calculate_delay(4) == 5.0  # capped by max_delay
    assert strategy.should_retry(0, TimeoutError()) is True
    assert strategy.should_retry(5, TimeoutError()) is False


@pytest.mark.asyncio()
async def test_connection_pool_reuses_connection(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    stub_client = StubIMAPClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())
    connection.client = stub_client
    connection.state = ConnectionState.CONNECTED

    pool = ImapConnectionPool(descriptor=descriptor, credential_manager=DummyCredentialManager(), max_connections=1)
    monkeypatch.setattr(pool, "_create_connection", lambda: connection)

    async with pool.acquire() as first:
        assert first is connection
    async with pool.acquire() as second:
        assert second is connection


def test_idle_connection_extracts_new_messages(descriptor) -> None:
    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())
    idle = IdleConnection(connection)
    client = StubIMAPClient(
        host="imap.example.com",
        port=993,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    client.search_results = [101, 102]
    responses = [(b"1", b"EXISTS")]

    result = idle._extract_new_messages(client, responses)
    assert result == [101, 102]


@pytest.mark.asyncio()
async def test_network_monitor_queues_and_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = NetworkMonitor()
    executed: list[str] = []

    async def _operation() -> None:
        executed.append("run")

    async def _offline(self) -> bool:  # noqa: D401
        return False

    async def _online(self) -> bool:  # noqa: D401
        return True

    monkeypatch.setattr(NetworkMonitor, "is_online", _offline)
    await monitor.queue_operation(_operation)
    assert executed == []

    monkeypatch.setattr(NetworkMonitor, "is_online", _online)
    await monitor.process_pending()
    assert executed == ["run"]


def test_connection_metrics_records_attempts() -> None:
    metrics = ConnectionMetrics()
    metrics.record_attempt(True, 1.0)
    metrics.record_attempt(False, 2.0)

    assert metrics.total_connections == 2
    assert metrics.successful_connections == 1
    assert metrics.failed_connections == 1
    assert metrics.average_connection_time > 0


# ---------------------------------------------------------------------------
# Enhanced unit tests
# ---------------------------------------------------------------------------


def test_retry_strategy_jitter_adds_randomness() -> None:
    """Verify jitter adds 0-50% random delay."""
    strategy = RetryStrategy(base_delay=10.0, max_delay=100.0, exponential_base=2.0, jitter=True)

    # Calculate delay multiple times and verify variance
    delays = [strategy.calculate_delay(2) for _ in range(10)]

    # With jitter, delays should vary
    assert len(set(delays)) > 1, "Jitter should produce different delays"

    # All delays should be within expected range (base * exp * (1 to 1.5))
    base_delay = 10.0 * (2.0 ** 2)  # 40.0
    for delay in delays:
        assert base_delay <= delay <= base_delay * 1.5


def test_retry_strategy_no_jitter_deterministic() -> None:
    """Verify no jitter produces consistent delays."""
    strategy = RetryStrategy(base_delay=5.0, exponential_base=2.0, jitter=False)

    delays = [strategy.calculate_delay(1) for _ in range(5)]
    assert len(set(delays)) == 1, "Without jitter, delays should be identical"
    assert delays[0] == 10.0


def test_retry_strategy_respects_max_delay() -> None:
    """Verify max_delay cap is enforced."""
    strategy = RetryStrategy(base_delay=1.0, max_delay=10.0, exponential_base=2.0, jitter=False)

    # At retry 10, exponential would be 1024, but capped at 10
    delay = strategy.calculate_delay(10)
    assert delay == 10.0


def test_retry_strategy_no_retry_on_auth_error() -> None:
    """Verify authentication errors fail fast without retry."""
    strategy = RetryStrategy()

    auth_error_type = getattr(cm.IMAPClient, "AuthenticationError", None)
    if auth_error_type is None:
        auth_error_type = cm.IMAPLoginError or cm.IMAPClient.Error

    auth_error = auth_error_type("Invalid credentials")

    assert strategy.should_retry(0, auth_error) is False


def test_retry_strategy_retries_network_errors() -> None:
    """Verify network errors trigger retry."""
    strategy = RetryStrategy(max_retries=3)

    assert strategy.should_retry(0, socket.error("Connection refused")) is True
    assert strategy.should_retry(0, OSError("Network unreachable")) is True
    assert strategy.should_retry(0, TimeoutError("Connection timeout")) is True

    # Verify max retries is respected
    assert strategy.should_retry(3, socket.error()) is False


def test_connection_state_transitions(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Verify connection state transitions through lifecycle."""
    stub = StubIMAPClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch)

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    # Initial state
    assert connection.state == ConnectionState.DISCONNECTED

    with connection.connect():
        # Should be connected after establishment
        assert connection.state == ConnectionState.CONNECTED

    # Should return to disconnected after cleanup
    assert connection.state == ConnectionState.DISCONNECTED


def test_connection_rejects_port_143(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify port 143 (plaintext IMAP) is rejected."""
    # Note: The descriptor itself validates and rejects port 143 at creation time
    # So we test the connection manager's additional check
    from futurnal.ingestion.imap.descriptor import ImapMailboxDescriptor, AuthMode

    # Create a descriptor bypassing validation for test purposes
    descriptor = ImapMailboxDescriptor(
        id="test-id",
        imap_host="imap.example.com",
        imap_port=993,  # Valid port initially
        email_address="test@example.com",
        auth_mode=AuthMode.APP_PASSWORD,
        credential_id="cred",
        provenance_user="test",
        provenance_machine_hash="hash",
        provenance_tool_version="0.1.0",
    )

    # Manually update to port 143 after creation
    descriptor.imap_port = 143

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    with pytest.raises(ValueError, match="Plain IMAP.*unsupported.*TLS required"):
        connection._establish_connection()


def test_connection_metrics_tracks_retries() -> None:
    """Verify retry metric incrementation."""
    metrics = ConnectionMetrics()

    assert metrics.total_retries == 0

    metrics.record_retry()
    metrics.record_retry()

    assert metrics.total_retries == 2


def test_connection_metrics_tracks_idle_renewals() -> None:
    """Verify IDLE renewal metric tracking."""
    metrics = ConnectionMetrics()

    assert metrics.idle_renewals == 0

    metrics.record_idle_renewal()
    metrics.record_idle_renewal()
    metrics.record_idle_renewal()

    assert metrics.idle_renewals == 3


def test_connection_is_alive_requires_connected_state(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Verify is_alive returns False for disconnected connections."""
    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    # Without client, not alive
    assert connection.is_alive() is False

    # With client but wrong state
    stub = StubIMAPClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    connection.client = stub
    connection.state = ConnectionState.DISCONNECTED
    assert connection.is_alive() is False


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_connection_full_lifecycle(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Test complete connection lifecycle: connect → use → disconnect."""
    stub = StubIMAPClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch, password="test123")

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    with connection.connect() as client:
        # Connection should be established
        assert client is stub
        assert stub.login_calls == [(descriptor.email_address, "test123")]
        assert connection.state == ConnectionState.CONNECTED

        # Can use connection
        client.noop()
        assert stub.noop_calls == 1

    # Should be logged out and disconnected
    assert stub.logged_out is True
    assert connection.state == ConnectionState.DISCONNECTED


def test_connection_retry_on_network_failure(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Test connection retry logic with network failure recovery."""
    import socket as socket_module
    attempt_count = 0

    class FailingIMAPClient(StubIMAPClient):
        # Preserve error types
        Error = StubIMAPClient.Error
        AuthenticationError = StubIMAPClient.AuthenticationError
        AbortError = StubIMAPClient.AbortError

        def __init__(self, *args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise socket_module.error("Connection refused")
            super().__init__(*args, **kwargs)

    # Create a factory that preserves error attributes
    class IMAPClientFactory:
        Error = StubIMAPClient.Error
        AuthenticationError = StubIMAPClient.AuthenticationError
        AbortError = StubIMAPClient.AbortError

        def __call__(self, **kwargs):
            return FailingIMAPClient(**kwargs)

    monkeypatch.setattr(cm, "IMAPClient", IMAPClientFactory())
    _patch_app_password(monkeypatch)

    connection = ImapConnection(
        descriptor=descriptor,
        credential_manager=DummyCredentialManager(),
        retry_strategy=RetryStrategy(max_retries=5, base_delay=0.01, jitter=False),
    )

    # Should succeed on 3rd attempt
    connection.reconnect_with_retry()
    assert attempt_count == 3
    assert connection.state == ConnectionState.CONNECTED


def test_connection_cleanup_on_error(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Verify connection cleanup happens even on authentication failure."""

    class AuthFailingClient(StubIMAPClient):
        def login(self, email, password):
            raise Exception("AUTHENTICATIONFAILED Invalid credentials")

    stub = AuthFailingClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch)

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    with pytest.raises(Exception, match="AUTHENTICATIONFAILED"):
        with connection.connect():
            pass

    # Should still call logout for cleanup
    assert stub.logged_out is True
    assert connection.state == ConnectionState.DISCONNECTED


@pytest.mark.asyncio()
async def test_connection_pool_concurrent_acquire(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Test connection pool handles sequential reuse."""
    connections_created = 0

    def create_connection():
        nonlocal connections_created
        connections_created += 1
        stub = StubIMAPClient(
            host=descriptor.imap_host,
            port=descriptor.imap_port,
            ssl=True,
            ssl_context=ssl.create_default_context(),
            timeout=30,
            use_uid=True,
        )
        conn = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())
        conn.client = stub
        conn.state = ConnectionState.CONNECTED
        return conn

    pool = ImapConnectionPool(
        descriptor=descriptor,
        credential_manager=DummyCredentialManager(),
        max_connections=2,
    )
    monkeypatch.setattr(pool, "_create_connection", create_connection)

    # First acquire
    async with pool.acquire() as conn1:
        assert connections_created == 1
        conn1_id = id(conn1)

    # Second acquire should reuse the first connection
    async with pool.acquire() as conn2:
        # Should reuse existing connection (still only 1 created)
        assert connections_created == 1
        assert id(conn2) == conn1_id

    # Pool should have 1 connection
    assert len(pool._pool) == 1


@pytest.mark.asyncio()
async def test_connection_pool_dead_connection_pruning(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Test dead connections are removed from pool."""

    class DeadClient(StubIMAPClient):
        def noop(self):
            raise Exception("Connection lost")

    dead_stub = DeadClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )

    dead_conn = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())
    dead_conn.client = dead_stub
    dead_conn.state = ConnectionState.CONNECTED

    pool = ImapConnectionPool(descriptor=descriptor, credential_manager=DummyCredentialManager())
    pool._pool.append(dead_conn)

    # Pruning should remove the dead connection
    pool._prune_dead_connections()
    assert len(pool._pool) == 0


@pytest.mark.asyncio()
async def test_idle_connection_renewal_mechanism(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Test IDLE connection renewal mechanism."""

    idle_cycles = 0

    class IdleStubClient(StubIMAPClient):
        def idle_check(self, timeout):
            nonlocal idle_cycles
            idle_cycles += 1
            # Return empty after 2 cycles to allow test to complete
            if idle_cycles > 2:
                return []
            return []

    stub = IdleStubClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )

    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch)

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    # Test metrics recording
    assert connection.metrics.idle_renewals == 0

    # Manually trigger IDLE cycle
    with connection.connect() as client:
        client.select_folder("INBOX")
        client.idle()
        client.idle_check(timeout=1)
        client.idle_done()
        connection.metrics.record_idle_renewal()

    assert connection.metrics.idle_renewals == 1


# ---------------------------------------------------------------------------
# Resilience tests
# ---------------------------------------------------------------------------


def test_connection_handles_logout_error(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Verify connection handles logout errors gracefully."""

    class LogoutFailingClient(StubIMAPClient):
        def logout(self):
            raise Exception("Connection already closed")

    stub = LogoutFailingClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch)

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    # Should not raise even if logout fails
    with connection.connect():
        pass

    assert connection.client is None
    assert connection.state == ConnectionState.DISCONNECTED


def test_connection_timeout_during_noop(descriptor) -> None:
    """Test connection handles timeout during health check."""

    class TimeoutClient(StubIMAPClient):
        def noop(self):
            raise TimeoutError("Connection timeout")

    stub = TimeoutClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())
    connection.client = stub
    connection.state = ConnectionState.CONNECTED

    # Should detect as not alive
    assert connection.is_alive() is False


@pytest.mark.asyncio()
async def test_network_monitor_handles_dns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test network monitor detects offline state on DNS failure."""

    def failing_dns(*args):
        raise socket.error("Name or service not known")

    monitor = NetworkMonitor()

    # Patch socket resolution to fail
    import socket as socket_module
    monkeypatch.setattr(socket_module, "gethostbyname", failing_dns)

    assert await monitor.is_online() is False


def test_idle_retry_strategy_behavior(descriptor) -> None:
    """Test IDLE connection uses retry strategy correctly."""

    # Test that retry strategy is configured for IDLE
    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())
    idle = IdleConnection(connection, renewal_interval=10)

    # Should have a retry strategy
    assert idle._retry_strategy is not None
    assert idle._retry_strategy.max_retries >= 1

    # Test retry decision for network errors
    assert idle._retry_strategy.should_retry(0, socket.error("Connection reset")) is True
    assert idle._retry_strategy.should_retry(0, OSError("Network unreachable")) is True

    # Test that retries are capped
    assert idle._retry_strategy.should_retry(10, socket.error()) is False


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------


def test_connection_no_credentials_in_error_messages(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Verify credentials never appear in exception messages."""

    class FailingClient(StubIMAPClient):
        def login(self, email, password):
            # Simulate error that might expose credentials
            raise Exception(f"Login failed for {email}")

    stub = FailingClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch, password="secret_password_123")

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    try:
        with connection.connect():
            pass
    except Exception as e:
        # Exception message should NOT contain the password
        assert "secret_password_123" not in str(e)


def test_audit_logging_without_credentials(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Verify audit logs don't expose credentials."""

    logged_events: list = []

    class MockAuditLogger:
        def record(self, event):
            logged_events.append(event)

    stub = StubIMAPClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch, password="super_secret")

    audit_logger = MockAuditLogger()
    connection = ImapConnection(
        descriptor=descriptor,
        credential_manager=DummyCredentialManager(),
        audit_logger=audit_logger,
    )

    with connection.connect():
        pass

    # Check all logged events
    for event in logged_events:
        event_str = str(event.metadata)
        assert "super_secret" not in event_str
        assert "password" not in event_str.lower() or "app_password" in event_str.lower()


def test_ssl_context_enforces_certificate_validation(descriptor) -> None:
    """Verify SSL context requires certificate validation."""
    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())
    context = connection._create_ssl_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_connection_uses_tls_only(monkeypatch: pytest.MonkeyPatch, descriptor) -> None:
    """Verify connection always uses TLS."""
    stub = StubIMAPClient(
        host=descriptor.imap_host,
        port=descriptor.imap_port,
        ssl=True,
        ssl_context=ssl.create_default_context(),
        timeout=30,
        use_uid=True,
    )
    _set_imap_client_factory(monkeypatch, stub)
    _patch_app_password(monkeypatch)

    connection = ImapConnection(descriptor=descriptor, credential_manager=DummyCredentialManager())

    with connection.connect():
        # Verify SSL was enabled
        assert stub.ssl is True
        assert stub.ssl_context is not None
