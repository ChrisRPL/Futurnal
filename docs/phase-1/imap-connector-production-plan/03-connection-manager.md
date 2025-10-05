Summary: Implement robust IMAP connection management with TLS enforcement, retry logic, and offline resilience.

# 03 · Connection Manager

## Purpose
Provide reliable, secure IMAP connection lifecycle management with TLS enforcement, exponential backoff retry (modOpt-inspired), connection pooling, and offline resilience. Ensures the Ghost maintains stable connectivity under varying network conditions while respecting privacy and security requirements.

## Scope
- IMAPClient wrapper with connection lifecycle management
- TLS/SSL enforcement (reject plaintext IMAP)
- Connection pooling for multi-folder sync
- Exponential backoff retry with jitter (modOpt-inspired)
- Connection health checks and automatic reconnection
- IDLE support with 10-minute renewal (per IMAPClient best practices)
- Offline detection and queue management
- Connection metrics and telemetry

## Requirements Alignment
- **TLS-only**: All IMAP connections must use TLS (port 993 or STARTTLS)
- **Privacy-first**: Connection errors logged without credential exposure
- **Resilience**: Graceful degradation during network issues
- **Efficiency**: Connection pooling to reduce authentication overhead
- **Observable**: Connection metrics for health monitoring

## Component Design

### ImapConnectionManager
```python
class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IDLE = "idle"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

class ImapConnection:
    """Represents a single IMAP connection."""

    def __init__(
        self,
        *,
        mailbox_descriptor: ImapMailboxDescriptor,
        credential_manager: CredentialManager,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.descriptor = mailbox_descriptor
        self.credential_manager = credential_manager
        self.audit_logger = audit_logger
        self.client: Optional[IMAPClient] = None
        self.state = ConnectionState.DISCONNECTED
        self.last_activity: Optional[datetime] = None
        self.connection_id = str(uuid.uuid4())
        self.retry_count = 0
        self.max_retries = 5

    @contextmanager
    def connect(self) -> Iterator[IMAPClient]:
        """Connect to IMAP server with automatic cleanup."""
        try:
            self._establish_connection()
            yield self.client
        finally:
            self._cleanup_connection()

    def _establish_connection(self) -> None:
        """Establish IMAP connection with TLS and authentication."""
        # Create SSL context
        ssl_context = self._create_ssl_context()

        # Connect to server
        self.state = ConnectionState.CONNECTING
        self.client = IMAPClient(
            host=self.descriptor.imap_host,
            port=self.descriptor.imap_port,
            ssl=True,
            ssl_context=ssl_context,
            timeout=30,
        )

        # Authenticate
        self._authenticate()

        self.state = ConnectionState.CONNECTED
        self.last_activity = datetime.utcnow()
        self.retry_count = 0

        # Log successful connection (without credentials)
        self._log_connection_event("connected", "success")

    def _authenticate(self) -> None:
        """Authenticate with IMAP server using stored credentials."""
        with secure_credential_context(
            self.credential_manager,
            self.descriptor.credential_id
        ) as credentials:
            if isinstance(credentials, OAuth2Tokens):
                # OAuth2 authentication (XOAUTH2)
                self.client.oauth2_login(
                    self.descriptor.email_address,
                    credentials.access_token,
                )
            elif isinstance(credentials, AppPassword):
                # Standard login
                self.client.login(
                    self.descriptor.email_address,
                    credentials.password,
                )

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create secure SSL context with modern settings."""
        context = ssl.create_default_context()
        # Use certifi for up-to-date CA certificates
        import certifi
        context.load_verify_locations(certifi.where())
        # Enforce TLS 1.2+
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context

    def _cleanup_connection(self) -> None:
        """Clean up connection resources."""
        if self.client:
            try:
                self.client.logout()
            except Exception as e:
                logger.warning("Error during logout", extra={"error": str(e)})
            finally:
                self.client = None
                self.state = ConnectionState.DISCONNECTED

    def is_alive(self) -> bool:
        """Check if connection is still alive."""
        if not self.client or self.state != ConnectionState.CONNECTED:
            return False

        try:
            # NOOP command to check connection
            self.client.noop()
            self.last_activity = datetime.utcnow()
            return True
        except Exception:
            return False

    def _log_connection_event(self, action: str, status: str, metadata: Optional[Dict] = None):
        """Log connection event without exposing credentials."""
        if not self.audit_logger:
            return

        event_metadata = {
            "connection_id": self.connection_id,
            "mailbox_id": self.descriptor.id,
            "host": self.descriptor.imap_host,
            "port": self.descriptor.imap_port,
            "auth_mode": self.descriptor.auth_mode.value,
            "retry_count": self.retry_count,
        }
        if metadata:
            event_metadata.update(metadata)

        self.audit_logger.record(AuditEvent(
            job_id=f"imap_connection_{self.connection_id}",
            source="imap_connection_manager",
            action=f"connection_{action}",
            status=status,
            timestamp=datetime.utcnow(),
            metadata=event_metadata,
        ))
```

### ImapConnectionPool
```python
class ImapConnectionPool:
    """Manages a pool of IMAP connections for efficient reuse."""

    def __init__(
        self,
        *,
        mailbox_descriptor: ImapMailboxDescriptor,
        credential_manager: CredentialManager,
        max_connections: int = 3,
        connection_timeout: int = 300,  # 5 minutes
    ):
        self.descriptor = mailbox_descriptor
        self.credential_manager = credential_manager
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._pool: List[ImapConnection] = []
        self._pool_lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[ImapConnection]:
        """Acquire connection from pool."""
        async with self._pool_lock:
            # Try to reuse existing connection
            connection = self._get_idle_connection()

            if not connection:
                # Create new connection if pool not full
                if len(self._pool) < self.max_connections:
                    connection = ImapConnection(
                        mailbox_descriptor=self.descriptor,
                        credential_manager=self.credential_manager,
                    )
                    self._pool.append(connection)
                else:
                    # Wait for connection to become available
                    connection = await self._wait_for_connection()

        try:
            yield connection
        finally:
            # Return connection to pool
            connection.last_activity = datetime.utcnow()

    def _get_idle_connection(self) -> Optional[ImapConnection]:
        """Get an idle connection from the pool."""
        for conn in self._pool:
            if conn.state in [ConnectionState.CONNECTED, ConnectionState.IDLE]:
                if conn.is_alive():
                    return conn
                else:
                    # Connection dead, remove from pool
                    self._pool.remove(conn)
        return None

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._pool_lock:
            for conn in self._pool:
                conn._cleanup_connection()
            self._pool.clear()
```

### Retry Logic (modOpt-inspired)
```python
class RetryStrategy:
    """Exponential backoff retry strategy with jitter."""

    def __init__(
        self,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate delay for given retry attempt."""
        delay = min(
            self.base_delay * (self.exponential_base ** retry_count),
            self.max_delay
        )

        if self.jitter:
            # Add random jitter (0-50% of delay)
            delay *= (1 + random.random() * 0.5)

        return delay

    def should_retry(self, retry_count: int, exception: Exception) -> bool:
        """Determine if operation should be retried."""
        if retry_count >= self.max_retries:
            return False

        # Retry on network errors
        if isinstance(exception, (socket.error, OSError, IMAPClient.Error)):
            return True

        # Don't retry on authentication errors
        if isinstance(exception, IMAPClient.AuthenticationError):
            return False

        return False


@retry_with_backoff
async def connect_with_retry(
    mailbox_descriptor: ImapMailboxDescriptor,
    credential_manager: CredentialManager,
    retry_strategy: Optional[RetryStrategy] = None,
) -> ImapConnection:
    """Connect to IMAP server with retry logic."""
    strategy = retry_strategy or RetryStrategy()
    retry_count = 0

    while True:
        try:
            connection = ImapConnection(
                mailbox_descriptor=mailbox_descriptor,
                credential_manager=credential_manager,
            )
            connection._establish_connection()
            return connection

        except Exception as e:
            if not strategy.should_retry(retry_count, e):
                raise

            delay = strategy.calculate_delay(retry_count)
            logger.warning(
                f"Connection failed, retrying in {delay:.1f}s",
                extra={
                    "retry_count": retry_count,
                    "max_retries": strategy.max_retries,
                    "error": str(e),
                }
            )
            await asyncio.sleep(delay)
            retry_count += 1
```

### IDLE Support
```python
class IdleConnection:
    """Manages IMAP IDLE connections with automatic renewal."""

    def __init__(
        self,
        connection: ImapConnection,
        renewal_interval: int = 600,  # 10 minutes
    ):
        self.connection = connection
        self.renewal_interval = renewal_interval
        self._idle_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start_idle(
        self,
        folder: str,
        callback: Callable[[List[int]], Awaitable[None]],
    ) -> None:
        """Start IDLE monitoring for new messages."""
        self.connection.client.select_folder(folder)

        while not self._stop_event.is_set():
            try:
                # Start IDLE
                self.connection.client.idle()
                self.connection.state = ConnectionState.IDLE

                # Wait for events or timeout
                responses = self.connection.client.idle_check(
                    timeout=self.renewal_interval
                )

                # Process new messages
                if responses:
                    new_message_uids = self._extract_new_messages(responses)
                    if new_message_uids:
                        await callback(new_message_uids)

                # Done IDLE (renew connection)
                self.connection.client.idle_done()
                self.connection.state = ConnectionState.CONNECTED

            except Exception as e:
                logger.error(f"IDLE error: {e}")
                # Reconnect and continue
                await self._reconnect()

    async def stop_idle(self) -> None:
        """Stop IDLE monitoring."""
        self._stop_event.set()
        if self._idle_task:
            self._idle_task.cancel()

    def _extract_new_messages(self, responses: List) -> List[int]:
        """Extract new message UIDs from IDLE responses."""
        # Parse IMAP IDLE responses like: [(b'1 EXISTS', b'OK')]
        new_uids = []
        for response in responses:
            if b'EXISTS' in response[0]:
                # New message arrived
                # Need to SEARCH for new UIDs
                pass
        return new_uids
```

### Offline Detection
```python
class NetworkMonitor:
    """Monitor network connectivity and queue operations when offline."""

    def __init__(self):
        self._online = True
        self._pending_operations: List[Callable] = []

    def is_online(self) -> bool:
        """Check if network is available."""
        try:
            # Try to resolve a known hostname
            socket.gethostbyname("www.google.com")
            self._online = True
            return True
        except socket.error:
            self._online = False
            return False

    async def queue_operation(self, operation: Callable) -> None:
        """Queue operation for later execution when online."""
        if self.is_online():
            await operation()
        else:
            self._pending_operations.append(operation)
            logger.info("Operation queued for later (offline)")

    async def process_pending(self) -> None:
        """Process pending operations when back online."""
        if not self.is_online():
            return

        while self._pending_operations:
            operation = self._pending_operations.pop(0)
            try:
                await operation()
            except Exception as e:
                logger.error(f"Failed to process pending operation: {e}")
```

## Acceptance Criteria

- ✅ All IMAP connections use TLS (port 993 or STARTTLS)
- ✅ Connection established successfully with OAuth2 credentials
- ✅ Connection established successfully with app password
- ✅ Connection retries with exponential backoff on network errors
- ✅ No retries on authentication errors (fail fast)
- ✅ Connection pool reuses idle connections efficiently
- ✅ Dead connections detected and removed from pool
- ✅ IDLE monitoring works with 10-minute renewal
- ✅ Offline detection queues operations for later
- ✅ Connection metrics logged (success rate, latency, retries)
- ✅ Credentials never logged during connection errors
- ✅ SSL certificate validation enforced
- ✅ Connection health checks via NOOP command

## Test Plan

### Unit Tests
- SSL context creation with certifi
- Retry strategy delay calculation
- Exponential backoff with jitter
- Should-retry decision logic
- Connection state transitions
- Offline detection logic

### Integration Tests
- End-to-end connection with mock IMAP server
- OAuth2 authentication flow
- App password authentication flow
- Connection retry on network failures
- Connection pool acquire/release
- IDLE monitoring with mock responses
- Connection cleanup on errors

### Resilience Tests
- Network interruption during IDLE
- Server timeout during operation
- SSL certificate validation failures
- Authentication token expiration during sync
- Connection pool exhaustion
- Offline queue processing

### Security Tests
- TLS enforcement (reject port 143 without STARTTLS)
- SSL certificate validation
- No credentials in exception messages
- Secure credential cleanup after use

## Implementation Notes

### Status Update · October 2025

Implementation landed in `src/futurnal/ingestion/imap/connection_manager.py` with
complementary tests in `tests/ingestion/imap/test_connection_manager.py`.
Key coverage:

- TLS-only enforcement with certifi-backed trust store
- Retry strategy with exponential backoff and authentication safeguards
- Connection pooling and lifecycle metrics
- IMAP IDLE renewal with automatic reconnection
- Offline-aware operation queue via `NetworkMonitor`

All unit tests pass: `pytest tests/ingestion/imap/test_connection_manager.py`.

### IMAPClient Configuration
```python
client = IMAPClient(
    host=descriptor.imap_host,
    port=descriptor.imap_port,
    ssl=True,
    ssl_context=ssl_context,
    timeout=30,
    use_uid=True,  # Always use UIDs for stability
)
```

### OAuth2 Authentication (XOAUTH2)
```python
def oauth2_login(client: IMAPClient, email: str, access_token: str) -> None:
    """Authenticate using OAuth2 (XOAUTH2 SASL)."""
    # IMAPClient has built-in oauth2_login method
    client.oauth2_login(email, access_token)
```

### Connection Metrics
```python
class ConnectionMetrics:
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_retries: int = 0
    average_connection_time: float = 0.0
    idle_renewals: int = 0
```

## Open Questions

- Should we support connection multiplexing (multiple folders on one connection)?
- How to handle server-side connection limits (e.g., Gmail max 15 concurrent connections)?
- Should we implement connection warming (pre-connect before needed)?
- How to handle IMAP proxy configurations?
- Should we support SOCKS proxy for additional privacy?
- How to handle server maintenance windows (graceful degradation)?

## Dependencies
- IMAPClient library (`pip install imapclient`)
- Python ssl module (standard library)
- certifi for CA certificates (`pip install certifi`)
- CredentialManager from task 02
- AuditLogger for connection event logging


