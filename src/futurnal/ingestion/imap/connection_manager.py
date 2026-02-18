"""IMAP connection lifecycle management for the Futurnal IMAP connector.

This module implements the connection management responsibilities described in
``docs/phase-1/imap-connector-production-plan/03-connection-manager.md``. It
provides a privacy-first, TLS-enforced connection layer that supports
resilient retries, connection pooling, IMAP IDLE renewals, offline queuing,
and structured audit logging hooks. The implementation is designed to operate
entirely on-device alongside the credential manager to honour Futurnal's
privacy guarantees.
"""

from __future__ import annotations

import asyncio
import logging
import random
import socket
import ssl
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Dict, Iterator, List, Optional

try:
    from imapclient import IMAPClient  # type: ignore
except Exception as exc:  # pragma: no cover - missing optional dependency
    raise RuntimeError(
        "imapclient must be installed to use the IMAP connection manager"
    ) from exc

try:
    from imapclient.exceptions import LoginError as IMAPLoginError  # type: ignore
except Exception:  # pragma: no cover - optional across imapclient versions
    IMAPLoginError = None

import certifi

from futurnal.privacy.audit import AuditEvent, AuditLogger

from .credential_manager import (
    AppPassword,
    CredentialManager,
    OAuth2Tokens,
    secure_credential_context,
)
from .descriptor import ImapMailboxDescriptor


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection state and metrics
# ---------------------------------------------------------------------------


class ConnectionState(str, Enum):
    """Lifecycle states for an IMAP connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IDLE = "idle"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class ConnectionMetrics:
    """Aggregated metrics for connection health reporting."""

    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_retries: int = 0
    average_connection_time: float = 0.0
    idle_renewals: int = 0

    def record_attempt(self, success: bool, elapsed: float) -> None:
        self.total_connections += 1
        if success:
            self.successful_connections += 1
            # incremental average to avoid large arrays
            self.average_connection_time += (
                elapsed - self.average_connection_time
            ) / max(1, self.successful_connections)
        else:
            self.failed_connections += 1

    def record_retry(self) -> None:
        self.total_retries += 1

    def record_idle_renewal(self) -> None:
        self.idle_renewals += 1


# ---------------------------------------------------------------------------
# Retry strategy
# ---------------------------------------------------------------------------


@dataclass
class RetryStrategy:
    """Exponential backoff retry configuration with jitter."""

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def calculate_delay(self, retry_count: int) -> float:
        delay = min(self.base_delay * (self.exponential_base**retry_count), self.max_delay)
        if self.jitter:
            delay *= 1 + random.random() * 0.5
        return delay

    def should_retry(self, retry_count: int, exc: Exception) -> bool:
        if retry_count >= self.max_retries:
            return False
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError, socket.timeout)):
            return True

        if isinstance(exc, IMAPClient.Error):
            if getattr(exc, "args", None):
                message = str(exc)
                # Authentication errors should not be retried automatically
                if "AUTHENTICATIONFAILED" in message.upper():
                    return False
        auth_error_type = getattr(IMAPClient, "AuthenticationError", None)
        if isinstance(auth_error_type, type) and isinstance(exc, auth_error_type):
            return False
        if IMAPLoginError is not None and isinstance(exc, IMAPLoginError):
            return False
        if isinstance(exc, (socket.error, OSError)):
            return True
        return isinstance(exc, IMAPClient.AbortError)


# ---------------------------------------------------------------------------
# Network monitor
# ---------------------------------------------------------------------------


class NetworkMonitor:
    """Detects offline mode and queues operations until connectivity resumes."""

    def __init__(self) -> None:
        self._online = True
        self._pending: Deque[Callable[[], Awaitable[Any]]] = deque()
        self._lock = asyncio.Lock()

    async def is_online(self) -> bool:
        """Check network connectivity via DNS resolve."""

        try:
            await asyncio.get_event_loop().run_in_executor(None, socket.gethostbyname, "www.google.com")
        except socket.error:
            self._online = False
            return False
        self._online = True
        return True

    async def queue_operation(self, operation: Callable[[], Awaitable[Any]]) -> None:
        async with self._lock:
            if await self.is_online():
                await operation()
            else:
                self._pending.append(operation)
                logger.info("Operation queued until network available")

    async def process_pending(self) -> None:
        if not await self.is_online():
            return
        while True:
            async with self._lock:
                if not self._pending:
                    break
                operation = self._pending.popleft()
            try:
                await operation()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed processing pending operation", exc_info=exc)


# ---------------------------------------------------------------------------
# Connection implementation
# ---------------------------------------------------------------------------


@dataclass
class ImapConnection:
    """Represents a single IMAP connection with TLS enforcement."""

    descriptor: ImapMailboxDescriptor
    credential_manager: CredentialManager
    retry_strategy: RetryStrategy = field(default_factory=RetryStrategy)
    audit_logger: Optional[AuditLogger] = None
    metrics: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    connection_timeout: int = 30

    client: Optional[IMAPClient] = field(default=None, init=False)
    state: ConnectionState = field(default=ConnectionState.DISCONNECTED, init=False)
    last_activity: Optional[datetime] = field(default=None, init=False)
    retry_count: int = field(default=0, init=False)

    @contextmanager
    def connect(self) -> Iterator[IMAPClient]:
        start = time.perf_counter()
        try:
            self._establish_connection()
            elapsed = time.perf_counter() - start
            self.metrics.record_attempt(True, elapsed)
            yield self.client  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            self.metrics.record_attempt(False, elapsed)
            self._log_event("connect", "failed", metadata={"error": str(exc)})
            self.state = ConnectionState.FAILED
            raise
        finally:
            self._cleanup_connection()

    def _establish_connection(self) -> None:
        if self.descriptor.imap_port == 143:
            raise ValueError("Plain IMAP (port 143) is unsupported; TLS required")
        self.state = ConnectionState.CONNECTING
        self._log_event("connect", "starting")
        ssl_context = self._create_ssl_context()
        self.client = IMAPClient(
            host=self.descriptor.imap_host,
            port=self.descriptor.imap_port,
            ssl=True,
            ssl_context=ssl_context,
            timeout=self.connection_timeout,
            use_uid=True,
        )
        self._authenticate()
        self.state = ConnectionState.CONNECTED
        self.last_activity = datetime.utcnow()
        self.retry_count = 0
        self._log_event("connect", "success")

    def _create_ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context()
        context.load_verify_locations(certifi.where())
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context

    def _authenticate(self) -> None:
        with secure_credential_context(self.credential_manager, self.descriptor.credential_id) as credentials:
            if isinstance(credentials, OAuth2Tokens):
                self.client.oauth2_login(self.descriptor.email_address, credentials.access_token)  # type: ignore[union-attr]
            elif isinstance(credentials, AppPassword):
                self.client.login(self.descriptor.email_address, credentials.password)  # type: ignore[union-attr]
            else:  # pragma: no cover - defensive
                raise ValueError("Unsupported credential type")

    def _cleanup_connection(self) -> None:
        if not self.client:
            return
        try:
            self.client.logout()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during logout", exc_info=exc)
        finally:
            self.client = None
            self.state = ConnectionState.DISCONNECTED

    def is_alive(self) -> bool:
        if not self.client or self.state not in {ConnectionState.CONNECTED, ConnectionState.IDLE}:
            return False
        try:
            self.client.noop()
            self.last_activity = datetime.utcnow()
            return True
        except Exception:  # noqa: BLE001
            return False

    def reconnect_with_retry(self) -> None:
        self.state = ConnectionState.RECONNECTING
        while True:
            try:
                self._establish_connection()
                return
            except Exception as exc:  # noqa: BLE001
                if not self.retry_strategy.should_retry(self.retry_count, exc):
                    self._log_event("reconnect", "failed", metadata={"error": str(exc)})
                    raise
                delay = self.retry_strategy.calculate_delay(self.retry_count)
                self.metrics.record_retry()
                self.retry_count += 1
                self._log_event(
                    "reconnect",
                    "retrying",
                    metadata={"retry": self.retry_count, "delay_seconds": round(delay, 2)},
                )
                time.sleep(delay)

    def _log_event(self, action: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.audit_logger:
            return
        metadata = metadata or {}
        event = AuditEvent(
            job_id=f"imap_connection_{self.descriptor.id}",
            source="imap_connection_manager",
            action=f"{action}",
            status=status,
            timestamp=datetime.utcnow(),
            metadata={
                "mailbox_id": self.descriptor.id,
                "host": self.descriptor.imap_host,
                "port": self.descriptor.imap_port,
                "auth_mode": self.descriptor.auth_mode.value,
                "retry_count": self.retry_count,
                **metadata,
            },
        )
        try:
            self.audit_logger.record(event)
        except Exception:  # pragma: no cover - audit failures should not break ingestion
            pass


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------


@dataclass
class ImapConnectionPool:
    """Manage a pool of reusable connections for a mailbox."""

    descriptor: ImapMailboxDescriptor
    credential_manager: CredentialManager
    max_connections: int = 3
    connection_timeout: int = 30
    retry_strategy: RetryStrategy = field(default_factory=RetryStrategy)
    audit_logger: Optional[AuditLogger] = None

    def __post_init__(self) -> None:
        self._pool: List[ImapConnection] = []
        self._pool_lock = asyncio.Lock()

    def _create_connection(self) -> ImapConnection:
        return ImapConnection(
            descriptor=self.descriptor,
            credential_manager=self.credential_manager,
            retry_strategy=self.retry_strategy,
            audit_logger=self.audit_logger,
            connection_timeout=self.connection_timeout,
        )

    async def close_all(self) -> None:
        async with self._pool_lock:
            for connection in self._pool:
                connection._cleanup_connection()
            self._pool.clear()

    def _prune_dead_connections(self) -> None:
        alive: List[ImapConnection] = []
        for connection in self._pool:
            if connection.is_alive():
                alive.append(connection)
            else:
                connection._cleanup_connection()
        self._pool = alive

    def _get_idle_connection(self) -> Optional[ImapConnection]:
        for connection in self._pool:
            if connection.state in {ConnectionState.CONNECTED, ConnectionState.IDLE}:
                if connection.is_alive():
                    return connection
        return None

    async def _wait_for_available(self) -> ImapConnection:
        while True:
            await asyncio.sleep(0.1)
            connection = self._get_idle_connection()
            if connection:
                return connection

    @asynccontextmanager
    async def acquire(self) -> Iterator[ImapConnection]:
        async with self._pool_lock:
            self._prune_dead_connections()
            connection = self._get_idle_connection()
            if connection is None:
                if len(self._pool) < self.max_connections:
                    connection = self._create_connection()
                    self._pool.append(connection)
                else:
                    connection = await self._wait_for_available()
        try:
            if connection.state == ConnectionState.DISCONNECTED:
                connection.reconnect_with_retry()
            yield connection
        finally:
            connection.last_activity = datetime.utcnow()


# ---------------------------------------------------------------------------
# IMAP IDLE manager
# ---------------------------------------------------------------------------


class IdleConnection:
    """Manage an IMAP IDLE loop with automatic renewal."""

    def __init__(
        self,
        connection: ImapConnection,
        renewal_interval: int = 600,
        retry_strategy: Optional[RetryStrategy] = None,
    ) -> None:
        self._connection = connection
        self._renewal_interval = renewal_interval
        self._retry_strategy = retry_strategy or RetryStrategy(max_retries=3, base_delay=2.0, max_delay=30.0)
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self, folder: str, callback: Callable[[List[int]], Awaitable[None]]) -> None:
        if self._task and not self._task.done():
            raise RuntimeError("Idle loop already running")
        self._stop.clear()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run(folder, callback))

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    async def _run(self, folder: str, callback: Callable[[List[int]], Awaitable[None]]) -> None:
        retry_count = 0
        while not self._stop.is_set():
            try:
                with self._connection.connect() as client:
                    client.select_folder(folder)
                    while not self._stop.is_set():
                        self._connection.state = ConnectionState.IDLE
                        client.idle()
                        responses = client.idle_check(timeout=self._renewal_interval)
                        client.idle_done()
                        self._connection.state = ConnectionState.CONNECTED
                        self._connection.metrics.record_idle_renewal()
                        new_message_uids = self._extract_new_messages(client, responses)
                        if new_message_uids:
                            await callback(new_message_uids)
                        if self._stop.is_set():
                            break
                retry_count = 0
            except Exception as exc:  # noqa: BLE001
                if not self._retry_strategy.should_retry(retry_count, exc):
                    self._connection._log_event("idle", "failed", metadata={"error": str(exc)})
                    raise
                delay = self._retry_strategy.calculate_delay(retry_count)
                retry_count += 1
                self._connection.metrics.record_retry()
                self._connection._log_event(
                    "idle",
                    "retrying",
                    metadata={"retry": retry_count, "delay_seconds": round(delay, 2)},
                )
                await asyncio.sleep(delay)

    def _extract_new_messages(self, client: IMAPClient, responses: List[Any]) -> List[int]:
        if not responses:
            return []
        # IDLE responses look like: [(b'3', b'EXISTS')]
        for response in responses:
            if isinstance(response, (tuple, list)) and len(response) >= 2:
                indicator = response[1]
                if isinstance(indicator, (bytes, str)) and b"EXISTS" in indicator if isinstance(indicator, bytes) else "EXISTS" in indicator:
                    status = client.search(["UNSEEN"])
                    return [int(uid) for uid in status]
        return []


__all__ = [
    "ConnectionMetrics",
    "ConnectionState",
    "IdleConnection",
    "ImapConnection",
    "ImapConnectionPool",
    "NetworkMonitor",
    "RetryStrategy",
]
