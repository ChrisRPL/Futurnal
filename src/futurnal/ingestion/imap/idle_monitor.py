"""IMAP IDLE monitoring for real-time sync.

This module implements the IDLE monitoring described in
``docs/phase-1/imap-connector-production-plan/06-incremental-sync-strategy.md``.
It provides push-based notification of mailbox changes with automatic 10-minute
renewal.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, List, Optional

try:
    from imapclient import IMAPClient  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "imapclient must be installed to use IDLE monitoring"
    ) from exc

from .connection_manager import ImapConnection, RetryStrategy
from .sync_state import SyncResult


logger = logging.getLogger(__name__)


class IdleMonitor:
    """Monitor folder for real-time changes using IMAP IDLE."""

    def __init__(
        self,
        *,
        connection: ImapConnection,
        folder: str,
        callback: Callable[[SyncResult], Awaitable[None]],
        renewal_interval: int = 600,  # 10 minutes
        retry_strategy: Optional[RetryStrategy] = None,
    ):
        """Initialize IDLE monitor.

        Args:
            connection: IMAP connection to use
            folder: Folder to monitor
            callback: Async callback when changes detected
            renewal_interval: IDLE renewal interval in seconds (default 10 min)
            retry_strategy: Optional retry strategy for connection failures
        """
        self.connection = connection
        self.folder = folder
        self.callback = callback
        self.renewal_interval = renewal_interval
        self.retry_strategy = retry_strategy or RetryStrategy(
            max_retries=3, base_delay=2.0, max_delay=30.0
        )
        self._stop_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start IDLE monitoring."""
        if self._monitor_task and not self._monitor_task.done():
            raise RuntimeError("IDLE monitor already running")

        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._monitor_task = loop.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop IDLE monitoring."""
        self._stop_event.set()
        if self._monitor_task:
            await self._monitor_task

    async def _monitor_loop(self) -> None:
        """Main IDLE monitoring loop."""
        retry_count = 0

        while not self._stop_event.is_set():
            try:
                with self.connection.connect() as client:
                    client.select_folder(self.folder)

                    while not self._stop_event.is_set():
                        # Start IDLE
                        client.idle()
                        logger.debug(f"IDLE started for {self.folder}")

                        # Wait for changes or timeout (renewal interval)
                        responses = client.idle_check(timeout=self.renewal_interval)

                        # Done IDLE (renew connection)
                        client.idle_done()

                        # Record idle renewal metric
                        self.connection.metrics.record_idle_renewal()

                        # Process changes
                        if responses:
                            has_changes = self._detect_changes(responses)
                            if has_changes:
                                logger.info(
                                    f"Changes detected in {self.folder}, triggering sync"
                                )
                                # Trigger sync callback with empty result
                                # The callback will use sync engine to get actual changes
                                await self.callback(SyncResult())

                        # Check if stop requested
                        if self._stop_event.is_set():
                            break

                # Reset retry count on successful loop
                retry_count = 0

            except Exception as exc:  # noqa: BLE001
                if not self.retry_strategy.should_retry(retry_count, exc):
                    self.connection._log_event(
                        "idle", "failed", metadata={"error": str(exc)}
                    )
                    logger.error(f"IDLE monitoring failed for {self.folder}: {exc}")
                    raise

                delay = self.retry_strategy.calculate_delay(retry_count)
                retry_count += 1
                self.connection.metrics.record_retry()
                self.connection._log_event(
                    "idle",
                    "retrying",
                    metadata={"retry": retry_count, "delay_seconds": round(delay, 2)},
                )
                logger.warning(
                    f"IDLE error, retrying in {delay:.1f}s (attempt {retry_count}): {exc}"
                )
                await asyncio.sleep(delay)

    def _detect_changes(self, responses: List[Any]) -> bool:
        """Detect if IDLE responses indicate changes.

        Args:
            responses: IDLE response list from server

        Returns:
            True if changes detected, False otherwise
        """
        for response in responses:
            if not isinstance(response, (tuple, list)) or len(response) < 1:
                continue

            response_text = response[0]
            if isinstance(response_text, bytes):
                response_text = response_text.decode("utf-8", errors="replace")

            # Check for EXISTS (new message)
            if "EXISTS" in response_text:
                return True

            # Check for EXPUNGE (deleted message)
            if "EXPUNGE" in response_text:
                return True

            # Check for FETCH (flag changes)
            if "FETCH" in response_text:
                return True

        return False


__all__ = ["IdleMonitor"]
