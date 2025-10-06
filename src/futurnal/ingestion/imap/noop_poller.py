"""NOOP-based polling for IMAP servers without IDLE support.

This module implements the NOOP polling fallback described in
``docs/phase-1/imap-connector-production-plan/06-incremental-sync-strategy.md``.
It provides periodic synchronization for servers that don't support IDLE.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .sync_engine import ImapSyncEngine
from .descriptor import ImapMailboxDescriptor


logger = logging.getLogger(__name__)


class NoopPoller:
    """Poll folder using NOOP when IDLE not available."""

    def __init__(
        self,
        *,
        sync_engine: ImapSyncEngine,
        mailbox_descriptor: ImapMailboxDescriptor,
        folder: str,
        poll_interval: int = 300,  # 5 minutes
    ):
        """Initialize NOOP poller.

        Args:
            sync_engine: Sync engine for performing syncs
            mailbox_descriptor: Mailbox descriptor
            folder: Folder to poll
            poll_interval: Polling interval in seconds (default 5 min)
        """
        self.sync_engine = sync_engine
        self.mailbox_descriptor = mailbox_descriptor
        self.folder = folder
        self.poll_interval = poll_interval
        self._stop_event = asyncio.Event()
        self._poll_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start NOOP polling."""
        if self._poll_task and not self._poll_task.done():
            raise RuntimeError("NOOP poller already running")

        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._poll_task = loop.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop NOOP polling."""
        self._stop_event.set()
        if self._poll_task:
            await self._poll_task

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                # Perform sync
                result = await self.sync_engine.sync_folder(self.folder)

                if result.has_changes:
                    logger.info(
                        f"NOOP poll found changes in {self.folder}: "
                        f"{len(result.new_messages)} new, "
                        f"{len(result.updated_messages)} updated, "
                        f"{len(result.deleted_messages)} deleted"
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"NOOP polling error for {self.folder}: {exc}",
                    exc_info=exc,
                )

            # Wait for next poll interval or stop signal
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval
                )
            except asyncio.TimeoutError:
                # Timeout is expected - continue polling
                pass


__all__ = ["NoopPoller"]
