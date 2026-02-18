"""IMAP incremental sync engine.

This module implements the sync strategies described in
``docs/phase-1/imap-connector-production-plan/06-incremental-sync-strategy.md``.
It provides UID-based and MODSEQ-based incremental synchronization with
automatic fallback and UIDVALIDITY change detection.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

try:
    from imapclient import IMAPClient  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "imapclient must be installed to use the IMAP sync engine"
    ) from exc

from futurnal.privacy.audit import AuditEvent, AuditLogger
from futurnal.privacy.consent import ConsentRequiredError

from .connection_manager import ImapConnectionPool
from .descriptor import ImapMailboxDescriptor
from .sync_state import ImapSyncState, ImapSyncStateStore, SyncResult

# Import privacy components (optional to maintain backward compatibility)
try:
    from .consent_manager import ImapConsentManager, ImapConsentScopes
    from .audit_events import log_email_sync_event, log_consent_check_failed
    PRIVACY_COMPONENTS_AVAILABLE = True
except ImportError:
    PRIVACY_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImapSyncEngine:
    """Manages incremental IMAP synchronization."""

    def __init__(
        self,
        *,
        connection_pool: ImapConnectionPool,
        state_store: ImapSyncStateStore,
        audit_logger: Optional[AuditLogger] = None,
        consent_manager: Optional[object] = None,  # ImapConsentManager (optional)
    ):
        """Initialize sync engine.

        Args:
            connection_pool: Connection pool for IMAP operations
            state_store: State store for sync persistence
            audit_logger: Optional audit logger for sync events
            consent_manager: Optional consent manager for enforcement (ImapConsentManager)
        """
        self.connection_pool = connection_pool
        self.state_store = state_store
        self.audit_logger = audit_logger
        self.consent_manager = consent_manager
        self.descriptor = connection_pool.descriptor

    async def sync_folder(self, folder: str) -> SyncResult:
        """Perform incremental sync of a folder.

        Args:
            folder: Folder name to sync

        Returns:
            Sync result with changes detected

        Raises:
            ConsentRequiredError: If required consent not granted
        """
        start_time = time.time()
        result = SyncResult()

        # Consent enforcement: Check required consents before sync
        if self.consent_manager and PRIVACY_COMPONENTS_AVAILABLE:
            try:
                # Require mailbox access consent
                self.consent_manager.require_consent(
                    mailbox_id=self.descriptor.id,
                    scope=ImapConsentScopes.MAILBOX_ACCESS.value,
                )
            except ConsentRequiredError as e:
                # Log consent check failure
                if self.audit_logger and PRIVACY_COMPONENTS_AVAILABLE:
                    log_consent_check_failed(
                        self.audit_logger,
                        mailbox_id=self.descriptor.id,
                        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
                        operation="folder_sync",
                    )
                raise

        try:
            # Load sync state
            state = self._load_sync_state(self.descriptor.id, folder)

            # Acquire connection
            async with self.connection_pool.acquire() as connection:
                with connection.connect() as client:
                    # Select folder
                    select_info = client.select_folder(folder)

                    # Validate UIDVALIDITY
                    current_uidvalidity = select_info[b"UIDVALIDITY"]
                    if state.uidvalidity != current_uidvalidity:
                        # UIDVALIDITY changed - full resync required
                        logger.warning(
                            f"UIDVALIDITY changed for {folder}, performing full resync",
                            extra={
                                "mailbox_id": self.descriptor.id,
                                "folder": folder,
                                "old_uidvalidity": state.uidvalidity,
                                "new_uidvalidity": current_uidvalidity,
                            },
                        )
                        result = await self._full_resync(
                            client, folder, current_uidvalidity
                        )
                        # _full_resync persists a fresh state snapshot; reload it to avoid
                        # writing stale pre-resync fields back to storage.
                        state = self._load_sync_state(self.descriptor.id, folder)
                    else:
                        # Detect server capabilities if not already known
                        if not state.supports_idle:
                            state.supports_idle = b"IDLE" in client.capabilities()
                        if not state.supports_modseq:
                            state.supports_modseq = b"CONDSTORE" in client.capabilities()
                        if not state.supports_qresync:
                            state.supports_qresync = b"QRESYNC" in client.capabilities()

                        # Perform incremental sync
                        if state.supports_modseq and state.highest_modseq:
                            result = await self._sync_with_modseq(
                                client, state, select_info
                            )
                        else:
                            result = await self._sync_with_uid(
                                client, state, select_info
                            )

            # Update sync state
            state.last_sync_time = datetime.utcnow()
            state.total_syncs += 1
            state.messages_synced += len(result.new_messages)
            state.messages_updated += len(result.updated_messages)
            state.messages_deleted += len(result.deleted_messages)
            self._save_sync_state(state)

            result.sync_duration_seconds = time.time() - start_time

            # Log sync event
            self._log_sync_event(folder, result)

            return result

        except Exception as exc:
            result.errors.append(str(exc))
            result.sync_duration_seconds = time.time() - start_time
            logger.error(
                f"Sync failed for {folder}",
                exc_info=exc,
                extra={"mailbox_id": self.descriptor.id, "folder": folder},
            )
            # Update error count
            try:
                state = self._load_sync_state(self.descriptor.id, folder)
                state.sync_errors += 1
                self._save_sync_state(state)
            except Exception:  # noqa: BLE001
                pass
            raise

    def _load_sync_state(self, mailbox_id: str, folder: str) -> ImapSyncState:
        """Load or initialize sync state.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name

        Returns:
            Sync state for the folder
        """
        state = self.state_store.fetch(mailbox_id, folder)
        if state:
            return state

        # Initialize new state
        return ImapSyncState(
            mailbox_id=mailbox_id,
            folder=folder,
            uidvalidity=0,  # Will be set on first sync
            last_sync_time=datetime.utcnow(),
        )

    def _save_sync_state(self, state: ImapSyncState) -> None:
        """Save sync state.

        Args:
            state: Sync state to persist
        """
        self.state_store.upsert(state)

    async def _sync_with_uid(
        self,
        client: IMAPClient,
        state: ImapSyncState,
        select_info: Dict[bytes, Any],
    ) -> SyncResult:
        """Sync using UID-based detection.

        Args:
            client: IMAP client
            state: Current sync state
            select_info: Folder select response

        Returns:
            Sync result
        """
        result = SyncResult()
        current_exists = select_info[b"EXISTS"]

        # Detect new messages
        if state.last_synced_uid > 0:
            # Search for messages newer than last synced UID
            search_criteria = [f"{state.last_synced_uid + 1}:*"]
            try:
                new_uids = client.search(search_criteria)
                result.new_messages = list(new_uids) if new_uids else []

                # Update highest UID
                if result.new_messages:
                    state.last_synced_uid = max(result.new_messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"UID search failed: {exc}")
                result.errors.append(f"UID search failed: {exc}")
        else:
            # First sync - get all messages
            try:
                all_uids = client.search(["ALL"])
                result.new_messages = list(all_uids) if all_uids else []
                if result.new_messages:
                    state.last_synced_uid = max(result.new_messages)
                    state.uidvalidity = select_info[b"UIDVALIDITY"]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Initial sync search failed: {exc}")
                result.errors.append(f"Initial sync failed: {exc}")

        # Detect deletions (EXISTS count decreased)
        if current_exists < state.last_exists_count:
            # Messages were deleted - find which UIDs are missing
            try:
                all_current_uids = set(client.search(["ALL"]) or [])
                all_known_uids = set(range(1, state.last_synced_uid + 1))
                deleted_uids = all_known_uids - all_current_uids
                result.deleted_messages = list(deleted_uids)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Deletion detection failed: {exc}")
                result.errors.append(f"Deletion detection failed: {exc}")

        # Update state
        state.last_exists_count = current_exists
        state.message_count = current_exists

        return result

    async def _sync_with_modseq(
        self,
        client: IMAPClient,
        state: ImapSyncState,
        select_info: Dict[bytes, Any],
    ) -> SyncResult:
        """Sync using MODSEQ for efficient delta detection.

        Args:
            client: IMAP client
            state: Current sync state
            select_info: Folder select response

        Returns:
            Sync result
        """
        result = SyncResult()

        try:
            # Search for messages changed since last MODSEQ
            search_criteria = ["MODSEQ", str(state.highest_modseq)]
            if state.last_synced_uid > 0:
                search_criteria.append(f"{state.last_synced_uid + 1}:*")

            changed_uids = client.search(search_criteria)

            if changed_uids:
                # Fetch MODSEQ values to determine if new or updated
                fetch_data = client.fetch(changed_uids, ["MODSEQ", "FLAGS"])

                for uid, data in fetch_data.items():
                    if uid > state.last_synced_uid:
                        result.new_messages.append(uid)
                    else:
                        result.updated_messages.append(uid)

                # Update state
                if changed_uids:
                    max_uid = max(changed_uids)
                    state.last_synced_uid = max(state.last_synced_uid, max_uid)

                # Update highest MODSEQ
                new_modseq = select_info.get(b"HIGHESTMODSEQ")
                if new_modseq:
                    state.highest_modseq = new_modseq

        except Exception as exc:
            logger.error(f"MODSEQ sync failed, falling back to UID: {exc}")
            # Fallback to UID-based sync
            return await self._sync_with_uid(client, state, select_info)

        # Detect deletions
        current_exists = select_info[b"EXISTS"]
        if current_exists < state.last_exists_count:
            try:
                all_current_uids = set(client.search(["ALL"]) or [])
                all_known_uids = set(range(1, state.last_synced_uid + 1))
                deleted_uids = all_known_uids - all_current_uids
                result.deleted_messages = list(deleted_uids)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Deletion detection failed: {exc}")
                result.errors.append(f"Deletion detection failed: {exc}")

        # Update state
        state.last_exists_count = current_exists
        state.message_count = current_exists

        return result

    async def _full_resync(
        self,
        client: IMAPClient,
        folder: str,
        new_uidvalidity: int,
    ) -> SyncResult:
        """Perform full resync when UIDVALIDITY changes.

        Args:
            client: IMAP client
            folder: Folder name
            new_uidvalidity: New UIDVALIDITY value

        Returns:
            Sync result
        """
        logger.info(f"Full resync required for {folder}")

        # Reset state with capability detection
        state = ImapSyncState(
            mailbox_id=self.descriptor.id,
            folder=folder,
            uidvalidity=new_uidvalidity,
            last_sync_time=datetime.utcnow(),
            supports_idle=b"IDLE" in client.capabilities(),
            supports_modseq=b"CONDSTORE" in client.capabilities(),
            supports_qresync=b"QRESYNC" in client.capabilities(),
        )

        # Get all messages
        all_uids = client.search(["ALL"])
        result = SyncResult(new_messages=list(all_uids) if all_uids else [])

        if result.new_messages:
            state.last_synced_uid = max(result.new_messages)
            state.message_count = len(result.new_messages)
            state.last_exists_count = state.message_count

        self._save_sync_state(state)
        return result

    def _log_sync_event(self, folder: str, result: SyncResult) -> None:
        """Log sync event without exposing content.

        Uses enhanced logging helper if available.

        Args:
            folder: Folder name
            result: Sync result
        """
        if not self.audit_logger:
            return

        # Use enhanced logging if available
        if PRIVACY_COMPONENTS_AVAILABLE:
            try:
                log_email_sync_event(
                    self.audit_logger,
                    mailbox_id=self.descriptor.id,
                    folder=folder,
                    new_messages=len(result.new_messages),
                    updated_messages=len(result.updated_messages),
                    deleted_messages=len(result.deleted_messages),
                    sync_duration_seconds=result.sync_duration_seconds,
                    errors=len(result.errors),
                )
            except Exception:  # pragma: no cover - audit failures should not break sync
                pass
        else:
            # Fallback to basic logging
            event = AuditEvent(
                job_id=f"imap_sync_{self.descriptor.id}_{folder}_{int(time.time())}",
                source="imap_sync_engine",
                action="folder_sync",
                status="success" if not result.errors else "partial_success",
                timestamp=datetime.utcnow(),
                metadata={
                    "mailbox_id": self.descriptor.id,
                    "folder": folder,
                    "new_messages": len(result.new_messages),
                    "updated_messages": len(result.updated_messages),
                    "deleted_messages": len(result.deleted_messages),
                    "total_changes": result.total_changes,
                    "duration_seconds": round(result.sync_duration_seconds, 2),
                    "errors": len(result.errors),
                },
            )

            try:
                self.audit_logger.record(event)
            except Exception:  # pragma: no cover - audit failures should not break sync
                pass


__all__ = ["ImapSyncEngine"]
