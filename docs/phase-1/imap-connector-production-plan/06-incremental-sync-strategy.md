Summary: Implement incremental IMAP sync with UID tracking, MODSEQ support, and IDLE/NOOP polling.

# 06 · Incremental Sync Strategy

## Purpose
Implement efficient incremental synchronization using IMAP UID tracking and MODSEQ (if supported), with IDLE/NOOP polling for real-time updates. Ensures the Ghost learns from new/updated/deleted emails within 5 minutes while minimizing network traffic and server load.

## Scope
- IMAP UID-based state tracking
- MODSEQ support for delta sync (RFC 7162)
- New message detection
- Updated message detection (flag changes)
- Deleted message detection
- IDLE support for push notifications
- NOOP fallback polling
- Sync state persistence via StateStore
- Folder sync orchestration
- Multi-mailbox sync coordination

## Requirements Alignment
- **5-minute detection window**: New/updated/deleted emails detected within 5 minutes
- **< 0.5% failure rate**: Reliable sync across diverse server implementations
- **Incremental learning**: Only process changed messages
- **Offline resilience**: Queue sync tasks when offline
- **Privacy-first**: State stored locally with audit trail

## Data Model

### ImapSyncState
```python
class ImapSyncState(BaseModel):
    """Persistent sync state for IMAP folder."""

    # Identity
    mailbox_id: str
    folder: str

    # IMAP state
    uidvalidity: int  # UIDVALIDITY value from server
    last_synced_uid: int = 0  # Highest UID seen
    highest_modseq: Optional[int] = None  # For MODSEQ support

    # Sync metadata
    last_sync_time: datetime
    message_count: int = 0
    last_exists_count: int = 0  # For detecting deletions

    # Sync statistics
    total_syncs: int = 0
    messages_synced: int = 0
    messages_updated: int = 0
    messages_deleted: int = 0
    sync_errors: int = 0

    # Server capabilities
    supports_idle: bool = False
    supports_modseq: bool = False
    supports_qresync: bool = False

    class Config:
        json_schema_extra = {
            "description": "Sync state for incremental IMAP synchronization"
        }


class SyncResult(BaseModel):
    """Result of a sync operation."""
    new_messages: List[int] = Field(default_factory=list)  # UIDs
    updated_messages: List[int] = Field(default_factory=list)
    deleted_messages: List[int] = Field(default_factory=list)
    sync_duration_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)
```

## Component Design

### ImapSyncEngine
```python
class ImapSyncEngine:
    """Manages incremental IMAP synchronization."""

    def __init__(
        self,
        *,
        connection_manager: ImapConnectionManager,
        state_store: StateStore,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.connection_manager = connection_manager
        self.state_store = state_store
        self.audit_logger = audit_logger

    async def sync_folder(
        self,
        mailbox_descriptor: ImapMailboxDescriptor,
        folder: str,
    ) -> SyncResult:
        """Perform incremental sync of a folder."""
        start_time = time.time()

        # Load sync state
        state = self._load_sync_state(mailbox_descriptor.id, folder)

        # Acquire connection
        async with self.connection_manager.acquire() as connection:
            with connection.connect() as client:
                # Select folder
                select_info = client.select_folder(folder)

                # Validate UIDVALIDITY
                current_uidvalidity = select_info[b'UIDVALIDITY']
                if state.uidvalidity != current_uidvalidity:
                    # UIDVALIDITY changed - full resync required
                    logger.warning("UIDVALIDITY changed, performing full resync")
                    return await self._full_resync(client, mailbox_descriptor, folder, current_uidvalidity)

                # Detect server capabilities
                if not state.supports_idle:
                    state.supports_idle = b'IDLE' in client.capabilities()
                if not state.supports_modseq:
                    state.supports_modseq = b'CONDSTORE' in client.capabilities()

                # Perform incremental sync
                if state.supports_modseq and state.highest_modseq:
                    result = await self._sync_with_modseq(client, state, select_info)
                else:
                    result = await self._sync_with_uid(client, state, select_info)

        # Update sync state
        state.last_sync_time = datetime.utcnow()
        state.total_syncs += 1
        state.messages_synced += len(result.new_messages)
        state.messages_updated += len(result.updated_messages)
        state.messages_deleted += len(result.deleted_messages)
        self._save_sync_state(state)

        result.sync_duration_seconds = time.time() - start_time

        # Log sync event
        self._log_sync_event(mailbox_descriptor.id, folder, result)

        return result

    def _load_sync_state(self, mailbox_id: str, folder: str) -> ImapSyncState:
        """Load or initialize sync state."""
        state_key = f"imap_sync:{mailbox_id}:{folder}"

        try:
            state_data = self.state_store.get(state_key)
            return ImapSyncState.model_validate_json(state_data)
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize new state
            return ImapSyncState(
                mailbox_id=mailbox_id,
                folder=folder,
                uidvalidity=0,
                last_sync_time=datetime.utcnow(),
            )

    def _save_sync_state(self, state: ImapSyncState) -> None:
        """Save sync state."""
        state_key = f"imap_sync:{state.mailbox_id}:{state.folder}"
        self.state_store.set(state_key, state.model_dump_json())

    async def _sync_with_uid(
        self,
        client: IMAPClient,
        state: ImapSyncState,
        select_info: Dict,
    ) -> SyncResult:
        """Sync using UID-based detection."""
        result = SyncResult()

        current_exists = select_info[b'EXISTS']

        # Detect new messages
        if state.last_synced_uid > 0:
            # Search for messages newer than last synced UID
            search_criteria = [f'{state.last_synced_uid + 1}:*']
            new_uids = client.search(search_criteria)
            result.new_messages = new_uids

            # Update highest UID
            if new_uids:
                state.last_synced_uid = max(new_uids)
        else:
            # First sync - get all messages
            all_uids = client.search(['ALL'])
            result.new_messages = all_uids
            if all_uids:
                state.last_synced_uid = max(all_uids)

        # Detect deletions (EXISTS count decreased)
        if current_exists < state.last_exists_count:
            # Messages were deleted
            # Need to check which UIDs are missing
            all_current_uids = set(client.search(['ALL']))
            all_known_uids = set(range(1, state.last_synced_uid + 1))
            deleted_uids = all_known_uids - all_current_uids
            result.deleted_messages = list(deleted_uids)

        state.last_exists_count = current_exists
        state.message_count = current_exists

        return result

    async def _sync_with_modseq(
        self,
        client: IMAPClient,
        state: ImapSyncState,
        select_info: Dict,
    ) -> SyncResult:
        """Sync using MODSEQ for efficient delta detection."""
        result = SyncResult()

        # Search for messages changed since last MODSEQ
        search_criteria = [
            'MODSEQ', state.highest_modseq,
            f'{state.last_synced_uid + 1}:*'
        ]

        try:
            changed_uids = client.search(search_criteria)

            # Fetch MODSEQ values to determine if new or updated
            if changed_uids:
                fetch_data = client.fetch(changed_uids, ['MODSEQ', 'FLAGS'])

                for uid, data in fetch_data.items():
                    if uid > state.last_synced_uid:
                        result.new_messages.append(uid)
                    else:
                        result.updated_messages.append(uid)

                # Update state
                if changed_uids:
                    state.last_synced_uid = max(state.last_synced_uid, max(changed_uids))

                # Update highest MODSEQ
                new_modseq = select_info.get(b'HIGHESTMODSEQ')
                if new_modseq:
                    state.highest_modseq = new_modseq

        except Exception as e:
            logger.error(f"MODSEQ sync failed, falling back to UID: {e}")
            return await self._sync_with_uid(client, state, select_info)

        # Detect deletions
        current_exists = select_info[b'EXISTS']
        if current_exists < state.last_exists_count:
            all_current_uids = set(client.search(['ALL']))
            all_known_uids = set(range(1, state.last_synced_uid + 1))
            deleted_uids = all_known_uids - all_current_uids
            result.deleted_messages = list(deleted_uids)

        state.last_exists_count = current_exists
        state.message_count = current_exists

        return result

    async def _full_resync(
        self,
        client: IMAPClient,
        mailbox_descriptor: ImapMailboxDescriptor,
        folder: str,
        new_uidvalidity: int,
    ) -> SyncResult:
        """Perform full resync when UIDVALIDITY changes."""
        logger.info(f"Full resync required for {folder}")

        # Reset state
        state = ImapSyncState(
            mailbox_id=mailbox_descriptor.id,
            folder=folder,
            uidvalidity=new_uidvalidity,
            last_sync_time=datetime.utcnow(),
        )

        # Get all messages
        all_uids = client.search(['ALL'])
        result = SyncResult(new_messages=all_uids)

        if all_uids:
            state.last_synced_uid = max(all_uids)
            state.message_count = len(all_uids)

        self._save_sync_state(state)
        return result
```

### IDLE Monitoring
```python
class IdleMonitor:
    """Monitor folder for real-time changes using IMAP IDLE."""

    def __init__(
        self,
        connection: ImapConnection,
        folder: str,
        callback: Callable[[SyncResult], Awaitable[None]],
    ):
        self.connection = connection
        self.folder = folder
        self.callback = callback
        self._stop_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self.renewal_interval = 600  # 10 minutes

    async def start(self) -> None:
        """Start IDLE monitoring."""
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop IDLE monitoring."""
        self._stop_event.set()
        if self._monitor_task:
            await self._monitor_task

    async def _monitor_loop(self) -> None:
        """Main IDLE monitoring loop."""
        with self.connection.connect() as client:
            client.select_folder(self.folder)

            while not self._stop_event.is_set():
                try:
                    # Start IDLE
                    client.idle()
                    logger.debug(f"IDLE started for {self.folder}")

                    # Wait for changes or timeout
                    responses = client.idle_check(timeout=self.renewal_interval)

                    # Done IDLE (renew connection)
                    client.idle_done()

                    # Process changes
                    if responses:
                        await self._process_idle_responses(responses)

                except Exception as e:
                    logger.error(f"IDLE error: {e}")
                    # Wait before retry
                    await asyncio.sleep(60)

    async def _process_idle_responses(self, responses: List) -> None:
        """Process IDLE responses and trigger sync."""
        has_changes = False

        for response in responses:
            if isinstance(response, tuple) and len(response) >= 1:
                response_text = response[0]
                if isinstance(response_text, bytes):
                    response_text = response_text.decode('utf-8', errors='replace')

                # Check for EXISTS (new message)
                if 'EXISTS' in response_text:
                    has_changes = True
                # Check for EXPUNGE (deleted message)
                elif 'EXPUNGE' in response_text:
                    has_changes = True
                # Check for FETCH (flag changes)
                elif 'FETCH' in response_text:
                    has_changes = True

        if has_changes:
            logger.info(f"Changes detected in {self.folder}, triggering sync")
            # Trigger sync callback
            # (callback will use sync engine to get details)
            await self.callback(SyncResult())
```

### NOOP Polling (Fallback)
```python
class NoopPoller:
    """Poll folder using NOOP when IDLE not available."""

    def __init__(
        self,
        sync_engine: ImapSyncEngine,
        mailbox_descriptor: ImapMailboxDescriptor,
        folder: str,
        poll_interval: int = 300,  # 5 minutes
    ):
        self.sync_engine = sync_engine
        self.mailbox_descriptor = mailbox_descriptor
        self.folder = folder
        self.poll_interval = poll_interval
        self._stop_event = asyncio.Event()
        self._poll_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start NOOP polling."""
        self._poll_task = asyncio.create_task(self._poll_loop())

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
                result = await self.sync_engine.sync_folder(
                    self.mailbox_descriptor,
                    self.folder,
                )

                if result.new_messages or result.updated_messages or result.deleted_messages:
                    logger.info(
                        f"Sync found changes: {len(result.new_messages)} new, "
                        f"{len(result.updated_messages)} updated, "
                        f"{len(result.deleted_messages)} deleted"
                    )

            except Exception as e:
                logger.error(f"Polling error: {e}")

            # Wait for next poll
            await asyncio.sleep(self.poll_interval)
```

## Acceptance Criteria

- ✅ New messages detected within 5 minutes
- ✅ Updated messages (flag changes) detected
- ✅ Deleted messages detected correctly
- ✅ UIDVALIDITY changes trigger full resync
- ✅ MODSEQ used when server supports it
- ✅ IDLE monitoring works for real-time updates
- ✅ IDLE renewed every 10 minutes
- ✅ NOOP polling works as fallback
- ✅ Sync state persisted correctly
- ✅ < 0.5% sync failure rate
- ✅ Multi-folder sync coordinated efficiently
- ✅ Offline sync queued for later execution

## Test Plan

### Unit Tests
- UID-based sync logic
- MODSEQ-based sync logic
- UIDVALIDITY change detection
- Deletion detection logic
- Sync state persistence

### Integration Tests
- End-to-end sync with mock IMAP server
- IDLE monitoring with simulated changes
- NOOP polling cycle
- Multi-folder sync coordination
- UIDVALIDITY change handling

### Reliability Tests
- Sync under network instability
- Server timeout handling
- Connection drops during IDLE
- Race conditions (concurrent syncs)
- Very large mailboxes (>10k messages)

## Implementation Notes

### UIDVALIDITY Handling
```python
# UIDVALIDITY is a server-assigned value that changes when
# folder contents are restructured (e.g., mailbox migration)
# When it changes, all UIDs are invalidated and full resync required
```

### MODSEQ Support (RFC 7162)
```python
# MODSEQ is a modification sequence number
# Each message has a MODSEQ that increases on any change
# Enables efficient "changed since" queries
```

## Open Questions

- Should we implement QRESYNC (RFC 7162) for even faster sync?
- How to handle very slow first sync (thousands of messages)?
- Should we prioritize certain folders (INBOX first)?
- How to handle server-side folder renames?
- Should we support partial folder sync (date range)?

## Dependencies
- ImapConnectionManager from task 03
- StateStore for sync state persistence
- AuditLogger for sync event tracking
- IngestionOrchestrator for scheduling


