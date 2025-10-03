Summary: Integrate IMAP connector with IngestionOrchestrator for scheduled sync and element processing.

# 09 · Connector & Orchestrator Integration

## Purpose
Integrate the IMAP email connector with the existing IngestionOrchestrator, ElementSink, and StateStore infrastructure, enabling scheduled sync operations, element processing pipeline, and unified job management alongside Obsidian and Local Files connectors.

## Scope
- ImapEmailConnector implementation (follows LocalFilesConnector pattern)
- ElementSink integration for processed emails and attachments
- StateStore integration for sync state persistence
- IngestionOrchestrator registration for IDLE/NOOP scheduling
- Job queue integration for async processing
- Unified telemetry and health monitoring
- CLI commands for mailbox management and sync operations

## Requirements Alignment
- **Unified architecture**: Follow established connector patterns
- **Scheduled sync**: IDLE/NOOP polling via APScheduler
- **Element pipeline**: Feed Unstructured.io elements to ElementSink
- **State management**: Persist sync state via StateStore
- **Observable**: Telemetry and health metrics

## Component Design

### ImapEmailConnector
```python
class ImapEmailConnector:
    """Main connector for IMAP email ingestion."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        state_store: StateStore,
        mailbox_registry: MailboxRegistry,
        element_sink: Optional[ElementSink] = None,
        audit_logger: Optional[AuditLogger] = None,
        consent_registry: Optional[ConsentRegistry] = None,
        credential_manager: Optional[CredentialManager] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.state_store = state_store
        self.mailbox_registry = mailbox_registry
        self.element_sink = element_sink
        self.audit_logger = audit_logger or AuditLogger(self.workspace_dir / "audit")
        self.consent_registry = consent_registry or ConsentRegistry(self.workspace_dir / "privacy")
        self.credential_manager = credential_manager or CredentialManager(
            audit_logger=self.audit_logger
        )

        # Initialize sub-components
        self.connection_manager = ImapConnectionManager(
            credential_manager=self.credential_manager,
            audit_logger=self.audit_logger,
        )

        self.sync_engine = ImapSyncEngine(
            connection_manager=self.connection_manager,
            state_store=self.state_store,
            audit_logger=self.audit_logger,
        )

        self.email_parser = EmailParser(
            audit_logger=self.audit_logger,
        )

        self.attachment_extractor = AttachmentExtractor(
            storage_dir=self.workspace_dir / "attachments",
        )

        self.attachment_processor = AttachmentProcessor()

        self.thread_reconstructor = ThreadReconstructor()

    async def sync_mailbox(
        self,
        mailbox_id: str,
    ) -> Dict[str, SyncResult]:
        """Sync all folders in a mailbox."""
        # Get mailbox descriptor
        descriptor = self.mailbox_registry.get(mailbox_id)

        # Check consent
        self.consent_registry.require_consent(
            source=f"mailbox:{mailbox_id}",
            scope=ImapConsentScopes.MAILBOX_ACCESS,
        )

        # Sync each folder
        results = {}
        for folder in descriptor.folders:
            result = await self.sync_folder(mailbox_id, folder)
            results[folder] = result

        return results

    async def sync_folder(
        self,
        mailbox_id: str,
        folder: str,
    ) -> SyncResult:
        """Sync a single folder."""
        descriptor = self.mailbox_registry.get(mailbox_id)

        # Perform incremental sync
        sync_result = await self.sync_engine.sync_folder(descriptor, folder)

        # Process new messages
        for uid in sync_result.new_messages:
            await self.process_email(mailbox_id, folder, uid)

        # Process updated messages (flag changes)
        for uid in sync_result.updated_messages:
            await self.process_email_update(mailbox_id, folder, uid)

        # Process deleted messages
        for uid in sync_result.deleted_messages:
            await self.process_email_deletion(mailbox_id, folder, uid)

        return sync_result

    async def process_email(
        self,
        mailbox_id: str,
        folder: str,
        uid: int,
    ) -> None:
        """Process a single email message."""
        descriptor = self.mailbox_registry.get(mailbox_id)

        # Fetch raw message
        async with self.connection_manager.acquire() as connection:
            with connection.connect() as client:
                client.select_folder(folder)
                fetch_data = client.fetch([uid], ['RFC822'])
                raw_message = fetch_data[uid][b'RFC822']

        # Parse email
        email_message = self.email_parser.parse_message(
            raw_message=raw_message,
            uid=uid,
            folder=folder,
            mailbox_id=mailbox_id,
        )

        # Extract attachments
        attachments = self.attachment_extractor.extract_attachments(
            raw_message=raw_message,
            message_id=email_message.message_id,
            mailbox_id=mailbox_id,
        )
        email_message.attachments = attachments

        # Add to thread reconstructor
        self.thread_reconstructor.add_message(email_message)

        # Process email body with Unstructured.io
        elements = await process_email_with_unstructured(
            email_message,
            EmailNormalizer(),
        )

        # Send to element sink
        if self.element_sink:
            for element in elements:
                self.element_sink.handle(element)

        # Process attachments
        for attachment in attachments:
            if attachment.processing_status == AttachmentProcessingStatus.PENDING:
                attachment_elements = await self.attachment_processor.process_attachment(attachment)
                if self.element_sink:
                    for element in attachment_elements:
                        self.element_sink.handle(element)

        # Generate semantic triples
        email_triples = extract_email_triples(email_message)
        attachment_triples = []
        for attachment in attachments:
            attachment_triples.extend(
                extract_attachment_triples(attachment, email_message.message_id)
            )

        # Send triples to element sink (as metadata)
        if self.element_sink:
            for triple in email_triples + attachment_triples:
                self.element_sink.handle({
                    'type': 'semantic_triple',
                    'triple': triple.to_dict(),
                })

    async def process_email_deletion(
        self,
        mailbox_id: str,
        folder: str,
        uid: int,
    ) -> None:
        """Process email deletion."""
        # Notify element sink
        if self.element_sink and hasattr(self.element_sink, 'handle_deletion'):
            self.element_sink.handle_deletion({
                'type': 'email',
                'uid': uid,
                'folder': folder,
                'mailbox_id': mailbox_id,
            })

    async def start_idle_monitor(
        self,
        mailbox_id: str,
        folder: str,
    ) -> IdleMonitor:
        """Start IDLE monitoring for a folder."""
        descriptor = self.mailbox_registry.get(mailbox_id)

        # Create connection for IDLE
        connection = await connect_with_retry(descriptor, self.credential_manager)

        # Create IDLE monitor
        async def idle_callback(result: SyncResult):
            """Callback when IDLE detects changes."""
            await self.sync_folder(mailbox_id, folder)

        monitor = IdleMonitor(connection, folder, idle_callback)
        await monitor.start()

        return monitor
```

### Orchestrator Registration
```python
class ImapSourceRegistration:
    """Registration helper for IMAP sources."""

    @staticmethod
    def register_mailbox(
        orchestrator: IngestionOrchestrator,
        mailbox_descriptor: ImapMailboxDescriptor,
        schedule: str = "@interval",
        interval_seconds: int = 300,  # 5 minutes
        priority: JobPriority = JobPriority.NORMAL,
    ) -> SourceRegistration:
        """Register IMAP mailbox with orchestrator."""

        # Create ingestion source
        source = LocalIngestionSource(
            name=f"imap-{mailbox_descriptor.email_address}",
            root_path=Path("/virtual/imap"),  # Virtual path for compatibility
            schedule=schedule,
            priority=priority.value,
        )

        # Register with orchestrator
        registration = SourceRegistration(
            source=source,
            schedule=schedule,
            interval_seconds=interval_seconds,
            priority=priority,
        )

        orchestrator.register_source(registration)

        return registration
```

### CLI Integration
```python
# Extend futurnal CLI with IMAP commands

import typer
from futurnal.cli import app

imap_app = typer.Typer(help="IMAP email connector commands")
app.add_typer(imap_app, name="imap")

@imap_app.command("add")
def add_mailbox(
    email: str = typer.Option(..., help="Email address"),
    name: Optional[str] = typer.Option(None, help="Mailbox name"),
    provider: Optional[str] = typer.Option(None, help="Provider (gmail, office365, generic)"),
    auth: str = typer.Option("oauth2", help="Auth mode (oauth2, app-password)"),
):
    """Add a new IMAP mailbox."""
    # Implementation from task 01
    pass

@imap_app.command("sync")
def sync_mailbox(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID"),
    folder: Optional[str] = typer.Option(None, help="Specific folder to sync"),
):
    """Manually trigger mailbox sync."""
    # Use ImapEmailConnector.sync_mailbox()
    pass

@imap_app.command("start-monitor")
def start_monitor(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID"),
    folder: str = typer.Option("INBOX", help="Folder to monitor"),
):
    """Start IDLE monitoring for real-time sync."""
    # Use ImapEmailConnector.start_idle_monitor()
    pass

@imap_app.command("status")
def mailbox_status(
    mailbox_id: str = typer.Argument(..., help="Mailbox ID"),
):
    """Show sync status and statistics."""
    # Load sync state from StateStore
    pass
```

### ElementSink Integration
```python
class UnifiedElementSink:
    """Unified sink for all connector types."""

    def __init__(
        self,
        pkg_storage: PKGStorage,
        vector_storage: VectorStorage,
    ):
        self.pkg_storage = pkg_storage
        self.vector_storage = vector_storage

    def handle(self, element: Dict) -> None:
        """Handle element from any connector."""
        element_type = element.get('type')

        if element_type == 'semantic_triple':
            # Store triple in PKG
            triple_data = element['triple']
            self.pkg_storage.store_triple(
                subject=triple_data['subject'],
                predicate=triple_data['predicate'],
                object=triple_data['object'],
                metadata=triple_data,
            )

        elif element_type in ['CompositeElement', 'Title', 'NarrativeText', 'Table', 'ListItem']:
            # Store Unstructured.io element
            # Generate vector embedding
            embedding = self._generate_embedding(element['text'])

            # Store in vector DB
            self.vector_storage.store(
                id=element['element_id'],
                embedding=embedding,
                metadata=element['metadata'],
            )

            # Store reference in PKG
            self.pkg_storage.store_element_reference(element)

    def handle_deletion(self, element: Dict) -> None:
        """Handle element deletion."""
        element_type = element.get('type')

        if element_type == 'email':
            # Remove email elements from PKG and vector store
            message_id = element.get('message_id')
            self.pkg_storage.delete_by_source(message_id)
            self.vector_storage.delete_by_metadata('source_message_id', message_id)
```

## Acceptance Criteria

- ✅ ImapEmailConnector implements connector interface
- ✅ Mailbox registered with IngestionOrchestrator successfully
- ✅ IDLE/NOOP polling scheduled via APScheduler
- ✅ Email elements sent to ElementSink
- ✅ Attachment elements sent to ElementSink
- ✅ Semantic triples sent to ElementSink
- ✅ Sync state persisted via StateStore
- ✅ CLI commands functional (add, sync, status, start-monitor)
- ✅ Deletion events propagate to PKG and vector stores
- ✅ Telemetry metrics collected (sync count, message count, errors)
- ✅ Health checks report sync status

## Test Plan

### Unit Tests
- ImapEmailConnector initialization
- Mailbox registration logic
- Element sink integration
- State store integration
- CLI command parsing

### Integration Tests
- End-to-end sync with orchestrator
- Element sink receives all element types
- State store persists sync checkpoints
- IDLE monitoring integration
- CLI workflow (add → sync → status)

### System Tests
- Multi-mailbox sync coordination
- Concurrent sync operations
- Long-running IDLE monitors
- Graceful shutdown (stop IDLE cleanly)
- Recovery after restart (resume from checkpoint)

## Implementation Notes

### Orchestrator Integration Pattern
```python
# Follow pattern from ObsidianConnector initialization
self._imap_connector = ImapEmailConnector(
    workspace_dir=self._workspace_dir,
    state_store=state_store,
    mailbox_registry=MailboxRegistry(),
    element_sink=element_sink,
    audit_logger=self._audit_logger,
    consent_registry=self._consent_registry,
)
```

### Job Scheduling
```python
# Schedule NOOP polling (fallback when IDLE not available)
orchestrator.register_source(SourceRegistration(
    source=imap_source,
    schedule="@interval",
    interval_seconds=300,  # 5 minutes
    priority=JobPriority.NORMAL,
))
```

## Open Questions

- Should we support multiple IDLE monitors per mailbox (one per folder)?
- How to coordinate IDLE and NOOP polling (prefer IDLE, fallback to NOOP)?
- Should we batch element processing or stream to sink?
- How to handle orchestrator shutdown (cleanly stop IDLE)?
- Should we expose sync progress via telemetry?

## Dependencies
- IngestionOrchestrator from orchestrator module
- ElementSink protocol
- StateStore from local connector
- All components from tasks 01-08


