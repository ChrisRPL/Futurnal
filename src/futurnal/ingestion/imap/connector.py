"""IMAP Email Connector - Grounds Ghost in user's email experiential data.

This connector enables the Ghost to learn from the user's email communications,
which represent a rich source of experiential knowledge including:
- Professional and personal interactions over time
- Project discussions and decision-making processes
- Relationship networks via participant analysis
- Temporal patterns of communication

The connector transforms email mailboxes into structured experiential memory
within the PKG, providing the Ghost with high-fidelity recall of the user's
communication history. This forms a critical component of Phase 1 (Archivist)
experiential memory construction.

Integration Architecture:
- Leverages existing IMAP components (EmailParser, SyncEngine, etc.)
- Feeds Unstructured.io pipeline for element extraction
- Generates semantic triples for PKG construction
- Compatible with existing privacy framework (consent, audit, redaction)
- Integrates with IngestionOrchestrator for scheduled sync operations
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from unstructured.partition.auto import partition

from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry, ConsentRequiredError
from futurnal.privacy.redaction import RedactionPolicy, build_policy, redact_path

from .attachment_extractor import AttachmentExtractor
from .attachment_processor import AttachmentProcessor
from .attachment_triples import extract_attachment_triples
from .connection_manager import ImapConnectionPool
from .credential_manager import CredentialManager
from .descriptor import ImapMailboxDescriptor, MailboxRegistry
from .email_normalizer import EmailNormalizer
from .email_parser import EmailParser
from .email_triples import extract_email_triples
from .idle_monitor import IdleMonitor
from .sync_engine import ImapSyncEngine
from .sync_state import ImapSyncStateStore, SyncResult
from .thread_reconstructor import ThreadReconstructor

logger = logging.getLogger(__name__)


def _log_extra(
    *,
    job_id: str | None = None,
    mailbox_id: str | None = None,
    folder: str | None = None,
    uid: int | None = None,
    policy: RedactionPolicy | None = None,
    **metadata: object,
) -> dict[str, object]:
    """Build structured logging context with privacy-aware handling."""
    extra: dict[str, object] = {}
    if job_id:
        extra["ingestion_job_id"] = job_id
    if mailbox_id:
        extra["ingestion_mailbox_id"] = mailbox_id
    if folder:
        extra["ingestion_folder"] = folder
    if uid:
        extra["ingestion_uid"] = uid
    for key, value in metadata.items():
        if value is not None:
            extra[f"ingestion_{key}"] = value
    return extra


@runtime_checkable
class ElementSink(Protocol):
    """Sink interface for handling parsed elements."""

    def handle(self, element: dict) -> None:
        ...

    def handle_deletion(self, element: dict) -> None:  # pragma: no cover - optional
        ...


class ImapEmailConnector:
    """Connector responsible for ingesting IMAP email into Futurnal pipelines.

    This connector provides the complete email ingestion pipeline:
    1. Incremental sync via ImapSyncEngine (UID/MODSEQ-based)
    2. Email parsing and normalization
    3. Element extraction via Unstructured.io
    4. Semantic triple generation for PKG
    5. Element delivery to sink (PKG + vector stores)

    Privacy-First Design:
    - All operations require explicit consent
    - Comprehensive audit logging without content exposure
    - Configurable privacy levels per mailbox
    - Redaction policies applied throughout

    Resource Management:
    - Connection pooling for IMAP operations
    - Batch processing with configurable limits
    - Timeout enforcement for attachment processing
    - Error handling with quarantine support
    """

    def __init__(
        self,
        *,
        workspace_dir: Path | str,
        state_store: ImapSyncStateStore,
        mailbox_registry: MailboxRegistry,
        element_sink: ElementSink | None = None,
        audit_logger: AuditLogger | None = None,
        consent_registry: ConsentRegistry | None = None,
        credential_manager: CredentialManager | None = None,
    ) -> None:
        """Initialize IMAP email connector.

        Args:
            workspace_dir: Root directory for workspace data
            state_store: IMAP sync state store for incremental sync
            mailbox_registry: Registry of configured mailboxes
            element_sink: Optional sink for processed elements
            audit_logger: Optional audit logger for privacy-compliant event recording
            consent_registry: Optional consent registry for enforcement
            credential_manager: Optional credential manager (defaults to system keychain)
        """
        self._workspace_dir = Path(workspace_dir)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        # Core storage directories
        self._parsed_dir = self._workspace_dir / "imap" / "parsed"
        self._parsed_dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_dir = self._workspace_dir / "imap" / "quarantine"
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._attachments_dir = self._workspace_dir / "imap" / "attachments"
        self._attachments_dir.mkdir(parents=True, exist_ok=True)

        # Core dependencies
        self._state_store = state_store
        self._mailbox_registry = mailbox_registry
        self._element_sink = element_sink
        self._audit_logger = audit_logger
        self._consent_registry = consent_registry
        self._credential_manager = credential_manager or CredentialManager(
            audit_logger=self._audit_logger
        )

        # Shared components
        self._thread_reconstructor = ThreadReconstructor()
        self._email_normalizer = EmailNormalizer()

        # Per-mailbox connection pools (lazy initialization)
        self._connection_pools: Dict[str, ImapConnectionPool] = {}

        logger.info(
            "ImapEmailConnector initialized",
            extra={
                "workspace_dir": str(self._workspace_dir),
                "has_element_sink": element_sink is not None,
                "has_audit_logger": audit_logger is not None,
                "has_consent_registry": consent_registry is not None,
            }
        )

    async def sync_mailbox(
        self,
        mailbox_id: str,
        *,
        job_id: str | None = None,
    ) -> Dict[str, SyncResult]:
        """Sync all folders in a mailbox.

        Args:
            mailbox_id: Mailbox identifier
            job_id: Optional job identifier for tracking

        Returns:
            Dictionary mapping folder names to sync results

        Raises:
            ConsentRequiredError: If required consent not granted
            FileNotFoundError: If mailbox not registered
        """
        active_job_id = job_id or uuid.uuid4().hex
        descriptor = self._mailbox_registry.get(mailbox_id)

        logger.info(
            "Starting mailbox sync",
            extra=_log_extra(
                job_id=active_job_id,
                mailbox_id=mailbox_id,
                folder_count=len(descriptor.folders),
                event="mailbox_sync_start",
            )
        )

        # Check consent for mailbox access
        if self._consent_registry:
            try:
                from .consent_manager import ImapConsentScopes
                self._consent_registry.require(
                    source=f"mailbox:{mailbox_id}",
                    scope=ImapConsentScopes.MAILBOX_ACCESS.value,
                )
            except ConsentRequiredError as exc:
                logger.error(
                    "Mailbox access consent missing",
                    extra=_log_extra(
                        job_id=active_job_id,
                        mailbox_id=mailbox_id,
                        event="consent_missing",
                    ),
                    exc_info=exc,
                )
                raise

        # Sync each folder
        results = {}
        for folder in descriptor.folders:
            try:
                result = await self.sync_folder(mailbox_id, folder, job_id=active_job_id)
                results[folder] = result
            except Exception as exc:
                logger.exception(
                    f"Folder sync failed for {folder}",
                    extra=_log_extra(
                        job_id=active_job_id,
                        mailbox_id=mailbox_id,
                        folder=folder,
                        event="folder_sync_failed",
                    ),
                )
                # Continue with other folders
                results[folder] = SyncResult(errors=[str(exc)])

        logger.info(
            "Mailbox sync completed",
            extra=_log_extra(
                job_id=active_job_id,
                mailbox_id=mailbox_id,
                folders_synced=len(results),
                event="mailbox_sync_complete",
            )
        )

        return results

    async def sync_folder(
        self,
        mailbox_id: str,
        folder: str,
        *,
        job_id: str | None = None,
    ) -> SyncResult:
        """Sync a single folder.

        Performs incremental sync via ImapSyncEngine, then processes
        all new/updated/deleted messages.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name to sync
            job_id: Optional job identifier for tracking

        Returns:
            Sync result with counts and errors

        Raises:
            ConsentRequiredError: If required consent not granted
        """
        active_job_id = job_id or uuid.uuid4().hex
        descriptor = self._mailbox_registry.get(mailbox_id)

        logger.debug(
            "Starting folder sync",
            extra=_log_extra(
                job_id=active_job_id,
                mailbox_id=mailbox_id,
                folder=folder,
                event="folder_sync_start",
            )
        )

        # Get or create connection pool
        pool = await self._get_connection_pool(descriptor)
        # Test doubles may return pools without a concrete descriptor id.
        # Sync engine requires a stable mailbox id for state-store keys.
        pool_descriptor_id = getattr(getattr(pool, "descriptor", None), "id", None)
        if not isinstance(pool_descriptor_id, str):
            try:
                pool.descriptor = descriptor
            except Exception:  # pragma: no cover - defensive for atypical pool stubs
                pass

        # Create sync engine
        # Note: consent_manager=None because consent is already checked at the connector level
        sync_engine = ImapSyncEngine(
            connection_pool=pool,
            state_store=self._state_store,
            audit_logger=self._audit_logger,
            consent_manager=None,
        )

        # Perform incremental sync
        sync_result = await sync_engine.sync_folder(folder)

        logger.info(
            "Folder sync completed",
            extra=_log_extra(
                job_id=active_job_id,
                mailbox_id=mailbox_id,
                folder=folder,
                new_messages=len(sync_result.new_messages),
                updated_messages=len(sync_result.updated_messages),
                deleted_messages=len(sync_result.deleted_messages),
                event="folder_sync_complete",
            )
        )

        # Process new messages
        for uid in sync_result.new_messages:
            try:
                await self.process_email(mailbox_id, folder, uid, job_id=active_job_id)
            except Exception as exc:
                logger.exception(
                    f"Failed to process new message {uid}",
                    extra=_log_extra(
                        job_id=active_job_id,
                        mailbox_id=mailbox_id,
                        folder=folder,
                        uid=uid,
                        event="process_email_failed",
                    ),
                )
                sync_result.errors.append(f"Process email {uid}: {exc}")

        # Process updated messages (flag changes)
        for uid in sync_result.updated_messages:
            try:
                await self.process_email_update(mailbox_id, folder, uid, job_id=active_job_id)
            except Exception as exc:
                logger.exception(
                    f"Failed to process updated message {uid}",
                    extra=_log_extra(
                        job_id=active_job_id,
                        mailbox_id=mailbox_id,
                        folder=folder,
                        uid=uid,
                        event="process_email_update_failed",
                    ),
                )
                sync_result.errors.append(f"Process update {uid}: {exc}")

        # Process deleted messages
        for uid in sync_result.deleted_messages:
            try:
                await self.process_email_deletion(mailbox_id, folder, uid, job_id=active_job_id)
            except Exception as exc:
                logger.exception(
                    f"Failed to process deleted message {uid}",
                    extra=_log_extra(
                        job_id=active_job_id,
                        mailbox_id=mailbox_id,
                        folder=folder,
                        uid=uid,
                        event="process_email_deletion_failed",
                    ),
                )
                sync_result.errors.append(f"Process deletion {uid}: {exc}")

        return sync_result

    async def process_email(
        self,
        mailbox_id: str,
        folder: str,
        uid: int,
        *,
        job_id: str | None = None,
    ) -> None:
        """Process a single email message through the full pipeline.

        Pipeline stages:
        1. Fetch raw RFC822 message from IMAP
        2. Parse with EmailParser
        3. Extract attachments with AttachmentExtractor
        4. Normalize email for Unstructured.io
        5. Extract elements via partition_text
        6. Send elements to ElementSink
        7. Process attachments through AttachmentProcessor
        8. Extract semantic triples (email + attachments)
        9. Send triples to ElementSink
        10. Add to ThreadReconstructor

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name
            uid: Message UID
            job_id: Optional job identifier for tracking
        """
        active_job_id = job_id or uuid.uuid4().hex
        descriptor = self._mailbox_registry.get(mailbox_id)
        policy = descriptor.build_redaction_policy()

        logger.debug(
            "Processing email",
            extra=_log_extra(
                job_id=active_job_id,
                mailbox_id=mailbox_id,
                folder=folder,
                uid=uid,
                event="process_email_start",
            )
        )

        try:
            # Fetch raw message
            pool = await self._get_connection_pool(descriptor)
            async with pool.acquire() as connection:
                with connection.connect() as client:
                    client.select_folder(folder)
                    fetch_data = client.fetch([uid], ['RFC822', 'FLAGS'])
                    if uid not in fetch_data:
                        raise ValueError(f"Message {uid} not found in folder {folder}")

                    raw_message = fetch_data[uid][b'RFC822']
                    flags = fetch_data[uid].get(b'FLAGS', [])
                    flags_str = [f.decode() if isinstance(f, bytes) else str(f) for f in flags]

            # Parse email
            # Note: consent_manager=None because consent is already checked at the connector level
            email_parser = EmailParser(
                privacy_policy=descriptor.privacy_settings,
                audit_logger=self._audit_logger,
                consent_manager=None,
            )

            email_message = email_parser.parse_message(
                raw_message=raw_message,
                uid=uid,
                folder=folder,
                mailbox_id=mailbox_id,
                flags=flags_str,
            )

            # Extract attachments
            attachment_extractor = AttachmentExtractor(
                storage_dir=self._attachments_dir,
                audit_logger=self._audit_logger,
            )

            attachments = attachment_extractor.extract_attachments(
                raw_message=raw_message,
                message_id=email_message.message_id,
                mailbox_id=mailbox_id,
            )
            email_message.attachments = attachments

            # Add to thread reconstructor
            self._thread_reconstructor.add_message(email_message)

            # Normalize email for Unstructured.io
            normalized_text = self._email_normalizer.normalize(email_message)

            # Save normalized text to temporary file for Unstructured.io
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                tmp_file.write(normalized_text)
                tmp_file_path = tmp_file.name

            try:
                # Process with Unstructured.io
                elements = partition(filename=tmp_file_path, strategy="fast", include_metadata=True)
            finally:
                # Clean up temp file
                Path(tmp_file_path).unlink(missing_ok=True)

            # Enrich and send elements to sink
            for element in elements:
                enriched = self._persist_element(
                    mailbox_id=mailbox_id,
                    folder=folder,
                    uid=uid,
                    message_id=email_message.message_id,
                    element=element,
                    job_id=active_job_id,
                )

                if self._element_sink:
                    try:
                        self._element_sink.handle(enriched)
                    except Exception as exc:
                        logger.exception(
                            "Element sink failed",
                            extra=_log_extra(
                                job_id=active_job_id,
                                mailbox_id=mailbox_id,
                                folder=folder,
                                uid=uid,
                                event="sink_failed",
                            ),
                        )
                        self._quarantine(
                            mailbox_id, folder, uid,
                            "sink_error", str(exc), policy
                        )
                        raise

            # Process attachments
            attachment_processor = AttachmentProcessor(
                audit_logger=self._audit_logger,
            )

            for attachment in attachments:
                try:
                    attachment_elements = await attachment_processor.process_attachment(attachment)

                    # Send attachment elements to sink
                    if self._element_sink:
                        for att_element in attachment_elements:
                            # Enrich with attachment metadata
                            att_element.setdefault('metadata', {}).update({
                                'source_attachment_id': attachment.attachment_id,
                                'source_message_id': email_message.message_id,
                                'source_mailbox_id': mailbox_id,
                                'source_folder': folder,
                                'source_uid': uid,
                            })
                            self._element_sink.handle(att_element)

                except Exception as exc:
                    logger.warning(
                        f"Attachment processing failed for {attachment.filename}",
                        extra=_log_extra(
                            job_id=active_job_id,
                            mailbox_id=mailbox_id,
                            folder=folder,
                            uid=uid,
                            attachment_id=attachment.attachment_id,
                            event="attachment_processing_failed",
                        ),
                    )
                    # Continue with other attachments

            # Generate semantic triples for email
            email_triples = extract_email_triples(email_message)

            # Generate semantic triples for attachments
            attachment_triples = []
            for attachment in attachments:
                attachment_triples.extend(
                    extract_attachment_triples(attachment, email_message.message_id)
                )

            # Send triples to element sink
            if self._element_sink:
                for triple in email_triples + attachment_triples:
                    try:
                        self._element_sink.handle({
                            'type': 'semantic_triple',
                            'triple': triple.to_dict(),
                            'metadata': {
                                'source': 'imap',
                                'mailbox_id': mailbox_id,
                                'folder': folder,
                                'uid': uid,
                                'message_id': email_message.message_id,
                            }
                        })
                    except Exception as exc:
                        logger.warning(
                            "Triple sink failed",
                            extra=_log_extra(
                                job_id=active_job_id,
                                mailbox_id=mailbox_id,
                                folder=folder,
                                uid=uid,
                                event="triple_sink_failed",
                            ),
                        )
                        # Continue with other triples

            logger.debug(
                "Email processing completed",
                extra=_log_extra(
                    job_id=active_job_id,
                    mailbox_id=mailbox_id,
                    folder=folder,
                    uid=uid,
                    element_count=len(elements),
                    attachment_count=len(attachments),
                    triple_count=len(email_triples) + len(attachment_triples),
                    event="process_email_complete",
                )
            )

        except Exception as exc:
            logger.exception(
                "Email processing failed",
                extra=_log_extra(
                    job_id=active_job_id,
                    mailbox_id=mailbox_id,
                    folder=folder,
                    uid=uid,
                    event="process_email_failed",
                ),
            )
            self._quarantine(
                mailbox_id, folder, uid,
                "processing_error", str(exc), policy
            )
            raise

    async def process_email_update(
        self,
        mailbox_id: str,
        folder: str,
        uid: int,
        *,
        job_id: str | None = None,
    ) -> None:
        """Process email flag update.

        Currently, flag updates trigger full reprocessing.
        Future optimization: only update metadata.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name
            uid: Message UID
            job_id: Optional job identifier for tracking
        """
        active_job_id = job_id or uuid.uuid4().hex

        logger.debug(
            "Processing email update",
            extra=_log_extra(
                job_id=active_job_id,
                mailbox_id=mailbox_id,
                folder=folder,
                uid=uid,
                event="process_email_update",
            )
        )

        # For now, reprocess the email completely
        # TODO: Optimize to only update flags/metadata
        await self.process_email(mailbox_id, folder, uid, job_id=active_job_id)

    async def process_email_deletion(
        self,
        mailbox_id: str,
        folder: str,
        uid: int,
        *,
        job_id: str | None = None,
    ) -> None:
        """Process email deletion.

        Notifies element sink to remove email and related elements
        from PKG and vector stores.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name
            uid: Message UID
            job_id: Optional job identifier for tracking
        """
        active_job_id = job_id or uuid.uuid4().hex

        logger.debug(
            "Processing email deletion",
            extra=_log_extra(
                job_id=active_job_id,
                mailbox_id=mailbox_id,
                folder=folder,
                uid=uid,
                event="process_email_deletion",
            )
        )

        # Notify element sink
        if self._element_sink and hasattr(self._element_sink, 'handle_deletion'):
            try:
                self._element_sink.handle_deletion({
                    'type': 'email',
                    'uid': uid,
                    'folder': folder,
                    'mailbox_id': mailbox_id,
                })
            except Exception as exc:
                logger.exception(
                    "Element sink deletion callback failed",
                    extra=_log_extra(
                        job_id=active_job_id,
                        mailbox_id=mailbox_id,
                        folder=folder,
                        uid=uid,
                        event="sink_deletion_failed",
                    ),
                )

    def ingest(
        self,
        mailbox_id: str,
        *,
        job_id: str | None = None,
    ) -> Iterable[dict]:
        """Synchronous ingest interface for orchestrator compatibility.

        This method provides a generator interface that yields processed elements,
        compatible with the LocalFilesConnector pattern used by IngestionOrchestrator.

        Internally, it runs the async sync_mailbox() method and yields results.

        Args:
            mailbox_id: Mailbox identifier to sync
            job_id: Optional job identifier for tracking

        Yields:
            Processed element dictionaries
        """
        active_job_id = job_id or uuid.uuid4().hex

        # Run async sync in an event loop (Python 3.11+ may not have a current loop by default).
        try:
            loop = asyncio.get_event_loop()
            created_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            created_loop = True

        try:
            if loop.is_running():
                raise RuntimeError(
                    "ingest() cannot run inside an active event loop; "
                    "use sync_mailbox() from async contexts"
                )
            results = loop.run_until_complete(
                self.sync_mailbox(mailbox_id, job_id=active_job_id)
            )
        finally:
            if created_loop:
                asyncio.set_event_loop(None)
                loop.close()

        # Yield summary result
        total_new = sum(len(r.new_messages) for r in results.values())
        total_updated = sum(len(r.updated_messages) for r in results.values())
        total_deleted = sum(len(r.deleted_messages) for r in results.values())

        yield {
            'source': f'imap-{mailbox_id}',
            'mailbox_id': mailbox_id,
            'folders_synced': len(results),
            'new_messages': total_new,
            'updated_messages': total_updated,
            'deleted_messages': total_deleted,
            'job_id': active_job_id,
        }

    async def start_idle_monitor(
        self,
        mailbox_id: str,
        folder: str = "INBOX",
        *,
        on_new_message: Optional[callable] = None,
    ) -> None:
        """Start IDLE monitoring for real-time email sync.

        This method uses IMAP IDLE to receive push notifications from the server
        when new messages arrive, providing real-time sync without polling.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder to monitor (default: INBOX)
            on_new_message: Optional callback for new message notifications

        Raises:
            ConsentRequiredError: If required consent not granted
            RuntimeError: If IDLE not supported by server

        Example:
            >>> connector = ImapEmailConnector(...)
            >>> async def handle_new(uid):
            ...     print(f"New message: {uid}")
            >>> await connector.start_idle_monitor("mailbox-id", on_new_message=handle_new)
        """
        descriptor = self._mailbox_registry.get(mailbox_id)

        logger.info(
            "Starting IDLE monitor",
            extra=_log_extra(
                mailbox_id=mailbox_id,
                folder=folder,
                event="idle_monitor_start",
            )
        )

        # Check consent for mailbox access
        if self._consent_registry:
            try:
                from .consent_manager import ImapConsentScopes
                self._consent_registry.require(
                    source=f"mailbox:{mailbox_id}",
                    scope=ImapConsentScopes.MAILBOX_ACCESS.value,
                )
            except ConsentRequiredError as exc:
                logger.error(
                    "Mailbox access consent missing for IDLE monitor",
                    extra=_log_extra(
                        mailbox_id=mailbox_id,
                        folder=folder,
                        event="consent_missing",
                    ),
                    exc_info=exc,
                )
                raise

        # Get connection pool
        pool = await self._get_connection_pool(descriptor)

        # Check if IDLE is supported
        async with pool.acquire() as connection:
            with connection.connect() as client:
                if b'IDLE' not in client.capabilities():
                    raise RuntimeError(
                        f"IMAP IDLE not supported by server {descriptor.imap_host}"
                    )

        # Create IDLE monitor
        idle_monitor = IdleMonitor(
            connection_pool=pool,
            folder=folder,
            on_new_message=self._handle_idle_new_message if on_new_message is None else on_new_message,
            on_expunge=self._handle_idle_expunge,
            audit_logger=self._audit_logger,
        )

        # Store mailbox_id for callbacks
        self._current_idle_mailbox_id = mailbox_id
        self._current_idle_folder = folder

        # Start monitoring (this will run until stopped)
        try:
            await idle_monitor.start()
        except KeyboardInterrupt:
            logger.info(
                "IDLE monitor stopped by user",
                extra=_log_extra(
                    mailbox_id=mailbox_id,
                    folder=folder,
                    event="idle_monitor_stopped",
                )
            )
        except Exception as exc:
            logger.exception(
                "IDLE monitor failed",
                extra=_log_extra(
                    mailbox_id=mailbox_id,
                    folder=folder,
                    event="idle_monitor_failed",
                )
            )
            raise

    async def _handle_idle_new_message(self, uid: int) -> None:
        """Handle new message notification from IDLE monitor.

        Args:
            uid: Message UID
        """
        logger.debug(
            "IDLE: New message detected",
            extra=_log_extra(
                mailbox_id=self._current_idle_mailbox_id,
                folder=self._current_idle_folder,
                uid=uid,
                event="idle_new_message",
            )
        )

        # Process the new message
        try:
            await self.process_email(
                self._current_idle_mailbox_id,
                self._current_idle_folder,
                uid,
            )
        except Exception as exc:
            logger.exception(
                "Failed to process IDLE new message",
                extra=_log_extra(
                    mailbox_id=self._current_idle_mailbox_id,
                    folder=self._current_idle_folder,
                    uid=uid,
                    event="idle_process_failed",
                )
            )

    async def _handle_idle_expunge(self, uid: int) -> None:
        """Handle message deletion notification from IDLE monitor.

        Args:
            uid: Message UID
        """
        logger.debug(
            "IDLE: Message expunged",
            extra=_log_extra(
                mailbox_id=self._current_idle_mailbox_id,
                folder=self._current_idle_folder,
                uid=uid,
                event="idle_expunge",
            )
        )

        # Process the deletion
        try:
            await self.process_email_deletion(
                self._current_idle_mailbox_id,
                self._current_idle_folder,
                uid,
            )
        except Exception as exc:
            logger.exception(
                "Failed to process IDLE expunge",
                extra=_log_extra(
                    mailbox_id=self._current_idle_mailbox_id,
                    folder=self._current_idle_folder,
                    uid=uid,
                    event="idle_expunge_failed",
                )
            )

    async def _get_connection_pool(self, descriptor: ImapMailboxDescriptor) -> ImapConnectionPool:
        """Get or create connection pool for mailbox.

        Args:
            descriptor: Mailbox descriptor

        Returns:
            Connection pool for the mailbox
        """
        if descriptor.id not in self._connection_pools:
            pool = ImapConnectionPool(
                descriptor=descriptor,
                credential_manager=self._credential_manager,
                max_connections=3,  # Conservative pool size for on-device usage
            )
            self._connection_pools[descriptor.id] = pool

        return self._connection_pools[descriptor.id]

    def _persist_element(
        self,
        *,
        mailbox_id: str,
        folder: str,
        uid: int,
        message_id: str,
        element: Any,
        job_id: str,
    ) -> dict:
        """Persist element to local storage and enrich with metadata.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name
            uid: Message UID
            message_id: Email message ID
            element: Element from Unstructured.io
            job_id: Job identifier

        Returns:
            Enriched element dictionary
        """
        # Generate element ID
        element_id = f"{mailbox_id}_{folder}_{uid}_{uuid.uuid4().hex[:8]}"
        storage_path = self._parsed_dir / f"{element_id}.json"

        # Convert element to dict
        if hasattr(element, "to_dict"):
            payload = element.to_dict()
        elif isinstance(element, dict):
            payload = element
        else:
            payload = {"text": str(element)}

        # Enrich with metadata
        payload.setdefault("metadata", {})
        payload["metadata"].update({
            "source": "imap",
            "mailbox_id": mailbox_id,
            "folder": folder,
            "uid": uid,
            "message_id": message_id,
            "element_id": element_id,
            "ingested_at": datetime.utcnow().isoformat(),
            "job_id": job_id,
        })

        # Persist to disk
        storage_path.write_text(json.dumps(payload, ensure_ascii=False))

        return {
            "source": "imap",
            "mailbox_id": mailbox_id,
            "folder": folder,
            "uid": uid,
            "message_id": message_id,
            "element_id": element_id,
            "element_path": str(storage_path),
            **payload,
        }

    def _quarantine(
        self,
        mailbox_id: str,
        folder: str,
        uid: int,
        reason: str,
        detail: str,
        policy: RedactionPolicy | None = None,
    ) -> None:
        """Quarantine failed email with retry metadata.

        Args:
            mailbox_id: Mailbox identifier
            folder: Folder name
            uid: Message UID
            reason: Failure reason category
            detail: Detailed error message
            policy: Redaction policy for logging
        """
        payload = {
            "mailbox_id": mailbox_id,
            "folder": folder,
            "uid": uid,
            "reason": reason,
            "detail": detail,
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": 0,
            "last_retry_at": None,
            "notes": [],
        }

        identifier = f"{mailbox_id}_{folder}_{uid}"
        quarantine_file = self._quarantine_dir / f"{identifier}.json"
        quarantine_file.write_text(json.dumps(payload, ensure_ascii=False))

        logger.warning(
            "Email quarantined",
            extra=_log_extra(
                mailbox_id=mailbox_id,
                folder=folder,
                uid=uid,
                event="quarantine",
                reason=reason,
            ),
        )


__all__ = ["ImapEmailConnector"]
