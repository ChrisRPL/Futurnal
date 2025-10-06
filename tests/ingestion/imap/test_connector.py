"""Unit tests for ImapEmailConnector."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

from futurnal.ingestion.imap.connector import ImapEmailConnector
from futurnal.ingestion.imap.descriptor import (
    AuthMode,
    ImapMailboxDescriptor,
    MailboxRegistry,
)
from futurnal.ingestion.imap.email_parser import EmailAddress, EmailMessage
from futurnal.ingestion.imap.sync_state import ImapSyncState, ImapSyncStateStore, SyncResult
from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry


@pytest.fixture
def workspace_dir(tmp_path):
    """Create temporary workspace directory."""
    return tmp_path / "workspace"


@pytest.fixture
def state_store(tmp_path):
    """Create IMAP state store."""
    db_path = tmp_path / "sync_state.db"
    return ImapSyncStateStore(path=db_path)


@pytest.fixture
def mailbox_registry(tmp_path):
    """Create mailbox registry."""
    registry_root = tmp_path / "registry"
    return MailboxRegistry(registry_root=registry_root)


@pytest.fixture
def audit_logger(tmp_path):
    """Create audit logger."""
    audit_dir = tmp_path / "audit"
    return AuditLogger(audit_dir)


@pytest.fixture
def consent_registry(tmp_path):
    """Create consent registry."""
    consent_dir = tmp_path / "consent"
    return ConsentRegistry(consent_dir)


@pytest.fixture
def test_descriptor():
    """Create test mailbox descriptor."""
    return ImapMailboxDescriptor.from_registration(
        email_address="test@example.com",
        imap_host="imap.example.com",
        imap_port=993,
        name="Test Mailbox",
        auth_mode=AuthMode.APP_PASSWORD,
        credential_id="test_cred_id",
        folders=["INBOX", "Sent"],
    )


@pytest.fixture
def credential_manager():
    """Create mock credential manager."""
    return MagicMock()


@pytest.fixture
def connector(workspace_dir, state_store, mailbox_registry, audit_logger, consent_registry, credential_manager):
    """Create ImapEmailConnector instance."""
    return ImapEmailConnector(
        workspace_dir=workspace_dir,
        state_store=state_store,
        mailbox_registry=mailbox_registry,
        element_sink=None,
        audit_logger=audit_logger,
        consent_registry=consent_registry,
        credential_manager=credential_manager,
    )


def test_connector_initialization(connector, workspace_dir):
    """Test connector initializes with correct directories."""
    assert connector._workspace_dir == workspace_dir
    assert (workspace_dir / "imap" / "parsed").exists()
    assert (workspace_dir / "imap" / "quarantine").exists()
    assert (workspace_dir / "imap" / "attachments").exists()


def test_connector_components_initialized(connector):
    """Test all sub-components are initialized."""
    assert connector._state_store is not None
    assert connector._mailbox_registry is not None
    assert connector._thread_reconstructor is not None
    assert connector._email_normalizer is not None
    assert connector._credential_manager is not None


@pytest.mark.asyncio
async def test_sync_mailbox_with_consent(connector, mailbox_registry, test_descriptor, consent_registry):
    """Test mailbox sync with proper consent."""
    # Register mailbox
    mailbox_registry.add_or_update(test_descriptor)

    # Grant consent
    from futurnal.ingestion.imap.consent_manager import ImapConsentScopes
    consent_registry.grant(
        source=f"mailbox:{test_descriptor.id}",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Mock sync engine
    with patch.object(connector, '_get_connection_pool') as mock_pool:
        mock_pool.return_value = AsyncMock()

        with patch('futurnal.ingestion.imap.connector.ImapSyncEngine') as mock_sync_engine_class:
            mock_engine = AsyncMock()
            mock_sync_engine_class.return_value = mock_engine

            # Mock empty sync result
            mock_engine.sync_folder.return_value = SyncResult()

            # Sync mailbox
            results = await connector.sync_mailbox(test_descriptor.id)

            # Verify results
            assert isinstance(results, dict)
            assert "INBOX" in results
            assert "Sent" in results
            assert mock_engine.sync_folder.call_count == 2


@pytest.mark.asyncio
async def test_sync_mailbox_without_consent(connector, mailbox_registry, test_descriptor):
    """Test mailbox sync fails without consent."""
    # Register mailbox
    mailbox_registry.add_or_update(test_descriptor)

    # Don't grant consent - should raise ConsentRequiredError
    from futurnal.privacy.consent import ConsentRequiredError

    with pytest.raises(ConsentRequiredError):
        await connector.sync_mailbox(test_descriptor.id)


@pytest.mark.asyncio
async def test_sync_folder(connector, mailbox_registry, test_descriptor, consent_registry):
    """Test folder sync operation."""
    # Setup
    mailbox_registry.add_or_update(test_descriptor)

    from futurnal.ingestion.imap.consent_manager import ImapConsentScopes
    consent_registry.grant(
        source=f"mailbox:{test_descriptor.id}",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Mock connection pool and sync engine
    with patch.object(connector, '_get_connection_pool') as mock_pool:
        mock_pool.return_value = AsyncMock()

        with patch('futurnal.ingestion.imap.connector.ImapSyncEngine') as mock_sync_engine_class:
            mock_engine = AsyncMock()
            mock_sync_engine_class.return_value = mock_engine

            # Mock sync result with new messages
            sync_result = SyncResult(
                new_messages=[1, 2, 3],
                updated_messages=[4],
                deleted_messages=[5],
            )
            mock_engine.sync_folder.return_value = sync_result

            # Mock process_email to avoid actual processing
            connector.process_email = AsyncMock()
            connector.process_email_update = AsyncMock()
            connector.process_email_deletion = AsyncMock()

            # Sync folder
            result = await connector.sync_folder(test_descriptor.id, "INBOX")

            # Verify
            assert result == sync_result
            assert connector.process_email.call_count == 3  # 3 new messages
            assert connector.process_email_update.call_count == 1  # 1 updated
            assert connector.process_email_deletion.call_count == 1  # 1 deleted


@pytest.mark.asyncio
async def test_process_email(connector, mailbox_registry, test_descriptor, consent_registry):
    """Test email processing pipeline."""
    # Setup
    mailbox_registry.add_or_update(test_descriptor)

    from futurnal.ingestion.imap.consent_manager import ImapConsentScopes
    consent_registry.grant(
        source=f"mailbox:{test_descriptor.id}",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )
    consent_registry.grant(
        source=f"mailbox:{test_descriptor.id}",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    # Create mock email message
    mock_email = EmailMessage(
        message_id="test-msg-id@example.com",
        uid=1,
        folder="INBOX",
        from_address=EmailAddress(address="sender@example.com"),
        to_addresses=[EmailAddress(address="test@example.com")],
        date=datetime.utcnow(),
        body_plain="Test email body",
        body_normalized="Test email body",
        size_bytes=100,
        retrieved_at=datetime.utcnow(),
        mailbox_id=test_descriptor.id,
    )

    # Mock connection pool
    with patch.object(connector, '_get_connection_pool') as mock_pool:
        mock_connection = MagicMock()
        mock_client = MagicMock()

        # Mock IMAP fetch
        mock_client.select_folder.return_value = None
        mock_client.fetch.return_value = {
            1: {
                b'RFC822': b'From: sender@example.com\r\nTo: test@example.com\r\n\r\nTest body',
                b'FLAGS': [b'\\Seen'],
            }
        }

        mock_connection.connect.return_value.__enter__.return_value = mock_client

        # Create async context manager mock for pool.acquire()
        mock_acquire = AsyncMock()
        mock_acquire.__aenter__.return_value = mock_connection
        mock_acquire.__aexit__.return_value = None

        # Create pool mock
        mock_pool_instance = MagicMock()
        mock_pool_instance.acquire.return_value = mock_acquire

        async def get_pool_mock(descriptor):
            return mock_pool_instance

        mock_pool.side_effect = get_pool_mock

        # Mock email parser
        with patch('futurnal.ingestion.imap.connector.EmailParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_message.return_value = mock_email

            # Mock attachment extractor
            with patch('futurnal.ingestion.imap.connector.AttachmentExtractor') as mock_extractor_class:
                mock_extractor = MagicMock()
                mock_extractor_class.return_value = mock_extractor
                mock_extractor.extract_attachments.return_value = []

                # Mock Unstructured.io partition
                with patch('futurnal.ingestion.imap.connector.partition') as mock_partition:
                    mock_element = MagicMock()
                    mock_element.to_dict.return_value = {"text": "Test content", "type": "NarrativeText"}
                    mock_partition.return_value = [mock_element]

                    # Mock triple extraction
                    with patch('futurnal.ingestion.imap.connector.extract_email_triples') as mock_triples:
                        mock_triple = MagicMock()
                        mock_triple.to_dict.return_value = {
                            "subject": "test",
                            "predicate": "sent_to",
                            "object": "recipient",
                        }
                        mock_triples.return_value = [mock_triple]

                        # Process email
                        await connector.process_email(test_descriptor.id, "INBOX", 1)

                        # Verify parsing was called
                        assert mock_parser.parse_message.called
                        assert mock_extractor.extract_attachments.called
                        assert mock_partition.called
                        assert mock_triples.called


def test_persist_element(connector, workspace_dir):
    """Test element persistence to disk."""
    mock_element = {
        "text": "Test content",
        "type": "NarrativeText",
    }

    result = connector._persist_element(
        mailbox_id="test-mailbox-id",
        folder="INBOX",
        uid=1,
        message_id="test-msg-id@example.com",
        element=mock_element,
        job_id="test-job-id",
    )

    # Verify enriched metadata
    assert result["source"] == "imap"
    assert result["mailbox_id"] == "test-mailbox-id"
    assert result["folder"] == "INBOX"
    assert result["uid"] == 1
    assert result["message_id"] == "test-msg-id@example.com"
    assert "element_id" in result
    assert "element_path" in result

    # Verify file was created
    element_path = Path(result["element_path"])
    assert element_path.exists()


def test_quarantine(connector, workspace_dir):
    """Test quarantine functionality."""
    connector._quarantine(
        mailbox_id="test-mailbox-id",
        folder="INBOX",
        uid=1,
        reason="parsing_error",
        detail="Test error detail",
        policy=None,
    )

    # Verify quarantine file exists
    quarantine_dir = workspace_dir / "imap" / "quarantine"
    quarantine_files = list(quarantine_dir.glob("*.json"))
    assert len(quarantine_files) == 1

    # Verify quarantine content
    import json
    quarantine_data = json.loads(quarantine_files[0].read_text())
    assert quarantine_data["mailbox_id"] == "test-mailbox-id"
    assert quarantine_data["folder"] == "INBOX"
    assert quarantine_data["uid"] == 1
    assert quarantine_data["reason"] == "parsing_error"
    assert quarantine_data["detail"] == "Test error detail"


def test_ingest_interface(connector, mailbox_registry, test_descriptor):
    """Test synchronous ingest interface for orchestrator compatibility."""
    # Register mailbox
    mailbox_registry.add_or_update(test_descriptor)

    # Mock async sync_mailbox
    async def mock_sync(mailbox_id, **kwargs):
        return {
            "INBOX": SyncResult(new_messages=[1, 2, 3]),
            "Sent": SyncResult(new_messages=[4, 5]),
        }

    with patch.object(connector, 'sync_mailbox', new=mock_sync):
        # Call ingest (synchronous interface)
        results = list(connector.ingest(test_descriptor.id))

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result["mailbox_id"] == test_descriptor.id
        assert result["folders_synced"] == 2
        assert result["new_messages"] == 5


@pytest.mark.asyncio
async def test_process_email_deletion(connector):
    """Test email deletion handling."""
    # Mock element sink with deletion support
    mock_sink = MagicMock()
    mock_sink.handle_deletion = MagicMock()
    connector._element_sink = mock_sink

    # Process deletion
    await connector.process_email_deletion("test-mailbox-id", "INBOX", 1)

    # Verify sink was notified
    mock_sink.handle_deletion.assert_called_once()
    call_args = mock_sink.handle_deletion.call_args[0][0]
    assert call_args["type"] == "email"
    assert call_args["uid"] == 1
    assert call_args["folder"] == "INBOX"
    assert call_args["mailbox_id"] == "test-mailbox-id"


@pytest.mark.asyncio
async def test_get_connection_pool_caching(connector, test_descriptor):
    """Test connection pool is cached per mailbox."""
    with patch('futurnal.ingestion.imap.connector.ImapConnectionPool') as mock_pool_class:
        mock_pool1 = AsyncMock()
        mock_pool2 = AsyncMock()
        mock_pool_class.side_effect = [mock_pool1, mock_pool2]

        # First call should create pool
        pool1 = await connector._get_connection_pool(test_descriptor)
        assert pool1 == mock_pool1

        # Second call should return cached pool
        pool2 = await connector._get_connection_pool(test_descriptor)
        assert pool2 == mock_pool1  # Same instance

        # Only one pool created
        assert mock_pool_class.call_count == 1


def test_element_sink_integration(connector):
    """Test element sink receives all element types."""
    # Create mock sink
    mock_sink = MagicMock()
    connector._element_sink = mock_sink

    # Test element handling
    test_element = {
        "type": "semantic_triple",
        "triple": {"subject": "test", "predicate": "is", "object": "data"},
    }

    connector._element_sink.handle(test_element)

    # Verify sink was called
    mock_sink.handle.assert_called_once_with(test_element)
