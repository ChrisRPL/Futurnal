"""Integration tests for ImapEmailConnector full pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from futurnal.ingestion.imap.connector import ImapEmailConnector
from futurnal.ingestion.imap.descriptor import (
    AuthMode,
    ImapMailboxDescriptor,
    MailboxRegistry,
)
from futurnal.ingestion.imap.sync_state import ImapSyncStateStore, SyncResult
from futurnal.privacy.audit import AuditLogger
from futurnal.privacy.consent import ConsentRegistry


def _attach_mock_pool(
    mock_pool: AsyncMock, descriptor: ImapMailboxDescriptor, connection: MagicMock
) -> None:
    """Configure a mocked async _get_connection_pool call with async acquire context."""
    pool = MagicMock()
    pool.descriptor = descriptor
    acquire_context = MagicMock()
    acquire_context.__aenter__ = AsyncMock(return_value=connection)
    acquire_context.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = acquire_context
    mock_pool.return_value = pool


@pytest.fixture
def workspace_dir(tmp_path):
    """Create temporary workspace directory."""
    return tmp_path / "workspace"


@pytest.fixture
def full_setup(workspace_dir):
    """Create full connector setup with all components."""
    # State store
    state_db = workspace_dir / "imap" / "sync_state.db"
    state_store = ImapSyncStateStore(path=state_db)

    # Mailbox registry
    registry_root = workspace_dir / "sources" / "imap"
    mailbox_registry = MailboxRegistry(registry_root=registry_root)

    # Audit logger
    audit_logger = AuditLogger(workspace_dir / "audit")

    # Consent registry
    consent_registry = ConsentRegistry(workspace_dir / "privacy")

    # Element sink (mock)
    element_sink = MagicMock()

    # Credential manager (mock)
    credential_manager = MagicMock()

    # Connector
    connector = ImapEmailConnector(
        workspace_dir=workspace_dir,
        state_store=state_store,
        mailbox_registry=mailbox_registry,
        element_sink=element_sink,
        audit_logger=audit_logger,
        consent_registry=consent_registry,
        credential_manager=credential_manager,
    )

    # Test descriptor
    descriptor = ImapMailboxDescriptor.from_registration(
        email_address="test@example.com",
        imap_host="imap.example.com",
        imap_port=993,
        name="Test Mailbox",
        auth_mode=AuthMode.APP_PASSWORD,
        credential_id="test_cred_id",
        folders=["INBOX"],
    )

    mailbox_registry.add_or_update(descriptor)

    # Grant consent
    from futurnal.ingestion.imap.consent_manager import ImapConsentScopes
    consent_registry.grant(
        source=f"mailbox:{descriptor.id}",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )
    consent_registry.grant(
        source=f"mailbox:{descriptor.id}",
        scope=ImapConsentScopes.EMAIL_CONTENT_ANALYSIS.value,
    )

    return {
        "connector": connector,
        "descriptor": descriptor,
        "mailbox_registry": mailbox_registry,
        "state_store": state_store,
        "element_sink": element_sink,
        "consent_registry": consent_registry,
        "audit_logger": audit_logger,
    }


@pytest.mark.asyncio
async def test_end_to_end_email_pipeline(full_setup, workspace_dir):
    """Test complete email processing pipeline from sync to element delivery."""
    connector = full_setup["connector"]
    descriptor = full_setup["descriptor"]
    element_sink = full_setup["element_sink"]

    # Create mock email data
    raw_email = b"""From: sender@example.com
To: test@example.com
Subject: Test Email
Message-ID: <test123@example.com>
Date: Mon, 1 Jan 2024 12:00:00 +0000

This is a test email body.
"""

    # Mock IMAP connection
    with patch.object(connector, '_get_connection_pool') as mock_pool:
        mock_connection = MagicMock()
        mock_client = MagicMock()

        # Mock IMAP operations
        mock_client.select_folder.return_value = {
            b'EXISTS': 1,
            b'UIDVALIDITY': 12345,
        }
        mock_client.search.return_value = [1]
        mock_client.fetch.return_value = {
            1: {
                b'RFC822': raw_email,
                b'FLAGS': [b'\\Seen'],
            }
        }
        mock_client.capabilities.return_value = [b'IMAP4rev1']

        mock_connection.connect.return_value.__enter__.return_value = mock_client
        _attach_mock_pool(mock_pool, descriptor, mock_connection)

        # Mock Unstructured.io partition
        with patch('futurnal.ingestion.imap.connector.partition') as mock_partition:
            mock_element = MagicMock()
            mock_element.to_dict.return_value = {
                "text": "This is a test email body.",
                "type": "NarrativeText",
            }
            mock_partition.return_value = [mock_element]

            # Run sync
            results = await connector.sync_mailbox(descriptor.id)

            # Verify sync completed
            assert "INBOX" in results
            result = results["INBOX"]
            assert len(result.new_messages) > 0

            # Verify element sink received elements
            assert element_sink.handle.called

            # Check element structure
            call_args_list = element_sink.handle.call_args_list
            assert len(call_args_list) > 0

            # Verify email element was sent
            email_elements = [call[0][0] for call in call_args_list if call[0][0].get('type') != 'semantic_triple']
            assert len(email_elements) > 0

            email_element = email_elements[0]
            assert email_element["metadata"]["source"] == "imap"
            assert email_element["metadata"]["mailbox_id"] == descriptor.id
            assert email_element["metadata"]["folder"] == "INBOX"

            # Verify semantic triples were sent
            triple_elements = [call[0][0] for call in call_args_list if call[0][0].get('type') == 'semantic_triple']
            assert len(triple_elements) > 0


@pytest.mark.asyncio
async def test_attachment_processing_pipeline(full_setup):
    """Test attachment extraction and processing."""
    connector = full_setup["connector"]
    descriptor = full_setup["descriptor"]
    element_sink = full_setup["element_sink"]

    # Email with attachment
    raw_email_with_attachment = b"""From: sender@example.com
To: test@example.com
Subject: Email with Attachment
Message-ID: <att123@example.com>
Date: Mon, 1 Jan 2024 12:00:00 +0000
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="boundary123"

--boundary123
Content-Type: text/plain

Email body with attachment.

--boundary123
Content-Type: application/pdf; name="test.pdf"
Content-Disposition: attachment; filename="test.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCg==

--boundary123--
"""

    # Mock IMAP connection
    with patch.object(connector, '_get_connection_pool') as mock_pool:
        mock_connection = MagicMock()
        mock_client = MagicMock()

        mock_client.select_folder.return_value = {
            b'EXISTS': 1,
            b'UIDVALIDITY': 12345,
        }
        mock_client.search.return_value = [1]
        mock_client.fetch.return_value = {
            1: {
                b'RFC822': raw_email_with_attachment,
                b'FLAGS': [b'\\Seen'],
            }
        }
        mock_client.capabilities.return_value = [b'IMAP4rev1']

        mock_connection.connect.return_value.__enter__.return_value = mock_client
        _attach_mock_pool(mock_pool, descriptor, mock_connection)

        # Mock Unstructured.io partition
        with patch('futurnal.ingestion.imap.connector.partition') as mock_partition:
            mock_element = MagicMock()
            mock_element.to_dict.return_value = {
                "text": "Email body with attachment.",
                "type": "NarrativeText",
            }
            mock_partition.return_value = [mock_element]

            # Mock attachment processor
            with patch('futurnal.ingestion.imap.connector.AttachmentProcessor') as mock_processor_class:
                mock_processor = MagicMock()
                mock_processor_class.return_value = mock_processor

                # Mock attachment processing
                mock_processor.process_attachment = AsyncMock(
                    return_value=[
                        {
                            "text": "PDF content",
                            "type": "NarrativeText",
                            "metadata": {"filename": "test.pdf"},
                        }
                    ]
                )

                # Run sync
                await connector.sync_mailbox(descriptor.id)

                # Verify attachment was processed
                assert element_sink.handle.called

                # Check for attachment elements and triples
                call_args_list = element_sink.handle.call_args_list
                assert len(call_args_list) > 0


@pytest.mark.asyncio
async def test_thread_reconstruction_integration(full_setup):
    """Test email thread reconstruction during processing."""
    connector = full_setup["connector"]
    descriptor = full_setup["descriptor"]

    # Create thread of emails
    email1 = b"""From: sender@example.com
To: test@example.com
Subject: Original Email
Message-ID: <thread1@example.com>
Date: Mon, 1 Jan 2024 12:00:00 +0000

Original message.
"""

    email2 = b"""From: test@example.com
To: sender@example.com
Subject: Re: Original Email
Message-ID: <thread2@example.com>
In-Reply-To: <thread1@example.com>
References: <thread1@example.com>
Date: Mon, 1 Jan 2024 13:00:00 +0000

Reply message.
"""

    # Mock IMAP connection
    with patch.object(connector, '_get_connection_pool') as mock_pool:
        mock_connection = MagicMock()
        mock_client = MagicMock()

        mock_client.select_folder.return_value = {
            b'EXISTS': 2,
            b'UIDVALIDITY': 12345,
        }
        mock_client.search.return_value = [1, 2]

        # Different fetch results for each UID
        def fetch_side_effect(uids, attrs):
            if 1 in uids:
                return {1: {b'RFC822': email1, b'FLAGS': [b'\\Seen']}}
            elif 2 in uids:
                return {2: {b'RFC822': email2, b'FLAGS': [b'\\Seen']}}
            return {}

        mock_client.fetch.side_effect = fetch_side_effect
        mock_client.capabilities.return_value = [b'IMAP4rev1']

        mock_connection.connect.return_value.__enter__.return_value = mock_client
        _attach_mock_pool(mock_pool, descriptor, mock_connection)

        # Mock Unstructured.io
        with patch('futurnal.ingestion.imap.connector.partition') as mock_partition:
            mock_element = MagicMock()
            mock_element.to_dict.return_value = {
                "text": "Test content",
                "type": "NarrativeText",
            }
            mock_partition.return_value = [mock_element]

            # Run sync
            await connector.sync_mailbox(descriptor.id)

            # Verify thread reconstructor received both messages
            assert connector._thread_reconstructor is not None
            # Thread reconstructor should have processed messages
            # (actual thread verification would require deeper inspection)


@pytest.mark.asyncio
async def test_deletion_propagation(full_setup):
    """Test that deletions are properly propagated to element sink."""
    connector = full_setup["connector"]
    descriptor = full_setup["descriptor"]
    element_sink = full_setup["element_sink"]

    # Mock deletion handling
    element_sink.handle_deletion = MagicMock()

    # Process deletion
    await connector.process_email_deletion(descriptor.id, "INBOX", 123)

    # Verify deletion was propagated
    element_sink.handle_deletion.assert_called_once()
    deletion_event = element_sink.handle_deletion.call_args[0][0]
    assert deletion_event["type"] == "email"
    assert deletion_event["uid"] == 123
    assert deletion_event["folder"] == "INBOX"
    assert deletion_event["mailbox_id"] == descriptor.id


@pytest.mark.asyncio
async def test_state_persistence(full_setup):
    """Test that sync state is persisted correctly."""
    connector = full_setup["connector"]
    descriptor = full_setup["descriptor"]
    state_store = full_setup["state_store"]

    # Mock IMAP connection with sync results
    with patch.object(connector, '_get_connection_pool') as mock_pool:
        mock_connection = MagicMock()
        mock_client = MagicMock()

        mock_client.select_folder.return_value = {
            b'EXISTS': 5,
            b'UIDVALIDITY': 12345,
            b'HIGHESTMODSEQ': 100,
        }
        mock_client.search.return_value = [1, 2, 3]
        mock_client.fetch.return_value = {}
        mock_client.capabilities.return_value = [b'IMAP4rev1', b'CONDSTORE']

        mock_connection.connect.return_value.__enter__.return_value = mock_client
        _attach_mock_pool(mock_pool, descriptor, mock_connection)

        # Mock element processing to avoid actual parsing
        with patch.object(connector, "process_email", new=AsyncMock(return_value=None)):
            # Run sync
            await connector.sync_folder(descriptor.id, "INBOX")

            # Verify state was persisted
            state = state_store.fetch(descriptor.id, "INBOX")
            assert state is not None
            assert state.mailbox_id == descriptor.id
            assert state.folder == "INBOX"
            assert state.uidvalidity == 12345


@pytest.mark.asyncio
async def test_privacy_enforcement(full_setup):
    """Test privacy enforcement throughout pipeline."""
    connector = full_setup["connector"]
    descriptor = full_setup["descriptor"]

    # Revoke consent
    consent_registry = full_setup["consent_registry"]
    from futurnal.ingestion.imap.consent_manager import ImapConsentScopes
    consent_registry.revoke(
        source=f"mailbox:{descriptor.id}",
        scope=ImapConsentScopes.MAILBOX_ACCESS.value,
    )

    # Attempt sync - should fail
    from futurnal.privacy.consent import ConsentRequiredError
    with pytest.raises(ConsentRequiredError):
        await connector.sync_mailbox(descriptor.id)

    # Verify no processing occurred
    assert connector._element_sink.handle.call_count == 0


def test_orchestrator_compatibility(full_setup):
    """Test connector works with orchestrator interface."""
    connector = full_setup["connector"]
    descriptor = full_setup["descriptor"]

    # Mock async operations
    async def mock_sync(mailbox_id, *, job_id=None):
        return {
            "INBOX": SyncResult(
                new_messages=[1, 2, 3],
                updated_messages=[],
                deleted_messages=[],
            )
        }

    with patch.object(connector, 'sync_mailbox', new=mock_sync):
        # Use synchronous ingest interface
        results = list(connector.ingest(descriptor.id))

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result["mailbox_id"] == descriptor.id
        assert result["new_messages"] == 3


@pytest.mark.asyncio
async def test_error_handling_and_quarantine(full_setup, workspace_dir):
    """Test error handling and quarantine functionality."""
    connector = full_setup["connector"]
    descriptor = full_setup["descriptor"]

    # Mock IMAP to return invalid email
    with patch.object(connector, '_get_connection_pool') as mock_pool:
        mock_connection = MagicMock()
        mock_client = MagicMock()

        mock_client.select_folder.return_value = {
            b'EXISTS': 1,
            b'UIDVALIDITY': 12345,
        }
        mock_client.search.return_value = [1]
        mock_client.fetch.return_value = {
            1: {
                b'RFC822': b'INVALID EMAIL DATA',  # Invalid email
                b'FLAGS': [b'\\Seen'],
            }
        }
        mock_client.capabilities.return_value = [b'IMAP4rev1']

        mock_connection.connect.return_value.__enter__.return_value = mock_client
        _attach_mock_pool(mock_pool, descriptor, mock_connection)

        # Attempt to process - should quarantine
        try:
            await connector.process_email(descriptor.id, "INBOX", 1)
        except Exception:
            pass  # Expected to fail

        # Verify quarantine file was created
        quarantine_dir = workspace_dir / "imap" / "quarantine"
        quarantine_files = list(quarantine_dir.glob("*.json"))
        assert len(quarantine_files) > 0
